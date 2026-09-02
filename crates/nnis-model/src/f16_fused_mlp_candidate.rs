//! Candidate-only fused F16 MLP gate/up projection and SiLU multiplication.
//!
//! One block owns one intermediate output. Gate and up projections preserve the
//! qualified 128-lane F32 FMA partition and reduction tree. Lane zero then
//! reproduces the visible F16 tensor boundaries of the reference decoder:
//! projection -> F16, SiLU -> F16, product -> F16.
//!
//! This is isolated qualification infrastructure. It does not change a runtime
//! plan or default.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const SOURCE: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void nnis_gate_up_silu_nk_f16_f32acc_fused_candidate(
    const __half* input,
    const __half* gate_weight_nk,
    const __half* up_weight_nk,
    __half* output,
    unsigned long long input_k,
    unsigned long long output_n
) {
    const unsigned long long col = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (col >= output_n) return;

    const __half* gate_weight = gate_weight_nk + col * input_k;
    const __half* up_weight = up_weight_nk + col * input_k;

    extern __shared__ float partial[];
    float* gate_partial = partial;
    float* up_partial = partial + blockDim.x;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (unsigned long long row = lane; row < input_k; row += blockDim.x) {
        const float x = __half2float(input[row]);
        gate_sum = fmaf(x, __half2float(gate_weight[row]), gate_sum);
        up_sum = fmaf(x, __half2float(up_weight[row]), up_sum);
    }
    gate_partial[lane] = gate_sum;
    up_partial[lane] = up_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            gate_partial[lane] += gate_partial[lane + stride];
            up_partial[lane] += up_partial[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0) {
        const __half gate = __float2half_rn(gate_partial[0]);
        const __half up = __float2half_rn(up_partial[0]);
        const float gate_f32 = __half2float(gate);
        const __half activated =
            __float2half_rn(gate_f32 / (1.0f + expf(-gate_f32)));
        output[col] = __float2half_rn(
            __half2float(activated) * __half2float(up));
    }
}
"#;

const REDUCTION_BLOCK_SIZE: u32 = 128;

#[derive(Debug)]
pub struct F16FusedMlpCandidate {
    context: Arc<Context>,
    gate_up_silu: Kernel,
}

impl F16FusedMlpCandidate {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        let code = compiler.compile_cubin(SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let gate_up_silu =
            module.get_function("nnis_gate_up_silu_nk_f16_f32acc_fused_candidate")?;
        let attrs = gate_up_silu.attributes()?;
        if REDUCTION_BLOCK_SIZE > attrs.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "F16 fused MLP block size {REDUCTION_BLOCK_SIZE} exceeds function limit {}",
                attrs.max_threads_per_block
            )));
        }
        let shared_bytes = 2usize
            .checked_mul(REDUCTION_BLOCK_SIZE as usize)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| NnisError::invalid_input("F16 fused MLP shared-memory size overflow"))?;
        if shared_bytes > attrs.max_dynamic_shared_memory_bytes as usize {
            return Err(NnisError::invalid_input(format!(
                "F16 fused MLP requires {shared_bytes} shared-memory bytes"
            )));
        }
        Ok(Self {
            context: Arc::clone(context),
            gate_up_silu,
        })
    }

    /// Project gate/up resident `[N,K]` weights and emit SiLU(gate) * up in one launch.
    ///
    /// # Safety
    ///
    /// The stream, kernel and buffers must remain alive and otherwise untouched
    /// until the launch completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_gate_up_silu_nk(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        gate_weight_nk: &DeviceBuffer<u16>,
        up_weight_nk: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        input_k: usize,
        output_n: usize,
    ) -> Result<()> {
        if input_k == 0 || output_n == 0 {
            return Err(NnisError::invalid_input(
                "F16 fused MLP requires non-zero input_k and output_n",
            ));
        }
        let weight_elements = input_k
            .checked_mul(output_n)
            .ok_or_else(|| NnisError::invalid_input("F16 fused MLP K*N overflows usize"))?;
        if input.len() != input_k
            || gate_weight_nk.len() != weight_elements
            || up_weight_nk.len() != weight_elements
            || output.len() != output_n
        {
            return Err(NnisError::invalid_input(format!(
                "F16 fused MLP expects input={input_k}, gate/up={weight_elements}, output={output_n}; got {}/{}/{}/{}",
                input.len(),
                gate_weight_nk.len(),
                up_weight_nk.len(),
                output.len()
            )));
        }
        for context in [
            stream.ctx(),
            input.ctx(),
            gate_weight_nk.ctx(),
            up_weight_nk.ctx(),
            output.ctx(),
        ] {
            if !Arc::ptr_eq(&self.context, context) {
                return Err(NnisError::invalid_input(
                    "F16 fused MLP candidate, stream and buffers must share one CUDA context",
                ));
            }
        }

        let grid = u32::try_from(output_n)
            .map_err(|_| NnisError::invalid_input("F16 fused MLP output_n exceeds u32"))?;
        let shared_bytes = 2 * REDUCTION_BLOCK_SIZE * std::mem::size_of::<f32>() as u32;
        let config = LaunchConfig::new(
            Dim3::new(grid, 1, 1),
            Dim3::new(REDUCTION_BLOCK_SIZE, 1, 1),
        )
        .with_dynamic_shared_memory(shared_bytes);
        let mut args = KernelArgs::with_capacity(6, 4);
        args.push_buffer(input)
            .push_buffer(gate_weight_nk)
            .push_buffer(up_weight_nk)
            .push_buffer(output)
            .push(input_k as u64)
            .push(output_n as u64);
        let launch = KernelLaunch::new(&self.gate_up_silu, stream, config);
        unsafe { launch.launch(&mut args) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        F16FusedProjectionGroupsCandidate, F16ReferenceKernels,
        F16TransposedProjectionCandidate,
    };
    use nnis_rt::gpu_context;

    fn narrow(
        context: &Arc<Context>,
        stream: &Stream,
        reference: &F16ReferenceKernels,
        values: &[f32],
    ) -> DeviceBuffer<u16> {
        let source = DeviceBuffer::from_host(context, stream, values).unwrap();
        let output = DeviceBuffer::<u16>::new(context, values.len()).unwrap();
        unsafe {
            reference
                .enqueue_narrow_from_f32(stream, &source, &output)
                .unwrap();
        }
        stream.synchronize().unwrap();
        output
    }

    fn deterministic(elements: usize, salt: usize) -> Vec<f32> {
        (0..elements)
            .map(|index| {
                let value = ((index.wrapping_mul(23 + salt) + 7 * salt) % 101) as i32 - 50;
                value as f32 * 0.015625
            })
            .collect()
    }

    #[test]
    fn fused_mlp_matches_grouped_projection_then_silu_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let compiler = JitCompiler::new();
        let reference = F16ReferenceKernels::load(&context, &compiler).unwrap();
        let transposed = F16TransposedProjectionCandidate::load(&context, &compiler).unwrap();
        let grouped = F16FusedProjectionGroupsCandidate::load(&context, &compiler).unwrap();
        let fused = F16FusedMlpCandidate::load(&context, &compiler).unwrap();

        let k = 128usize;
        let n = 160usize;
        let input = narrow(&context, &stream, &reference, &deterministic(k, 1));
        let prepare_weight = |salt: usize| {
            let kn = narrow(&context, &stream, &reference, &deterministic(k * n, salt));
            let nk = DeviceBuffer::<u16>::new(&context, k * n).unwrap();
            unsafe {
                transposed
                    .enqueue_transpose_kn_to_nk(&stream, &kn, &nk, k, n)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            nk
        };
        let gate_weight = prepare_weight(2);
        let up_weight = prepare_weight(3);
        let gate = DeviceBuffer::<u16>::new(&context, n).unwrap();
        let up = DeviceBuffer::<u16>::new(&context, n).unwrap();
        let reference_output = DeviceBuffer::<u16>::new(&context, n).unwrap();
        let fused_output = DeviceBuffer::<u16>::new(&context, n).unwrap();

        unsafe {
            grouped
                .enqueue_gate_up_nk(
                    &stream,
                    &input,
                    &gate_weight,
                    &up_weight,
                    &gate,
                    &up,
                    k,
                    n,
                )
                .unwrap();
            reference
                .enqueue_silu_multiply(&stream, &gate, &up, &reference_output)
                .unwrap();
            fused
                .enqueue_gate_up_silu_nk(
                    &stream,
                    &input,
                    &gate_weight,
                    &up_weight,
                    &fused_output,
                    k,
                    n,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(
            reference_output.to_vec(&stream).unwrap(),
            fused_output.to_vec(&stream).unwrap()
        );
    }
}
