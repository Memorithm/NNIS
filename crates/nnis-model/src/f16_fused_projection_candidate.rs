//! Candidate-only grouped launches for resident `[N,K]` F16 projections.
//!
//! The qualified transposed projection path launches Q, K, V separately and
//! gate/up separately even though each group consumes the same input vector.
//! This candidate preserves the exact per-output 128-lane F32 FMA partition,
//! shared-memory reduction tree and F16 rounding boundary while combining each
//! group into one CUDA launch.
//!
//! This module is isolated qualification infrastructure. It does not change a
//! runtime plan or default.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const SOURCE: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void nnis_qkv_nk_f16_f32acc_fused_candidate(
    const __half* input,
    const __half* q_weight_nk,
    const __half* k_weight_nk,
    const __half* v_weight_nk,
    __half* q_output,
    __half* k_output,
    __half* v_output,
    unsigned long long input_k,
    unsigned long long q_n,
    unsigned long long kv_n
) {
    const unsigned long long logical_col = blockIdx.x;
    const unsigned int lane = threadIdx.x;

    const __half* row_weight = nullptr;
    __half* output = nullptr;
    unsigned long long col = 0;

    if (logical_col < q_n) {
        col = logical_col;
        row_weight = q_weight_nk + col * input_k;
        output = q_output;
    } else if (logical_col < q_n + kv_n) {
        col = logical_col - q_n;
        row_weight = k_weight_nk + col * input_k;
        output = k_output;
    } else {
        col = logical_col - q_n - kv_n;
        if (col >= kv_n) return;
        row_weight = v_weight_nk + col * input_k;
        output = v_output;
    }

    extern __shared__ float partial[];
    float sum = 0.0f;
    for (unsigned long long row = lane; row < input_k; row += blockDim.x) {
        sum = fmaf(
            __half2float(input[row]),
            __half2float(row_weight[row]),
            sum);
    }
    partial[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        output[col] = __float2half_rn(partial[0]);
    }
}

extern "C" __global__ void nnis_gate_up_nk_f16_f32acc_fused_candidate(
    const __half* input,
    const __half* gate_weight_nk,
    const __half* up_weight_nk,
    __half* gate_output,
    __half* up_output,
    unsigned long long input_k,
    unsigned long long output_n
) {
    const unsigned long long logical_col = blockIdx.x;
    const unsigned int lane = threadIdx.x;

    const bool is_up = logical_col >= output_n;
    const unsigned long long col = is_up ? logical_col - output_n : logical_col;
    if (col >= output_n) return;
    const __half* row_weight =
        (is_up ? up_weight_nk : gate_weight_nk) + col * input_k;
    __half* output = is_up ? up_output : gate_output;

    extern __shared__ float partial[];
    float sum = 0.0f;
    for (unsigned long long row = lane; row < input_k; row += blockDim.x) {
        sum = fmaf(
            __half2float(input[row]),
            __half2float(row_weight[row]),
            sum);
    }
    partial[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        output[col] = __float2half_rn(partial[0]);
    }
}
"#;

const REDUCTION_BLOCK_SIZE: u32 = 128;

#[derive(Debug)]
pub struct F16FusedProjectionGroupsCandidate {
    context: Arc<Context>,
    qkv: Kernel,
    gate_up: Kernel,
}

impl F16FusedProjectionGroupsCandidate {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        let code = compiler.compile_cubin(SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let qkv = module.get_function("nnis_qkv_nk_f16_f32acc_fused_candidate")?;
        let gate_up = module.get_function("nnis_gate_up_nk_f16_f32acc_fused_candidate")?;
        let shared_bytes = REDUCTION_BLOCK_SIZE as usize * std::mem::size_of::<f32>();
        for (name, kernel) in [("qkv", &qkv), ("gate_up", &gate_up)] {
            let attrs = kernel.attributes()?;
            if REDUCTION_BLOCK_SIZE > attrs.max_threads_per_block {
                return Err(NnisError::invalid_input(format!(
                    "F16 fused projection {name} block size {REDUCTION_BLOCK_SIZE} exceeds function limit {}",
                    attrs.max_threads_per_block
                )));
            }
            if shared_bytes > attrs.max_dynamic_shared_memory_bytes as usize {
                return Err(NnisError::invalid_input(format!(
                    "F16 fused projection {name} requires {shared_bytes} shared-memory bytes"
                )));
            }
        }
        Ok(Self {
            context: Arc::clone(context),
            qkv,
            gate_up,
        })
    }

    /// Launch Q/K/V resident `[N,K]` projections as one CUDA kernel.
    ///
    /// # Safety
    ///
    /// The stream, kernels and buffers must remain alive and otherwise untouched
    /// until the launch completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_qkv_nk(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        q_weight_nk: &DeviceBuffer<u16>,
        k_weight_nk: &DeviceBuffer<u16>,
        v_weight_nk: &DeviceBuffer<u16>,
        q_output: &DeviceBuffer<u16>,
        k_output: &DeviceBuffer<u16>,
        v_output: &DeviceBuffer<u16>,
        input_k: usize,
        q_n: usize,
        kv_n: usize,
    ) -> Result<()> {
        self.validate_projection(stream, input, q_weight_nk, q_output, input_k, q_n, "q")?;
        self.validate_projection(stream, input, k_weight_nk, k_output, input_k, kv_n, "k")?;
        self.validate_projection(stream, input, v_weight_nk, v_output, input_k, kv_n, "v")?;
        let blocks = q_n
            .checked_add(
                kv_n.checked_mul(2).ok_or_else(|| {
                    NnisError::invalid_input("F16 fused QKV 2*kv_n overflows usize")
                })?,
            )
            .ok_or_else(|| NnisError::invalid_input("F16 fused QKV grid overflows usize"))?;
        if blocks == 0 {
            return Ok(());
        }
        let grid = u32::try_from(blocks)
            .map_err(|_| NnisError::invalid_input("F16 fused QKV grid exceeds u32"))?;
        let config =
            LaunchConfig::new(Dim3::new(grid, 1, 1), Dim3::new(REDUCTION_BLOCK_SIZE, 1, 1))
                .with_dynamic_shared_memory(
                    REDUCTION_BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
                );
        let mut args = KernelArgs::with_capacity(10, 7);
        args.push_buffer(input)
            .push_buffer(q_weight_nk)
            .push_buffer(k_weight_nk)
            .push_buffer(v_weight_nk)
            .push_buffer(q_output)
            .push_buffer(k_output)
            .push_buffer(v_output)
            .push(input_k as u64)
            .push(q_n as u64)
            .push(kv_n as u64);
        let launch = KernelLaunch::new(&self.qkv, stream, config);
        unsafe { launch.launch(&mut args) }
    }

    /// Launch gate/up resident `[N,K]` projections as one CUDA kernel.
    ///
    /// # Safety
    ///
    /// The stream, kernels and buffers must remain alive and otherwise untouched
    /// until the launch completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_gate_up_nk(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        gate_weight_nk: &DeviceBuffer<u16>,
        up_weight_nk: &DeviceBuffer<u16>,
        gate_output: &DeviceBuffer<u16>,
        up_output: &DeviceBuffer<u16>,
        input_k: usize,
        output_n: usize,
    ) -> Result<()> {
        self.validate_projection(
            stream,
            input,
            gate_weight_nk,
            gate_output,
            input_k,
            output_n,
            "gate",
        )?;
        self.validate_projection(
            stream,
            input,
            up_weight_nk,
            up_output,
            input_k,
            output_n,
            "up",
        )?;
        let blocks = output_n
            .checked_mul(2)
            .ok_or_else(|| NnisError::invalid_input("F16 fused gate/up grid overflows usize"))?;
        if blocks == 0 {
            return Ok(());
        }
        let grid = u32::try_from(blocks)
            .map_err(|_| NnisError::invalid_input("F16 fused gate/up grid exceeds u32"))?;
        let config =
            LaunchConfig::new(Dim3::new(grid, 1, 1), Dim3::new(REDUCTION_BLOCK_SIZE, 1, 1))
                .with_dynamic_shared_memory(
                    REDUCTION_BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
                );
        let mut args = KernelArgs::with_capacity(7, 5);
        args.push_buffer(input)
            .push_buffer(gate_weight_nk)
            .push_buffer(up_weight_nk)
            .push_buffer(gate_output)
            .push_buffer(up_output)
            .push(input_k as u64)
            .push(output_n as u64);
        let launch = KernelLaunch::new(&self.gate_up, stream, config);
        unsafe { launch.launch(&mut args) }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_projection(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        weight: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        k: usize,
        n: usize,
        name: &str,
    ) -> Result<()> {
        if k == 0 || n == 0 {
            return Err(NnisError::invalid_input(format!(
                "F16 fused projection {name} requires non-zero K and N"
            )));
        }
        let elements = k.checked_mul(n).ok_or_else(|| {
            NnisError::invalid_input(format!("F16 fused projection {name} K*N overflows usize"))
        })?;
        if input.len() != k || weight.len() != elements || output.len() != n {
            return Err(NnisError::invalid_input(format!(
                "F16 fused projection {name} expects input={k}, weight={elements}, output={n}; got {}/{}/{}",
                input.len(),
                weight.len(),
                output.len()
            )));
        }
        if !Arc::ptr_eq(&self.context, stream.ctx())
            || !Arc::ptr_eq(&self.context, input.ctx())
            || !Arc::ptr_eq(&self.context, weight.ctx())
            || !Arc::ptr_eq(&self.context, output.ctx())
        {
            return Err(NnisError::invalid_input(format!(
                "F16 fused projection {name} candidate, stream and buffers must share one CUDA context"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{F16ReferenceKernels, F16TransposedProjectionCandidate};
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
            .map(|index| ((index.wrapping_mul(17 + salt) % 61) as i32 - 30) as f32 * 0.03125)
            .collect()
    }

    #[test]
    fn fused_groups_match_sequential_transposed_projection_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let compiler = JitCompiler::new();
        let reference = F16ReferenceKernels::load(&context, &compiler).unwrap();
        let sequential = F16TransposedProjectionCandidate::load(&context, &compiler).unwrap();
        let fused = F16FusedProjectionGroupsCandidate::load(&context, &compiler).unwrap();

        let k = 128usize;
        let q_n = 96usize;
        let kv_n = 32usize;
        let mlp_n = 160usize;
        let input = narrow(&context, &stream, &reference, &deterministic(k, 1));

        let prepare_weight = |n: usize, salt: usize| {
            let kn = narrow(&context, &stream, &reference, &deterministic(k * n, salt));
            let nk = DeviceBuffer::<u16>::new(&context, k * n).unwrap();
            unsafe {
                sequential
                    .enqueue_transpose_kn_to_nk(&stream, &kn, &nk, k, n)
                    .unwrap();
            }
            stream.synchronize().unwrap();
            nk
        };

        let q_w = prepare_weight(q_n, 2);
        let k_w = prepare_weight(kv_n, 3);
        let v_w = prepare_weight(kv_n, 4);
        let gate_w = prepare_weight(mlp_n, 5);
        let up_w = prepare_weight(mlp_n, 6);

        let q_seq = DeviceBuffer::<u16>::new(&context, q_n).unwrap();
        let k_seq = DeviceBuffer::<u16>::new(&context, kv_n).unwrap();
        let v_seq = DeviceBuffer::<u16>::new(&context, kv_n).unwrap();
        let q_fused = DeviceBuffer::<u16>::new(&context, q_n).unwrap();
        let k_fused = DeviceBuffer::<u16>::new(&context, kv_n).unwrap();
        let v_fused = DeviceBuffer::<u16>::new(&context, kv_n).unwrap();
        let gate_seq = DeviceBuffer::<u16>::new(&context, mlp_n).unwrap();
        let up_seq = DeviceBuffer::<u16>::new(&context, mlp_n).unwrap();
        let gate_fused = DeviceBuffer::<u16>::new(&context, mlp_n).unwrap();
        let up_fused = DeviceBuffer::<u16>::new(&context, mlp_n).unwrap();

        unsafe {
            sequential
                .enqueue_project_nk(&stream, &input, &q_w, &q_seq, k, q_n)
                .unwrap();
            sequential
                .enqueue_project_nk(&stream, &input, &k_w, &k_seq, k, kv_n)
                .unwrap();
            sequential
                .enqueue_project_nk(&stream, &input, &v_w, &v_seq, k, kv_n)
                .unwrap();
            fused
                .enqueue_qkv_nk(
                    &stream, &input, &q_w, &k_w, &v_w, &q_fused, &k_fused, &v_fused, k, q_n, kv_n,
                )
                .unwrap();
            sequential
                .enqueue_project_nk(&stream, &input, &gate_w, &gate_seq, k, mlp_n)
                .unwrap();
            sequential
                .enqueue_project_nk(&stream, &input, &up_w, &up_seq, k, mlp_n)
                .unwrap();
            fused
                .enqueue_gate_up_nk(
                    &stream,
                    &input,
                    &gate_w,
                    &up_w,
                    &gate_fused,
                    &up_fused,
                    k,
                    mlp_n,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();

        assert_eq!(
            q_seq.to_vec(&stream).unwrap(),
            q_fused.to_vec(&stream).unwrap()
        );
        assert_eq!(
            k_seq.to_vec(&stream).unwrap(),
            k_fused.to_vec(&stream).unwrap()
        );
        assert_eq!(
            v_seq.to_vec(&stream).unwrap(),
            v_fused.to_vec(&stream).unwrap()
        );
        assert_eq!(
            gate_seq.to_vec(&stream).unwrap(),
            gate_fused.to_vec(&stream).unwrap()
        );
        assert_eq!(
            up_seq.to_vec(&stream).unwrap(),
            up_fused.to_vec(&stream).unwrap()
        );
    }
}
