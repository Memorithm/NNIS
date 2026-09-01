//! Candidate-only transposed-layout F16 projections for Thor decode optimization.
//!
//! The qualified F16 reference runtime stores decoder projection weights in
//! model-format orientation `[K, N]`. Its reference projection kernel assigns one
//! block to one output column, so lanes within that block read `weight[row*N+col]`
//! with a large stride. This candidate keeps the exact same lane partition,
//! F32 FMA accumulation order, shared-memory reduction tree, and F16 output
//! boundary, but consumes a one-time resident `[N, K]` transpose so lanes read
//! contiguous weights within each output row.
//!
//! This module is isolated research infrastructure. It does not change the
//! default or qualified F16 runtime path.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const SOURCE: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void nnis_transpose_kn_to_nk_f16(
    const __half* source,
    __half* destination,
    unsigned long long k,
    unsigned long long n
) {
    const unsigned long long linear =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long elements = k * n;
    if (linear >= elements) return;
    const unsigned long long row = linear / n;
    const unsigned long long col = linear % n;
    destination[col * k + row] = source[row * n + col];
}

extern "C" __global__ void nnis_project_nk_f16_f32acc_candidate(
    const __half* input,
    const __half* weight_nk,
    __half* output,
    unsigned long long k,
    unsigned long long n
) {
    const unsigned long long col = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (col >= n) return;

    extern __shared__ float partial[];
    const __half* row_weight = weight_nk + col * k;
    float sum = 0.0f;
    for (unsigned long long row = lane; row < k; row += blockDim.x) {
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

extern "C" __global__ void nnis_lm_head_nk_f16_to_f32_candidate(
    const __half* input,
    const __half* weight_nk,
    float* output,
    unsigned long long k,
    unsigned long long n
) {
    const unsigned long long col = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (col >= n) return;

    extern __shared__ float partial[];
    const __half* row_weight = weight_nk + col * k;
    float sum = 0.0f;
    for (unsigned long long row = lane; row < k; row += blockDim.x) {
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
        output[col] = __half2float(__float2half_rn(partial[0]));
    }
}
"#;

const REDUCTION_BLOCK_SIZE: u32 = 128;
const TRANSPOSE_BLOCK_SIZE: u32 = 256;

#[derive(Debug)]
pub struct F16TransposedProjectionCandidate {
    context: Arc<Context>,
    transpose: Kernel,
    project_nk: Kernel,
    lm_head_nk: Kernel,
}

impl F16TransposedProjectionCandidate {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        let code = compiler.compile_cubin(SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let transpose = module.get_function("nnis_transpose_kn_to_nk_f16")?;
        let project_nk = module.get_function("nnis_project_nk_f16_f32acc_candidate")?;
        let lm_head_nk = module.get_function("nnis_lm_head_nk_f16_to_f32_candidate")?;

        for (name, kernel, block) in [
            ("transpose", &transpose, TRANSPOSE_BLOCK_SIZE),
            ("project_nk", &project_nk, REDUCTION_BLOCK_SIZE),
            ("lm_head_nk", &lm_head_nk, REDUCTION_BLOCK_SIZE),
        ] {
            let attrs = kernel.attributes()?;
            if block > attrs.max_threads_per_block {
                return Err(NnisError::invalid_input(format!(
                    "F16 transposed candidate kernel {name} block size {block} exceeds function limit {}",
                    attrs.max_threads_per_block
                )));
            }
        }
        let shared_bytes = REDUCTION_BLOCK_SIZE as usize * std::mem::size_of::<f32>();
        for (name, kernel) in [("project_nk", &project_nk), ("lm_head_nk", &lm_head_nk)] {
            if shared_bytes > kernel.attributes()?.max_dynamic_shared_memory_bytes as usize {
                return Err(NnisError::invalid_input(format!(
                    "F16 transposed candidate kernel {name} requires {shared_bytes} shared-memory bytes"
                )));
            }
        }

        Ok(Self {
            context: Arc::clone(context),
            transpose,
            project_nk,
            lm_head_nk,
        })
    }

    /// Transpose one resident F16 `[K,N]` matrix into candidate `[N,K]` layout.
    ///
    /// # Safety
    ///
    /// The stream, candidate kernels, source, and destination must remain alive
    /// and otherwise untouched until this launch completes.
    pub unsafe fn enqueue_transpose_kn_to_nk(
        &self,
        stream: &Stream,
        source: &DeviceBuffer<u16>,
        destination: &DeviceBuffer<u16>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let elements = self.validate_matrix(stream, source, destination, k, n)?;
        if elements == 0 {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(4, 2);
        args.push_buffer(source)
            .push_buffer(destination)
            .push(k as u64)
            .push(n as u64);
        let launch = KernelLaunch::new(
            &self.transpose,
            stream,
            LaunchConfig::for_num_elements(elements, TRANSPOSE_BLOCK_SIZE)?,
        );
        unsafe { launch.launch(&mut args) }
    }

    /// Project one F16 row against a resident `[N,K]` F16 matrix.
    ///
    /// Arithmetic order is intentionally identical to the qualified `[K,N]`
    /// reference kernel; only weight addressing differs.
    ///
    /// # Safety
    ///
    /// The stream, candidate kernels, input, weight, and output must remain
    /// alive and otherwise untouched until this launch completes.
    pub unsafe fn enqueue_project_nk(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        weight_nk: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        self.validate_projection(stream, input, weight_nk, output.ctx(), output.len(), k, n)?;
        if n == 0 {
            return Ok(());
        }
        let grid = u32::try_from(n)
            .map_err(|_| NnisError::invalid_input("F16 transposed projection N exceeds u32"))?;
        let config = LaunchConfig::new(
            Dim3::new(grid, 1, 1),
            Dim3::new(REDUCTION_BLOCK_SIZE, 1, 1),
        )
        .with_dynamic_shared_memory(REDUCTION_BLOCK_SIZE * std::mem::size_of::<f32>() as u32);
        let mut args = KernelArgs::with_capacity(5, 3);
        args.push_buffer(input)
            .push_buffer(weight_nk)
            .push_buffer(output)
            .push(k as u64)
            .push(n as u64);
        let launch = KernelLaunch::new(&self.project_nk, stream, config);
        unsafe { launch.launch(&mut args) }
    }

    /// Candidate LM-head projection with `[N,K]` F16 weights and F32 logits.
    ///
    /// # Safety
    ///
    /// The stream, candidate kernels, input, weight, and output must remain
    /// alive and otherwise untouched until this launch completes.
    pub unsafe fn enqueue_lm_head_nk_f32_logits(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        weight_nk: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        self.validate_projection(stream, input, weight_nk, output.ctx(), output.len(), k, n)?;
        if n == 0 {
            return Ok(());
        }
        let grid = u32::try_from(n)
            .map_err(|_| NnisError::invalid_input("F16 transposed LM-head N exceeds u32"))?;
        let config = LaunchConfig::new(
            Dim3::new(grid, 1, 1),
            Dim3::new(REDUCTION_BLOCK_SIZE, 1, 1),
        )
        .with_dynamic_shared_memory(REDUCTION_BLOCK_SIZE * std::mem::size_of::<f32>() as u32);
        let mut args = KernelArgs::with_capacity(5, 3);
        args.push_buffer(input)
            .push_buffer(weight_nk)
            .push_buffer(output)
            .push(k as u64)
            .push(n as u64);
        let launch = KernelLaunch::new(&self.lm_head_nk, stream, config);
        unsafe { launch.launch(&mut args) }
    }

    fn validate_matrix(
        &self,
        stream: &Stream,
        source: &DeviceBuffer<u16>,
        destination: &DeviceBuffer<u16>,
        k: usize,
        n: usize,
    ) -> Result<usize> {
        let elements = k
            .checked_mul(n)
            .ok_or_else(|| NnisError::invalid_input("F16 transpose K*N overflows usize"))?;
        if source.len() != elements || destination.len() != elements {
            return Err(NnisError::invalid_input(format!(
                "F16 transpose expects {elements} elements; got {}/{}",
                source.len(),
                destination.len()
            )));
        }
        self.validate_contexts(stream, &[source.ctx(), destination.ctx()])?;
        Ok(elements)
    }

    fn validate_projection(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        weight: &DeviceBuffer<u16>,
        output_context: &Arc<Context>,
        output_len: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let elements = k
            .checked_mul(n)
            .ok_or_else(|| NnisError::invalid_input("F16 transposed projection K*N overflows usize"))?;
        if input.len() != k || weight.len() != elements || output_len != n {
            return Err(NnisError::invalid_input(format!(
                "F16 transposed projection expects input={k}, weight={elements}, output={n}; got {}/{}/{}",
                input.len(),
                weight.len(),
                output_len
            )));
        }
        self.validate_contexts(stream, &[input.ctx(), weight.ctx(), output_context])
    }

    fn validate_contexts(&self, stream: &Stream, contexts: &[&Arc<Context>]) -> Result<()> {
        if !Arc::ptr_eq(&self.context, stream.ctx())
            || contexts
                .iter()
                .any(|context| !Arc::ptr_eq(&self.context, context))
        {
            return Err(NnisError::invalid_input(
                "F16 transposed candidate kernels, stream, and buffers must share one CUDA context",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F16ReferenceKernels;
    use nnis_rt::gpu_context;

    #[test]
    fn transposed_projection_is_bitwise_equal_to_reference_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let compiler = JitCompiler::new();
        let reference = F16ReferenceKernels::load(&context, &compiler).unwrap();
        let candidate = F16TransposedProjectionCandidate::load(&context, &compiler).unwrap();

        let k = 7usize;
        let n = 5usize;
        let input_host: Vec<f32> = (0..k)
            .map(|index| ((index as f32) - 3.0) * 0.125)
            .collect();
        let weight_host: Vec<f32> = (0..k * n)
            .map(|index| ((index as i32 % 11) - 5) as f32 * 0.0625)
            .collect();
        let input_f32 = DeviceBuffer::from_host(&context, &stream, &input_host).unwrap();
        let weight_f32 = DeviceBuffer::from_host(&context, &stream, &weight_host).unwrap();
        let input_f16 = DeviceBuffer::<u16>::new(&context, k).unwrap();
        let weight_kn = DeviceBuffer::<u16>::new(&context, k * n).unwrap();
        let weight_nk = DeviceBuffer::<u16>::new(&context, k * n).unwrap();
        let reference_output = DeviceBuffer::<u16>::new(&context, n).unwrap();
        let candidate_output = DeviceBuffer::<u16>::new(&context, n).unwrap();
        let reference_logits = DeviceBuffer::<f32>::new(&context, n).unwrap();
        let candidate_logits = DeviceBuffer::<f32>::new(&context, n).unwrap();

        unsafe {
            reference
                .enqueue_narrow_from_f32(&stream, &input_f32, &input_f16)
                .unwrap();
            reference
                .enqueue_narrow_from_f32(&stream, &weight_f32, &weight_kn)
                .unwrap();
            candidate
                .enqueue_transpose_kn_to_nk(&stream, &weight_kn, &weight_nk, k, n)
                .unwrap();
            reference
                .enqueue_project_kn(&stream, &input_f16, &weight_kn, &reference_output, k, n)
                .unwrap();
            candidate
                .enqueue_project_nk(&stream, &input_f16, &weight_nk, &candidate_output, k, n)
                .unwrap();
            reference
                .enqueue_lm_head_f32_logits(
                    &stream,
                    &input_f16,
                    &weight_kn,
                    &reference_logits,
                    k,
                    n,
                )
                .unwrap();
            candidate
                .enqueue_lm_head_nk_f32_logits(
                    &stream,
                    &input_f16,
                    &weight_nk,
                    &candidate_logits,
                    k,
                    n,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();

        assert_eq!(
            reference_output.to_vec(&stream).unwrap(),
            candidate_output.to_vec(&stream).unwrap()
        );
        let left = reference_logits.to_vec(&stream).unwrap();
        let right = candidate_logits.to_vec(&stream).unwrap();
        assert_eq!(left.len(), right.len());
        for (index, (&a, &b)) in left.iter().zip(&right).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "LM-head candidate differs at output {index}: {a} != {b}"
            );
        }
    }
}
