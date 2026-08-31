//! Batch-one projection with f32 activations and packed-bf16 weights.
//!
//! This is a representation experiment primitive, not a claim that the
//! whole decoder is BF16. Each packed weight is widened exactly to f32,
//! then accumulated with the same increasing-K explicit-FMA order as
//! [`crate::F32Gemv::project_kn`]. The output remains f32.

use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const SOURCE: &str = r#"
__device__ __forceinline__ float nnis_bf16_bits_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}

extern "C" __global__ void nnis_project_kn_f32_bf16_weight(
    const float* input,
    const unsigned short* weight,
    float* output,
    unsigned long long k,
    unsigned long long n
) {
    const unsigned long long col =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    float value = 0.0f;
    for (unsigned long long row = 0; row < k; ++row) {
        const float w = nnis_bf16_bits_to_f32(weight[row * n + col]);
        value = fmaf(input[row], w, value);
    }
    output[col] = value;
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 64;

/// Context-bound `[1,K] × [K,N] -> [1,N]` projection using f32
/// activations, packed-bf16 weights and f32 accumulation/output.
#[derive(Debug)]
pub struct F32Bf16Gemv {
    project_kn: Kernel,
    block_size: u32,
}

impl F32Bf16Gemv {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(NnisError::invalid_input(format!(
                "f32-bf16 project block size {block_size} is not a non-zero power of two"
            )));
        }
        let code = compiler.compile_cubin(SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let project_kn = module.get_function("nnis_project_kn_f32_bf16_weight")?;
        let attributes = project_kn.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "f32-bf16 project block size {block_size} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        Ok(Self {
            project_kn,
            block_size,
        })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn project_kn(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result =
            unsafe { self.enqueue_project_kn(stream, input, weight, output, k, n) };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue the mixed-representation projection without synchronizing.
    ///
    /// # Safety
    ///
    /// All buffers, the stream and this kernel must remain alive and
    /// otherwise untouched until the stream completes.
    pub unsafe fn enqueue_project_kn(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        self.validate_execution(stream, input, weight, output, k, n)?;
        if n == 0 {
            return Ok(());
        }
        if k == 0 {
            // SAFETY: the output lifetime obligation is documented above.
            return unsafe { output.zero_async(stream) };
        }
        let k_arg = u64::try_from(k)
            .map_err(|_| NnisError::invalid_input("f32-bf16 project K exceeds u64::MAX"))?;
        let n_arg = u64::try_from(n)
            .map_err(|_| NnisError::invalid_input("f32-bf16 project N exceeds u64::MAX"))?;
        let config = LaunchConfig::for_num_elements(n, self.block_size)?;
        let mut arguments = KernelArgs::with_capacity(5, 3);
        arguments
            .push_buffer(input)
            .push_buffer(weight)
            .push_buffer(output)
            .push(k_arg)
            .push(n_arg);
        let launch = KernelLaunch::new(&self.project_kn, stream, config);
        // SAFETY: argument order/widths match the CUDA kernel; the caller
        // owns the remaining asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    fn validate_execution(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let expected_weight = k
            .checked_mul(n)
            .ok_or_else(|| NnisError::invalid_input("f32-bf16 project shape overflows usize"))?;
        if input.len() != k {
            return Err(NnisError::invalid_input(format!(
                "f32-bf16 project input has {} elements; shape requires {k}",
                input.len()
            )));
        }
        if weight.len() != expected_weight {
            return Err(NnisError::invalid_input(format!(
                "f32-bf16 project weight has {} elements; shape ({k}, {n}) requires {expected_weight}",
                weight.len()
            )));
        }
        if output.len() != n {
            return Err(NnisError::invalid_input(format!(
                "f32-bf16 project output has {} elements; shape requires {n}",
                output.len()
            )));
        }
        let context = self.project_kn.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, input.ctx())
            || !Arc::ptr_eq(context, weight.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "f32-bf16 project stream, buffers and kernel must share one context",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F32Gemv;
    use nnis_rt::gpu_context;

    fn exact_bf16_value(index: usize) -> f32 {
        ((index * 17 % 41) as f32 - 20.0) * 0.125
    }

    #[test]
    fn mixed_project_matches_f32_project_for_exact_bf16_weights() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let k = 17usize;
        let n = 23usize;
        let input_host = (0..k)
            .map(|index| ((index * 11 % 29) as f32 - 14.0) * 0.0625)
            .collect::<Vec<_>>();
        let weight_f32 = (0..k * n).map(exact_bf16_value).collect::<Vec<_>>();
        let weight_bf16 = weight_f32
            .iter()
            .map(|value| {
                let bits = value.to_bits();
                assert_eq!(bits & 0xffff, 0);
                (bits >> 16) as u16
            })
            .collect::<Vec<_>>();
        let compiler = JitCompiler::new();
        let reference = F32Gemv::load_with_block_size(&context, &compiler, 64).unwrap();
        let candidate = F32Bf16Gemv::load_with_block_size(&context, &compiler, 64).unwrap();
        let stream = Stream::new(&context).unwrap();
        let input = DeviceBuffer::from_host(&context, &stream, &input_host).unwrap();
        let f32_weight = DeviceBuffer::from_host(&context, &stream, &weight_f32).unwrap();
        let bf16_weight = DeviceBuffer::from_host(&context, &stream, &weight_bf16).unwrap();
        let f32_output = DeviceBuffer::<f32>::new(&context, n).unwrap();
        let bf16_output = DeviceBuffer::<f32>::new(&context, n).unwrap();
        reference
            .project_kn(&stream, &input, &f32_weight, &f32_output, k, n)
            .unwrap();
        candidate
            .project_kn(&stream, &input, &bf16_weight, &bf16_output, k, n)
            .unwrap();
        let expected = f32_output.to_vec(&stream).unwrap();
        let actual = bf16_output.to_vec(&stream).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "mismatch at output {index}: actual={actual}, expected={expected}"
            );
        }
    }
}
