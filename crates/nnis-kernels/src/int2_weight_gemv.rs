//! Batch-one projection with F32 activations and packed ternary INT2 weights.
//!
//! Four 2-bit codes are stored per byte. The reference mapping is:
//! `0 -> 0`, `1 -> +1`, `2 -> -1`; code `3` is reserved and rejected
//! by the model-level representation validator. The kernel multiplies the
//! decoded signed level by one resident F32 scale and accumulates in F32.
//!
//! This is an isolated execution primitive, not a full-model INT2 promotion
//! and not a performance claim.

use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const SOURCE: &str = r#"
__device__ __forceinline__ int nnis_decode_ternary_int2(
    unsigned char packed,
    unsigned int lane
) {
    const unsigned int code = ((unsigned int)packed >> (lane * 2u)) & 3u;
    if (code == 1u) return 1;
    if (code == 2u) return -1;
    return 0;
}

extern "C" __global__ void nnis_project_kn_f32_int2_weight(
    const float* input,
    const unsigned char* packed_weight,
    const float* scale,
    float* output,
    unsigned long long k,
    unsigned long long n
) {
    const unsigned long long col =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    const float weight_scale = scale[0];
    float value = 0.0f;
    for (unsigned long long row = 0; row < k; ++row) {
        const unsigned long long flat = row * n + col;
        const unsigned char packed = packed_weight[flat >> 2];
        const unsigned int lane = (unsigned int)(flat & 3ull);
        const int q = nnis_decode_ternary_int2(packed, lane);
        const float weight = (float)q * weight_scale;
        value = fmaf(input[row], weight, value);
    }
    output[col] = value;
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 64;

/// Context-bound `[1,K] x [K,N] -> [1,N]` projection using F32 activations,
/// packed ternary INT2 weights, one resident F32 scale and F32 accumulation.
#[derive(Debug)]
pub struct F32Int2Gemv {
    project_kn: Kernel,
    block_size: u32,
}

impl F32Int2Gemv {
    /// Compile and load the default INT2 projection primitive.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load with an explicit non-zero power-of-two thread-block width.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(NnisError::invalid_input(format!(
                "f32-int2 project block size {block_size} is not a non-zero power of two"
            )));
        }
        let code = compiler.compile_cubin(SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let project_kn = module.get_function("nnis_project_kn_f32_int2_weight")?;
        let attributes = project_kn.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "f32-int2 project block size {block_size} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        Ok(Self {
            project_kn,
            block_size,
        })
    }

    #[must_use]
    pub const fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Execute one packed-INT2 projection and synchronize.
    #[allow(clippy::too_many_arguments)]
    pub fn project_kn(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        packed_weight: &DeviceBuffer<u8>,
        scale: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result =
            unsafe { self.enqueue_project_kn(stream, input, packed_weight, scale, output, k, n) };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue one packed-INT2 projection without synchronizing.
    ///
    /// # Safety
    ///
    /// All buffers, the stream and this kernel must remain alive and otherwise
    /// untouched until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_project_kn(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        packed_weight: &DeviceBuffer<u8>,
        scale: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        self.validate_execution(stream, input, packed_weight, scale, output, k, n)?;
        if n == 0 {
            return Ok(());
        }
        if k == 0 {
            // SAFETY: the output lifetime obligation is documented above.
            return unsafe { output.zero_async(stream) };
        }

        let k_arg = u64::try_from(k)
            .map_err(|_| NnisError::invalid_input("f32-int2 project K exceeds u64::MAX"))?;
        let n_arg = u64::try_from(n)
            .map_err(|_| NnisError::invalid_input("f32-int2 project N exceeds u64::MAX"))?;
        let config = LaunchConfig::for_num_elements(n, self.block_size)?;
        let mut arguments = KernelArgs::with_capacity(6, 4);
        arguments
            .push_buffer(input)
            .push_buffer(packed_weight)
            .push_buffer(scale)
            .push_buffer(output)
            .push(k_arg)
            .push(n_arg);
        let launch = KernelLaunch::new(&self.project_kn, stream, config);
        // SAFETY: argument order and widths match the CUDA kernel. The caller
        // owns the remaining asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_execution(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        packed_weight: &DeviceBuffer<u8>,
        scale: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let logical_weight_elements = k
            .checked_mul(n)
            .ok_or_else(|| NnisError::invalid_input("f32-int2 project shape overflows usize"))?;
        let expected_packed = logical_weight_elements
            .checked_add(3)
            .ok_or_else(|| NnisError::invalid_input("f32-int2 packed length overflows usize"))?
            / 4;
        if input.len() != k {
            return Err(NnisError::invalid_input(format!(
                "f32-int2 project input has {} elements; shape requires {k}",
                input.len()
            )));
        }
        if packed_weight.len() != expected_packed {
            return Err(NnisError::invalid_input(format!(
                "f32-int2 project packed weight has {} bytes; shape ({k}, {n}) requires {expected_packed}",
                packed_weight.len()
            )));
        }
        if scale.len() != 1 {
            return Err(NnisError::invalid_input(format!(
                "f32-int2 project scale buffer has {} values; exactly one is required",
                scale.len()
            )));
        }
        if output.len() != n {
            return Err(NnisError::invalid_input(format!(
                "f32-int2 project output has {} elements; shape requires {n}",
                output.len()
            )));
        }
        let context = self.project_kn.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, input.ctx())
            || !Arc::ptr_eq(context, packed_weight.ctx())
            || !Arc::ptr_eq(context, scale.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "f32-int2 project stream, buffers and kernel must share one context",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    fn quantize_reference(values: &[f32]) -> (Vec<u8>, f32, Vec<f32>) {
        let max_abs = values.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs };
        let threshold = scale * 0.5;
        let mut packed = vec![0_u8; values.len().div_ceil(4)];
        let mut reconstructed = Vec::with_capacity(values.len());
        for (index, value) in values.iter().copied().enumerate() {
            let q = if value >= threshold {
                1_i8
            } else if value <= -threshold {
                -1_i8
            } else {
                0_i8
            };
            let code = match q {
                1 => 1_u8,
                -1 => 2_u8,
                _ => 0_u8,
            };
            packed[index / 4] |= code << ((index % 4) * 2);
            reconstructed.push(f32::from(q) * scale);
        }
        (packed, scale, reconstructed)
    }

    fn ordered_projection(input: &[f32], weight: &[f32], k: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|col| {
                (0..k).fold(0.0_f32, |value, row| {
                    input[row].mul_add(weight[row * n + col], value)
                })
            })
            .collect()
    }

    #[test]
    fn int2_projection_matches_quantized_ordered_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let k = 17_usize;
        let n = 23_usize;
        let input_host = (0..k)
            .map(|index| ((index * 11 % 29) as f32 - 14.0) * 0.0625)
            .collect::<Vec<_>>();
        let weight_host = (0..k * n)
            .map(|index| (((index * 17 % 101) as f32 - 50.0) * 0.03125) + 0.013)
            .collect::<Vec<_>>();
        let (packed_host, scale_host, reconstructed) = quantize_reference(&weight_host);
        let expected = ordered_projection(&input_host, &reconstructed, k, n);

        let compiler = JitCompiler::new();
        let kernel = F32Int2Gemv::load_with_block_size(&context, &compiler, 64).unwrap();
        let stream = Stream::new(&context).unwrap();
        let input = DeviceBuffer::from_host(&context, &stream, &input_host).unwrap();
        let packed = DeviceBuffer::from_host(&context, &stream, &packed_host).unwrap();
        let scale = DeviceBuffer::from_host(&context, &stream, &[scale_host]).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, n).unwrap();

        kernel
            .project_kn(&stream, &input, &packed, &scale, &output, k, n)
            .unwrap();
        let actual = output.to_vec(&stream).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "INT2 ordered projection mismatch at output {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn int2_projection_rejects_shape_contract_violations_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(F32Int2Gemv::load_with_block_size(&context, &compiler, 0).is_err());
        assert!(F32Int2Gemv::load_with_block_size(&context, &compiler, 48).is_err());

        let kernel = F32Int2Gemv::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let input = DeviceBuffer::<f32>::new(&context, 4).unwrap();
        let packed = DeviceBuffer::<u8>::new(&context, 3).unwrap();
        let scale = DeviceBuffer::<f32>::new(&context, 1).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 3).unwrap();

        let error = kernel
            .project_kn(&stream, &input, &packed, &scale, &output, 4, 4)
            .unwrap_err();
        assert!(error.to_string().contains("requires 4"), "{error}");

        let packed = DeviceBuffer::<u8>::new(&context, 4).unwrap();
        let bad_scale = DeviceBuffer::<f32>::new(&context, 2).unwrap();
        let error = kernel
            .project_kn(&stream, &input, &packed, &bad_scale, &output, 4, 4)
            .unwrap_err();
        assert!(error.to_string().contains("exactly one"), "{error}");
    }
}
