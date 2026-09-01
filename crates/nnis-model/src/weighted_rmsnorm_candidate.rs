//! Candidate-only weighted F32 RMSNorm with an explicit launch width.
//!
//! The production decoder continues to use [`crate::F32DecoderKernels`] and its
//! historical block size.  This type exists only so destination-owned
//! qualification code (including Forge adapters) can compare launch widths for
//! the *same weighted RMSNorm semantics used by the decoder* before any runtime
//! plan is considered.
//!
//! No block size chosen through this API is a runtime-default promotion.

use nnis_jit::{CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const WEIGHTED_RMSNORM_SOURCE: &str = r#"
extern "C" __global__ void nnis_weighted_rmsnorm_f32_candidate(
    const float* input,
    const float* weight,
    float* output,
    unsigned long long cols,
    float inv_cols,
    float epsilon
) {
    extern __shared__ float partial[];
    const unsigned int lane = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const float* source = input + row * cols;
    float* destination = output + row * cols;

    float sumsq = 0.0f;
    for (unsigned long long col = lane; col < cols; col += blockDim.x) {
        const float value = source[col];
        sumsq = fmaf(value, value, sumsq);
    }
    partial[lane] = sumsq;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }

    const float scale = rsqrtf(partial[0] * inv_cols + epsilon);
    for (unsigned long long col = lane; col < cols; col += blockDim.x) {
        destination[col] = source[col] * scale * weight[col];
    }
}
"#;

fn validate_block_size(block_size: u32) -> Result<()> {
    if block_size == 0 || !block_size.is_power_of_two() {
        return Err(NnisError::invalid_input(format!(
            "weighted RMSNorm candidate block size {block_size} is not a non-zero power of two"
        )));
    }
    Ok(())
}

/// Explicit-launch candidate for the decoder's weighted F32 RMSNorm semantics.
#[derive(Debug)]
pub struct F32WeightedRmsNormCandidate {
    kernel: Kernel,
    block_size: u32,
}

impl F32WeightedRmsNormCandidate {
    /// Compile and load one explicit launch-width candidate.
    ///
    /// The caller must opt in to the block size. This API has no implicit
    /// hardware policy and does not alter [`crate::F32DecoderKernels::load`].
    pub fn load(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        validate_block_size(block_size)?;
        let code = compiler.compile_cubin(
            WEIGHTED_RMSNORM_SOURCE,
            &CompileOptions::for_device(context),
        )?;
        let module = Module::load(context, &code)?;
        let kernel = module.get_function("nnis_weighted_rmsnorm_f32_candidate")?;
        let attributes = kernel.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "weighted RMSNorm candidate block size {block_size} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        let shared_memory_bytes = block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| {
                NnisError::invalid_input("weighted RMSNorm candidate shared-memory size overflows")
            })?;
        if shared_memory_bytes > attributes.max_dynamic_shared_memory_bytes {
            return Err(NnisError::invalid_input(format!(
                "weighted RMSNorm candidate needs {shared_memory_bytes} dynamic shared-memory bytes; function limit is {}",
                attributes.max_dynamic_shared_memory_bytes
            )));
        }
        Ok(Self { kernel, block_size })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    #[allow(clippy::too_many_arguments)]
    pub fn weighted_rms_norm(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) -> Result<()> {
        // SAFETY: all borrowed resources remain alive until synchronization.
        let result = unsafe {
            self.enqueue_weighted_rms_norm(stream, input, weight, output, rows, cols, epsilon)
        };
        match result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue weighted RMSNorm without synchronizing.
    ///
    /// # Safety
    ///
    /// The stream, kernel and all buffers must remain alive and otherwise
    /// untouched until the stream completes the launch.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_weighted_rms_norm(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) -> Result<()> {
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("weighted RMSNorm candidate shape overflows usize"))?;
        if input.len() != count || output.len() != count || weight.len() != cols {
            return Err(NnisError::invalid_input(format!(
                "weighted RMSNorm candidate expects input/output {count} and weight {cols}; got {}/{}/{}",
                input.len(),
                output.len(),
                weight.len()
            )));
        }
        if !Arc::ptr_eq(self.context(), stream.ctx())
            || !Arc::ptr_eq(self.context(), input.ctx())
            || !Arc::ptr_eq(self.context(), weight.ctx())
            || !Arc::ptr_eq(self.context(), output.ctx())
        {
            return Err(NnisError::invalid_input(
                "weighted RMSNorm candidate, stream and buffers must share one CUDA context",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(NnisError::invalid_input(format!(
                "weighted RMSNorm candidate epsilon must be finite and positive; got {epsilon}"
            )));
        }
        if rows == 0 || cols == 0 {
            return Ok(());
        }

        let grid_rows = u32::try_from(rows)
            .map_err(|_| NnisError::invalid_input("weighted RMSNorm candidate rows exceed u32"))?;
        let shared_memory_bytes = self
            .block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| {
                NnisError::invalid_input("weighted RMSNorm candidate shared-memory size overflows")
            })?;
        let launch_config = LaunchConfig::new(
            Dim3::new(grid_rows, 1, 1),
            Dim3::new(self.block_size, 1, 1),
        )
        .with_dynamic_shared_memory(shared_memory_bytes);
        let cols_u64 = u64::try_from(cols)
            .map_err(|_| NnisError::invalid_input("weighted RMSNorm candidate cols exceed u64"))?;
        let mut args = KernelArgs::with_capacity(6, 3);
        args.push_buffer(input)
            .push_buffer(weight)
            .push_buffer(output)
            .push(cols_u64)
            .push(1.0_f32 / cols as f32)
            .push(epsilon);
        let launch = KernelLaunch::new(&self.kernel, stream, launch_config);
        // SAFETY: argument order and widths match the CUDA entry point; caller
        // owns asynchronous lifetime obligations.
        unsafe { launch.launch(&mut args) }
    }

    fn context(&self) -> &Arc<Context> {
        self.kernel.context()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    #[test]
    fn launch_width_validation_is_fail_closed_without_cuda() {
        assert!(validate_block_size(0).is_err());
        assert!(validate_block_size(3).is_err());
        assert!(validate_block_size(255).is_err());
        assert!(validate_block_size(256).is_ok());
        assert!(validate_block_size(512).is_ok());
    }

    #[test]
    fn block_256_and_512_match_weighted_host_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let stream = Stream::new(&context).unwrap();
        let rows = 2usize;
        let cols = 2048usize;
        let epsilon = 1.0e-6_f32;
        let input_host: Vec<f32> = (0..rows * cols)
            .map(|index| ((index % 97) as f32 - 48.0) * 0.03125)
            .collect();
        let weight_host: Vec<f32> = (0..cols)
            .map(|index| 0.75 + (index % 29) as f32 * 0.0078125)
            .collect();
        let input = DeviceBuffer::from_host(&context, &stream, &input_host).unwrap();
        let weight = DeviceBuffer::from_host(&context, &stream, &weight_host).unwrap();
        let output_256 = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
        let output_512 = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
        let candidate_256 = F32WeightedRmsNormCandidate::load(&context, &compiler, 256).unwrap();
        let candidate_512 = F32WeightedRmsNormCandidate::load(&context, &compiler, 512).unwrap();
        candidate_256
            .weighted_rms_norm(
                &stream,
                &input,
                &weight,
                &output_256,
                rows,
                cols,
                epsilon,
            )
            .unwrap();
        candidate_512
            .weighted_rms_norm(
                &stream,
                &input,
                &weight,
                &output_512,
                rows,
                cols,
                epsilon,
            )
            .unwrap();

        let mut expected = Vec::with_capacity(rows * cols);
        for row in input_host.chunks_exact(cols) {
            let mean_square = row
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / cols as f32;
            let inverse = 1.0 / (mean_square + epsilon).sqrt();
            expected.extend(
                row.iter()
                    .zip(weight_host.iter())
                    .map(|(&value, &weight)| value * inverse * weight),
            );
        }
        for actual in [output_256.to_vec(&stream).unwrap(), output_512.to_vec(&stream).unwrap()] {
            for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= 5.0e-5,
                    "mismatch at {index}: {actual} != {expected}"
                );
            }
        }
    }
}
