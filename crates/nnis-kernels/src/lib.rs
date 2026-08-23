//! Reusable NVIDIA-native inference kernels built through [`nnis_jit`].
//!
//! The first kernel family provides `f32` elementwise primitives. It exposes
//! safe, synchronizing operations for ordinary use and explicitly unsafe
//! enqueue operations for callers that manage asynchronous buffer lifetimes.

mod bf16;
mod gemv;
mod layernorm;
mod reduction;
mod rms_norm;
mod rope;
mod row_softmax;
mod softmax;

pub use bf16::Bf16Elementwise;
pub use gemv::F32Gemv;
pub use layernorm::{F32LayerNorm, F32LayerNormWorkspace};
pub use reduction::{F32Reduction, F32ReductionWorkspace};
pub use rms_norm::F32RmsNorm;
pub use rope::F32Rope;
pub use row_softmax::{F32Softmax2D, F32Softmax2DWorkspace};
pub use softmax::F32Softmax;

use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
    OccupancyRecommendation,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const ELEMENTWISE_SOURCE: &str = r#"
extern "C" __global__ void nnis_vector_add_f32(
    const float* left,
    const float* right,
    float* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = left[index] + right[index];
    }
}

extern "C" __global__ void nnis_scale_f32(
    const float* input,
    float* output,
    float scale,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = input[index] * scale;
    }
}

extern "C" __global__ void nnis_affine_f32(
    const float* input,
    float* output,
    float scale,
    float bias,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = fmaf(input[index], scale, bias);
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Occupancy recommendations for each kernel in [`F32Elementwise`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32ElementwiseOccupancy {
    pub vector_add: OccupancyRecommendation,
    pub scale: OccupancyRecommendation,
    pub affine: OccupancyRecommendation,
}

/// Active blocks per multiprocessor for this kernel set's configured width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32ElementwiseActiveBlocks {
    pub vector_add: u32,
    pub scale: u32,
    pub affine: u32,
}

/// Context-bound `f32` elementwise kernels compiled for the active GPU.
#[derive(Debug)]
pub struct F32Elementwise {
    vector_add: Kernel,
    scale: Kernel,
    affine: Kernel,
    block_size: u32,
}

impl F32Elementwise {
    /// Compile (or reuse cached CUBIN) and load the elementwise kernel module.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load the kernel family with one explicitly selected thread-block width.
    ///
    /// This constructor supports reproducible tuning while keeping launch
    /// details behind the operation API. The width is checked against every
    /// function's compiled CUDA limit before the kernel set is returned.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 {
            return Err(NnisError::invalid_input("elementwise block size is zero"));
        }
        let options = CompileOptions::for_device(context);
        let code = compiler.compile_cubin(ELEMENTWISE_SOURCE, &options)?;
        let module = Module::load(context, &code)?;
        let vector_add = module.get_function("nnis_vector_add_f32")?;
        let scale = module.get_function("nnis_scale_f32")?;
        let affine = module.get_function("nnis_affine_f32")?;
        validate_block_size("vector_add", &vector_add, block_size)?;
        validate_block_size("scale", &scale, block_size)?;
        validate_block_size("affine", &affine, block_size)?;
        Ok(Self {
            vector_add,
            scale,
            affine,
            block_size,
        })
    }

    /// CUDA thread-block width used by this kernel family.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// CUDA's occupancy-based launch recommendation for every operation.
    pub fn occupancy(&self) -> Result<F32ElementwiseOccupancy> {
        Ok(F32ElementwiseOccupancy {
            vector_add: self.vector_add.recommend_occupancy(0, None)?,
            scale: self.scale.recommend_occupancy(0, None)?,
            affine: self.affine.recommend_occupancy(0, None)?,
        })
    }

    /// Resource-limited active blocks per SM at the configured block width.
    pub fn active_blocks_per_multiprocessor(&self) -> Result<F32ElementwiseActiveBlocks> {
        Ok(F32ElementwiseActiveBlocks {
            vector_add: self
                .vector_add
                .max_active_blocks_per_multiprocessor(self.block_size, 0)?,
            scale: self
                .scale
                .max_active_blocks_per_multiprocessor(self.block_size, 0)?,
            affine: self
                .affine
                .max_active_blocks_per_multiprocessor(self.block_size, 0)?,
        })
    }

    /// Add two equal-length buffers and wait for completion.
    pub fn vector_add(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        // SAFETY: this method retains every buffer borrow until synchronization.
        unsafe { self.enqueue_vector_add(stream, left, right, output)? };
        stream.synchronize()
    }

    /// Multiply each input element by `scale` and wait for completion.
    pub fn scale(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        scale: f32,
    ) -> Result<()> {
        // SAFETY: this method retains every buffer borrow until synchronization.
        unsafe { self.enqueue_scale(stream, input, output, scale)? };
        stream.synchronize()
    }

    /// Compute `output = input * scale + bias` with one rounded FMA and wait.
    pub fn affine(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        scale: f32,
        bias: f32,
    ) -> Result<()> {
        // SAFETY: this method retains every buffer borrow until synchronization.
        unsafe { self.enqueue_affine(stream, input, output, scale, bias)? };
        stream.synchronize()
    }

    /// Enqueue vector addition without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// The stream, all three buffers, and this kernel set must remain alive and
    /// otherwise untouched until the stream has completed the launch.
    pub unsafe fn enqueue_vector_add(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        validate_lengths(
            "vector_add",
            &[
                ("left", left.len()),
                ("right", right.len()),
                ("output", output.len()),
            ],
        )?;
        if output.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(4, 3);
        args.push_buffer(left)
            .push_buffer(right)
            .push_buffer(output)
            .push(output.len() as u64);
        let launch = KernelLaunch::new(
            &self.vector_add,
            stream,
            LaunchConfig::for_num_elements(output.len(), self.block_size)?,
        );
        // SAFETY: argument order and widths exactly match `nnis_vector_add_f32`;
        // the remaining asynchronous lifetime obligation is documented above.
        unsafe { launch.launch(&mut args) }
    }

    /// Enqueue elementwise scaling without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// The stream, both buffers, and this kernel set must remain alive and
    /// otherwise untouched until the stream has completed the launch.
    pub unsafe fn enqueue_scale(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        scale: f32,
    ) -> Result<()> {
        validate_lengths("scale", &[("input", input.len()), ("output", output.len())])?;
        if output.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(4, 2);
        args.push_buffer(input)
            .push_buffer(output)
            .push(scale)
            .push(output.len() as u64);
        let launch = KernelLaunch::new(
            &self.scale,
            stream,
            LaunchConfig::for_num_elements(output.len(), self.block_size)?,
        );
        // SAFETY: argument order and widths exactly match `nnis_scale_f32`;
        // the remaining asynchronous lifetime obligation is documented above.
        unsafe { launch.launch(&mut args) }
    }

    /// Enqueue fused affine transformation without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// The stream, both buffers, and this kernel set must remain alive and
    /// otherwise untouched until the stream has completed the launch.
    pub unsafe fn enqueue_affine(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        scale: f32,
        bias: f32,
    ) -> Result<()> {
        validate_lengths(
            "affine",
            &[("input", input.len()), ("output", output.len())],
        )?;
        if output.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(5, 2);
        args.push_buffer(input)
            .push_buffer(output)
            .push(scale)
            .push(bias)
            .push(output.len() as u64);
        let launch = KernelLaunch::new(
            &self.affine,
            stream,
            LaunchConfig::for_num_elements(output.len(), self.block_size)?,
        );
        // SAFETY: argument order and widths exactly match `nnis_affine_f32`;
        // the remaining asynchronous lifetime obligation is documented above.
        unsafe { launch.launch(&mut args) }
    }
}

fn validate_block_size(operation: &str, kernel: &Kernel, block_size: u32) -> Result<()> {
    if block_size == 0 {
        return Err(NnisError::invalid_input(format!(
            "{operation} block size is zero"
        )));
    }
    let maximum = kernel.attributes()?.max_threads_per_block;
    if block_size > maximum {
        return Err(NnisError::invalid_input(format!(
            "{operation} block size {block_size} exceeds function limit {maximum}"
        )));
    }
    Ok(())
}

fn validate_lengths(operation: &str, buffers: &[(&str, usize)]) -> Result<()> {
    let expected = buffers.first().map_or(0, |(_, length)| *length);
    if let Some((name, actual)) = buffers
        .iter()
        .copied()
        .find(|(_, length)| *length != expected)
    {
        return Err(NnisError::invalid_input(format!(
            "{operation} buffer {name} has {actual} elements; expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    const TEST_SIZES: &[usize] = &[0, 1, 31, 32, 255, 256, 257, 1_025, 4_097];

    fn assert_close(actual: &[f32], expected: &[f32], operation: &str) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 2.0e-6_f32.max(expected.abs() * 2.0e-6);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{operation} mismatch at {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn elementwise_kernels_match_cpu_oracles_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let kernels = F32Elementwise::load(&context, &compiler).unwrap();
        let occupancy = kernels.occupancy().unwrap();
        for recommendation in [occupancy.vector_add, occupancy.scale, occupancy.affine] {
            assert!(recommendation.block_size > 0);
            assert!(recommendation.minimum_grid_size > 0);
            assert!(recommendation.active_blocks_per_multiprocessor > 0);
        }
        let active_blocks = kernels.active_blocks_per_multiprocessor().unwrap();
        assert!(active_blocks.vector_add > 0);
        assert!(active_blocks.scale > 0);
        assert!(active_blocks.affine > 0);
        let stream = Stream::new(&context).unwrap();

        for &size in TEST_SIZES {
            let left_host = (0..size)
                .map(|index| (index as f32 - 37.0) * 0.03125)
                .collect::<Vec<_>>();
            let right_host = (0..size)
                .map(|index| (index as f32 % 19.0) * -0.125 + 0.75)
                .collect::<Vec<_>>();
            let left = DeviceBuffer::from_host(&context, &stream, &left_host).unwrap();
            let right = DeviceBuffer::from_host(&context, &stream, &right_host).unwrap();
            let output = DeviceBuffer::<f32>::new(&context, size).unwrap();

            kernels.vector_add(&stream, &left, &right, &output).unwrap();
            let actual = output.to_vec(&stream).unwrap();
            let expected = left_host
                .iter()
                .zip(&right_host)
                .map(|(&left, &right)| left + right)
                .collect::<Vec<_>>();
            assert_close(&actual, &expected, "vector_add");

            let scale = -0.625_f32;
            kernels.scale(&stream, &left, &output, scale).unwrap();
            let actual = output.to_vec(&stream).unwrap();
            let expected = left_host
                .iter()
                .map(|&value| value * scale)
                .collect::<Vec<_>>();
            assert_close(&actual, &expected, "scale");

            let bias = 1.75_f32;
            kernels
                .affine(&stream, &left, &output, scale, bias)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            let expected = left_host
                .iter()
                .map(|&value| value.mul_add(scale, bias))
                .collect::<Vec<_>>();
            assert_close(&actual, &expected, "affine");
        }
    }

    #[test]
    fn elementwise_rejects_length_mismatches_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(F32Elementwise::load_with_block_size(&context, &compiler, 0).is_err());
        assert!(F32Elementwise::load_with_block_size(
            &context,
            &compiler,
            context.props().max_threads_per_block + 1,
        )
        .is_err());
        let kernels = F32Elementwise::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let short = DeviceBuffer::<f32>::new(&context, 3).unwrap();
        let long = DeviceBuffer::<f32>::new(&context, 4).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 3).unwrap();

        let error = kernels
            .vector_add(&stream, &short, &long, &output)
            .unwrap_err();
        assert!(
            error.to_string().contains("right has 4 elements"),
            "{error}"
        );
    }
}
