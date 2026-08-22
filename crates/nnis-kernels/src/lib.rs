//! Reusable NVIDIA-native inference kernels built through [`nnis_jit`].
//!
//! The first kernel family provides `f32` elementwise primitives. It exposes
//! safe, synchronizing operations for ordinary use and explicitly unsafe
//! enqueue operations for callers that manage asynchronous buffer lifetimes.

use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
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
        let options = CompileOptions::for_device(context);
        let code = compiler.compile_cubin(ELEMENTWISE_SOURCE, &options)?;
        let module = Module::load(context, &code)?;
        Ok(Self {
            vector_add: module.get_function("nnis_vector_add_f32")?,
            scale: module.get_function("nnis_scale_f32")?,
            affine: module.get_function("nnis_affine_f32")?,
            block_size: DEFAULT_BLOCK_SIZE,
        })
    }

    /// CUDA thread-block width used by this kernel family.
    pub fn block_size(&self) -> u32 {
        self.block_size
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
        let mut args = KernelArgs::new();
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
        let mut args = KernelArgs::new();
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
        let mut args = KernelArgs::new();
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
