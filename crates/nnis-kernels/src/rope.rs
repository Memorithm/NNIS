//! Rotary position embeddings (`f32`) for row-major `[rows, cols]` tensors
//! with even `cols`, plus per-row angle caches of shape `[rows, cols / 2]`.
//!
//! Two pairing conventions ship as separate kernels:
//!
//! - **interleaved** (original RoPE): adjacent pairs `(x[2j], x[2j+1])`
//! - **rotate-half** (GPT-NeoX / Llama): half pairs `(x[j], x[j+half])`
//!
//! Both apply, per pair `j` of row `r` with cache entries `(c, s)`:
//! `even_out = e*c - o*s`, `odd_out = o*c + e*s`. Angle caches are caller
//! supplied so the GPU never evaluates transcendental functions; host and
//! oracle therefore share exact inputs.
//!
//! Safe wrappers synchronize once; enqueue variants never do.

use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const ROPE_SOURCE: &str = r#"
extern "C" __global__ void nnis_rope_interleaved_f32(
    const float* input,
    const float* cos_cache,
    const float* sin_cache,
    float* output,
    unsigned long long rows,
    unsigned long long half
) {
    const unsigned long long count = rows * half;
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned long long row = index / half;
        const unsigned long long j = index % half;
        const float even = input[row * half * 2 + 2 * j];
        const float odd = input[row * half * 2 + 2 * j + 1];
        const float c = cos_cache[index];
        const float s = sin_cache[index];
        output[row * half * 2 + 2 * j] = even * c - odd * s;
        output[row * half * 2 + 2 * j + 1] = odd * c + even * s;
    }
}

extern "C" __global__ void nnis_rope_rotate_half_f32(
    const float* input,
    const float* cos_cache,
    const float* sin_cache,
    float* output,
    unsigned long long rows,
    unsigned long long half
) {
    const unsigned long long count = rows * half;
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned long long row = index / half;
        const unsigned long long j = index % half;
        const unsigned long long base = row * half * 2;
        const float first = input[base + j];
        const float second = input[base + half + j];
        const float c = cos_cache[index];
        const float s = sin_cache[index];
        output[base + j] = first * c - second * s;
        output[base + half + j] = second * c + first * s;
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Context-bound rotary position embedding kernels.
#[derive(Debug)]
pub struct F32Rope {
    interleaved: Kernel,
    rotate_half: Kernel,
    block_size: u32,
}

impl F32Rope {
    /// Compile (or reuse cached CUBIN) and load the rope kernel set.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load the family with an explicitly selected thread-block width.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 {
            return Err(NnisError::invalid_input("rope block size is zero"));
        }
        let code = compiler.compile_cubin(ROPE_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let interleaved = module.get_function("nnis_rope_interleaved_f32")?;
        let rotate_half = module.get_function("nnis_rope_rotate_half_f32")?;
        for (name, function) in [("interleaved", &interleaved), ("rotate_half", &rotate_half)] {
            let attributes = function.attributes()?;
            if block_size > attributes.max_threads_per_block {
                return Err(NnisError::invalid_input(format!(
                    "rope {name} block size {block_size} exceeds function limit {}",
                    attributes.max_threads_per_block
                )));
            }
        }
        Ok(Self {
            interleaved,
            rotate_half,
            block_size,
        })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Apply original-style (adjacent-pair) rotation and wait.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_interleaved(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        cos: &DeviceBuffer<f32>,
        sin: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        // SAFETY: borrows retained until synchronization below.
        let result =
            unsafe { self.enqueue_interleaved(stream, input, cos, sin, output, rows, cols) };
        match result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Apply NeoX/Llama-style (half-split pair) rotation and wait.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_rotate_half(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        cos: &DeviceBuffer<f32>,
        sin: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        // SAFETY: borrows retained until synchronization below.
        let result =
            unsafe { self.enqueue_rotate_half(stream, input, cos, sin, output, rows, cols) };
        match result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue the interleaved kernel without synchronizing.
    ///
    /// # Safety
    ///
    /// All buffers and the stream must remain alive and otherwise untouched
    /// until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_interleaved(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        cos: &DeviceBuffer<f32>,
        sin: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        self.validate(stream, input, cos, sin, output, rows, cols)?;
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(6, 4);
        args.push_buffer(input)
            .push_buffer(cos)
            .push_buffer(sin)
            .push_buffer(output)
            .push(rows as u64)
            .push((cols / 2) as u64);
        let launch = KernelLaunch::new(
            &self.interleaved,
            stream,
            LaunchConfig::for_num_elements(rows * (cols / 2), self.block_size)?,
        );
        // SAFETY: argument order/widths match `nnis_rope_interleaved_f32`.
        unsafe { launch.launch(&mut args) }
    }

    /// Enqueue the rotate-half kernel without synchronizing.
    ///
    /// # Safety
    ///
    /// See [`Self::enqueue_interleaved`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_rotate_half(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        cos: &DeviceBuffer<f32>,
        sin: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        self.validate(stream, input, cos, sin, output, rows, cols)?;
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(6, 4);
        args.push_buffer(input)
            .push_buffer(cos)
            .push_buffer(sin)
            .push_buffer(output)
            .push(rows as u64)
            .push((cols / 2) as u64);
        let launch = KernelLaunch::new(
            &self.rotate_half,
            stream,
            LaunchConfig::for_num_elements(rows * (cols / 2), self.block_size)?,
        );
        // SAFETY: argument order/widths match `nnis_rope_rotate_half_f32`.
        unsafe { launch.launch(&mut args) }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        cos: &DeviceBuffer<f32>,
        sin: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if cols % 2 != 0 {
            return Err(NnisError::invalid_input(format!(
                "rope width must be even; got {cols}"
            )));
        }
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("rope shape overflows usize"))?;
        let cache_len = rows * (cols / 2);
        if input.len() != count {
            return Err(NnisError::invalid_input(format!(
                "rope input has {} elements; shape ({rows}, {cols}) requires {count}",
                input.len()
            )));
        }
        if output.len() != count {
            return Err(NnisError::invalid_input(format!(
                "rope output has {} elements; shape ({rows}, {cols}) requires {count}",
                output.len()
            )));
        }
        for (name, cache) in [("cos", cos), ("sin", sin)] {
            if cache.len() != cache_len {
                return Err(NnisError::invalid_input(format!(
                    "rope {name} cache holds {} elements; expected {cache_len}",
                    cache.len()
                )));
            }
        }
        if !Arc::ptr_eq(self.context(), stream.ctx())
            || !Arc::ptr_eq(self.context(), input.ctx())
            || !Arc::ptr_eq(self.context(), output.ctx())
            || !Arc::ptr_eq(self.context(), cos.ctx())
            || !Arc::ptr_eq(self.context(), sin.ctx())
        {
            return Err(NnisError::invalid_input(
                "rope stream, buffers, and kernels must share one context",
            ));
        }
        Ok(())
    }

    fn context(&self) -> &Arc<Context> {
        self.interleaved.context()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    const SHAPES: &[(usize, usize)] = &[
        (1, 2),
        (1, 8),
        (3, 32),
        (5, 96),
        (7, 128),
        (13, 256),
        (9, 1_024),
    ];

    fn host_values(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| ((i % 53) as f32 - 26.0) * 0.1875)
            .collect()
    }

    fn reference(
        input: &[f32],
        cos: &[f32],
        sin: &[f32],
        rows: usize,
        cols: usize,
        rotate_half: bool,
    ) -> Vec<f64> {
        let half = cols / 2;
        let mut out = vec![0.0_f64; rows * cols];
        for row in 0..rows {
            for j in 0..half {
                let (e, o) = if rotate_half {
                    (
                        f64::from(input[row * cols + j]),
                        f64::from(input[row * cols + half + j]),
                    )
                } else {
                    (
                        f64::from(input[row * cols + 2 * j]),
                        f64::from(input[row * cols + 2 * j + 1]),
                    )
                };
                let idx = row * half + j;
                let (c, s) = (f64::from(cos[idx]), f64::from(sin[idx]));
                let (new_e, new_o) = (e * c - o * s, o * c + e * s);
                if rotate_half {
                    out[row * cols + j] = new_e;
                    out[row * cols + half + j] = new_o;
                } else {
                    out[row * cols + 2 * j] = new_e;
                    out[row * cols + 2 * j + 1] = new_o;
                }
            }
        }
        out
    }

    fn assert_close(actual: &[f32], expected: &[f64], label: &str) {
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 2.0e-6_f32.max((expected.abs() as f32) * 2.0e-6);
            assert!(
                (actual - expected as f32).abs() <= tolerance,
                "{label} mismatch at {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn rope_matches_oracle_for_both_conventions_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let rope = F32Rope::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(rows, cols) in SHAPES {
            let half = cols / 2;
            let host = host_values(rows, cols);
            // Deterministic pseudo-angles covering several periods.
            let cos_host: Vec<f32> = (0..rows * half)
                .map(|i| ((i % 37) as f32 * 0.17).cos())
                .collect();
            let sin_host: Vec<f32> = (0..rows * half)
                .map(|i| ((i % 37) as f32 * 0.17).sin())
                .collect();
            let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
            let cos = DeviceBuffer::from_host(&context, &stream, &cos_host).unwrap();
            let sin = DeviceBuffer::from_host(&context, &stream, &sin_host).unwrap();

            for rotate_half in [false, true] {
                let output = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
                if rotate_half {
                    rope.apply_rotate_half(&stream, &input, &cos, &sin, &output, rows, cols)
                        .unwrap();
                } else {
                    rope.apply_interleaved(&stream, &input, &cos, &sin, &output, rows, cols)
                        .unwrap();
                }
                let actual = output.to_vec(&stream).unwrap();
                assert_close(
                    &actual,
                    &reference(&host, &cos_host, &sin_host, rows, cols, rotate_half),
                    &format!(
                        "{} ({rows}, {cols})",
                        if rotate_half { "half" } else { "inter" }
                    ),
                );
            }
        }

        // Identity caches leave every convention's output equal to input.
        let (rows, cols) = (4_usize, 16_usize);
        let host = host_values(rows, cols);
        let ones = vec![1.0_f32; rows * cols / 2];
        let zeros = vec![0.0_f32; rows * cols / 2];
        let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
        let cos = DeviceBuffer::from_host(&context, &stream, &ones).unwrap();
        let sin = DeviceBuffer::from_host(&context, &stream, &zeros).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
        rope.apply_interleaved(&stream, &input, &cos, &sin, &output, rows, cols)
            .unwrap();
        assert_eq!(output.to_vec(&stream).unwrap(), host);
        rope.apply_rotate_half(&stream, &input, &cos, &sin, &output, rows, cols)
            .unwrap();
        assert_eq!(output.to_vec(&stream).unwrap(), host);
    }

    #[test]
    fn rope_rejects_invalid_shapes_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(F32Rope::load_with_block_size(&context, &compiler, 0).is_err());
        let rope = F32Rope::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let input = DeviceBuffer::<f32>::new(&context, 12).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 12).unwrap();
        let ok_cache = DeviceBuffer::<f32>::new(&context, 6).unwrap();
        let short_cache = DeviceBuffer::<f32>::new(&context, 5).unwrap();

        // Odd width rejected outright.
        let error = rope
            .apply_interleaved(&stream, &input, &ok_cache, &ok_cache, &output, 2, 3)
            .unwrap_err();
        assert!(error.to_string().contains("must be even"), "{error}");

        // Short cache rejected before launch.
        let error = rope
            .apply_rotate_half(&stream, &input, &short_cache, &ok_cache, &output, 2, 6)
            .unwrap_err();
        assert!(error.to_string().contains("holds 5"), "{error}");
    }
}
