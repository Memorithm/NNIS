//! Row-batched `f32` RMS normalization for row-major matrices.
//!
//! Each row is transformed as
//! `output = input * rsqrt(mean(input^2) + epsilon) * gamma`.
//! Unlike layer norm there is no mean subtraction and no bias, so one
//! sum-of-squares reduction per row replaces the two-pass statistics.
//!
//! Two execution paths share one CUDA source module:
//!
//! - staged: one block reduces each row's sum of squares into a device
//!   column, then an elementwise kernel applies the scaling
//! - fused: one block owns one entire row through dynamic shared memory,
//!   reading the matrix once and writing it once
//!
//! Safe wrappers synchronize exactly once; enqueue variants never do.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const RMS_NORM_SOURCE: &str = r#"
extern "C" __global__ void nnis_rmsnorm_row_sumsq_f32(
    const float* input,
    float* row_sumsq,
    unsigned long long cols
) {
    extern __shared__ float partial[];
    const unsigned int lane = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const float* row_input = input + row * cols;

    float value = 0.0f;
    for (unsigned long long index = lane; index < cols; index += blockDim.x) {
        value += row_input[index] * row_input[index];
    }
    partial[lane] = value;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        row_sumsq[row] = partial[0];
    }
}

extern "C" __global__ void nnis_rmsnorm_normalize_f32(
    const float* input,
    float* output,
    const float* row_sumsq,
    unsigned long long rows,
    unsigned long long cols,
    float inv_cols,
    float epsilon,
    float gamma
) {
    const unsigned long long count = rows * cols;
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned long long row = index / cols;
        const float scale = rsqrtf(row_sumsq[row] * inv_cols + epsilon);
        output[index] = (input[index] * scale) * gamma;
    }
}

// One block owns one entire row: the row is staged in dynamic shared
// memory, so the matrix is read once and written once. Shared layout is
// [row values (cols floats)][reduction partials (blockDim.x floats)].
extern "C" __global__ void nnis_rmsnorm_row_fused_f32(
    const float* input,
    float* output,
    unsigned long long cols,
    float inv_cols,
    float epsilon,
    float gamma
) {
    extern __shared__ float shared[];
    float* values = shared;
    float* partial = shared + cols;

    const unsigned int lane = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const float* source = input + row * cols;
    float* destination = output + row * cols;

    for (unsigned long long index = lane; index < cols; index += blockDim.x) {
        values[index] = source[index];
    }
    __syncthreads();

    float value = 0.0f;
    for (unsigned long long index = lane; index < cols; index += blockDim.x) {
        value += values[index] * values[index];
    }
    partial[lane] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] * inv_cols + epsilon);
    __syncthreads();

    for (unsigned long long index = lane; index < cols; index += blockDim.x) {
        destination[index] = (values[index] * scale) * gamma;
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Context-bound row-batched `f32` RMS normalization.
///
/// A [`F32RmsNormWorkspace`] holds the device-side per-row sums of squares
/// and may be reused across non-overlapping calls with equal or smaller
/// row counts.
#[derive(Debug)]
pub struct F32RmsNorm {
    row_sumsq: Kernel,
    normalize: Kernel,
    fused: Kernel,
    block_size: u32,
}

/// Reusable per-row sum-of-squares storage for [`F32RmsNorm`].
#[derive(Debug)]
pub struct F32RmsNormWorkspace {
    max_rows: usize,
    row_sumsq: DeviceBuffer<f32>,
}

impl F32RmsNormWorkspace {
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    fn context(&self) -> &Arc<Context> {
        self.row_sumsq.ctx()
    }
}

impl F32RmsNorm {
    /// Compile (or reuse cached CUBIN) and load the RMS-norm kernel set.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load the family with an explicitly selected thread-block width.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(NnisError::invalid_input(format!(
                "rms-norm block size {block_size} is not a non-zero power of two"
            )));
        }
        let shared_memory_bytes = block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| NnisError::invalid_input("rms-norm shared memory overflows"))?;
        let code = compiler.compile_cubin(RMS_NORM_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let row_sumsq = module.get_function("nnis_rmsnorm_row_sumsq_f32")?;
        let normalize = module.get_function("nnis_rmsnorm_normalize_f32")?;
        let fused = module.get_function("nnis_rmsnorm_row_fused_f32")?;
        for (name, function) in [
            ("row_sumsq", &row_sumsq),
            ("normalize", &normalize),
            ("fused", &fused),
        ] {
            let attributes = function.attributes()?;
            if block_size > attributes.max_threads_per_block {
                return Err(NnisError::invalid_input(format!(
                    "rms-norm {name} block size {block_size} exceeds function limit {}",
                    attributes.max_threads_per_block
                )));
            }
            if name != "normalize"
                && shared_memory_bytes > attributes.max_dynamic_shared_memory_bytes
            {
                return Err(NnisError::invalid_input(format!(
                    "rms-norm {name} requires {shared_memory_bytes} shared-memory bytes; \
                     function limit is {}",
                    attributes.max_dynamic_shared_memory_bytes
                )));
            }
        }
        Ok(Self {
            row_sumsq,
            normalize,
            fused,
            block_size,
        })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Whether the fused single-kernel path can hold a row of `cols`
    /// floats plus its reduction partials in dynamic shared memory.
    pub fn fused_available(&self, cols: usize) -> bool {
        let Ok(attributes) = self.fused.attributes() else {
            return false;
        };
        self.fused_shared_memory(cols)
            .map(|bytes| bytes <= attributes.max_dynamic_shared_memory_bytes as usize)
            .unwrap_or(false)
    }

    /// Normalize each row choosing the best available path and wait once:
    /// the fused single kernel when the row fits dynamic shared memory,
    /// otherwise the staged pipeline with freshly allocated scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn normalize_rows_dispatched(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
        gamma: f32,
    ) -> Result<()> {
        if self.fused_available(cols) {
            return self.fused_normalize_rows(stream, input, output, rows, cols, epsilon, gamma);
        }
        let workspace = self.workspace(stream.ctx(), rows)?;
        self.normalize_rows(
            stream, input, output, rows, cols, epsilon, gamma, &workspace,
        )
    }

    /// Allocate per-row sum-of-squares storage reusable up to `max_rows`.
    pub fn workspace(
        &self,
        context: &Arc<Context>,
        max_rows: usize,
    ) -> Result<F32RmsNormWorkspace> {
        if !Arc::ptr_eq(context, self.context()) {
            return Err(NnisError::invalid_input(
                "rms-norm and workspace contexts do not match",
            ));
        }
        Ok(F32RmsNormWorkspace {
            max_rows,
            row_sumsq: DeviceBuffer::new(context, max_rows)?,
        })
    }

    /// Normalize each row into `output` through the staged pipeline and
    /// wait once.
    #[allow(clippy::too_many_arguments)]
    pub fn normalize_rows(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
        gamma: f32,
        workspace: &F32RmsNormWorkspace,
    ) -> Result<()> {
        // SAFETY: every buffer borrow remains live until synchronization.
        let enqueue_result = unsafe {
            self.enqueue_normalize_rows(
                stream, input, output, rows, cols, epsilon, gamma, workspace,
            )
        };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Normalize each row with the single-kernel fused path and wait once.
    ///
    /// The row must fit in dynamic shared memory together with the
    /// reduction partials: `(cols + block_size) * 4` bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_normalize_rows(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
        gamma: f32,
    ) -> Result<()> {
        // SAFETY: every buffer borrow remains live until synchronization.
        let enqueue_result =
            unsafe { self.enqueue_fused_rows(stream, input, output, rows, cols, epsilon, gamma) };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue the staged pipeline without synchronizing.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, this kernel set, and the workspace must
    /// remain alive and otherwise untouched until the stream completes.
    /// The workspace may not be shared by overlapping operations,
    /// including on other streams.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_normalize_rows(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
        gamma: f32,
        workspace: &F32RmsNormWorkspace,
    ) -> Result<()> {
        self.validate_execution(stream, input, output, rows, cols, workspace)?;
        if rows == 0 || cols == 0 {
            return Ok(());
        }

        // SAFETY: stages are ordered on one stream; the normalize stage
        // reads only stats-stage output under this method's contract.
        unsafe {
            self.launch_row_sumsq(stream, input, &workspace.row_sumsq, rows, cols)?;
            self.launch_normalize(
                stream,
                input,
                output,
                &workspace.row_sumsq,
                rows,
                cols,
                epsilon,
                gamma,
            )?;
        }
        Ok(())
    }

    /// Enqueue the single-kernel fused path without synchronizing.
    ///
    /// # Safety
    ///
    /// All buffers and the stream must remain alive and otherwise untouched
    /// until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_fused_rows(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
        gamma: f32,
    ) -> Result<()> {
        self.validate_shapes(stream, input, output, rows, cols)?;
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let attributes = self.fused.attributes()?;
        let shared_memory_bytes = self.fused_shared_memory(cols)?;
        if shared_memory_bytes > attributes.max_dynamic_shared_memory_bytes as usize {
            return Err(NnisError::invalid_input(format!(
                "fused rms norm needs {shared_memory_bytes} shared-memory bytes for \
                 {cols} columns; function limit is {}",
                attributes.max_dynamic_shared_memory_bytes
            )));
        }
        // SAFETY: the caller owns the asynchronous lifetime of both buffers.
        unsafe { self.launch_fused(stream, input, output, rows, cols, epsilon, gamma) }
    }

    fn fused_shared_memory(&self, cols: usize) -> Result<usize> {
        let element = std::mem::size_of::<f32>();
        let values = element
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("fused rms norm shared size overflows"))?;
        let partials = element
            .checked_mul(self.block_size as usize)
            .ok_or_else(|| NnisError::invalid_input("fused rms norm shared size overflows"))?;
        values
            .checked_add(partials)
            .ok_or_else(|| NnisError::invalid_input("fused rms norm shared size overflows"))
    }

    fn validate_shapes(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("rms-norm shape overflows usize"))?;
        if input.len() != count {
            return Err(NnisError::invalid_input(format!(
                "rms-norm input has {} elements; shape ({rows}, {cols}) requires {count}",
                input.len()
            )));
        }
        if output.len() != count {
            return Err(NnisError::invalid_input(format!(
                "rms-norm output has {} elements; shape ({rows}, {cols}) requires {count}",
                output.len()
            )));
        }
        if !Arc::ptr_eq(self.context(), stream.ctx())
            || !Arc::ptr_eq(self.context(), input.ctx())
            || !Arc::ptr_eq(self.context(), output.ctx())
        {
            return Err(NnisError::invalid_input(
                "rms-norm stream, buffers, and kernels must share one context",
            ));
        }
        Ok(())
    }

    fn validate_execution(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        workspace: &F32RmsNormWorkspace,
    ) -> Result<()> {
        self.validate_shapes(stream, input, output, rows, cols)?;
        if rows > workspace.max_rows {
            return Err(NnisError::invalid_input(format!(
                "rms-norm has {rows} rows; workspace capacity is {}",
                workspace.max_rows
            )));
        }
        if !Arc::ptr_eq(self.context(), workspace.context()) {
            return Err(NnisError::invalid_input(
                "rms-norm stream, buffers, and kernels must share one context",
            ));
        }
        Ok(())
    }

    /// # Safety
    ///
    /// Caller owns the asynchronous lifetime of every buffer; see
    /// [`F32RmsNorm::enqueue_normalize_rows`].
    unsafe fn launch_row_sumsq(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        row_sumsq: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let grid_size = u32::try_from(rows)
            .map_err(|_| NnisError::invalid_input("rms-norm exceeds u32::MAX rows"))?;
        let cols = u64::try_from(cols)
            .map_err(|_| NnisError::invalid_input("rms-norm width exceeds u64::MAX"))?;
        let shared_memory_bytes = self
            .block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| NnisError::invalid_input("rms-norm shared memory overflows"))?;
        let config = LaunchConfig::new(Dim3::x(grid_size), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(shared_memory_bytes);
        let mut arguments = KernelArgs::with_capacity(3, 2);
        arguments
            .push_buffer(input)
            .push_buffer(row_sumsq)
            .push(cols);
        let launch = KernelLaunch::new(&self.row_sumsq, stream, config);
        // SAFETY: argument order/widths match `nnis_rmsnorm_row_sumsq_f32`;
        // the caller owns lifetimes.
        unsafe { launch.launch(&mut arguments) }
    }

    /// # Safety
    ///
    /// Caller owns the asynchronous lifetime of every buffer; see
    /// [`F32RmsNorm::enqueue_normalize_rows`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_normalize(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        row_sumsq: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
        gamma: f32,
    ) -> Result<()> {
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("rms-norm length overflows usize"))?;
        let (rows, cols) = self.u64_shape(rows, cols)?;
        let mut arguments = KernelArgs::with_capacity(8, 3);
        arguments
            .push_buffer(input)
            .push_buffer(output)
            .push_buffer(row_sumsq)
            .push(rows)
            .push(cols)
            .push(1.0_f32 / cols as f32)
            .push(epsilon)
            .push(gamma);
        let launch = KernelLaunch::new(
            &self.normalize,
            stream,
            LaunchConfig::for_num_elements(count, self.block_size)?,
        );
        // SAFETY: argument order/widths match `nnis_rmsnorm_normalize_f32`;
        // same-stream ordering guarantees each row's sum is final before use.
        unsafe { launch.launch(&mut arguments) }
    }

    fn u64_shape(&self, rows: usize, cols: usize) -> Result<(u64, u64)> {
        let rows = u64::try_from(rows)
            .map_err(|_| NnisError::invalid_input("rms-norm exceeds u64::MAX rows"))?;
        let cols = u64::try_from(cols)
            .map_err(|_| NnisError::invalid_input("rms-norm width exceeds u64::MAX"))?;
        Ok((rows, cols))
    }

    /// # Safety
    ///
    /// Caller owns the asynchronous lifetime of every buffer; see
    /// [`F32RmsNorm::enqueue_fused_rows`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_fused(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
        gamma: f32,
    ) -> Result<()> {
        let grid_size = u32::try_from(rows)
            .map_err(|_| NnisError::invalid_input("rms-norm exceeds u32::MAX rows"))?;
        let cols_u64 = u64::try_from(cols)
            .map_err(|_| NnisError::invalid_input("rms-norm width exceeds u64::MAX"))?;
        let shared_memory_bytes = self.fused_shared_memory(cols)?;
        let config = LaunchConfig::new(Dim3::x(grid_size), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(
                u32::try_from(shared_memory_bytes)
                    .map_err(|_| NnisError::invalid_input("fused shared memory exceeds u32"))?,
            );
        let mut arguments = KernelArgs::with_capacity(6, 2);
        arguments
            .push_buffer(input)
            .push_buffer(output)
            .push(cols_u64)
            .push(1.0_f32 / cols as f32)
            .push(epsilon)
            .push(gamma);
        let launch = KernelLaunch::new(&self.fused, stream, config);
        // SAFETY: argument order/widths match `nnis_rmsnorm_row_fused_f32`;
        // the caller owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    fn context(&self) -> &Arc<Context> {
        self.row_sumsq.context()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    const SHAPES: &[(usize, usize)] = &[
        (1, 1),
        (1, 7),
        (2, 31),
        (3, 32),
        (5, 255),
        (4, 256),
        (7, 257),
        (13, 1_025),
        (17, 4_097),
    ];

    const EPSILON: f32 = 1.0e-6;
    const GAMMA: f32 = 1.625;

    fn host_values(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|index| {
                let drift = ((index % 17) as f32 - 8.0) * 11.25;
                let ripple = ((index * 7 % 89) as f32 - 44.0) * 0.0625;
                drift + ripple
            })
            .collect()
    }

    /// Replays the kernel's evaluation order: strided per-lane accumulation
    /// of squares, shared-memory tree reduction, then the affine scaling.
    fn reference_rows(input: &[f32], rows: usize, cols: usize) -> Vec<f64> {
        input
            .chunks(cols)
            .take(rows)
            .flat_map(|slice| {
                let count = slice.len() as f64;
                let mean_square: f64 = slice
                    .iter()
                    .map(|&value| {
                        let widened = f64::from(value);
                        widened * widened
                    })
                    .sum::<f64>()
                    / count;
                let scale = (mean_square + f64::from(EPSILON)).sqrt().recip();
                slice
                    .iter()
                    .map(move |&value| f64::from(value) * scale * f64::from(GAMMA))
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f64], label: &str) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 2.0e-6_f32.max((expected.abs() as f32) * 1.0e-5);
            assert!(
                (actual - expected as f32).abs() <= tolerance,
                "{label} mismatch at {index}: {actual} != {expected}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn rms_norm_matches_high_precision_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let rms_norm = F32RmsNorm::load(&context, &compiler).unwrap();
        let max_rows = SHAPES.iter().map(|&(rows, _)| rows).max().unwrap();
        let workspace = rms_norm.workspace(&context, max_rows).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(rows, cols) in SHAPES {
            let host = host_values(rows, cols);
            let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
            let output = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
            rms_norm
                .normalize_rows(
                    &stream, &input, &output, rows, cols, EPSILON, GAMMA, &workspace,
                )
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            assert_close(
                &actual,
                &reference_rows(&host, rows, cols),
                &format!("staged ({rows}, {cols})"),
            );
        }

        // A constant row c normalizes to c/RMS(c) = sign(c):
        // out = c * gamma / sqrt(c^2 + eps) ~= gamma.
        let (rows, cols) = (4_usize, 129_usize);
        let input =
            DeviceBuffer::from_host(&context, &stream, &vec![3.5_f32; rows * cols]).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
        let small_workspace = rms_norm.workspace(&context, rows).unwrap();
        rms_norm
            .normalize_rows(
                &stream,
                &input,
                &output,
                rows,
                cols,
                EPSILON,
                GAMMA,
                &small_workspace,
            )
            .unwrap();
        let expected = 3.5_f32 / (3.5_f32 * 3.5_f32 + EPSILON).sqrt() * GAMMA;
        for (index, value) in output.to_vec(&stream).unwrap().iter().enumerate() {
            assert!(
                (value - expected).abs() <= 1.0e-4,
                "constant-row mismatch at {index}: {value} != {expected}"
            );
        }
    }

    #[test]
    fn fused_rms_norm_matches_high_precision_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let rms_norm = F32RmsNorm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(rows, cols) in SHAPES {
            let host = host_values(rows, cols);
            let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
            let output = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
            rms_norm
                .fused_normalize_rows(&stream, &input, &output, rows, cols, EPSILON, GAMMA)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            assert_close(
                &actual,
                &reference_rows(&host, rows, cols),
                &format!("fused ({rows}, {cols})"),
            );
        }
    }

    #[test]
    fn fused_rms_norm_rejects_oversized_rows() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let rms_norm = F32RmsNorm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        // (cols + block_size) * 4 bytes must fit the function's dynamic
        // shared-memory limit; 20,000 columns cannot.
        let cols = 20_000_usize;
        let rows = 2_usize;
        let input = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
        let error = rms_norm
            .fused_normalize_rows(&stream, &input, &output, rows, cols, EPSILON, GAMMA)
            .unwrap_err();
        assert!(error.to_string().contains("shared-memory"), "{error}");
    }

    #[test]
    fn dispatched_path_matches_oracle_and_selects_fused_when_possible() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let rms_norm = F32RmsNorm::load(&context, &compiler).unwrap();
        assert!(rms_norm.fused_available(2_048));
        // A row that cannot fit shared memory must still normalize correctly
        // through the staged fallback.
        assert!(!rms_norm.fused_available(20_000));
        let stream = Stream::new(&context).unwrap();

        for &(rows, cols, fused_expected) in &[
            (5_usize, 2_048_usize, true),
            (3_usize, 257_usize, true),
            (2_usize, 20_000_usize, false),
        ] {
            let host = host_values(rows, cols);
            let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
            let output = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
            rms_norm
                .normalize_rows_dispatched(&stream, &input, &output, rows, cols, EPSILON, GAMMA)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            assert_close(
                &actual,
                &reference_rows(&host, rows, cols),
                &format!("dispatched ({rows}, {cols})"),
            );
            assert_eq!(fused_expected, rms_norm.fused_available(cols));
        }
    }

    #[test]
    fn rms_norm_rejects_invalid_shapes_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(F32RmsNorm::load_with_block_size(&context, &compiler, 0).is_err());
        assert!(F32RmsNorm::load_with_block_size(&context, &compiler, 300).is_err());
        let rms_norm = F32RmsNorm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let input = DeviceBuffer::<f32>::new(&context, 16).unwrap();
        let short_output = DeviceBuffer::<f32>::new(&context, 10).unwrap();
        let workspace = rms_norm.workspace(&context, 3).unwrap();
        let error = rms_norm
            .normalize_rows(
                &stream,
                &input,
                &short_output,
                3,
                4,
                EPSILON,
                GAMMA,
                &workspace,
            )
            .unwrap_err();
        assert!(error.to_string().contains("requires 12"), "{error}");

        let output = DeviceBuffer::<f32>::new(&context, 16).unwrap();
        let error = rms_norm
            .normalize_rows(&stream, &input, &output, 4, 4, EPSILON, GAMMA, &workspace)
            .unwrap_err();
        assert!(
            error.to_string().contains("workspace capacity is 3"),
            "{error}"
        );
    }
}
