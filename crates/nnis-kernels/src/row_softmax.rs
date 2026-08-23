//! Row-batched numerically stable `f32` softmax for row-major matrices.
//!
//! Each stage is one native kernel launch over an `rows x cols` matrix:
//!
//! 1. one thread block reduces each row's maximum into a device column
//! 2. `output = exp(input - row_max[row])`
//! 3. one thread block reduces each row's exponential sum
//! 4. every element divides by its device-resident row total
//!
//! No host synchronization occurs between stages; safe wrappers synchronize
//! exactly once after all four stages.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const ROW_SOFTMAX_SOURCE: &str = r#"
extern "C" __global__ void nnis_softmax_row_max_f32(
    const float* input,
    float* row_max,
    unsigned long long cols
) {
    extern __shared__ float partial[];
    const unsigned int lane = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const float* row_input = input + row * cols;

    float value = __int_as_float(0xff800000);
    for (unsigned long long index = lane; index < cols; index += blockDim.x) {
        value = fmaxf(value, row_input[index]);
    }
    partial[lane] = value;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] = fmaxf(partial[lane], partial[lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0) {
        row_max[row] = partial[0];
    }
}

extern "C" __global__ void nnis_softmax_row_exp_shift_f32(
    const float* input,
    float* output,
    const float* row_max,
    unsigned long long rows,
    unsigned long long cols
) {
    const unsigned long long count = rows * cols;
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned long long row = index / cols;
        output[index] = expf(input[index] - row_max[row]);
    }
}

extern "C" __global__ void nnis_softmax_row_sum_f32(
    const float* input,
    float* row_total,
    unsigned long long cols
) {
    extern __shared__ float partial[];
    const unsigned int lane = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const float* row_input = input + row * cols;

    float value = 0.0f;
    for (unsigned long long index = lane; index < cols; index += blockDim.x) {
        value += row_input[index];
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
        row_total[row] = partial[0];
    }
}

extern "C" __global__ void nnis_softmax_row_normalize_f32(
    float* data,
    const float* row_total,
    unsigned long long rows,
    unsigned long long cols
) {
    const unsigned long long count = rows * cols;
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const unsigned long long row = index / cols;
        data[index] /= row_total[row];
    }
}

// One block owns one entire row: the row is staged in dynamic shared
// memory, so the matrix is read once and written once. Shared layout is
// [row values (cols floats)][reduction partials (blockDim.x floats)].
extern "C" __global__ void nnis_softmax_row_fused_f32(
    const float* input,
    float* output,
    unsigned long long cols
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

    float value = __int_as_float(0xff800000);
    for (unsigned long long index = lane; index < cols; index += blockDim.x) {
        value = fmaxf(value, values[index]);
    }
    partial[lane] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] = fmaxf(partial[lane], partial[lane + stride]);
        }
        __syncthreads();
    }
    const float maximum = partial[0];
    __syncthreads();

    float total = 0.0f;
    for (unsigned long long index = lane; index < cols; index += blockDim.x) {
        const float exponential = expf(values[index] - maximum);
        values[index] = exponential;
        total += exponential;
    }
    partial[lane] = total;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    const float sum = partial[0];
    __syncthreads();

    for (unsigned long long index = lane; index < cols; index += blockDim.x) {
        destination[index] = values[index] / sum;
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Context-bound row-batched stable `f32` softmax.
///
/// A [`F32Softmax2DWorkspace`] holds the device-side per-row scalar columns
/// and may be reused across non-overlapping calls with equal or smaller row
/// counts.
#[derive(Debug)]
pub struct F32Softmax2D {
    row_max: Kernel,
    exp_shift: Kernel,
    row_sum: Kernel,
    normalize: Kernel,
    fused: Kernel,
    block_size: u32,
}

/// Reusable per-row scalar storage for [`F32Softmax2D`].
#[derive(Debug)]
pub struct F32Softmax2DWorkspace {
    max_rows: usize,
    row_scalars: DeviceBuffer<f32>,
}

impl F32Softmax2DWorkspace {
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    fn context(&self) -> &Arc<Context> {
        self.row_scalars.ctx()
    }
}

impl F32Softmax2D {
    /// Compile (or reuse cached CUBINs) and load the row-softmax kernel set.
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
                "row-softmax block size {block_size} is not a non-zero power of two"
            )));
        }
        let shared_memory_bytes = block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| NnisError::invalid_input("row-softmax shared memory overflows"))?;
        let code =
            compiler.compile_cubin(ROW_SOFTMAX_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let row_max = module.get_function("nnis_softmax_row_max_f32")?;
        let exp_shift = module.get_function("nnis_softmax_row_exp_shift_f32")?;
        let row_sum = module.get_function("nnis_softmax_row_sum_f32")?;
        let normalize = module.get_function("nnis_softmax_row_normalize_f32")?;
        let fused = module.get_function("nnis_softmax_row_fused_f32")?;
        for (name, function) in [
            ("row_max", &row_max),
            ("exp_shift", &exp_shift),
            ("row_sum", &row_sum),
            ("normalize", &normalize),
            ("fused", &fused),
        ] {
            let attributes = function.attributes()?;
            if block_size > attributes.max_threads_per_block {
                return Err(NnisError::invalid_input(format!(
                    "row-softmax {name} block size {block_size} exceeds function limit {}",
                    attributes.max_threads_per_block
                )));
            }
            if (name == "row_max" || name == "row_sum")
                && shared_memory_bytes > attributes.max_dynamic_shared_memory_bytes
            {
                return Err(NnisError::invalid_input(format!(
                    "row-softmax {name} requires {shared_memory_bytes} shared-memory bytes; \
                     function limit is {}",
                    attributes.max_dynamic_shared_memory_bytes
                )));
            }
        }
        Ok(Self {
            row_max,
            exp_shift,
            row_sum,
            normalize,
            fused,
            block_size,
        })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Allocate per-row scalar storage reusable up to `max_rows`.
    pub fn workspace(
        &self,
        context: &Arc<Context>,
        max_rows: usize,
    ) -> Result<F32Softmax2DWorkspace> {
        if !Arc::ptr_eq(context, self.row_max.context()) {
            return Err(NnisError::invalid_input(
                "row-softmax and workspace contexts do not match",
            ));
        }
        Ok(F32Softmax2DWorkspace {
            max_rows,
            row_scalars: DeviceBuffer::new(context, max_rows)?,
        })
    }

    /// Compute each row's stable softmax into `output` and wait once.
    pub fn softmax_rows(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        workspace: &F32Softmax2DWorkspace,
    ) -> Result<()> {
        // SAFETY: every buffer borrow remains live until synchronization.
        let enqueue_result =
            unsafe { self.enqueue_softmax_rows(stream, input, output, rows, cols, workspace) };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Compute each row's stable softmax with the single-kernel fused path
    /// and wait once.
    ///
    /// The row must fit in dynamic shared memory together with the
    /// reduction partials: `(cols + block_size) * 4` bytes. No workspace is
    /// required because no intermediate leaves the block.
    pub fn fused_softmax_rows(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        // SAFETY: every buffer borrow remains live until synchronization.
        let enqueue_result = unsafe { self.enqueue_fused_rows(stream, input, output, rows, cols) };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue the complete four-stage row softmax without synchronizing.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, this kernel set, and the workspace must
    /// remain alive and otherwise untouched until the stream completes. The
    /// workspace may not be shared by overlapping operations, including on
    /// other streams.
    pub unsafe fn enqueue_softmax_rows(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        workspace: &F32Softmax2DWorkspace,
    ) -> Result<()> {
        self.validate_execution(stream, input, output, rows, cols, workspace)?;
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let row_scalars = &workspace.row_scalars;

        // SAFETY: stages are ordered on one stream; each reads only
        // prior-stage output under this method's lifetime contract.
        unsafe {
            self.launch_row_reduction(&self.row_max, stream, input, row_scalars, rows, cols)?;
            self.launch_exp_shift(stream, input, output, row_scalars, rows, cols)?;
            self.launch_row_reduction(&self.row_sum, stream, output, row_scalars, rows, cols)?;
            self.launch_normalize(stream, output, row_scalars, rows, cols)?;
        }
        Ok(())
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
            .ok_or_else(|| NnisError::invalid_input("row-softmax shape overflows usize"))?;
        if input.len() != count {
            return Err(NnisError::invalid_input(format!(
                "row-softmax input has {} elements; shape ({rows}, {cols}) requires {count}",
                input.len()
            )));
        }
        if output.len() != count {
            return Err(NnisError::invalid_input(format!(
                "row-softmax output has {} elements; shape ({rows}, {cols}) requires {count}",
                output.len()
            )));
        }
        if !Arc::ptr_eq(self.context(), stream.ctx())
            || !Arc::ptr_eq(self.context(), input.ctx())
            || !Arc::ptr_eq(self.context(), output.ctx())
        {
            return Err(NnisError::invalid_input(
                "row-softmax stream, buffers, and kernels must share one context",
            ));
        }
        Ok(())
    }

    /// Enqueue the single-kernel fused row softmax without synchronizing.
    ///
    /// # Safety
    ///
    /// All buffers and the stream must remain alive and otherwise untouched
    /// until the stream completes.
    pub unsafe fn enqueue_fused_rows(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        self.validate_shapes(stream, input, output, rows, cols)?;
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let attributes = self.fused.attributes()?;
        let shared_memory_bytes = self.fused_shared_memory(cols)?;
        if shared_memory_bytes > attributes.max_dynamic_shared_memory_bytes as usize {
            return Err(NnisError::invalid_input(format!(
                "fused row softmax needs {shared_memory_bytes} shared-memory bytes for \
                 {cols} columns; function limit is {}",
                attributes.max_dynamic_shared_memory_bytes
            )));
        }
        // SAFETY: the caller owns the asynchronous lifetime of both buffers.
        unsafe { self.launch_fused(stream, input, output, rows, cols, shared_memory_bytes) }
    }

    fn fused_shared_memory(&self, cols: usize) -> Result<usize> {
        let element = std::mem::size_of::<f32>();
        let values = element
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("fused row softmax shared size overflows"))?;
        let partials = element
            .checked_mul(self.block_size as usize)
            .ok_or_else(|| NnisError::invalid_input("fused row softmax shared size overflows"))?;
        values
            .checked_add(partials)
            .ok_or_else(|| NnisError::invalid_input("fused row softmax shared size overflows"))
    }

    fn validate_execution(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        workspace: &F32Softmax2DWorkspace,
    ) -> Result<()> {
        self.validate_shapes(stream, input, output, rows, cols)?;
        if rows > workspace.max_rows {
            return Err(NnisError::invalid_input(format!(
                "row-softmax has {rows} rows; workspace capacity is {}",
                workspace.max_rows
            )));
        }
        if !Arc::ptr_eq(self.context(), workspace.context()) {
            return Err(NnisError::invalid_input(
                "row-softmax stream, buffers, and kernels must share one context",
            ));
        }
        Ok(())
    }

    /// # Safety
    ///
    /// Caller owns the asynchronous lifetime of every buffer; see
    /// [`F32Softmax2D::enqueue_softmax_rows`].
    unsafe fn launch_row_reduction(
        &self,
        kernel: &Kernel,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        row_scalars: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let grid_size = u32::try_from(rows)
            .map_err(|_| NnisError::invalid_input("row-softmax exceeds u32::MAX rows"))?;
        let cols = u64::try_from(cols)
            .map_err(|_| NnisError::invalid_input("row-softmax width exceeds u64::MAX"))?;
        let shared_memory_bytes = self
            .block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| NnisError::invalid_input("row-softmax shared memory overflows"))?;
        let config = LaunchConfig::new(Dim3::x(grid_size), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(shared_memory_bytes);
        let mut arguments = KernelArgs::with_capacity(3, 2);
        arguments
            .push_buffer(input)
            .push_buffer(row_scalars)
            .push(cols);
        let launch = KernelLaunch::new(kernel, stream, config);
        // SAFETY: argument order/widths match both row reduction kernels,
        // which share one signature; the caller owns lifetimes.
        unsafe { launch.launch(&mut arguments) }
    }

    /// # Safety
    ///
    /// Caller owns the asynchronous lifetime of every buffer; see
    /// [`F32Softmax2D::enqueue_softmax_rows`].
    unsafe fn launch_exp_shift(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        row_max: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("row-softmax length overflows usize"))?;
        let (rows, cols) = self.u64_shape(rows, cols)?;
        let mut arguments = KernelArgs::with_capacity(5, 3);
        arguments
            .push_buffer(input)
            .push_buffer(output)
            .push_buffer(row_max)
            .push(rows)
            .push(cols);
        let launch = KernelLaunch::new(
            &self.exp_shift,
            stream,
            LaunchConfig::for_num_elements(count, self.block_size)?,
        );
        // SAFETY: argument order/widths match
        // `nnis_softmax_row_exp_shift_f32`; the caller owns lifetimes.
        unsafe { launch.launch(&mut arguments) }
    }

    /// # Safety
    ///
    /// Caller owns the asynchronous lifetime of every buffer; see
    /// [`F32Softmax2D::enqueue_softmax_rows`]. Normalization writes `data`
    /// in place.
    unsafe fn launch_normalize(
        &self,
        stream: &Stream,
        data: &DeviceBuffer<f32>,
        row_total: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("row-softmax length overflows usize"))?;
        let (rows, cols) = self.u64_shape(rows, cols)?;
        let mut arguments = KernelArgs::with_capacity(4, 2);
        arguments
            .push_buffer(data)
            .push_buffer(row_total)
            .push(rows)
            .push(cols);
        let launch = KernelLaunch::new(
            &self.normalize,
            stream,
            LaunchConfig::for_num_elements(count, self.block_size)?,
        );
        // SAFETY: argument order/widths match
        // `nnis_softmax_row_normalize_f32`; same-stream ordering guarantees
        // each row total is final before use.
        unsafe { launch.launch(&mut arguments) }
    }

    fn u64_shape(&self, rows: usize, cols: usize) -> Result<(u64, u64)> {
        let rows = u64::try_from(rows)
            .map_err(|_| NnisError::invalid_input("row-softmax exceeds u64::MAX rows"))?;
        let cols = u64::try_from(cols)
            .map_err(|_| NnisError::invalid_input("row-softmax width exceeds u64::MAX"))?;
        Ok((rows, cols))
    }

    /// # Safety
    ///
    /// Caller owns the asynchronous lifetime of every buffer; see
    /// [`F32Softmax2D::enqueue_softmax_rows`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_fused(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        shared_memory_bytes: usize,
    ) -> Result<()> {
        let grid_size = u32::try_from(rows)
            .map_err(|_| NnisError::invalid_input("row-softmax exceeds u32::MAX rows"))?;
        let cols = u64::try_from(cols)
            .map_err(|_| NnisError::invalid_input("row-softmax width exceeds u64::MAX"))?;
        let config = LaunchConfig::new(Dim3::x(grid_size), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(
                u32::try_from(shared_memory_bytes)
                    .map_err(|_| NnisError::invalid_input("fused shared memory exceeds u32"))?,
            );
        let mut arguments = KernelArgs::with_capacity(3, 2);
        arguments.push_buffer(input).push_buffer(output).push(cols);
        let launch = KernelLaunch::new(&self.fused, stream, config);
        // SAFETY: argument order/widths match `nnis_softmax_row_fused_f32`;
        // the caller owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    fn context(&self) -> &Arc<Context> {
        self.row_max.context()
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

    fn host_values(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|index| {
                let spread = ((index % 23) as f32 - 11.0) * 29.5;
                let ripple = ((index * 11 % 127) as f32 - 63.0) * 0.125;
                spread + ripple
            })
            .collect()
    }

    fn reference_rows(input: &[f32], rows: usize, cols: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(input.len());
        for row in 0..rows {
            let slice = &input[row * cols..(row + 1) * cols];
            let maximum = slice
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, |acc, value| acc.max(f64::from(value)));
            let exponentials: Vec<f64> = slice
                .iter()
                .map(|&value| f64::from(value) - maximum)
                .map(f64::exp)
                .collect();
            let total: f64 = exponentials.iter().sum();
            result.extend(exponentials.into_iter().map(|value| value / total));
        }
        result
    }

    #[test]
    fn row_softmax_matches_high_precision_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let softmax = F32Softmax2D::load(&context, &compiler).unwrap();
        let max_rows = SHAPES.iter().map(|&(rows, _)| rows).max().unwrap();
        let max_elements = SHAPES.iter().map(|&(r, c)| r * c).max().unwrap();
        let workspace = softmax.workspace(&context, max_rows).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(rows, cols) in SHAPES {
            let host = host_values(rows, cols);
            let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
            let output = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
            softmax
                .softmax_rows(&stream, &input, &output, rows, cols, &workspace)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            let expected = reference_rows(&host, rows, cols);
            assert_eq!(actual.len(), expected.len());
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let tolerance = 1.0e-6_f32.max((expected.abs() as f32) * 1.0e-5);
                assert!(
                    (actual - expected as f32).abs() <= tolerance,
                    "row softmax mismatch at {index} shape ({rows}, {cols}): \
                     {actual} != {expected}, tolerance={tolerance}"
                );
            }
        }

        // Every row of a constant matrix is uniform.
        let (rows, cols) = (6_usize, 129_usize);
        let input =
            DeviceBuffer::from_host(&context, &stream, &vec![3.5_f32; rows * cols]).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
        let small_workspace = softmax.workspace(&context, rows).unwrap();
        softmax
            .softmax_rows(&stream, &input, &output, rows, cols, &small_workspace)
            .unwrap();
        for (index, value) in output.to_vec(&stream).unwrap().iter().enumerate() {
            let expected = 1.0 / cols as f32;
            assert!(
                (value - expected).abs() <= 1.0e-6,
                "uniform mismatch at {index}: {value} != {expected}"
            );
        }
        let _ = max_elements;
    }

    #[test]
    fn fused_row_softmax_matches_high_precision_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let softmax = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(rows, cols) in SHAPES {
            let host = host_values(rows, cols);
            let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
            let output = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
            softmax
                .fused_softmax_rows(&stream, &input, &output, rows, cols)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            let expected = reference_rows(&host, rows, cols);
            assert_eq!(actual.len(), expected.len());
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let tolerance = 1.0e-6_f32.max((expected.abs() as f32) * 1.0e-5);
                assert!(
                    (actual - expected as f32).abs() <= tolerance,
                    "fused row softmax mismatch at {index} shape ({rows}, {cols}): \
                     {actual} != {expected}, tolerance={tolerance}"
                );
            }
        }
    }

    #[test]
    fn fused_row_softmax_rejects_oversized_rows() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let softmax = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        // (cols + block_size) * 4 bytes must fit the function's dynamic
        // shared-memory limit; 20,000 columns cannot.
        let cols = 20_000_usize;
        let rows = 2_usize;
        let input = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, rows * cols).unwrap();
        let error = softmax
            .fused_softmax_rows(&stream, &input, &output, rows, cols)
            .unwrap_err();
        assert!(error.to_string().contains("shared-memory"), "{error}");
    }

    #[test]
    fn row_softmax_rejects_invalid_shapes_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(F32Softmax2D::load_with_block_size(&context, &compiler, 0).is_err());
        assert!(F32Softmax2D::load_with_block_size(&context, &compiler, 300).is_err());
        let softmax = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let input = DeviceBuffer::<f32>::new(&context, 16).unwrap();
        let short_output = DeviceBuffer::<f32>::new(&context, 10).unwrap();
        let workspace = softmax.workspace(&context, 3).unwrap();
        let error = softmax
            .softmax_rows(&stream, &input, &short_output, 3, 4, &workspace)
            .unwrap_err();
        assert!(error.to_string().contains("requires 12"), "{error}");

        let output = DeviceBuffer::<f32>::new(&context, 16).unwrap();
        let error = softmax
            .softmax_rows(&stream, &input, &output, 4, 4, &workspace)
            .unwrap_err();
        assert!(
            error.to_string().contains("workspace capacity is 3"),
            "{error}"
        );
    }
}
