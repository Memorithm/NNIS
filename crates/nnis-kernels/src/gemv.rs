//! Matrix-vector product `y = A * x` for row-major `f32` matrices.
//!
//! One thread block computes one output element: threads stride across the
//! row accumulating explicit-FMA products, then a shared-memory tree reduces
//! the per-thread partials. Explicit `fmaf`/`mul_add` on both sides keeps the
//! GPU result bit-for-bit reproducible against the CPU oracle regardless of
//! compiler contraction settings.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const GEMV_SOURCE: &str = r#"
extern "C" __global__ void nnis_gemv_f32(
    const float* matrix,
    const float* vector,
    float* output,
    unsigned long long cols
) {
    extern __shared__ float partial[];
    const unsigned int lane = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const float* row_data = matrix + row * cols;

    float value = 0.0f;
    for (unsigned long long index = lane; index < cols; index += blockDim.x) {
        value = fmaf(row_data[index], vector[index], value);
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
        output[row] = partial[0];
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Context-bound `f32` matrix-vector product.
#[derive(Debug)]
pub struct F32Gemv {
    gemv: Kernel,
    block_size: u32,
}

impl F32Gemv {
    /// Compile (or reuse cached CUBIN) and load the default GEMV kernel.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load the kernel with an explicitly selected power-of-two width.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(NnisError::invalid_input(format!(
                "gemv block size {block_size} is not a non-zero power of two"
            )));
        }
        let shared_memory_bytes = block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| NnisError::invalid_input("gemv shared-memory size overflows"))?;
        let code = compiler.compile_cubin(GEMV_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let gemv = module.get_function("nnis_gemv_f32")?;
        let attributes = gemv.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "gemv block size {block_size} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        if shared_memory_bytes as usize > attributes.max_dynamic_shared_memory_bytes as usize {
            return Err(NnisError::invalid_input(format!(
                "gemv requires {shared_memory_bytes} shared-memory bytes; function limit is {}",
                attributes.max_dynamic_shared_memory_bytes
            )));
        }
        Ok(Self { gemv, block_size })
    }

    /// CUDA thread-block width used along each row.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Compute `output = matrix * vector` and wait for completion.
    ///
    /// Shapes: `matrix` holds `rows * cols` row-major elements, `vector`
    /// holds `cols`, and `output` receives `rows`.
    pub fn gemv(
        &self,
        stream: &Stream,
        matrix: &DeviceBuffer<f32>,
        vector: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result =
            unsafe { self.enqueue_gemv(stream, matrix, vector, output, rows, cols) };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue the GEMV without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, and this kernel must remain alive and
    /// otherwise untouched until the stream completes.
    pub unsafe fn enqueue_gemv(
        &self,
        stream: &Stream,
        matrix: &DeviceBuffer<f32>,
        vector: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        self.validate_execution(stream, matrix, vector, output, rows, cols)?;
        if rows == 0 || cols == 0 {
            // SAFETY: the output lifetime obligation is documented above.
            return unsafe { output.zero_async(stream) };
        }
        let shared_memory_bytes = self
            .block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| NnisError::invalid_input("gemv shared-memory size overflows"))?;
        let grid_size = u32::try_from(rows)
            .map_err(|_| NnisError::invalid_input("gemv exceeds u32::MAX rows"))?;
        let cols = u64::try_from(cols)
            .map_err(|_| NnisError::invalid_input("gemv width exceeds u64::MAX"))?;
        let config = LaunchConfig::new(Dim3::x(grid_size), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(shared_memory_bytes);
        let mut arguments = KernelArgs::with_capacity(4, 3);
        arguments
            .push_buffer(matrix)
            .push_buffer(vector)
            .push_buffer(output)
            .push(cols);
        let launch = KernelLaunch::new(&self.gemv, stream, config);
        // SAFETY: argument order/widths match `nnis_gemv_f32`; the caller
        // owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    fn validate_execution(
        &self,
        stream: &Stream,
        matrix: &DeviceBuffer<f32>,
        vector: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let expected_matrix = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("gemv shape overflows usize"))?;
        if matrix.len() != expected_matrix {
            return Err(NnisError::invalid_input(format!(
                "gemv matrix has {} elements; shape ({rows}, {cols}) requires {expected_matrix}",
                matrix.len()
            )));
        }
        if vector.len() != cols {
            return Err(NnisError::invalid_input(format!(
                "gemv vector has {} elements; shape requires {cols}",
                vector.len()
            )));
        }
        if output.len() != rows {
            return Err(NnisError::invalid_input(format!(
                "gemv output has {} elements; shape requires {rows}",
                output.len()
            )));
        }
        let context = self.gemv.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, matrix.ctx())
            || !Arc::ptr_eq(context, vector.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "gemv stream, buffers, and kernel must share one context",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    const SHAPES: &[(usize, usize)] = &[
        (1, 1),
        (1, 7),
        (3, 31),
        (4, 32),
        (5, 255),
        (6, 256),
        (7, 257),
        (13, 1_023),
        (9, 1_024),
        (11, 1_025),
    ];

    fn host_matrix(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|index| (((index * 13 % 97) as f32 - 48.0) * 0.0625) + ((index % 5) as f32 - 2.0))
            .collect()
    }

    fn host_vector(cols: usize) -> Vec<f32> {
        (0..cols)
            .map(|index| ((index * 29 % 61) as f32 - 30.0) * 0.125)
            .collect()
    }

    /// Replays the kernel's evaluation order exactly: per-thread strided
    /// explicit-FMA accumulation followed by a shared-memory tree reduction.
    fn reference_gemv(
        matrix: &[f32],
        vector: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) -> Vec<f32> {
        matrix
            .chunks(cols)
            .take(rows)
            .map(|row_data| {
                let mut partials: Vec<f32> = (0..block_size)
                    .map(|lane| {
                        row_data
                            .iter()
                            .skip(lane)
                            .step_by(block_size)
                            .zip(vector.iter().skip(lane).step_by(block_size))
                            .fold(0.0_f32, |value, (&m, &v)| m.mul_add(v, value))
                    })
                    .collect();
                let mut stride = block_size / 2;
                while stride > 0 {
                    for lane in 0..stride {
                        partials[lane] += partials[lane + stride];
                    }
                    stride /= 2;
                }
                partials[0]
            })
            .collect()
    }

    #[test]
    fn gemv_matches_ordered_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let gemv = F32Gemv::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let max_rows = SHAPES.iter().map(|&(rows, _)| rows).max().unwrap();
        let max_elements = SHAPES.iter().map(|&(r, c)| r * c).max().unwrap();

        for &(rows, cols) in SHAPES {
            let matrix_host = host_matrix(rows, cols);
            let vector_host = host_vector(cols);
            // Pad allocations to the largest shape so buffers are reused
            // across iterations while the logical shape varies.
            let matrix = DeviceBuffer::from_host(&context, &stream, &matrix_host).unwrap();
            let _ = max_rows;
            let _ = max_elements;
            let vector = DeviceBuffer::from_host(&context, &stream, &vector_host).unwrap();
            let output = DeviceBuffer::<f32>::new(&context, rows).unwrap();

            gemv.gemv(&stream, &matrix, &vector, &output, rows, cols)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            let expected = reference_gemv(
                &matrix_host,
                &vector_host,
                rows,
                cols,
                gemv.block_size() as usize,
            );
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "ordered gemv mismatch at row {index} shape ({rows}, {cols}): \
                     {actual} != {expected}"
                );
            }

            // Independent f64 check inside an explicit tolerance.
            for index in 0..rows {
                let high_precision: f64 = (0..cols)
                    .map(|col| {
                        f64::from(matrix_host[index * cols + col]) * f64::from(vector_host[col])
                    })
                    .sum();
                assert!(
                    (f64::from(actual[index]) - high_precision).abs()
                        <= 1.0e-3_f64.max(high_precision.abs() * 1.0e-5),
                    "gemv f64 mismatch at row {index}: {actual:?} vs {high_precision}"
                );
            }
        }
    }

    #[test]
    fn gemv_rejects_invalid_shapes_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(F32Gemv::load_with_block_size(&context, &compiler, 0).is_err());
        assert!(F32Gemv::load_with_block_size(&context, &compiler, 192).is_err());
        let gemv = F32Gemv::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let matrix = DeviceBuffer::<f32>::new(&context, 12).unwrap(); // 3 x 4
        let short_vector = DeviceBuffer::<f32>::new(&context, 3).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 3).unwrap();
        let error = gemv
            .gemv(&stream, &matrix, &short_vector, &output, 3, 4)
            .unwrap_err();
        assert!(error.to_string().contains("requires 4"), "{error}");

        let vector = DeviceBuffer::<f32>::new(&context, 4).unwrap();
        let long_output = DeviceBuffer::<f32>::new(&context, 4).unwrap();
        let error = gemv
            .gemv(&stream, &matrix, &vector, &long_output, 3, 4)
            .unwrap_err();
        assert!(error.to_string().contains("shape requires 3"), "{error}");
    }
}
