//! Batch-one projection with F32 activations and sparse CSC F32 weights.
//!
//! Each output column owns one contiguous slice of retained weights described by
//! `column_offsets[col]..column_offsets[col + 1]`. Retained row indices and
//! F32 values are traversed in stored order, so the accumulation order is
//! deterministic for one canonical CSC payload.
//!
//! This is an isolated structural execution primitive. It does not by itself
//! qualify a full sparse model representation or claim a speedup.

use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const SOURCE: &str = r#"
extern "C" __global__ void nnis_project_kn_f32_sparse_csc(
    const float* input,
    const unsigned long long* column_offsets,
    const unsigned int* row_indices,
    const float* values,
    float* output,
    unsigned long long k,
    unsigned long long n,
    unsigned long long nnz
) {
    const unsigned long long col =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    const unsigned long long begin = column_offsets[col];
    const unsigned long long end = column_offsets[col + 1ull];
    if (begin > end || end > nnz) {
        output[col] = __int_as_float(0x7fffffff);
        return;
    }

    float value = 0.0f;
    for (unsigned long long index = begin; index < end; ++index) {
        const unsigned int row = row_indices[index];
        if ((unsigned long long)row >= k) {
            output[col] = __int_as_float(0x7fffffff);
            return;
        }
        value = fmaf(input[row], values[index], value);
    }
    output[col] = value;
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 64;

/// Context-bound sparse CSC `[1,K] x [K,N] -> [1,N]` projection.
#[derive(Debug)]
pub struct F32SparseCscGemv {
    project_kn: Kernel,
    block_size: u32,
}

impl F32SparseCscGemv {
    /// Compile and load the default sparse CSC projection primitive.
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
                "f32-sparse-csc project block size {block_size} is not a non-zero power of two"
            )));
        }
        let code = compiler.compile_cubin(SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let project_kn = module.get_function("nnis_project_kn_f32_sparse_csc")?;
        let attributes = project_kn.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "f32-sparse-csc project block size {block_size} exceeds function limit {}",
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

    /// Execute one sparse CSC projection and synchronize.
    #[allow(clippy::too_many_arguments)]
    pub fn project_kn(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        column_offsets: &DeviceBuffer<u64>,
        row_indices: &DeviceBuffer<u32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe {
            self.enqueue_project_kn(
                stream,
                input,
                column_offsets,
                row_indices,
                values,
                output,
                k,
                n,
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

    /// Enqueue one sparse CSC projection without synchronizing.
    ///
    /// # Safety
    ///
    /// The stream, kernel and all buffers must remain alive and otherwise
    /// untouched until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_project_kn(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        column_offsets: &DeviceBuffer<u64>,
        row_indices: &DeviceBuffer<u32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        self.validate_execution(
            stream,
            input,
            column_offsets,
            row_indices,
            values,
            output,
            k,
            n,
        )?;
        if n == 0 {
            return Ok(());
        }
        if values.is_empty() {
            // SAFETY: the output lifetime obligation is documented above.
            return unsafe { output.zero_async(stream) };
        }

        let k_arg = u64::try_from(k)
            .map_err(|_| NnisError::invalid_input("f32-sparse-csc project K exceeds u64::MAX"))?;
        let n_arg = u64::try_from(n)
            .map_err(|_| NnisError::invalid_input("f32-sparse-csc project N exceeds u64::MAX"))?;
        let nnz_arg = u64::try_from(values.len())
            .map_err(|_| NnisError::invalid_input("f32-sparse-csc NNZ exceeds u64::MAX"))?;
        let config = LaunchConfig::for_num_elements(n, self.block_size)?;
        let mut arguments = KernelArgs::with_capacity(8, 6);
        arguments
            .push_buffer(input)
            .push_buffer(column_offsets)
            .push_buffer(row_indices)
            .push_buffer(values)
            .push_buffer(output)
            .push(k_arg)
            .push(n_arg)
            .push(nnz_arg);
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
        column_offsets: &DeviceBuffer<u64>,
        row_indices: &DeviceBuffer<u32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        if input.len() != k {
            return Err(NnisError::invalid_input(format!(
                "f32-sparse-csc project input has {} elements; shape requires {k}",
                input.len()
            )));
        }
        let expected_offsets = n.checked_add(1).ok_or_else(|| {
            NnisError::invalid_input("f32-sparse-csc offset count overflows usize")
        })?;
        if column_offsets.len() != expected_offsets {
            return Err(NnisError::invalid_input(format!(
                "f32-sparse-csc column offsets has {} entries; shape requires {expected_offsets}",
                column_offsets.len()
            )));
        }
        if row_indices.len() != values.len() {
            return Err(NnisError::invalid_input(format!(
                "f32-sparse-csc row indices has {} entries; values has {}",
                row_indices.len(),
                values.len()
            )));
        }
        if k == 0 && !values.is_empty() {
            return Err(NnisError::invalid_input(
                "f32-sparse-csc cannot retain values when K is zero",
            ));
        }
        if output.len() != n {
            return Err(NnisError::invalid_input(format!(
                "f32-sparse-csc project output has {} elements; shape requires {n}",
                output.len()
            )));
        }

        let context = self.project_kn.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, input.ctx())
            || !Arc::ptr_eq(context, column_offsets.ctx())
            || !Arc::ptr_eq(context, row_indices.ctx())
            || !Arc::ptr_eq(context, values.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "f32-sparse-csc project stream, buffers and kernel must share one context",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    fn ordered_csc_projection(
        input: &[f32],
        column_offsets: &[u64],
        row_indices: &[u32],
        values: &[f32],
        n: usize,
    ) -> Vec<f32> {
        (0..n)
            .map(|col| {
                let begin = column_offsets[col] as usize;
                let end = column_offsets[col + 1] as usize;
                (begin..end).fold(0.0_f32, |acc, index| {
                    input[row_indices[index] as usize].mul_add(values[index], acc)
                })
            })
            .collect()
    }

    #[test]
    fn sparse_csc_projection_matches_ordered_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let k = 5_usize;
        let n = 4_usize;
        let input_host = [0.5_f32, -1.0, 2.0, 0.25, -0.75];
        let offsets_host = [0_u64, 2, 3, 5, 7];
        let rows_host = [0_u32, 3, 1, 0, 4, 2, 3];
        let values_host = [2.0_f32, -1.0, 0.5, -0.25, 4.0, 1.5, -2.0];
        let expected =
            ordered_csc_projection(&input_host, &offsets_host, &rows_host, &values_host, n);

        let compiler = JitCompiler::new();
        let kernel = F32SparseCscGemv::load_with_block_size(&context, &compiler, 64).unwrap();
        let stream = Stream::new(&context).unwrap();
        let input = DeviceBuffer::from_host(&context, &stream, &input_host).unwrap();
        let offsets = DeviceBuffer::from_host(&context, &stream, &offsets_host).unwrap();
        let rows = DeviceBuffer::from_host(&context, &stream, &rows_host).unwrap();
        let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, n).unwrap();

        kernel
            .project_kn(&stream, &input, &offsets, &rows, &values, &output, k, n)
            .unwrap();
        let actual = output.to_vec(&stream).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "sparse CSC ordered projection mismatch at output {index}"
            );
        }
    }

    #[test]
    fn sparse_csc_projection_rejects_shape_contract_violations_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(F32SparseCscGemv::load_with_block_size(&context, &compiler, 0).is_err());
        assert!(F32SparseCscGemv::load_with_block_size(&context, &compiler, 48).is_err());

        let kernel = F32SparseCscGemv::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let input = DeviceBuffer::<f32>::new(&context, 4).unwrap();
        let offsets = DeviceBuffer::<u64>::new(&context, 4).unwrap();
        let rows = DeviceBuffer::<u32>::new(&context, 2).unwrap();
        let values = DeviceBuffer::<f32>::new(&context, 3).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 3).unwrap();

        let error = kernel
            .project_kn(&stream, &input, &offsets, &rows, &values, &output, 4, 3)
            .unwrap_err();
        assert!(error.to_string().contains("row indices"), "{error}");

        let rows = DeviceBuffer::<u32>::new(&context, 3).unwrap();
        let short_offsets = DeviceBuffer::<u64>::new(&context, 3).unwrap();
        let error = kernel
            .project_kn(
                &stream,
                &input,
                &short_offsets,
                &rows,
                &values,
                &output,
                4,
                3,
            )
            .unwrap_err();
        assert!(error.to_string().contains("requires 4"), "{error}");
    }
}
