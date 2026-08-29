//! Row gathering for embedding lookups over row-major tables.
//!
//! `output[i * cols + j] = table[indices[i] * cols + j]` - a pure copy of
//! stored bit patterns, so results are bit-for-bit identical to the table
//! contents for both `f32` and packed-bf16 storage. The safe wrappers
//! validate every index against the table row count on the host before any
//! launch; the unsafe enqueues skip that check for asynchronous callers
//! that have already validated.

use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const GATHER_SOURCE: &str = r#"
extern "C" __global__ void nnis_gather_rows_f32(
    const float* table,
    const unsigned int* indices,
    float* output,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long count
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long elements = count * cols;
    if (index >= elements) {
        return;
    }
    const unsigned int row = indices[index / cols];
    output[index] = table[(unsigned long long)row * cols + index % cols];
}

extern "C" __global__ void nnis_gather_rows_bf16(
    const unsigned short* table,
    const unsigned int* indices,
    unsigned short* output,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long count
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long elements = count * cols;
    if (index >= elements) {
        return;
    }
    const unsigned int row = indices[index / cols];
    output[index] = table[(unsigned long long)row * cols + index % cols];
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Context-bound row gather over an `f32` embedding table.
#[derive(Debug)]
pub struct F32Gather {
    kernel: Kernel,
    block_size: u32,
}

impl F32Gather {
    /// Compile (or reuse cached CUBIN) and load the default gather kernel.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load with an explicit power-of-two thread-block width.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(NnisError::invalid_input(format!(
                "gather block size {block_size} is not a non-zero power of two"
            )));
        }
        let code = compiler.compile_cubin(GATHER_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let kernel = module.get_function("nnis_gather_rows_f32")?;
        let attributes = kernel.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "gather block size {block_size} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        Ok(Self { kernel, block_size })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Copy the selected rows into `output` and wait for completion.
    ///
    /// `table` holds `rows * cols` row-major elements, `indices` holds
    /// `count` row numbers (each validated against `rows`), and `output`
    /// receives `count * cols`. Every index must be strictly smaller than
    /// `rows`; duplicates and unsorted orders are allowed.
    ///
    /// An empty `count` leaves nothing to read or write.
    pub fn gather(
        &self,
        stream: &Stream,
        table: &DeviceBuffer<f32>,
        indices: &DeviceBuffer<u32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        // Host-side bound check keeps the safe API total: no launch happens
        // with an index the kernel would read out of the table.
        let index_host = indices.to_vec(stream)?;
        if let Some((position, index)) = index_host
            .iter()
            .enumerate()
            .find(|(_, &i)| i as usize >= rows)
        {
            return Err(NnisError::invalid_input(format!(
                "gather index {index} at position {position} is out of range for {rows} rows"
            )));
        }
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result =
            unsafe { self.enqueue_gather(stream, table, indices, output, rows, cols) };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue the gather without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, and this kernel must remain alive and
    /// otherwise untouched until the stream completes. Every index value
    /// must be smaller than `rows`; out-of-range indices read out of bounds.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_gather(
        &self,
        stream: &Stream,
        table: &DeviceBuffer<f32>,
        indices: &DeviceBuffer<u32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        self.validate_common(stream, table, indices, output.len(), rows, cols, "gather")?;
        if indices.is_empty() {
            return Ok(());
        }
        self.launch(
            &self.kernel,
            table.device_ptr(),
            indices,
            output.device_ptr(),
            rows,
            cols,
            indices.len(),
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        kernel: &Kernel,
        table: u64,
        indices: &DeviceBuffer<u32>,
        output: u64,
        rows: usize,
        cols: usize,
        count: usize,
        stream: &Stream,
    ) -> Result<()> {
        let mut arguments = KernelArgs::with_capacity(6, 3);
        arguments
            .push(table)
            .push_buffer(indices)
            .push(output)
            .push(rows as u64)
            .push(cols as u64)
            .push(count as u64);
        let launch = KernelLaunch::new(
            kernel,
            stream,
            LaunchConfig::for_num_elements(count.saturating_mul(cols), self.block_size)?,
        );
        // SAFETY: argument order/widths match the gather signatures; the
        // caller owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_common(
        &self,
        stream: &Stream,
        table: &DeviceBuffer<f32>,
        indices: &DeviceBuffer<u32>,
        output_len: usize,
        rows: usize,
        cols: usize,
        label: &str,
    ) -> Result<()> {
        let expected_table = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input(format!("{label} shape overflows usize")))?;
        if table.len() != expected_table {
            return Err(NnisError::invalid_input(format!(
                "{label} table has {} elements; shape ({rows}, {cols}) requires {expected_table}",
                table.len()
            )));
        }
        let expected_output = indices
            .len()
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("gather shape overflows usize"))?;
        if output_len != expected_output {
            return Err(NnisError::invalid_input(format!(
                "{label} output has {output_len} elements; {} indices x {cols} columns \
                 requires {expected_output}",
                indices.len()
            )));
        }
        let context = self.kernel.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, table.ctx())
            || !Arc::ptr_eq(context, indices.ctx())
        {
            return Err(NnisError::invalid_input(
                "gather stream, buffers, and kernel must share one context",
            ));
        }
        Ok(())
    }
}

/// Context-bound row gather over a packed-bf16 `u16` embedding table.
#[derive(Debug)]
pub struct Bf16Gather {
    kernel: Kernel,
    block_size: u32,
}

impl Bf16Gather {
    /// Compile (or reuse cached CUBIN) and load the default gather kernel.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load with an explicit power-of-two thread-block width.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(NnisError::invalid_input(format!(
                "bf16 gather block size {block_size} is not a non-zero power of two"
            )));
        }
        let code = compiler.compile_cubin(GATHER_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let kernel = module.get_function("nnis_gather_rows_bf16")?;
        let attributes = kernel.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "bf16 gather block size {block_size} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        Ok(Self { kernel, block_size })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Copy the selected rows into `output` and wait for completion; see
    /// [`F32Gather::gather`] for the shared contract. Bit patterns are
    /// copied verbatim.
    pub fn gather(
        &self,
        stream: &Stream,
        table: &DeviceBuffer<u16>,
        indices: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u16>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let index_host = indices.to_vec(stream)?;
        if let Some((position, index)) = index_host
            .iter()
            .enumerate()
            .find(|(_, &i)| i as usize >= rows)
        {
            return Err(NnisError::invalid_input(format!(
                "bf16 gather index {index} at position {position} is out of range for {rows} rows"
            )));
        }
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result =
            unsafe { self.enqueue_gather(stream, table, indices, output, rows, cols) };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue the gather without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, and this kernel must remain alive and
    /// otherwise untouched until the stream completes. Every index value
    /// must be smaller than `rows`; out-of-range indices read out of bounds.
    pub unsafe fn enqueue_gather(
        &self,
        stream: &Stream,
        table: &DeviceBuffer<u16>,
        indices: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u16>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let context = self.kernel.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, table.ctx())
            || !Arc::ptr_eq(context, indices.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "bf16 gather stream, buffers, and kernel must share one context",
            ));
        }
        let expected_table = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("bf16 gather shape overflows usize"))?;
        if table.len() != expected_table {
            return Err(NnisError::invalid_input(format!(
                "bf16 gather table has {} elements; shape ({rows}, {cols}) requires \
                 {expected_table}",
                table.len()
            )));
        }
        let expected_output = indices.len().saturating_mul(cols);
        if output.len() != expected_output {
            return Err(NnisError::invalid_input(format!(
                "bf16 gather output has {} elements; {} indices x {cols} columns \
                 requires {expected_output}",
                output.len(),
                indices.len()
            )));
        }
        if indices.is_empty() {
            return Ok(());
        }
        self.launch(
            &self.kernel,
            table.device_ptr(),
            indices,
            output.device_ptr(),
            rows,
            cols,
            indices.len(),
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        kernel: &Kernel,
        table: u64,
        indices: &DeviceBuffer<u32>,
        output: u64,
        rows: usize,
        cols: usize,
        count: usize,
        stream: &Stream,
    ) -> Result<()> {
        let mut arguments = KernelArgs::with_capacity(6, 3);
        arguments
            .push(table)
            .push_buffer(indices)
            .push(output)
            .push(rows as u64)
            .push(cols as u64)
            .push(count as u64);
        let launch = KernelLaunch::new(
            kernel,
            stream,
            LaunchConfig::for_num_elements(count.saturating_mul(cols), self.block_size)?,
        );
        // SAFETY: argument order/widths match the gather signatures; the
        // caller owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    const CASES: &[(usize, usize, usize)] = &[
        // (rows, cols, count)
        (1, 1, 1),
        (4, 8, 4),
        (16, 1, 7),
        (33, 65, 129),
        (256, 32, 300),
    ];

    fn host_table(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|index| ((index * 13 % 97) as f32 - 48.0) * 0.0625)
            .collect()
    }

    fn host_indices(rows: usize, count: usize) -> Vec<u32> {
        // Deterministic duplicates and unsorted orders.
        (0..count)
            .map(|index| ((index * 17 + index / 3) % rows) as u32)
            .collect()
    }

    #[test]
    fn gather_rows_match_table_contents_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let f32_gather = F32Gather::load(&context, &compiler).unwrap();
        let bf16_gather = Bf16Gather::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let narrow = nnis_rt::f32_to_bf16_rne;

        for &(rows, cols, count) in CASES {
            let table_host = host_table(rows, cols);
            let indices_host = host_indices(rows, count);
            let indices = DeviceBuffer::from_host(&context, &stream, &indices_host).unwrap();

            // f32: bit-exact copy of stored values.
            let table_f32 = DeviceBuffer::from_host(&context, &stream, &table_host).unwrap();
            let poisoned_f32 = vec![f32::NAN; count * cols];
            let output_f32 = DeviceBuffer::from_host(&context, &stream, &poisoned_f32).unwrap();
            f32_gather
                .gather(&stream, &table_f32, &indices, &output_f32, rows, cols)
                .unwrap();
            let actual_f32 = output_f32.to_vec(&stream).unwrap();
            for position in 0..count {
                let row = indices_host[position] as usize;
                for col in 0..cols {
                    assert_eq!(
                        actual_f32[position * cols + col].to_bits(),
                        table_host[row * cols + col].to_bits(),
                        "f32 mismatch at ({position}, {col}) shape ({rows}, {cols}, {count})"
                    );
                }
            }

            // Packed bf16: bit patterns copied verbatim.
            let bits: Vec<u16> = table_host.iter().copied().map(narrow).collect();
            let table_bf16 = DeviceBuffer::from_host(&context, &stream, &bits).unwrap();
            let poisoned_bf16 = vec![u16::MAX; count * cols];
            let output_bf16 = DeviceBuffer::from_host(&context, &stream, &poisoned_bf16).unwrap();
            bf16_gather
                .gather(&stream, &table_bf16, &indices, &output_bf16, rows, cols)
                .unwrap();
            let actual_bf16 = output_bf16.to_vec(&stream).unwrap();
            for position in 0..count {
                let row = indices_host[position] as usize;
                for col in 0..cols {
                    assert_eq!(
                        actual_bf16[position * cols + col],
                        bits[row * cols + col],
                        "bf16 mismatch at ({position}, {col}) shape ({rows}, {cols}, {count})"
                    );
                }
            }
        }
    }

    #[test]
    fn gather_rejects_out_of_range_indices_before_launch_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let f32_gather = F32Gather::load(&context, &compiler).unwrap();
        let bf16_gather = Bf16Gather::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        let table = DeviceBuffer::<f32>::new(&context, 4 * 8).unwrap(); // 4 rows x 8
        let table_bf16 = DeviceBuffer::<u16>::new(&context, 4 * 8).unwrap();
        let indices = DeviceBuffer::from_host(&context, &stream, &[0_u32, 2, 9]).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 3 * 8).unwrap();
        let output_bf16 = DeviceBuffer::<u16>::new(&context, 3 * 8).unwrap();

        let error = f32_gather
            .gather(&stream, &table, &indices, &output, 4, 8)
            .unwrap_err();
        assert!(error.to_string().contains("index 9"), "{error}");

        let error = bf16_gather
            .gather(&stream, &table_bf16, &indices, &output_bf16, 4, 8)
            .unwrap_err();
        assert!(error.to_string().contains("index 9"), "{error}");

        // Wrong table size is rejected before any launch.
        let short_table = DeviceBuffer::<f32>::new(&context, 31).unwrap();
        let valid = DeviceBuffer::from_host(&context, &stream, &[1_u32, 3]).unwrap();
        let error = f32_gather
            .gather(&stream, &short_table, &valid, &output, 4, 8)
            .unwrap_err();
        assert!(error.to_string().contains("requires 32"), "{error}");
    }
}
