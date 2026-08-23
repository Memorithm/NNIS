//! Row scattering: write source rows to destination positions.
//!
//! `output[positions[i] * cols + j] = rows[i * cols + j]` - the inverse of
//! [`crate::gather`], suitable for KV-cache appends and positional writes.
//! Bit patterns are copied verbatim for both `f32` and packed-bf16
//! storage. Safe wrappers validate every position on the host before any
//! launch; positions must be unique or writes race (documented), so the
//! checked path rejects duplicates instead of leaving undefined content.

use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const SCATTER_SOURCE: &str = r#"
extern "C" __global__ void nnis_scatter_rows_f32(
    const float* source,
    const unsigned int* positions,
    float* output,
    unsigned long long output_rows,
    unsigned long long cols,
    unsigned long long count
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count * cols) {
        return;
    }
    const unsigned int row = positions[index / cols];
    output[(unsigned long long)row * cols + index % cols] = source[index];
}

extern "C" __global__ void nnis_scatter_rows_bf16(
    const unsigned short* source,
    const unsigned int* positions,
    unsigned short* output,
    unsigned long long output_rows,
    unsigned long long cols,
    unsigned long long count
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count * cols) {
        return;
    }
    const unsigned int row = positions[index / cols];
    output[(unsigned long long)row * cols + index % cols] = source[index];
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

macro_rules! scatter_family {
    ($name:ident, $kernel_name:literal, $dtype:ty, $label:literal) => {
        /// Context-bound row scatter; see the module documentation for the
        /// contract shared by both storage widths.
        #[derive(Debug)]
        pub struct $name {
            kernel: Kernel,
            block_size: u32,
        }

        impl $name {
            /// Compile (or reuse cached CUBIN) and load the default kernel.
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
                        concat!($label, " block size {} is not a non-zero power of two"),
                        block_size
                    )));
                }
                let code =
                    compiler.compile_cubin(SCATTER_SOURCE, &CompileOptions::for_device(context))?;
                let module = Module::load(context, &code)?;
                let kernel = module.get_function($kernel_name)?;
                let attributes = kernel.attributes()?;
                if block_size > attributes.max_threads_per_block {
                    return Err(NnisError::invalid_input(format!(
                        concat!($label, " block size {} exceeds function limit {}"),
                        block_size, attributes.max_threads_per_block
                    )));
                }
                Ok(Self { kernel, block_size })
            }

            pub fn block_size(&self) -> u32 {
                self.block_size
            }

            /// Write `count` source rows into `output` at `positions` and
            /// wait for completion.
            ///
            /// `source` holds `count * cols` row-major elements, `positions`
            /// holds `count` row numbers each validated on the host against
            /// `output_rows`, and duplicates are rejected because overlapping
            /// writes would be racy. Other rows of `output` are untouched.
            pub fn scatter(
                &self,
                stream: &Stream,
                source: &DeviceBuffer<$dtype>,
                positions: &DeviceBuffer<u32>,
                output: &DeviceBuffer<$dtype>,
                output_rows: usize,
                cols: usize,
            ) -> Result<()> {
                let position_host = positions.to_vec(stream)?;
                let mut seen = std::collections::HashSet::with_capacity(position_host.len());
                for (position_at, &row) in position_host.iter().enumerate() {
                    if row as usize >= output_rows {
                        return Err(NnisError::invalid_input(format!(
                            concat!(
                                $label,
                                " position {} at slot {} is out of range for {} rows"
                            ),
                            row, position_at, output_rows
                        )));
                    }
                    if !seen.insert(row) {
                        return Err(NnisError::invalid_input(format!(
                            concat!($label, " duplicate position {} would race"),
                            row
                        )));
                    }
                }
                // SAFETY: all borrows remain live until synchronization below.
                let enqueue_result = unsafe {
                    self.enqueue_scatter(stream, source, positions, output, output_rows, cols)
                };
                match enqueue_result {
                    Ok(()) => stream.synchronize(),
                    Err(error) => {
                        let _ = stream.synchronize();
                        Err(error)
                    }
                }
            }

            /// Enqueue without synchronizing the stream.
            ///
            /// # Safety
            ///
            /// All buffers, the stream, and this kernel must remain alive and
            /// otherwise untouched until the stream completes. Every position
            /// must be smaller than `output_rows`; duplicate positions make
            /// writes racy; out-of-range positions write out of bounds.
            pub unsafe fn enqueue_scatter(
                &self,
                stream: &Stream,
                source: &DeviceBuffer<$dtype>,
                positions: &DeviceBuffer<u32>,
                output: &DeviceBuffer<$dtype>,
                output_rows: usize,
                cols: usize,
            ) -> Result<()> {
                let context = self.kernel.context();
                if !Arc::ptr_eq(context, stream.ctx())
                    || !Arc::ptr_eq(context, source.ctx())
                    || !Arc::ptr_eq(context, positions.ctx())
                    || !Arc::ptr_eq(context, output.ctx())
                {
                    return Err(NnisError::invalid_input(concat!(
                        $label,
                        " stream, buffers, and kernel must share one context"
                    )));
                }
                let expected_source = positions.len().checked_mul(cols).ok_or_else(|| {
                    NnisError::invalid_input(concat!($label, " shape overflows usize"))
                })?;
                if source.len() != expected_source {
                    return Err(NnisError::invalid_input(format!(
                        concat!(
                            $label,
                            " source has {} elements; {} positions x {} columns \
                                 requires {}"
                        ),
                        source.len(),
                        positions.len(),
                        cols,
                        expected_source
                    )));
                }
                let expected_output = output_rows.checked_mul(cols).ok_or_else(|| {
                    NnisError::invalid_input(concat!($label, " shape overflows usize"))
                })?;
                if output.len() != expected_output {
                    return Err(NnisError::invalid_input(format!(
                        concat!(
                            $label,
                            " output has {} elements; shape ({}, {}) requires {}"
                        ),
                        output.len(),
                        output_rows,
                        cols,
                        expected_output
                    )));
                }
                if positions.is_empty() {
                    return Ok(());
                }
                let mut arguments = KernelArgs::with_capacity(6, 3);
                arguments
                    .push_buffer(source)
                    .push_buffer(positions)
                    .push_buffer(output)
                    .push(output_rows as u64)
                    .push(cols as u64)
                    .push(positions.len() as u64);
                let launch = KernelLaunch::new(
                    &self.kernel,
                    stream,
                    LaunchConfig::for_num_elements(
                        positions.len().saturating_mul(cols),
                        self.block_size,
                    )?,
                );
                // SAFETY: argument order/widths match the scatter signatures;
                // the caller owns the asynchronous lifetime obligation.
                unsafe { launch.launch(&mut arguments) }
            }
        }
    };
}

scatter_family!(F32Scatter, "nnis_scatter_rows_f32", f32, "scatter");
scatter_family!(Bf16Scatter, "nnis_scatter_rows_bf16", u16, "bf16 scatter");

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    #[test]
    fn scatter_writes_selected_rows_and_leaves_others_untouched_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let f32_scatter = F32Scatter::load(&context, &compiler).unwrap();
        let bf16_scatter = Bf16Scatter::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        const CASES: &[(usize, usize, usize)] = &[(4, 8, 4), (16, 1, 7), (33, 65, 16)];
        for &(rows, cols, count) in CASES {
            assert!(count <= rows);
            let source: Vec<f32> = (0..count * cols)
                .map(|i| ((i * 13 % 97) as f32 - 48.0) * 0.0625)
                .collect();
            // Descending order: unique by construction and exercises
            // non-monotonic write patterns.
            let positions: Vec<u32> = (0..count).map(|i| (rows - 1 - i) as u32).collect();

            // Pre-fill outputs with a sentinel distinct from any payload.
            let output_f32 =
                DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; rows * cols]).unwrap();
            let source_buf = DeviceBuffer::from_host(&context, &stream, &source).unwrap();
            let positions_buf = DeviceBuffer::from_host(&context, &stream, &positions).unwrap();
            f32_scatter
                .scatter(
                    &stream,
                    &source_buf,
                    &positions_buf,
                    &output_f32,
                    rows,
                    cols,
                )
                .unwrap();
            let actual = output_f32.to_vec(&stream).unwrap();
            for row in 0..rows {
                match positions.iter().position(|&p| p as usize == row) {
                    Some(slot) => {
                        for col in 0..cols {
                            assert_eq!(
                                actual[row * cols + col].to_bits(),
                                source[slot * cols + col].to_bits(),
                                "f32 scattered mismatch ({row},{col}) case {rows}x{cols}x{count}"
                            );
                        }
                    }
                    None => {
                        for col in 0..cols {
                            assert!(
                                actual[row * cols + col].is_nan(),
                                "untouched row {row} was modified"
                            );
                        }
                    }
                }
            }

            // bf16: same flow over bit patterns with a distinct sentinel.
            let narrow = nnis_rt::f32_to_bf16_rne;
            let source_bits: Vec<u16> = source.iter().copied().map(narrow).collect();
            let output_bf16 =
                DeviceBuffer::from_host(&context, &stream, &vec![0xAAAA_u16; rows * cols]).unwrap();
            let source_bits_buf = DeviceBuffer::from_host(&context, &stream, &source_bits).unwrap();
            bf16_scatter
                .scatter(
                    &stream,
                    &source_bits_buf,
                    &positions_buf,
                    &output_bf16,
                    rows,
                    cols,
                )
                .unwrap();
            let actual_bits = output_bf16.to_vec(&stream).unwrap();
            for row in 0..rows {
                match positions.iter().position(|&p| p as usize == row) {
                    Some(slot) => {
                        for col in 0..cols {
                            assert_eq!(
                                actual_bits[row * cols + col],
                                source_bits[slot * cols + col],
                                "bf16 scattered mismatch ({row},{col})"
                            );
                        }
                    }
                    None => {
                        assert!(
                            actual_bits[row * cols..(row + 1) * cols]
                                .iter()
                                .all(|&b| b == 0xAAAA),
                            "untouched bf16 row {row} was modified"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn scatter_rejects_out_of_range_and_duplicate_positions_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let scatter = F32Scatter::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let source = DeviceBuffer::<f32>::new(&context, 3 * 8).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 4 * 8).unwrap();

        let out_of_range = DeviceBuffer::from_host(&context, &stream, &[0_u32, 2, 9]).unwrap();
        let error = scatter
            .scatter(&stream, &source, &out_of_range, &output, 4, 8)
            .unwrap_err();
        assert!(error.to_string().contains("out of range"), "{error}");

        let duplicated = DeviceBuffer::from_host(&context, &stream, &[1_u32, 3, 1]).unwrap();
        let error = scatter
            .scatter(&stream, &source, &duplicated, &output, 4, 8)
            .unwrap_err();
        assert!(error.to_string().contains("duplicate"), "{error}");

        // Wrong source size is rejected before any launch.
        let valid = DeviceBuffer::from_host(&context, &stream, &[0_u32, 2]).unwrap();
        let short_source = DeviceBuffer::<f32>::new(&context, 15).unwrap();
        let error = scatter
            .scatter(&stream, &short_source, &valid, &output, 4, 8)
            .unwrap_err();
        assert!(error.to_string().contains("requires 16"), "{error}");
    }
}
