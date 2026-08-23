//! Multi-pass `f32` reductions.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const REDUCTION_SOURCE: &str = r#"
extern "C" __global__ void nnis_reduce_sum_f32(
    const float* input,
    float* output,
    unsigned long long elements
) {
    extern __shared__ float partial[];
    const unsigned int lane = threadIdx.x;
    const unsigned long long first =
        ((unsigned long long)blockIdx.x * blockDim.x * 2) + lane;
    const unsigned long long second = first + blockDim.x;

    float value = 0.0f;
    if (first < elements) {
        value = input[first];
    }
    if (second < elements) {
        value += input[second];
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
        output[blockIdx.x] = partial[0];
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Reusable scratch storage for an [`F32Reduction`] sum.
///
/// A workspace is context- and block-size-specific. It may be reused for any
/// input no larger than `max_elements`, but not by overlapping asynchronous
/// operations.
#[derive(Debug)]
pub struct F32ReductionWorkspace {
    max_elements: usize,
    block_size: u32,
    scratch_a: DeviceBuffer<f32>,
    scratch_b: DeviceBuffer<f32>,
}

impl F32ReductionWorkspace {
    pub fn max_elements(&self) -> usize {
        self.max_elements
    }

    pub fn scratch_capacity(&self) -> usize {
        self.scratch_a.len()
    }

    fn context(&self) -> &Arc<Context> {
        self.scratch_a.ctx()
    }
}

/// Context-bound, multi-pass `f32` sum reduction.
#[derive(Debug)]
pub struct F32Reduction {
    sum: Kernel,
    block_size: u32,
}

impl F32Reduction {
    /// Compile (or reuse cached CUBIN) and load the default reduction kernel.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load the reduction with an explicit power-of-two thread-block width.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(NnisError::invalid_input(format!(
                "reduction block size {block_size} is not a non-zero power of two"
            )));
        }
        let shared_memory_bytes = block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| NnisError::invalid_input("reduction shared-memory size overflows"))?;
        let code =
            compiler.compile_cubin(REDUCTION_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let sum = module.get_function("nnis_reduce_sum_f32")?;
        let attributes = sum.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "reduction block size {block_size} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        if shared_memory_bytes > attributes.max_dynamic_shared_memory_bytes {
            return Err(NnisError::invalid_input(format!(
                "reduction requires {shared_memory_bytes} shared-memory bytes; function limit is {}",
                attributes.max_dynamic_shared_memory_bytes
            )));
        }
        Ok(Self { sum, block_size })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Allocate scratch storage reusable for inputs up to `max_elements`.
    pub fn workspace(
        &self,
        context: &Arc<Context>,
        max_elements: usize,
    ) -> Result<F32ReductionWorkspace> {
        if !Arc::ptr_eq(context, self.sum.context()) {
            return Err(NnisError::invalid_input(
                "reduction and workspace contexts do not match",
            ));
        }
        let scratch_elements = partial_count(max_elements, self.block_size)?;
        Ok(F32ReductionWorkspace {
            max_elements,
            block_size: self.block_size,
            scratch_a: DeviceBuffer::new(context, scratch_elements)?,
            scratch_b: DeviceBuffer::new(context, scratch_elements)?,
        })
    }

    /// Reduce an input and return its host scalar, waiting for completion.
    pub fn sum(&self, stream: &Stream, input: &DeviceBuffer<f32>) -> Result<f32> {
        let workspace = self.workspace(input.ctx(), input.len())?;
        let output = DeviceBuffer::<f32>::new(input.ctx(), 1)?;
        self.sum_into(stream, input, &output, &workspace)?;
        Ok(output.to_vec(stream)?[0])
    }

    /// Reduce into a one-element device buffer and wait for completion.
    pub fn sum_into(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        workspace: &F32ReductionWorkspace,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe { self.enqueue_sum(stream, input, output, workspace) };
        let synchronize_result = stream.synchronize();
        match enqueue_result {
            Ok(()) => synchronize_result,
            Err(error) => {
                // Even a later-pass submission failure may follow successful
                // asynchronous passes. Synchronization above retains every
                // borrow through their completion before returning the cause.
                let _ = synchronize_result;
                Err(error)
            }
        }
    }

    /// Enqueue every reduction pass without synchronizing the stream.
    ///
    /// On return, `output[0]` is scheduled to receive the sum. For an empty
    /// input the scheduled result is `0.0`.
    ///
    /// # Safety
    ///
    /// The reduction, stream, input, output, and workspace must remain alive
    /// and otherwise untouched until the stream completes. The workspace may
    /// not be shared by overlapping operations, including on other streams.
    pub unsafe fn enqueue_sum(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        workspace: &F32ReductionWorkspace,
    ) -> Result<()> {
        self.validate_execution(stream, input, output, workspace)?;
        if input.is_empty() {
            // SAFETY: the output and stream lifetime obligation is inherited by
            // this method's contract.
            return unsafe { output.zero_async(stream) };
        }

        let mut current = input;
        let mut current_elements = input.len();
        let mut write_scratch_a = true;
        loop {
            let output_elements = partial_count(current_elements, self.block_size)?;
            let destination = if output_elements == 1 {
                output
            } else if write_scratch_a {
                &workspace.scratch_a
            } else {
                &workspace.scratch_b
            };
            // SAFETY: buffers are distinct workspace/output allocations, all
            // passes are ordered on one stream, and caller owns the lifetime.
            unsafe {
                self.enqueue_pass(stream, current, destination, current_elements)?;
            }
            if output_elements == 1 {
                break;
            }
            current = destination;
            current_elements = output_elements;
            write_scratch_a = !write_scratch_a;
        }
        Ok(())
    }

    fn validate_execution(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        workspace: &F32ReductionWorkspace,
    ) -> Result<()> {
        if output.len() != 1 {
            return Err(NnisError::invalid_input(format!(
                "reduction output has {} elements; expected 1",
                output.len()
            )));
        }
        if input.len() > workspace.max_elements {
            return Err(NnisError::invalid_input(format!(
                "reduction input has {} elements; workspace capacity is {}",
                input.len(),
                workspace.max_elements
            )));
        }
        if workspace.block_size != self.block_size {
            return Err(NnisError::invalid_input(format!(
                "workspace block size {} does not match reduction block size {}",
                workspace.block_size, self.block_size
            )));
        }
        let context = self.sum.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, input.ctx())
            || !Arc::ptr_eq(context, output.ctx())
            || !Arc::ptr_eq(context, workspace.context())
        {
            return Err(NnisError::invalid_input(
                "reduction stream, buffers, workspace, and kernel must share one context",
            ));
        }
        Ok(())
    }

    unsafe fn enqueue_pass(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        elements: usize,
    ) -> Result<()> {
        let output_elements = partial_count(elements, self.block_size)?;
        if output_elements == 0 || output.len() < output_elements {
            return Err(NnisError::invalid_input(format!(
                "reduction pass needs {output_elements} output elements; allocation has {}",
                output.len()
            )));
        }
        let grid_size = u32::try_from(output_elements)
            .map_err(|_| NnisError::invalid_input("reduction grid exceeds u32::MAX blocks"))?;
        let elements = u64::try_from(elements)
            .map_err(|_| NnisError::invalid_input("reduction length exceeds u64::MAX"))?;
        let shared_memory_bytes = self
            .block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| NnisError::invalid_input("reduction shared-memory size overflows"))?;
        let config = LaunchConfig::new(Dim3::x(grid_size), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(shared_memory_bytes);
        let mut arguments = KernelArgs::with_capacity(3, 2);
        arguments
            .push_buffer(input)
            .push_buffer(output)
            .push(elements);
        let launch = KernelLaunch::new(&self.sum, stream, config);
        // SAFETY: argument order/widths match `nnis_reduce_sum_f32`; the
        // enclosing operation owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }
}

fn partial_count(elements: usize, block_size: u32) -> Result<usize> {
    if elements == 0 {
        return Ok(0);
    }
    let elements_per_block = (block_size as usize)
        .checked_mul(2)
        .ok_or_else(|| NnisError::invalid_input("reduction block span overflows usize"))?;
    Ok(elements.div_ceil(elements_per_block))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    const TEST_SIZES: &[usize] = &[
        0, 1, 2, 3, 31, 32, 255, 256, 257, 511, 512, 513, 1_023, 1_024, 1_025, 131_071, 1_000_003,
    ];

    fn reference_tree_sum(input: &[f32], block_size: usize) -> f32 {
        if input.is_empty() {
            return 0.0;
        }
        let mut current = input.to_vec();
        loop {
            let output_elements = current.len().div_ceil(block_size * 2);
            let mut output = vec![0.0_f32; output_elements];
            for (block, result) in output.iter_mut().enumerate() {
                let mut shared = vec![0.0_f32; block_size];
                let base = block * block_size * 2;
                for (lane, value) in shared.iter_mut().enumerate() {
                    if let Some(first) = current.get(base + lane) {
                        *value = *first;
                    }
                    if let Some(second) = current.get(base + block_size + lane) {
                        *value += *second;
                    }
                }
                let mut stride = block_size / 2;
                while stride > 0 {
                    for lane in 0..stride {
                        shared[lane] += shared[lane + stride];
                    }
                    stride /= 2;
                }
                *result = shared[0];
            }
            if output.len() == 1 {
                return output[0];
            }
            current = output;
        }
    }

    fn forward_error_bound(input: &[f32]) -> f32 {
        if input.is_empty() {
            return 0.0;
        }
        let depth = (usize::BITS - (input.len() - 1).leading_zeros()) as f64 + 1.0;
        let epsilon = f32::EPSILON as f64;
        let gamma = depth * epsilon / (1.0 - depth * epsilon);
        let magnitude = input
            .iter()
            .map(|value| f64::from(value.abs()))
            .sum::<f64>();
        (gamma * magnitude).max(f64::from(f32::EPSILON)) as f32
    }

    #[test]
    fn sum_matches_ordered_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let reduction = F32Reduction::load(&context, &compiler).unwrap();
        let maximum = *TEST_SIZES.iter().max().unwrap();
        let workspace = reduction.workspace(&context, maximum).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 1).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &size in TEST_SIZES {
            let host = (0..size)
                .map(|index| {
                    let numerator = (index * 37 % 1_009) as f32 - 504.0;
                    numerator / 127.0
                })
                .collect::<Vec<_>>();
            let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
            reduction
                .sum_into(&stream, &input, &output, &workspace)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap()[0];
            let ordered = reference_tree_sum(&host, reduction.block_size() as usize);
            assert_eq!(
                actual.to_bits(),
                ordered.to_bits(),
                "ordered sum mismatch for {size} elements: {actual} != {ordered}"
            );

            let high_precision = host.iter().map(|&value| f64::from(value)).sum::<f64>();
            let error = (f64::from(actual) - high_precision).abs();
            let tolerance = f64::from(forward_error_bound(&host));
            assert!(
                error <= tolerance,
                "sum error for {size} elements is {error}, tolerance is {tolerance}"
            );
        }

        let singleton = DeviceBuffer::from_host(&context, &stream, &[3.25_f32]).unwrap();
        assert_eq!(reduction.sum(&stream, &singleton).unwrap(), 3.25);
    }

    #[test]
    fn reduction_rejects_invalid_shapes_and_workspace() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(F32Reduction::load_with_block_size(&context, &compiler, 0).is_err());
        assert!(F32Reduction::load_with_block_size(&context, &compiler, 192).is_err());
        let reduction = F32Reduction::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let input = DeviceBuffer::from_host(&context, &stream, &[1.0_f32; 4]).unwrap();
        let short_workspace = reduction.workspace(&context, 3).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 1).unwrap();
        assert!(reduction
            .sum_into(&stream, &input, &output, &short_workspace)
            .is_err());

        let workspace = reduction.workspace(&context, input.len()).unwrap();
        let wrong_output = DeviceBuffer::<f32>::new(&context, 2).unwrap();
        assert!(reduction
            .sum_into(&stream, &input, &wrong_output, &workspace)
            .is_err());
    }
}
