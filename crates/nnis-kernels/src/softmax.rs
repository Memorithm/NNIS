//! Numerically stable `f32` softmax built from native multi-pass stages.
//!
//! The pipeline enqueues, on one stream and without host roundtrips:
//!
//! 1. device-side maximum of the input ([`F32Reduction::enqueue_max`])
//! 2. `output = exp(input - max)`
//! 3. device-side sum of the exponentials ([`F32Reduction::enqueue_sum`])
//! 4. in-place normalization by that device-resident sum
//!
//! Keeping the maximum and total on the device between stages avoids two
//! device-to-host synchronizations per call; a safe wrapper performs exactly
//! one synchronization after all four stages.

use crate::reduction::{F32Reduction, F32ReductionWorkspace};
use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{
    Context, DeviceBuffer, NnisError, PooledBuffer, Result, Stream, StreamOrderedAllocator,
};
use std::sync::Arc;

const SOFTMAX_SOURCE: &str = r#"
extern "C" __global__ void nnis_softmax_exp_shift_f32(
    const float* input,
    float* output,
    const float* max_value,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = expf(input[index] - max_value[0]);
    }
}

extern "C" __global__ void nnis_softmax_normalize_f32(
    float* data,
    const float* total,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        data[index] /= total[0];
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Context-bound numerically stable `f32` softmax.
#[derive(Debug)]
pub struct F32Softmax {
    reduction: F32Reduction,
    exp_shift: Kernel,
    normalize: Kernel,
    block_size: u32,
}

impl F32Softmax {
    /// Compile (or reuse cached CUBINs) and load the softmax kernel set.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load the softmax family with an explicitly selected thread-block width.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 {
            return Err(NnisError::invalid_input("softmax block size is zero"));
        }
        let reduction = F32Reduction::load(context, compiler)?;
        let code = compiler.compile_cubin(SOFTMAX_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let exp_shift = module.get_function("nnis_softmax_exp_shift_f32")?;
        let normalize = module.get_function("nnis_softmax_normalize_f32")?;
        for (name, function) in [("exp_shift", &exp_shift), ("normalize", &normalize)] {
            let attributes = function.attributes()?;
            if block_size > attributes.max_threads_per_block {
                return Err(NnisError::invalid_input(format!(
                    "softmax {name} block size {block_size} exceeds function limit {}",
                    attributes.max_threads_per_block
                )));
            }
        }
        Ok(Self {
            reduction,
            exp_shift,
            normalize,
            block_size,
        })
    }

    /// CUDA thread-block width used by the elementwise stages.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Underlying sum/max reductions shared by the softmax pipeline.
    pub fn reduction(&self) -> &F32Reduction {
        &self.reduction
    }

    /// Compute the stable softmax of `input` into `output` and wait.
    ///
    /// Scratch storage is allocated for this call. `input` and `output` must
    /// have equal lengths; an empty input leaves an empty output untouched.
    pub fn softmax(
        &self,
        context: &Arc<Context>,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        let workspace = self.reduction.workspace(context, input.len())?;
        self.softmax_into(stream, input, output, &workspace)
    }

    /// Compute the stable softmax reusing caller-provided reduction scratch
    /// and wait for completion.
    pub fn softmax_into(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        workspace: &F32ReductionWorkspace,
    ) -> Result<()> {
        let max_scratch = DeviceBuffer::<f32>::new(stream.ctx(), 1)?;
        let sum_scratch = DeviceBuffer::<f32>::new(stream.ctx(), 1)?;
        // SAFETY: every buffer borrow remains live until synchronization.
        let enqueue_result = unsafe {
            self.enqueue_softmax(stream, input, output, &max_scratch, &sum_scratch, workspace)
        };
        // Even a late submission failure may follow successful asynchronous
        // stages, so the stream is drained before any error is reported.
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Compute the stable softmax with every scratch allocation taken from a
    /// [`StreamOrderedAllocator`] and wait once: the reduction-tree workspace,
    /// and both device-side scalars are stream-ordered pool allocations whose
    /// drops enqueue cheap asynchronous frees after synchronization.
    ///
    /// This is the design-note pipeline case: identical GPU work to
    /// [`Self::softmax`], but steady-state calls pay no synchronous
    /// allocator cost.
    pub fn softmax_pooled(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        allocator: &StreamOrderedAllocator,
    ) -> Result<()> {
        let workspace = self.reduction.pooled_workspace(allocator, input.len())?;
        let max_scratch = allocator.alloc::<f32>(1)?;
        let sum_scratch = allocator.alloc::<f32>(1)?;
        // SAFETY: every buffer borrow remains live until synchronization
        // below; the pooled temporaries drop only afterwards.
        let enqueue_result = unsafe {
            self.enqueue_softmax_pooled(
                stream,
                input,
                output,
                &max_scratch,
                &sum_scratch,
                &workspace,
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

    /// Enqueue the complete four-stage softmax over pooled scratch without
    /// synchronizing.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, the allocator's pool, and the workspace must
    /// remain alive and otherwise untouched until the stream completes. The
    /// workspace and scalar scratches may not be shared by overlapping
    /// operations, including on other streams.
    pub unsafe fn enqueue_softmax_pooled(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        max_scratch: &PooledBuffer<f32>,
        sum_scratch: &PooledBuffer<f32>,
        workspace: &F32ReductionWorkspace,
    ) -> Result<()> {
        self.validate_pooled_execution(stream, input, output, max_scratch, sum_scratch)?;
        if input.is_empty() {
            return Ok(());
        }
        // SAFETY: each stage retains its borrows under this method's contract;
        // same-stream ordering guarantees each stage reads only prior-stage
        // output.
        unsafe {
            self.reduction
                .enqueue_max_ptr(stream, input, max_scratch.device_ptr(), workspace)?;
            self.enqueue_exp_shift_ptr(
                stream,
                input,
                output,
                max_scratch.device_ptr(),
                input.len(),
            )?;
            self.reduction
                .enqueue_sum_ptr(stream, output, sum_scratch.device_ptr(), workspace)?;
            self.enqueue_normalize_ptr(stream, output, sum_scratch.device_ptr())?;
        }
        Ok(())
    }

    fn validate_pooled_execution(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        max_scratch: &PooledBuffer<f32>,
        sum_scratch: &PooledBuffer<f32>,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(NnisError::invalid_input(format!(
                "softmax input has {} elements but output has {}",
                input.len(),
                output.len()
            )));
        }
        for (name, scalar) in [("max_scratch", max_scratch), ("sum_scratch", sum_scratch)] {
            if scalar.len() != 1 {
                return Err(NnisError::invalid_input(format!(
                    "softmax {name} holds {} elements; expected exactly 1",
                    scalar.len()
                )));
            }
        }
        if max_scratch.device_ptr() == sum_scratch.device_ptr() {
            return Err(NnisError::invalid_input(
                "softmax max_scratch and sum_scratch must be distinct buffers",
            ));
        }
        if !Arc::ptr_eq(self.context(), stream.ctx())
            || !Arc::ptr_eq(self.context(), input.ctx())
            || !Arc::ptr_eq(self.context(), output.ctx())
            || !Arc::ptr_eq(self.context(), max_scratch.ctx())
            || !Arc::ptr_eq(self.context(), sum_scratch.ctx())
        {
            return Err(NnisError::invalid_input(
                "softmax stream, buffers, and kernels must share one context",
            ));
        }
        Ok(())
    }

    /// Enqueue the complete four-stage softmax without synchronizing.
    ///
    /// On return, `output` is scheduled to hold one probability per input
    /// element. An empty input schedules no work. `max_scratch` and
    /// `sum_scratch` must be distinct one-element buffers receiving the
    /// intermediate device-side scalars.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, this kernel set, and the workspace must
    /// remain alive and otherwise untouched until the stream completes. The
    /// workspace and scalar scratches may not be shared by overlapping
    /// operations, including on other streams.
    pub unsafe fn enqueue_softmax(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        max_scratch: &DeviceBuffer<f32>,
        sum_scratch: &DeviceBuffer<f32>,
        workspace: &F32ReductionWorkspace,
    ) -> Result<()> {
        self.validate_execution(stream, input, output, max_scratch, sum_scratch)?;
        if input.is_empty() {
            return Ok(());
        }
        // SAFETY: each stage retains its borrows under this method's contract;
        // same-stream ordering guarantees each stage reads only prior-stage
        // output.
        unsafe {
            self.reduction
                .enqueue_max(stream, input, max_scratch, workspace)?;
            self.enqueue_exp_shift(stream, input, output, max_scratch)?;
            self.reduction
                .enqueue_sum(stream, output, sum_scratch, workspace)?;
            self.enqueue_normalize(stream, output, sum_scratch)?;
        }
        Ok(())
    }

    fn validate_execution(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        max_scratch: &DeviceBuffer<f32>,
        sum_scratch: &DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(NnisError::invalid_input(format!(
                "softmax input has {} elements but output has {}",
                input.len(),
                output.len()
            )));
        }
        for (name, scalar) in [("max_scratch", max_scratch), ("sum_scratch", sum_scratch)] {
            if scalar.len() != 1 {
                return Err(NnisError::invalid_input(format!(
                    "softmax {name} holds {} elements; expected exactly 1",
                    scalar.len()
                )));
            }
        }
        if max_scratch.device_ptr() == sum_scratch.device_ptr() {
            return Err(NnisError::invalid_input(
                "softmax max_scratch and sum_scratch must be distinct buffers",
            ));
        }
        if !Arc::ptr_eq(self.context(), stream.ctx())
            || !Arc::ptr_eq(self.context(), input.ctx())
            || !Arc::ptr_eq(self.context(), output.ctx())
            || !Arc::ptr_eq(self.context(), max_scratch.ctx())
            || !Arc::ptr_eq(self.context(), sum_scratch.ctx())
        {
            return Err(NnisError::invalid_input(
                "softmax stream, buffers, and kernels must share one context",
            ));
        }
        Ok(())
    }

    /// # Safety
    ///
    /// Caller owns the asynchronous lifetime of every buffer; see
    /// [`F32Softmax::enqueue_softmax`].
    unsafe fn enqueue_exp_shift(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        max_value: &DeviceBuffer<f32>,
    ) -> Result<()> {
        // SAFETY: typed wrapper over the pointer form; lifetimes owned by
        // the enclosing operation.
        unsafe {
            self.enqueue_exp_shift_ptr(stream, input, output, max_value.device_ptr(), input.len())
        }
    }

    /// # Safety
    ///
    /// `max_value_ptr` must name a live one-element `f32` allocation; see
    /// the enclosing operation's documented lifetime contract.
    unsafe fn enqueue_exp_shift_ptr(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        max_value_ptr: u64,
        elements_len: usize,
    ) -> Result<()> {
        let elements = u64::try_from(elements_len)
            .map_err(|_| NnisError::invalid_input("softmax length exceeds u64::MAX"))?;
        let mut arguments = KernelArgs::with_capacity(4, 2);
        arguments
            .push_buffer(input)
            .push_buffer(output)
            .push(max_value_ptr)
            .push(elements);
        let launch = KernelLaunch::new(
            &self.exp_shift,
            stream,
            LaunchConfig::for_num_elements(elements_len, self.block_size)?,
        );
        // SAFETY: argument order/widths match `nnis_softmax_exp_shift_f32`;
        // the caller owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    /// # Safety
    ///
    /// Caller owns the asynchronous lifetime of every buffer; see
    /// [`F32Softmax::enqueue_softmax`]. Normalization writes `data` in place.
    unsafe fn enqueue_normalize(
        &self,
        stream: &Stream,
        data: &DeviceBuffer<f32>,
        total: &DeviceBuffer<f32>,
    ) -> Result<()> {
        // SAFETY: typed wrapper over the pointer form; lifetimes owned by
        // the enclosing operation.
        unsafe { self.enqueue_normalize_ptr(stream, data, total.device_ptr()) }
    }

    /// # Safety
    ///
    /// `total_ptr` must name a live one-element `f32` allocation; see the
    /// enclosing operation's documented lifetime contract. Normalization
    /// writes `data` in place.
    unsafe fn enqueue_normalize_ptr(
        &self,
        stream: &Stream,
        data: &DeviceBuffer<f32>,
        total_ptr: u64,
    ) -> Result<()> {
        let elements = u64::try_from(data.len())
            .map_err(|_| NnisError::invalid_input("softmax length exceeds u64::MAX"))?;
        let mut arguments = KernelArgs::with_capacity(3, 1);
        arguments.push_buffer(data).push(total_ptr).push(elements);
        let launch = KernelLaunch::new(
            &self.normalize,
            stream,
            LaunchConfig::for_num_elements(data.len(), self.block_size)?,
        );
        // SAFETY: argument order/widths match `nnis_softmax_normalize_f32`;
        // same-stream ordering guarantees the total is final before use.
        unsafe { launch.launch(&mut arguments) }
    }

    fn context(&self) -> &Arc<Context> {
        self.exp_shift.context()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    const TEST_SIZES: &[usize] = &[0, 1, 31, 32, 255, 256, 257, 1_025, 4_097];

    /// Deterministic pseudo-random values spanning several orders of
    /// magnitude, including large magnitudes that would overflow a naive
    /// `expf(x)` implementation.
    fn host_values(size: usize) -> Vec<f32> {
        (0..size)
            .map(|index| {
                let spread = ((index % 17) as f32 - 8.0) * 37.5;
                let ripple = ((index * 7 % 101) as f32 - 50.0) * 0.25;
                spread + ripple
            })
            .collect()
    }

    fn reference_softmax(input: &[f32]) -> Vec<f64> {
        if input.is_empty() {
            return Vec::new();
        }
        let maximum = input
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |acc, value| acc.max(f64::from(value)));
        let shifted: Vec<f64> = input
            .iter()
            .map(|&value| f64::from(value) - maximum)
            .collect();
        let exponentials: Vec<f64> = shifted.iter().map(|&value| value.exp()).collect();
        let total: f64 = exponentials.iter().sum();
        exponentials
            .into_iter()
            .map(|value| value / total)
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            // expf contributes at most a few ulps per element and the f32
            // accumulation of totals dominates; 1e-5 relative with a small
            // absolute floor covers both without weakening validation.
            let tolerance = 1.0e-6_f32.max((expected.abs() as f32) * 1.0e-5);
            assert!(
                (actual - expected as f32).abs() <= tolerance,
                "softmax mismatch at {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn softmax_matches_high_precision_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let softmax = F32Softmax::load(&context, &compiler).unwrap();
        let maximum = *TEST_SIZES.iter().max().unwrap();
        let workspace = softmax.reduction().workspace(&context, maximum).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &size in TEST_SIZES {
            let host = host_values(size);
            let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
            let output = DeviceBuffer::<f32>::new(&context, size).unwrap();
            softmax
                .softmax_into(&stream, &input, &output, &workspace)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            assert_close(&actual, &reference_softmax(&host));
            if size == 0 {
                continue;
            }
            let probability_sum: f32 = actual.iter().sum();
            let sum_tolerance = 2.0e-4_f32.max(f32::EPSILON * size as f32);
            assert!(
                (probability_sum - 1.0).abs() <= sum_tolerance,
                "probabilities sum to {probability_sum} for {size} elements"
            );
        }

        // A single element always yields exactly 1.0.
        let singleton_input = DeviceBuffer::from_host(&context, &stream, &[-123.5_f32]).unwrap();
        let singleton_output = DeviceBuffer::<f32>::new(&context, 1).unwrap();
        softmax
            .softmax(&context, &stream, &singleton_input, &singleton_output)
            .unwrap();
        assert_eq!(singleton_output.to_vec(&stream).unwrap()[0], 1.0);

        // Constant inputs yield a uniform distribution.
        let uniform_input = DeviceBuffer::from_host(&context, &stream, &[9.75_f32; 100]).unwrap();
        let uniform_output = DeviceBuffer::<f32>::new(&context, 100).unwrap();
        softmax
            .softmax(&context, &stream, &uniform_input, &uniform_output)
            .unwrap();
        for (index, value) in uniform_output.to_vec(&stream).unwrap().iter().enumerate() {
            assert!(
                (value - 0.01).abs() <= 1.0e-6,
                "uniform mismatch at {index}: {value}"
            );
        }
    }

    #[test]
    fn pooled_softmax_matches_oracle_across_sizes_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let allocator = StreamOrderedAllocator::new(&stream).unwrap();
        let compiler = JitCompiler::new();
        let softmax = F32Softmax::load(&context, &compiler).unwrap();

        for &size in TEST_SIZES {
            let host = host_values(size);
            let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
            let output = DeviceBuffer::<f32>::new(&context, size).unwrap();
            softmax
                .softmax_pooled(&stream, &input, &output, &allocator)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            let expected = reference_softmax(&host);
            assert_eq!(actual.len(), expected.len());
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let tolerance = 1.0e-6_f32.max((expected.abs() as f32) * 1.0e-5);
                assert!(
                    (actual - expected as f32).abs() <= tolerance,
                    "pooled mismatch at {index} size {size}: {actual} != {expected}"
                );
            }
        }

        // Singleton through the pooled path must be exactly 1.0.
        let input = DeviceBuffer::from_host(&context, &stream, &[7.5_f32]).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 1).unwrap();
        softmax
            .softmax_pooled(&stream, &input, &output, &allocator)
            .unwrap();
        let value = output.to_vec(&stream).unwrap()[0];
        assert!((value - 1.0).abs() <= 1.0e-7, "singleton: {value}");

        // Undersized pooled workspace is rejected before any launch.
        let small_allocator_input =
            DeviceBuffer::from_host(&context, &stream, &host_values(64)).unwrap();
        let small_output = DeviceBuffer::<f32>::new(&context, 64).unwrap();
        let strict_allocator = StreamOrderedAllocator::new(&stream).unwrap();
        let workspace = softmax
            .reduction()
            .pooled_workspace(&strict_allocator, 8)
            .unwrap();
        let max_scratch = strict_allocator.alloc::<f32>(1).unwrap();
        let sum_scratch = strict_allocator.alloc::<f32>(1).unwrap();
        let error = unsafe {
            softmax.enqueue_softmax_pooled(
                &stream,
                &small_allocator_input,
                &small_output,
                &max_scratch,
                &sum_scratch,
                &workspace,
            )
        }
        .unwrap_err();
        assert!(error.to_string().contains("capacity is 8"), "{error}");
    }

    #[test]
    fn softmax_rejects_invalid_shapes_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(F32Softmax::load_with_block_size(&context, &compiler, 0).is_err());
        let softmax = F32Softmax::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let short = DeviceBuffer::<f32>::new(&context, 3).unwrap();
        let long = DeviceBuffer::<f32>::new(&context, 4).unwrap();
        let error = softmax
            .softmax(&context, &stream, &short, &long)
            .unwrap_err();
        assert!(error.to_string().contains("but output has 4"), "{error}");

        let workspace = softmax.reduction().workspace(&context, 4).unwrap();
        let undersized = softmax.reduction().workspace(&context, 2).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 4).unwrap();
        let input = DeviceBuffer::<f32>::new(&context, 4).unwrap();
        softmax
            .softmax_into(&stream, &input, &output, &workspace)
            .unwrap();
        assert!(softmax
            .softmax_into(&stream, &input, &output, &undersized)
            .is_err());
    }
}
