//! Multi-pass `f32` reductions.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{
    Context, DeviceBuffer, NnisError, PooledBuffer, Result, Stream, StreamOrderedAllocator,
};
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

extern "C" __global__ void nnis_reduce_max_f32(
    const float* input,
    float* output,
    unsigned long long elements
) {
    extern __shared__ float partial[];
    const unsigned int lane = threadIdx.x;
    const unsigned long long first =
        ((unsigned long long)blockIdx.x * blockDim.x * 2) + lane;
    const unsigned long long second = first + blockDim.x;

    float value = __int_as_float(0xff800000);
    if (first < elements) {
        value = input[first];
    }
    if (second < elements) {
        value = fmaxf(value, input[second]);
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
        output[blockIdx.x] = partial[0];
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Scratch storage for the reduction tree: plain or stream-ordered pooled.
///
/// Both variants expose the same launch surface; the launch path only needs
/// a device pointer and an element count.
#[derive(Debug)]
pub(crate) enum TreeScratch {
    Plain(DeviceBuffer<f32>),
    Pooled(PooledBuffer<f32>),
}

impl TreeScratch {
    pub(crate) fn ptr(&self) -> u64 {
        match self {
            Self::Plain(buffer) => buffer.device_ptr(),
            Self::Pooled(buffer) => buffer.device_ptr(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Plain(buffer) => buffer.len(),
            Self::Pooled(buffer) => buffer.len(),
        }
    }

    pub(crate) fn ctx(&self) -> &Arc<Context> {
        match self {
            Self::Plain(buffer) => buffer.ctx(),
            Self::Pooled(buffer) => buffer.ctx(),
        }
    }
}

/// Reusable scratch storage for an [`F32Reduction`] sum.
///
/// A workspace is context- and block-size-specific. It may be reused for any
/// input no larger than `max_elements`, but not by overlapping asynchronous
/// operations. The scratch slots may be plain or pooled stream-ordered
/// storage; pooling is opt-in through [`F32Reduction::pooled_workspace`].
#[derive(Debug)]
pub struct F32ReductionWorkspace {
    max_elements: usize,
    block_size: u32,
    scratch_a: TreeScratch,
    scratch_b: TreeScratch,
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

/// Context-bound, multi-pass `f32` sum and max reductions.
///
/// Both operations share the same tree structure, so a workspace serves
/// either one.
#[derive(Debug)]
pub struct F32Reduction {
    sum: Kernel,
    max: Kernel,
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
        let max = module.get_function("nnis_reduce_max_f32")?;
        let attributes = sum.attributes()?;
        let max_attributes = max.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "reduction block size {block_size} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        if block_size > max_attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "reduction block size {block_size} exceeds function limit {}",
                max_attributes.max_threads_per_block
            )));
        }
        if shared_memory_bytes > attributes.max_dynamic_shared_memory_bytes {
            return Err(NnisError::invalid_input(format!(
                "reduction requires {shared_memory_bytes} shared-memory bytes; function limit is {}",
                attributes.max_dynamic_shared_memory_bytes
            )));
        }
        if shared_memory_bytes > max_attributes.max_dynamic_shared_memory_bytes {
            return Err(NnisError::invalid_input(format!(
                "reduction requires {shared_memory_bytes} shared-memory bytes; function limit is {}",
                max_attributes.max_dynamic_shared_memory_bytes
            )));
        }
        Ok(Self {
            sum,
            max,
            block_size,
        })
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
            scratch_a: TreeScratch::Plain(DeviceBuffer::new(context, scratch_elements)?),
            scratch_b: TreeScratch::Plain(DeviceBuffer::new(context, scratch_elements)?),
        })
    }

    /// Build a workspace whose tree scratch is allocated stream-ordered
    /// from `allocator` (pooled frees enqueue on the allocator's stream at
    /// drop). Capacity semantics match [`Self::workspace`].
    pub fn pooled_workspace(
        &self,
        allocator: &StreamOrderedAllocator,
        max_elements: usize,
    ) -> Result<F32ReductionWorkspace> {
        if !Arc::ptr_eq(allocator.context(), self.sum.context()) {
            return Err(NnisError::invalid_input(
                "reduction and allocator contexts do not match",
            ));
        }
        let scratch_elements = partial_count(max_elements, self.block_size)?;
        Ok(F32ReductionWorkspace {
            max_elements,
            block_size: self.block_size,
            scratch_a: TreeScratch::Pooled(allocator.alloc(scratch_elements)?),
            scratch_b: TreeScratch::Pooled(allocator.alloc(scratch_elements)?),
        })
    }

    /// Reduce an input and return its host scalar, waiting for completion.
    pub fn sum(&self, stream: &Stream, input: &DeviceBuffer<f32>) -> Result<f32> {
        self.sum_scalar(stream, input)
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
        Self::join(enqueue_result, stream.synchronize())
    }

    /// Return the maximum element as a host scalar, waiting for completion.
    /// An empty input yields `-infinity`, matching the kernel identity. `NaN`
    /// inputs follow `fmaxf` semantics: non-NaN operands win.
    pub fn max(&self, stream: &Stream, input: &DeviceBuffer<f32>) -> Result<f32> {
        self.max_scalar(stream, input)
    }

    /// Reduce the maximum into a one-element device buffer and wait.
    ///
    /// An empty input leaves `output[0]` untouched.
    pub fn max_into(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        workspace: &F32ReductionWorkspace,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe { self.enqueue_max(stream, input, output, workspace) };
        Self::join(enqueue_result, stream.synchronize())
    }

    /// Enqueue every sum-reduction pass without synchronizing the stream.
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
        // SAFETY: identical buffers and lifetime obligations as this method's
        // documented contract; only the reduction operator differs.
        unsafe {
            self.enqueue_tree(
                &self.sum,
                stream,
                input,
                output.device_ptr(),
                output.len(),
                workspace,
                TreeKind::Sum,
            )
        }
    }

    /// Pointer-form [`Self::enqueue_sum`]: the one-element scalar
    /// destination is passed by device address.
    ///
    /// # Safety
    ///
    /// `output_ptr` must name a live one-element `f32` allocation distinct
    /// from every workspace slot; remaining obligations match
    /// [`Self::enqueue_sum`].
    pub unsafe fn enqueue_sum_ptr(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output_ptr: u64,
        workspace: &F32ReductionWorkspace,
    ) -> Result<()> {
        // SAFETY: caller owns the pointer's lifetime per this method's docs.
        unsafe {
            self.enqueue_tree(
                &self.sum,
                stream,
                input,
                output_ptr,
                1,
                workspace,
                TreeKind::Sum,
            )
        }
    }

    /// Enqueue every max-reduction pass without synchronizing the stream.
    ///
    /// On return, `output[0]` is scheduled to receive the maximum. For an
    /// empty input nothing is enqueued and `output[0]` is left untouched.
    ///
    /// # Safety
    ///
    /// Same obligations as [`F32Reduction::enqueue_sum`].
    pub unsafe fn enqueue_max(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        workspace: &F32ReductionWorkspace,
    ) -> Result<()> {
        // SAFETY: identical buffers and lifetime obligations as `enqueue_sum`.
        unsafe {
            self.enqueue_tree(
                &self.max,
                stream,
                input,
                output.device_ptr(),
                output.len(),
                workspace,
                TreeKind::Max,
            )
        }
    }

    /// Pointer-form [`Self::enqueue_max`].
    ///
    /// # Safety
    ///
    /// `output_ptr` must name a live one-element `f32` allocation distinct
    /// from every workspace slot; remaining obligations match
    /// [`Self::enqueue_sum`].
    pub unsafe fn enqueue_max_ptr(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output_ptr: u64,
        workspace: &F32ReductionWorkspace,
    ) -> Result<()> {
        // SAFETY: caller owns the pointer's lifetime per this method's docs.
        unsafe {
            self.enqueue_tree(
                &self.max,
                stream,
                input,
                output_ptr,
                1,
                workspace,
                TreeKind::Max,
            )
        }
    }

    /// # Safety
    ///
    /// `output_ptr`/`output_len` describe a live allocation; obligations
    /// mirror [`Self::enqueue_sum`] / [`Self::enqueue_max`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn enqueue_tree(
        &self,
        kernel: &Kernel,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output_ptr: u64,
        output_len: usize,
        workspace: &F32ReductionWorkspace,
        kind: TreeKind,
    ) -> Result<()> {
        if output_len != 1 {
            return Err(NnisError::invalid_input(format!(
                "reduction output has {output_len} elements; expected 1"
            )));
        }
        self.validate_execution(stream, input, output_ptr, workspace)?;
        if input.is_empty() {
            if matches!(kind, TreeKind::Sum) {
                let context = self.sum.context();
                context.set_current()?;
                let api = nnis_sys::driver::api()?;
                // SAFETY: the pointer names a live one-element allocation;
                // lifetime inherited by this method's contract.
                let rc = unsafe { (api.cuMemsetD8Async)(output_ptr as usize, 0, 4, stream.raw()) };
                if rc != 0 {
                    return Err(NnisError::driver("cuMemsetD8Async", rc).with("bytes", 4));
                }
            }
            return Ok(());
        }

        let mut current_ptr = input.device_ptr();
        let mut current_elements = input.len();
        let mut write_scratch_a = true;
        loop {
            let output_elements = partial_count(current_elements, self.block_size)?;
            let (destination_ptr, destination_capacity) = if output_elements == 1 {
                (output_ptr, output_len)
            } else if write_scratch_a {
                (workspace.scratch_a.ptr(), workspace.scratch_a.len())
            } else {
                (workspace.scratch_b.ptr(), workspace.scratch_b.len())
            };
            // SAFETY: pointers name distinct live allocations (input/output
            // or workspace slots), all passes are ordered on one stream, and
            // the caller owns every asynchronous lifetime involved.
            unsafe {
                self.enqueue_pass(
                    kernel,
                    stream,
                    current_ptr,
                    current_elements,
                    destination_ptr,
                    destination_capacity,
                )?;
            }
            if output_elements == 1 {
                break;
            }
            current_ptr = destination_ptr;
            current_elements = output_elements;
            write_scratch_a = !write_scratch_a;
        }
        Ok(())
    }

    fn validate_execution(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        _output_ptr: u64,
        workspace: &F32ReductionWorkspace,
    ) -> Result<()> {
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
            || !Arc::ptr_eq(context, workspace.context())
        {
            return Err(NnisError::invalid_input(
                "reduction stream, buffers, workspace, and kernel must share one context",
            ));
        }
        Ok(())
    }

    /// Allocate scratch and scalar storage, run one sum reduction, and copy
    /// the scalar back to the host.
    fn sum_scalar(&self, stream: &Stream, input: &DeviceBuffer<f32>) -> Result<f32> {
        let workspace = self.workspace(input.ctx(), input.len())?;
        let output = DeviceBuffer::<f32>::new(input.ctx(), 1)?;
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe { self.enqueue_sum(stream, input, &output, &workspace) };
        Self::join(enqueue_result, stream.synchronize())?;
        Ok(output.to_vec(stream)?[0])
    }

    /// Allocate scratch and scalar storage, run one max reduction, and copy
    /// the scalar back to the host.
    fn max_scalar(&self, stream: &Stream, input: &DeviceBuffer<f32>) -> Result<f32> {
        let workspace = self.workspace(input.ctx(), input.len())?;
        let output = DeviceBuffer::<f32>::new(input.ctx(), 1)?;
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe { self.enqueue_max(stream, input, &output, &workspace) };
        Self::join(enqueue_result, stream.synchronize())?;
        Ok(output.to_vec(stream)?[0])
    }

    /// Synchronize after an enqueue result, always draining the stream so
    /// submitted asynchronous work completes before any error is reported.
    fn join(enqueue_result: Result<()>, synchronize_result: Result<()>) -> Result<()> {
        match enqueue_result {
            Ok(()) => synchronize_result,
            Err(error) => {
                // Even a later-pass submission failure may follow successful
                // asynchronous passes. The synchronization above retains every
                // borrow through their completion before returning the cause.
                let _ = synchronize_result;
                Err(error)
            }
        }
    }

    /// # Safety
    ///
    /// Both pointers must name distinct live `f32` device allocations with
    /// at least the stated capacities; see the enclosing operation's
    /// documented asynchronous-lifetime contract.
    unsafe fn enqueue_pass(
        &self,
        kernel: &Kernel,
        stream: &Stream,
        input_ptr: u64,
        elements: usize,
        output_ptr: u64,
        output_capacity: usize,
    ) -> Result<()> {
        let output_elements = partial_count(elements, self.block_size)?;
        if output_elements == 0 || output_capacity < output_elements {
            return Err(NnisError::invalid_input(format!(
                "reduction pass needs {output_elements} output elements; allocation has {output_capacity}"
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
        let mut arguments = KernelArgs::with_capacity(3, 0);
        arguments.push(input_ptr).push(output_ptr).push(elements);
        let launch = KernelLaunch::new(kernel, stream, config);
        // SAFETY: argument order/widths match both reduction kernels, which
        // share one signature; the enclosing operation owns the asynchronous
        // lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }
}

/// Reduction operator selector for the shared tree implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeKind {
    Sum,
    Max,
}

pub(crate) fn partial_count(elements: usize, block_size: u32) -> Result<usize> {
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
    fn max_matches_cpu_oracle_on_gpu() {
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
            // The max tree never launches for an empty input and therefore
            // leaves the destination untouched; that contract is asserted
            // explicitly after this loop.
            if size == 0 {
                continue;
            }
            // Deterministic values with a wide dynamic range and an exact
            // maximum placed at varying offsets.
            let mut host: Vec<f32> = (0..size)
                .map(|index| ((index % 61) as f32 - 30.0) * 0.5)
                .collect();
            if size > 0 {
                let peak_index = (size * 13) % size;
                host[peak_index] = 41.75;
            }
            let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
            reduction
                .max_into(&stream, &input, &output, &workspace)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap()[0];
            let expected = host.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "exact max mismatch for {size} elements: {actual} != {expected}"
            );
            assert_eq!(reduction.max(&stream, &input).unwrap(), expected);
        }

        // Empty inputs leave the destination untouched for max.
        let empty = DeviceBuffer::<f32>::new(&context, 0).unwrap();
        let untouched = DeviceBuffer::from_host(&context, &stream, &[7.5_f32]).unwrap();
        reduction
            .max_into(&stream, &empty, &untouched, &workspace)
            .unwrap();
        assert_eq!(untouched.to_vec(&stream).unwrap()[0], 7.5);

        // Negative-only inputs must not be clamped by any zero identity.
        let negatives =
            DeviceBuffer::from_host(&context, &stream, &[-4.5_f32, -1.25, -9.0]).unwrap();
        assert_eq!(reduction.max(&stream, &negatives).unwrap(), -1.25);
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
        assert!(reduction
            .max_into(&stream, &input, &wrong_output, &workspace)
            .is_err());
    }
}
