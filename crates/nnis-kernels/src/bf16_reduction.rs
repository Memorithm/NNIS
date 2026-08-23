//! Multi-pass reductions over packed `bf16` inputs.
//!
//! Storage is `u16` holding `bf16` bit patterns; every arithmetic step runs
//! in `f32`. The first tree pass widens while reducing, later passes are the
//! plain `f32` tree kernels, so results are identical to reducing the widened
//! buffer with [`crate::F32Reduction`] at the same block size.

use crate::reduction::{partial_count, TreeScratch};
use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const BF16_REDUCTION_SOURCE: &str = r#"
extern "C" __global__ void nnis_bf16_reduce_sum_f32(
    const unsigned short* input,
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
        value = __uint_as_float(((unsigned int)input[first]) << 16);
    }
    if (second < elements) {
        value += __uint_as_float(((unsigned int)input[second]) << 16);
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

extern "C" __global__ void nnis_bf16_reduce_max_f32(
    const unsigned short* input,
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
        value = __uint_as_float(((unsigned int)input[first]) << 16);
    }
    if (second < elements) {
        value = fmaxf(value, __uint_as_float(((unsigned int)input[second]) << 16));
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

/// Reusable scratch storage for a [`Bf16Reduction`] tree.
#[derive(Debug)]
pub struct Bf16ReductionWorkspace {
    max_elements: usize,
    block_size: u32,
    scratch_a: TreeScratch,
    scratch_b: TreeScratch,
}

impl Bf16ReductionWorkspace {
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

/// Context-bound, multi-pass sum and max reductions over packed `bf16`.
///
/// Inputs are `DeviceBuffer<u16>` whose elements are `bf16` bit patterns.
/// Accumulation happens in `f32`; the max comparison operates on exactly
/// widened values and is therefore bit-exact against a host scan.
#[derive(Debug)]
pub struct Bf16Reduction {
    bf16_sum: Kernel,
    bf16_max: Kernel,
    sum: Kernel,
    max: Kernel,
    block_size: u32,
}

impl Bf16Reduction {
    /// Compile (or reuse cached CUBIN) and load the default bf16 reduction.
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
                "bf16 reduction block size {block_size} is not a non-zero power of two"
            )));
        }
        let shared_memory_bytes = block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| NnisError::invalid_input("bf16 shared-memory size overflows"))?;
        let code =
            compiler.compile_cubin(BF16_REDUCTION_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let bf16_sum = module.get_function("nnis_bf16_reduce_sum_f32")?;
        let bf16_max = module.get_function("nnis_bf16_reduce_max_f32")?;
        let sum = module.get_function("nnis_reduce_sum_f32")?;
        let max = module.get_function("nnis_reduce_max_f32")?;
        for kernel in [&bf16_sum, &bf16_max, &sum, &max] {
            let attributes = kernel.attributes()?;
            if block_size > attributes.max_threads_per_block {
                return Err(NnisError::invalid_input(format!(
                    "bf16 reduction block size {block_size} exceeds function limit {}",
                    attributes.max_threads_per_block
                )));
            }
            if shared_memory_bytes > attributes.max_dynamic_shared_memory_bytes {
                return Err(NnisError::invalid_input(format!(
                    "bf16 reduction requires {shared_memory_bytes} shared-memory bytes; \
                     function limit is {}",
                    attributes.max_dynamic_shared_memory_bytes
                )));
            }
        }
        Ok(Self {
            bf16_sum,
            bf16_max,
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
    ) -> Result<Bf16ReductionWorkspace> {
        if !Arc::ptr_eq(context, self.bf16_sum.context()) {
            return Err(NnisError::invalid_input(
                "bf16 reduction and workspace contexts do not match",
            ));
        }
        let scratch_elements = partial_count(max_elements, self.block_size)?;
        Ok(Bf16ReductionWorkspace {
            max_elements,
            block_size: self.block_size,
            scratch_a: TreeScratch::Plain(DeviceBuffer::new(context, scratch_elements)?),
            scratch_b: TreeScratch::Plain(DeviceBuffer::new(context, scratch_elements)?),
        })
    }

    /// Reduce an input and return its host scalar, waiting for completion.
    /// An empty input yields `0.0` for sum.
    pub fn sum(&self, stream: &Stream, input: &DeviceBuffer<u16>) -> Result<f32> {
        let workspace = self.workspace(input.ctx(), input.len())?;
        let output = DeviceBuffer::<f32>::new(input.ctx(), 1)?;
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe { self.enqueue_sum(stream, input, &output, &workspace) };
        Self::join(enqueue_result, stream.synchronize())?;
        Ok(output.to_vec(stream)?[0])
    }

    /// Return the maximum element as a host scalar, waiting for completion.
    /// An empty input yields `-infinity`, matching the kernel identity.
    pub fn max(&self, stream: &Stream, input: &DeviceBuffer<u16>) -> Result<f32> {
        if input.is_empty() {
            return Ok(f32::NEG_INFINITY);
        }
        let workspace = self.workspace(input.ctx(), input.len())?;
        let output = DeviceBuffer::<f32>::new(input.ctx(), 1)?;
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe { self.enqueue_max(stream, input, &output, &workspace) };
        Self::join(enqueue_result, stream.synchronize())?;
        Ok(output.to_vec(stream)?[0])
    }

    /// Enqueue every sum-reduction pass without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// The reduction, stream, input, output, and workspace must remain alive
    /// and otherwise untouched until the stream completes. The workspace may
    /// not be shared by overlapping operations, including on other streams.
    pub unsafe fn enqueue_sum(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
        workspace: &Bf16ReductionWorkspace,
    ) -> Result<()> {
        // SAFETY: documented lifetime obligations carry over unchanged.
        unsafe { self.enqueue_tree(&self.bf16_sum, &self.sum, stream, input, output, workspace) }
    }

    /// Enqueue every max-reduction pass without synchronizing the stream.
    ///
    /// For an empty input nothing is enqueued and `output[0]` stays untouched.
    ///
    /// # Safety
    ///
    /// Same obligations as [`Self::enqueue_sum`].
    pub unsafe fn enqueue_max(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
        workspace: &Bf16ReductionWorkspace,
    ) -> Result<()> {
        // SAFETY: documented lifetime obligations carry over unchanged.
        unsafe { self.enqueue_tree(&self.bf16_max, &self.max, stream, input, output, workspace) }
    }

    /// # Safety
    ///
    /// Pointers name live allocations owned by the caller per the public
    /// enqueue contracts; all passes are ordered on one stream.
    unsafe fn enqueue_tree(
        &self,
        first_pass: &Kernel,
        later_pass: &Kernel,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
        workspace: &Bf16ReductionWorkspace,
    ) -> Result<()> {
        if output.len() != 1 {
            return Err(NnisError::invalid_input(format!(
                "bf16 reduction output has {} elements; expected 1",
                output.len()
            )));
        }
        self.validate_execution(stream, input, output, workspace)?;
        if input.is_empty() {
            return Ok(());
        }

        let mut current_ptr = input.device_ptr();
        let mut current_elements = input.len();
        let mut first = true;
        let mut write_scratch_a = true;
        loop {
            let output_elements = partial_count(current_elements, self.block_size)?;
            let destination_ptr = if output_elements == 1 {
                output.device_ptr()
            } else if write_scratch_a {
                workspace.scratch_a.ptr()
            } else {
                workspace.scratch_b.ptr()
            };
            // SAFETY: pointers name distinct live allocations; passes are
            // ordered on one stream and lifetimes are the caller's burden.
            unsafe {
                self.enqueue_pass(
                    if first { first_pass } else { later_pass },
                    stream,
                    current_ptr,
                    current_elements,
                    destination_ptr,
                )?;
            }
            first = false;
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
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
        workspace: &Bf16ReductionWorkspace,
    ) -> Result<()> {
        if input.len() > workspace.max_elements {
            return Err(NnisError::invalid_input(format!(
                "bf16 reduction input has {} elements; workspace capacity is {}",
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
        let context = self.bf16_sum.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, input.ctx())
            || !Arc::ptr_eq(context, output.ctx())
            || !Arc::ptr_eq(context, workspace.context())
        {
            return Err(NnisError::invalid_input(
                "bf16 reduction stream, buffers, workspace, and kernel must share one context",
            ));
        }
        Ok(())
    }

    fn join(enqueue_result: Result<()>, synchronize_result: Result<()>) -> Result<()> {
        match enqueue_result {
            Ok(()) => synchronize_result,
            Err(error) => {
                let _ = synchronize_result;
                Err(error)
            }
        }
    }

    /// # Safety
    ///
    /// Both pointers must name distinct live device allocations; the input
    /// element width depends on whether this is the widening first pass or a
    /// later `f32` pass.
    unsafe fn enqueue_pass(
        &self,
        kernel: &Kernel,
        stream: &Stream,
        input_ptr: u64,
        elements: usize,
        output_ptr: u64,
    ) -> Result<()> {
        let output_elements = partial_count(elements, self.block_size)?;
        if output_elements == 0 {
            return Err(NnisError::invalid_input(
                "bf16 reduction pass requires a non-empty input",
            ));
        }
        let grid_size = u32::try_from(output_elements)
            .map_err(|_| NnisError::invalid_input("bf16 grid exceeds u32::MAX blocks"))?;
        let elements = u64::try_from(elements)
            .map_err(|_| NnisError::invalid_input("bf16 length exceeds u64::MAX"))?;
        let shared_memory_bytes = self
            .block_size
            .checked_mul(std::mem::size_of::<f32>() as u32)
            .ok_or_else(|| NnisError::invalid_input("bf16 shared-memory size overflows"))?;
        let config = LaunchConfig::new(Dim3::x(grid_size), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(shared_memory_bytes);
        let mut arguments = KernelArgs::with_capacity(3, 0);
        arguments.push(input_ptr).push(output_ptr).push(elements);
        let launch = KernelLaunch::new(kernel, stream, config);
        // SAFETY: both kernel pairs share this argument signature
        // (pointer, pointer, count); only the pointed-to element width
        // differs between the widening first pass and later passes.
        unsafe { launch.launch(&mut arguments) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    const TEST_SIZES: &[usize] = &[
        0, 1, 2, 3, 31, 32, 255, 256, 257, 511, 512, 513, 1_023, 1_024, 1_025, 131_071, 1_000_003,
    ];

    /// Deterministic values that stay exactly representable in bf16.
    fn host_values(size: usize) -> Vec<f32> {
        (0..size)
            .map(|index| {
                let numerator = (index * 37 % 1_009) as f32 - 504.0;
                numerator / 64.0
            })
            .collect()
    }

    fn to_bits(values: &[f32]) -> Vec<u16> {
        values
            .iter()
            .map(|&value| nnis_rt::f32_to_bf16_rne(value))
            .collect()
    }

    fn reference_tree_sum(bits: &[u16], block_size: usize) -> f32 {
        if bits.is_empty() {
            return 0.0;
        }
        let mut current: Vec<f32> = bits.iter().map(|&b| nnis_rt::bf16_bits_to_f32(b)).collect();
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

    #[test]
    fn bf16_sum_matches_ordered_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let reduction = Bf16Reduction::load(&context, &compiler).unwrap();
        let maximum = *TEST_SIZES.iter().max().unwrap();
        let _workspace = reduction.workspace(&context, maximum).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &size in TEST_SIZES {
            let bits =
                DeviceBuffer::from_host(&context, &stream, &to_bits(&host_values(size))).unwrap();
            let actual = reduction.sum(&stream, &bits).unwrap();

            // Bit-exact against the same-order f32 tree over widened inputs.
            let ordered = reference_tree_sum(
                &to_bits(&host_values(size)),
                reduction.block_size() as usize,
            );
            assert_eq!(
                actual.to_bits(),
                ordered.to_bits(),
                "ordered bf16 sum mismatch for {size} elements"
            );

            // High-precision bound on the accumulated rounding error.
            let exact: f64 = host_values(size)
                .iter()
                .map(|&value| f64::from(nnis_rt::bf16_bits_to_f32(nnis_rt::f32_to_bf16_rne(value))))
                .sum();
            let depth = (usize::BITS - (size.max(1) - 1).leading_zeros()) as f64 + 1.0;
            let gamma = depth * f32::EPSILON as f64 / (1.0 - depth * f32::EPSILON as f64);
            let magnitude: f64 = host_values(size)
                .iter()
                .take(size.max(1))
                .map(|&value| {
                    f64::from(nnis_rt::bf16_bits_to_f32(nnis_rt::f32_to_bf16_rne(value))).abs()
                })
                .sum();
            let error = (f64::from(actual) - exact).abs();
            assert!(
                size == 0 || error <= (gamma * magnitude).max(f64::from(f32::EPSILON)),
                "bf16 sum error for {size} elements is {error}"
            );
        }
    }

    #[test]
    fn bf16_max_matches_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let reduction = Bf16Reduction::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &size in TEST_SIZES {
            let host = host_values(size);
            let bits = DeviceBuffer::from_host(&context, &stream, &to_bits(&host)).unwrap();
            let actual = reduction.max(&stream, &bits).unwrap();
            if size == 0 {
                assert_eq!(actual.to_bits(), f32::NEG_INFINITY.to_bits());
                continue;
            }
            let expected = host
                .iter()
                .map(|&value| nnis_rt::bf16_bits_to_f32(nnis_rt::f32_to_bf16_rne(value)))
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "bf16 max mismatch for {size} elements"
            );
        }
    }
}
