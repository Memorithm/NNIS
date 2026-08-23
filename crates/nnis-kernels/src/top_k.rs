//! Deterministic top-k selection over `f32` scores.
//!
//! Selection runs `k` rounds. Each round performs a multi-pass tree argmax
//! over a private scratch copy of the input, writes the winning value/index
//! pair to the caller's outputs, and masks the winner to `-infinity` so the
//! next round cannot reselect it. Comparisons prefer the larger value and
//! break ties toward the lower index, matching the CPU oracle bit for bit.
//! Inputs containing `NaN` select in an order-dependent way and are outside
//! the contract.

use crate::reduction::partial_count;
use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const TOP_K_SOURCE: &str = r#"
extern "C" __global__ void nnis_topk_fill_f32(
    const float* input,
    float* scratch,
    unsigned long long elements
) {
    const unsigned long long i =
        ((unsigned long long)blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < elements) {
        scratch[i] = input[i];
    }
}

extern "C" __global__ void nnis_topk_mask_f32(
    float* scratch,
    unsigned long long slot,
    const unsigned int* index_buffer
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scratch[index_buffer[slot]] = __int_as_float(0xff800000);
    }
}

extern "C" __global__ void nnis_topk_argmax_first_f32(
    const float* input,
    float* out_value,
    unsigned int* out_index,
    unsigned long long elements
) {
    extern __shared__ unsigned char shared[];
    float* partial_value = (float*)shared;
    unsigned int* partial_index =
        (unsigned int*)(shared + (unsigned long long)blockDim.x * sizeof(float));
    const unsigned int lane = threadIdx.x;
    const unsigned long long first =
        ((unsigned long long)blockIdx.x * blockDim.x * 2) + lane;
    const unsigned long long second = first + blockDim.x;

    float value = __int_as_float(0xff800000);
    unsigned int index = 0xffffffffu;
    if (first < elements) {
        value = input[first];
        index = (unsigned int)first;
    }
    if (second < elements) {
        const float candidate = input[second];
        const unsigned int candidate_index = (unsigned int)second;
        if (candidate > value ||
            (candidate == value && candidate_index < index)) {
            value = candidate;
            index = candidate_index;
        }
    }
    partial_value[lane] = value;
    partial_index[lane] = index;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float candidate = partial_value[lane + stride];
            const unsigned int candidate_index = partial_index[lane + stride];
            if (candidate > partial_value[lane] ||
                (candidate == partial_value[lane] &&
                 candidate_index < partial_index[lane])) {
                partial_value[lane] = candidate;
                partial_index[lane] = candidate_index;
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        out_value[blockIdx.x] = partial_value[0];
        out_index[blockIdx.x] = partial_index[0];
    }
}

extern "C" __global__ void nnis_topk_argmax_pair_f32(
    const float* in_value,
    const unsigned int* in_index,
    float* out_value,
    unsigned int* out_index,
    unsigned long long elements
) {
    extern __shared__ unsigned char shared[];
    float* partial_value = (float*)shared;
    unsigned int* partial_index =
        (unsigned int*)(shared + (unsigned long long)blockDim.x * sizeof(float));
    const unsigned int lane = threadIdx.x;
    const unsigned long long first =
        ((unsigned long long)blockIdx.x * blockDim.x * 2) + lane;
    const unsigned long long second = first + blockDim.x;

    float value = __int_as_float(0xff800000);
    unsigned int index = 0xffffffffu;
    if (first < elements) {
        value = in_value[first];
        index = in_index[first];
    }
    if (second < elements) {
        const float candidate = in_value[second];
        const unsigned int candidate_index = in_index[second];
        if (candidate > value ||
            (candidate == value && candidate_index < index)) {
            value = candidate;
            index = candidate_index;
        }
    }
    partial_value[lane] = value;
    partial_index[lane] = index;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float candidate = partial_value[lane + stride];
            const unsigned int candidate_index = partial_index[lane + stride];
            if (candidate > partial_value[lane] ||
                (candidate == partial_value[lane] &&
                 candidate_index < partial_index[lane])) {
                partial_value[lane] = candidate;
                partial_index[lane] = candidate_index;
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        out_value[blockIdx.x] = partial_value[0];
        out_index[blockIdx.x] = partial_index[0];
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Reusable scratch storage for [`F32TopK`] selections.
#[derive(Debug)]
pub struct F32TopKWorkspace {
    max_elements: usize,
    block_size: u32,
    data: DeviceBuffer<f32>,
    values_a: DeviceBuffer<f32>,
    values_b: DeviceBuffer<f32>,
    indices_a: DeviceBuffer<u32>,
    indices_b: DeviceBuffer<u32>,
}

impl F32TopKWorkspace {
    pub fn max_elements(&self) -> usize {
        self.max_elements
    }

    fn context(&self) -> &Arc<Context> {
        self.data.ctx()
    }
}

/// Context-bound deterministic top-k selection over `f32` inputs.
///
/// Outputs are descending by value; equal values are emitted in ascending
/// index order.
#[derive(Debug)]
pub struct F32TopK {
    fill: Kernel,
    mask: Kernel,
    argmax_first: Kernel,
    argmax_pair: Kernel,
    block_size: u32,
}

impl F32TopK {
    /// Compile (or reuse cached CUBIN) and load the default top-k selector.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load the selector with an explicit power-of-two thread-block width.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(NnisError::invalid_input(format!(
                "top-k block size {block_size} is not a non-zero power of two"
            )));
        }
        let scan_shared_memory_bytes = block_size
            .checked_mul(std::mem::size_of::<f32>() as u32 * 2)
            .ok_or_else(|| NnisError::invalid_input("top-k shared-memory size overflows"))?;
        let code = compiler.compile_cubin(TOP_K_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let fill = module.get_function("nnis_topk_fill_f32")?;
        let mask = module.get_function("nnis_topk_mask_f32")?;
        let argmax_first = module.get_function("nnis_topk_argmax_first_f32")?;
        let argmax_pair = module.get_function("nnis_topk_argmax_pair_f32")?;
        for kernel in [&fill, &mask, &argmax_first, &argmax_pair] {
            let attributes = kernel.attributes()?;
            if block_size > attributes.max_threads_per_block {
                return Err(NnisError::invalid_input(format!(
                    "top-k block size {block_size} exceeds function limit {}",
                    attributes.max_threads_per_block
                )));
            }
        }
        for kernel in [&argmax_first, &argmax_pair] {
            let attributes = kernel.attributes()?;
            if scan_shared_memory_bytes > attributes.max_dynamic_shared_memory_bytes {
                return Err(NnisError::invalid_input(format!(
                    "top-k requires {scan_shared_memory_bytes} shared-memory bytes; \
                     function limit is {}",
                    attributes.max_dynamic_shared_memory_bytes
                )));
            }
        }
        Ok(Self {
            fill,
            mask,
            argmax_first,
            argmax_pair,
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
    ) -> Result<F32TopKWorkspace> {
        if !Arc::ptr_eq(context, self.fill.context()) {
            return Err(NnisError::invalid_input(
                "top-k and workspace contexts do not match",
            ));
        }
        let partial_elements = partial_count(max_elements, self.block_size)?;
        Ok(F32TopKWorkspace {
            max_elements,
            block_size: self.block_size,
            data: DeviceBuffer::new(context, max_elements)?,
            values_a: DeviceBuffer::new(context, partial_elements)?,
            values_b: DeviceBuffer::new(context, partial_elements)?,
            indices_a: DeviceBuffer::new(context, partial_elements)?,
            indices_b: DeviceBuffer::new(context, partial_elements)?,
        })
    }

    /// Select the top `k` elements, waiting for completion.
    ///
    /// `values` receives the `k` largest inputs in descending order and
    /// `indices` their positions, ties broken toward the lower index.
    pub fn top_k(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        indices: &DeviceBuffer<u32>,
        k: usize,
    ) -> Result<()> {
        let workspace = self.workspace(input.ctx(), input.len())?;
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result =
            unsafe { self.enqueue_top_k(stream, input, values, indices, k, &workspace) };
        Self::join(enqueue_result, stream.synchronize())?;
        Ok(())
    }

    /// Enqueue every selection pass without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// The selector, stream, input, outputs, and workspace must remain alive
    /// and otherwise untouched until the stream completes. The workspace may
    /// not be shared by overlapping operations, including on other streams.
    pub unsafe fn enqueue_top_k(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        indices: &DeviceBuffer<u32>,
        k: usize,
        workspace: &F32TopKWorkspace,
    ) -> Result<()> {
        self.validate_execution(stream, input, values, indices, k, workspace)?;

        // SAFETY: pointers name live allocations owned by the caller per the
        // public enqueue contract; every launch is ordered on one stream.
        unsafe {
            self.enqueue_fill(stream, input, workspace)?;
            for round in 0..k {
                let winner_value_ptr =
                    values.device_ptr() + round as u64 * std::mem::size_of::<f32>() as u64;
                let winner_index_ptr =
                    indices.device_ptr() + round as u64 * std::mem::size_of::<u32>() as u64;
                self.enqueue_round(
                    stream,
                    input.len(),
                    workspace,
                    winner_value_ptr,
                    winner_index_ptr,
                )?;
                if round + 1 < k {
                    self.enqueue_mask(stream, indices.device_ptr(), round as u64, workspace)?;
                }
            }
        }
        Ok(())
    }

    /// # Safety
    ///
    /// Pointers name live allocations owned by the caller per the public
    /// enqueue contracts; all passes are ordered on one stream.
    unsafe fn enqueue_round(
        &self,
        stream: &Stream,
        input_elements: usize,
        workspace: &F32TopKWorkspace,
        winner_value_ptr: u64,
        winner_index_ptr: u64,
    ) -> Result<()> {
        let mut current_value_ptr = workspace.data.device_ptr();
        let mut current_index_ptr: Option<u64> = None;
        let mut current_elements = input_elements;
        let mut write_scratch_a = true;
        loop {
            let output_elements = partial_count(current_elements, self.block_size)?;
            let (destination_value_ptr, destination_index_ptr) = if output_elements == 1 {
                (winner_value_ptr, winner_index_ptr)
            } else if write_scratch_a {
                (
                    workspace.values_a.device_ptr(),
                    workspace.indices_a.device_ptr(),
                )
            } else {
                (
                    workspace.values_b.device_ptr(),
                    workspace.indices_b.device_ptr(),
                )
            };
            // SAFETY: the plain-scan pass reads raw `f32` scores while
            // pair-scan passes read `(value, index)` pairs; both write
            // pairs. All buffers are distinct live allocations and every
            // launch is ordered on one stream.
            unsafe {
                match current_index_ptr {
                    None => self.enqueue_scan_plain(
                        stream,
                        current_value_ptr,
                        destination_value_ptr,
                        destination_index_ptr,
                        current_elements,
                    )?,
                    Some(index_pointer) => self.enqueue_scan_pair(
                        stream,
                        current_value_ptr,
                        index_pointer,
                        destination_value_ptr,
                        destination_index_ptr,
                        current_elements,
                    )?,
                }
            }
            if output_elements == 1 {
                break;
            }
            current_value_ptr = destination_value_ptr;
            current_index_ptr = Some(destination_index_ptr);
            current_elements = output_elements;
            write_scratch_a = !write_scratch_a;
        }
        Ok(())
    }

    fn validate_execution(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        indices: &DeviceBuffer<u32>,
        k: usize,
        workspace: &F32TopKWorkspace,
    ) -> Result<()> {
        if k == 0 {
            return Err(NnisError::invalid_input(
                "top-k requires at least one element",
            ));
        }
        if input.len() < k {
            return Err(NnisError::invalid_input(format!(
                "top-k input has {} elements; selecting {k} exceeds the input length",
                input.len()
            )));
        }
        if u64::try_from(input.len()).map(|length| length > u32::MAX as u64) == Ok(true) {
            return Err(NnisError::invalid_input(
                "top-k input length exceeds u32::MAX",
            ));
        }
        if values.len() != k || indices.len() != k {
            return Err(NnisError::invalid_input(format!(
                "top-k outputs must hold exactly {k} elements; got {} values and {} indices",
                values.len(),
                indices.len()
            )));
        }
        if input.len() > workspace.max_elements {
            return Err(NnisError::invalid_input(format!(
                "top-k input has {} elements; workspace capacity is {}",
                input.len(),
                workspace.max_elements
            )));
        }
        if workspace.block_size != self.block_size {
            return Err(NnisError::invalid_input(format!(
                "workspace block size {} does not match top-k block size {}",
                workspace.block_size, self.block_size
            )));
        }
        let context = self.fill.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, input.ctx())
            || !Arc::ptr_eq(context, values.ctx())
            || !Arc::ptr_eq(context, indices.ctx())
            || !Arc::ptr_eq(context, workspace.context())
        {
            return Err(NnisError::invalid_input(
                "top-k stream, buffers, workspace, and kernel must share one context",
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
    /// Both pointers must name live device allocations owned by the caller.
    unsafe fn enqueue_fill(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        workspace: &F32TopKWorkspace,
    ) -> Result<()> {
        let elements = u64::try_from(input.len())
            .map_err(|_| NnisError::invalid_input("top-k length exceeds u64::MAX"))?;
        let grid_size = u32::try_from(input.len().div_ceil(self.block_size as usize))
            .map_err(|_| NnisError::invalid_input("top-k grid exceeds u32::MAX blocks"))?;
        let config = LaunchConfig::new(Dim3::x(grid_size), Dim3::x(self.block_size));
        let mut arguments = KernelArgs::with_capacity(3, 0);
        arguments
            .push(input.device_ptr())
            .push(workspace.data.device_ptr())
            .push(elements);
        let launch = KernelLaunch::new(&self.fill, stream, config);
        // SAFETY: argument order matches the kernel signature exactly.
        unsafe { launch.launch(&mut arguments) }
    }

    /// # Safety
    ///
    /// The index pointer names a live device allocation owned by the caller.
    unsafe fn enqueue_mask(
        &self,
        stream: &Stream,
        index_buffer_ptr: u64,
        slot: u64,
        workspace: &F32TopKWorkspace,
    ) -> Result<()> {
        let config = LaunchConfig::new(Dim3::x(1), Dim3::x(1));
        let mut arguments = KernelArgs::with_capacity(3, 0);
        arguments
            .push(workspace.data.device_ptr())
            .push(slot)
            .push(index_buffer_ptr);
        let launch = KernelLaunch::new(&self.mask, stream, config);
        // SAFETY: argument order matches the kernel signature exactly.
        unsafe { launch.launch(&mut arguments) }
    }

    /// # Safety
    ///
    /// All three pointers must name live device allocations owned by the
    /// caller.
    unsafe fn enqueue_scan_plain(
        &self,
        stream: &Stream,
        input_value_ptr: u64,
        output_value_ptr: u64,
        output_index_ptr: u64,
        elements: usize,
    ) -> Result<()> {
        let config = self.scan_config(elements)?;
        let elements = u64::try_from(elements)
            .map_err(|_| NnisError::invalid_input("top-k length exceeds u64::MAX"))?;
        let mut arguments = KernelArgs::with_capacity(4, 0);
        arguments
            .push(input_value_ptr)
            .push(output_value_ptr)
            .push(output_index_ptr)
            .push(elements);
        let launch = KernelLaunch::new(&self.argmax_first, stream, config);
        // SAFETY: argument order matches the kernel signature exactly.
        unsafe { launch.launch(&mut arguments) }
    }

    /// # Safety
    ///
    /// All four pointers must name live device allocations owned by the
    /// caller.
    unsafe fn enqueue_scan_pair(
        &self,
        stream: &Stream,
        input_value_ptr: u64,
        input_index_ptr: u64,
        output_value_ptr: u64,
        output_index_ptr: u64,
        elements: usize,
    ) -> Result<()> {
        let config = self.scan_config(elements)?;
        let elements = u64::try_from(elements)
            .map_err(|_| NnisError::invalid_input("top-k length exceeds u64::MAX"))?;
        let mut arguments = KernelArgs::with_capacity(5, 0);
        arguments
            .push(input_value_ptr)
            .push(input_index_ptr)
            .push(output_value_ptr)
            .push(output_index_ptr)
            .push(elements);
        let launch = KernelLaunch::new(&self.argmax_pair, stream, config);
        // SAFETY: argument order matches the kernel signature exactly.
        unsafe { launch.launch(&mut arguments) }
    }

    fn scan_config(&self, elements: usize) -> Result<LaunchConfig> {
        let output_elements = partial_count(elements, self.block_size)?;
        let grid_size = u32::try_from(output_elements)
            .map_err(|_| NnisError::invalid_input("top-k grid exceeds u32::MAX blocks"))?;
        let shared_memory_bytes =
            self.block_size * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>()) as u32;
        Ok(
            LaunchConfig::new(Dim3::x(grid_size), Dim3::x(self.block_size))
                .with_dynamic_shared_memory(shared_memory_bytes),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    const TEST_PAIRS: &[(usize, usize)] = &[
        (1, 1),
        (2, 1),
        (2, 2),
        (3, 3),
        (31, 5),
        (256, 9),
        (257, 40),
        (1_024, 17),
        (131_071, 64),
    ];

    fn host_values(elements: usize) -> Vec<f32> {
        (0..elements)
            .map(|index| ((index * 37 % 1_009) as f32 - 504.0) / 64.0)
            .collect()
    }

    /// Replicates the device comparator exactly: larger value wins, ties
    /// break toward the lower index, winners are masked to -infinity.
    fn reference_top_k(data: &[f32], k: usize) -> (Vec<f32>, Vec<u32>) {
        let mut scratch = data.to_vec();
        let mut values = Vec::with_capacity(k);
        let mut indices = Vec::with_capacity(k);
        for _ in 0..k {
            let mut best = 0usize;
            for candidate in 1..scratch.len() {
                let value = scratch[candidate];
                let incumbent = scratch[best];
                if value > incumbent || (value == incumbent && candidate < best) {
                    best = candidate;
                }
            }
            values.push(scratch[best]);
            indices.push(best as u32);
            scratch[best] = f32::NEG_INFINITY;
        }
        (values, indices)
    }

    #[test]
    fn top_k_matches_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let compiler = JitCompiler::new();
        let top_k = F32TopK::load(&context, &compiler).unwrap();

        for &(elements, k) in TEST_PAIRS {
            let host = host_values(elements);
            let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
            let values = DeviceBuffer::<f32>::new(&context, k).unwrap();
            let indices = DeviceBuffer::<u32>::new(&context, k).unwrap();
            let workspace = top_k.workspace(&context, elements).unwrap();
            // SAFETY: every borrow stays live through the synchronize below.
            unsafe {
                top_k
                    .enqueue_top_k(&stream, &input, &values, &indices, k, &workspace)
                    .unwrap();
            }
            stream.synchronize().unwrap();

            let (expected_values, expected_indices) = reference_top_k(&host, k);
            let actual_values = values.to_vec(&stream).unwrap();
            let actual_indices = indices.to_vec(&stream).unwrap();
            assert_eq!(
                actual_indices, expected_indices,
                "indices mismatch at {elements} elements, k={k}"
            );
            for (actual, expected) in actual_values.iter().zip(expected_values.iter()) {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "value bits mismatch at {elements} elements, k={k}"
                );
            }
        }
    }

    #[test]
    fn top_k_breaks_ties_toward_lower_indices_on_gpu() {
        let Some(context) = gpu_context() else {
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let compiler = JitCompiler::new();
        let top_k = F32TopK::load(&context, &compiler).unwrap();

        let elements = 512;
        let k = 16;
        let host = vec![7.0_f32; elements];
        let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
        let values = DeviceBuffer::<f32>::new(&context, k).unwrap();
        let indices = DeviceBuffer::<u32>::new(&context, k).unwrap();
        top_k.top_k(&stream, &input, &values, &indices, k).unwrap();

        let actual_indices = indices.to_vec(&stream).unwrap();
        let expected_indices: Vec<u32> = (0..k as u32).collect();
        assert_eq!(actual_indices, expected_indices);
        for value in values.to_vec(&stream).unwrap() {
            assert_eq!(value.to_bits(), 7.0_f32.to_bits());
        }
    }

    #[test]
    fn top_k_rejects_invalid_executions_on_gpu() {
        let Some(context) = gpu_context() else {
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let compiler = JitCompiler::new();
        let top_k = F32TopK::load(&context, &compiler).unwrap();

        let host = vec![1.0_f32; 8];
        let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
        let values = DeviceBuffer::<f32>::new(&context, 4).unwrap();
        let indices = DeviceBuffer::<u32>::new(&context, 4).unwrap();
        let workspace = top_k.workspace(&context, 8).unwrap();

        macro_rules! expect_rejected {
            ($call:expr) => {
                // SAFETY: borrows outlive the rejected enqueue; nothing runs.
                unsafe {
                    assert!($call.is_err());
                }
            };
        }
        expect_rejected!(top_k.enqueue_top_k(&stream, &input, &values, &indices, 0, &workspace));
        expect_rejected!(top_k.enqueue_top_k(&stream, &input, &values, &indices, 9, &workspace));
        let short_values = DeviceBuffer::<f32>::new(&context, 3).unwrap();
        expect_rejected!(top_k.enqueue_top_k(
            &stream,
            &input,
            &short_values,
            &indices,
            4,
            &workspace
        ));
        let small_workspace = top_k.workspace(&context, 4).unwrap();
        expect_rejected!(top_k.enqueue_top_k(
            &stream,
            &input,
            &values,
            &indices,
            4,
            &small_workspace
        ));
        let narrow = F32TopK::load_with_block_size(&context, &compiler, 128).unwrap();
        expect_rejected!(narrow.enqueue_top_k(&stream, &input, &values, &indices, 4, &workspace));
    }
}
