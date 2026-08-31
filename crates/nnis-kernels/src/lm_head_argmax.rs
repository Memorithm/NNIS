//! Fused batch-one f32 LM-head projection and greedy argmax candidate.
//!
//! This kernel family targets `[1,K] × [K,V]` decoder LM-head weights in the
//! exact physical orientation already used by NNIS. Each vocabulary logit is
//! accumulated by one CUDA thread with explicit `fmaf` in strictly increasing
//! K order, matching [`crate::F32Gemv::project_kn`] bit-for-bit for finite
//! inputs. The full vocabulary row is never materialized: each block emits one
//! `(value, token)` candidate and a second deterministic reduction selects the
//! global winner, breaking ties toward the lower token id.
//!
//! As with [`crate::F32TopK`], NaN-containing scores are outside the contract.
//! This is a candidate primitive only; model-runtime promotion requires a
//! separate end-to-end correctness and throughput gate.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const LM_HEAD_ARGMAX_SOURCE: &str = r#"
extern "C" __global__ void nnis_lm_head_argmax_candidates_f32(
    const float* input,
    const float* weight,
    float* candidate_values,
    unsigned int* candidate_indices,
    unsigned long long k,
    unsigned long long vocab
) {
    extern __shared__ unsigned char shared[];
    float* values = (float*)shared;
    unsigned int* indices =
        (unsigned int*)(shared + (unsigned long long)blockDim.x * sizeof(float));

    const unsigned int lane = threadIdx.x;
    const unsigned long long token =
        (unsigned long long)blockIdx.x * blockDim.x + lane;

    float value = -__int_as_float(0x7f800000);
    unsigned int index = 0xffffffffu;
    if (token < vocab) {
        value = 0.0f;
        for (unsigned long long row = 0; row < k; ++row) {
            value = fmaf(input[row], weight[row * vocab + token], value);
        }
        index = (unsigned int)token;
    }

    values[lane] = value;
    indices[lane] = index;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float rhs_value = values[lane + stride];
            const unsigned int rhs_index = indices[lane + stride];
            const float lhs_value = values[lane];
            const unsigned int lhs_index = indices[lane];
            if (rhs_index != 0xffffffffu &&
                (rhs_value > lhs_value ||
                 (rhs_value == lhs_value && rhs_index < lhs_index))) {
                values[lane] = rhs_value;
                indices[lane] = rhs_index;
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        candidate_values[blockIdx.x] = values[0];
        candidate_indices[blockIdx.x] = indices[0];
    }
}

extern "C" __global__ void nnis_lm_head_argmax_reduce_f32(
    const float* candidate_values,
    const unsigned int* candidate_indices,
    float* output_value,
    unsigned int* output_index,
    unsigned long long candidates
) {
    extern __shared__ unsigned char shared[];
    float* values = (float*)shared;
    unsigned int* indices =
        (unsigned int*)(shared + (unsigned long long)blockDim.x * sizeof(float));

    const unsigned int lane = threadIdx.x;
    float best_value = -__int_as_float(0x7f800000);
    unsigned int best_index = 0xffffffffu;

    for (unsigned long long i = lane; i < candidates; i += blockDim.x) {
        const float value = candidate_values[i];
        const unsigned int index = candidate_indices[i];
        if (index != 0xffffffffu &&
            (value > best_value ||
             (value == best_value && index < best_index))) {
            best_value = value;
            best_index = index;
        }
    }

    values[lane] = best_value;
    indices[lane] = best_index;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const float rhs_value = values[lane + stride];
            const unsigned int rhs_index = indices[lane + stride];
            const float lhs_value = values[lane];
            const unsigned int lhs_index = indices[lane];
            if (rhs_index != 0xffffffffu &&
                (rhs_value > lhs_value ||
                 (rhs_value == lhs_value && rhs_index < lhs_index))) {
                values[lane] = rhs_value;
                indices[lane] = rhs_index;
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        output_value[0] = values[0];
        output_index[0] = indices[0];
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 64;

#[derive(Debug)]
pub struct F32LmHeadArgmaxWorkspace {
    vocab: usize,
    block_size: u32,
    candidate_values: DeviceBuffer<f32>,
    candidate_indices: DeviceBuffer<u32>,
}

impl F32LmHeadArgmaxWorkspace {
    #[must_use]
    pub const fn vocab(&self) -> usize {
        self.vocab
    }

    #[must_use]
    pub fn candidate_count(&self) -> usize {
        self.candidate_values.len()
    }
}

/// Context-bound fused f32 LM-head projection + greedy top-1 candidate.
#[derive(Debug)]
pub struct F32LmHeadArgmax {
    candidates: Kernel,
    reduce: Kernel,
    block_size: u32,
}

impl F32LmHeadArgmax {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(NnisError::invalid_input(format!(
                "LM-head argmax block size {block_size} is not a non-zero power of two"
            )));
        }
        let shared_memory_bytes = block_size
            .checked_mul((std::mem::size_of::<f32>() + std::mem::size_of::<u32>()) as u32)
            .ok_or_else(|| {
                NnisError::invalid_input("LM-head argmax shared-memory size overflows")
            })?;
        let code =
            compiler.compile_cubin(LM_HEAD_ARGMAX_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let candidates = module.get_function("nnis_lm_head_argmax_candidates_f32")?;
        let reduce = module.get_function("nnis_lm_head_argmax_reduce_f32")?;
        for kernel in [&candidates, &reduce] {
            let attributes = kernel.attributes()?;
            if block_size > attributes.max_threads_per_block {
                return Err(NnisError::invalid_input(format!(
                    "LM-head argmax block size {block_size} exceeds function limit {}",
                    attributes.max_threads_per_block
                )));
            }
            if shared_memory_bytes as usize > attributes.max_dynamic_shared_memory_bytes as usize {
                return Err(NnisError::invalid_input(format!(
                    "LM-head argmax requires {shared_memory_bytes} shared-memory bytes; function limit is {}",
                    attributes.max_dynamic_shared_memory_bytes
                )));
            }
        }
        Ok(Self {
            candidates,
            reduce,
            block_size,
        })
    }

    #[must_use]
    pub const fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn workspace(
        &self,
        context: &Arc<Context>,
        vocab: usize,
    ) -> Result<F32LmHeadArgmaxWorkspace> {
        if vocab == 0 {
            return Err(NnisError::invalid_input(
                "LM-head argmax vocabulary must be non-zero",
            ));
        }
        if vocab > u32::MAX as usize {
            return Err(NnisError::invalid_input(
                "LM-head argmax vocabulary exceeds u32::MAX",
            ));
        }
        if !Arc::ptr_eq(context, self.candidates.context()) {
            return Err(NnisError::invalid_input(
                "LM-head argmax and workspace contexts do not match",
            ));
        }
        let width = self.block_size as usize;
        let candidate_count = vocab
            .checked_add(width - 1)
            .ok_or_else(|| NnisError::invalid_input("LM-head argmax candidate count overflows"))?
            / width;
        Ok(F32LmHeadArgmaxWorkspace {
            vocab,
            block_size: self.block_size,
            candidate_values: DeviceBuffer::new(context, candidate_count)?,
            candidate_indices: DeviceBuffer::new(context, candidate_count)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn argmax_kn(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        output_value: &DeviceBuffer<f32>,
        output_index: &DeviceBuffer<u32>,
        k: usize,
        vocab: usize,
        workspace: &F32LmHeadArgmaxWorkspace,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe {
            self.enqueue_argmax_kn(
                stream,
                input,
                weight,
                output_value,
                output_index,
                k,
                vocab,
                workspace,
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

    /// Enqueue fused `[1,K] × [K,V] -> greedy top-1` without materializing V logits.
    ///
    /// # Safety
    ///
    /// The kernel, stream, all buffers and workspace must remain alive and
    /// otherwise untouched until the stream completes. The workspace must not
    /// be used by overlapping operations.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_argmax_kn(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        output_value: &DeviceBuffer<f32>,
        output_index: &DeviceBuffer<u32>,
        k: usize,
        vocab: usize,
        workspace: &F32LmHeadArgmaxWorkspace,
    ) -> Result<()> {
        self.validate_execution(
            stream,
            input,
            weight,
            output_value,
            output_index,
            k,
            vocab,
            workspace,
        )?;

        let k_arg = u64::try_from(k)
            .map_err(|_| NnisError::invalid_input("LM-head argmax K exceeds u64::MAX"))?;
        let vocab_arg = u64::try_from(vocab)
            .map_err(|_| NnisError::invalid_input("LM-head argmax vocab exceeds u64::MAX"))?;
        let candidate_count = workspace.candidate_count();
        let candidate_count_u32 = u32::try_from(candidate_count).map_err(|_| {
            NnisError::invalid_input("LM-head argmax candidate grid exceeds u32::MAX")
        })?;
        let candidate_count_arg = u64::try_from(candidate_count).map_err(|_| {
            NnisError::invalid_input("LM-head argmax candidate count exceeds u64::MAX")
        })?;
        let shared_memory_bytes = self
            .block_size
            .checked_mul((std::mem::size_of::<f32>() + std::mem::size_of::<u32>()) as u32)
            .ok_or_else(|| {
                NnisError::invalid_input("LM-head argmax shared-memory size overflows")
            })?;

        let candidate_config =
            LaunchConfig::new(Dim3::x(candidate_count_u32), Dim3::x(self.block_size))
                .with_dynamic_shared_memory(shared_memory_bytes);
        let mut candidate_args = KernelArgs::with_capacity(6, 4);
        candidate_args
            .push_buffer(input)
            .push_buffer(weight)
            .push_buffer(&workspace.candidate_values)
            .push_buffer(&workspace.candidate_indices)
            .push(k_arg)
            .push(vocab_arg);
        let candidate_launch = KernelLaunch::new(&self.candidates, stream, candidate_config);
        // SAFETY: argument order/widths match `nnis_lm_head_argmax_candidates_f32`.
        unsafe { candidate_launch.launch(&mut candidate_args)? };

        let reduce_config = LaunchConfig::new(Dim3::x(1), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(shared_memory_bytes);
        let mut reduce_args = KernelArgs::with_capacity(5, 4);
        reduce_args
            .push_buffer(&workspace.candidate_values)
            .push_buffer(&workspace.candidate_indices)
            .push_buffer(output_value)
            .push_buffer(output_index)
            .push(candidate_count_arg);
        let reduce_launch = KernelLaunch::new(&self.reduce, stream, reduce_config);
        // SAFETY: argument order/widths match `nnis_lm_head_argmax_reduce_f32`.
        unsafe { reduce_launch.launch(&mut reduce_args) }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_execution(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        output_value: &DeviceBuffer<f32>,
        output_index: &DeviceBuffer<u32>,
        k: usize,
        vocab: usize,
        workspace: &F32LmHeadArgmaxWorkspace,
    ) -> Result<()> {
        if vocab == 0 {
            return Err(NnisError::invalid_input(
                "LM-head argmax vocabulary must be non-zero",
            ));
        }
        if vocab > u32::MAX as usize {
            return Err(NnisError::invalid_input(
                "LM-head argmax vocabulary exceeds u32::MAX",
            ));
        }
        let expected_weight = k
            .checked_mul(vocab)
            .ok_or_else(|| NnisError::invalid_input("LM-head argmax shape overflows usize"))?;
        if input.len() != k {
            return Err(NnisError::invalid_input(format!(
                "LM-head argmax input has {} elements; shape requires {k}",
                input.len()
            )));
        }
        if weight.len() != expected_weight {
            return Err(NnisError::invalid_input(format!(
                "LM-head argmax weight has {} elements; shape ({k}, {vocab}) requires {expected_weight}",
                weight.len()
            )));
        }
        if output_value.len() != 1 || output_index.len() != 1 {
            return Err(NnisError::invalid_input(format!(
                "LM-head argmax outputs must each hold one element; got {} values and {} indices",
                output_value.len(),
                output_index.len()
            )));
        }
        if workspace.vocab != vocab || workspace.block_size != self.block_size {
            return Err(NnisError::invalid_input(
                "LM-head argmax workspace shape or block size does not match the operation",
            ));
        }
        let context = self.candidates.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, input.ctx())
            || !Arc::ptr_eq(context, weight.ctx())
            || !Arc::ptr_eq(context, output_value.ctx())
            || !Arc::ptr_eq(context, output_index.ctx())
            || !Arc::ptr_eq(context, workspace.candidate_values.ctx())
            || !Arc::ptr_eq(context, workspace.candidate_indices.ctx())
        {
            return Err(NnisError::invalid_input(
                "LM-head argmax stream, buffers, workspace and kernel must share one context",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    fn cpu_argmax_kn(input: &[f32], weight: &[f32], k: usize, vocab: usize) -> (f32, u32) {
        let mut best_value = f32::NEG_INFINITY;
        let mut best_index = u32::MAX;
        for token in 0..vocab {
            let mut value = 0.0_f32;
            for row in 0..k {
                value = input[row].mul_add(weight[row * vocab + token], value);
            }
            if value > best_value || (value == best_value && (token as u32) < best_index) {
                best_value = value;
                best_index = token as u32;
            }
        }
        (best_value, best_index)
    }

    #[test]
    fn rejects_non_power_of_two_block_size_before_cuda_work() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(F32LmHeadArgmax::load_with_block_size(&context, &compiler, 96).is_err());
    }

    #[test]
    fn fused_winner_matches_ordered_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let compiler = JitCompiler::new();
        let kernel = F32LmHeadArgmax::load_with_block_size(&context, &compiler, 64).unwrap();
        let k = 7usize;
        let vocab = 131usize;
        let input_host: Vec<f32> = (0..k)
            .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.125)
            .collect();
        let weight_host: Vec<f32> = (0..k * vocab)
            .map(|i| ((i * 19 % 29) as f32 - 14.0) * 0.03125)
            .collect();
        let expected = cpu_argmax_kn(&input_host, &weight_host, k, vocab);
        let input = DeviceBuffer::from_host(&context, &stream, &input_host).unwrap();
        let weight = DeviceBuffer::from_host(&context, &stream, &weight_host).unwrap();
        let output_value = DeviceBuffer::<f32>::new(&context, 1).unwrap();
        let output_index = DeviceBuffer::<u32>::new(&context, 1).unwrap();
        let workspace = kernel.workspace(&context, vocab).unwrap();
        kernel
            .argmax_kn(
                &stream,
                &input,
                &weight,
                &output_value,
                &output_index,
                k,
                vocab,
                &workspace,
            )
            .unwrap();
        let actual_value = output_value.to_vec(&stream).unwrap()[0];
        let actual_index = output_index.to_vec(&stream).unwrap()[0];
        assert_eq!(actual_index, expected.1);
        assert_eq!(actual_value.to_bits(), expected.0.to_bits());
    }
}
