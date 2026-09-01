//! Candidate-only F16 execution primitives for NNML5 reference alignment.
//!
//! TensorRT Edge-LLM v0.10.0 executes the qualified SmolLM2 reference with
//! F16 runtime bindings.  This module provides the narrow set of native CUDA
//! operations needed to build an explicit NNIS F16 qualification path without
//! changing model format v1, the historical F32 runtime, or any default plan.
//!
//! Numeric policy:
//! - persisted NNIS model values remain the model-format-v1 F32 graph;
//! - weight conversion uses CUDA round-to-nearest-even F32 -> IEEE binary16;
//! - decoder activations and KV storage are binary16;
//! - reductions / dot-product accumulators are F32 where the kernel requires
//!   numerical range;
//! - outputs are narrowed at the same high-level tensor boundaries that the
//!   Edge-LLM Llama graph exposes as F16;
//! - LM-head results are rounded to F16, then widened to F32 logits.
//!
//! These kernels are correctness infrastructure, not a performance claim.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, KvCache, NnisError, Result, Stream};
use std::sync::Arc;

const F16_REFERENCE_SOURCE: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void nnis_f16_narrow_from_f32(
    const float* input,
    __half* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __float2half_rn(input[index]);
    }
}

extern "C" __global__ void nnis_f16_widen_to_f32(
    const __half* input,
    float* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __half2float(input[index]);
    }
}

extern "C" __global__ void nnis_gather_f16(
    const __half* table,
    const unsigned int* indices,
    __half* output,
    unsigned long long rows,
    unsigned long long cols,
    unsigned long long index_count
) {
    const unsigned long long linear =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long elements = index_count * cols;
    if (linear >= elements) {
        return;
    }
    const unsigned long long index_slot = linear / cols;
    const unsigned long long col = linear % cols;
    const unsigned long long row = (unsigned long long)indices[index_slot];
    if (row < rows) {
        output[linear] = table[row * cols + col];
    }
}

// Edge-LLM's ordinary Llama RMSNorm computes the variance in F32, narrows the
// normalized value to F16, then applies the F16 gamma. Preserve that visible
// tensor boundary instead of fusing both multiplications into one F32 result.
extern "C" __global__ void nnis_weighted_rmsnorm_f16_f32acc(
    const __half* input,
    const __half* weight,
    __half* output,
    unsigned long long cols,
    float inv_cols,
    float epsilon
) {
    extern __shared__ float partial[];
    const unsigned int lane = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const __half* source = input + row * cols;
    __half* destination = output + row * cols;

    float sumsq = 0.0f;
    for (unsigned long long col = lane; col < cols; col += blockDim.x) {
        const float value = __half2float(source[col]);
        sumsq = fmaf(value, value, sumsq);
    }
    partial[lane] = sumsq;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }

    const float scale = rsqrtf(partial[0] * inv_cols + epsilon);
    for (unsigned long long col = lane; col < cols; col += blockDim.x) {
        const __half normalized =
            __float2half_rn(__half2float(source[col]) * scale);
        destination[col] = __float2half_rn(
            __half2float(normalized) * __half2float(weight[col]));
    }
}

// Internal NNIS projection orientation is [K, N]. One block owns one output
// column. Multiplication and accumulation are F32; the tensor output boundary
// is rounded once to F16.
extern "C" __global__ void nnis_project_kn_f16_f32acc(
    const __half* input,
    const __half* weight,
    __half* output,
    unsigned long long k,
    unsigned long long n
) {
    const unsigned long long col = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (col >= n) {
        return;
    }
    extern __shared__ float partial[];
    float sum = 0.0f;
    for (unsigned long long row = lane; row < k; row += blockDim.x) {
        sum = fmaf(
            __half2float(input[row]),
            __half2float(weight[row * n + col]),
            sum);
    }
    partial[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        output[col] = __float2half_rn(partial[0]);
    }
}

// Llama-style rotate-half RoPE. Cos/sin remain F32 as in the Edge-LLM graph;
// projected Q/K are F16 and the rotated tensor boundary remains F16.
extern "C" __global__ void nnis_rope_rotate_half_position_f16(
    const __half* input,
    const float* cos_cache,
    const float* sin_cache,
    __half* output,
    unsigned long long heads,
    unsigned long long head_dim,
    unsigned long long position,
    unsigned long long max_positions
) {
    const unsigned long long half = head_dim / 2;
    const unsigned long long pair_index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long pairs = heads * half;
    if (pair_index >= pairs || position >= max_positions) {
        return;
    }

    const unsigned long long head = pair_index / half;
    const unsigned long long pair = pair_index % half;
    const unsigned long long row = head * head_dim;
    const unsigned long long cache_index = position * half + pair;
    const float left = __half2float(input[row + pair]);
    const float right = __half2float(input[row + half + pair]);
    const float cosine = cos_cache[cache_index];
    const float sine = sin_cache[cache_index];
    output[row + pair] = __float2half_rn(left * cosine - right * sine);
    output[row + half + pair] = __float2half_rn(right * cosine + left * sine);
}

// One block owns one query head and one lane owns one value/output dimension.
// Lane zero preserves a serial score + online-softmax chain in F32 while each
// lane keeps its output accumulator in F32 until the final F16 tensor boundary.
extern "C" __global__ void nnis_cached_attention_decode_f16_f32acc(
    const __half* query,
    const __half* keys,
    const __half* values,
    __half* output,
    unsigned long long layer,
    unsigned long long query_heads,
    unsigned long long kv_heads,
    unsigned long long capacity,
    unsigned long long head_dim,
    unsigned long long kv_rows,
    float scale
) {
    const unsigned long long query_head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (query_head >= query_heads || lane >= head_dim) {
        return;
    }

    const unsigned long long group_size = query_heads / kv_heads;
    const unsigned long long kv_head = query_head / group_size;
    const __half* q = query + query_head * head_dim;
    const unsigned long long cache_base =
        ((layer * kv_heads + kv_head) * capacity) * head_dim;
    const __half* head_keys = keys + cache_base;
    const __half* head_values = values + cache_base;
    __half* destination = output + query_head * head_dim;

    __shared__ float old_weight_shared;
    __shared__ float new_weight_shared;
    __shared__ float inverse_sum_shared;

    float accumulator = 0.0f;
    float running_max = -3.402823466e+38F;
    float running_sum = 0.0f;

    for (unsigned long long pos = 0; pos < kv_rows; ++pos) {
        const __half* value = head_values + pos * head_dim;
        if (lane == 0) {
            const __half* key = head_keys + pos * head_dim;
            float score = 0.0f;
            for (unsigned long long dim = 0; dim < head_dim; ++dim) {
                score = fmaf(
                    __half2float(q[dim]),
                    __half2float(key[dim]),
                    score);
            }
            score *= scale;
            const float next_max = fmaxf(running_max, score);
            const float old_weight = running_sum == 0.0f
                ? 0.0f
                : expf(running_max - next_max);
            const float new_weight = expf(score - next_max);
            old_weight_shared = old_weight;
            new_weight_shared = new_weight;
            running_sum = running_sum * old_weight + new_weight;
            running_max = next_max;
        }
        __syncthreads();
        accumulator = accumulator * old_weight_shared
            + __half2float(value[lane]) * new_weight_shared;
        __syncthreads();
    }

    if (lane == 0) {
        inverse_sum_shared = 1.0f / running_sum;
    }
    __syncthreads();
    destination[lane] = __float2half_rn(accumulator * inverse_sum_shared);
}

extern "C" __global__ void nnis_vector_add_f16(
    const __half* left,
    const __half* right,
    __half* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __float2half_rn(
            __half2float(left[index]) + __half2float(right[index]));
    }
}

// Preserve the F16 activation boundary: SiLU is rounded to F16 before the
// elementwise product with the up projection.
extern "C" __global__ void nnis_silu_multiply_f16(
    const __half* gate,
    const __half* up,
    __half* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float x = __half2float(gate[index]);
        const __half activated = __float2half_rn(x / (1.0f + expf(-x)));
        output[index] = __float2half_rn(
            __half2float(activated) * __half2float(up[index]));
    }
}

// Edge-LLM's Llama network marks logits as F32 after the F16 lm_head tensor.
// Reproduce that high-level boundary: F32 accumulation -> F16 rounding -> F32.
extern "C" __global__ void nnis_lm_head_kn_f16_to_f32(
    const __half* input,
    const __half* weight,
    float* output,
    unsigned long long k,
    unsigned long long n
) {
    const unsigned long long col = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (col >= n) {
        return;
    }
    extern __shared__ float partial[];
    float sum = 0.0f;
    for (unsigned long long row = lane; row < k; row += blockDim.x) {
        sum = fmaf(
            __half2float(input[row]),
            __half2float(weight[row * n + col]),
            sum);
    }
    partial[lane] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        output[col] = __half2float(__float2half_rn(partial[0]));
    }
}
"#;

const REDUCTION_BLOCK_SIZE: u32 = 128;
const ELEMENTWISE_BLOCK_SIZE: u32 = 256;

#[derive(Debug)]
pub struct F16ReferenceKernels {
    context: Arc<Context>,
    narrow: Kernel,
    widen: Kernel,
    gather: Kernel,
    weighted_rms_norm: Kernel,
    project_kn: Kernel,
    rope_position: Kernel,
    cached_attention_decode: Kernel,
    vector_add: Kernel,
    silu_multiply: Kernel,
    lm_head: Kernel,
    attention_max_threads: u32,
}

impl F16ReferenceKernels {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        let code =
            compiler.compile_cubin(F16_REFERENCE_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let narrow = module.get_function("nnis_f16_narrow_from_f32")?;
        let widen = module.get_function("nnis_f16_widen_to_f32")?;
        let gather = module.get_function("nnis_gather_f16")?;
        let weighted_rms_norm = module.get_function("nnis_weighted_rmsnorm_f16_f32acc")?;
        let project_kn = module.get_function("nnis_project_kn_f16_f32acc")?;
        let rope_position = module.get_function("nnis_rope_rotate_half_position_f16")?;
        let cached_attention_decode =
            module.get_function("nnis_cached_attention_decode_f16_f32acc")?;
        let vector_add = module.get_function("nnis_vector_add_f16")?;
        let silu_multiply = module.get_function("nnis_silu_multiply_f16")?;
        let lm_head = module.get_function("nnis_lm_head_kn_f16_to_f32")?;

        for (name, kernel, block) in [
            ("narrow", &narrow, ELEMENTWISE_BLOCK_SIZE),
            ("widen", &widen, ELEMENTWISE_BLOCK_SIZE),
            ("gather", &gather, ELEMENTWISE_BLOCK_SIZE),
            (
                "weighted_rms_norm",
                &weighted_rms_norm,
                REDUCTION_BLOCK_SIZE,
            ),
            ("project_kn", &project_kn, REDUCTION_BLOCK_SIZE),
            ("rope_position", &rope_position, ELEMENTWISE_BLOCK_SIZE),
            ("vector_add", &vector_add, ELEMENTWISE_BLOCK_SIZE),
            ("silu_multiply", &silu_multiply, ELEMENTWISE_BLOCK_SIZE),
            ("lm_head", &lm_head, REDUCTION_BLOCK_SIZE),
        ] {
            let limit = kernel.attributes()?.max_threads_per_block;
            if block > limit {
                return Err(NnisError::invalid_input(format!(
                    "F16 reference kernel {name} block size {block} exceeds function limit {limit}"
                )));
            }
        }
        let shared_bytes = REDUCTION_BLOCK_SIZE as usize * std::mem::size_of::<f32>();
        for (name, kernel) in [
            ("weighted_rms_norm", &weighted_rms_norm),
            ("project_kn", &project_kn),
            ("lm_head", &lm_head),
        ] {
            if shared_bytes > kernel.attributes()?.max_dynamic_shared_memory_bytes as usize {
                return Err(NnisError::invalid_input(format!(
                    "F16 reference kernel {name} requires {shared_bytes} bytes of dynamic shared memory"
                )));
            }
        }
        let attention_max_threads = cached_attention_decode.attributes()?.max_threads_per_block;

        Ok(Self {
            context: Arc::clone(context),
            narrow,
            widen,
            gather,
            weighted_rms_norm,
            project_kn,
            rope_position,
            cached_attention_decode,
            vector_add,
            silu_multiply,
            lm_head,
            attention_max_threads,
        })
    }

    /// Narrow an F32 device tensor to IEEE binary16 without synchronizing.
    ///
    /// # Safety
    ///
    /// The stream, kernel set, input, and output buffers must remain alive and
    /// otherwise untouched until the stream reaches this launch.
    pub unsafe fn enqueue_narrow_from_f32(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        self.validate_len("F16 narrow", input.len(), output.len())?;
        self.validate_contexts(stream, &[input.ctx(), output.ctx()])?;
        if input.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(3, 2);
        args.push_buffer(input)
            .push_buffer(output)
            .push(input.len() as u64);
        let launch = KernelLaunch::new(
            &self.narrow,
            stream,
            LaunchConfig::for_num_elements(input.len(), ELEMENTWISE_BLOCK_SIZE)?,
        );
        unsafe { launch.launch(&mut args) }
    }

    /// Widen an IEEE binary16 device tensor to F32 without synchronizing.
    ///
    /// # Safety
    ///
    /// The stream, kernel set, input, and output buffers must remain alive and
    /// otherwise untouched until the stream reaches this launch.
    pub unsafe fn enqueue_widen_to_f32(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        self.validate_len("F16 widen", input.len(), output.len())?;
        self.validate_contexts(stream, &[input.ctx(), output.ctx()])?;
        if input.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(3, 2);
        args.push_buffer(input)
            .push_buffer(output)
            .push(input.len() as u64);
        let launch = KernelLaunch::new(
            &self.widen,
            stream,
            LaunchConfig::for_num_elements(input.len(), ELEMENTWISE_BLOCK_SIZE)?,
        );
        unsafe { launch.launch(&mut args) }
    }

    /// Gather F16 rows selected by device-resident token IDs.
    ///
    /// # Safety
    ///
    /// The stream, kernel set, table, indices, and output buffers must remain
    /// alive and otherwise untouched until the stream reaches this launch.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_gather(
        &self,
        stream: &Stream,
        table: &DeviceBuffer<u16>,
        indices: &DeviceBuffer<u32>,
        output: &DeviceBuffer<u16>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let table_len = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("F16 gather table shape overflows usize"))?;
        let output_len = indices
            .len()
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("F16 gather output shape overflows usize"))?;
        if table.len() != table_len || output.len() != output_len {
            return Err(NnisError::invalid_input(format!(
                "F16 gather expects table {table_len} and output {output_len}; got {}/{}",
                table.len(),
                output.len()
            )));
        }
        self.validate_contexts(stream, &[table.ctx(), indices.ctx(), output.ctx()])?;
        if output_len == 0 {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(6, 3);
        args.push_buffer(table)
            .push_buffer(indices)
            .push_buffer(output)
            .push(rows as u64)
            .push(cols as u64)
            .push(indices.len() as u64);
        let launch = KernelLaunch::new(
            &self.gather,
            stream,
            LaunchConfig::for_num_elements(output_len, ELEMENTWISE_BLOCK_SIZE)?,
        );
        unsafe { launch.launch(&mut args) }
    }

    /// Apply weighted RMSNorm over F16 storage with F32 reduction state.
    ///
    /// # Safety
    ///
    /// The stream, kernel set, input, weight, and output buffers must remain
    /// alive and otherwise untouched until the stream reaches this launch.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_weighted_rms_norm(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        weight: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) -> Result<()> {
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("F16 RMSNorm shape overflows usize"))?;
        if input.len() != count || output.len() != count || weight.len() != cols {
            return Err(NnisError::invalid_input(format!(
                "F16 RMSNorm expects input/output {count} and weight {cols}; got {}/{}/{}",
                input.len(),
                output.len(),
                weight.len()
            )));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(NnisError::invalid_input(format!(
                "F16 RMSNorm epsilon must be finite and positive; got {epsilon}"
            )));
        }
        self.validate_contexts(stream, &[input.ctx(), weight.ctx(), output.ctx()])?;
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let grid_rows = u32::try_from(rows)
            .map_err(|_| NnisError::invalid_input("F16 RMSNorm rows exceed u32"))?;
        let config = LaunchConfig::new(
            Dim3::new(grid_rows, 1, 1),
            Dim3::new(REDUCTION_BLOCK_SIZE, 1, 1),
        )
        .with_dynamic_shared_memory(REDUCTION_BLOCK_SIZE * std::mem::size_of::<f32>() as u32);
        let mut args = KernelArgs::with_capacity(6, 3);
        args.push_buffer(input)
            .push_buffer(weight)
            .push_buffer(output)
            .push(cols as u64)
            .push(1.0_f32 / cols as f32)
            .push(epsilon);
        let launch = KernelLaunch::new(&self.weighted_rms_norm, stream, config);
        unsafe { launch.launch(&mut args) }
    }

    /// Project one F16 row against an internal `[K, N]` F16 weight matrix.
    ///
    /// # Safety
    ///
    /// The stream, kernel set, input, weight, and output buffers must remain
    /// alive and otherwise untouched until the stream reaches this launch.
    pub unsafe fn enqueue_project_kn(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        weight: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        self.validate_project_shapes(
            "F16 projection",
            input.len(),
            weight.len(),
            output.len(),
            k,
            n,
        )?;
        self.validate_contexts(stream, &[input.ctx(), weight.ctx(), output.ctx()])?;
        if n == 0 {
            return Ok(());
        }
        let grid = u32::try_from(n)
            .map_err(|_| NnisError::invalid_input("F16 projection N exceeds u32"))?;
        let config =
            LaunchConfig::new(Dim3::new(grid, 1, 1), Dim3::new(REDUCTION_BLOCK_SIZE, 1, 1))
                .with_dynamic_shared_memory(
                    REDUCTION_BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
                );
        let mut args = KernelArgs::with_capacity(5, 3);
        args.push_buffer(input)
            .push_buffer(weight)
            .push_buffer(output)
            .push(k as u64)
            .push(n as u64);
        let launch = KernelLaunch::new(&self.project_kn, stream, config);
        unsafe { launch.launch(&mut args) }
    }

    /// Apply one position of Llama rotate-half RoPE to F16 Q/K storage.
    ///
    /// # Safety
    ///
    /// The stream, kernel set, input, RoPE caches, and output must remain alive
    /// and otherwise untouched until the stream reaches this launch.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_rope_position(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        cos_cache: &DeviceBuffer<f32>,
        sin_cache: &DeviceBuffer<f32>,
        output: &DeviceBuffer<u16>,
        heads: usize,
        head_dim: usize,
        position: usize,
        max_positions: usize,
    ) -> Result<()> {
        if heads == 0 || head_dim == 0 || head_dim % 2 != 0 {
            return Err(NnisError::invalid_input(format!(
                "F16 RoPE requires non-zero heads and even head_dim; got heads={heads}, head_dim={head_dim}"
            )));
        }
        let width = heads
            .checked_mul(head_dim)
            .ok_or_else(|| NnisError::invalid_input("F16 RoPE width overflows usize"))?;
        let cache_len = max_positions
            .checked_mul(head_dim / 2)
            .ok_or_else(|| NnisError::invalid_input("F16 RoPE cache shape overflows usize"))?;
        if input.len() != width
            || output.len() != width
            || cos_cache.len() != cache_len
            || sin_cache.len() != cache_len
        {
            return Err(NnisError::invalid_input(
                "F16 RoPE input/output/cache shapes do not match the declared heads/head_dim/positions",
            ));
        }
        if position >= max_positions {
            return Err(NnisError::invalid_input(format!(
                "F16 RoPE position {position} exceeds max position {max_positions}"
            )));
        }
        self.validate_contexts(
            stream,
            &[input.ctx(), cos_cache.ctx(), sin_cache.ctx(), output.ctx()],
        )?;
        let pairs = heads
            .checked_mul(head_dim / 2)
            .ok_or_else(|| NnisError::invalid_input("F16 RoPE pair count overflows usize"))?;
        let mut args = KernelArgs::with_capacity(8, 4);
        args.push_buffer(input)
            .push_buffer(cos_cache)
            .push_buffer(sin_cache)
            .push_buffer(output)
            .push(heads as u64)
            .push(head_dim as u64)
            .push(position as u64)
            .push(max_positions as u64);
        let launch = KernelLaunch::new(
            &self.rope_position,
            stream,
            LaunchConfig::for_num_elements(pairs, ELEMENTWISE_BLOCK_SIZE)?,
        );
        unsafe { launch.launch(&mut args) }
    }

    /// Decode one F16 query token against an owned F16 KV cache.
    ///
    /// # Safety
    ///
    /// The query/output buffers, cache allocations, stream, and kernel set must
    /// remain alive until completion. The cache must not be reset, appended on
    /// another stream, or otherwise mutated concurrently with this launch.
    pub unsafe fn enqueue_cached_attention_decode(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<u16>,
        cache: &KvCache<u16>,
        layer: usize,
        output: &DeviceBuffer<u16>,
        scale: f32,
    ) -> Result<()> {
        let config = cache.config();
        let kv_rows = cache.len(layer)?;
        if query.len() != output.len() || query.len() % config.head_dim != 0 {
            return Err(NnisError::invalid_input(
                "F16 cached attention query/output shape is inconsistent with KV head_dim",
            ));
        }
        let query_heads = query.len() / config.head_dim;
        if query_heads == 0 || config.heads == 0 || query_heads % config.heads != 0 {
            return Err(NnisError::invalid_input(format!(
                "F16 cached attention requires query heads divisible by KV heads; got {query_heads}/{}",
                config.heads
            )));
        }
        if kv_rows == 0 {
            return Err(NnisError::invalid_input(
                "F16 cached attention requires at least one valid KV position",
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(NnisError::invalid_input(format!(
                "F16 cached attention scale must be finite and positive; got {scale}"
            )));
        }
        self.validate_contexts(
            stream,
            &[
                query.ctx(),
                cache.keys().ctx(),
                cache.values().ctx(),
                output.ctx(),
            ],
        )?;
        if stream.raw() != cache.stream().raw() {
            return Err(NnisError::invalid_input(
                "F16 cached attention must execute on the KV cache owning stream",
            ));
        }
        let threads = u32::try_from(config.head_dim)
            .map_err(|_| NnisError::invalid_input("F16 attention head_dim exceeds u32"))?;
        if threads == 0 || threads > self.attention_max_threads {
            return Err(NnisError::unsupported(format!(
                "F16 attention head_dim {threads} exceeds kernel limit {}",
                self.attention_max_threads
            )));
        }
        let grid = u32::try_from(query_heads)
            .map_err(|_| NnisError::invalid_input("F16 query head count exceeds u32"))?;
        let mut args = KernelArgs::with_capacity(11, 4);
        args.push_buffer(query)
            .push_buffer(cache.keys())
            .push_buffer(cache.values())
            .push_buffer(output)
            .push(layer as u64)
            .push(query_heads as u64)
            .push(config.heads as u64)
            .push(config.capacity as u64)
            .push(config.head_dim as u64)
            .push(kv_rows as u64)
            .push(scale);
        let launch = KernelLaunch::new(
            &self.cached_attention_decode,
            stream,
            LaunchConfig::new(Dim3::new(grid, 1, 1), Dim3::new(threads, 1, 1)),
        );
        unsafe { launch.launch(&mut args) }
    }

    /// Add two F16 vectors and round the result back to F16.
    ///
    /// # Safety
    ///
    /// The stream, kernel set, both inputs, and output must remain alive and
    /// otherwise untouched until the stream reaches this launch.
    pub unsafe fn enqueue_vector_add(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<u16>,
        right: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        if left.len() != right.len() || left.len() != output.len() {
            return Err(NnisError::invalid_input(format!(
                "F16 vector-add length mismatch: {}/{}/{}",
                left.len(),
                right.len(),
                output.len()
            )));
        }
        self.validate_contexts(stream, &[left.ctx(), right.ctx(), output.ctx()])?;
        if output.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(4, 3);
        args.push_buffer(left)
            .push_buffer(right)
            .push_buffer(output)
            .push(output.len() as u64);
        let launch = KernelLaunch::new(
            &self.vector_add,
            stream,
            LaunchConfig::for_num_elements(output.len(), ELEMENTWISE_BLOCK_SIZE)?,
        );
        unsafe { launch.launch(&mut args) }
    }

    /// Apply SiLU to an F16 gate, round it to F16, then multiply by F16 up.
    ///
    /// # Safety
    ///
    /// The stream, kernel set, gate, up, and output buffers must remain alive
    /// and otherwise untouched until the stream reaches this launch.
    pub unsafe fn enqueue_silu_multiply(
        &self,
        stream: &Stream,
        gate: &DeviceBuffer<u16>,
        up: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        if gate.len() != up.len() || gate.len() != output.len() {
            return Err(NnisError::invalid_input(format!(
                "F16 SiLU-multiply length mismatch: {}/{}/{}",
                gate.len(),
                up.len(),
                output.len()
            )));
        }
        self.validate_contexts(stream, &[gate.ctx(), up.ctx(), output.ctx()])?;
        if output.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(4, 3);
        args.push_buffer(gate)
            .push_buffer(up)
            .push_buffer(output)
            .push(output.len() as u64);
        let launch = KernelLaunch::new(
            &self.silu_multiply,
            stream,
            LaunchConfig::for_num_elements(output.len(), ELEMENTWISE_BLOCK_SIZE)?,
        );
        unsafe { launch.launch(&mut args) }
    }

    /// Project the F16 hidden state through the F16 LM head into F32 logits.
    ///
    /// # Safety
    ///
    /// The stream, kernel set, input, weight, and output buffers must remain
    /// alive and otherwise untouched until the stream reaches this launch.
    pub unsafe fn enqueue_lm_head_f32_logits(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        weight: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        self.validate_project_shapes("F16 LM head", input.len(), weight.len(), output.len(), k, n)?;
        self.validate_contexts(stream, &[input.ctx(), weight.ctx(), output.ctx()])?;
        if n == 0 {
            return Ok(());
        }
        let grid =
            u32::try_from(n).map_err(|_| NnisError::invalid_input("F16 LM-head N exceeds u32"))?;
        let config =
            LaunchConfig::new(Dim3::new(grid, 1, 1), Dim3::new(REDUCTION_BLOCK_SIZE, 1, 1))
                .with_dynamic_shared_memory(
                    REDUCTION_BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
                );
        let mut args = KernelArgs::with_capacity(5, 3);
        args.push_buffer(input)
            .push_buffer(weight)
            .push_buffer(output)
            .push(k as u64)
            .push(n as u64);
        let launch = KernelLaunch::new(&self.lm_head, stream, config);
        unsafe { launch.launch(&mut args) }
    }

    fn validate_contexts(&self, stream: &Stream, contexts: &[&Arc<Context>]) -> Result<()> {
        if !Arc::ptr_eq(&self.context, stream.ctx())
            || contexts
                .iter()
                .any(|context| !Arc::ptr_eq(&self.context, context))
        {
            return Err(NnisError::invalid_input(
                "F16 reference kernels, stream and buffers must share one CUDA context",
            ));
        }
        Ok(())
    }

    fn validate_len(&self, name: &str, left: usize, right: usize) -> Result<()> {
        if left != right {
            return Err(NnisError::invalid_input(format!(
                "{name} length mismatch: {left} != {right}"
            )));
        }
        Ok(())
    }

    fn validate_project_shapes(
        &self,
        name: &str,
        input_len: usize,
        weight_len: usize,
        output_len: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        let expected_weight = k
            .checked_mul(n)
            .ok_or_else(|| NnisError::invalid_input(format!("{name} K*N overflows usize")))?;
        if input_len != k || weight_len != expected_weight || output_len != n {
            return Err(NnisError::invalid_input(format!(
                "{name} expects input={k}, weight={expected_weight}, output={n}; got {input_len}/{weight_len}/{output_len}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::{gpu_context, KvCacheConfig};

    fn approx_eq(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected} (tol={tolerance})"
        );
    }

    #[test]
    fn f16_reference_primitives_execute_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let kernels = F16ReferenceKernels::load(&context, &JitCompiler::new()).unwrap();

        let host = [1.0_f32, -2.0, 0.5, 3.0];
        let f32_values = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
        let half_values = DeviceBuffer::<u16>::new(&context, host.len()).unwrap();
        let widened = DeviceBuffer::<f32>::new(&context, host.len()).unwrap();
        unsafe {
            kernels
                .enqueue_narrow_from_f32(&stream, &f32_values, &half_values)
                .unwrap();
            kernels
                .enqueue_widen_to_f32(&stream, &half_values, &widened)
                .unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(widened.to_vec(&stream).unwrap(), host);

        let table_f32 =
            DeviceBuffer::from_host(&context, &stream, &[1.0_f32, 2.0, 0.5, 3.0]).unwrap();
        let table_f16 = DeviceBuffer::<u16>::new(&context, 4).unwrap();
        let indices = DeviceBuffer::from_host(&context, &stream, &[1_u32]).unwrap();
        let gathered = DeviceBuffer::<u16>::new(&context, 2).unwrap();
        unsafe {
            kernels
                .enqueue_narrow_from_f32(&stream, &table_f32, &table_f16)
                .unwrap();
            kernels
                .enqueue_gather(&stream, &table_f16, &indices, &gathered, 2, 2)
                .unwrap();
        }

        let weight_f32 =
            DeviceBuffer::from_host(&context, &stream, &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let weight_f16 = DeviceBuffer::<u16>::new(&context, 4).unwrap();
        let projected = DeviceBuffer::<u16>::new(&context, 2).unwrap();
        let projected_f32 = DeviceBuffer::<f32>::new(&context, 2).unwrap();
        unsafe {
            kernels
                .enqueue_narrow_from_f32(&stream, &weight_f32, &weight_f16)
                .unwrap();
            kernels
                .enqueue_project_kn(&stream, &gathered, &weight_f16, &projected, 2, 2)
                .unwrap();
            kernels
                .enqueue_widen_to_f32(&stream, &projected, &projected_f32)
                .unwrap();
        }

        let residual = DeviceBuffer::<u16>::new(&context, 2).unwrap();
        let residual_f32 = DeviceBuffer::<f32>::new(&context, 2).unwrap();
        unsafe {
            kernels
                .enqueue_vector_add(&stream, &gathered, &gathered, &residual)
                .unwrap();
            kernels
                .enqueue_widen_to_f32(&stream, &residual, &residual_f32)
                .unwrap();
        }

        let cos = DeviceBuffer::from_host(&context, &stream, &[1.0_f32]).unwrap();
        let sin = DeviceBuffer::from_host(&context, &stream, &[0.0_f32]).unwrap();
        let roped = DeviceBuffer::<u16>::new(&context, 2).unwrap();
        let roped_f32 = DeviceBuffer::<f32>::new(&context, 2).unwrap();
        unsafe {
            kernels
                .enqueue_rope_position(&stream, &gathered, &cos, &sin, &roped, 1, 2, 0, 1)
                .unwrap();
            kernels
                .enqueue_widen_to_f32(&stream, &roped, &roped_f32)
                .unwrap();
        }

        let mut cache =
            KvCache::<u16>::new(&stream, KvCacheConfig::new(1, 1, 2, 4).unwrap()).unwrap();
        let key = Arc::new(DeviceBuffer::<u16>::new(&context, 2).unwrap());
        let value = Arc::new(DeviceBuffer::<u16>::new(&context, 2).unwrap());
        let key_f32 = DeviceBuffer::from_host(&context, &stream, &[1.0_f32, 0.0]).unwrap();
        let value_f32 = DeviceBuffer::from_host(&context, &stream, &[0.5_f32, 2.0]).unwrap();
        unsafe {
            kernels
                .enqueue_narrow_from_f32(&stream, &key_f32, &key)
                .unwrap();
            kernels
                .enqueue_narrow_from_f32(&stream, &value_f32, &value)
                .unwrap();
        }
        cache
            .append_layer(0, Arc::clone(&key), Arc::clone(&value), 1)
            .unwrap();
        let query_f32 = DeviceBuffer::from_host(&context, &stream, &[1.0_f32, 0.0]).unwrap();
        let query = DeviceBuffer::<u16>::new(&context, 2).unwrap();
        let attention = DeviceBuffer::<u16>::new(&context, 2).unwrap();
        let attention_f32 = DeviceBuffer::<f32>::new(&context, 2).unwrap();
        unsafe {
            kernels
                .enqueue_narrow_from_f32(&stream, &query_f32, &query)
                .unwrap();
            kernels
                .enqueue_cached_attention_decode(&stream, &query, &cache, 0, &attention, 1.0)
                .unwrap();
            kernels
                .enqueue_widen_to_f32(&stream, &attention, &attention_f32)
                .unwrap();
        }

        let logits = DeviceBuffer::<f32>::new(&context, 2).unwrap();
        unsafe {
            kernels
                .enqueue_lm_head_f32_logits(&stream, &gathered, &weight_f16, &logits, 2, 2)
                .unwrap();
        }

        let norm_weight_f32 = DeviceBuffer::from_host(&context, &stream, &[1.0_f32, 1.0]).unwrap();
        let norm_weight = DeviceBuffer::<u16>::new(&context, 2).unwrap();
        let normed = DeviceBuffer::<u16>::new(&context, 2).unwrap();
        let normed_f32 = DeviceBuffer::<f32>::new(&context, 2).unwrap();
        unsafe {
            kernels
                .enqueue_narrow_from_f32(&stream, &norm_weight_f32, &norm_weight)
                .unwrap();
            kernels
                .enqueue_weighted_rms_norm(&stream, &gathered, &norm_weight, &normed, 1, 2, 1.0e-5)
                .unwrap();
            kernels
                .enqueue_widen_to_f32(&stream, &normed, &normed_f32)
                .unwrap();
        }

        let silu = DeviceBuffer::<u16>::new(&context, 2).unwrap();
        let silu_f32 = DeviceBuffer::<f32>::new(&context, 2).unwrap();
        unsafe {
            kernels
                .enqueue_silu_multiply(&stream, &gathered, &gathered, &silu)
                .unwrap();
            kernels
                .enqueue_widen_to_f32(&stream, &silu, &silu_f32)
                .unwrap();
        }
        stream.synchronize().unwrap();

        assert_eq!(projected_f32.to_vec(&stream).unwrap(), vec![9.5, 13.0]);
        assert_eq!(residual_f32.to_vec(&stream).unwrap(), vec![1.0, 6.0]);
        assert_eq!(roped_f32.to_vec(&stream).unwrap(), vec![0.5, 3.0]);
        assert_eq!(attention_f32.to_vec(&stream).unwrap(), vec![0.5, 2.0]);
        assert_eq!(logits.to_vec(&stream).unwrap(), vec![9.5, 13.0]);

        let normed = normed_f32.to_vec(&stream).unwrap();
        let rms = ((0.5_f32 * 0.5 + 3.0 * 3.0) / 2.0 + 1.0e-5).sqrt();
        approx_eq(normed[0], 0.5 / rms, 1.0e-3);
        approx_eq(normed[1], 3.0 / rms, 1.0e-3);

        let silu = silu_f32.to_vec(&stream).unwrap();
        let silu0 = 0.5_f32 / (1.0 + (-0.5_f32).exp());
        let silu1 = 3.0_f32 / (1.0 + (-3.0_f32).exp());
        approx_eq(silu[0], silu0 * 0.5, 2.0e-3);
        approx_eq(silu[1], silu1 * 3.0, 2.0e-3);
    }
}
