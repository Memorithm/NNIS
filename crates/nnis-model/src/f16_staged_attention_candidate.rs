//! Candidate-only F16 cached-attention decode primitive that removes per-position barriers.
//!
//! The qualified F16 reference kernel keeps the query/key score and online-softmax
//! chain serial on lane zero, but synchronizes the whole block twice for every KV
//! position so the value lanes can consume two shared scalar weights. This candidate
//! preserves the exact score FMA order, online-softmax order, per-output value update
//! order, F32 accumulation, and final F16 boundary while staging every old/new weight
//! pair in dynamic shared memory first. One block-wide barrier then releases the value
//! lanes to consume all positions without further synchronization.
//!
//! Nothing in this module changes runtime policy by itself.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, KvCache, NnisError, Result, Stream};
use std::sync::Arc;

const F16_STAGED_ATTENTION_SOURCE: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void nnis_cached_attention_decode_f16_staged_weights(
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

    extern __shared__ float staged[];
    float* old_weights = staged;
    float* new_weights = staged + kv_rows;
    __shared__ float inverse_sum_shared;

    if (lane == 0) {
        float running_max = -3.402823466e+38F;
        float running_sum = 0.0f;

        for (unsigned long long pos = 0; pos < kv_rows; ++pos) {
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
            old_weights[pos] = old_weight;
            new_weights[pos] = new_weight;
            running_sum = running_sum * old_weight + new_weight;
            running_max = next_max;
        }
        inverse_sum_shared = 1.0f / running_sum;
    }

    __syncthreads();

    float accumulator = 0.0f;
    for (unsigned long long pos = 0; pos < kv_rows; ++pos) {
        const __half* value = head_values + pos * head_dim;
        accumulator = accumulator * old_weights[pos]
            + __half2float(value[lane]) * new_weights[pos];
    }
    destination[lane] = __float2half_rn(accumulator * inverse_sum_shared);
}
"#;

/// Candidate that stages serial softmax weights once and removes the reference
/// kernel's two block-wide barriers per KV position.
#[derive(Debug)]
pub struct F16CachedAttentionStagedWeightsCandidate {
    context: Arc<Context>,
    kernel: Kernel,
    max_threads_per_block: u32,
}

impl F16CachedAttentionStagedWeightsCandidate {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        let code = compiler.compile_cubin(
            F16_STAGED_ATTENTION_SOURCE,
            &CompileOptions::for_device(context),
        )?;
        let module = Module::load(context, &code)?;
        let kernel = module.get_function("nnis_cached_attention_decode_f16_staged_weights")?;
        let max_threads_per_block = kernel.attributes()?.max_threads_per_block;
        Ok(Self {
            context: Arc::clone(context),
            kernel,
            max_threads_per_block,
        })
    }

    pub fn cached_attention_decode(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<u16>,
        cache: &KvCache<u16>,
        layer: usize,
        output: &DeviceBuffer<u16>,
        scale: f32,
    ) -> Result<()> {
        // SAFETY: this method keeps all borrowed resources alive until the
        // immediately following stream synchronization completes.
        unsafe {
            self.enqueue_cached_attention_decode(stream, query, cache, layer, output, scale)?
        };
        stream.synchronize()
    }

    /// Enqueue the candidate without synchronizing.
    ///
    /// # Safety
    ///
    /// The stream, kernel, query/output buffers, and cache allocations must stay
    /// alive and otherwise untouched until the stream completes. The cache must
    /// not be reset or mutated concurrently from another stream.
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
        if query.len() != output.len() {
            return Err(NnisError::invalid_input(format!(
                "staged F16 attention query/output lengths differ: {}/{}",
                query.len(),
                output.len()
            )));
        }
        if query.len() % config.head_dim != 0 {
            return Err(NnisError::invalid_input(format!(
                "staged F16 attention query width {} is not divisible by head_dim {}",
                query.len(),
                config.head_dim
            )));
        }
        let query_heads = query.len() / config.head_dim;
        if query_heads == 0 || config.heads == 0 || query_heads % config.heads != 0 {
            return Err(NnisError::invalid_input(format!(
                "staged F16 attention requires query heads divisible by KV heads; got {query_heads}/{}",
                config.heads
            )));
        }
        if kv_rows == 0 {
            return Err(NnisError::invalid_input(
                "staged F16 attention requires at least one valid KV position",
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(NnisError::invalid_input(format!(
                "staged F16 attention scale must be finite and positive; got {scale}"
            )));
        }
        if !Arc::ptr_eq(&self.context, stream.ctx())
            || !Arc::ptr_eq(&self.context, query.ctx())
            || !Arc::ptr_eq(&self.context, cache.keys().ctx())
            || !Arc::ptr_eq(&self.context, cache.values().ctx())
            || !Arc::ptr_eq(&self.context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "staged F16 attention kernel, stream, cache, and buffers must share one CUDA context",
            ));
        }
        if stream.raw() != cache.stream().raw() {
            return Err(NnisError::invalid_input(
                "staged F16 attention must execute on the KV cache owning stream",
            ));
        }

        let threads = u32::try_from(config.head_dim)
            .map_err(|_| NnisError::invalid_input("staged F16 attention head_dim exceeds u32"))?;
        if threads == 0 || threads > self.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "staged F16 attention head_dim {threads} exceeds function thread limit {}",
                self.max_threads_per_block
            )));
        }
        let query_heads_u32 = u32::try_from(query_heads)
            .map_err(|_| NnisError::invalid_input("staged F16 query head count exceeds u32"))?;
        let shared_bytes = kv_rows
            .checked_mul(2)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                NnisError::invalid_input("staged F16 attention shared-memory size overflow")
            })?;
        let shared_bytes = u32::try_from(shared_bytes).map_err(|_| {
            NnisError::invalid_input("staged F16 attention shared-memory size exceeds u32")
        })?;

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
            &self.kernel,
            stream,
            LaunchConfig::new(Dim3::new(query_heads_u32, 1, 1), Dim3::new(threads, 1, 1))
                .with_dynamic_shared_memory(shared_bytes),
        );
        // SAFETY: argument order and widths match the CUDA signature exactly;
        // asynchronous lifetime obligations are stated on this method.
        unsafe { launch.launch(&mut args) }
    }
}
