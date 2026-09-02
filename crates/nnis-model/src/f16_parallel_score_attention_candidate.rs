//! Candidate-only F16 cached-attention kernel with parallel Q·K score reduction.
//!
//! KA15 shows the qualified F16 decode-attention kernel as the largest remaining
//! GPU-time contributor after projection/MLP fusion. The existing staged-weights
//! candidate removes per-position barriers but intentionally leaves every Q·K dot
//! product serial on lane zero. This candidate instead assigns KV positions to
//! warps, computes each F16×F16 score with a warp-parallel F32 reduction, then
//! preserves the serial online-softmax recurrence and the per-output value update
//! order in F32 before the final F16 tensor boundary.
//!
//! The score reduction tree differs from the qualified serial FMA order. Therefore
//! this module is research-only until physical evidence quantifies numerical drift
//! and end-to-end trajectory stability. It changes no runtime plan or default.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, KvCache, NnisError, Result, Stream};
use std::sync::Arc;

const F16_PARALLEL_SCORE_ATTENTION_SOURCE: &str = r#"
#include <cuda_fp16.h>

extern "C" __global__ void nnis_cached_attention_decode_f16_parallel_scores(
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
    if (query_head >= query_heads) {
        return;
    }

    const unsigned int thread = threadIdx.x;
    const unsigned int warp_lane = thread & 31U;
    const unsigned int warp = thread >> 5U;
    const unsigned int warp_count = blockDim.x >> 5U;

    const unsigned long long group_size = query_heads / kv_heads;
    const unsigned long long kv_head = query_head / group_size;
    const __half* q = query + query_head * head_dim;
    const unsigned long long cache_base =
        ((layer * kv_heads + kv_head) * capacity) * head_dim;
    const __half* head_keys = keys + cache_base;
    const __half* head_values = values + cache_base;
    __half* destination = output + query_head * head_dim;

    extern __shared__ float staged[];
    float* scores = staged;
    float* old_weights = scores + kv_rows;
    float* new_weights = old_weights + kv_rows;
    __shared__ float inverse_sum_shared;

    // Several warps score independent KV positions concurrently. Within one
    // position, lanes reduce the F32 partial products using a fixed warp tree.
    for (unsigned long long pos = warp; pos < kv_rows; pos += warp_count) {
        const __half* key = head_keys + pos * head_dim;
        float partial = 0.0f;
        for (unsigned long long dim = warp_lane; dim < head_dim; dim += 32ULL) {
            partial = fmaf(
                __half2float(q[dim]),
                __half2float(key[dim]),
                partial);
        }
        for (unsigned int offset = 16U; offset > 0U; offset >>= 1U) {
            partial += __shfl_down_sync(0xffffffffU, partial, offset);
        }
        if (warp_lane == 0U) {
            scores[pos] = partial * scale;
        }
    }

    __syncthreads();

    // Preserve the qualified serial online-softmax ordering across KV rows.
    if (thread == 0U) {
        float running_max = -3.402823466e+38F;
        float running_sum = 0.0f;
        for (unsigned long long pos = 0; pos < kv_rows; ++pos) {
            const float score = scores[pos];
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

    // Preserve the per-output value accumulation order used by the reference
    // and staged-weights kernels. Extra score warps simply idle in this phase.
    if ((unsigned long long)thread < head_dim) {
        float accumulator = 0.0f;
        for (unsigned long long pos = 0; pos < kv_rows; ++pos) {
            const __half* value = head_values + pos * head_dim;
            accumulator = accumulator * old_weights[pos]
                + __half2float(value[thread]) * new_weights[pos];
        }
        destination[thread] = __float2half_rn(accumulator * inverse_sum_shared);
    }
}
"#;

#[derive(Debug)]
pub struct F16CachedAttentionParallelScoreCandidate {
    context: Arc<Context>,
    kernel: Kernel,
    max_threads_per_block: u32,
    max_dynamic_shared_memory_bytes: u32,
}

impl F16CachedAttentionParallelScoreCandidate {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        let code = compiler.compile_cubin(
            F16_PARALLEL_SCORE_ATTENTION_SOURCE,
            &CompileOptions::for_device(context),
        )?;
        let module = Module::load(context, &code)?;
        let kernel = module.get_function("nnis_cached_attention_decode_f16_parallel_scores")?;
        let attributes = kernel.attributes()?;
        Ok(Self {
            context: Arc::clone(context),
            kernel,
            max_threads_per_block: attributes.max_threads_per_block,
            max_dynamic_shared_memory_bytes: attributes.max_dynamic_shared_memory_bytes,
        })
    }

    #[must_use]
    pub fn max_supported_kv_rows(&self) -> usize {
        let limit = self
            .context
            .props()
            .shared_memory_per_block
            .min(self.max_dynamic_shared_memory_bytes);
        limit as usize / (3 * std::mem::size_of::<f32>())
    }

    #[must_use]
    pub fn supports_kv_rows(&self, kv_rows: usize) -> bool {
        kv_rows > 0 && kv_rows <= self.max_supported_kv_rows()
    }

    pub fn cached_attention_decode(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<u16>,
        cache: &KvCache<u16>,
        layer: usize,
        output: &DeviceBuffer<u16>,
        scale: f32,
        threads_per_block: u32,
    ) -> Result<()> {
        // SAFETY: all borrowed resources remain alive until the immediately
        // following stream synchronization completes.
        unsafe {
            self.enqueue_cached_attention_decode(
                stream,
                query,
                cache,
                layer,
                output,
                scale,
                threads_per_block,
            )?;
        }
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
        threads_per_block: u32,
    ) -> Result<()> {
        let config = cache.config();
        let kv_rows = cache.len(layer)?;
        if query.len() != output.len() {
            return Err(NnisError::invalid_input(format!(
                "parallel-score F16 attention query/output lengths differ: {}/{}",
                query.len(),
                output.len()
            )));
        }
        if query.len() % config.head_dim != 0 {
            return Err(NnisError::invalid_input(format!(
                "parallel-score F16 attention query width {} is not divisible by head_dim {}",
                query.len(),
                config.head_dim
            )));
        }
        let query_heads = query.len() / config.head_dim;
        if query_heads == 0 || config.heads == 0 || query_heads % config.heads != 0 {
            return Err(NnisError::invalid_input(format!(
                "parallel-score F16 attention requires query heads divisible by KV heads; got {query_heads}/{}",
                config.heads
            )));
        }
        if kv_rows == 0 {
            return Err(NnisError::invalid_input(
                "parallel-score F16 attention requires at least one valid KV position",
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(NnisError::invalid_input(format!(
                "parallel-score F16 attention scale must be finite and positive; got {scale}"
            )));
        }
        if !Arc::ptr_eq(&self.context, stream.ctx())
            || !Arc::ptr_eq(&self.context, query.ctx())
            || !Arc::ptr_eq(&self.context, cache.keys().ctx())
            || !Arc::ptr_eq(&self.context, cache.values().ctx())
            || !Arc::ptr_eq(&self.context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "parallel-score F16 attention kernel, stream, cache, and buffers must share one CUDA context",
            ));
        }
        if stream.raw() != cache.stream().raw() {
            return Err(NnisError::invalid_input(
                "parallel-score F16 attention must execute on the KV cache owning stream",
            ));
        }

        let head_dim_u32 = u32::try_from(config.head_dim).map_err(|_| {
            NnisError::invalid_input("parallel-score F16 attention head_dim exceeds u32")
        })?;
        let warp_aligned = threads_per_block & 31 == 0;
        if threads_per_block < head_dim_u32
            || !warp_aligned
            || threads_per_block > self.max_threads_per_block
        {
            return Err(NnisError::invalid_input(format!(
                "parallel-score F16 attention block size {threads_per_block} must be a multiple of 32, at least head_dim {head_dim_u32}, and at most function limit {}",
                self.max_threads_per_block
            )));
        }
        if !self.supports_kv_rows(kv_rows) {
            return Err(NnisError::unsupported(format!(
                "parallel-score F16 attention requires {kv_rows} KV rows but this device/kernel supports at most {} within validated dynamic shared memory",
                self.max_supported_kv_rows()
            )));
        }

        let query_heads_u32 = u32::try_from(query_heads)
            .map_err(|_| NnisError::invalid_input("parallel-score query head count exceeds u32"))?;
        let shared_bytes = kv_rows
            .checked_mul(3)
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                NnisError::invalid_input("parallel-score F16 attention shared-memory size overflow")
            })?;
        let shared_bytes = u32::try_from(shared_bytes).map_err(|_| {
            NnisError::invalid_input("parallel-score F16 attention shared-memory size exceeds u32")
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
            LaunchConfig::new(
                Dim3::new(query_heads_u32, 1, 1),
                Dim3::new(threads_per_block, 1, 1),
            )
            .with_dynamic_shared_memory(shared_bytes),
        );
        // SAFETY: argument order and widths match the CUDA signature exactly;
        // asynchronous lifetime obligations are stated on this method.
        unsafe { launch.launch(&mut args) }
    }
}
