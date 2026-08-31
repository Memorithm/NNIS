//! Candidate-only cached-attention decode primitive.
//!
//! The production decoder currently assigns one CUDA thread to each query head.
//! This candidate preserves the serial score/online-softmax chain on lane zero,
//! but lets one thread own each independent value/output dimension. The score
//! FMA order and the per-output update order across KV positions are unchanged.
//! Nothing in this module changes decoder policy by itself.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, KvCache, NnisError, Result, Stream};
use std::sync::Arc;

const CACHED_ATTENTION_PARALLEL_VALUE_SOURCE: &str = r#"
extern "C" __global__ void nnis_cached_attention_decode_parallel_value_f32(
    const float* query,
    const float* keys,
    const float* values,
    float* output,
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
    const float* q = query + query_head * head_dim;
    const unsigned long long cache_base =
        ((layer * kv_heads + kv_head) * capacity) * head_dim;
    const float* head_keys = keys + cache_base;
    const float* head_values = values + cache_base;
    float* destination = output + query_head * head_dim;

    __shared__ float old_weight_shared;
    __shared__ float new_weight_shared;
    __shared__ float inverse_sum_shared;

    destination[lane] = 0.0f;
    __syncthreads();

    float running_max = -3.402823466e+38F;
    float running_sum = 0.0f;
    for (unsigned long long position = 0; position < kv_rows; ++position) {
        const float* value = head_values + position * head_dim;
        if (lane == 0) {
            const float* key = head_keys + position * head_dim;
            float score = 0.0f;
            for (unsigned long long dim = 0; dim < head_dim; ++dim) {
                score = fmaf(q[dim], key[dim], score);
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

        destination[lane] =
            destination[lane] * old_weight_shared + value[lane] * new_weight_shared;
        __syncthreads();
    }

    if (lane == 0) {
        inverse_sum_shared = 1.0f / running_sum;
    }
    __syncthreads();
    destination[lane] *= inverse_sum_shared;
}
"#;

/// Candidate one-token cached attention that parallelizes only the value/output
/// dimension while preserving lane-zero score and online-softmax sequencing.
#[derive(Debug)]
pub struct F32CachedAttentionDecodeParallelValue {
    context: Arc<Context>,
    kernel: Kernel,
    max_threads_per_block: u32,
}

impl F32CachedAttentionDecodeParallelValue {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        let code = compiler.compile_cubin(
            CACHED_ATTENTION_PARALLEL_VALUE_SOURCE,
            &CompileOptions::for_device(context),
        )?;
        let module = Module::load(context, &code)?;
        let kernel = module.get_function("nnis_cached_attention_decode_parallel_value_f32")?;
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
        query: &DeviceBuffer<f32>,
        cache: &KvCache<f32>,
        layer: usize,
        output: &DeviceBuffer<f32>,
        scale: f32,
    ) -> Result<()> {
        // SAFETY: this method retains all borrows until the stream is drained.
        unsafe {
            self.enqueue_cached_attention_decode(stream, query, cache, layer, output, scale)?
        };
        stream.synchronize()
    }

    /// Enqueue the candidate without synchronizing.
    ///
    /// # Safety
    ///
    /// The stream, kernel, query/output buffers and cache allocations must stay
    /// alive and otherwise untouched until the stream completes. The cache must
    /// not be reset or mutated concurrently from another stream.
    pub unsafe fn enqueue_cached_attention_decode(
        &self,
        stream: &Stream,
        query: &DeviceBuffer<f32>,
        cache: &KvCache<f32>,
        layer: usize,
        output: &DeviceBuffer<f32>,
        scale: f32,
    ) -> Result<()> {
        let config = cache.config();
        let kv_rows = cache.len(layer)?;
        if query.len() != output.len() {
            return Err(NnisError::invalid_input(format!(
                "candidate cached attention query/output lengths differ: {}/{}",
                query.len(),
                output.len()
            )));
        }
        if query.len() % config.head_dim != 0 {
            return Err(NnisError::invalid_input(format!(
                "candidate cached attention query width {} is not divisible by head_dim {}",
                query.len(),
                config.head_dim
            )));
        }
        let query_heads = query.len() / config.head_dim;
        if query_heads == 0 || config.heads == 0 || query_heads % config.heads != 0 {
            return Err(NnisError::invalid_input(format!(
                "candidate cached attention requires query heads divisible by KV heads; got {query_heads}/{}",
                config.heads
            )));
        }
        if kv_rows == 0 {
            return Err(NnisError::invalid_input(
                "candidate cached attention requires at least one valid KV position",
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(NnisError::invalid_input(format!(
                "candidate attention scale must be finite and positive; got {scale}"
            )));
        }
        if !Arc::ptr_eq(&self.context, stream.ctx())
            || !Arc::ptr_eq(&self.context, query.ctx())
            || !Arc::ptr_eq(&self.context, cache.keys().ctx())
            || !Arc::ptr_eq(&self.context, cache.values().ctx())
            || !Arc::ptr_eq(&self.context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "candidate cached attention kernel, stream, cache and buffers must share one CUDA context",
            ));
        }
        if stream.raw() != cache.stream().raw() {
            return Err(NnisError::invalid_input(
                "candidate cached attention must execute on the KV cache owning stream",
            ));
        }

        let threads = u32::try_from(config.head_dim)
            .map_err(|_| NnisError::invalid_input("candidate attention head_dim exceeds u32"))?;
        if threads == 0 || threads > self.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "candidate attention head_dim {threads} exceeds function thread limit {}",
                self.max_threads_per_block
            )));
        }
        let query_heads_u32 = u32::try_from(query_heads)
            .map_err(|_| NnisError::invalid_input("candidate query head count exceeds u32"))?;

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
            LaunchConfig::new(Dim3::new(query_heads_u32, 1, 1), Dim3::new(threads, 1, 1)),
        );
        // SAFETY: argument order and widths match the CUDA signature exactly;
        // asynchronous lifetime obligations are stated on this method.
        unsafe { launch.launch(&mut args) }
    }
}
