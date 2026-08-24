//! Scaled dot-product attention for row-major `f32` heads.
//!
//! Computes `O = softmax(Q * K^T * scale) * V` where `Q` is
//! `query_rows x head_dim`, `K` is `kv_rows x head_dim`, and `V` is
//! `kv_rows x value_dim`. Two execution paths exist behind one family:
//!
//! - Fused: one thread block streams its whole query row; key/value chunks
//!   are staged in dynamic shared memory, scores are explicit-FMA chains,
//!   and an online (chunk-wise) running max/sum softmax keeps every
//!   intermediate device-resident. Nothing the size of the score matrix ever
//!   exists in global memory.
//! - Composed: the existing transposed-B GEMM, elementwise scaling, row
//!   softmax dispatch, and plain GEMM families materialize the probability
//!   matrix through three kernel launches per stage.
//!
//! Both paths accept an optional causal mask (query row `i` attends only to
//! keys `j <= i`), which requires square score shapes. Causal positions are
//! excluded before exponentiation, so they contribute exactly zero weight.
//!
//! Scores themselves are deterministic f32 chains, but `expf` and the
//! running-maximum rescaling differ from host transcendentals by ulps, so
//! correctness tests validate against an f64 oracle inside explicit
//! tolerances (the established softmax precedent) rather than bit-exactly.

/// Attention mask selecting which key positions each query may attend to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMask {
    /// Every query attends to every key.
    None,
    /// Query row `i` attends only to keys `j <= i`. Requires
    /// `query_rows == kv_rows`.
    Causal,
}

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const ATTENTION_SOURCE: &str = r#"
__device__ __forceinline__ float nnis_neg_inf() {
    return __int_as_float(0xff800000u);
}

extern "C" __global__ void nnis_attention_fused_f32(
    const float* queries,
    const float* keys,
    const float* values,
    float* output,
    unsigned long long query_rows,
    unsigned long long head_dim,
    unsigned long long kv_rows,
    unsigned long long value_dim,
    float scale,
    unsigned int causal
) {
    extern __shared__ float smem[];
    const unsigned int threads = blockDim.x;
    const unsigned int tid = threadIdx.x;
    float* q_sh = smem;                          // head_dim
    float* k_tile = q_sh + head_dim;             // threads * head_dim
    float* v_tile = k_tile + threads * head_dim; // threads * value_dim
    float* scores = v_tile + threads * value_dim;// threads
    float* scratch = scores + threads;           // threads
    float* acc = scratch + threads;              // value_dim

    const unsigned long long row = blockIdx.x;

    for (unsigned int i = tid; i < head_dim; i += threads) {
        q_sh[i] = queries[row * head_dim + i];
    }
    for (unsigned int c = tid; c < value_dim; c += threads) {
        acc[c] = 0.0f;
    }

    float running_max = nnis_neg_inf();
    float running_sum = 0.0f;

    for (unsigned long long base = 0; base < kv_rows; base += threads) {
        for (unsigned int idx = tid; idx < threads * head_dim; idx += threads) {
            const unsigned int r = idx / head_dim;
            const unsigned int c = idx % head_dim;
            const unsigned long long global = base + r;
            k_tile[idx] =
                global < kv_rows ? keys[global * head_dim + c] : 0.0f;
        }
        for (unsigned int idx = tid; idx < threads * value_dim; idx += threads) {
            const unsigned int r = idx / value_dim;
            const unsigned int c = idx % value_dim;
            const unsigned long long global = base + r;
            v_tile[idx] =
                global < kv_rows ? values[global * value_dim + c] : 0.0f;
        }
        __syncthreads();

        float score = nnis_neg_inf();
        const unsigned int key = base + tid;
        const int unmasked = !causal || key <= row;
        if (key < kv_rows && unmasked) {
            score = 0.0f;
            for (unsigned int e = 0; e < head_dim; ++e) {
                score = fmaf(q_sh[e], k_tile[tid * head_dim + e], score);
            }
            score *= scale;
        }
        scores[tid] = score;
        scratch[tid] = score;
        __syncthreads();

        for (unsigned int stride = threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            }
            __syncthreads();
        }
        const float new_max = fmaxf(running_max, scratch[0]);
        const float rescale = expf(running_max - new_max);

        const float weight = expf(score - new_max);
        scores[tid] = weight;
        scratch[tid] = weight;
        __syncthreads();

        for (unsigned int stride = threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] += scratch[tid + stride];
            }
            __syncthreads();
        }
        running_sum = running_sum * rescale + scratch[0];
        running_max = new_max;

        for (unsigned int c = tid; c < value_dim; c += threads) {
            acc[c] *= rescale;
        }
        __syncthreads();

        for (unsigned int c = tid; c < value_dim; c += threads) {
            float total = acc[c];
            for (unsigned int t = 0; t < threads; ++t) {
                total += scores[t] * v_tile[t * value_dim + c];
            }
            acc[c] = total;
        }
        __syncthreads();
    }

    for (unsigned int c = tid; c < value_dim; c += threads) {
        output[row * value_dim + c] = acc[c] / running_sum;
    }
}

extern "C" __global__ void nnis_attention_fused_multihead_f32(
    const float* queries,
    const float* keys,
    const float* values,
    float* output,
    unsigned int heads,
    unsigned long long query_rows,
    unsigned long long head_dim,
    unsigned long long kv_rows,
    unsigned long long value_dim,
    float scale,
    unsigned int causal
) {
    extern __shared__ float smem[];
    const unsigned int threads = blockDim.x;
    const unsigned int tid = threadIdx.x;
    float* q_sh = smem;                          // head_dim
    float* k_tile = q_sh + head_dim;             // threads * head_dim
    float* v_tile = k_tile + threads * head_dim; // threads * value_dim
    float* scores = v_tile + threads * value_dim;// threads
    float* scratch = scores + threads;           // threads
    float* acc = scratch + threads;              // value_dim

    // One block owns one (head, query row) pair over the packed
    // [heads][rows][dim] layout.
    const unsigned long long block = blockIdx.x;
    const unsigned long long row = block % query_rows;
    const unsigned long long head = block / query_rows;
    const float* q_head = queries + head * query_rows * head_dim;
    const float* k_head = keys + head * kv_rows * head_dim;
    const float* v_head = values + head * kv_rows * value_dim;
    float* o_head = output + head * query_rows * value_dim;

    for (unsigned int i = tid; i < head_dim; i += threads) {
        q_sh[i] = q_head[row * head_dim + i];
    }
    for (unsigned int c = tid; c < value_dim; c += threads) {
        acc[c] = 0.0f;
    }

    float running_max = nnis_neg_inf();
    float running_sum = 0.0f;

    for (unsigned long long base = 0; base < kv_rows; base += threads) {
        for (unsigned int idx = tid; idx < threads * head_dim; idx += threads) {
            const unsigned int r = idx / head_dim;
            const unsigned int c = idx % head_dim;
            const unsigned long long global = base + r;
            k_tile[idx] =
                global < kv_rows ? k_head[global * head_dim + c] : 0.0f;
        }
        for (unsigned int idx = tid; idx < threads * value_dim; idx += threads) {
            const unsigned int r = idx / value_dim;
            const unsigned int c = idx % value_dim;
            const unsigned long long global = base + r;
            v_tile[idx] =
                global < kv_rows ? v_head[global * value_dim + c] : 0.0f;
        }
        __syncthreads();

        float score = nnis_neg_inf();
        const unsigned int key = base + tid;
        const int unmasked = !causal || key <= row;
        if (key < kv_rows && unmasked) {
            score = 0.0f;
            for (unsigned int e = 0; e < head_dim; ++e) {
                score = fmaf(q_sh[e], k_tile[tid * head_dim + e], score);
            }
            score *= scale;
        }
        scores[tid] = score;
        scratch[tid] = score;
        __syncthreads();

        for (unsigned int stride = threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            }
            __syncthreads();
        }
        const float new_max = fmaxf(running_max, scratch[0]);
        const float rescale = expf(running_max - new_max);

        const float weight = expf(score - new_max);
        scores[tid] = weight;
        scratch[tid] = weight;
        __syncthreads();

        for (unsigned int stride = threads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] += scratch[tid + stride];
            }
            __syncthreads();
        }
        running_sum = running_sum * rescale + scratch[0];
        running_max = new_max;

        for (unsigned int c = tid; c < value_dim; c += threads) {
            acc[c] *= rescale;
        }
        __syncthreads();

        for (unsigned int c = tid; c < value_dim; c += threads) {
            float total = acc[c];
            for (unsigned int t = 0; t < threads; ++t) {
                total += scores[t] * v_tile[t * value_dim + c];
            }
            acc[c] = total;
        }
        __syncthreads();
    }

    for (unsigned int c = tid; c < value_dim; c += threads) {
        o_head[row * value_dim + c] = acc[c] / running_sum;
    }
}

extern "C" __global__ void nnis_attention_scale_causal_f32(
    const float* scores,
    float* output,
    unsigned long long query_rows,
    unsigned long long kv_rows,
    float scale
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long elements = query_rows * kv_rows;
    if (index >= elements) {
        return;
    }
    const unsigned long long row = index / kv_rows;
    const unsigned long long col = index % kv_rows;
    if (col > row) {
        output[index] = nnis_neg_inf();
    } else {
        output[index] = scores[index] * scale;
    }
}
extern "C" __global__ void nnis_attention_scale_causal_multihead_f32(
    const float* scores,
    float* output,
    unsigned long long elements,
    unsigned long long query_rows,
    unsigned long long kv_rows,
    float scale
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    // Packed [heads][query_rows][kv_rows]: the causal row repeats per head.
    const unsigned long long col = index % kv_rows;
    const unsigned long long row = (index / kv_rows) % query_rows;
    if (col > row) {
        output[index] = nnis_neg_inf();
    } else {
        output[index] = scores[index] * scale;
    }
}
"#;
const CAUSAL_BLOCK_SIZE: u32 = 256;
const DEFAULT_BLOCK_SIZE: u32 = 64;

/// Context-bound scaled dot-product attention.
#[derive(Debug)]
pub struct F32Attention {
    fused: Kernel,
    fused_multihead: Kernel,
    scale_causal: Kernel,
    scale_causal_multihead: Kernel,
    block_size: u32,
}

impl F32Attention {
    /// Compile (or reuse cached CUBIN) and load the attention kernel set.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load the fused kernel with an explicitly selected power-of-two block
    /// width. The same width is the key-chunk size streamed through shared
    /// memory.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 || !block_size.is_power_of_two() {
            return Err(NnisError::invalid_input(format!(
                "attention block size {block_size} is not a non-zero power of two"
            )));
        }
        let code =
            compiler.compile_cubin(ATTENTION_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let fused = module.get_function("nnis_attention_fused_f32")?;
        let fused_multihead = module.get_function("nnis_attention_fused_multihead_f32")?;
        let scale_causal = module.get_function("nnis_attention_scale_causal_f32")?;
        let scale_causal_multihead =
            module.get_function("nnis_attention_scale_causal_multihead_f32")?;
        let attributes = fused.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "attention block size {block_size} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        let attributes_causal = scale_causal.attributes()?;
        if attributes.max_threads_per_block < attributes_causal.max_threads_per_block {
            // Kept uniform so one validation covers both launches.
            return Err(NnisError::invalid_input(
                "attention scale-causal kernel exceeds fused thread limit",
            ));
        }
        Ok(Self {
            fused,
            fused_multihead,
            scale_causal,
            scale_causal_multihead,
            block_size,
        })
    }

    /// CUDA thread-block width; also the streamed key-chunk length.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Dynamic shared-memory bytes the fused path needs for these head
    /// shapes at this family's configured block width.
    pub fn fused_shared_memory_bytes(&self, head_dim: usize, value_dim: usize) -> Option<usize> {
        let threads = self.block_size as usize;
        let floats = threads
            .checked_mul(head_dim)?
            .checked_add(threads.checked_mul(value_dim)?)?
            .checked_add(threads.checked_mul(2)?)?
            .checked_add(head_dim)?
            .checked_add(value_dim)?;
        floats.checked_mul(std::mem::size_of::<f32>())
    }

    /// Whether the fused single-kernel path can run these head shapes with
    /// the function's dynamic shared-memory limit.
    pub fn fused_available(&self, head_dim: usize, value_dim: usize) -> bool {
        let Some(required) = self.fused_shared_memory_bytes(head_dim, value_dim) else {
            return false;
        };
        let Ok(attributes) = self.fused.attributes() else {
            return false;
        };
        required <= attributes.max_dynamic_shared_memory_bytes as usize
    }

    /// Fused scaled dot-product attention for one logical head and wait for
    /// completion.
    ///
    /// Shapes: `queries` holds `query_rows * head_dim`, `keys` holds
    /// `kv_rows * head_dim`, `values` holds `kv_rows * value_dim`, and
    /// `output` receives `query_rows * value_dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_fused(
        &self,
        stream: &Stream,
        queries: &DeviceBuffer<f32>,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
        scale: f32,
        mask: AttentionMask,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe {
            self.enqueue_attention_fused(
                stream, queries, keys, values, output, query_rows, head_dim, kv_rows, value_dim,
                scale, mask,
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

    /// Enqueue the fused attention without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, and this kernel must remain alive and
    /// otherwise untouched until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_attention_fused(
        &self,
        stream: &Stream,
        queries: &DeviceBuffer<f32>,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
        scale: f32,
        mask: AttentionMask,
    ) -> Result<()> {
        self.validate_execution(
            stream, queries, keys, values, output, query_rows, head_dim, kv_rows, value_dim,
        )?;
        if mask == AttentionMask::Causal && query_rows != kv_rows {
            return Err(NnisError::invalid_input(format!(
                "causal attention requires query_rows == kv_rows; got {query_rows} vs {kv_rows}"
            )));
        }
        if !self.fused_available(head_dim, value_dim) {
            return Err(NnisError::invalid_input(format!(
                "attention fused path needs {} shared-memory bytes for \
                 head ({head_dim}, {value_dim}) at block {}",
                self.fused_shared_memory_bytes(head_dim, value_dim)
                    .unwrap_or_default(),
                self.block_size
            )));
        }
        if query_rows == 0 {
            return Ok(());
        }
        let shared_memory_bytes = self
            .fused_shared_memory_bytes(head_dim, value_dim)
            .expect("validated above");
        let grid_x = u32::try_from(query_rows)
            .map_err(|_| NnisError::invalid_input("attention exceeds u32::MAX query rows"))?;
        let (query_rows, head_dim, kv_rows, value_dim) = (
            u64::try_from(query_rows)
                .map_err(|_| NnisError::invalid_input("attention rows exceed u64"))?,
            u64::try_from(head_dim)
                .map_err(|_| NnisError::invalid_input("attention head dim exceeds u64"))?,
            u64::try_from(kv_rows)
                .map_err(|_| NnisError::invalid_input("attention kv rows exceed u64"))?,
            u64::try_from(value_dim)
                .map_err(|_| NnisError::invalid_input("attention value dim exceeds u64"))?,
        );
        let config = LaunchConfig::new(Dim3::x(grid_x), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(
                u32::try_from(shared_memory_bytes)
                    .map_err(|_| NnisError::invalid_input("attention shared memory exceeds u32"))?,
            );
        let mut arguments = KernelArgs::with_capacity(10, 4);
        arguments
            .push_buffer(queries)
            .push_buffer(keys)
            .push_buffer(values)
            .push_buffer(output)
            .push(query_rows)
            .push(head_dim)
            .push(kv_rows)
            .push(value_dim)
            .push(scale)
            .push(u32::from(mask == AttentionMask::Causal));
        let launch = KernelLaunch::new(&self.fused, stream, config);
        // SAFETY: argument order/widths match `nnis_attention_fused_f32`;
        // the caller owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    /// Fused scaled dot-product attention over packed multi-head inputs and
    /// wait for completion.
    ///
    /// Layout: every buffer holds `[heads][rows][dim]` contiguously -
    /// `queries` holds `num_heads * query_rows * head_dim`, `keys` holds
    /// `num_heads * kv_rows * head_dim`, `values` holds
    /// `num_heads * kv_rows * value_dim`, and `output` receives
    /// `num_heads * query_rows * value_dim`. One launch covers every head;
    /// each block owns one (head, query row) pair with a trajectory
    /// identical to [`Self::attention_fused`].
    #[allow(clippy::too_many_arguments)]
    pub fn attention_fused_multihead(
        &self,
        stream: &Stream,
        queries: &DeviceBuffer<f32>,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        num_heads: usize,
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
        scale: f32,
        mask: AttentionMask,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe {
            self.enqueue_attention_fused_multihead(
                stream, queries, keys, values, output, num_heads, query_rows, head_dim, kv_rows,
                value_dim, scale, mask,
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

    /// Enqueue the multi-head fused attention without synchronizing the
    /// stream.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, and this kernel must remain alive and
    /// otherwise untouched until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_attention_fused_multihead(
        &self,
        stream: &Stream,
        queries: &DeviceBuffer<f32>,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        num_heads: usize,
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
        scale: f32,
        mask: AttentionMask,
    ) -> Result<()> {
        self.validate_multihead_execution(
            stream, queries, keys, values, output, num_heads, query_rows, head_dim, kv_rows,
            value_dim,
        )?;
        if mask == AttentionMask::Causal && query_rows != kv_rows {
            return Err(NnisError::invalid_input(format!(
                "causal attention requires query_rows == kv_rows; got {query_rows} vs {kv_rows}"
            )));
        }
        if !self.fused_available(head_dim, value_dim) {
            return Err(NnisError::invalid_input(format!(
                "attention fused path needs {} shared-memory bytes for \
                 head ({head_dim}, {value_dim}) at block {}",
                self.fused_shared_memory_bytes(head_dim, value_dim)
                    .unwrap_or_default(),
                self.block_size
            )));
        }
        if query_rows == 0 {
            return Ok(());
        }
        let total_blocks = num_heads
            .checked_mul(query_rows)
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let grid_x = u32::try_from(total_blocks).map_err(|_| {
            NnisError::invalid_input("attention exceeds u32::MAX head-by-query blocks")
        })?;
        let shared_memory_bytes = self
            .fused_shared_memory_bytes(head_dim, value_dim)
            .expect("validated above");
        let (heads, query_rows, head_dim, kv_rows, value_dim) = (
            u32::try_from(num_heads)
                .map_err(|_| NnisError::invalid_input("attention heads exceed u32"))?,
            u64::try_from(query_rows)
                .map_err(|_| NnisError::invalid_input("attention rows exceed u64"))?,
            u64::try_from(head_dim)
                .map_err(|_| NnisError::invalid_input("attention head dim exceeds u64"))?,
            u64::try_from(kv_rows)
                .map_err(|_| NnisError::invalid_input("attention kv rows exceed u64"))?,
            u64::try_from(value_dim)
                .map_err(|_| NnisError::invalid_input("attention value dim exceeds u64"))?,
        );
        let config = LaunchConfig::new(Dim3::x(grid_x), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(
                u32::try_from(shared_memory_bytes)
                    .map_err(|_| NnisError::invalid_input("attention shared memory exceeds u32"))?,
            );
        let mut arguments = KernelArgs::with_capacity(11, 4);
        arguments
            .push_buffer(queries)
            .push_buffer(keys)
            .push_buffer(values)
            .push_buffer(output)
            .push(heads)
            .push(query_rows)
            .push(head_dim)
            .push(kv_rows)
            .push(value_dim)
            .push(scale)
            .push(u32::from(mask == AttentionMask::Causal));
        let launch = KernelLaunch::new(&self.fused_multihead, stream, config);
        // SAFETY: argument order/widths match
        // `nnis_attention_fused_multihead_f32`; the caller owns the
        // asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    /// Composed scaled dot-product attention over packed multi-head inputs
    /// and wait for completion.
    ///
    /// Layout matches [`Self::attention_fused_multihead`]: every buffer
    /// holds `[heads][rows][dim]` contiguously. One batched pipeline runs
    /// end-to-end - transposed-B GEMM per head via gridDim.z, an in-place
    /// elementwise scale (with causal masking from a packed-layout variant
    /// of the single-head kernel), a row softmax whose rows are
    /// independent so `heads * query_rows` rows run unchanged, and a
    /// batched GEMM against the per-head values. The score scratch is one
    /// buffer instead of two; per-head results are identical to looping
    /// [`Self::attention_composed`], which tests assert bit-for-bit.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_composed_multihead(
        &self,
        gemm: &crate::F32Gemm,
        elementwise: &crate::F32Elementwise,
        softmax_2d: &crate::F32Softmax2D,
        stream: &Stream,
        queries: &DeviceBuffer<f32>,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        num_heads: usize,
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
        scale: f32,
        mask: AttentionMask,
    ) -> Result<()> {
        self.validate_multihead_execution(
            stream, queries, keys, values, output, num_heads, query_rows, head_dim, kv_rows,
            value_dim,
        )?;
        if mask == AttentionMask::Causal && query_rows != kv_rows {
            return Err(NnisError::invalid_input(format!(
                "causal attention requires query_rows == kv_rows; got {query_rows} vs {kv_rows}"
            )));
        }
        if query_rows == 0 {
            return Ok(());
        }
        let score_elements = num_heads
            .checked_mul(query_rows)
            .and_then(|rows| rows.checked_mul(kv_rows))
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let context = stream.ctx();
        let probabilities = DeviceBuffer::<f32>::new(context, score_elements)?;

        gemm.gemm_transposed_b_batched(
            stream,
            queries,
            keys,
            &probabilities,
            num_heads,
            query_rows,
            kv_rows,
            head_dim,
        )?;
        if mask == AttentionMask::Causal {
            // SAFETY: the probability borrow remains live through the
            // synchronizing call below; each index is written from its own
            // read.
            unsafe {
                self.enqueue_scale_causal_multihead(
                    stream,
                    &probabilities,
                    &probabilities,
                    score_elements,
                    query_rows,
                    kv_rows,
                    scale,
                )?;
            }
            stream.synchronize()?;
        } else {
            // In-place scaling is safe: each element is read and written by
            // exactly one thread at its own index.
            elementwise.scale(stream, &probabilities, &probabilities, scale)?;
        }
        softmax_2d.softmax_rows_dispatched(
            stream,
            &probabilities,
            &probabilities,
            num_heads * query_rows,
            kv_rows,
        )?;
        gemm.gemm_batched(
            stream,
            &probabilities,
            values,
            output,
            num_heads,
            query_rows,
            value_dim,
            kv_rows,
        )
    }

    /// Enqueue the fused scale-and-causal-mask pass over a packed
    /// multi-head score matrix without synchronizing the stream. Positions
    /// above the within-head diagonal become `-infinity`; others are
    /// scaled.
    ///
    /// # Safety
    ///
    /// The stream, both buffers, and this kernel family must remain alive
    /// and otherwise untouched until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_scale_causal_multihead(
        &self,
        stream: &Stream,
        scores: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        elements: usize,
        query_rows: usize,
        kv_rows: usize,
        scale: f32,
    ) -> Result<()> {
        if scores.len() != elements || output.len() != elements {
            return Err(NnisError::invalid_input(format!(
                "attention multi-head scale-causal buffers have {}/{} elements; \
                 requires {elements}",
                scores.len(),
                output.len()
            )));
        }
        let context = self.fused.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, scores.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "attention multi-head scale-causal stream and buffers must share one context",
            ));
        }
        if elements == 0 {
            return Ok(());
        }
        let total_elements = elements;
        let (elements, query_rows, kv_rows) = (
            u64::try_from(elements)
                .map_err(|_| NnisError::invalid_input("attention elements exceed u64"))?,
            u64::try_from(query_rows)
                .map_err(|_| NnisError::invalid_input("attention rows exceed u64"))?,
            u64::try_from(kv_rows)
                .map_err(|_| NnisError::invalid_input("attention kv rows exceed u64"))?,
        );
        let config = LaunchConfig::for_num_elements(total_elements, CAUSAL_BLOCK_SIZE)?;
        let mut arguments = KernelArgs::with_capacity(6, 2);
        arguments
            .push_buffer(scores)
            .push_buffer(output)
            .push(elements)
            .push(query_rows)
            .push(kv_rows)
            .push(scale);
        let launch = KernelLaunch::new(&self.scale_causal_multihead, stream, config);
        // SAFETY: argument order/widths match
        // `nnis_attention_scale_causal_multihead_f32`; the caller owns the
        // asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    /// Enqueue the fused scale-and-causal-mask pass over a score matrix.
    /// Positions above the diagonal become `-infinity`; others are scaled.
    ///
    /// # Safety
    ///
    /// The stream, both buffers, and this kernel family must remain alive
    /// and otherwise untouched until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_scale_causal(
        &self,
        stream: &Stream,
        scores: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        query_rows: usize,
        kv_rows: usize,
        scale: f32,
    ) -> Result<()> {
        let elements = query_rows
            .checked_mul(kv_rows)
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        if scores.len() != elements || output.len() != elements {
            return Err(NnisError::invalid_input(format!(
                "attention scale-causal buffers have {}/{} elements; \
                 shape ({query_rows}, {kv_rows}) requires {elements}",
                scores.len(),
                output.len()
            )));
        }
        let context = self.fused.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, scores.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "attention scale-causal stream and buffers must share one context",
            ));
        }
        if elements == 0 {
            return Ok(());
        }
        let (query_rows, kv_rows) = (
            u64::try_from(query_rows)
                .map_err(|_| NnisError::invalid_input("attention rows exceed u64"))?,
            u64::try_from(kv_rows)
                .map_err(|_| NnisError::invalid_input("attention kv rows exceed u64"))?,
        );
        let config = LaunchConfig::for_num_elements(elements, CAUSAL_BLOCK_SIZE)?;
        let mut arguments = KernelArgs::with_capacity(5, 2);
        arguments
            .push_buffer(scores)
            .push_buffer(output)
            .push(query_rows)
            .push(kv_rows)
            .push(scale);
        let launch = KernelLaunch::new(&self.scale_causal, stream, config);
        // SAFETY: argument order/widths match `nnis_attention_scale_causal_f32`;
        // the caller owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_execution(
        &self,
        stream: &Stream,
        queries: &DeviceBuffer<f32>,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
    ) -> Result<()> {
        if head_dim == 0 || kv_rows == 0 || value_dim == 0 {
            return Err(NnisError::invalid_input(format!(
                "attention requires non-empty keys/head/value dimensions; \
                 got head_dim={head_dim}, kv_rows={kv_rows}, value_dim={value_dim}"
            )));
        }
        let expected_q = query_rows
            .checked_mul(head_dim)
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let expected_k = kv_rows
            .checked_mul(head_dim)
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let expected_v = kv_rows
            .checked_mul(value_dim)
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let expected_o = query_rows
            .checked_mul(value_dim)
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        if queries.len() != expected_q {
            return Err(NnisError::invalid_input(format!(
                "attention queries have {} elements; shape ({query_rows}, {head_dim}) \
                 requires {expected_q}",
                queries.len()
            )));
        }
        if keys.len() != expected_k {
            return Err(NnisError::invalid_input(format!(
                "attention keys have {} elements; shape ({kv_rows}, {head_dim}) \
                 requires {expected_k}",
                keys.len()
            )));
        }
        if values.len() != expected_v {
            return Err(NnisError::invalid_input(format!(
                "attention values have {} elements; shape ({kv_rows}, {value_dim}) \
                 requires {expected_v}",
                values.len()
            )));
        }
        if output.len() != expected_o {
            return Err(NnisError::invalid_input(format!(
                "attention output has {} elements; shape ({query_rows}, {value_dim}) \
                 requires {expected_o}",
                output.len()
            )));
        }
        let context = self.fused.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, queries.ctx())
            || !Arc::ptr_eq(context, keys.ctx())
            || !Arc::ptr_eq(context, values.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "attention stream, buffers, and kernel must share one context",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_multihead_execution(
        &self,
        stream: &Stream,
        queries: &DeviceBuffer<f32>,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        num_heads: usize,
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
    ) -> Result<()> {
        if num_heads == 0 {
            return Err(NnisError::invalid_input(
                "attention requires at least one head",
            ));
        }
        let expected_q = num_heads
            .checked_mul(query_rows)
            .and_then(|elements| elements.checked_mul(head_dim))
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let expected_k = num_heads
            .checked_mul(kv_rows)
            .and_then(|elements| elements.checked_mul(head_dim))
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let expected_v = num_heads
            .checked_mul(kv_rows)
            .and_then(|elements| elements.checked_mul(value_dim))
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let expected_o = num_heads
            .checked_mul(query_rows)
            .and_then(|elements| elements.checked_mul(value_dim))
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        if queries.len() != expected_q {
            return Err(NnisError::invalid_input(format!(
                "attention queries have {} elements; {} heads of shape ({query_rows}, {head_dim}) \
                 requires {expected_q}",
                queries.len(),
                num_heads
            )));
        }
        if keys.len() != expected_k {
            return Err(NnisError::invalid_input(format!(
                "attention keys have {} elements; {num_heads} heads of shape \
                 ({kv_rows}, {head_dim}) requires {expected_k}",
                keys.len()
            )));
        }
        if values.len() != expected_v {
            return Err(NnisError::invalid_input(format!(
                "attention values have {} elements; {num_heads} heads of shape \
                 ({kv_rows}, {value_dim}) requires {expected_v}",
                values.len()
            )));
        }
        if output.len() != expected_o {
            return Err(NnisError::invalid_input(format!(
                "attention output has {} elements; {num_heads} heads of shape \
                 ({query_rows}, {value_dim}) requires {expected_o}",
                output.len()
            )));
        }
        let context = self.fused.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, queries.ctx())
            || !Arc::ptr_eq(context, keys.ctx())
            || !Arc::ptr_eq(context, values.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "attention stream, buffers, and kernel must share one context",
            ));
        }
        Ok(())
    }

    /// Composed attention through the existing kernel families and wait for
    /// completion: transposed-B GEMM for the scores, elementwise scaling,
    /// dispatched row softmax, and a plain GEMM against `values`.
    ///
    /// Two score-sized scratch buffers are allocated per call; prefer the
    /// fused path when [`Self::fused_available`] returns true.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_composed(
        &self,
        gemm: &crate::F32Gemm,
        elementwise: &crate::F32Elementwise,
        softmax_2d: &crate::F32Softmax2D,
        stream: &Stream,
        queries: &DeviceBuffer<f32>,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
        scale: f32,
        mask: AttentionMask,
    ) -> Result<()> {
        if head_dim == 0 || kv_rows == 0 || value_dim == 0 {
            return Err(NnisError::invalid_input(format!(
                "attention requires non-empty keys/head/value dimensions; \
                 got head_dim={head_dim}, kv_rows={kv_rows}, value_dim={value_dim}"
            )));
        }
        if mask == AttentionMask::Causal && query_rows != kv_rows {
            return Err(NnisError::invalid_input(format!(
                "causal attention requires query_rows == kv_rows; got {query_rows} vs {kv_rows}"
            )));
        }
        let score_elements = query_rows
            .checked_mul(kv_rows)
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let expected_q = query_rows
            .checked_mul(head_dim)
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let expected_k = kv_rows
            .checked_mul(head_dim)
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let expected_v = kv_rows
            .checked_mul(value_dim)
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        let expected_o = query_rows
            .checked_mul(value_dim)
            .ok_or_else(|| NnisError::invalid_input("attention shape overflows usize"))?;
        if queries.len() != expected_q
            || keys.len() != expected_k
            || values.len() != expected_v
            || output.len() != expected_o
        {
            return Err(NnisError::invalid_input(format!(
                "attention shapes require queries={expected_q}, keys={expected_k}, \
                 values={expected_v}, output={expected_o}"
            )));
        }
        let context = stream.ctx();
        let probabilities = DeviceBuffer::<f32>::new(context, score_elements)?;
        let scores = DeviceBuffer::<f32>::new(context, score_elements)?;

        gemm.gemm_transposed_b(
            stream, queries, keys, &scores, query_rows, kv_rows, head_dim,
        )?;
        if mask == AttentionMask::Causal {
            // SAFETY: scratch borrows remain live through the later
            // synchronizing calls on this method's path.
            unsafe {
                self.enqueue_scale_causal(
                    stream,
                    &scores,
                    &probabilities,
                    query_rows,
                    kv_rows,
                    scale,
                )?;
            }
        } else {
            elementwise.scale(stream, &scores, &probabilities, scale)?;
        }
        softmax_2d.softmax_rows_dispatched(
            stream,
            &probabilities,
            &probabilities,
            query_rows,
            kv_rows,
        )?;
        gemm.gemm(
            stream,
            &probabilities,
            values,
            output,
            query_rows,
            value_dim,
            kv_rows,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{F32Elementwise, F32Gemm, F32Softmax2D};
    use nnis_rt::gpu_context;

    /// (query_rows, head_dim, kv_rows, value_dim); all fit the fused path
    /// at the default block width except where a test wants rejection.
    const SHAPES: &[(usize, usize, usize, usize)] = &[
        (1, 16, 3, 16),
        (2, 64, 64, 64),
        (5, 64, 200, 64),
        (17, 32, 70, 96),
        (33, 48, 129, 48),
    ];

    fn host_values(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| (((index * 13 % 97) as f32 - 48.0) * 0.0625) + ((index % 5) as f32 - 2.0))
            .collect()
    }

    /// f64 attention reference; scores share the kernel's ascending-dot
    /// order and the softmax uses one shared maximum per query row.
    #[allow(clippy::too_many_arguments)]
    fn reference_attention(
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
        scale: f64,
        causal: bool,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; query_rows * value_dim];
        for row in 0..query_rows {
            let mut scores = vec![0.0_f64; kv_rows];
            for key in 0..kv_rows {
                if causal && key > row {
                    break;
                }
                let score: f64 = (0..head_dim)
                    .map(|e| {
                        f64::from(queries[row * head_dim + e]) * f64::from(keys[key * head_dim + e])
                    })
                    .sum();
                scores[key] = score * scale;
            }
            let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let total: f64 = weights.iter().sum();
            for col in 0..value_dim {
                let value: f64 = (0..kv_rows)
                    .map(|key| weights[key] * f64::from(values[key * value_dim + col]))
                    .sum();
                output[row * value_dim + col] = (value / total) as f32;
            }
        }
        output
    }

    fn assert_close(actual: &[f32], expected: &[f32], context: &str) {
        assert_eq!(actual.len(), expected.len());
        for index in 0..actual.len() {
            let difference = (f64::from(actual[index]) - f64::from(expected[index])).abs();
            let tolerance = 1.0e-4_f64.max(f64::from(expected[index].abs()) * 2.0e-4);
            assert!(
                difference <= tolerance,
                "{context} mismatch at {index}: {} vs {}, tolerance {tolerance}",
                actual[index],
                expected[index]
            );
        }
    }

    #[test]
    fn attention_fused_matches_f64_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = F32Attention::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(query_rows, head_dim, kv_rows, value_dim) in SHAPES {
            let queries_host = host_values(query_rows * head_dim);
            let keys_host = host_values(kv_rows * head_dim);
            let values_host = host_values(kv_rows * value_dim);
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_host).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_host).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
            // Pre-fill so a skipped kernel cannot pass silently.
            let output_host = vec![f32::NAN; query_rows * value_dim];
            let output = DeviceBuffer::from_host(&context, &stream, &output_host).unwrap();

            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            attention
                .attention_fused(
                    &stream,
                    &queries,
                    &keys,
                    &values,
                    &output,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    scale,
                    AttentionMask::None,
                )
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            let expected = reference_attention(
                &queries_host,
                &keys_host,
                &values_host,
                query_rows,
                head_dim,
                kv_rows,
                value_dim,
                f64::from(scale),
                false,
            );
            assert_close(
                &actual,
                &expected,
                &format!("fused ({query_rows},{head_dim},{kv_rows},{value_dim})"),
            );
        }
    }

    #[test]
    fn attention_composed_matches_f64_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = F32Attention::load(&context, &compiler).unwrap();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let elementwise = F32Elementwise::load(&context, &compiler).unwrap();
        let softmax_2d = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(query_rows, head_dim, kv_rows, value_dim) in SHAPES {
            let queries_host = host_values(query_rows * head_dim);
            let keys_host = host_values(kv_rows * head_dim);
            let values_host = host_values(kv_rows * value_dim);
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_host).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_host).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
            let output_host = vec![f32::NAN; query_rows * value_dim];
            let output = DeviceBuffer::from_host(&context, &stream, &output_host).unwrap();

            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            attention
                .attention_composed(
                    &gemm,
                    &elementwise,
                    &softmax_2d,
                    &stream,
                    &queries,
                    &keys,
                    &values,
                    &output,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    scale,
                    AttentionMask::None,
                )
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            let expected = reference_attention(
                &queries_host,
                &keys_host,
                &values_host,
                query_rows,
                head_dim,
                kv_rows,
                value_dim,
                f64::from(scale),
                false,
            );
            assert_close(
                &actual,
                &expected,
                &format!("composed ({query_rows},{head_dim},{kv_rows},{value_dim})"),
            );
        }
    }

    #[test]
    fn attention_rejects_invalid_shapes_and_oversized_heads_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = F32Attention::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        // Zero-length key dimension is outside the contract.
        let queries = DeviceBuffer::<f32>::new(&context, 8).unwrap(); // 1 x 8
        let empty_kv = DeviceBuffer::<f32>::new(&context, 0).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 8).unwrap();
        let error = attention
            .attention_fused(
                &stream,
                &queries,
                &empty_kv,
                &empty_kv,
                &output,
                1,
                8,
                0,
                8,
                0.25,
                AttentionMask::None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("non-empty"), "{error}");

        // Short key buffer.
        let queries = DeviceBuffer::<f32>::new(&context, 64).unwrap(); // 1 x 64
        let short_keys = DeviceBuffer::<f32>::new(&context, 63).unwrap(); // needs 128
        let values = DeviceBuffer::<f32>::new(&context, 256).unwrap(); // 2 x 128
        let error = attention
            .attention_fused(
                &stream,
                &queries,
                &short_keys,
                &values,
                &output,
                1,
                64,
                2,
                8,
                0.125,
                AttentionMask::None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("requires 128"), "{error}");

        // Head shapes beyond the dynamic shared-memory limit are rejected
        // before any launch.
        assert!(!attention.fused_available(4096, 4096));
        let big_queries = DeviceBuffer::<f32>::new(&context, 4096).unwrap(); // 1 x 4096
        let big_keys = DeviceBuffer::<f32>::new(&context, 32_768).unwrap(); // 8 x 4096
        let values = DeviceBuffer::<f32>::new(&context, 64).unwrap(); // 8 x 8
        let small_out = DeviceBuffer::<f32>::new(&context, 8).unwrap(); // 1 x 8
        let error = attention
            .attention_fused(
                &stream,
                &big_queries,
                &big_keys,
                &values,
                &small_out,
                1,
                4096,
                8,
                8,
                0.03125,
                AttentionMask::None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("shared-memory"), "{error}");
    }

    #[test]
    fn attention_causal_matches_f64_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = F32Attention::load(&context, &compiler).unwrap();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let elementwise = F32Elementwise::load(&context, &compiler).unwrap();
        let softmax_2d = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        // Square score shapes only; includes shapes spanning many chunks.
        const CAUSAL_SHAPES: &[(usize, usize)] = &[(1, 16), (7, 64), (33, 48), (70, 32)];

        for &(kv_rows, head_dim) in CAUSAL_SHAPES {
            let query_rows = kv_rows;
            let value_dim = head_dim;
            let queries_host = host_values(query_rows * head_dim);
            let keys_host = host_values(kv_rows * head_dim);
            let values_host = host_values(kv_rows * value_dim);
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_host).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_host).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            for mask in [AttentionMask::Causal] {
                // Pre-fill so a skipped kernel cannot pass silently.
                let poisoned = vec![f32::NAN; query_rows * value_dim];
                let fused_output =
                    DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();
                let composed_output =
                    DeviceBuffer::from_host(&context, &stream, &poisoned).unwrap();

                attention
                    .attention_fused(
                        &stream,
                        &queries,
                        &keys,
                        &values,
                        &fused_output,
                        query_rows,
                        head_dim,
                        kv_rows,
                        value_dim,
                        scale,
                        mask,
                    )
                    .unwrap();
                attention
                    .attention_composed(
                        &gemm,
                        &elementwise,
                        &softmax_2d,
                        &stream,
                        &queries,
                        &keys,
                        &values,
                        &composed_output,
                        query_rows,
                        head_dim,
                        kv_rows,
                        value_dim,
                        scale,
                        mask,
                    )
                    .unwrap();

                let expected = reference_attention(
                    &queries_host,
                    &keys_host,
                    &values_host,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    f64::from(scale),
                    true,
                );
                for (name, buffer) in [("fused", &fused_output), ("composed", &composed_output)] {
                    let actual = buffer.to_vec(&stream).unwrap();
                    assert_close(
                        &actual,
                        &expected,
                        &format!("causal {name} ({query_rows},{head_dim})"),
                    );
                    // Causal positions must be exactly zero-weight: row r may
                    // only be a convex mix of value rows 0..=r.
                    let _ = name;
                }
            }
        }
    }

    #[test]
    fn attention_rejects_causal_on_rectangular_scores_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = F32Attention::load(&context, &compiler).unwrap();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let elementwise = F32Elementwise::load(&context, &compiler).unwrap();
        let softmax_2d = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let queries = DeviceBuffer::<f32>::new(&context, 2 * 8).unwrap();
        let keys = DeviceBuffer::<f32>::new(&context, 5 * 8).unwrap();
        let values = DeviceBuffer::<f32>::new(&context, 5 * 4).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 2 * 4).unwrap();

        let error = attention
            .attention_fused(
                &stream,
                &queries,
                &keys,
                &values,
                &output,
                2,
                8,
                5,
                4,
                0.25,
                AttentionMask::Causal,
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("query_rows == kv_rows"),
            "{error}"
        );

        let error = attention
            .attention_composed(
                &gemm,
                &elementwise,
                &softmax_2d,
                &stream,
                &queries,
                &keys,
                &values,
                &output,
                2,
                8,
                5,
                4,
                0.25,
                AttentionMask::Causal,
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("query_rows == kv_rows"),
            "{error}"
        );
    }

    /// (heads, query_rows, head_dim, kv_rows, value_dim).
    const MULTIHEAD_SHAPES: &[(usize, usize, usize, usize, usize)] = &[
        (1, 1, 16, 3, 16),
        (2, 5, 64, 200, 64),
        (3, 17, 32, 70, 96),
        (4, 33, 48, 129, 48),
    ];

    /// Per-head host data with distinct seeds so cross-head mixing cannot
    /// pass silently.
    fn multihead_values(heads: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                (((index * 13 % 97) as f32 - 48.0) * 0.0625)
                    + (((index + heads * 11) % 5) as f32 - 2.0)
            })
            .collect()
    }

    #[test]
    fn attention_multihead_bit_matches_per_head_fused_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = F32Attention::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(heads, query_rows, head_dim, kv_rows, value_dim) in MULTIHEAD_SHAPES {
            // Interleave per-head hosts into the packed [heads][rows][dim]
            // layout.
            let mut queries_host = Vec::with_capacity(heads * query_rows * head_dim);
            let mut keys_host = Vec::with_capacity(heads * kv_rows * head_dim);
            let mut values_host = Vec::with_capacity(heads * kv_rows * value_dim);
            for head in 0..heads {
                queries_host.extend(multihead_values(head, query_rows * head_dim));
                keys_host.extend(multihead_values(head + 97, kv_rows * head_dim));
                values_host.extend(multihead_values(head + 193, kv_rows * value_dim));
            }
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_host).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_host).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
            // Pre-fill so a skipped kernel cannot pass silently.
            let poisoned = vec![f32::NAN; heads * query_rows * value_dim];
            let batched_output =
                DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();

            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            attention
                .attention_fused_multihead(
                    &stream,
                    &queries,
                    &keys,
                    &values,
                    &batched_output,
                    heads,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    scale,
                    AttentionMask::None,
                )
                .unwrap();
            let batched_actual = batched_output.to_vec(&stream).unwrap();

            for head in 0..heads {
                let head_queries = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head, query_rows * head_dim),
                )
                .unwrap();
                let head_keys = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head + 97, kv_rows * head_dim),
                )
                .unwrap();
                let head_values = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head + 193, kv_rows * value_dim),
                )
                .unwrap();
                let head_poisoned = vec![f32::NAN; query_rows * value_dim];
                let head_output =
                    DeviceBuffer::from_host(&context, &stream, &head_poisoned).unwrap();
                attention
                    .attention_fused(
                        &stream,
                        &head_queries,
                        &head_keys,
                        &head_values,
                        &head_output,
                        query_rows,
                        head_dim,
                        kv_rows,
                        value_dim,
                        scale,
                        AttentionMask::None,
                    )
                    .unwrap();
                let head_actual = head_output.to_vec(&stream).unwrap();
                let range = head * query_rows * value_dim..(head + 1) * query_rows * value_dim;
                for (index, actual) in batched_actual[range].iter().enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        head_actual[index].to_bits(),
                        "multihead {heads}h bit mismatch at head {head} element {index} \
                         shape ({query_rows},{head_dim},{kv_rows},{value_dim})"
                    );
                }
            }
        }
    }

    #[test]
    fn attention_multihead_causal_bit_matches_per_head_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = F32Attention::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        const CAUSAL_HEADS_SHAPES: &[(usize, usize, usize)] =
            &[(2, 7, 64), (3, 33, 48), (4, 70, 32)];

        for &(heads, rows, head_dim) in CAUSAL_HEADS_SHAPES {
            let value_dim = head_dim;
            let mut queries_host = Vec::with_capacity(heads * rows * head_dim);
            let mut keys_host = Vec::with_capacity(heads * rows * head_dim);
            let mut values_host = Vec::with_capacity(heads * rows * value_dim);
            for head in 0..heads {
                queries_host.extend(multihead_values(head, rows * head_dim));
                keys_host.extend(multihead_values(head + 97, rows * head_dim));
                values_host.extend(multihead_values(head + 193, rows * value_dim));
            }
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_host).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_host).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
            let poisoned = vec![f32::NAN; heads * rows * value_dim];
            let batched_output =
                DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            attention
                .attention_fused_multihead(
                    &stream,
                    &queries,
                    &keys,
                    &values,
                    &batched_output,
                    heads,
                    rows,
                    head_dim,
                    rows,
                    value_dim,
                    scale,
                    AttentionMask::Causal,
                )
                .unwrap();
            let batched_actual = batched_output.to_vec(&stream).unwrap();

            for head in 0..heads {
                let head_queries = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head, rows * head_dim),
                )
                .unwrap();
                let head_keys = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head + 97, rows * head_dim),
                )
                .unwrap();
                let head_values = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head + 193, rows * value_dim),
                )
                .unwrap();
                let head_poisoned = vec![f32::NAN; rows * value_dim];
                let head_output =
                    DeviceBuffer::from_host(&context, &stream, &head_poisoned).unwrap();
                attention
                    .attention_fused(
                        &stream,
                        &head_queries,
                        &head_keys,
                        &head_values,
                        &head_output,
                        rows,
                        head_dim,
                        rows,
                        value_dim,
                        scale,
                        AttentionMask::Causal,
                    )
                    .unwrap();
                let head_actual = head_output.to_vec(&stream).unwrap();
                let range = head * rows * value_dim..(head + 1) * rows * value_dim;
                for (index, actual) in batched_actual[range].iter().enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        head_actual[index].to_bits(),
                        "causal multihead {heads}h bit mismatch at head {head} element {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn attention_composed_multihead_bit_matches_per_head_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = F32Attention::load(&context, &compiler).unwrap();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let elementwise = F32Elementwise::load(&context, &compiler).unwrap();
        let softmax_2d = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(heads, query_rows, head_dim, kv_rows, value_dim) in MULTIHEAD_SHAPES {
            let mut queries_host = Vec::with_capacity(heads * query_rows * head_dim);
            let mut keys_host = Vec::with_capacity(heads * kv_rows * head_dim);
            let mut values_host = Vec::with_capacity(heads * kv_rows * value_dim);
            for head in 0..heads {
                queries_host.extend(multihead_values(head, query_rows * head_dim));
                keys_host.extend(multihead_values(head + 97, kv_rows * head_dim));
                values_host.extend(multihead_values(head + 193, kv_rows * value_dim));
            }
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_host).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_host).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
            let poisoned = vec![f32::NAN; heads * query_rows * value_dim];
            let batched_output =
                DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();

            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            attention
                .attention_composed_multihead(
                    &gemm,
                    &elementwise,
                    &softmax_2d,
                    &stream,
                    &queries,
                    &keys,
                    &values,
                    &batched_output,
                    heads,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    scale,
                    AttentionMask::None,
                )
                .unwrap();
            let batched_actual = batched_output.to_vec(&stream).unwrap();

            for head in 0..heads {
                let head_queries = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head, query_rows * head_dim),
                )
                .unwrap();
                let head_keys = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head + 97, kv_rows * head_dim),
                )
                .unwrap();
                let head_values = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head + 193, kv_rows * value_dim),
                )
                .unwrap();
                let head_output = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &vec![f32::NAN; query_rows * value_dim],
                )
                .unwrap();
                attention
                    .attention_composed(
                        &gemm,
                        &elementwise,
                        &softmax_2d,
                        &stream,
                        &head_queries,
                        &head_keys,
                        &head_values,
                        &head_output,
                        query_rows,
                        head_dim,
                        kv_rows,
                        value_dim,
                        scale,
                        AttentionMask::None,
                    )
                    .unwrap();
                let head_actual = head_output.to_vec(&stream).unwrap();
                let range = head * query_rows * value_dim..(head + 1) * query_rows * value_dim;
                for index in 0..query_rows * value_dim {
                    assert_eq!(
                        batched_actual[range.start + index].to_bits(),
                        head_actual[index].to_bits(),
                        "composed multihead {heads}h bit mismatch at head {head} element \
                         {index} shape ({query_rows},{head_dim},{kv_rows},{value_dim})"
                    );
                }
            }
        }

        // Causal square shapes: same bit-exact contract.
        const CAUSAL_HEADS_SHAPES: &[(usize, usize, usize)] = &[(2, 7, 64), (3, 33, 48)];
        for &(heads, rows, head_dim) in CAUSAL_HEADS_SHAPES {
            let value_dim = head_dim;
            let mut queries_host = Vec::with_capacity(heads * rows * head_dim);
            let mut keys_host = Vec::with_capacity(heads * rows * head_dim);
            let mut values_host = Vec::with_capacity(heads * rows * value_dim);
            for head in 0..heads {
                queries_host.extend(multihead_values(head, rows * head_dim));
                keys_host.extend(multihead_values(head + 97, rows * head_dim));
                values_host.extend(multihead_values(head + 193, rows * value_dim));
            }
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_host).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_host).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
            let poisoned = vec![f32::NAN; heads * rows * value_dim];
            let batched_output =
                DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            attention
                .attention_composed_multihead(
                    &gemm,
                    &elementwise,
                    &softmax_2d,
                    &stream,
                    &queries,
                    &keys,
                    &values,
                    &batched_output,
                    heads,
                    rows,
                    head_dim,
                    rows,
                    value_dim,
                    scale,
                    AttentionMask::Causal,
                )
                .unwrap();
            let batched_actual = batched_output.to_vec(&stream).unwrap();

            for head in 0..heads {
                let head_queries = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head, rows * head_dim),
                )
                .unwrap();
                let head_keys = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head + 97, rows * head_dim),
                )
                .unwrap();
                let head_values = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &multihead_values(head + 193, rows * value_dim),
                )
                .unwrap();
                let head_output =
                    DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; rows * value_dim])
                        .unwrap();
                attention
                    .attention_composed(
                        &gemm,
                        &elementwise,
                        &softmax_2d,
                        &stream,
                        &head_queries,
                        &head_keys,
                        &head_values,
                        &head_output,
                        rows,
                        head_dim,
                        rows,
                        value_dim,
                        scale,
                        AttentionMask::Causal,
                    )
                    .unwrap();
                let head_actual = head_output.to_vec(&stream).unwrap();
                let range = head * rows * value_dim..(head + 1) * rows * value_dim;
                for index in 0..rows * value_dim {
                    assert_eq!(
                        batched_actual[range.start + index].to_bits(),
                        head_actual[index].to_bits(),
                        "causal composed multihead {heads}h bit mismatch at head {head} \
                         element {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn attention_multihead_rejects_invalid_shapes_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = F32Attention::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        // Zero heads is outside the contract.
        let queries = DeviceBuffer::<f32>::new(&context, 8).unwrap();
        let keys = DeviceBuffer::<f32>::new(&context, 8).unwrap();
        let values = DeviceBuffer::<f32>::new(&context, 8).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 8).unwrap();
        let error = attention
            .attention_fused_multihead(
                &stream,
                &queries,
                &keys,
                &values,
                &output,
                0,
                1,
                8,
                1,
                8,
                0.25,
                AttentionMask::None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("at least one head"), "{error}");

        // Short key buffer under the packed multi-head contract.
        let error = attention
            .attention_fused_multihead(
                &stream,
                &queries,
                &keys,
                &values,
                &output,
                2,
                1,
                8,
                1,
                8,
                0.25,
                AttentionMask::None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("2 heads"), "{error}");

        // Rectangular causal scores remain rejected.
        let queries = DeviceBuffer::<f32>::new(&context, 2 * 2 * 8).unwrap();
        let keys = DeviceBuffer::<f32>::new(&context, 2 * 5 * 8).unwrap();
        let values = DeviceBuffer::<f32>::new(&context, 2 * 5 * 4).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 2 * 2 * 4).unwrap();
        let error = attention
            .attention_fused_multihead(
                &stream,
                &queries,
                &keys,
                &values,
                &output,
                2,
                2,
                8,
                5,
                4,
                0.25,
                AttentionMask::Causal,
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("query_rows == kv_rows"),
            "{error}"
        );

        // Oversized heads are rejected before launch exactly like the
        // single-head path.
        assert!(!attention.fused_available(4096, 4096));
    }
}
