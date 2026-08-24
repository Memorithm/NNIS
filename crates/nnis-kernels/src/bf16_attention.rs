//! Scaled dot-product attention over packed-bf16 `u16` heads.
//!
//! Numeric policy matches every bf16 family in this crate: bf16 storage,
//! f32 compute. Queries, keys, and values are stored as bf16 bit patterns;
//! the fused kernel widens them with exact bit shifts while staging shared
//! memory and from there follows the [`F32Attention`] fused trajectory
//! verbatim - explicit-FMA score chains, an online running-max/sum softmax,
//! and an f32 output accumulator narrowed once at the final store with
//! round-to-nearest-even. Nothing the size of the score matrix exists in
//! global memory.
//!
//! The composed path reuses validated families end-to-end: exact widening
//! of Q/K/V, the full f32 composed attention (transposed-B GEMM, scaling,
//! dispatched row softmax, plain GEMM), and one final narrowing. Because
//! widening is exact and position independent, this is arithmetically
//! identical to a native path that kept f32 score scratch behind a bf16
//! GEMM-NT, without adding a quantization step in front of the softmax.
//! Materializing bf16 scores instead would inject bf16 rounding noise into
//! every logit before exponentiation; that stricter-storage variant remains
//! available to a downstream project that prefers throughput over fidelity,
//! and Thor measurements (identical medians for f32 and bf16 tiled GEMM,
//! issue-bound) give it no bandwidth justification today.
//!
//! Scores are deterministic f32 chains, but `expf` and the running-maximum
//! rescaling differ from host transcendentals by ulps, so absolute
//! correctness is validated against an f64 oracle inside tolerances that
//! include the final bf16 quantization (half-ulp below 2^-8 relative).
//! Because widening is exact, results must additionally match the f32
//! family evaluated on the same widened inputs BIT-FOR-BIT after one host
//! round-to-nearest-even narrowing; tests assert both properties.

use crate::{
    AttentionMask, Bf16Elementwise, Bf16Gemm, F32Attention, F32Elementwise, F32Gemm, F32Softmax2D,
};
use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const BF16_ATTENTION_SOURCE: &str = r#"
__device__ __forceinline__ float nnis_neg_inf() {
    return __int_as_float(0xff800000u);
}

__device__ __forceinline__ float bf16_bits_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}

__device__ __forceinline__ unsigned short f32_to_bf16_bits(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        // NaN: quiet it and avoid rounding into infinity.
        bits |= 0x00400000u;
        return (unsigned short)(bits >> 16);
    }
    unsigned int lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void nnis_attention_fused_bf16(
    const unsigned short* queries,
    const unsigned short* keys,
    const unsigned short* values,
    unsigned short* output,
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
        q_sh[i] = bf16_bits_to_f32(queries[row * head_dim + i]);
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
                global < kv_rows ? bf16_bits_to_f32(keys[global * head_dim + c]) : 0.0f;
        }
        for (unsigned int idx = tid; idx < threads * value_dim; idx += threads) {
            const unsigned int r = idx / value_dim;
            const unsigned int c = idx % value_dim;
            const unsigned long long global = base + r;
            v_tile[idx] =
                global < kv_rows ? bf16_bits_to_f32(values[global * value_dim + c]) : 0.0f;
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
        output[row * value_dim + c] = f32_to_bf16_bits(acc[c] / running_sum);
    }
}

extern "C" __global__ void nnis_attention_fused_multihead_bf16(
    const unsigned short* queries,
    const unsigned short* keys,
    const unsigned short* values,
    unsigned short* output,
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
    const unsigned short* q_head = queries + head * query_rows * head_dim;
    const unsigned short* k_head = keys + head * kv_rows * head_dim;
    const unsigned short* v_head = values + head * kv_rows * value_dim;
    unsigned short* o_head = output + head * query_rows * value_dim;

    for (unsigned int i = tid; i < head_dim; i += threads) {
        q_sh[i] = bf16_bits_to_f32(q_head[row * head_dim + i]);
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
                global < kv_rows ? bf16_bits_to_f32(k_head[global * head_dim + c]) : 0.0f;
        }
        for (unsigned int idx = tid; idx < threads * value_dim; idx += threads) {
            const unsigned int r = idx / value_dim;
            const unsigned int c = idx % value_dim;
            const unsigned long long global = base + r;
            v_tile[idx] =
                global < kv_rows ? bf16_bits_to_f32(v_head[global * value_dim + c]) : 0.0f;
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
        o_head[row * value_dim + c] = f32_to_bf16_bits(acc[c] / running_sum);
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 64;

/// Context-bound scaled dot-product attention over packed-bf16 heads.
#[derive(Debug)]
pub struct Bf16Attention {
    fused: Kernel,
    fused_multihead: Kernel,
    block_size: u32,
}

impl Bf16Attention {
    /// Compile (or reuse cached CUBIN) and load the bf16 attention kernel.
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
                "bf16 attention block size {block_size} is not a non-zero power of two"
            )));
        }
        let code =
            compiler.compile_cubin(BF16_ATTENTION_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let fused = module.get_function("nnis_attention_fused_bf16")?;
        let fused_multihead = module.get_function("nnis_attention_fused_multihead_bf16")?;
        let attributes = fused.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "bf16 attention block size {block_size} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        Ok(Self {
            fused,
            fused_multihead,
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
    /// `output` receives `query_rows * value_dim`; every buffer holds
    /// packed-bf16 bit patterns.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_fused(
        &self,
        stream: &Stream,
        queries: &DeviceBuffer<u16>,
        keys: &DeviceBuffer<u16>,
        values: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
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
        queries: &DeviceBuffer<u16>,
        keys: &DeviceBuffer<u16>,
        values: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
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
                "bf16 attention fused path needs {} shared-memory bytes for \
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
            .map_err(|_| NnisError::invalid_input("bf16 attention exceeds u32::MAX query rows"))?;
        let (query_rows, head_dim, kv_rows, value_dim) = (
            u64::try_from(query_rows)
                .map_err(|_| NnisError::invalid_input("bf16 attention rows exceed u64"))?,
            u64::try_from(head_dim)
                .map_err(|_| NnisError::invalid_input("bf16 attention head dim exceeds u64"))?,
            u64::try_from(kv_rows)
                .map_err(|_| NnisError::invalid_input("bf16 attention kv rows exceed u64"))?,
            u64::try_from(value_dim)
                .map_err(|_| NnisError::invalid_input("bf16 attention value dim exceeds u64"))?,
        );
        let config = LaunchConfig::new(Dim3::x(grid_x), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(u32::try_from(shared_memory_bytes).map_err(|_| {
                NnisError::invalid_input("bf16 attention shared memory exceeds u32")
            })?);
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
        // SAFETY: argument order/widths match `nnis_attention_fused_bf16`;
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
    /// `num_heads * query_rows * value_dim`; all packed-bf16 bit patterns.
    /// One launch covers every head with a per-head trajectory identical to
    /// [`Self::attention_fused`].
    #[allow(clippy::too_many_arguments)]
    pub fn attention_fused_multihead(
        &self,
        stream: &Stream,
        queries: &DeviceBuffer<u16>,
        keys: &DeviceBuffer<u16>,
        values: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
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

    /// Enqueue the multi-head fused bf16 attention without synchronizing
    /// the stream.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, and this kernel must remain alive and
    /// otherwise untouched until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_attention_fused_multihead(
        &self,
        stream: &Stream,
        queries: &DeviceBuffer<u16>,
        keys: &DeviceBuffer<u16>,
        values: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
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
                "bf16 attention fused path needs {} shared-memory bytes for \
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
            .ok_or_else(|| NnisError::invalid_input("bf16 attention shape overflows usize"))?;
        let grid_x = u32::try_from(total_blocks).map_err(|_| {
            NnisError::invalid_input("bf16 attention exceeds u32::MAX head-by-query blocks")
        })?;
        let shared_memory_bytes = self
            .fused_shared_memory_bytes(head_dim, value_dim)
            .expect("validated above");
        let (heads, query_rows, head_dim, kv_rows, value_dim) = (
            u32::try_from(num_heads)
                .map_err(|_| NnisError::invalid_input("bf16 attention heads exceed u32"))?,
            u64::try_from(query_rows)
                .map_err(|_| NnisError::invalid_input("bf16 attention rows exceed u64"))?,
            u64::try_from(head_dim)
                .map_err(|_| NnisError::invalid_input("bf16 attention head dim exceeds u64"))?,
            u64::try_from(kv_rows)
                .map_err(|_| NnisError::invalid_input("bf16 attention kv rows exceed u64"))?,
            u64::try_from(value_dim)
                .map_err(|_| NnisError::invalid_input("bf16 attention value dim exceeds u64"))?,
        );
        let config = LaunchConfig::new(Dim3::x(grid_x), Dim3::x(self.block_size))
            .with_dynamic_shared_memory(u32::try_from(shared_memory_bytes).map_err(|_| {
                NnisError::invalid_input("bf16 attention shared memory exceeds u32")
            })?);
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
        // `nnis_attention_fused_multihead_bf16`; the caller owns the
        // asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    /// Composed bf16 attention through validated families and wait for
    /// completion: exact widening of every operand, the full f32 composed
    /// attention pipeline, and one final round-to-nearest-even narrowing.
    ///
    /// Four f32 scratch buffers sized like the operands and output are
    /// allocated per call on top of the f32 pipeline's two score-sized
    /// scratches; prefer the fused path when [`Self::fused_available`]
    /// returns true.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_composed(
        &self,
        conversions: &Bf16Elementwise,
        attention: &F32Attention,
        gemm: &F32Gemm,
        elementwise: &F32Elementwise,
        softmax_2d: &F32Softmax2D,
        stream: &Stream,
        queries: &DeviceBuffer<u16>,
        keys: &DeviceBuffer<u16>,
        values: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
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
        if query_rows == 0 {
            return Ok(());
        }
        let context = self.fused.context();
        let wide_queries = DeviceBuffer::<f32>::new(context, query_rows * head_dim)?;
        let wide_keys = DeviceBuffer::<f32>::new(context, kv_rows * head_dim)?;
        let wide_values = DeviceBuffer::<f32>::new(context, kv_rows * value_dim)?;
        let wide_output = DeviceBuffer::<f32>::new(context, query_rows * value_dim)?;

        conversions.widen(stream, queries, &wide_queries)?;
        conversions.widen(stream, keys, &wide_keys)?;
        conversions.widen(stream, values, &wide_values)?;
        attention.attention_composed(
            gemm,
            elementwise,
            softmax_2d,
            stream,
            &wide_queries,
            &wide_keys,
            &wide_values,
            &wide_output,
            query_rows,
            head_dim,
            kv_rows,
            value_dim,
            scale,
            mask,
        )?;
        conversions.narrow(stream, &wide_output, output)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_execution(
        &self,
        stream: &Stream,
        queries: &DeviceBuffer<u16>,
        keys: &DeviceBuffer<u16>,
        values: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
    ) -> Result<()> {
        if head_dim == 0 || kv_rows == 0 || value_dim == 0 {
            return Err(NnisError::invalid_input(format!(
                "bf16 attention requires non-empty keys/head/value dimensions; \
                 got head_dim={head_dim}, kv_rows={kv_rows}, value_dim={value_dim}"
            )));
        }
        let expected_q = query_rows
            .checked_mul(head_dim)
            .ok_or_else(|| NnisError::invalid_input("bf16 attention shape overflows usize"))?;
        let expected_k = kv_rows
            .checked_mul(head_dim)
            .ok_or_else(|| NnisError::invalid_input("bf16 attention shape overflows usize"))?;
        let expected_v = kv_rows
            .checked_mul(value_dim)
            .ok_or_else(|| NnisError::invalid_input("bf16 attention shape overflows usize"))?;
        let expected_o = query_rows
            .checked_mul(value_dim)
            .ok_or_else(|| NnisError::invalid_input("bf16 attention shape overflows usize"))?;
        if queries.len() != expected_q {
            return Err(NnisError::invalid_input(format!(
                "bf16 attention queries have {} elements; shape ({query_rows}, {head_dim}) \
                 requires {expected_q}",
                queries.len()
            )));
        }
        if keys.len() != expected_k {
            return Err(NnisError::invalid_input(format!(
                "bf16 attention keys have {} elements; shape ({kv_rows}, {head_dim}) \
                 requires {expected_k}",
                keys.len()
            )));
        }
        if values.len() != expected_v {
            return Err(NnisError::invalid_input(format!(
                "bf16 attention values have {} elements; shape ({kv_rows}, {value_dim}) \
                 requires {expected_v}",
                values.len()
            )));
        }
        if output.len() != expected_o {
            return Err(NnisError::invalid_input(format!(
                "bf16 attention output has {} elements; shape ({query_rows}, {value_dim}) \
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
                "bf16 attention stream, buffers, and kernel must share one context",
            ));
        }
        Ok(())
    }

    /// Composed bf16 attention with fully quantized intermediates and wait
    /// for completion - the opt-in opposite of [`Self::attention_composed`].
    ///
    /// Policy: scores are materialized as bf16 by
    /// [`Bf16Gemm::gemm_transposed_b`] (f32 accumulate, one RNE narrow),
    /// widened for the f32 scale/mask/softmax stages in place, narrowed
    /// back to bf16, and multiplied against packed-bf16 values by
    /// [`Bf16Gemm::gemm`]. Both the logits and the probabilities therefore
    /// carry bf16 rounding BEFORE exponentiation and accumulation; expect
    /// visibly larger error than the default composed path. The score
    /// scratch also shrinks from two f32 buffers to one bf16 + one f32
    /// buffer, and the materialized probability map is inspectable at
    /// packed-bf16 width.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_composed_quantized(
        &self,
        conversions: &Bf16Elementwise,
        attention: &F32Attention,
        bf16_gemm: &Bf16Gemm,
        elementwise: &F32Elementwise,
        softmax_2d: &F32Softmax2D,
        stream: &Stream,
        queries: &DeviceBuffer<u16>,
        keys: &DeviceBuffer<u16>,
        values: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
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
        if query_rows == 0 {
            return Ok(());
        }
        let context = self.fused.context();
        let score_elements = query_rows
            .checked_mul(kv_rows)
            .ok_or_else(|| NnisError::invalid_input("bf16 attention shape overflows usize"))?;
        // Reused twice: bf16 logits first, then the narrowed probabilities.
        let packed_scores = DeviceBuffer::<u16>::new(context, score_elements)?;
        let wide_probs = DeviceBuffer::<f32>::new(context, score_elements)?;

        bf16_gemm.gemm_transposed_b(
            stream,
            queries,
            keys,
            &packed_scores,
            query_rows,
            kv_rows,
            head_dim,
        )?;
        conversions.widen(stream, &packed_scores, &wide_probs)?;
        if mask == AttentionMask::Causal {
            // SAFETY: both borrows remain live through the synchronizing
            // call below; the kernel writes each index from its own read.
            unsafe {
                attention.enqueue_scale_causal(
                    stream,
                    &wide_probs,
                    &wide_probs,
                    query_rows,
                    kv_rows,
                    scale,
                )?;
            }
            stream.synchronize()?;
        } else {
            elementwise.scale(stream, &wide_probs, &wide_probs, scale)?;
        }
        softmax_2d.softmax_rows_dispatched(
            stream,
            &wide_probs,
            &wide_probs,
            query_rows,
            kv_rows,
        )?;
        conversions.narrow(stream, &wide_probs, &packed_scores)?;
        bf16_gemm.gemm(
            stream,
            &packed_scores,
            values,
            output,
            query_rows,
            value_dim,
            kv_rows,
        )
    }

    /// Composed packed-bf16 multi-head attention through validated families
    /// and wait for completion: exact widening of every operand, the f32
    /// composed multi-head pipeline above, and one final round-to-nearest-
    /// even narrowing over the packed output.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_composed_multihead(
        &self,
        conversions: &Bf16Elementwise,
        attention: &F32Attention,
        gemm: &F32Gemm,
        elementwise: &F32Elementwise,
        softmax_2d: &F32Softmax2D,
        stream: &Stream,
        queries: &DeviceBuffer<u16>,
        keys: &DeviceBuffer<u16>,
        values: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
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
        let context = self.fused.context();
        let wide_queries = DeviceBuffer::<f32>::new(context, num_heads * query_rows * head_dim)?;
        let wide_keys = DeviceBuffer::<f32>::new(context, num_heads * kv_rows * head_dim)?;
        let wide_values = DeviceBuffer::<f32>::new(context, num_heads * kv_rows * value_dim)?;
        let wide_output = DeviceBuffer::<f32>::new(context, num_heads * query_rows * value_dim)?;

        conversions.widen(stream, queries, &wide_queries)?;
        conversions.widen(stream, keys, &wide_keys)?;
        conversions.widen(stream, values, &wide_values)?;
        attention.attention_composed_multihead(
            gemm,
            elementwise,
            softmax_2d,
            stream,
            &wide_queries,
            &wide_keys,
            &wide_values,
            &wide_output,
            num_heads,
            query_rows,
            head_dim,
            kv_rows,
            value_dim,
            scale,
            mask,
        )?;
        conversions.narrow(stream, &wide_output, output)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_multihead_execution(
        &self,
        stream: &Stream,
        queries: &DeviceBuffer<u16>,
        keys: &DeviceBuffer<u16>,
        values: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        num_heads: usize,
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
    ) -> Result<()> {
        if num_heads == 0 {
            return Err(NnisError::invalid_input(
                "bf16 attention requires at least one head",
            ));
        }
        let expected_q = num_heads
            .checked_mul(query_rows)
            .and_then(|elements| elements.checked_mul(head_dim))
            .ok_or_else(|| NnisError::invalid_input("bf16 attention shape overflows usize"))?;
        let expected_k = num_heads
            .checked_mul(kv_rows)
            .and_then(|elements| elements.checked_mul(head_dim))
            .ok_or_else(|| NnisError::invalid_input("bf16 attention shape overflows usize"))?;
        let expected_v = num_heads
            .checked_mul(kv_rows)
            .and_then(|elements| elements.checked_mul(value_dim))
            .ok_or_else(|| NnisError::invalid_input("bf16 attention shape overflows usize"))?;
        let expected_o = num_heads
            .checked_mul(query_rows)
            .and_then(|elements| elements.checked_mul(value_dim))
            .ok_or_else(|| NnisError::invalid_input("bf16 attention shape overflows usize"))?;
        if queries.len() != expected_q {
            return Err(NnisError::invalid_input(format!(
                "bf16 attention queries have {} elements; {num_heads} heads of shape \
                 ({query_rows}, {head_dim}) requires {expected_q}",
                queries.len()
            )));
        }
        if keys.len() != expected_k {
            return Err(NnisError::invalid_input(format!(
                "bf16 attention keys have {} elements; {num_heads} heads of shape \
                 ({kv_rows}, {head_dim}) requires {expected_k}",
                keys.len()
            )));
        }
        if values.len() != expected_v {
            return Err(NnisError::invalid_input(format!(
                "bf16 attention values have {} elements; {num_heads} heads of shape \
                 ({kv_rows}, {value_dim}) requires {expected_v}",
                values.len()
            )));
        }
        if output.len() != expected_o {
            return Err(NnisError::invalid_input(format!(
                "bf16 attention output has {} elements; {num_heads} heads of shape \
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
                "bf16 attention stream, buffers, and kernel must share one context",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::{bf16_bits_to_f32, f32_to_bf16_rne, gpu_context};

    /// (query_rows, head_dim, kv_rows, value_dim); all fit the fused path
    /// at the default block width except where a test wants rejection.
    const SHAPES: &[(usize, usize, usize, usize)] = &[
        (1, 16, 3, 16),
        (2, 64, 64, 64),
        (5, 64, 200, 64),
        (17, 32, 70, 96),
        (33, 48, 129, 48),
    ];

    /// Absolute correctness tolerance against the f64 oracle. The output is
    /// bf16: round-to-nearest-even contributes up to 2^-8 relative
    /// (half-ulp), and the f32 softmax chain adds far smaller ulp-level
    /// error, so the bound is the quantization term with headroom.
    fn oracle_tolerance(expected: f64) -> f64 {
        5.0e-3_f64.max(expected.abs() * 8.0e-3)
    }

    fn host_values(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| (((index * 13 % 97) as f32 - 48.0) * 0.0625) + ((index % 5) as f32 - 2.0))
            .collect()
    }

    fn to_bits(values: &[f32]) -> Vec<u16> {
        values.iter().copied().map(f32_to_bf16_rne).collect()
    }

    fn widened(bits: &[u16]) -> Vec<f32> {
        bits.iter().copied().map(bf16_bits_to_f32).collect()
    }

    /// f64 attention reference evaluated on the widened bf16 inputs; scores
    /// share the kernel's ascending-dot order and the softmax uses one
    /// shared maximum per query row.
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

    /// Asserts bf16 output bits equal the f32 family's result on the same
    /// widened inputs after one host RNE narrowing, and that the widened
    /// values track the f64 oracle inside quantization-aware tolerances.
    fn assert_bit_exact_and_close(
        actual_bits: &[u16],
        f32_reference_bits: &[u16],
        expected: &[f32],
        context: &str,
    ) {
        assert_eq!(actual_bits.len(), f32_reference_bits.len());
        assert_eq!(actual_bits.len(), expected.len());
        let mut max_difference = 0.0_f64;
        for index in 0..actual_bits.len() {
            assert_eq!(
                actual_bits[index], f32_reference_bits[index],
                "{context} bit mismatch at {index}: {:04x} vs {:04x}",
                actual_bits[index], f32_reference_bits[index]
            );
            let actual = f64::from(bf16_bits_to_f32(actual_bits[index]));
            let oracle = f64::from(expected[index]);
            let difference = (actual - oracle).abs();
            max_difference = max_difference.max(difference);
            let tolerance = oracle_tolerance(oracle);
            assert!(
                difference <= tolerance,
                "{context} mismatch at {index}: {actual} vs {oracle}, tolerance {tolerance}"
            );
        }
        eprintln!("{context}: max element error {max_difference:.3e} vs f64 oracle");
    }

    #[test]
    fn bf16_attention_fused_bit_matches_f32_family_and_f64_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let bf16_attention = Bf16Attention::load(&context, &compiler).unwrap();
        let f32_attention = F32Attention::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(query_rows, head_dim, kv_rows, value_dim) in SHAPES {
            let queries_bits = to_bits(&host_values(query_rows * head_dim));
            let keys_bits = to_bits(&host_values(kv_rows * head_dim));
            let values_bits = to_bits(&host_values(kv_rows * value_dim));
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_bits).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_bits).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_bits).unwrap();
            // Pre-fill so a skipped kernel cannot pass silently.
            let poisoned = vec![0xFFFF_u16; query_rows * value_dim];
            let bf16_output =
                DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();
            let f32_output =
                DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; query_rows * value_dim])
                    .unwrap();

            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            bf16_attention
                .attention_fused(
                    &stream,
                    &queries,
                    &keys,
                    &values,
                    &bf16_output,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    scale,
                    AttentionMask::None,
                )
                .unwrap();
            // The f32 family consumes exactly-widened copies of the same bits.
            let wide_queries =
                DeviceBuffer::from_host(&context, &stream, &widened(&queries_bits)).unwrap();
            let wide_keys =
                DeviceBuffer::from_host(&context, &stream, &widened(&keys_bits)).unwrap();
            let wide_values =
                DeviceBuffer::from_host(&context, &stream, &widened(&values_bits)).unwrap();
            f32_attention
                .attention_fused(
                    &stream,
                    &wide_queries,
                    &wide_keys,
                    &wide_values,
                    &f32_output,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    scale,
                    AttentionMask::None,
                )
                .unwrap();

            let actual_bits = bf16_output.to_vec(&stream).unwrap();
            let f32_actual = f32_output.to_vec(&stream).unwrap();
            let f32_narrowed: Vec<u16> = f32_actual.iter().copied().map(f32_to_bf16_rne).collect();
            let expected = reference_attention(
                &widened(&queries_bits),
                &widened(&keys_bits),
                &widened(&values_bits),
                query_rows,
                head_dim,
                kv_rows,
                value_dim,
                f64::from(scale),
                false,
            );
            assert_bit_exact_and_close(
                &actual_bits,
                &f32_narrowed,
                &expected,
                &format!("fused ({query_rows},{head_dim},{kv_rows},{value_dim})"),
            );
        }
    }

    #[test]
    fn bf16_attention_composed_bit_matches_f32_family_and_f64_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let bf16_attention = Bf16Attention::load(&context, &compiler).unwrap();
        let conversions = Bf16Elementwise::load(&context, &compiler).unwrap();
        let f32_attention = F32Attention::load(&context, &compiler).unwrap();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let elementwise = F32Elementwise::load(&context, &compiler).unwrap();
        let softmax_2d = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(query_rows, head_dim, kv_rows, value_dim) in SHAPES {
            let queries_bits = to_bits(&host_values(query_rows * head_dim));
            let keys_bits = to_bits(&host_values(kv_rows * head_dim));
            let values_bits = to_bits(&host_values(kv_rows * value_dim));
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_bits).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_bits).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_bits).unwrap();
            let poisoned = vec![0xFFFF_u16; query_rows * value_dim];
            let bf16_output =
                DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();
            let f32_output =
                DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; query_rows * value_dim])
                    .unwrap();

            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            bf16_attention
                .attention_composed(
                    &conversions,
                    &f32_attention,
                    &gemm,
                    &elementwise,
                    &softmax_2d,
                    &stream,
                    &queries,
                    &keys,
                    &values,
                    &bf16_output,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    scale,
                    AttentionMask::None,
                )
                .unwrap();

            let wide_queries =
                DeviceBuffer::from_host(&context, &stream, &widened(&queries_bits)).unwrap();
            let wide_keys =
                DeviceBuffer::from_host(&context, &stream, &widened(&keys_bits)).unwrap();
            let wide_values =
                DeviceBuffer::from_host(&context, &stream, &widened(&values_bits)).unwrap();
            f32_attention
                .attention_composed(
                    &gemm,
                    &elementwise,
                    &softmax_2d,
                    &stream,
                    &wide_queries,
                    &wide_keys,
                    &wide_values,
                    &f32_output,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    scale,
                    AttentionMask::None,
                )
                .unwrap();

            let actual_bits = bf16_output.to_vec(&stream).unwrap();
            let f32_actual = f32_output.to_vec(&stream).unwrap();
            let f32_narrowed: Vec<u16> = f32_actual.iter().copied().map(f32_to_bf16_rne).collect();
            let expected = reference_attention(
                &widened(&queries_bits),
                &widened(&keys_bits),
                &widened(&values_bits),
                query_rows,
                head_dim,
                kv_rows,
                value_dim,
                f64::from(scale),
                false,
            );
            assert_bit_exact_and_close(
                &actual_bits,
                &f32_narrowed,
                &expected,
                &format!("composed ({query_rows},{head_dim},{kv_rows},{value_dim})"),
            );
        }
    }

    #[test]
    fn bf16_attention_causal_matches_f32_family_and_f64_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let bf16_attention = Bf16Attention::load(&context, &compiler).unwrap();
        let conversions = Bf16Elementwise::load(&context, &compiler).unwrap();
        let f32_attention = F32Attention::load(&context, &compiler).unwrap();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let elementwise = F32Elementwise::load(&context, &compiler).unwrap();
        let softmax_2d = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        // Square score shapes only; includes shapes spanning many chunks.
        const CAUSAL_SHAPES: &[(usize, usize)] = &[(1, 16), (7, 64), (33, 48), (70, 32)];

        for &(kv_rows, head_dim) in CAUSAL_SHAPES {
            let query_rows = kv_rows;
            let value_dim = head_dim;
            let queries_bits = to_bits(&host_values(query_rows * head_dim));
            let keys_bits = to_bits(&host_values(kv_rows * head_dim));
            let values_bits = to_bits(&host_values(kv_rows * value_dim));
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_bits).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_bits).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_bits).unwrap();
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            let poisoned = vec![0xFFFF_u16; query_rows * value_dim];
            let fused_output =
                DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();
            let composed_output = DeviceBuffer::from_host(&context, &stream, &poisoned).unwrap();
            let f32_fused_output =
                DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; query_rows * value_dim])
                    .unwrap();
            let f32_composed_output =
                DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; query_rows * value_dim])
                    .unwrap();

            bf16_attention
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
                    AttentionMask::Causal,
                )
                .unwrap();
            bf16_attention
                .attention_composed(
                    &conversions,
                    &f32_attention,
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
                    AttentionMask::Causal,
                )
                .unwrap();

            let wide_queries =
                DeviceBuffer::from_host(&context, &stream, &widened(&queries_bits)).unwrap();
            let wide_keys =
                DeviceBuffer::from_host(&context, &stream, &widened(&keys_bits)).unwrap();
            let wide_values =
                DeviceBuffer::from_host(&context, &stream, &widened(&values_bits)).unwrap();
            f32_attention
                .attention_fused(
                    &stream,
                    &wide_queries,
                    &wide_keys,
                    &wide_values,
                    &f32_fused_output,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    scale,
                    AttentionMask::Causal,
                )
                .unwrap();
            f32_attention
                .attention_composed(
                    &gemm,
                    &elementwise,
                    &softmax_2d,
                    &stream,
                    &wide_queries,
                    &wide_keys,
                    &wide_values,
                    &f32_composed_output,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    scale,
                    AttentionMask::Causal,
                )
                .unwrap();

            let expected = reference_attention(
                &widened(&queries_bits),
                &widened(&keys_bits),
                &widened(&values_bits),
                query_rows,
                head_dim,
                kv_rows,
                value_dim,
                f64::from(scale),
                true,
            );
            let f32_fused = f32_fused_output.to_vec(&stream).unwrap();
            let f32_composed = f32_composed_output.to_vec(&stream).unwrap();
            let f32_fused_narrowed: Vec<u16> =
                f32_fused.iter().copied().map(f32_to_bf16_rne).collect();
            let f32_composed_narrowed: Vec<u16> =
                f32_composed.iter().copied().map(f32_to_bf16_rne).collect();
            assert_bit_exact_and_close(
                &fused_output.to_vec(&stream).unwrap(),
                &f32_fused_narrowed,
                &expected,
                &format!("causal fused ({query_rows},{head_dim})"),
            );
            assert_bit_exact_and_close(
                &composed_output.to_vec(&stream).unwrap(),
                &f32_composed_narrowed,
                &expected,
                &format!("causal composed ({query_rows},{head_dim})"),
            );
        }
    }

    #[test]
    fn bf16_attention_rejects_invalid_shapes_and_oversized_heads_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(Bf16Attention::load_with_block_size(&context, &compiler, 0).is_err());
        assert!(Bf16Attention::load_with_block_size(&context, &compiler, 96).is_err());
        let bf16_attention = Bf16Attention::load(&context, &compiler).unwrap();
        let conversions = Bf16Elementwise::load(&context, &compiler).unwrap();
        let f32_attention = F32Attention::load(&context, &compiler).unwrap();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let elementwise = F32Elementwise::load(&context, &compiler).unwrap();
        let softmax_2d = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        // Zero-length key dimension is outside the contract.
        let queries = DeviceBuffer::<u16>::new(&context, 8).unwrap(); // 1 x 8
        let empty_kv = DeviceBuffer::<u16>::new(&context, 0).unwrap();
        let output = DeviceBuffer::<u16>::new(&context, 8).unwrap();
        let error = bf16_attention
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
        let queries = DeviceBuffer::<u16>::new(&context, 64).unwrap(); // 1 x 64
        let short_keys = DeviceBuffer::<u16>::new(&context, 63).unwrap(); // needs 128
        let values = DeviceBuffer::<u16>::new(&context, 256).unwrap(); // 2 x 128
        let error = bf16_attention
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
        assert!(!bf16_attention.fused_available(4096, 4096));
        let big_queries = DeviceBuffer::<u16>::new(&context, 4096).unwrap(); // 1 x 4096
        let big_keys = DeviceBuffer::<u16>::new(&context, 32_768).unwrap(); // 8 x 4096
        let values = DeviceBuffer::<u16>::new(&context, 64).unwrap(); // 8 x 8
        let small_out = DeviceBuffer::<u16>::new(&context, 8).unwrap(); // 1 x 8
        let error = bf16_attention
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

        // Causal masking requires square score shapes on both paths.
        let queries = DeviceBuffer::<u16>::new(&context, 2 * 8).unwrap();
        let keys = DeviceBuffer::<u16>::new(&context, 5 * 8).unwrap();
        let values = DeviceBuffer::<u16>::new(&context, 5 * 4).unwrap();
        let output = DeviceBuffer::<u16>::new(&context, 2 * 4).unwrap();
        let error = bf16_attention
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
        let error = bf16_attention
            .attention_composed(
                &conversions,
                &f32_attention,
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

    /// Replays the fully quantized policy exactly: bit-exact bf16 GEMM-NT
    /// logit chain, f64 stable softmax over the widened bf16 logits, f64
    /// probability/value product over widened bf16 values, one final RNE
    /// narrowing.
    #[allow(clippy::too_many_arguments)]
    fn quantized_reference(
        queries_bits: &[u16],
        keys_bits: &[u16],
        values_bits: &[u16],
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
        scale: f64,
        causal: bool,
    ) -> Vec<u16> {
        let mut output = vec![0_u16; query_rows * value_dim];
        for row in 0..query_rows {
            let mut scores = vec![0.0_f64; kv_rows];
            for key in 0..kv_rows {
                if causal && key > row {
                    break;
                }
                let logit = (0..head_dim).fold(0.0_f32, |value, depth| {
                    bf16_bits_to_f32(queries_bits[row * head_dim + depth])
                        .mul_add(bf16_bits_to_f32(keys_bits[key * head_dim + depth]), value)
                });
                scores[key] = f64::from(bf16_bits_to_f32(f32_to_bf16_rne(logit))) * scale;
            }
            // Masked positions must contribute exactly zero weight, like the
            // kernel's pre-exponentiation -infinity exclusion.
            if causal && row + 1 < kv_rows {
                for score in scores[row + 1..].iter_mut() {
                    *score = f64::NEG_INFINITY;
                }
            }
            let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let total: f64 = weights.iter().sum();
            // The device narrows its normalized f32 probabilities to bf16
            // before the final GEMM; replay that quantization exactly.
            let packed_weights: Vec<f64> = weights
                .iter()
                .map(|&weight| {
                    f64::from(bf16_bits_to_f32(f32_to_bf16_rne((weight / total) as f32)))
                })
                .collect();
            for col in 0..value_dim {
                let value: f64 = (0..kv_rows)
                    .map(|key| {
                        packed_weights[key]
                            * f64::from(bf16_bits_to_f32(values_bits[key * value_dim + col]))
                    })
                    .sum();
                output[row * value_dim + col] = f32_to_bf16_rne(value as f32);
            }
        }
        output
    }

    #[test]
    fn bf16_attention_composed_quantized_matches_quantized_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = Bf16Attention::load(&context, &compiler).unwrap();
        let conversions = Bf16Elementwise::load(&context, &compiler).unwrap();
        let f32_attention = F32Attention::load(&context, &compiler).unwrap();
        let bf16_gemm = crate::Bf16Gemm::load(&context, &compiler).unwrap();
        let elementwise = F32Elementwise::load(&context, &compiler).unwrap();
        let softmax_2d = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(query_rows, head_dim, kv_rows, value_dim) in SHAPES {
            let queries_bits = to_bits(&host_values(query_rows * head_dim));
            let keys_bits = to_bits(&host_values(kv_rows * head_dim));
            let values_bits = to_bits(&host_values(kv_rows * value_dim));
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_bits).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_bits).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_bits).unwrap();
            let poisoned = vec![0xFFFF_u16; query_rows * value_dim];
            let output = DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();

            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            attention
                .attention_composed_quantized(
                    &conversions,
                    &f32_attention,
                    &bf16_gemm,
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
            let actual_bits = output.to_vec(&stream).unwrap();
            let expected_bits = quantized_reference(
                &queries_bits,
                &keys_bits,
                &values_bits,
                query_rows,
                head_dim,
                kv_rows,
                value_dim,
                f64::from(scale),
                false,
            );
            let mut max_difference = 0.0_f64;
            for index in 0..actual_bits.len() {
                let actual = f64::from(bf16_bits_to_f32(actual_bits[index]));
                let oracle = f64::from(bf16_bits_to_f32(expected_bits[index]));
                let difference = (actual - oracle).abs();
                max_difference = max_difference.max(difference);
                let tolerance = oracle_tolerance(oracle);
                assert!(
                    difference <= tolerance,
                    "quantized ({query_rows},{head_dim},{kv_rows},{value_dim}) mismatch at \
                     {index}: {actual} vs {oracle}, tolerance {tolerance}"
                );
            }
            eprintln!(
                "quantized ({query_rows},{head_dim},{kv_rows},{value_dim}): \
                 max element error {max_difference:.3e} vs quantized f64 oracle"
            );
        }
    }

    #[test]
    fn bf16_attention_composed_quantized_causal_and_rejections() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = Bf16Attention::load(&context, &compiler).unwrap();
        let conversions = Bf16Elementwise::load(&context, &compiler).unwrap();
        let f32_attention = F32Attention::load(&context, &compiler).unwrap();
        let bf16_gemm = crate::Bf16Gemm::load(&context, &compiler).unwrap();
        let elementwise = F32Elementwise::load(&context, &compiler).unwrap();
        let softmax_2d = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        const CAUSAL_SHAPES: &[(usize, usize)] = &[(7, 64), (33, 48)];
        for &(rows, head_dim) in CAUSAL_SHAPES {
            let value_dim = head_dim;
            let queries_bits = to_bits(&host_values(rows * head_dim));
            let keys_bits = to_bits(&host_values(rows * head_dim));
            let values_bits = to_bits(&host_values(rows * value_dim));
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_bits).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_bits).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_bits).unwrap();
            let poisoned = vec![0xFFFF_u16; rows * value_dim];
            let output = DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();

            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            attention
                .attention_composed_quantized(
                    &conversions,
                    &f32_attention,
                    &bf16_gemm,
                    &elementwise,
                    &softmax_2d,
                    &stream,
                    &queries,
                    &keys,
                    &values,
                    &output,
                    rows,
                    head_dim,
                    rows,
                    value_dim,
                    scale,
                    AttentionMask::Causal,
                )
                .unwrap();
            let actual_bits = output.to_vec(&stream).unwrap();
            let expected_bits = quantized_reference(
                &queries_bits,
                &keys_bits,
                &values_bits,
                rows,
                head_dim,
                rows,
                value_dim,
                f64::from(scale),
                true,
            );
            for index in 0..actual_bits.len() {
                let difference = (f64::from(bf16_bits_to_f32(actual_bits[index]))
                    - f64::from(bf16_bits_to_f32(expected_bits[index])))
                .abs();
                let tolerance = oracle_tolerance(f64::from(bf16_bits_to_f32(expected_bits[index])));
                assert!(
                    difference <= tolerance,
                    "quantized causal ({rows},{head_dim}) mismatch at {index}: \
                     tolerance {tolerance}"
                );
            }
        }

        // Rectangular causal shapes stay rejected before launch.
        let queries = DeviceBuffer::<u16>::new(&context, 2 * 8).unwrap();
        let keys = DeviceBuffer::<u16>::new(&context, 5 * 8).unwrap();
        let values = DeviceBuffer::<u16>::new(&context, 5 * 4).unwrap();
        let output = DeviceBuffer::<u16>::new(&context, 2 * 4).unwrap();
        let error = attention
            .attention_composed_quantized(
                &conversions,
                &f32_attention,
                &bf16_gemm,
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

        // Short packed buffers stay rejected before launch.
        let short_keys = DeviceBuffer::<u16>::new(&context, 39).unwrap(); // needs 40
        let error = attention
            .attention_composed_quantized(
                &conversions,
                &f32_attention,
                &bf16_gemm,
                &elementwise,
                &softmax_2d,
                &stream,
                &queries,
                &short_keys,
                &values,
                &output,
                2,
                8,
                5,
                4,
                0.25,
                AttentionMask::None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("requires 40"), "{error}");
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
    fn multihead_values(head: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                (((index * 13 % 97) as f32 - 48.0) * 0.0625)
                    + (((index + head * 11) % 5) as f32 - 2.0)
            })
            .collect()
    }

    #[test]
    fn bf16_attention_multihead_bit_matches_per_head_fused_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = Bf16Attention::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(heads, query_rows, head_dim, kv_rows, value_dim) in MULTIHEAD_SHAPES {
            // Interleave per-head hosts into the packed [heads][rows][dim]
            // layout.
            let mut queries_host: Vec<u16> = Vec::with_capacity(heads * query_rows * head_dim);
            let mut keys_host: Vec<u16> = Vec::with_capacity(heads * kv_rows * head_dim);
            let mut values_host: Vec<u16> = Vec::with_capacity(heads * kv_rows * value_dim);
            for head in 0..heads {
                queries_host.extend(to_bits(&multihead_values(head, query_rows * head_dim)));
                keys_host.extend(to_bits(&multihead_values(head + 97, kv_rows * head_dim)));
                values_host.extend(to_bits(&multihead_values(head + 193, kv_rows * value_dim)));
            }
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_host).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_host).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
            // Pre-fill so a skipped kernel cannot pass silently.
            let poisoned = vec![0xFFFF_u16; heads * query_rows * value_dim];
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
                    &to_bits(&multihead_values(head, query_rows * head_dim)),
                )
                .unwrap();
                let head_keys = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &to_bits(&multihead_values(head + 97, kv_rows * head_dim)),
                )
                .unwrap();
                let head_values = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &to_bits(&multihead_values(head + 193, kv_rows * value_dim)),
                )
                .unwrap();
                let head_poisoned = vec![0xFFFF_u16; query_rows * value_dim];
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
                        actual, &head_actual[index],
                        "bf16 multihead {heads}h bit mismatch at head {head} element {index} \
                         shape ({query_rows},{head_dim},{kv_rows},{value_dim})"
                    );
                }
            }
        }
    }

    #[test]
    fn bf16_attention_multihead_causal_bit_matches_per_head_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = Bf16Attention::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        const CAUSAL_HEADS_SHAPES: &[(usize, usize, usize)] =
            &[(2, 7, 64), (3, 33, 48), (4, 70, 32)];

        for &(heads, rows, head_dim) in CAUSAL_HEADS_SHAPES {
            let value_dim = head_dim;
            let mut queries_host: Vec<u16> = Vec::with_capacity(heads * rows * head_dim);
            let mut keys_host: Vec<u16> = Vec::with_capacity(heads * rows * head_dim);
            let mut values_host: Vec<u16> = Vec::with_capacity(heads * rows * value_dim);
            for head in 0..heads {
                queries_host.extend(to_bits(&multihead_values(head, rows * head_dim)));
                keys_host.extend(to_bits(&multihead_values(head + 97, rows * head_dim)));
                values_host.extend(to_bits(&multihead_values(head + 193, rows * value_dim)));
            }
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_host).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_host).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
            let poisoned = vec![0xFFFF_u16; heads * rows * value_dim];
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
                    &to_bits(&multihead_values(head, rows * head_dim)),
                )
                .unwrap();
                let head_keys = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &to_bits(&multihead_values(head + 97, rows * head_dim)),
                )
                .unwrap();
                let head_values = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &to_bits(&multihead_values(head + 193, rows * value_dim)),
                )
                .unwrap();
                let head_poisoned = vec![0xFFFF_u16; rows * value_dim];
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
                        actual, &head_actual[index],
                        "bf16 causal multihead {heads}h bit mismatch at head {head} \
                         element {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn bf16_attention_composed_multihead_bit_matches_per_head_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = Bf16Attention::load(&context, &compiler).unwrap();
        let conversions = Bf16Elementwise::load(&context, &compiler).unwrap();
        let f32_attention = F32Attention::load(&context, &compiler).unwrap();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let elementwise = F32Elementwise::load(&context, &compiler).unwrap();
        let softmax_2d = F32Softmax2D::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(heads, query_rows, head_dim, kv_rows, value_dim) in MULTIHEAD_SHAPES {
            let mut queries_host: Vec<u16> = Vec::with_capacity(heads * query_rows * head_dim);
            let mut keys_host: Vec<u16> = Vec::with_capacity(heads * kv_rows * head_dim);
            let mut values_host: Vec<u16> = Vec::with_capacity(heads * kv_rows * value_dim);
            for head in 0..heads {
                queries_host.extend(to_bits(&multihead_values(head, query_rows * head_dim)));
                keys_host.extend(to_bits(&multihead_values(head + 97, kv_rows * head_dim)));
                values_host.extend(to_bits(&multihead_values(head + 193, kv_rows * value_dim)));
            }
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_host).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_host).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
            let poisoned = vec![0xFFFF_u16; heads * query_rows * value_dim];
            let batched_output =
                DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();

            let scale = 1.0_f32 / (head_dim as f32).sqrt();
            attention
                .attention_composed_multihead(
                    &conversions,
                    &f32_attention,
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
                    &to_bits(&multihead_values(head, query_rows * head_dim)),
                )
                .unwrap();
                let head_keys = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &to_bits(&multihead_values(head + 97, kv_rows * head_dim)),
                )
                .unwrap();
                let head_values = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &to_bits(&multihead_values(head + 193, kv_rows * value_dim)),
                )
                .unwrap();
                let head_output = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &vec![0xFFFF_u16; query_rows * value_dim],
                )
                .unwrap();
                attention
                    .attention_composed(
                        &conversions,
                        &f32_attention,
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
                for index in 0..query_rows * value_dim {
                    assert_eq!(
                        batched_actual[head * query_rows * value_dim + index],
                        head_actual[index],
                        "bf16 composed multihead {heads}h bit mismatch at head {head} \
                         element {index} shape ({query_rows},{head_dim},{kv_rows},{value_dim})"
                    );
                }
            }
        }

        // Causal square shapes: same bit-exact contract.
        const CAUSAL_HEADS_SHAPES: &[(usize, usize, usize)] = &[(2, 7, 64), (3, 33, 48)];
        for &(heads, rows, head_dim) in CAUSAL_HEADS_SHAPES {
            let value_dim = head_dim;
            let mut queries_host: Vec<u16> = Vec::with_capacity(heads * rows * head_dim);
            let mut keys_host: Vec<u16> = Vec::with_capacity(heads * rows * head_dim);
            let mut values_host: Vec<u16> = Vec::with_capacity(heads * rows * value_dim);
            for head in 0..heads {
                queries_host.extend(to_bits(&multihead_values(head, rows * head_dim)));
                keys_host.extend(to_bits(&multihead_values(head + 97, rows * head_dim)));
                values_host.extend(to_bits(&multihead_values(head + 193, rows * value_dim)));
            }
            let queries = DeviceBuffer::from_host(&context, &stream, &queries_host).unwrap();
            let keys = DeviceBuffer::from_host(&context, &stream, &keys_host).unwrap();
            let values = DeviceBuffer::from_host(&context, &stream, &values_host).unwrap();
            let poisoned = vec![0xFFFF_u16; heads * rows * value_dim];
            let batched_output =
                DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();
            let scale = 1.0_f32 / (head_dim as f32).sqrt();

            attention
                .attention_composed_multihead(
                    &conversions,
                    &f32_attention,
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
                    &to_bits(&multihead_values(head, rows * head_dim)),
                )
                .unwrap();
                let head_keys = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &to_bits(&multihead_values(head + 97, rows * head_dim)),
                )
                .unwrap();
                let head_values = DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &to_bits(&multihead_values(head + 193, rows * value_dim)),
                )
                .unwrap();
                let head_output =
                    DeviceBuffer::from_host(&context, &stream, &vec![0xFFFF_u16; rows * value_dim])
                        .unwrap();
                attention
                    .attention_composed(
                        &conversions,
                        &f32_attention,
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
                for index in 0..rows * value_dim {
                    assert_eq!(
                        batched_actual[head * rows * value_dim + index],
                        head_actual[index],
                        "bf16 causal composed multihead {heads}h bit mismatch at head \
                         {head} element {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn bf16_attention_multihead_rejects_invalid_shapes_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let attention = Bf16Attention::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        // Zero heads is outside the contract.
        let queries = DeviceBuffer::<u16>::new(&context, 8).unwrap();
        let keys = DeviceBuffer::<u16>::new(&context, 8).unwrap();
        let values = DeviceBuffer::<u16>::new(&context, 8).unwrap();
        let output = DeviceBuffer::<u16>::new(&context, 8).unwrap();
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
        let queries = DeviceBuffer::<u16>::new(&context, 2 * 2 * 8).unwrap();
        let keys = DeviceBuffer::<u16>::new(&context, 2 * 5 * 8).unwrap();
        let values = DeviceBuffer::<u16>::new(&context, 2 * 5 * 4).unwrap();
        let output = DeviceBuffer::<u16>::new(&context, 2 * 2 * 4).unwrap();
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
