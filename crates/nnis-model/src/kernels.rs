//! CUDA operations required by a decoder model but not provided by the
//! existing generic NNIS primitive families.
//!
//! These are deliberately narrow correctness kernels: per-channel RMSNorm,
//! elementwise multiplication for SwiGLU, and single-token attention over the
//! capacity-strided owned KV cache. They are not performance claims.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, KvCache, NnisError, Result, Stream};
use std::sync::Arc;

const MODEL_KERNELS_SOURCE: &str = r#"
extern "C" __global__ void nnis_weighted_rmsnorm_f32(
    const float* input,
    const float* weight,
    float* output,
    unsigned long long cols,
    float inv_cols,
    float epsilon
) {
    extern __shared__ float partial[];
    const unsigned int lane = threadIdx.x;
    const unsigned long long row = blockIdx.x;
    const float* source = input + row * cols;
    float* destination = output + row * cols;

    float sumsq = 0.0f;
    for (unsigned long long col = lane; col < cols; col += blockDim.x) {
        const float value = source[col];
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
        destination[col] = source[col] * scale * weight[col];
    }
}

extern "C" __global__ void nnis_multiply_f32(
    const float* left,
    const float* right,
    float* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = left[index] * right[index];
    }
}

// Correctness-first single-token decoder attention over the fixed-capacity
// cache layout [layer][kv_head][capacity][head_dim]. One CUDA thread owns one
// query head. Consecutive groups of query heads share a KV head.
extern "C" __global__ void nnis_cached_attention_decode_f32(
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
    if (threadIdx.x != 0 || query_head >= query_heads) {
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

    for (unsigned long long dim = 0; dim < head_dim; ++dim) {
        destination[dim] = 0.0f;
    }

    float running_max = -3.402823466e+38F;
    float running_sum = 0.0f;
    for (unsigned long long position = 0; position < kv_rows; ++position) {
        const float* key = head_keys + position * head_dim;
        const float* value = head_values + position * head_dim;
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
        for (unsigned long long dim = 0; dim < head_dim; ++dim) {
            destination[dim] =
                destination[dim] * old_weight + value[dim] * new_weight;
        }
        running_sum = running_sum * old_weight + new_weight;
        running_max = next_max;
    }

    const float inverse_sum = 1.0f / running_sum;
    for (unsigned long long dim = 0; dim < head_dim; ++dim) {
        destination[dim] *= inverse_sum;
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

fn gqa_group_size(query_heads: usize, kv_heads: usize) -> Result<usize> {
    if query_heads == 0 || kv_heads == 0 {
        return Err(NnisError::invalid_input(
            "cached attention requires non-zero query and KV head counts",
        ));
    }
    if query_heads % kv_heads != 0 {
        return Err(NnisError::invalid_input(format!(
            "query head count {query_heads} is not divisible by KV head count {kv_heads}"
        )));
    }
    Ok(query_heads / kv_heads)
}

/// Exact CUDA operations needed to assemble the first decoder runtime.
#[derive(Debug)]
pub struct F32DecoderKernels {
    weighted_rms_norm: Kernel,
    multiply: Kernel,
    cached_attention_decode: Kernel,
    block_size: u32,
}

impl F32DecoderKernels {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        let code =
            compiler.compile_cubin(MODEL_KERNELS_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let weighted_rms_norm = module.get_function("nnis_weighted_rmsnorm_f32")?;
        let multiply = module.get_function("nnis_multiply_f32")?;
        let cached_attention_decode = module.get_function("nnis_cached_attention_decode_f32")?;

        for (name, kernel) in [
            ("weighted_rms_norm", &weighted_rms_norm),
            ("multiply", &multiply),
        ] {
            let attributes = kernel.attributes()?;
            if DEFAULT_BLOCK_SIZE > attributes.max_threads_per_block {
                return Err(NnisError::invalid_input(format!(
                    "decoder {name} block size {DEFAULT_BLOCK_SIZE} exceeds function limit {}",
                    attributes.max_threads_per_block
                )));
            }
        }
        let shared_memory_bytes = DEFAULT_BLOCK_SIZE as usize * std::mem::size_of::<f32>();
        if shared_memory_bytes
            > weighted_rms_norm
                .attributes()?
                .max_dynamic_shared_memory_bytes as usize
        {
            return Err(NnisError::invalid_input(format!(
                "weighted RMSNorm requires {shared_memory_bytes} shared-memory bytes"
            )));
        }

        Ok(Self {
            weighted_rms_norm,
            multiply,
            cached_attention_decode,
            block_size: DEFAULT_BLOCK_SIZE,
        })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    #[allow(clippy::too_many_arguments)]
    pub fn weighted_rms_norm(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) -> Result<()> {
        let result = unsafe {
            self.enqueue_weighted_rms_norm(stream, input, weight, output, rows, cols, epsilon)
        };
        match result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// # Safety
    ///
    /// The stream, kernel set and all buffers must remain alive and otherwise
    /// untouched until the stream completes the launch.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_weighted_rms_norm(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) -> Result<()> {
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("weighted RMSNorm shape overflows usize"))?;
        if input.len() != count || output.len() != count || weight.len() != cols {
            return Err(NnisError::invalid_input(format!(
                "weighted RMSNorm expects input/output {count} and weight {cols}; got {}/{}/{}",
                input.len(),
                output.len(),
                weight.len()
            )));
        }
        self.validate_contexts(stream, &[input, weight, output])?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(NnisError::invalid_input(format!(
                "weighted RMSNorm epsilon must be finite and positive; got {epsilon}"
            )));
        }
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        let grid_rows = u32::try_from(rows)
            .map_err(|_| NnisError::invalid_input("weighted RMSNorm rows exceed u32"))?;
        let config =
            LaunchConfig::new(Dim3::new(grid_rows, 1, 1), Dim3::new(self.block_size, 1, 1))
                .with_dynamic_shared_memory(self.block_size * std::mem::size_of::<f32>() as u32);
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

    pub fn multiply(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        let result = unsafe { self.enqueue_multiply(stream, left, right, output) };
        match result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// # Safety
    ///
    /// The stream, kernel set and all buffers must remain alive and otherwise
    /// untouched until the stream completes the launch.
    pub unsafe fn enqueue_multiply(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        if left.len() != right.len() || left.len() != output.len() {
            return Err(NnisError::invalid_input(format!(
                "multiply length mismatch: left={}, right={}, output={}",
                left.len(),
                right.len(),
                output.len()
            )));
        }
        self.validate_contexts(stream, &[left, right, output])?;
        if output.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(4, 3);
        args.push_buffer(left)
            .push_buffer(right)
            .push_buffer(output)
            .push(output.len() as u64);
        let launch = KernelLaunch::new(
            &self.multiply,
            stream,
            LaunchConfig::for_num_elements(output.len(), self.block_size)?,
        );
        unsafe { launch.launch(&mut args) }
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
        let result = unsafe {
            self.enqueue_cached_attention_decode(stream, query, cache, layer, output, scale)
        };
        match result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue one-token multi-head attention over the active prefix of an
    /// owned capacity-strided KV cache.
    ///
    /// # Safety
    ///
    /// The stream, kernel set, query/output and cache allocations must remain
    /// alive and otherwise untouched until the stream completes. The cache must
    /// not be reset or appended from another stream concurrently.
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
                "cached attention query/output lengths differ: {}/{}",
                query.len(),
                output.len()
            )));
        }
        if query.len() % config.head_dim != 0 {
            return Err(NnisError::invalid_input(format!(
                "cached attention query width {} is not divisible by head_dim {}",
                query.len(),
                config.head_dim
            )));
        }
        let query_heads = query.len() / config.head_dim;
        let _group_size = gqa_group_size(query_heads, config.heads)?;
        if kv_rows == 0 {
            return Err(NnisError::invalid_input(
                "cached attention requires at least one valid KV position",
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(NnisError::invalid_input(format!(
                "attention scale must be finite and positive; got {scale}"
            )));
        }
        self.validate_contexts(stream, &[query, cache.keys(), cache.values(), output])?;
        if stream.raw() != cache.stream().raw() {
            return Err(NnisError::invalid_input(
                "cached attention must execute on the KV cache's owning stream",
            ));
        }
        let query_heads_u32 = u32::try_from(query_heads)
            .map_err(|_| NnisError::invalid_input("query head count exceeds u32"))?;
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
            LaunchConfig::new(Dim3::new(query_heads_u32, 1, 1), Dim3::new(1, 1, 1)),
        );
        unsafe { launch.launch(&mut args) }
    }

    fn validate_contexts(&self, stream: &Stream, buffers: &[&DeviceBuffer<f32>]) -> Result<()> {
        if !Arc::ptr_eq(self.context(), stream.ctx())
            || buffers
                .iter()
                .any(|buffer| !Arc::ptr_eq(self.context(), buffer.ctx()))
        {
            return Err(NnisError::invalid_input(
                "decoder kernels, stream and buffers must share one CUDA context",
            ));
        }
        Ok(())
    }

    fn context(&self) -> &Arc<Context> {
        self.weighted_rms_norm.context()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::{gpu_context, KvCacheConfig};

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "mismatch at {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn grouped_query_head_mapping_is_validated_without_cuda() {
        assert_eq!(gqa_group_size(2, 2).unwrap(), 1);
        assert_eq!(gqa_group_size(4, 2).unwrap(), 2);
        assert_eq!(gqa_group_size(9, 3).unwrap(), 3);
        assert!(gqa_group_size(3, 2).is_err());
        assert!(gqa_group_size(0, 1).is_err());
    }

    #[test]
    fn exact_decoder_kernels_match_host_oracles_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let kernels = F32DecoderKernels::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        let input_host = [1.0_f32, -2.0, 3.0, -4.0, 2.0, 1.0, -1.0, 0.5];
        let weight_host = [0.5_f32, 1.0, 1.5, 2.0];
        let input = DeviceBuffer::from_host(&context, &stream, &input_host).unwrap();
        let weight = DeviceBuffer::from_host(&context, &stream, &weight_host).unwrap();
        let normalized = DeviceBuffer::<f32>::new(&context, input_host.len()).unwrap();
        kernels
            .weighted_rms_norm(&stream, &input, &weight, &normalized, 2, 4, 1.0e-5)
            .unwrap();
        let mut expected = Vec::with_capacity(input_host.len());
        for row in input_host.chunks_exact(4) {
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / 4.0;
            let inverse = 1.0 / (mean_square + 1.0e-5).sqrt();
            expected.extend(
                row.iter()
                    .zip(weight_host)
                    .map(|(&value, weight)| value * inverse * weight),
            );
        }
        assert_close(&normalized.to_vec(&stream).unwrap(), &expected, 2.0e-5);

        let right_host = [2.0_f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let right = DeviceBuffer::from_host(&context, &stream, &right_host).unwrap();
        let product = DeviceBuffer::<f32>::new(&context, input_host.len()).unwrap();
        kernels.multiply(&stream, &input, &right, &product).unwrap();
        let expected_product: Vec<f32> = input_host
            .iter()
            .zip(right_host)
            .map(|(&left, right)| left * right)
            .collect();
        assert_eq!(product.to_vec(&stream).unwrap(), expected_product);

        let mut cache =
            KvCache::<f32>::new(&stream, KvCacheConfig::new(1, 2, 2, 4).unwrap()).unwrap();
        let keys = Arc::new(
            DeviceBuffer::from_host(
                &context,
                &stream,
                &[1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0],
            )
            .unwrap(),
        );
        let values = Arc::new(
            DeviceBuffer::from_host(
                &context,
                &stream,
                &[10.0_f32, 0.0, 0.0, 20.0, 1.0, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );
        cache.append_layer(0, keys, values, 2).unwrap();
        let query = DeviceBuffer::from_host(&context, &stream, &[1.0_f32, 0.0, 0.0, 1.0]).unwrap();
        let attended = DeviceBuffer::<f32>::new(&context, 4).unwrap();
        kernels
            .cached_attention_decode(&stream, &query, &cache, 0, &attended, 1.0)
            .unwrap();
        let e = 1.0_f32.exp();
        let expected_attention = [10.0 * e / (e + 1.0), 20.0 / (e + 1.0), 2.0, 3.0];
        assert_close(
            &attended.to_vec(&stream).unwrap(),
            &expected_attention,
            2.0e-5,
        );

        let gqa_query = DeviceBuffer::from_host(
            &context,
            &stream,
            &[1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let gqa_attended = DeviceBuffer::<f32>::new(&context, 8).unwrap();
        kernels
            .cached_attention_decode(&stream, &gqa_query, &cache, 0, &gqa_attended, 1.0)
            .unwrap();
        let e2 = 2.0_f32.exp();
        let expected_gqa = [
            10.0 * e / (e + 1.0),
            20.0 / (e + 1.0),
            10.0 / (e + 1.0),
            20.0 * e / (e + 1.0),
            (e2 + 3.0) / (e2 + 1.0),
            (2.0 * e2 + 4.0) / (e2 + 1.0),
            2.0,
            3.0,
        ];
        assert_close(
            &gqa_attended.to_vec(&stream).unwrap(),
            &expected_gqa,
            2.0e-5,
        );
    }
}
