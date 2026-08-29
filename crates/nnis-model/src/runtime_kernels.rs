//! Small runtime-only CUDA operations for device-resident token flow and
//! position-aware RoPE. These close interface gaps between existing generic
//! NNIS primitives; they are not standalone kernel-development targets.

use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const RUNTIME_SOURCE: &str = r#"
extern "C" __global__ void nnis_select_token_u32(
    const unsigned int* input_ids,
    unsigned int* current_token,
    unsigned long long position,
    unsigned long long count
) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && position < count) {
        current_token[0] = input_ids[position];
    }
}

extern "C" __global__ void nnis_record_token_u32(
    const unsigned int* current_token,
    unsigned int* output_ids,
    unsigned long long position,
    unsigned long long capacity
) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && position < capacity) {
        output_ids[position] = current_token[0];
    }
}

// Llama-style rotate-half RoPE over packed [heads][head_dim] rows. The
// position cache is generated once on the host and stored as
// [max_positions][head_dim/2] cos/sin values. This kernel only selects the
// requested position, avoiding a host copy or a temporary exact-length view.
extern "C" __global__ void nnis_rope_rotate_half_position_f32(
    const float* input,
    const float* cos_cache,
    const float* sin_cache,
    float* output,
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
    const float left = input[row + pair];
    const float right = input[row + half + pair];
    const float cosine = cos_cache[cache_index];
    const float sine = sin_cache[cache_index];
    output[row + pair] = left * cosine - right * sine;
    output[row + half + pair] = right * cosine + left * sine;
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

#[derive(Debug)]
pub struct F32RuntimeKernels {
    select_token: Kernel,
    record_token: Kernel,
    rope_position: Kernel,
    block_size: u32,
}

impl F32RuntimeKernels {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        let code = compiler.compile_cubin(RUNTIME_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let select_token = module.get_function("nnis_select_token_u32")?;
        let record_token = module.get_function("nnis_record_token_u32")?;
        let rope_position = module.get_function("nnis_rope_rotate_half_position_f32")?;
        let attributes = rope_position.attributes()?;
        if DEFAULT_BLOCK_SIZE > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "runtime RoPE block size {DEFAULT_BLOCK_SIZE} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        Ok(Self {
            select_token,
            record_token,
            rope_position,
            block_size: DEFAULT_BLOCK_SIZE,
        })
    }

    /// # Safety
    ///
    /// Buffers and stream must remain alive until the stream reaches this
    /// operation. `position` must be in range; it is validated before launch.
    pub unsafe fn enqueue_select_token(
        &self,
        stream: &Stream,
        input_ids: &DeviceBuffer<u32>,
        current_token: &DeviceBuffer<u32>,
        position: usize,
    ) -> Result<()> {
        self.validate_context_u32(stream, input_ids, current_token)?;
        if current_token.len() != 1 {
            return Err(NnisError::invalid_input(format!(
                "current-token buffer must contain exactly one u32; got {}",
                current_token.len()
            )));
        }
        if position >= input_ids.len() {
            return Err(NnisError::invalid_input(format!(
                "token position {position} is out of range for {} input ids",
                input_ids.len()
            )));
        }
        let mut args = KernelArgs::with_capacity(4, 2);
        args.push_buffer(input_ids)
            .push_buffer(current_token)
            .push(position as u64)
            .push(input_ids.len() as u64);
        let launch = KernelLaunch::new(
            &self.select_token,
            stream,
            LaunchConfig::for_num_elements(1, 1)?,
        );
        unsafe { launch.launch(&mut args) }
    }

    /// # Safety
    ///
    /// Buffers and stream must remain alive until the stream reaches this
    /// operation. `position` must be in range; it is validated before launch.
    pub unsafe fn enqueue_record_token(
        &self,
        stream: &Stream,
        current_token: &DeviceBuffer<u32>,
        output_ids: &DeviceBuffer<u32>,
        position: usize,
    ) -> Result<()> {
        self.validate_context_u32(stream, current_token, output_ids)?;
        if current_token.len() != 1 {
            return Err(NnisError::invalid_input(format!(
                "current-token buffer must contain exactly one u32; got {}",
                current_token.len()
            )));
        }
        if position >= output_ids.len() {
            return Err(NnisError::invalid_input(format!(
                "generated-token position {position} is out of range for capacity {}",
                output_ids.len()
            )));
        }
        let mut args = KernelArgs::with_capacity(4, 2);
        args.push_buffer(current_token)
            .push_buffer(output_ids)
            .push(position as u64)
            .push(output_ids.len() as u64);
        let launch = KernelLaunch::new(
            &self.record_token,
            stream,
            LaunchConfig::for_num_elements(1, 1)?,
        );
        unsafe { launch.launch(&mut args) }
    }

    /// # Safety
    ///
    /// All buffers, stream and this kernel set must remain alive until the
    /// stream reaches the launch. The output may be consumed by later work on
    /// the same stream.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_rope_position(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        cos_cache: &DeviceBuffer<f32>,
        sin_cache: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        heads: usize,
        head_dim: usize,
        position: usize,
        max_positions: usize,
    ) -> Result<()> {
        if head_dim == 0 || head_dim % 2 != 0 || heads == 0 {
            return Err(NnisError::invalid_input(format!(
                "RoPE requires non-zero heads and even head_dim; got heads={heads}, head_dim={head_dim}"
            )));
        }
        let width = heads
            .checked_mul(head_dim)
            .ok_or_else(|| NnisError::invalid_input("RoPE width overflows usize"))?;
        if input.len() != width || output.len() != width {
            return Err(NnisError::invalid_input(format!(
                "RoPE input/output lengths must be {width}; got {}/{}",
                input.len(),
                output.len()
            )));
        }
        let cache_len = max_positions
            .checked_mul(head_dim / 2)
            .ok_or_else(|| NnisError::invalid_input("RoPE cache shape overflows usize"))?;
        if cos_cache.len() != cache_len || sin_cache.len() != cache_len {
            return Err(NnisError::invalid_input(format!(
                "RoPE caches must each contain {cache_len} elements; got {}/{}",
                cos_cache.len(),
                sin_cache.len()
            )));
        }
        if position >= max_positions {
            return Err(NnisError::invalid_input(format!(
                "RoPE position {position} exceeds max position {max_positions}"
            )));
        }
        self.validate_context_f32(stream, &[input, cos_cache, sin_cache, output])?;
        let pairs = heads
            .checked_mul(head_dim / 2)
            .ok_or_else(|| NnisError::invalid_input("RoPE pair count overflows usize"))?;
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
            LaunchConfig::for_num_elements(pairs, self.block_size)?,
        );
        unsafe { launch.launch(&mut args) }
    }

    fn validate_context_u32(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<u32>,
        right: &DeviceBuffer<u32>,
    ) -> Result<()> {
        let context = self.select_token.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, left.ctx())
            || !Arc::ptr_eq(context, right.ctx())
        {
            return Err(NnisError::invalid_input(
                "runtime token buffers, stream and kernels must share one CUDA context",
            ));
        }
        Ok(())
    }

    fn validate_context_f32(
        &self,
        stream: &Stream,
        buffers: &[&DeviceBuffer<f32>],
    ) -> Result<()> {
        let context = self.rope_position.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || buffers
                .iter()
                .any(|buffer| !Arc::ptr_eq(context, buffer.ctx()))
        {
            return Err(NnisError::invalid_input(
                "runtime f32 buffers, stream and kernels must share one CUDA context",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    #[test]
    fn token_flow_and_position_rope_match_host_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let kernels = F32RuntimeKernels::load(&context, &JitCompiler::new()).unwrap();

        let ids = DeviceBuffer::from_host(&context, &stream, &[4_u32, 7, 9]).unwrap();
        let current = DeviceBuffer::<u32>::new(&context, 1).unwrap();
        let recorded = DeviceBuffer::<u32>::new_zeroed(&context, 2, &stream).unwrap();
        unsafe {
            kernels
                .enqueue_select_token(&stream, &ids, &current, 1)
                .unwrap();
            kernels
                .enqueue_record_token(&stream, &current, &recorded, 0)
                .unwrap();
        }
        stream.synchronize().unwrap();
        assert_eq!(current.to_vec(&stream).unwrap(), vec![7]);
        assert_eq!(recorded.to_vec(&stream).unwrap(), vec![7, 0]);

        let input = DeviceBuffer::from_host(
            &context,
            &stream,
            &[1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0],
        )
        .unwrap();
        let cos = DeviceBuffer::from_host(
            &context,
            &stream,
            &[1.0_f32, 1.0, 0.5, -0.25, -0.5, 0.75],
        )
        .unwrap();
        let sin = DeviceBuffer::from_host(
            &context,
            &stream,
            &[0.0_f32, 0.0, 0.25, 0.75, 0.5, -0.25],
        )
        .unwrap();
        let output = DeviceBuffer::<f32>::new(&context, 8).unwrap();
        unsafe {
            kernels
                .enqueue_rope_position(&stream, &input, &cos, &sin, &output, 2, 4, 1, 3)
                .unwrap();
        }
        stream.synchronize().unwrap();
        let actual = output.to_vec(&stream).unwrap();
        let expected = [
            1.0 * 0.5 - 3.0 * 0.25,
            2.0 * -0.25 - 4.0 * 0.75,
            3.0 * 0.5 + 1.0 * 0.25,
            4.0 * -0.25 + 2.0 * 0.75,
            -1.0 * 0.5 - 2.0 * 0.25,
            0.5 * -0.25 - -3.0 * 0.75,
            2.0 * 0.5 + -1.0 * 0.25,
            -3.0 * -0.25 + 0.5 * 0.75,
        ];
        for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "RoPE mismatch at {index}: {actual} != {expected}"
            );
        }
    }
}
