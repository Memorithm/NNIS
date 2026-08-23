//! `bfloat16` storage with `f32` compute: the NNIS inference numeric policy.
//!
//! Every kernel widens its 16-bit inputs to `f32` with exact bit shifts,
//! computes in full precision, and narrows back with round-to-nearest-even.
//! The host-side helpers in `nnis-rt` are bit-identical to the device
//! implementations, so oracles can assert exact equality.

use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const BF16_SOURCE: &str = r#"
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

extern "C" __global__ void nnis_bf16_vector_add_f32acc(
    const unsigned short* left,
    const unsigned short* right,
    unsigned short* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float sum = bf16_bits_to_f32(left[index]) + bf16_bits_to_f32(right[index]);
        output[index] = f32_to_bf16_bits(sum);
    }
}

extern "C" __global__ void nnis_bf16_scale_f32acc(
    const unsigned short* input,
    unsigned short* output,
    float scale,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float product = bf16_bits_to_f32(input[index]) * scale;
        output[index] = f32_to_bf16_bits(product);
    }
}

extern "C" __global__ void nnis_bf16_affine_f32acc(
    const unsigned short* input,
    unsigned short* output,
    float scale,
    float bias,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float widened = bf16_bits_to_f32(input[index]);
        output[index] = f32_to_bf16_bits(fmaf(widened, scale, bias));
    }
}

extern "C" __global__ void nnis_bf16_relu_f32acc(
    const unsigned short* input,
    unsigned short* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float widened = bf16_bits_to_f32(input[index]);
        output[index] = f32_to_bf16_bits(fmaxf(widened, 0.0f));
    }
}

extern "C" __global__ void nnis_bf16_silu_f32acc(
    const unsigned short* input,
    unsigned short* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float x = bf16_bits_to_f32(input[index]);
        output[index] = f32_to_bf16_bits(x / (1.0f + expf(-x)));
    }
}

extern "C" __global__ void nnis_bf16_gelu_tanh_f32acc(
    const unsigned short* input,
    unsigned short* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float x = bf16_bits_to_f32(input[index]);
        const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        output[index] = f32_to_bf16_bits(0.5f * x * (1.0f + tanhf(inner)));
    }
}

extern "C" __global__ void nnis_bf16_widen_f32(
    const unsigned short* input,
    float* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = bf16_bits_to_f32(input[index]);
    }
}

extern "C" __global__ void nnis_bf16_narrow_from_f32(
    const float* input,
    unsigned short* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = f32_to_bf16_bits(input[index]);
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Context-bound bf16-storage/f32-compute elementwise kernels.
#[derive(Debug)]
pub struct Bf16Elementwise {
    vector_add: Kernel,
    scale: Kernel,
    affine: Kernel,
    relu: Kernel,
    silu: Kernel,
    gelu_tanh: Kernel,
    widen: Kernel,
    narrow: Kernel,
    block_size: u32,
}

impl Bf16Elementwise {
    /// Compile (or reuse cached CUBIN) and load the bf16 kernel set.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    /// Load the family with an explicitly selected thread-block width.
    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 {
            return Err(NnisError::invalid_input("bf16 block size is zero"));
        }
        let code = compiler.compile_cubin(BF16_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let vector_add = module.get_function("nnis_bf16_vector_add_f32acc")?;
        let scale = module.get_function("nnis_bf16_scale_f32acc")?;
        let affine = module.get_function("nnis_bf16_affine_f32acc")?;
        let widen = module.get_function("nnis_bf16_widen_f32")?;
        let narrow = module.get_function("nnis_bf16_narrow_from_f32")?;
        let relu = module.get_function("nnis_bf16_relu_f32acc")?;
        let silu = module.get_function("nnis_bf16_silu_f32acc")?;
        let gelu_tanh = module.get_function("nnis_bf16_gelu_tanh_f32acc")?;
        for (name, function) in [
            ("vector_add", &vector_add),
            ("scale", &scale),
            ("affine", &affine),
            ("relu", &relu),
            ("silu", &silu),
            ("gelu_tanh", &gelu_tanh),
            ("widen", &widen),
            ("narrow", &narrow),
        ] {
            let attributes = function.attributes()?;
            if block_size > attributes.max_threads_per_block {
                return Err(NnisError::invalid_input(format!(
                    "bf16 {name} block size {block_size} exceeds function limit {}",
                    attributes.max_threads_per_block
                )));
            }
        }
        Ok(Self {
            vector_add,
            scale,
            affine,
            relu,
            silu,
            gelu_tanh,
            widen,
            narrow,
            block_size,
        })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Widen bf16 storage to exact `f32` and wait.
    pub fn widen(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        // SAFETY: borrows retained until synchronization.
        unsafe { self.enqueue_widen(stream, input, output)? };
        stream.synchronize()
    }

    /// Narrow `f32` to bf16 storage (round-to-nearest-even) and wait.
    pub fn narrow(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        // SAFETY: borrows retained until synchronization.
        unsafe { self.enqueue_narrow(stream, input, output)? };
        stream.synchronize()
    }

    /// Add two equal-length bf16 buffers (f32 accumulate) and wait.
    pub fn vector_add(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<u16>,
        right: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        // SAFETY: borrows retained until synchronization.
        unsafe { self.enqueue_vector_add(stream, left, right, output)? };
        stream.synchronize()
    }

    /// Multiply each bf16 element by an `f32` scalar and wait.
    pub fn scale(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        scale: f32,
    ) -> Result<()> {
        // SAFETY: borrows retained until synchronization.
        unsafe { self.enqueue_scale(stream, input, output, scale)? };
        stream.synchronize()
    }

    /// Compute `output = input * scale + bias` (one rounded FMA on widened
    /// values) and wait.
    pub fn affine(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        scale: f32,
        bias: f32,
    ) -> Result<()> {
        // SAFETY: borrows retained until synchronization.
        unsafe { self.enqueue_affine(stream, input, output, scale, bias)? };
        stream.synchronize()
    }

    /// Apply rectified linear units over packed-bf16 storage and wait.
    pub fn relu(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        // SAFETY: borrows retained until synchronization.
        unsafe { self.enqueue_relu(stream, input, output)? };
        stream.synchronize()
    }

    /// Apply SiLU over packed-bf16 storage with f32 compute and wait.
    pub fn silu(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        // SAFETY: borrows retained until synchronization.
        unsafe { self.enqueue_silu(stream, input, output)? };
        stream.synchronize()
    }

    /// Apply tanh-approximated GELU over packed-bf16 storage and wait.
    pub fn gelu_tanh(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        // SAFETY: borrows retained until synchronization.
        unsafe { self.enqueue_gelu_tanh(stream, input, output)? };
        stream.synchronize()
    }

    /// # Safety
    ///
    /// The stream, both buffers, and this kernel set must remain alive and
    /// otherwise untouched until the stream completes.
    pub unsafe fn enqueue_relu(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        self.validate_lengths(&[input.len(), output.len()])?;
        if input.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(3, 2);
        args.push_buffer(input)
            .push_buffer(output)
            .push(input.len() as u64);
        self.launch(&self.relu, args, input.len(), stream)
    }

    /// # Safety
    ///
    /// See [`Self::enqueue_relu`].
    pub unsafe fn enqueue_silu(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        self.validate_lengths(&[input.len(), output.len()])?;
        if input.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(3, 2);
        args.push_buffer(input)
            .push_buffer(output)
            .push(input.len() as u64);
        self.launch(&self.silu, args, input.len(), stream)
    }

    /// # Safety
    ///
    /// See [`Self::enqueue_relu`].
    pub unsafe fn enqueue_gelu_tanh(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        self.validate_lengths(&[input.len(), output.len()])?;
        if input.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(3, 2);
        args.push_buffer(input)
            .push_buffer(output)
            .push(input.len() as u64);
        self.launch(&self.gelu_tanh, args, input.len(), stream)
    }

    /// # Safety
    ///
    /// All buffers and the stream must remain alive and otherwise untouched
    /// until the stream completes; see the typed wrappers.
    pub unsafe fn enqueue_widen(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        self.validate_lengths(&[input.len(), output.len()])?;
        if input.is_empty() {
            return Ok(());
        }
        self.launch_convert(
            &self.widen,
            input.device_ptr(),
            output.device_ptr(),
            input.len(),
            stream,
        )
    }

    /// # Safety
    ///
    /// See [`Self::enqueue_widen`].
    pub unsafe fn enqueue_narrow(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        self.validate_lengths(&[input.len(), output.len()])?;
        if input.is_empty() {
            return Ok(());
        }
        self.launch_convert(
            &self.narrow,
            input.device_ptr(),
            output.device_ptr(),
            input.len(),
            stream,
        )
    }

    /// # Safety
    ///
    /// See [`Self::enqueue_widen`].
    pub unsafe fn enqueue_vector_add(
        &self,
        stream: &Stream,
        left: &DeviceBuffer<u16>,
        right: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
    ) -> Result<()> {
        self.validate_lengths(&[left.len(), right.len(), output.len()])?;
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
            LaunchConfig::for_num_elements(output.len(), self.block_size)?,
        );
        // SAFETY: argument order/widths match `nnis_bf16_vector_add_f32acc`.
        unsafe { launch.launch(&mut args) }
    }

    /// # Safety
    ///
    /// See [`Self::enqueue_widen`].
    pub unsafe fn enqueue_scale(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        scale: f32,
    ) -> Result<()> {
        self.validate_lengths(&[input.len(), output.len()])?;
        if output.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(4, 2);
        args.push(input.device_ptr())
            .push(output.device_ptr())
            .push(scale)
            .push(input.len() as u64);
        self.launch(&self.scale, args, input.len(), stream)
    }

    /// # Safety
    ///
    /// See [`Self::enqueue_widen`].
    pub unsafe fn enqueue_affine(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        scale: f32,
        bias: f32,
    ) -> Result<()> {
        self.validate_lengths(&[input.len(), output.len()])?;
        if output.is_empty() {
            return Ok(());
        }
        let mut args = KernelArgs::with_capacity(5, 2);
        args.push(input.device_ptr())
            .push(output.device_ptr())
            .push(scale)
            .push(bias)
            .push(input.len() as u64);
        self.launch(&self.affine, args, input.len(), stream)
    }

    fn validate_lengths(&self, lengths: &[usize]) -> Result<()> {
        let expected = lengths[0];
        for (position, actual) in lengths.iter().enumerate().skip(1) {
            if *actual != expected {
                return Err(NnisError::invalid_input(format!(
                    "bf16 buffer {position} has {actual} elements; expected {expected}"
                )));
            }
        }
        Ok(())
    }

    /// Launcher for the conversion kernels: (in, out, count).
    fn launch_convert(
        &self,
        kernel: &Kernel,
        input: u64,
        output: u64,
        elements: usize,
        stream: &Stream,
    ) -> Result<()> {
        let mut args = KernelArgs::with_capacity(3, 0);
        args.push(input).push(output).push(elements as u64);
        let launch = KernelLaunch::new(
            kernel,
            stream,
            LaunchConfig::for_num_elements(elements, self.block_size)?,
        );
        // SAFETY: argument order/widths match the conversion signatures;
        // the enclosing typed wrapper owns lifetime obligations.
        unsafe { launch.launch(&mut args) }
    }

    /// Common launch tail: config from element count + submitted pack.
    fn launch(
        &self,
        kernel: &Kernel,
        mut args: KernelArgs<'_>,
        elements: usize,
        stream: &Stream,
    ) -> Result<()> {
        let launch = KernelLaunch::new(
            kernel,
            stream,
            LaunchConfig::for_num_elements(elements, self.block_size)?,
        );
        // SAFETY: each caller submits entries exactly matching its kernel's
        // declared parameter list in order and width.
        unsafe { launch.launch(&mut args) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::{bf16_bits_to_f32, f32_to_bf16_rne, gpu_context};

    const TEST_SIZES: &[usize] = &[1, 31, 255, 256, 4_097];

    fn host_values(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| ((i % 61) as f32 - 30.0) * 0.125)
            .collect()
    }

    #[test]
    fn bf16_kernels_match_host_oracles_bit_exact_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let bf16 = Bf16Elementwise::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &size in TEST_SIZES {
            let left_host = host_values(size);
            let right_host = (0..size)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.25)
                .collect::<Vec<_>>();
            let left_bf16: Vec<u16> = left_host.iter().copied().map(f32_to_bf16_rne).collect();
            let right_bf16: Vec<u16> = right_host.iter().copied().map(f32_to_bf16_rne).collect();
            let left = DeviceBuffer::from_host(&context, &stream, &left_bf16).unwrap();
            let right = DeviceBuffer::from_host(&context, &stream, &right_bf16).unwrap();
            let output = DeviceBuffer::<u16>::new(&context, size).unwrap();

            // Widening is exact.
            let wide = DeviceBuffer::<f32>::new(&context, size).unwrap();
            bf16.widen(&stream, &left, &wide).unwrap();
            for (index, value) in wide.to_vec(&stream).unwrap().iter().enumerate() {
                assert_eq!(
                    *value,
                    bf16_bits_to_f32(left_bf16[index]),
                    "widen at {index}"
                );
            }

            // Narrowing matches RNE.
            bf16.narrow(&stream, &wide, &output).unwrap();
            assert_eq!(output.to_vec(&stream).unwrap(), left_bf16);

            // Vector add: identical f32 arithmetic then identical rounding.
            bf16.vector_add(&stream, &left, &right, &output).unwrap();
            let actual = output.to_vec(&stream).unwrap();
            for index in 0..size {
                let expected_sum =
                    bf16_bits_to_f32(left_bf16[index]) + bf16_bits_to_f32(right_bf16[index]);
                assert_eq!(
                    actual[index],
                    f32_to_bf16_rne(expected_sum),
                    "add at {index}"
                );
            }

            // Scale: single multiply, no contraction possible.
            let scale = -1.375_f32;
            bf16.scale(&stream, &left, &output, scale).unwrap();
            let actual = output.to_vec(&stream).unwrap();
            for index in 0..size {
                let expected_product = bf16_bits_to_f32(left_bf16[index]) * scale;
                assert_eq!(
                    actual[index],
                    f32_to_bf16_rne(expected_product),
                    "scale at {index}"
                );
            }

            // Affine: explicit FMA on both sides.
            let bias = 0.875_f32;
            bf16.affine(&stream, &left, &output, scale, bias).unwrap();
            let actual = output.to_vec(&stream).unwrap();
            for index in 0..size {
                let expected_value = bf16_bits_to_f32(left_bf16[index]).mul_add(scale, bias);
                assert_eq!(
                    actual[index],
                    f32_to_bf16_rne(expected_value),
                    "affine at {index}"
                );
            }
        }
    }

    #[test]
    fn bf16_rejects_length_mismatches_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let bf16 = Bf16Elementwise::load(&context, &compiler).unwrap();
        assert!(Bf16Elementwise::load_with_block_size(&context, &compiler, 0).is_err());
        let stream = Stream::new(&context).unwrap();
        let short = DeviceBuffer::<u16>::new(&context, 3).unwrap();
        let long = DeviceBuffer::<u16>::new(&context, 4).unwrap();
        let error = bf16.vector_add(&stream, &short, &long, &short).unwrap_err();
        assert!(error.to_string().contains("expected 3"), "{error}");
    }
    #[test]
    fn bf16_activations_match_host_oracles_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let kernels = Bf16Elementwise::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        const TEST_SIZES: &[usize] = &[1, 63, 256, 1_025];
        for &size in TEST_SIZES {
            let input_host: Vec<f32> = (0..size)
                .map(|index| (index as f32 - size as f32 / 2.0) * 0.25)
                .collect();
            let input_bits: Vec<u16> = input_host.iter().copied().map(f32_to_bf16_rne).collect();
            let input = DeviceBuffer::from_host(&context, &stream, &input_bits).unwrap();
            let poisoned = vec![0xFFFF_u16; size];
            let output = DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();

            // ReLU: widen -> fmax(0) -> narrow is deterministic bit math.
            kernels.relu(&stream, &input, &output).unwrap();
            let actual = output.to_vec(&stream).unwrap();
            for index in 0..size {
                let expected = f32_to_bf16_rne(bf16_bits_to_f32(input_bits[index]).max(0.0));
                assert_eq!(
                    actual[index], expected,
                    "bf16 relu mismatch at {index} size {size}"
                );
            }

            // SiLU: transcendental, f64 tolerance oracle over widened values.
            let _ = output.copy_from_host(&stream, &poisoned);
            kernels.silu(&stream, &input, &output).unwrap();
            let actual = output.to_vec(&stream).unwrap();
            for index in 0..size {
                let x = f64::from(bf16_bits_to_f32(input_bits[index]));
                let expected_f32 = (x / (1.0 + (-x).exp())) as f32;
                assert!(
                    (f64::from(bf16_bits_to_f32(actual[index])) - f64::from(expected_f32)).abs()
                        <= 1.0e-2_f64.max(f64::from(expected_f32.abs()) * 1.0e-2),
                    "bf16 silu mismatch at {index} size {size}"
                );
            }

            // GELU (tanh approximation): same tolerance policy.
            let _ = output.copy_from_host(&stream, &poisoned);
            kernels.gelu_tanh(&stream, &input, &output).unwrap();
            let actual = output.to_vec(&stream).unwrap();
            for index in 0..size {
                let x = f64::from(bf16_bits_to_f32(input_bits[index]));
                let inner = 0.7978845608028654_f64 * (x + 0.044_715_f64 * x * x * x);
                let expected_f32 = (0.5 * x * (1.0 + inner.tanh())) as f32;
                assert!(
                    (f64::from(bf16_bits_to_f32(actual[index])) - f64::from(expected_f32)).abs()
                        <= 1.0e-2_f64.max(f64::from(expected_f32.abs()) * 1.0e-2),
                    "bf16 gelu mismatch at {index} size {size}"
                );
            }
        }
    }
}
