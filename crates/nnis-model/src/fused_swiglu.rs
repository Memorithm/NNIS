//! Candidate-only fused SwiGLU activation primitive.
//!
//! This module does not alter decoder policy. It exposes the exact operation
//! needed to test whether replacing the current `SiLU(gate)` launch followed by
//! `activated * up` with one kernel is worthwhile on physical hardware.

use nnis_jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const SILU_MULTIPLY_SOURCE: &str = r#"
extern "C" __global__ void nnis_silu_multiply_f32(
    const float* gate,
    const float* up,
    float* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        const float x = gate[index];
        const float activated = x / (1.0f + expf(-x));
        output[index] = activated * up[index];
    }
}
"#;

const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Candidate fused `SiLU(gate) * up` kernel.
///
/// The operation is intentionally kept outside the decoder plan until isolated
/// and end-to-end evidence justify integrating it. The current decoder remains
/// unchanged simply by adding this primitive.
#[derive(Debug)]
pub struct F32SiluMultiply {
    context: Arc<Context>,
    kernel: Kernel,
    block_size: u32,
}

impl F32SiluMultiply {
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_block_size(context, compiler, DEFAULT_BLOCK_SIZE)
    }

    pub fn load_with_block_size(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        block_size: u32,
    ) -> Result<Self> {
        if block_size == 0 {
            return Err(NnisError::invalid_input(
                "SiLU-multiply block size must be non-zero",
            ));
        }
        let code = compiler.compile_cubin(
            SILU_MULTIPLY_SOURCE,
            &CompileOptions::for_device(context),
        )?;
        let module = Module::load(context, &code)?;
        let kernel = module.get_function("nnis_silu_multiply_f32")?;
        let attributes = kernel.attributes()?;
        if block_size > attributes.max_threads_per_block {
            return Err(NnisError::invalid_input(format!(
                "SiLU-multiply block size {block_size} exceeds function limit {}",
                attributes.max_threads_per_block
            )));
        }
        Ok(Self {
            context: Arc::clone(context),
            kernel,
            block_size,
        })
    }

    #[must_use]
    pub const fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn silu_multiply(
        &self,
        stream: &Stream,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        // SAFETY: this method keeps all borrows alive until synchronization.
        unsafe { self.enqueue_silu_multiply(stream, gate, up, output)? };
        stream.synchronize()
    }

    /// Enqueue `output[i] = silu(gate[i]) * up[i]` without synchronizing.
    ///
    /// # Safety
    ///
    /// The stream, kernel object and all buffers must remain alive and otherwise
    /// untouched until the stream completes this launch.
    pub unsafe fn enqueue_silu_multiply(
        &self,
        stream: &Stream,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        if gate.len() != up.len() || gate.len() != output.len() {
            return Err(NnisError::invalid_input(format!(
                "SiLU-multiply length mismatch: gate={}, up={}, output={}",
                gate.len(),
                up.len(),
                output.len()
            )));
        }
        if !Arc::ptr_eq(&self.context, stream.ctx())
            || !Arc::ptr_eq(&self.context, gate.ctx())
            || !Arc::ptr_eq(&self.context, up.ctx())
            || !Arc::ptr_eq(&self.context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "SiLU-multiply kernel, stream and buffers must share one CUDA context",
            ));
        }
        if output.is_empty() {
            return Ok(());
        }

        let mut args = KernelArgs::with_capacity(4, 3);
        args.push_buffer(gate)
            .push_buffer(up)
            .push_buffer(output)
            .push(output.len() as u64);
        let launch = KernelLaunch::new(
            &self.kernel,
            stream,
            LaunchConfig::for_num_elements(output.len(), self.block_size)?,
        );
        // SAFETY: the argument layout exactly matches `nnis_silu_multiply_f32`;
        // remaining asynchronous lifetime obligations are documented above.
        unsafe { launch.launch(&mut args) }
    }
}
