//! Public facade for the NVIDIA Native Inference Stack.
//!
//! Common runtime, JIT, and kernel workflows are available from this crate;
//! raw CUDA FFI remains confined to `nnis-sys`.

use std::sync::Arc;

/// Safe CUDA device, context, stream, event, and allocation APIs.
pub mod runtime {
    pub use nnis_rt::{
        gpu_context, Context, Device, DeviceBuffer, DevicePod, DeviceProps, ErrorKind, Event,
        NnisError, PinnedBuffer, Result, Stream,
    };
}

/// Runtime compilation, module ownership, and validated launch APIs.
pub mod jit {
    pub use nnis_jit::{
        CodeKind, CompileOptions, CompiledCode, Dim3, JitCompiler, JitProgram, Kernel, KernelArgs,
        KernelLaunch, KernelParameter, LaunchConfig, Module, ProgramCacheKey,
    };
}

/// Reusable NNIS native kernel families.
pub mod kernels {
    pub use nnis_kernels::F32Elementwise;
}

pub use jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
pub use kernels::F32Elementwise;
pub use runtime::{
    Context, Device, DeviceBuffer, DevicePod, DeviceProps, ErrorKind, Event, NnisError,
    PinnedBuffer, Result, Stream,
};

/// Imports for the typical NNIS execution path.
pub mod prelude {
    pub use crate::{
        CompileOptions, Context, Device, DeviceBuffer, DevicePod, Event, F32Elementwise,
        JitCompiler, KernelArgs, KernelLaunch, LaunchConfig, Module, NnisError, Result, Session,
        Stream,
    };
}

/// Ready-to-use context, stream, compiler cache, and standard kernel set.
#[derive(Debug)]
pub struct Session {
    context: Arc<Context>,
    stream: Stream,
    compiler: JitCompiler,
    elementwise: F32Elementwise,
}

impl Session {
    /// Create a session on the first visible CUDA device.
    pub fn first() -> Result<Self> {
        let device = Device::first()?;
        Self::new(&device)
    }

    /// Create a session for an explicitly selected device.
    pub fn new(device: &Device) -> Result<Self> {
        Self::from_context(Context::new(device)?)
    }

    /// Build a session around an existing primary-context reference.
    pub fn from_context(context: Arc<Context>) -> Result<Self> {
        let stream = Stream::new(&context)?;
        let compiler = JitCompiler::new();
        let elementwise = F32Elementwise::load(&context, &compiler)?;
        Ok(Self {
            context,
            stream,
            compiler,
            elementwise,
        })
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn compiler(&self) -> &JitCompiler {
        &self.compiler
    }

    pub fn elementwise(&self) -> &F32Elementwise {
        &self.elementwise
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

    #[test]
    fn session_executes_standard_kernel_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let session = Session::from_context(context).unwrap();
        let host = (0..257)
            .map(|index| index as f32 * 0.25 - 4.0)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(session.context(), session.stream(), &host).unwrap();
        let output = DeviceBuffer::<f32>::new(session.context(), host.len()).unwrap();

        session
            .elementwise()
            .affine(session.stream(), &input, &output, -0.5, 1.25)
            .unwrap();
        let actual = output.to_vec(session.stream()).unwrap();
        for (index, (&actual, &input)) in actual.iter().zip(&host).enumerate() {
            assert_eq!(actual, input.mul_add(-0.5, 1.25), "mismatch at {index}");
        }
    }
}
