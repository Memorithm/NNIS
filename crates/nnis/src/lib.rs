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
        KernelAttributes, KernelLaunch, KernelParameter, LaunchConfig, Module,
        OccupancyRecommendation, ProgramCacheKey,
    };
}

/// Reusable NNIS native kernel families.
pub mod kernels {
    pub use nnis_kernels::{
        AttentionMask, Bf16Elementwise, Bf16Gather, Bf16Gemm, Bf16Reduction,
        Bf16ReductionWorkspace, Bf16Scatter, F32Attention, F32Elementwise,
        F32ElementwiseActiveBlocks, F32ElementwiseOccupancy, F32Gather, F32Gemm, F32Gemv,
        F32LayerNorm, F32LayerNormWorkspace, F32Reduction, F32ReductionWorkspace, F32RmsNorm,
        F32Rope, F32Scatter, F32Softmax, F32Softmax2D, F32Softmax2DWorkspace, F32TopK,
        F32TopKWorkspace,
    };
}

pub use jit::{
    CompileOptions, JitCompiler, Kernel, KernelArgs, KernelAttributes, KernelLaunch, LaunchConfig,
    Module, OccupancyRecommendation,
};
pub use kernels::{
    AttentionMask, Bf16Elementwise, Bf16Gather, Bf16Gemm, Bf16Reduction, Bf16ReductionWorkspace,
    Bf16Scatter, F32Attention, F32Elementwise, F32ElementwiseActiveBlocks, F32ElementwiseOccupancy,
    F32Gather, F32Gemm, F32Gemv, F32LayerNorm, F32LayerNormWorkspace, F32Reduction,
    F32ReductionWorkspace, F32RmsNorm, F32Rope, F32Scatter, F32Softmax, F32Softmax2D,
    F32Softmax2DWorkspace, F32TopK, F32TopKWorkspace,
};
pub use runtime::{
    Context, Device, DeviceBuffer, DevicePod, DeviceProps, ErrorKind, Event, NnisError,
    PinnedBuffer, Result, Stream,
};

/// Imports for the typical NNIS execution path.
pub mod prelude {
    pub use crate::{
        CompileOptions, Context, Device, DeviceBuffer, DevicePod, Event, F32Elementwise, F32Gemm,
        F32Gemv, F32LayerNorm, F32Reduction, F32RmsNorm, F32Softmax, F32Softmax2D, JitCompiler,
        KernelArgs, KernelLaunch, LaunchConfig, Module, NnisError, Result, Session, Stream,
    };
}

/// Ready-to-use context, stream, compiler cache, and standard kernel set.
#[derive(Debug)]
pub struct Session {
    context: Arc<Context>,
    stream: Stream,
    compiler: JitCompiler,
    elementwise: F32Elementwise,
    reduction: F32Reduction,
    softmax: F32Softmax,
    softmax_2d: F32Softmax2D,
    gemv: F32Gemv,
    gemm: F32Gemm,
    layer_norm: F32LayerNorm,
    rms_norm: F32RmsNorm,
    bf16_elementwise: Bf16Elementwise,
    bf16_gemm: Bf16Gemm,
    bf16_reduction: Bf16Reduction,
    rope: F32Rope,
    top_k: F32TopK,
    attention: F32Attention,
    gather: F32Gather,
    bf16_gather: Bf16Gather,
    scatter: F32Scatter,
    bf16_scatter: Bf16Scatter,
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
        let reduction = F32Reduction::load(&context, &compiler)?;
        let softmax = F32Softmax::load(&context, &compiler)?;
        let softmax_2d = F32Softmax2D::load(&context, &compiler)?;
        let gemv = F32Gemv::load(&context, &compiler)?;
        let gemm = F32Gemm::load(&context, &compiler)?;
        let layer_norm = F32LayerNorm::load(&context, &compiler)?;
        let rms_norm = F32RmsNorm::load(&context, &compiler)?;
        let bf16_elementwise = Bf16Elementwise::load(&context, &compiler)?;
        let bf16_gemm = Bf16Gemm::load(&context, &compiler)?;
        let bf16_reduction = Bf16Reduction::load(&context, &compiler)?;
        let rope = F32Rope::load(&context, &compiler)?;
        let top_k = F32TopK::load(&context, &compiler)?;
        let attention = F32Attention::load(&context, &compiler)?;
        let gather = F32Gather::load(&context, &compiler)?;
        let bf16_gather = Bf16Gather::load(&context, &compiler)?;
        let scatter = F32Scatter::load(&context, &compiler)?;
        let bf16_scatter = Bf16Scatter::load(&context, &compiler)?;
        Ok(Self {
            context,
            stream,
            compiler,
            elementwise,
            reduction,
            softmax,
            softmax_2d,
            gemv,
            gemm,
            layer_norm,
            rms_norm,
            bf16_elementwise,
            bf16_gemm,
            bf16_reduction,
            rope,
            top_k,
            attention,
            gather,
            bf16_gather,
            scatter,
            bf16_scatter,
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

    pub fn reduction(&self) -> &F32Reduction {
        &self.reduction
    }

    pub fn softmax(&self) -> &F32Softmax {
        &self.softmax
    }

    pub fn softmax_2d(&self) -> &F32Softmax2D {
        &self.softmax_2d
    }

    pub fn gemv(&self) -> &F32Gemv {
        &self.gemv
    }

    pub fn gemm(&self) -> &F32Gemm {
        &self.gemm
    }

    pub fn layer_norm(&self) -> &F32LayerNorm {
        &self.layer_norm
    }

    pub fn rms_norm(&self) -> &F32RmsNorm {
        &self.rms_norm
    }

    pub fn bf16_elementwise(&self) -> &Bf16Elementwise {
        &self.bf16_elementwise
    }

    pub fn bf16_gemm(&self) -> &Bf16Gemm {
        &self.bf16_gemm
    }

    pub fn bf16_reduction(&self) -> &Bf16Reduction {
        &self.bf16_reduction
    }

    pub fn top_k(&self) -> &F32TopK {
        &self.top_k
    }

    pub fn rope(&self) -> &F32Rope {
        &self.rope
    }

    pub fn attention(&self) -> &F32Attention {
        &self.attention
    }

    pub fn gather(&self) -> &F32Gather {
        &self.gather
    }

    pub fn bf16_gather(&self) -> &Bf16Gather {
        &self.bf16_gather
    }

    pub fn scatter(&self) -> &F32Scatter {
        &self.scatter
    }

    pub fn bf16_scatter(&self) -> &Bf16Scatter {
        &self.bf16_scatter
    }

    /// Composed scaled dot-product attention over this session's kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_composed(
        &self,
        queries: &DeviceBuffer<f32>,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        query_rows: usize,
        head_dim: usize,
        kv_rows: usize,
        value_dim: usize,
        scale: f32,
        mask: kernels::AttentionMask,
    ) -> Result<()> {
        self.attention.attention_composed(
            self.gemm(),
            self.elementwise(),
            self.softmax_2d(),
            &self.stream,
            queries,
            keys,
            values,
            output,
            query_rows,
            head_dim,
            kv_rows,
            value_dim,
            scale,
            mask,
        )
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
        let expected_sum = host.iter().sum::<f32>();
        assert_eq!(
            session.reduction().sum(session.stream(), &input).unwrap(),
            expected_sum
        );
        let probabilities = DeviceBuffer::<f32>::new(session.context(), host.len()).unwrap();
        session
            .softmax()
            .softmax(session.context(), session.stream(), &input, &probabilities)
            .unwrap();
        let actual = probabilities.to_vec(session.stream()).unwrap();
        let maximum = host
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |acc, value| acc.max(f64::from(value)));
        let total: f64 = host
            .iter()
            .map(|&value| f64::from(value) - maximum)
            .map(f64::exp)
            .sum();
        for (index, (&actual, &input)) in actual.iter().zip(&host).enumerate() {
            let expected = ((f64::from(input) - maximum).exp() / total) as f32;
            assert!(
                (actual - expected).abs() <= 1.0e-5_f32.max(expected.abs() * 1.0e-5),
                "softmax mismatch at {index}: {actual} != {expected}"
            );
        }
    }
}
