//! Model-neutral decoder-only transformer runtime for NNIS.
//!
//! This crate owns model configuration, device weight graphs and the exact
//! model-level CUDA operations that are not already covered by `nnis-kernels`.
//! It intentionally does not claim Hugging Face compatibility.

mod cached_attention_candidate;
mod config;
mod format;
mod fused_swiglu;
mod fusion_plan;
mod kernels;
mod projection_plan;
mod representation_plan;
mod runtime;
mod runtime_kernels;
mod weights;

pub use cached_attention_candidate::F32CachedAttentionDecodeParallelValue;
pub use config::{Activation, GenerationConfig, ModelConfig, WeightDType};
pub use format::{
    load_model_directory, ModelManifest, TensorManifest, NNIS_MODEL_FORMAT, NNIS_MODEL_MANIFEST,
    NNIS_MODEL_VERSION,
};
pub use fused_swiglu::F32SiluMultiply;
pub use fusion_plan::{F32FusionPlan, F32SiluMultiplyKernel, F32_FUSION_PLAN_VERSION};
pub use kernels::F32DecoderKernels;
pub use projection_plan::{F32ProjectionKernel, F32ProjectionPlan};
pub use representation_plan::{
    load_model_directory_with_representation_plan, PhysicalWeightRepresentation,
    WeightRepresentationPlan, WEIGHT_REPRESENTATION_PLAN_VERSION,
};
pub use runtime::{InferenceSession, Model};
pub use runtime_kernels::F32RuntimeKernels;
pub use weights::{DecoderLayerWeights, DeviceTensor, MatrixWeight, ModelWeights, VectorWeight};
