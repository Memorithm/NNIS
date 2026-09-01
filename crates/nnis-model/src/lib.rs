//! Model-neutral decoder-only transformer runtime for NNIS.
//!
//! This crate owns model configuration, device weight graphs and the exact
//! model-level CUDA operations that are not already covered by `nnis-kernels`.
//! It intentionally does not claim Hugging Face compatibility.

mod attention_plan;
mod cached_attention_candidate;
mod config;
mod f16_reference_execution_plan;
mod f16_reference_kernels;
mod f16_reference_runtime;
mod f16_staged_attention_candidate;
mod f16_transposed_projection_candidate;
mod format;
mod fused_swiglu;
mod fusion_plan;
mod kernels;
mod projection_plan;
mod representation_plan;
mod runtime;
mod runtime_kernels;
mod weighted_rmsnorm_candidate;
mod weights;

pub use attention_plan::{F32AttentionPlan, F32CachedAttentionKernel, F32_ATTENTION_PLAN_VERSION};
pub use cached_attention_candidate::F32CachedAttentionDecodeParallelValue;
pub use config::{Activation, GenerationConfig, ModelConfig, WeightDType};
pub use f16_reference_execution_plan::{
    F16ReferenceExecutionPlan, F16ReferenceProjectionLayout, F16_REFERENCE_EXECUTION_PLAN_VERSION,
};
pub use f16_reference_kernels::F16ReferenceKernels;
pub use f16_reference_runtime::{
    F16ReferenceAccumulator, F16ReferenceGenerationProfile, F16ReferenceLogits, F16ReferenceModel,
    F16ReferencePlan, F16ReferenceSession, F16ReferenceStorage, F16_REFERENCE_PLAN_VERSION,
};
pub use f16_staged_attention_candidate::F16CachedAttentionStagedWeightsCandidate;
pub use f16_transposed_projection_candidate::F16TransposedProjectionCandidate;
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
pub use weighted_rmsnorm_candidate::F32WeightedRmsNormCandidate;
pub use weights::{DecoderLayerWeights, DeviceTensor, MatrixWeight, ModelWeights, VectorWeight};
