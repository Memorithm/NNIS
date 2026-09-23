//! Model-neutral decoder-only transformer runtime for NNIS.
//!
//! This crate owns model configuration, device weight graphs and the exact
//! model-level CUDA operations that are not already covered by `nnis-kernels`.
//! It intentionally does not claim Hugging Face compatibility.

mod attention_plan;
mod cached_attention_candidate;
mod config;
mod da_luc_evidence;
mod da_luc_plan;
mod decoder_capabilities;
mod dense_weight_materialization;
mod exact_checkpoint_spec;
mod execution_transition;
mod f16_attention_plan;
mod f16_fused_mlp_candidate;
mod f16_fused_projection_candidate;
mod f16_materialization_memory;
mod f16_parallel_score_attention_candidate;
mod f16_reference_execution_plan;
mod f16_reference_kernels;
mod f16_reference_runtime;
mod f16_staged_attention_candidate;
mod f16_transposed_projection_candidate;
mod format;
mod fused_swiglu;
mod fusion_plan;
mod generated_token_evidence;
mod int2_model_storage;
mod int2_reference;
mod int4_reference;
mod kernels;
mod projection_plan;
mod representation_plan;
#[path = "runtime/mod.rs"]
mod runtime;
mod runtime_kernels;
mod safetensors_loader;
mod safetensors_preflight;
mod sampling;
mod session_batch;
mod sparse_csc_reference;
mod sparse_model_storage;
mod sparse_reference;
mod streaming;
mod weight_capability_manifest;
mod weight_full_model_evidence;
mod weight_qualification_bundle;
mod weight_representation_accounting;
mod weight_representation_qualification;
mod weighted_rmsnorm_candidate;
mod weights;

pub use attention_plan::{F32AttentionPlan, F32CachedAttentionKernel, F32_ATTENTION_PLAN_VERSION};
pub use cached_attention_candidate::F32CachedAttentionDecodeParallelValue;
pub use config::{Activation, GenerationConfig, ModelConfig, WeightDType};
pub use da_luc_evidence::{
    NnisDalucFlatOracleEvidence, NnisDalucFlatViewSnapshot, NnisDalucOracleErrorStats,
    NnisDalucOracleReconstructionEvidence, NnisDalucOracleStorageEvidence,
    FLAT_DA_LUC_ORACLE_REPOSITORY, NNIS_DA_LUC_ORACLE_EVIDENCE_VERSION,
    SUPPORTED_FLAT_DA_LUC_ORACLE_PAYLOAD_VERSION,
};
pub use da_luc_plan::{
    NnisDalucBackendCapabilities, NnisDalucBitOrder, NnisDalucCandidatePlan,
    NnisDalucCodebookScope, NnisDalucConsumptionMode, NnisDalucCudaPhysicalLayout,
    NnisDalucFloatDType, NnisDalucHeadGeometry, NnisDalucKeyRepresentation, NnisDalucPaddingRule,
    NnisDalucResidualSemantics, NnisDalucRowOrder, NnisDalucStorageTopology,
    NnisDalucValueRepresentation, NnisDalucViewLayout, NnisDalucZeroPointStorage,
    NnisKvExecutionPolicy, NNIS_DA_LUC_PLAN_VERSION, SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION,
};
pub use decoder_capabilities::{
    DecoderAttentionTopology, DecoderExecutionCapabilities, DecoderMlpSemantics,
    DecoderRopeSemantics, NNIS_DECODER_CAPABILITY_VERSION,
};
pub use dense_weight_materialization::{
    DenseWeightMaterializationEvidenceV1, DenseWeightMaterializationEvidenceV2,
    NNIS_DENSE_WEIGHT_MATERIALIZATION_EVIDENCE_V2_VERSION,
    NNIS_DENSE_WEIGHT_MATERIALIZATION_EVIDENCE_VERSION,
};
pub use exact_checkpoint_spec::{
    ExactDecoderCheckpointSpec, NNIS_EXACT_DECODER_CHECKPOINT_SPEC_VERSION, SMOLLM2_135M_BF16,
    TINYLLAMA_1P1B_CHAT_BF16,
};
pub use execution_transition::{
    F16ExecutionTransitionMode, F16ExecutionTransitionRequirementsV1,
    NNIS_F16_EXECUTION_TRANSITION_REQUIREMENTS_VERSION,
};
pub use f16_attention_plan::{
    F16AttentionPlan, F16CachedAttentionKernel, F16ParallelScorePolicy, F16_ATTENTION_PLAN_VERSION,
    F16_PARALLEL_SCORE_KA17_MAX_KV_ROWS,
};
pub use f16_fused_mlp_candidate::F16FusedMlpCandidate;
pub use f16_fused_projection_candidate::F16FusedProjectionGroupsCandidate;
pub use f16_materialization_memory::{
    F16WeightMaterializationEventKindV1, F16WeightMaterializationEventV1,
    F16WeightMaterializationFailureEvidenceV1, F16WeightMaterializationMemoryEvidenceV1,
    NNIS_F16_WEIGHT_MATERIALIZATION_FAILURE_EVIDENCE_VERSION,
    NNIS_F16_WEIGHT_MATERIALIZATION_MEMORY_EVIDENCE_VERSION,
};
pub use f16_parallel_score_attention_candidate::F16CachedAttentionParallelScoreCandidate;
pub use f16_reference_execution_plan::{
    F16ReferenceExecutionPlan, F16ReferenceProjectionLayout, F16_REFERENCE_EXECUTION_PLAN_VERSION,
};
pub use f16_reference_kernels::F16ReferenceKernels;
pub use f16_reference_runtime::{
    F16ReferenceAccumulator, F16ReferenceGenerationProfile, F16ReferenceLogits, F16ReferenceModel,
    F16ReferenceModelConstructionFailure, F16ReferencePlan, F16ReferenceSession,
    F16ReferenceStorage, F16_REFERENCE_PLAN_VERSION,
};
pub use f16_staged_attention_candidate::F16CachedAttentionStagedWeightsCandidate;
pub use f16_transposed_projection_candidate::F16TransposedProjectionCandidate;
pub use format::{
    load_model_directory, ModelManifest, TensorManifest, NNIS_MODEL_FORMAT, NNIS_MODEL_MANIFEST,
    NNIS_MODEL_VERSION,
};
pub use fused_swiglu::F32SiluMultiply;
pub use fusion_plan::{F32FusionPlan, F32SiluMultiplyKernel, F32_FUSION_PLAN_VERSION};
pub use generated_token_evidence::{
    validate_finite_runtime_output, GeneratedTokenEvidenceV1, NNIS_GENERATED_TOKEN_EVIDENCE_VERSION,
};
pub use int2_model_storage::{
    int2_dense_full_model_evidence_v1, Int2DenseMaterializedModelV1,
    Int2ReferenceAllocationSummaryV1, Int2ReferenceModelStorageV1, Int2ReferenceStorageSummaryV1,
};
pub use int2_reference::{
    dequantize_int2_ternary_reference_v1, quantize_int2_ternary_reference_v1,
    Int2ReferenceProjectionPlanV1, Int2ReferenceQuantizedTensorV1,
    NNIS_INT2_REFERENCE_ACCUMULATION_V1, NNIS_INT2_REFERENCE_CODE_NEGATIVE,
    NNIS_INT2_REFERENCE_CODE_POSITIVE, NNIS_INT2_REFERENCE_CODE_RESERVED,
    NNIS_INT2_REFERENCE_CODE_ZERO, NNIS_INT2_REFERENCE_DEQUANTIZATION_V1,
    NNIS_INT2_REFERENCE_PROJECTION_PLAN_VERSION, NNIS_INT2_REFERENCE_SERIALIZED_HEADER_BYTES,
    NNIS_INT2_REFERENCE_STORAGE_VERSION,
};
pub use int4_reference::{
    dequantize_int4_symmetric_reference_v1, int4_dense_full_model_evidence_v1,
    quantize_int4_symmetric_reference_v1, Int4DenseMaterializedModelV1,
    Int4ReferenceAllocationSummaryV1, Int4ReferenceModelStorageV1, Int4ReferenceProjectionPlanV1,
    Int4ReferenceQuantizedTensorV1, Int4ReferenceStorageSummaryV1,
    NNIS_INT4_REFERENCE_ACCUMULATION_V1, NNIS_INT4_REFERENCE_DEQUANTIZATION_V1,
    NNIS_INT4_REFERENCE_PROJECTION_PLAN_VERSION, NNIS_INT4_REFERENCE_QUANT_MAX,
    NNIS_INT4_REFERENCE_QUANT_MIN, NNIS_INT4_REFERENCE_SERIALIZED_HEADER_BYTES,
    NNIS_INT4_REFERENCE_STORAGE_VERSION,
};
pub use kernels::F32DecoderKernels;
pub use nnis_rt::KvCacheTelemetry;
pub use projection_plan::{F32ProjectionKernel, F32ProjectionPlan};
pub use representation_plan::{
    load_model_directory_with_representation_plan, PhysicalWeightRepresentation,
    WeightRepresentationPlan, WEIGHT_REPRESENTATION_PLAN_VERSION,
};
pub use runtime::{InferenceSession, Model};
pub use runtime_kernels::F32RuntimeKernels;
pub use safetensors_loader::{
    load_model_from_safetensors, load_model_from_safetensors_f32, LoadedSafetensorsModel,
    SafetensorsLoadConfig, SafetensorsMetadata,
};
pub use safetensors_preflight::{
    preflight_hf_safetensors_source, HfSafetensorsPreflightReportV1,
    NNIS_HF_SAFETENSORS_PREFLIGHT_VERSION,
};
pub use sampling::{SamplingConfig, NNIS_SAMPLING_POLICY_VERSION};
pub use session_batch::{SampledBatchRequest, SampledSessionBatch};
pub use sparse_csc_reference::{
    densify_matrix_csc_reference_v1, sparsify_matrix_csc_reference_v1, SparseCscProjectionPlanV1,
    SparseCscReferenceMatrixV1, NNIS_SPARSE_CSC_ACCUMULATION_V1,
    NNIS_SPARSE_CSC_PROJECTION_PLAN_VERSION, NNIS_SPARSE_CSC_REFERENCE_VERSION,
    NNIS_SPARSE_CSC_SERIALIZED_HEADER_BYTES,
};
pub use sparse_model_storage::{
    sparse_dense_full_model_evidence_v1, SparseDenseMaterializedModelV1,
    SparseReferenceAllocationSummaryV1, SparseReferenceModelStorageSummaryV1,
    SparseReferenceModelStorageV1, NNIS_SPARSE_MODEL_STORAGE_VERSION,
};
pub use sparse_reference::{
    densify_sparse_reference_v1, sparsify_magnitude_reference_v1, SparseReferenceTensorV1,
    NNIS_SPARSE_REFERENCE_SERIALIZED_HEADER_BYTES, NNIS_SPARSE_REFERENCE_STORAGE_VERSION,
};
pub use streaming::GenerationStreamControl;
pub use weight_capability_manifest::{
    reference_weight_capability_manifest_v1, WeightCapabilityManifestV1,
    WeightRepresentationCapabilityV1, NNIS_WEIGHT_CAPABILITY_MANIFEST_VERSION,
};
pub use weight_full_model_evidence::{
    PhysicalWeightExecutionObservationV1, WeightFullModelExecutionEvidenceV1,
    NNIS_PHYSICAL_WEIGHT_EXECUTION_OBSERVATION_VERSION, NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION,
};
pub use weight_qualification_bundle::{
    WeightQualificationBundleV1, NNIS_WEIGHT_QUALIFICATION_BUNDLE_VERSION,
};
pub use weight_representation_accounting::{
    CanonicalWeightDenominatorV1, WeightRepresentationAccountingV1,
    NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION,
};
pub use weight_representation_qualification::{
    int2_storage_qualification_record_v1, int4_storage_qualification_record_v1,
    WeightExecutionQualificationLevelV1, WeightRepresentationFamilyV1,
    WeightRepresentationQualificationRecordV1, NNIS_WEIGHT_REPRESENTATION_QUALIFICATION_VERSION,
};
pub use weighted_rmsnorm_candidate::F32WeightedRmsNormCandidate;
pub use weights::{
    DecoderLayerWeights, DeviceTensor, MatrixWeight, ModelWeights, VectorWeight,
    WeightAllocationDTypeV1, WeightAllocationSegmentV1, WeightAllocationSummaryV1,
    NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION,
};
