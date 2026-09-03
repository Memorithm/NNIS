use crate::{Activation, F16ReferencePlan, ModelConfig};
use nnis_rt::{Context, NnisError, Result};
use serde::{Deserialize, Serialize};

pub const F16_REFERENCE_EXECUTION_PLAN_VERSION: u32 = 1;

/// Physical resident layout and candidate launch policy used by F16 decoder projections.
///
/// This is deliberately separate from [`F16ReferencePlan`], which remains the
/// stable numeric contract. Candidate variants never change arithmetic or weight
/// storage precision; they only select an explicit physical representation and
/// launch envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F16ReferenceProjectionLayout {
    /// Qualified reference layout: model-format orientation `[K, N]`.
    KnReference,
    /// Candidate layout: resident one-time transpose `[N, K]`.
    NkTransposedCandidate,
    /// Candidate layout: resident `[N, K]` plus grouped QKV and gate/up launches.
    ///
    /// O-projection, down-projection and LM-head launches remain the existing
    /// transposed kernels. This variant remains an explicit comparison surface.
    NkTransposedFusedGroupsCandidate,
    /// Resident `[N, K]`, grouped QKV, and fused gate/up/SiLU.
    ///
    /// The schema name is retained for compatibility with the qualification
    /// evidence produced while this layout was still candidate-only. KA13 and
    /// KA14 subsequently qualified this layout for the pinned SmolLM2-135M
    /// workload on NVIDIA Thor. Generic constructors still do not select it.
    NkTransposedFusedMlpCandidate,
}

/// Versioned physical execution plan for the NNML5 F16 runtime.
///
/// Existing `F16ReferenceModel::{new,load_directory}` APIs continue to use
/// [`Self::reference`] and therefore preserve the qualified `[K, N]` path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct F16ReferenceExecutionPlan {
    pub schema_version: u32,
    pub numeric: F16ReferencePlan,
    pub projection_layout: F16ReferenceProjectionLayout,
}

impl F16ReferenceExecutionPlan {
    /// Preserve the qualified physical projection layout for an existing numeric plan.
    pub const fn reference(numeric: F16ReferencePlan) -> Self {
        Self {
            schema_version: F16_REFERENCE_EXECUTION_PLAN_VERSION,
            numeric,
            projection_layout: F16ReferenceProjectionLayout::KnReference,
        }
    }

    /// Explicit Thor candidate measured in PR #74.
    pub const fn edge_llm_v0_10_0_transposed_projection_candidate() -> Self {
        Self {
            schema_version: F16_REFERENCE_EXECUTION_PLAN_VERSION,
            numeric: F16ReferencePlan::edge_llm_v0_10_0_alignment(),
            projection_layout: F16ReferenceProjectionLayout::NkTransposedCandidate,
        }
    }

    /// Explicit grouped-launch comparison surface measured in PR #79 / KA9.
    pub const fn edge_llm_v0_10_0_transposed_fused_groups_candidate() -> Self {
        Self {
            schema_version: F16_REFERENCE_EXECUTION_PLAN_VERSION,
            numeric: F16ReferencePlan::edge_llm_v0_10_0_alignment(),
            projection_layout: F16ReferenceProjectionLayout::NkTransposedFusedGroupsCandidate,
        }
    }

    /// Historical candidate constructor retained for evidence/schema compatibility.
    pub const fn edge_llm_v0_10_0_transposed_fused_mlp_candidate() -> Self {
        Self {
            schema_version: F16_REFERENCE_EXECUTION_PLAN_VERSION,
            numeric: F16ReferencePlan::edge_llm_v0_10_0_alignment(),
            projection_layout: F16ReferenceProjectionLayout::NkTransposedFusedMlpCandidate,
        }
    }

    /// Qualified min-latency plan for the pinned SmolLM2-135M execution domain on Thor.
    ///
    /// Promotion basis: two independent eight-round ABBA campaigns (KA13 and KA14)
    /// on the exact runtime commit `4101b8924f1e5400a7871259b9c1b732ae3c77bb`.
    /// All 16 paired observations favored this layout, both runs preserved the
    /// qualified 32-token greedy trajectory, and the median paired improvement
    /// across runs was 7.77%. This constructor is deliberately fail-closed outside
    /// the measured SmolLM2 geometry and NVIDIA Thor GPU class.
    pub fn smollm2_135m_thor_min_latency(config: &ModelConfig, context: &Context) -> Result<Self> {
        let plan = Self::edge_llm_v0_10_0_transposed_fused_mlp_candidate();
        plan.validate_smollm2_135m_min_latency_model_domain(config)?;

        let properties = context.props();
        if properties.name != "NVIDIA Thor"
            || properties.compute_capability != (11, 0)
            || properties.multiprocessor_count != 20
        {
            return Err(NnisError::unsupported(format!(
                "qualified SmolLM2 F16 min-latency plan requires NVIDIA Thor cc 11.0 with 20 SMs; observed {} cc {}.{} with {} SMs",
                properties.name,
                properties.compute_capability.0,
                properties.compute_capability.1,
                properties.multiprocessor_count
            )));
        }
        Ok(plan)
    }

    /// Validate the model-side qualification domain independently of CUDA hardware.
    ///
    /// This does not authorize the plan by itself; use
    /// [`Self::smollm2_135m_thor_min_latency`] for the hardware-scoped selector.
    pub fn validate_smollm2_135m_min_latency_model_domain(
        &self,
        config: &ModelConfig,
    ) -> Result<()> {
        self.validate(config)?;
        if *self != Self::edge_llm_v0_10_0_transposed_fused_mlp_candidate() {
            return Err(NnisError::unsupported(
                "SmolLM2 F16 min-latency qualification requires the fused-MLP execution layout",
            ));
        }
        if config.vocab_size != 49_152
            || config.eos_token_id != Some(0)
            || config.hidden_size != 576
            || config.intermediate_size != 1_536
            || config.num_hidden_layers != 30
            || config.num_attention_heads != 9
            || config.num_key_value_heads != 3
            || config.max_position_embeddings != 8_192
            || config.rms_norm_eps != 1.0e-5
            || config.rope_theta != 100_000.0
            || config.activation != Activation::Silu
        {
            return Err(NnisError::unsupported(
                "qualified F16 min-latency plan is restricted to the pinned SmolLM2-135M model geometry",
            ));
        }
        Ok(())
    }

    pub fn validate(&self, config: &ModelConfig) -> Result<()> {
        if self.schema_version != F16_REFERENCE_EXECUTION_PLAN_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported F16 reference execution-plan schema {}; expected {}",
                self.schema_version, F16_REFERENCE_EXECUTION_PLAN_VERSION
            )));
        }
        self.numeric.validate(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightDType;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 4,
            eos_token_id: Some(0),
            hidden_size: 4,
            intermediate_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 8,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10_000.0,
            activation: Activation::Silu,
            weight_dtype: WeightDType::F32,
        }
    }

    fn smollm2_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 49_152,
            eos_token_id: Some(0),
            hidden_size: 576,
            intermediate_size: 1_536,
            num_hidden_layers: 30,
            num_attention_heads: 9,
            num_key_value_heads: 3,
            max_position_embeddings: 8_192,
            rms_norm_eps: 1.0e-5,
            rope_theta: 100_000.0,
            activation: Activation::Silu,
            weight_dtype: WeightDType::F32,
        }
    }

    #[test]
    fn reference_execution_plan_is_versioned_explicit_and_fail_closed() {
        let config = tiny_config();
        let numeric = F16ReferencePlan::edge_llm_v0_10_0_alignment();
        let reference = F16ReferenceExecutionPlan::reference(numeric);
        reference.validate(&config).unwrap();
        assert_eq!(
            reference.projection_layout,
            F16ReferenceProjectionLayout::KnReference
        );

        let candidate =
            F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_projection_candidate();
        candidate.validate(&config).unwrap();
        assert_eq!(
            candidate.projection_layout,
            F16ReferenceProjectionLayout::NkTransposedCandidate
        );
        let encoded = serde_json::to_string(&candidate).unwrap();
        assert!(encoded.contains("\"schema_version\":1"));
        assert!(encoded.contains("\"projection_layout\":\"nk_transposed_candidate\""));
        assert!(encoded.contains("\"numeric\""));

        let fused = F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_fused_groups_candidate();
        fused.validate(&config).unwrap();
        assert_eq!(
            fused.projection_layout,
            F16ReferenceProjectionLayout::NkTransposedFusedGroupsCandidate
        );
        let encoded_fused = serde_json::to_string(&fused).unwrap();
        assert!(encoded_fused
            .contains("\"projection_layout\":\"nk_transposed_fused_groups_candidate\""));

        let fused_mlp =
            F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_fused_mlp_candidate();
        fused_mlp.validate(&config).unwrap();
        assert_eq!(
            fused_mlp.projection_layout,
            F16ReferenceProjectionLayout::NkTransposedFusedMlpCandidate
        );
        let encoded_fused_mlp = serde_json::to_string(&fused_mlp).unwrap();
        assert!(encoded_fused_mlp
            .contains("\"projection_layout\":\"nk_transposed_fused_mlp_candidate\""));

        let mut future = candidate;
        future.schema_version = F16_REFERENCE_EXECUTION_PLAN_VERSION + 1;
        assert!(future.validate(&config).is_err());
    }

    #[test]
    fn smollm2_min_latency_model_domain_is_fail_closed() {
        let plan = F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_fused_mlp_candidate();
        plan.validate_smollm2_135m_min_latency_model_domain(&smollm2_config())
            .unwrap();
        assert!(plan
            .validate_smollm2_135m_min_latency_model_domain(&tiny_config())
            .is_err());
        assert!(
            F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_fused_groups_candidate()
                .validate_smollm2_135m_min_latency_model_domain(&smollm2_config())
                .is_err()
        );
    }
}
