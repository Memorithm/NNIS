use crate::{F16ReferencePlan, ModelConfig};
use nnis_rt::{NnisError, Result};
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
    /// transposed kernels. This variant is qualification-only until a separate
    /// end-to-end promotion gate succeeds.
    NkTransposedFusedGroupsCandidate,
    /// Candidate layout: resident `[N, K]`, grouped QKV, and fused gate/up/SiLU.
    ///
    /// The MLP candidate preserves the existing F16 projection, SiLU and product
    /// rounding boundaries while replacing grouped gate/up + SiLU with one launch.
    /// O-projection, down-projection, LM-head and attention remain unchanged.
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

    /// Explicit grouped-launch candidate measured in PR #79 / KA9.
    pub const fn edge_llm_v0_10_0_transposed_fused_groups_candidate() -> Self {
        Self {
            schema_version: F16_REFERENCE_EXECUTION_PLAN_VERSION,
            numeric: F16ReferencePlan::edge_llm_v0_10_0_alignment(),
            projection_layout: F16ReferenceProjectionLayout::NkTransposedFusedGroupsCandidate,
        }
    }

    /// Explicit fused-MLP candidate measured in PR #82 / KA12.
    pub const fn edge_llm_v0_10_0_transposed_fused_mlp_candidate() -> Self {
        Self {
            schema_version: F16_REFERENCE_EXECUTION_PLAN_VERSION,
            numeric: F16ReferencePlan::edge_llm_v0_10_0_alignment(),
            projection_layout: F16ReferenceProjectionLayout::NkTransposedFusedMlpCandidate,
        }
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
    use crate::{Activation, WeightDType};

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
}
