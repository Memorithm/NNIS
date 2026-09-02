use crate::ModelConfig;
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

pub const F16_ATTENTION_PLAN_VERSION: u32 = 1;

/// Cached-attention implementation selected by the explicit F16 attention plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F16CachedAttentionKernel {
    /// Qualified reference kernel with two block barriers per KV position.
    ReferencePerPositionBarriers,
    /// PR #77 candidate that stages serial softmax weights once in shared memory.
    StagedWeightsCandidate,
}

/// Versioned F16 attention-kernel policy, separate from numeric and projection plans.
///
/// The staged candidate is used only at or above `staged_min_kv_rows` and only
/// while the kernel can represent the active KV rows within its validated
/// dynamic-shared-memory limit. Otherwise execution falls back to the qualified
/// reference kernel. The fallback changes no arithmetic semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct F16AttentionPlan {
    pub schema_version: u32,
    pub kernel: F16CachedAttentionKernel,
    pub staged_min_kv_rows: usize,
}

impl F16AttentionPlan {
    /// Preserve the qualified reference attention path for every KV length.
    pub const fn reference() -> Self {
        Self {
            schema_version: F16_ATTENTION_PLAN_VERSION,
            kernel: F16CachedAttentionKernel::ReferencePerPositionBarriers,
            staged_min_kv_rows: 0,
        }
    }

    /// Explicit Thor candidate selected from KV row 16 onward.
    ///
    /// The threshold comes from the PR #77 physical sweep: the staged kernel was
    /// slower at one row, tied at two, effectively neutral at four/eight, and
    /// first showed a material isolated reduction at 16 rows. Runtime promotion
    /// still requires separate end-to-end evidence.
    pub const fn thor_staged_weights_candidate() -> Self {
        Self {
            schema_version: F16_ATTENTION_PLAN_VERSION,
            kernel: F16CachedAttentionKernel::StagedWeightsCandidate,
            staged_min_kv_rows: 16,
        }
    }

    pub fn validate(&self, config: &ModelConfig) -> Result<()> {
        if self.schema_version != F16_ATTENTION_PLAN_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported F16 attention-plan schema {}; expected {}",
                self.schema_version, F16_ATTENTION_PLAN_VERSION
            )));
        }
        config.validate_execution_support()?;
        match self.kernel {
            F16CachedAttentionKernel::ReferencePerPositionBarriers => {
                if self.staged_min_kv_rows != 0 {
                    return Err(NnisError::invalid_input(
                        "reference F16 attention plan requires staged_min_kv_rows=0",
                    ));
                }
            }
            F16CachedAttentionKernel::StagedWeightsCandidate => {
                if self.staged_min_kv_rows == 0
                    || self.staged_min_kv_rows > config.max_position_embeddings
                {
                    return Err(NnisError::invalid_input(format!(
                        "staged F16 attention threshold {} must be within 1..={}",
                        self.staged_min_kv_rows, config.max_position_embeddings
                    )));
                }
            }
        }
        Ok(())
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
            max_position_embeddings: 32,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10_000.0,
            activation: Activation::Silu,
            weight_dtype: WeightDType::F32,
        }
    }

    #[test]
    fn attention_plan_is_versioned_explicit_and_fail_closed() {
        let config = tiny_config();
        let reference = F16AttentionPlan::reference();
        reference.validate(&config).unwrap();

        let candidate = F16AttentionPlan::thor_staged_weights_candidate();
        candidate.validate(&config).unwrap();
        assert_eq!(candidate.staged_min_kv_rows, 16);
        assert_eq!(
            candidate.kernel,
            F16CachedAttentionKernel::StagedWeightsCandidate
        );

        let encoded = serde_json::to_string(&candidate).unwrap();
        assert!(encoded.contains("\"schema_version\":1"));
        assert!(encoded.contains("\"kernel\":\"staged_weights_candidate\""));
        assert!(encoded.contains("\"staged_min_kv_rows\":16"));

        let mut future = candidate;
        future.schema_version = F16_ATTENTION_PLAN_VERSION + 1;
        assert!(future.validate(&config).is_err());

        let mut invalid_reference = reference;
        invalid_reference.staged_min_kv_rows = 1;
        assert!(invalid_reference.validate(&config).is_err());
    }
}
