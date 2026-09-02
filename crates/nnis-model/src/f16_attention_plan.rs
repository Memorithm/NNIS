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
    /// KA17-qualified candidate with warp-parallel Q·K score reduction.
    ///
    /// The launch policy is evidence-bounded to the pinned SmolLM2 short-context
    /// geometry. Execution falls back to the qualified reference kernel outside
    /// the measured KV range rather than extrapolating an unqualified launch.
    ParallelScoreKa17Candidate,
}

/// Versioned F16 attention-kernel policy, separate from numeric and projection plans.
///
/// The staged candidate is used only at or above `staged_min_kv_rows` and only
/// while the kernel can represent the active KV rows within its validated
/// dynamic-shared-memory limit. The KA17 parallel-score plan instead carries its
/// fixed launch policy in the enum variant itself; `staged_min_kv_rows` must be
/// zero for that variant. Existing reference and staged plan serialization remains
/// unchanged.
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

    /// Explicit KA17 launch policy for the pinned SmolLM2 F16 attention geometry.
    ///
    /// KA17 fixed this policy before its dense-row confirmation campaign and then
    /// observed bitwise-equal F16 outputs for all 840 tested cases. The policy is
    /// intentionally bounded to KV rows 4..=35; rows outside that range use the
    /// qualified reference kernel until separately qualified.
    pub const fn thor_ka17_parallel_score_candidate() -> Self {
        Self {
            schema_version: F16_ATTENTION_PLAN_VERSION,
            kernel: F16CachedAttentionKernel::ParallelScoreKa17Candidate,
            staged_min_kv_rows: 0,
        }
    }

    /// Return the KA17 threads-per-block selection for a qualified KV length.
    ///
    /// `None` means use the qualified reference kernel. This is deliberately not
    /// an interpolated or adaptive policy: each interval is the policy declared
    /// before KA17 and rows above 35 remain unqualified.
    #[must_use]
    pub const fn parallel_score_threads_per_block(&self, kv_rows: usize) -> Option<u32> {
        if !matches!(
            self.kernel,
            F16CachedAttentionKernel::ParallelScoreKa17Candidate
        ) {
            return None;
        }
        match kv_rows {
            4 => Some(128),
            5..=16 => Some(256),
            17..=35 => Some(512),
            _ => None,
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
            F16CachedAttentionKernel::ParallelScoreKa17Candidate => {
                if self.staged_min_kv_rows != 0 {
                    return Err(NnisError::invalid_input(
                        "KA17 parallel-score F16 attention plan requires staged_min_kv_rows=0",
                    ));
                }
                if config.hidden_size != 576
                    || config.num_attention_heads != 9
                    || config.num_key_value_heads != 3
                    || config.head_dim() != 64
                    || config.max_position_embeddings < 35
                {
                    return Err(NnisError::unsupported(
                        "KA17 parallel-score F16 attention plan is restricted to the qualified SmolLM2 attention geometry",
                    ));
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

        let parallel = F16AttentionPlan::thor_ka17_parallel_score_candidate();
        parallel.validate(&smollm2_config()).unwrap();
        assert_eq!(
            parallel.kernel,
            F16CachedAttentionKernel::ParallelScoreKa17Candidate
        );
        assert_eq!(parallel.parallel_score_threads_per_block(1), None);
        assert_eq!(parallel.parallel_score_threads_per_block(3), None);
        assert_eq!(parallel.parallel_score_threads_per_block(4), Some(128));
        assert_eq!(parallel.parallel_score_threads_per_block(5), Some(256));
        assert_eq!(parallel.parallel_score_threads_per_block(16), Some(256));
        assert_eq!(parallel.parallel_score_threads_per_block(17), Some(512));
        assert_eq!(parallel.parallel_score_threads_per_block(35), Some(512));
        assert_eq!(parallel.parallel_score_threads_per_block(36), None);
        assert!(parallel.validate(&tiny_config()).is_err());
        let encoded_parallel = serde_json::to_string(&parallel).unwrap();
        assert!(encoded_parallel.contains("\"kernel\":\"parallel_score_ka17_candidate\""));

        let mut future = candidate;
        future.schema_version = F16_ATTENTION_PLAN_VERSION + 1;
        assert!(future.validate(&config).is_err());

        let mut invalid_reference = reference;
        invalid_reference.staged_min_kv_rows = 1;
        assert!(invalid_reference.validate(&config).is_err());
    }
}
