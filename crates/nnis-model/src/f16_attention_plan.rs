use crate::ModelConfig;
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

pub const F16_ATTENTION_PLAN_VERSION: u32 = 1;
pub const F16_PARALLEL_SCORE_KA17_MAX_KV_ROWS: usize = 35;

/// Cached-attention implementation selected by the explicit F16 attention plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F16CachedAttentionKernel {
    /// Qualified reference kernel with two block barriers per KV position.
    ReferencePerPositionBarriers,
    /// PR #77 candidate that stages serial softmax weights once in shared memory.
    StagedWeightsCandidate,
    /// KA17-qualified SmolLM2 short-context candidate with parallel Q·K scores.
    ParallelScoreKa17Candidate,
}

/// Versioned F16 attention-kernel policy, separate from numeric and projection plans.
///
/// `staged_min_kv_rows` is used only by [`F16CachedAttentionKernel::StagedWeightsCandidate`].
/// The KA17 parallel-score policy is intentionally fixed in code to the launch
/// schedule declared before the dense KA17 qualification run. It falls back to
/// the reference kernel outside the measured KV-row domain rather than
/// extrapolating to unseen context lengths.
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

    /// Explicit KA17 candidate for the pinned SmolLM2-135M short-context domain.
    ///
    /// KA17 qualified all four candidate launch widths bitwise at the final F16
    /// output boundary for KV rows 1..=35 and six deterministic fixture families.
    /// The launch policy was declared before that dense-row measurement and must
    /// not be retuned from KA17 after the fact. This constructor remains opt-in;
    /// it does not alter the reference runtime default.
    pub const fn smollm2_135m_thor_parallel_score_ka17_candidate() -> Self {
        Self {
            schema_version: F16_ATTENTION_PLAN_VERSION,
            kernel: F16CachedAttentionKernel::ParallelScoreKa17Candidate,
            staged_min_kv_rows: 0,
        }
    }

    /// Return the predeclared KA17 launch width for one active KV length.
    ///
    /// `None` means use the qualified reference attention kernel. In particular,
    /// the candidate is not extrapolated beyond the physically qualified
    /// 1..=35 short-context domain.
    #[must_use]
    pub const fn parallel_score_ka17_threads_per_block(&self, kv_rows: usize) -> Option<u32> {
        match self.kernel {
            F16CachedAttentionKernel::ParallelScoreKa17Candidate => {
                if kv_rows <= 3 || kv_rows > F16_PARALLEL_SCORE_KA17_MAX_KV_ROWS {
                    None
                } else if kv_rows == 4 {
                    Some(128)
                } else if kv_rows <= 16 {
                    Some(256)
                } else {
                    Some(512)
                }
            }
            F16CachedAttentionKernel::ReferencePerPositionBarriers
            | F16CachedAttentionKernel::StagedWeightsCandidate => None,
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
                    || config.max_position_embeddings < F16_PARALLEL_SCORE_KA17_MAX_KV_ROWS
                {
                    return Err(NnisError::unsupported(
                        "KA17 parallel-score F16 attention plan is restricted to the qualified SmolLM2 9-query-head/3-KV-head/head-dim-64 short-context geometry",
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

        let mut future = candidate;
        future.schema_version = F16_ATTENTION_PLAN_VERSION + 1;
        assert!(future.validate(&config).is_err());

        let mut invalid_reference = reference;
        invalid_reference.staged_min_kv_rows = 1;
        assert!(invalid_reference.validate(&config).is_err());
    }

    #[test]
    fn ka17_parallel_score_policy_is_pinned_and_fail_closed() {
        let candidate = F16AttentionPlan::smollm2_135m_thor_parallel_score_ka17_candidate();
        candidate.validate(&smollm2_config()).unwrap();
        assert!(candidate.validate(&tiny_config()).is_err());
        assert_eq!(candidate.parallel_score_ka17_threads_per_block(1), None);
        assert_eq!(candidate.parallel_score_ka17_threads_per_block(3), None);
        assert_eq!(candidate.parallel_score_ka17_threads_per_block(4), Some(128));
        assert_eq!(candidate.parallel_score_ka17_threads_per_block(5), Some(256));
        assert_eq!(candidate.parallel_score_ka17_threads_per_block(16), Some(256));
        assert_eq!(candidate.parallel_score_ka17_threads_per_block(17), Some(512));
        assert_eq!(candidate.parallel_score_ka17_threads_per_block(35), Some(512));
        assert_eq!(candidate.parallel_score_ka17_threads_per_block(36), None);

        let encoded = serde_json::to_string(&candidate).unwrap();
        assert!(encoded.contains("\"kernel\":\"parallel_score_ka17_candidate\""));
        assert!(encoded.contains("\"staged_min_kv_rows\":0"));
    }
}
