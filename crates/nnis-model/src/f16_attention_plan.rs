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
    /// PR #89 candidate with warp-parallel Q·K score reduction.
    ParallelScoreCandidate,
}

/// Fixed launch policy carried by the KA17-qualified parallel-score candidate.
///
/// This is deliberately an enum instead of mutable thresholds: the policy was
/// declared before KA17 and must not be retuned from the same evidence. Rows
/// outside the qualified 1..=35 corpus use the reference kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F16ParallelScorePolicy {
    Ka17SmolLm2ShortContextV1,
}

impl F16ParallelScorePolicy {
    #[must_use]
    pub const fn threads_per_block(self, kv_rows: usize) -> Option<u32> {
        match self {
            Self::Ka17SmolLm2ShortContextV1 => match kv_rows {
                4 => Some(128),
                5..=16 => Some(256),
                17..=F16_PARALLEL_SCORE_KA17_MAX_KV_ROWS => Some(512),
                _ => None,
            },
        }
    }
}

/// Versioned F16 attention-kernel policy, separate from numeric and projection plans.
///
/// Candidate plans remain opt-in. The reference constructor is unchanged. The
/// staged candidate may fall back to reference when below threshold or outside
/// resource support. The KA17 parallel-score plan uses the reference kernel
/// outside its explicitly qualified short-context domain and fails closed if a
/// selected candidate launch is resource-incompatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct F16AttentionPlan {
    pub schema_version: u32,
    pub kernel: F16CachedAttentionKernel,
    pub staged_min_kv_rows: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_score_policy: Option<F16ParallelScorePolicy>,
}

impl F16AttentionPlan {
    /// Preserve the qualified reference attention path for every KV length.
    pub const fn reference() -> Self {
        Self {
            schema_version: F16_ATTENTION_PLAN_VERSION,
            kernel: F16CachedAttentionKernel::ReferencePerPositionBarriers,
            staged_min_kv_rows: 0,
            parallel_score_policy: None,
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
            parallel_score_policy: None,
        }
    }

    /// Candidate-only plan carrying the launch policy fixed before KA17.
    ///
    /// KA17 qualified the final F16 output boundary for SmolLM2-135M attention
    /// geometry across KV rows 1..=35 and six deterministic fixture families.
    /// This constructor does not promote the candidate: rows <=3 and >35 retain
    /// the reference kernel, and rows 4..=35 use the predeclared 128/256/512
    /// schedule. End-to-end greedy and timing evidence remains mandatory.
    pub const fn thor_ka17_parallel_score_candidate() -> Self {
        Self {
            schema_version: F16_ATTENTION_PLAN_VERSION,
            kernel: F16CachedAttentionKernel::ParallelScoreCandidate,
            staged_min_kv_rows: 0,
            parallel_score_policy: Some(F16ParallelScorePolicy::Ka17SmolLm2ShortContextV1),
        }
    }

    #[must_use]
    pub const fn parallel_score_threads_per_block(&self, kv_rows: usize) -> Option<u32> {
        match self.parallel_score_policy {
            Some(policy) => policy.threads_per_block(kv_rows),
            None => None,
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
                if self.staged_min_kv_rows != 0 || self.parallel_score_policy.is_some() {
                    return Err(NnisError::invalid_input(
                        "reference F16 attention plan requires no candidate thresholds or policy",
                    ));
                }
            }
            F16CachedAttentionKernel::StagedWeightsCandidate => {
                if self.parallel_score_policy.is_some() {
                    return Err(NnisError::invalid_input(
                        "staged F16 attention plan cannot carry a parallel-score policy",
                    ));
                }
                if self.staged_min_kv_rows == 0
                    || self.staged_min_kv_rows > config.max_position_embeddings
                {
                    return Err(NnisError::invalid_input(format!(
                        "staged F16 attention threshold {} must be within 1..={}",
                        self.staged_min_kv_rows, config.max_position_embeddings
                    )));
                }
            }
            F16CachedAttentionKernel::ParallelScoreCandidate => {
                if self.staged_min_kv_rows != 0 {
                    return Err(NnisError::invalid_input(
                        "parallel-score F16 attention plan requires staged_min_kv_rows=0",
                    ));
                }
                if self.parallel_score_policy
                    != Some(F16ParallelScorePolicy::Ka17SmolLm2ShortContextV1)
                {
                    return Err(NnisError::invalid_input(
                        "parallel-score F16 attention plan requires the fixed KA17 SmolLM2 short-context policy",
                    ));
                }
                if config.hidden_size != 576
                    || config.intermediate_size != 1_536
                    || config.num_hidden_layers != 30
                    || config.num_attention_heads != 9
                    || config.num_key_value_heads != 3
                    || config.head_dim() != 64
                    || config.max_position_embeddings < F16_PARALLEL_SCORE_KA17_MAX_KV_ROWS
                {
                    return Err(NnisError::unsupported(format!(
                        "KA17 parallel-score policy is qualified only for SmolLM2-135M geometry with at least {} positions; got {config:?}",
                        F16_PARALLEL_SCORE_KA17_MAX_KV_ROWS
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
        assert!(!encoded.contains("parallel_score_policy"));

        let mut future = candidate;
        future.schema_version = F16_ATTENTION_PLAN_VERSION + 1;
        assert!(future.validate(&config).is_err());

        let mut invalid_reference = reference;
        invalid_reference.staged_min_kv_rows = 1;
        assert!(invalid_reference.validate(&config).is_err());
    }

    #[test]
    fn ka17_parallel_score_policy_is_fixed_and_geometry_gated() {
        let plan = F16AttentionPlan::thor_ka17_parallel_score_candidate();
        plan.validate(&smollm2_config()).unwrap();
        assert_eq!(
            plan.kernel,
            F16CachedAttentionKernel::ParallelScoreCandidate
        );
        assert_eq!(plan.parallel_score_threads_per_block(1), None);
        assert_eq!(plan.parallel_score_threads_per_block(3), None);
        assert_eq!(plan.parallel_score_threads_per_block(4), Some(128));
        assert_eq!(plan.parallel_score_threads_per_block(5), Some(256));
        assert_eq!(plan.parallel_score_threads_per_block(16), Some(256));
        assert_eq!(plan.parallel_score_threads_per_block(17), Some(512));
        assert_eq!(plan.parallel_score_threads_per_block(35), Some(512));
        assert_eq!(plan.parallel_score_threads_per_block(36), None);
        assert!(plan.validate(&tiny_config()).is_err());

        let encoded = serde_json::to_string(&plan).unwrap();
        assert!(encoded.contains("\"kernel\":\"parallel_score_candidate\""));
        assert!(encoded.contains(
            "\"parallel_score_policy\":\"ka17_smol_lm2_short_context_v1\""
        ));
    }
}
