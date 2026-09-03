use crate::{Activation, F16ReferenceExecutionPlan, ModelConfig, WeightDType};
use nnis_rt::{Context, NnisError, Result};
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
/// Generic APIs keep the reference constructor unchanged. Candidate plans remain
/// explicit comparison surfaces. The promoted SmolLM2/Thor selector is separately
/// fail-closed to the exact qualified model, parent execution plan, and GPU class.
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

    /// Historical candidate constructor carrying the policy fixed before KA17.
    ///
    /// KA17 qualified the final F16 output boundary for SmolLM2-135M attention
    /// geometry across KV rows 1..=35 and six deterministic fixture families.
    /// Rows <=3 and >35 retain the reference kernel; rows 4..=35 use the
    /// predeclared 128/256/512 schedule. This constructor remains available for
    /// evidence replay and explicit comparisons.
    pub const fn thor_ka17_parallel_score_candidate() -> Self {
        Self {
            schema_version: F16_ATTENTION_PLAN_VERSION,
            kernel: F16CachedAttentionKernel::ParallelScoreCandidate,
            staged_min_kv_rows: 0,
            parallel_score_policy: Some(F16ParallelScorePolicy::Ka17SmolLm2ShortContextV1),
        }
    }

    /// Qualified minimum-latency attention plan for the pinned SmolLM2-135M
    /// short-context execution domain on NVIDIA Thor, under the exact parent
    /// projection plan used by KA18 and KA19.
    ///
    /// Promotion basis: KA17 established 840/840 bitwise-identical final-F16
    /// comparisons across KV rows 1..=35, six fixture families and four launch
    /// widths. KA18 and KA19 then independently requalified the frozen policy on
    /// the exact SmolLM2 decode32 trajectory with repeated ABBA ordering. Each
    /// campaign won all six paired GPU rounds by at least 3%, preserved every
    /// 32-token greedy trajectory, and reported median paired generation-stage
    /// GPU improvements above 20%. The independent consensus verifier accepted
    /// both campaigns under one compatible stable physical environment.
    ///
    /// Both campaigns used
    /// [`F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_projection_candidate`]
    /// as the parent execution plan. Combining this attention policy with another
    /// projection/MLP plan requires a separate physical end-to-end qualification.
    /// This selector intentionally does not change [`Self::reference`].
    pub fn smollm2_135m_thor_min_latency(
        config: &ModelConfig,
        execution_plan: F16ReferenceExecutionPlan,
        context: &Context,
    ) -> Result<Self> {
        let plan = Self::thor_ka17_parallel_score_candidate();
        plan.validate_smollm2_135m_min_latency_domain(config, execution_plan)?;

        let properties = context.props();
        if properties.name != "NVIDIA Thor"
            || properties.compute_capability != (11, 0)
            || properties.multiprocessor_count != 20
        {
            return Err(NnisError::unsupported(format!(
                "qualified SmolLM2 F16 attention min-latency plan requires NVIDIA Thor cc 11.0 with 20 SMs; observed {} cc {}.{} with {} SMs",
                properties.name,
                properties.compute_capability.0,
                properties.compute_capability.1,
                properties.multiprocessor_count
            )));
        }
        Ok(plan)
    }

    /// Validate the exact model and parent-execution-plan promotion domain
    /// independently of CUDA hardware. Hardware authorization still requires
    /// [`Self::smollm2_135m_thor_min_latency`].
    pub fn validate_smollm2_135m_min_latency_domain(
        &self,
        config: &ModelConfig,
        execution_plan: F16ReferenceExecutionPlan,
    ) -> Result<()> {
        self.validate(config)?;
        if *self != Self::thor_ka17_parallel_score_candidate() {
            return Err(NnisError::unsupported(
                "SmolLM2 F16 attention min-latency qualification requires the KA17 parallel-score policy",
            ));
        }
        if execution_plan
            != F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_projection_candidate()
        {
            return Err(NnisError::unsupported(
                "SmolLM2 F16 attention min-latency qualification requires the exact transposed-projection parent used by KA18 and KA19",
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
            || config.weight_dtype != WeightDType::F32
        {
            return Err(NnisError::unsupported(
                "qualified F16 attention min-latency plan is restricted to the pinned SmolLM2-135M model identity",
            ));
        }
        Ok(())
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
        assert!(encoded.contains("\"parallel_score_policy\":\"ka17_smol_lm2_short_context_v1\""));
    }

    #[test]
    fn smollm2_min_latency_domain_is_fail_closed_to_ka18_parent() {
        let plan = F16AttentionPlan::thor_ka17_parallel_score_candidate();
        let qualified_parent =
            F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_projection_candidate();
        plan.validate_smollm2_135m_min_latency_domain(&smollm2_config(), qualified_parent)
            .unwrap();

        assert!(plan
            .validate_smollm2_135m_min_latency_domain(&tiny_config(), qualified_parent)
            .is_err());
        assert!(F16AttentionPlan::reference()
            .validate_smollm2_135m_min_latency_domain(&smollm2_config(), qualified_parent)
            .is_err());

        let fused_mlp_parent =
            F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_fused_mlp_candidate();
        assert!(plan
            .validate_smollm2_135m_min_latency_domain(&smollm2_config(), fused_mlp_parent)
            .is_err());

        let mut wrong_identity = smollm2_config();
        wrong_identity.rope_theta = 10_000.0;
        assert!(plan
            .validate_smollm2_135m_min_latency_domain(&wrong_identity, qualified_parent)
            .is_err());
    }
}
