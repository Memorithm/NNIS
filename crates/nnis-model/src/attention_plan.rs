use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Current schema version for explicit cached-attention execution plans.
pub const F32_ATTENTION_PLAN_VERSION: u32 = 1;

/// Kernel choice for one-token cached decoder attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kernel", rename_all = "snake_case")]
pub enum F32CachedAttentionKernel {
    /// Historical correctness-first path: one CUDA thread owns one query head.
    SerialSingleThread,
    /// R2 candidate: lane zero preserves the serial score/softmax chain while
    /// independent value/output dimensions are updated by separate threads.
    ParallelValue { threads_per_query_head: u32 },
}

/// Versioned cached-attention execution plan.
///
/// This axis is deliberately separate from projection selection, weight
/// representation and fusion selection. The default reproduces the historical
/// one-thread-per-query-head path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct F32AttentionPlan {
    pub schema_version: u32,
    pub cached_decode: F32CachedAttentionKernel,
}

impl F32AttentionPlan {
    /// Historical decoder attention path.
    #[must_use]
    pub const fn baseline() -> Self {
        Self {
            schema_version: F32_ATTENTION_PLAN_VERSION,
            cached_decode: F32CachedAttentionKernel::SerialSingleThread,
        }
    }

    /// R2 candidate selected after the physical SmolLM2-shaped isolated gate.
    ///
    /// The candidate is explicit and does not become the runtime default merely
    /// because this constructor exists.
    #[must_use]
    pub const fn r2_parallel_value_candidate() -> Self {
        Self {
            schema_version: F32_ATTENTION_PLAN_VERSION,
            cached_decode: F32CachedAttentionKernel::ParallelValue {
                threads_per_query_head: 64,
            },
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.schema_version != F32_ATTENTION_PLAN_VERSION {
            return Err(NnisError::unsupported(format!(
                "f32 attention plan schema {}; supported version is {}",
                self.schema_version, F32_ATTENTION_PLAN_VERSION
            )));
        }
        if let F32CachedAttentionKernel::ParallelValue {
            threads_per_query_head,
        } = self.cached_decode
        {
            if threads_per_query_head != 64 {
                return Err(NnisError::unsupported(format!(
                    "R2 parallel-value cached-attention plan currently admits only the physically evaluated 64-thread geometry; got {threads_per_query_head}"
                )));
            }
        }
        Ok(())
    }
}

impl Default for F32AttentionPlan {
    fn default() -> Self {
        Self::baseline()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_is_serial_and_versioned() {
        let plan = F32AttentionPlan::default();
        assert_eq!(plan.schema_version, F32_ATTENTION_PLAN_VERSION);
        assert_eq!(
            plan.cached_decode,
            F32CachedAttentionKernel::SerialSingleThread
        );
        plan.validate().unwrap();
    }

    #[test]
    fn r2_candidate_is_explicit_and_pinned_to_measured_geometry() {
        let plan = F32AttentionPlan::r2_parallel_value_candidate();
        assert_eq!(
            plan.cached_decode,
            F32CachedAttentionKernel::ParallelValue {
                threads_per_query_head: 64
            }
        );
        plan.validate().unwrap();

        let unsupported = F32AttentionPlan {
            schema_version: F32_ATTENTION_PLAN_VERSION,
            cached_decode: F32CachedAttentionKernel::ParallelValue {
                threads_per_query_head: 32,
            },
        };
        assert!(unsupported.validate().is_err());
    }

    #[test]
    fn serialized_shape_is_stable_for_evidence_consumers() {
        assert_eq!(
            serde_json::to_value(F32AttentionPlan::baseline()).unwrap(),
            json!({
                "schema_version": 1,
                "cached_decode": {"kernel": "serial_single_thread"}
            })
        );
        assert_eq!(
            serde_json::to_value(F32AttentionPlan::r2_parallel_value_candidate()).unwrap(),
            json!({
                "schema_version": 1,
                "cached_decode": {
                    "kernel": "parallel_value",
                    "threads_per_query_head": 64
                }
            })
        );
    }

    #[test]
    fn future_schema_fails_closed() {
        let mut plan = F32AttentionPlan::baseline();
        plan.schema_version += 1;
        assert!(plan.validate().is_err());
    }
}
