use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Current schema version for explicit decoder fusion plans.
pub const F32_FUSION_PLAN_VERSION: u32 = 1;

/// Kernel choice for the SwiGLU `SiLU(gate) * up` stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kernel", rename_all = "snake_case")]
pub enum F32SiluMultiplyKernel {
    Separate,
    Fused { block_size: u32 },
}

/// Versioned decoder fusion plan.
///
/// This axis is deliberately separate from projection selection and weight
/// representation. The default reproduces the historical two-launch path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct F32FusionPlan {
    pub schema_version: u32,
    pub silu_multiply: F32SiluMultiplyKernel,
}

impl F32FusionPlan {
    /// Historical decoder path: separate SiLU and multiply launches.
    #[must_use]
    pub const fn baseline() -> Self {
        Self {
            schema_version: F32_FUSION_PLAN_VERSION,
            silu_multiply: F32SiluMultiplyKernel::Separate,
        }
    }

    /// R2 candidate selected after the physical 1536-element isolated gate.
    ///
    /// The candidate is explicit and does not become the runtime default merely
    /// because this constructor exists.
    #[must_use]
    pub const fn r2_silu_multiply_fused_candidate() -> Self {
        Self {
            schema_version: F32_FUSION_PLAN_VERSION,
            silu_multiply: F32SiluMultiplyKernel::Fused { block_size: 256 },
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.schema_version != F32_FUSION_PLAN_VERSION {
            return Err(NnisError::unsupported(format!(
                "f32 fusion plan schema {}; supported version is {}",
                self.schema_version, F32_FUSION_PLAN_VERSION
            )));
        }
        if let F32SiluMultiplyKernel::Fused { block_size } = self.silu_multiply {
            if block_size != 256 {
                return Err(NnisError::unsupported(format!(
                    "R2 fused SiLU-multiply plan currently admits only the physically evaluated block size 256; got {block_size}"
                )));
            }
        }
        Ok(())
    }
}

impl Default for F32FusionPlan {
    fn default() -> Self {
        Self::baseline()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_is_unfused_and_versioned() {
        let plan = F32FusionPlan::default();
        assert_eq!(plan.schema_version, F32_FUSION_PLAN_VERSION);
        assert_eq!(plan.silu_multiply, F32SiluMultiplyKernel::Separate);
        plan.validate().unwrap();
    }

    #[test]
    fn r2_candidate_is_explicit_and_pinned_to_measured_geometry() {
        let plan = F32FusionPlan::r2_silu_multiply_fused_candidate();
        assert_eq!(
            plan.silu_multiply,
            F32SiluMultiplyKernel::Fused { block_size: 256 }
        );
        plan.validate().unwrap();

        let unsupported = F32FusionPlan {
            schema_version: F32_FUSION_PLAN_VERSION,
            silu_multiply: F32SiluMultiplyKernel::Fused { block_size: 128 },
        };
        assert!(unsupported.validate().is_err());
    }

    #[test]
    fn serialized_shape_is_stable_for_evidence_consumers() {
        assert_eq!(
            serde_json::to_value(F32FusionPlan::baseline()).unwrap(),
            json!({
                "schema_version": 1,
                "silu_multiply": {"kernel": "separate"}
            })
        );
        assert_eq!(
            serde_json::to_value(F32FusionPlan::r2_silu_multiply_fused_candidate()).unwrap(),
            json!({
                "schema_version": 1,
                "silu_multiply": {"kernel": "fused", "block_size": 256}
            })
        );
    }

    #[test]
    fn future_schema_fails_closed() {
        let mut plan = F32FusionPlan::baseline();
        plan.schema_version += 1;
        assert!(plan.validate().is_err());
    }
}
