use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Physical kernel selected for one f32 `[1,K] × [K,N]` projection family.
///
/// This describes execution only. It does not change logical model weights or
/// their f32 representation; representation elasticity is a separate plan axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kernel", rename_all = "snake_case")]
pub enum F32ProjectionKernel {
    Gemm,
    Gemv { block_size: u32 },
}

impl F32ProjectionKernel {
    pub fn validate(self) -> Result<()> {
        if let Self::Gemv { block_size } = self {
            if block_size == 0 || !block_size.is_power_of_two() {
                return Err(NnisError::invalid_input(format!(
                    "projection GEMV block size {block_size} is not a non-zero power of two"
                )));
            }
        }
        Ok(())
    }
}

/// Per-shape f32 decoder projection plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct F32ProjectionPlan {
    pub q_o: F32ProjectionKernel,
    pub k_v: F32ProjectionKernel,
    pub gate_up: F32ProjectionKernel,
    pub down: F32ProjectionKernel,
    pub lm_head: F32ProjectionKernel,
}

impl F32ProjectionPlan {
    /// Historical NNIS behavior: every projection uses the general f32 GEMM.
    #[must_use]
    pub const fn baseline_gemm() -> Self {
        Self {
            q_o: F32ProjectionKernel::Gemm,
            k_v: F32ProjectionKernel::Gemm,
            gate_up: F32ProjectionKernel::Gemm,
            down: F32ProjectionKernel::Gemm,
            lm_head: F32ProjectionKernel::Gemm,
        }
    }

    /// E1 plan selected from the physical Thor MAXN candidate sweep.
    ///
    /// This is evidence-scoped to the qualified SmolLM2 f32 experiment and is
    /// not claimed to be universally optimal on Thor or other models.
    #[must_use]
    pub const fn thor_e1_smollm2() -> Self {
        Self {
            q_o: F32ProjectionKernel::Gemv { block_size: 512 },
            k_v: F32ProjectionKernel::Gemm,
            gate_up: F32ProjectionKernel::Gemv { block_size: 128 },
            down: F32ProjectionKernel::Gemm,
            lm_head: F32ProjectionKernel::Gemv { block_size: 64 },
        }
    }

    pub fn validate(self) -> Result<()> {
        for choice in [self.q_o, self.k_v, self.gate_up, self.down, self.lm_head] {
            choice.validate()?;
        }
        Ok(())
    }
}

impl Default for F32ProjectionPlan {
    fn default() -> Self {
        Self::baseline_gemm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_preserves_all_gemm_execution() {
        let plan = F32ProjectionPlan::baseline_gemm();
        assert_eq!(plan.q_o, F32ProjectionKernel::Gemm);
        assert_eq!(plan.k_v, F32ProjectionKernel::Gemm);
        assert_eq!(plan.gate_up, F32ProjectionKernel::Gemm);
        assert_eq!(plan.down, F32ProjectionKernel::Gemm);
        assert_eq!(plan.lm_head, F32ProjectionKernel::Gemm);
        plan.validate().unwrap();
    }

    #[test]
    fn thor_e1_plan_matches_physical_sweep_promotion() {
        let plan = F32ProjectionPlan::thor_e1_smollm2();
        assert_eq!(plan.q_o, F32ProjectionKernel::Gemv { block_size: 512 });
        assert_eq!(plan.k_v, F32ProjectionKernel::Gemm);
        assert_eq!(plan.gate_up, F32ProjectionKernel::Gemv { block_size: 128 });
        assert_eq!(plan.down, F32ProjectionKernel::Gemm);
        assert_eq!(plan.lm_head, F32ProjectionKernel::Gemv { block_size: 64 });
        plan.validate().unwrap();
    }

    #[test]
    fn invalid_gemv_width_is_fail_closed() {
        assert!(F32ProjectionKernel::Gemv { block_size: 192 }
            .validate()
            .is_err());
    }
}
