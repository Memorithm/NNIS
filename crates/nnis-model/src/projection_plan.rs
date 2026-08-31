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

    /// Diagnostic E1 plan: only Q/O projections use the measured GEMV candidate.
    #[must_use]
    pub const fn thor_e1_qo_only() -> Self {
        Self {
            q_o: F32ProjectionKernel::Gemv { block_size: 512 },
            ..Self::baseline_gemm()
        }
    }

    /// Diagnostic E1 plan: only gate/up projections use the measured GEMV candidate.
    #[must_use]
    pub const fn thor_e1_gate_up_only() -> Self {
        Self {
            gate_up: F32ProjectionKernel::Gemv { block_size: 128 },
            ..Self::baseline_gemm()
        }
    }

    /// Diagnostic E1 plan: only the LM head uses the measured GEMV candidate.
    #[must_use]
    pub const fn thor_e1_lm_head_only() -> Self {
        Self {
            lm_head: F32ProjectionKernel::Gemv { block_size: 64 },
            ..Self::baseline_gemm()
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
    fn single_family_diagnostics_change_only_one_family() {
        let baseline = F32ProjectionPlan::baseline_gemm();
        let qo = F32ProjectionPlan::thor_e1_qo_only();
        assert_eq!(qo.q_o, F32ProjectionKernel::Gemv { block_size: 512 });
        assert_eq!(
            (qo.k_v, qo.gate_up, qo.down, qo.lm_head),
            (
                baseline.k_v,
                baseline.gate_up,
                baseline.down,
                baseline.lm_head
            )
        );
        let gate_up = F32ProjectionPlan::thor_e1_gate_up_only();
        assert_eq!(
            gate_up.gate_up,
            F32ProjectionKernel::Gemv { block_size: 128 }
        );
        assert_eq!(
            (gate_up.q_o, gate_up.k_v, gate_up.down, gate_up.lm_head),
            (baseline.q_o, baseline.k_v, baseline.down, baseline.lm_head)
        );
        let lm_head = F32ProjectionPlan::thor_e1_lm_head_only();
        assert_eq!(
            lm_head.lm_head,
            F32ProjectionKernel::Gemv { block_size: 64 }
        );
        assert_eq!(
            (lm_head.q_o, lm_head.k_v, lm_head.gate_up, lm_head.down),
            (baseline.q_o, baseline.k_v, baseline.gate_up, baseline.down)
        );
    }

    #[test]
    fn invalid_gemv_width_is_fail_closed() {
        assert!(F32ProjectionKernel::Gemv { block_size: 192 }
            .validate()
            .is_err());
    }
}
