use crate::F16ReferenceExecutionPlan;
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

pub const NNIS_F16_EXECUTION_TRANSITION_REQUIREMENTS_VERSION: u32 = 1;

/// How an already-materialized NNIS F16 model can move between physical
/// execution plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F16ExecutionTransitionMode {
    /// The target plan is selected while constructing a new
    /// [`crate::F16ReferenceModel`]. The current model is not mutated in place.
    ModelRebuildRequired,
}

/// Versioned transition requirements for an NNIS F16 physical execution plan.
///
/// This contract is intentionally narrower than a generic adaptive-runtime API.
/// It describes what NNIS actually implements today so external controllers can
/// fail closed instead of assuming an in-place model transition exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct F16ExecutionTransitionRequirementsV1 {
    pub schema_version: u32,
    pub transition_mode: F16ExecutionTransitionMode,
    /// The original logical/source weights must remain available to construct
    /// the target physical resident layout.
    pub source_weights_required: bool,
    /// Existing inference sessions are not transferable to the rebuilt model.
    pub active_sessions_preserved: bool,
    /// Existing KV state is not transferable to the rebuilt model.
    pub kv_state_preserved: bool,
    /// NNIS does not authorize mutation of an already-materialized F16 model to
    /// another execution plan in place.
    pub live_transition_authorized: bool,
}

impl F16ExecutionTransitionRequirementsV1 {
    /// Requirements implemented by every schema-v1 F16 execution plan today.
    #[must_use]
    pub const fn model_rebuild_required() -> Self {
        Self {
            schema_version: NNIS_F16_EXECUTION_TRANSITION_REQUIREMENTS_VERSION,
            transition_mode: F16ExecutionTransitionMode::ModelRebuildRequired,
            source_weights_required: true,
            active_sessions_preserved: false,
            kv_state_preserved: false,
            live_transition_authorized: false,
        }
    }

    /// Validate the versioned requirements and their fail-closed invariants.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_F16_EXECUTION_TRANSITION_REQUIREMENTS_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported F16 execution-transition requirements schema {}; expected {}",
                self.schema_version, NNIS_F16_EXECUTION_TRANSITION_REQUIREMENTS_VERSION
            )));
        }

        match self.transition_mode {
            F16ExecutionTransitionMode::ModelRebuildRequired => {
                if !self.source_weights_required
                    || self.active_sessions_preserved
                    || self.kv_state_preserved
                    || self.live_transition_authorized
                {
                    return Err(NnisError::invalid_input(
                        "schema-v1 F16 model-rebuild transition requirements must require source weights, invalidate sessions/KV state, and forbid live in-place transition",
                    ));
                }
            }
        }

        Ok(())
    }
}

impl F16ReferenceExecutionPlan {
    /// Machine-readable requirements for changing from an already-materialized
    /// F16 model to this execution plan.
    ///
    /// The plan remains a construction-time choice: its projection layout is
    /// used while resident weights and candidate kernels are materialized.
    #[must_use]
    pub const fn transition_requirements(&self) -> F16ExecutionTransitionRequirementsV1 {
        F16ExecutionTransitionRequirementsV1::model_rebuild_required()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F16ReferencePlan;

    #[test]
    fn every_current_f16_execution_layout_requires_model_rebuild() {
        let plans = [
            F16ReferenceExecutionPlan::reference(
                F16ReferencePlan::edge_llm_v0_10_0_alignment(),
            ),
            F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_projection_candidate(),
            F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_fused_groups_candidate(),
            F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_fused_mlp_candidate(),
        ];

        for plan in plans {
            let requirements = plan.transition_requirements();
            requirements.validate().unwrap();
            assert_eq!(
                requirements.transition_mode,
                F16ExecutionTransitionMode::ModelRebuildRequired
            );
            assert!(requirements.source_weights_required);
            assert!(!requirements.active_sessions_preserved);
            assert!(!requirements.kv_state_preserved);
            assert!(!requirements.live_transition_authorized);
        }
    }

    #[test]
    fn transition_requirements_wire_contract_is_strict_and_versioned() {
        let requirements = F16ExecutionTransitionRequirementsV1::model_rebuild_required();
        let json = serde_json::to_string(&requirements).unwrap();
        let decoded: F16ExecutionTransitionRequirementsV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, requirements);
        decoded.validate().unwrap();

        let unknown = format!(
            "{{\"schema_version\":{},\"transition_mode\":\"model_rebuild_required\",\"source_weights_required\":true,\"active_sessions_preserved\":false,\"kv_state_preserved\":false,\"live_transition_authorized\":false,\"unknown\":1}}",
            NNIS_F16_EXECUTION_TRANSITION_REQUIREMENTS_VERSION
        );
        assert!(serde_json::from_str::<F16ExecutionTransitionRequirementsV1>(&unknown).is_err());

        let mut future = requirements;
        future.schema_version = NNIS_F16_EXECUTION_TRANSITION_REQUIREMENTS_VERSION + 1;
        assert!(future.validate().is_err());
    }

    #[test]
    fn requirements_reject_false_live_transition_claims() {
        let mut requirements = F16ExecutionTransitionRequirementsV1::model_rebuild_required();
        requirements.live_transition_authorized = true;
        assert!(requirements.validate().is_err());

        let mut requirements = F16ExecutionTransitionRequirementsV1::model_rebuild_required();
        requirements.active_sessions_preserved = true;
        assert!(requirements.validate().is_err());

        let mut requirements = F16ExecutionTransitionRequirementsV1::model_rebuild_required();
        requirements.kv_state_preserved = true;
        assert!(requirements.validate().is_err());
    }
}
