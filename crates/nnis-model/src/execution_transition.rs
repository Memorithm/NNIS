use crate::{F16ReferenceExecutionPlan, F16_REFERENCE_EXECUTION_PLAN_VERSION};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

pub const NNIS_F16_EXECUTION_TRANSITION_REQUIREMENTS_VERSION: u32 = 1;
pub const NNIS_F16_REBUILD_TRANSITION_VERSION: u32 = 1;

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

/// Explicit memory admission envelope for one model-rebuild transition.
///
/// `target_peak_owned_bytes` is the target model's measured or conservatively
/// estimated peak owned allocation while the source model remains resident.
/// It must not be relabelled as process-wide VRAM or physical page residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct F16RebuildMemoryBudgetV1 {
    pub schema_version: u32,
    pub source_resident_owned_bytes: u64,
    pub target_peak_owned_bytes: u64,
    pub transition_budget_bytes: u64,
}

impl F16RebuildMemoryBudgetV1 {
    pub fn new(
        source_resident_owned_bytes: u64,
        target_peak_owned_bytes: u64,
        transition_budget_bytes: u64,
    ) -> Result<Self> {
        let budget = Self {
            schema_version: NNIS_F16_REBUILD_TRANSITION_VERSION,
            source_resident_owned_bytes,
            target_peak_owned_bytes,
            transition_budget_bytes,
        };
        budget.validate()?;
        Ok(budget)
    }

    pub fn required_double_resident_bytes(&self) -> Result<u64> {
        self.source_resident_owned_bytes
            .checked_add(self.target_peak_owned_bytes)
            .ok_or_else(|| {
                NnisError::invalid_input(
                    "F16 rebuild double-residency byte accounting overflowed u64",
                )
            })
    }

    pub fn headroom_bytes(&self) -> Result<u64> {
        let required = self.required_double_resident_bytes()?;
        self.transition_budget_bytes.checked_sub(required).ok_or_else(|| {
            NnisError::invalid_input(format!(
                "F16 rebuild requires {required} owned bytes while the source remains resident, exceeding transition budget {}",
                self.transition_budget_bytes
            ))
        })
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_F16_REBUILD_TRANSITION_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported F16 rebuild memory-budget schema {}; expected {}",
                self.schema_version, NNIS_F16_REBUILD_TRANSITION_VERSION
            )));
        }
        if self.source_resident_owned_bytes == 0 || self.target_peak_owned_bytes == 0 {
            return Err(NnisError::invalid_input(
                "F16 rebuild memory admission requires non-zero source-resident and target-peak owned bytes",
            ));
        }
        self.headroom_bytes()?;
        Ok(())
    }
}

/// Observable state of the one-shot rebuild transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F16RebuildTransitionPhase {
    Prepared,
    Quiescing,
    Rebuilding,
    Verifying,
    InvalidatingSessionState,
    Committing,
    Committed,
    RollingBack,
    RolledBack,
}

/// Runtime-owned operations required by the NNIS rebuild coordinator.
///
/// Implementations must retain the source model until `commit_candidate`
/// succeeds, reject new sessions after `begin_quiescence`, and make
/// `rollback_to_source` idempotent. The coordinator never treats this sequence
/// as an in-place or zero-downtime hot switch.
pub trait F16RebuildTransitionRuntime {
    type Candidate;

    fn begin_quiescence(&mut self) -> Result<()>;
    fn active_session_count(&self) -> usize;
    fn source_weights_available(&self) -> bool;
    fn rebuild_candidate(
        &mut self,
        target_plan: F16ReferenceExecutionPlan,
    ) -> Result<Self::Candidate>;
    fn verify_candidate(&mut self, candidate: &Self::Candidate) -> Result<()>;
    fn invalidate_sessions_and_kv(&mut self) -> Result<()>;
    fn commit_candidate(&mut self, candidate: Self::Candidate) -> Result<()>;
    fn rollback_to_source(&mut self) -> Result<()>;
}

/// Evidence returned only after a verified candidate has been committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct F16RebuildTransitionReceiptV1 {
    pub schema_version: u32,
    pub source_plan: F16ReferenceExecutionPlan,
    pub target_plan: F16ReferenceExecutionPlan,
    pub required_double_resident_bytes: u64,
    pub budget_headroom_bytes: u64,
    pub source_retained_until_commit: bool,
    pub sessions_invalidated: bool,
    pub kv_state_reset: bool,
    pub candidate_verified: bool,
    pub committed: bool,
    pub live_transition_claimed: bool,
}

impl F16RebuildTransitionReceiptV1 {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_F16_REBUILD_TRANSITION_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported F16 rebuild receipt schema {}; expected {}",
                self.schema_version, NNIS_F16_REBUILD_TRANSITION_VERSION
            )));
        }
        if self.source_plan == self.target_plan
            || !self.source_retained_until_commit
            || !self.sessions_invalidated
            || !self.kv_state_reset
            || !self.candidate_verified
            || !self.committed
            || self.live_transition_claimed
        {
            return Err(NnisError::invalid_input(
                "F16 rebuild receipt violates the verified rebuild, state-reset, or no-hot-switch contract",
            ));
        }
        Ok(())
    }
}

/// One-shot, fail-closed coordinator for changing F16 execution plans.
///
/// The source stays authoritative through rebuild and verification. Session/KV
/// state is invalidated before commit, exactly as declared by
/// [`F16ExecutionTransitionRequirementsV1`]. Every failure attempts rollback;
/// a second execution is rejected even after rollback so callers must create a
/// fresh, fully re-admitted transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct F16RebuildTransitionV1 {
    pub schema_version: u32,
    pub source_plan: F16ReferenceExecutionPlan,
    pub target_plan: F16ReferenceExecutionPlan,
    pub memory_budget: F16RebuildMemoryBudgetV1,
    pub phase: F16RebuildTransitionPhase,
}

impl F16RebuildTransitionV1 {
    pub fn new(
        source_plan: F16ReferenceExecutionPlan,
        target_plan: F16ReferenceExecutionPlan,
        memory_budget: F16RebuildMemoryBudgetV1,
    ) -> Result<Self> {
        let transaction = Self {
            schema_version: NNIS_F16_REBUILD_TRANSITION_VERSION,
            source_plan,
            target_plan,
            memory_budget,
            phase: F16RebuildTransitionPhase::Prepared,
        };
        transaction.validate_preconditions()?;
        Ok(transaction)
    }

    /// Revalidate every public/wire field before the runtime can be mutated.
    ///
    /// This is intentionally separate from `new`: serde and public-field users
    /// can construct values without calling the constructor.
    fn validate_preconditions(&self) -> Result<()> {
        if self.schema_version != NNIS_F16_REBUILD_TRANSITION_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported F16 rebuild transaction schema {}; expected {}",
                self.schema_version, NNIS_F16_REBUILD_TRANSITION_VERSION
            )));
        }
        if self.phase != F16RebuildTransitionPhase::Prepared {
            return Err(NnisError::invalid_input(
                "F16 rebuild transaction must be in prepared state",
            ));
        }
        if self.source_plan.schema_version != F16_REFERENCE_EXECUTION_PLAN_VERSION
            || self.target_plan.schema_version != F16_REFERENCE_EXECUTION_PLAN_VERSION
        {
            return Err(NnisError::unsupported(
                "F16 rebuild requires supported source and target execution-plan schemas",
            ));
        }
        if self.source_plan == self.target_plan {
            return Err(NnisError::invalid_input(
                "F16 rebuild source and target execution plans must differ",
            ));
        }
        self.target_plan.transition_requirements().validate()?;
        self.memory_budget.validate()
    }

    fn rollback<R: F16RebuildTransitionRuntime>(
        &mut self,
        runtime: &mut R,
        primary: NnisError,
    ) -> Result<F16RebuildTransitionReceiptV1> {
        self.phase = F16RebuildTransitionPhase::RollingBack;
        let rollback = runtime.rollback_to_source();
        self.phase = F16RebuildTransitionPhase::RolledBack;
        match rollback {
            Ok(()) => Err(primary),
            Err(rollback_error) => Err(NnisError::invalid_input(format!(
                "F16 rebuild failed ({primary}); rollback also failed ({rollback_error})"
            ))),
        }
    }

    pub fn execute<R: F16RebuildTransitionRuntime>(
        &mut self,
        runtime: &mut R,
    ) -> Result<F16RebuildTransitionReceiptV1> {
        self.validate_preconditions()?;

        self.phase = F16RebuildTransitionPhase::Quiescing;
        if let Err(error) = runtime.begin_quiescence() {
            return self.rollback(runtime, error);
        }
        let active_sessions = runtime.active_session_count();
        if active_sessions != 0 {
            return self.rollback(
                runtime,
                NnisError::invalid_input(format!(
                    "F16 rebuild quiescence left {active_sessions} active sessions"
                )),
            );
        }
        if !runtime.source_weights_available() {
            return self.rollback(
                runtime,
                NnisError::invalid_input("F16 rebuild requires source weights after quiescence"),
            );
        }

        self.phase = F16RebuildTransitionPhase::Rebuilding;
        let candidate = match runtime.rebuild_candidate(self.target_plan) {
            Ok(candidate) => candidate,
            Err(error) => return self.rollback(runtime, error),
        };

        self.phase = F16RebuildTransitionPhase::Verifying;
        if let Err(error) = runtime.verify_candidate(&candidate) {
            drop(candidate);
            return self.rollback(runtime, error);
        }

        self.phase = F16RebuildTransitionPhase::InvalidatingSessionState;
        if let Err(error) = runtime.invalidate_sessions_and_kv() {
            drop(candidate);
            return self.rollback(runtime, error);
        }

        self.phase = F16RebuildTransitionPhase::Committing;
        if let Err(error) = runtime.commit_candidate(candidate) {
            return self.rollback(runtime, error);
        }
        self.phase = F16RebuildTransitionPhase::Committed;

        let receipt = F16RebuildTransitionReceiptV1 {
            schema_version: NNIS_F16_REBUILD_TRANSITION_VERSION,
            source_plan: self.source_plan,
            target_plan: self.target_plan,
            required_double_resident_bytes: self.memory_budget.required_double_resident_bytes()?,
            budget_headroom_bytes: self.memory_budget.headroom_bytes()?,
            source_retained_until_commit: true,
            sessions_invalidated: true,
            kv_state_reset: true,
            candidate_verified: true,
            committed: true,
            live_transition_claimed: false,
        };
        receipt.validate()?;
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F16ReferencePlan;

    #[test]
    fn every_current_f16_execution_layout_requires_model_rebuild() {
        let plans = [
            F16ReferenceExecutionPlan::reference(F16ReferencePlan::edge_llm_v0_10_0_alignment()),
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

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum FailAt {
        Quiesce,
        Rebuild,
        Verify,
        Invalidate,
        Commit,
    }

    #[derive(Default)]
    struct FakeTransitionRuntime {
        active_sessions: usize,
        source_weights_available: bool,
        fail_at: Option<FailAt>,
        events: Vec<&'static str>,
        committed: bool,
    }

    impl FakeTransitionRuntime {
        fn admitted() -> Self {
            Self {
                active_sessions: 0,
                source_weights_available: true,
                ..Self::default()
            }
        }

        fn maybe_fail(&self, point: FailAt) -> Result<()> {
            if self.fail_at == Some(point) {
                Err(NnisError::invalid_input("injected transition failure"))
            } else {
                Ok(())
            }
        }
    }

    impl F16RebuildTransitionRuntime for FakeTransitionRuntime {
        type Candidate = u64;

        fn begin_quiescence(&mut self) -> Result<()> {
            self.events.push("quiesce");
            self.maybe_fail(FailAt::Quiesce)
        }

        fn active_session_count(&self) -> usize {
            self.active_sessions
        }

        fn source_weights_available(&self) -> bool {
            self.source_weights_available
        }

        fn rebuild_candidate(
            &mut self,
            _target_plan: F16ReferenceExecutionPlan,
        ) -> Result<Self::Candidate> {
            self.events.push("rebuild");
            self.maybe_fail(FailAt::Rebuild)?;
            Ok(7)
        }

        fn verify_candidate(&mut self, candidate: &Self::Candidate) -> Result<()> {
            self.events.push("verify");
            assert_eq!(*candidate, 7);
            self.maybe_fail(FailAt::Verify)
        }

        fn invalidate_sessions_and_kv(&mut self) -> Result<()> {
            self.events.push("invalidate");
            self.maybe_fail(FailAt::Invalidate)
        }

        fn commit_candidate(&mut self, candidate: Self::Candidate) -> Result<()> {
            self.events.push("commit");
            assert_eq!(candidate, 7);
            self.maybe_fail(FailAt::Commit)?;
            self.committed = true;
            Ok(())
        }

        fn rollback_to_source(&mut self) -> Result<()> {
            self.events.push("rollback");
            self.committed = false;
            Ok(())
        }
    }

    fn rebuild_transaction() -> F16RebuildTransitionV1 {
        F16RebuildTransitionV1::new(
            F16ReferenceExecutionPlan::reference(F16ReferencePlan::edge_llm_v0_10_0_alignment()),
            F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_projection_candidate(),
            F16RebuildMemoryBudgetV1::new(100, 140, 256).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn rebuild_transition_commits_only_after_quiesce_verify_and_state_reset() {
        let mut transaction = rebuild_transaction();
        let mut runtime = FakeTransitionRuntime::admitted();
        let receipt = transaction.execute(&mut runtime).unwrap();

        assert_eq!(
            runtime.events,
            ["quiesce", "rebuild", "verify", "invalidate", "commit"]
        );
        assert!(runtime.committed);
        assert_eq!(transaction.phase, F16RebuildTransitionPhase::Committed);
        assert_eq!(receipt.required_double_resident_bytes, 240);
        assert_eq!(receipt.budget_headroom_bytes, 16);
        assert!(receipt.sessions_invalidated);
        assert!(receipt.kv_state_reset);
        assert!(!receipt.live_transition_claimed);
        receipt.validate().unwrap();
    }

    #[test]
    fn memory_admission_fails_before_quiescence() {
        let source =
            F16ReferenceExecutionPlan::reference(F16ReferencePlan::edge_llm_v0_10_0_alignment());
        let target = F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_projection_candidate();
        assert!(F16RebuildMemoryBudgetV1::new(100, 140, 239).is_err());
        assert!(F16RebuildTransitionV1::new(
            source,
            target,
            F16RebuildMemoryBudgetV1 {
                schema_version: NNIS_F16_REBUILD_TRANSITION_VERSION,
                source_resident_owned_bytes: 100,
                target_peak_owned_bytes: 140,
                transition_budget_bytes: 239,
            },
        )
        .is_err());
    }

    #[test]
    fn active_sessions_or_missing_source_weights_roll_back() {
        let mut transaction = rebuild_transaction();
        let mut runtime = FakeTransitionRuntime::admitted();
        runtime.active_sessions = 1;
        assert!(transaction.execute(&mut runtime).is_err());
        assert_eq!(runtime.events, ["quiesce", "rollback"]);
        assert_eq!(transaction.phase, F16RebuildTransitionPhase::RolledBack);

        let mut transaction = rebuild_transaction();
        let mut runtime = FakeTransitionRuntime::admitted();
        runtime.source_weights_available = false;
        assert!(transaction.execute(&mut runtime).is_err());
        assert_eq!(runtime.events, ["quiesce", "rollback"]);
    }

    #[test]
    fn every_runtime_failure_rolls_back_without_commit() {
        for fail_at in [
            FailAt::Quiesce,
            FailAt::Rebuild,
            FailAt::Verify,
            FailAt::Invalidate,
            FailAt::Commit,
        ] {
            let mut transaction = rebuild_transaction();
            let mut runtime = FakeTransitionRuntime::admitted();
            runtime.fail_at = Some(fail_at);
            assert!(transaction.execute(&mut runtime).is_err());
            assert_eq!(runtime.events.last(), Some(&"rollback"));
            assert!(!runtime.committed);
            assert_eq!(transaction.phase, F16RebuildTransitionPhase::RolledBack);
            assert!(transaction.execute(&mut runtime).is_err());
        }
    }

    #[test]
    fn rebuild_wire_contracts_are_strict_and_do_not_claim_hot_switch() {
        let transaction = rebuild_transaction();
        let json = serde_json::to_string(&transaction).unwrap();
        let decoded: F16RebuildTransitionV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, transaction);
        assert!(json.contains("\"phase\":\"prepared\""));

        let unknown = json.replace(
            "\"phase\":\"prepared\"",
            "\"phase\":\"prepared\",\"unknown\":true",
        );
        assert!(serde_json::from_str::<F16RebuildTransitionV1>(&unknown).is_err());
    }

    #[test]
    fn deserialized_or_public_field_drift_is_rejected_before_quiescence() {
        let source =
            F16ReferenceExecutionPlan::reference(F16ReferencePlan::edge_llm_v0_10_0_alignment());

        let mut equal_plans = rebuild_transaction();
        equal_plans.target_plan = source;
        let mut runtime = FakeTransitionRuntime::admitted();
        assert!(equal_plans.execute(&mut runtime).is_err());
        assert!(runtime.events.is_empty());

        let mut unsupported_source = rebuild_transaction();
        unsupported_source.source_plan.schema_version = F16_REFERENCE_EXECUTION_PLAN_VERSION + 1;
        let mut runtime = FakeTransitionRuntime::admitted();
        assert!(unsupported_source.execute(&mut runtime).is_err());
        assert!(runtime.events.is_empty());

        let mut unsupported_target = rebuild_transaction();
        unsupported_target.target_plan.schema_version = F16_REFERENCE_EXECUTION_PLAN_VERSION + 1;
        let mut runtime = FakeTransitionRuntime::admitted();
        assert!(unsupported_target.execute(&mut runtime).is_err());
        assert!(runtime.events.is_empty());
    }
}
