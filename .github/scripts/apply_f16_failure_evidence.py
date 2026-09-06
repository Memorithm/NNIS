from pathlib import Path

memory = Path('crates/nnis-model/src/f16_materialization_memory.rs')
text = memory.read_text()
text = text.replace(
    'pub const NNIS_F16_WEIGHT_MATERIALIZATION_MEMORY_EVIDENCE_VERSION: u32 = 1;\n',
    'pub const NNIS_F16_WEIGHT_MATERIALIZATION_MEMORY_EVIDENCE_VERSION: u32 = 1;\n'
    'pub const NNIS_F16_WEIGHT_MATERIALIZATION_FAILURE_EVIDENCE_VERSION: u32 = 1;\n',
    1,
)
marker = '#[derive(Debug)]\npub(crate) struct F16WeightMaterializationTracker'
failure_struct = '''#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct F16WeightMaterializationFailureEvidenceV1 {
    pub schema_version: u32,
    pub execution_plan: F16ReferenceExecutionPlan,
    pub source_weight_allocations: WeightAllocationSummaryV1,
    pub failure_operation: Option<String>,
    pub failure_driver_code: Option<i32>,
    pub failure_message: String,
    pub live_f16_allocation_bytes_at_failure_detection: u64,
    pub live_temporary_f16_allocation_bytes_at_failure_detection: u64,
    pub scoped_owned_allocation_bytes_at_failure_detection: u64,
    pub peak_live_f16_allocation_bytes: u64,
    pub peak_live_temporary_f16_allocation_bytes: u64,
    pub peak_scoped_owned_allocation_bytes: u64,
    pub events: Vec<F16WeightMaterializationEventV1>,
}

'''
if 'pub struct F16WeightMaterializationFailureEvidenceV1' not in text:
    if marker not in text:
        raise SystemExit('tracker marker missing')
    text = text.replace(marker, failure_struct + marker, 1)

finish_marker = '    pub(crate) fn finish(\n'
failure_method = '''    pub(crate) fn failure_evidence(
        self,
        execution_plan: F16ReferenceExecutionPlan,
        source_weight_allocations: WeightAllocationSummaryV1,
        error: &NnisError,
    ) -> Result<F16WeightMaterializationFailureEvidenceV1> {
        if source_weight_allocations.owned_device_allocation_bytes
            != self.source_owned_allocation_bytes
        {
            return Err(NnisError::invalid_input(
                "source weight summary changed during failed F16 materialization accounting",
            ));
        }
        let scoped_owned_allocation_bytes_at_failure_detection = self
            .source_owned_allocation_bytes
            .checked_add(self.live_f16_allocation_bytes)
            .ok_or_else(|| {
                NnisError::invalid_input(
                    "failed F16 materialization scoped allocation bytes overflow u64",
                )
            })?;
        let failure_operation = if error.op().is_empty() {
            None
        } else {
            Some(error.op().to_string())
        };
        Ok(F16WeightMaterializationFailureEvidenceV1 {
            schema_version: NNIS_F16_WEIGHT_MATERIALIZATION_FAILURE_EVIDENCE_VERSION,
            execution_plan,
            source_weight_allocations,
            failure_operation,
            failure_driver_code: error.driver_code(),
            failure_message: error.to_string(),
            live_f16_allocation_bytes_at_failure_detection: self.live_f16_allocation_bytes,
            live_temporary_f16_allocation_bytes_at_failure_detection: self
                .live_temporary_f16_allocation_bytes,
            scoped_owned_allocation_bytes_at_failure_detection,
            peak_live_f16_allocation_bytes: self.peak_live_f16_allocation_bytes,
            peak_live_temporary_f16_allocation_bytes: self
                .peak_live_temporary_f16_allocation_bytes,
            peak_scoped_owned_allocation_bytes: self.peak_scoped_owned_allocation_bytes,
            events: self.events,
        })
    }

'''
if 'pub(crate) fn failure_evidence(' not in text:
    if finish_marker not in text:
        raise SystemExit('finish marker missing')
    text = text.replace(finish_marker, failure_method + finish_marker, 1)

test_anchor = '    #[test]\n    fn final_resident_mismatch_fails_closed() {'
if 'fn failure_evidence_preserves_pre_cleanup_peak_and_driver_error()' not in text:
    test = '''    #[test]
    fn failure_evidence_preserves_pre_cleanup_peak_and_driver_error() {
        let mut tracker = F16WeightMaterializationTracker::new(100).unwrap();
        tracker.allocate_resident("prior", 20).unwrap();
        tracker
            .allocate_temporary("proj.kn_temporary", 30)
            .unwrap();
        tracker.allocate_resident("proj", 30).unwrap();
        let error = NnisError::driver("transpose", 2).with("weight", "proj");
        let evidence = tracker
            .failure_evidence(
                plan(F16ReferenceProjectionLayout::NkTransposedCandidate),
                summary(100, WeightAllocationDTypeV1::F32),
                &error,
            )
            .unwrap();

        assert_eq!(
            evidence.schema_version,
            NNIS_F16_WEIGHT_MATERIALIZATION_FAILURE_EVIDENCE_VERSION
        );
        assert_eq!(evidence.failure_operation.as_deref(), Some("transpose"));
        assert_eq!(evidence.failure_driver_code, Some(2));
        assert!(evidence.failure_message.contains("weight: proj"));
        assert_eq!(evidence.live_f16_allocation_bytes_at_failure_detection, 80);
        assert_eq!(
            evidence.live_temporary_f16_allocation_bytes_at_failure_detection,
            30
        );
        assert_eq!(evidence.scoped_owned_allocation_bytes_at_failure_detection, 180);
        assert_eq!(evidence.peak_scoped_owned_allocation_bytes, 180);
        assert_eq!(evidence.events.len(), 3);
    }

    #[test]
    fn failure_evidence_before_first_f16_allocation_reports_source_only() {
        let tracker = F16WeightMaterializationTracker::new(100).unwrap();
        let error = NnisError::invalid_input("synthetic pre-allocation materialization failure");
        let evidence = tracker
            .failure_evidence(
                plan(F16ReferenceProjectionLayout::KnReference),
                summary(100, WeightAllocationDTypeV1::F32),
                &error,
            )
            .unwrap();
        assert_eq!(evidence.failure_operation, None);
        assert_eq!(evidence.failure_driver_code, None);
        assert_eq!(evidence.live_f16_allocation_bytes_at_failure_detection, 0);
        assert_eq!(evidence.scoped_owned_allocation_bytes_at_failure_detection, 100);
        assert_eq!(evidence.peak_scoped_owned_allocation_bytes, 100);
        assert!(evidence.events.is_empty());
    }

'''
    if test_anchor not in text:
        raise SystemExit('test anchor missing')
    text = text.replace(test_anchor, test + test_anchor, 1)
memory.write_text(text)

runtime = Path('crates/nnis-model/src/f16_reference_runtime.rs')
text = runtime.read_text()
text = text.replace(
    '    F16WeightMaterializationMemoryEvidenceV1, F16WeightMaterializationTracker,\n',
    '    F16WeightMaterializationFailureEvidenceV1, F16WeightMaterializationMemoryEvidenceV1,\n'
    '    F16WeightMaterializationTracker,\n',
    1,
)

model_marker = '#[derive(Debug)]\npub struct F16ReferenceModel {'
internal_types = '''#[derive(Debug)]
enum F16WeightMaterializationBuildAttempt {
    Completed(F16ModelWeights, F16WeightMaterializationTracker),
    Failed {
        error: NnisError,
        tracker: F16WeightMaterializationTracker,
    },
}

struct F16MaterializationFailureSink<'a> {
    evidence: &'a mut Option<F16WeightMaterializationFailureEvidenceV1>,
    evidence_error: &'a mut Option<String>,
}

#[derive(Debug)]
pub struct F16ReferenceModelConstructionFailure {
    error: NnisError,
    materialization_failure_evidence: Option<F16WeightMaterializationFailureEvidenceV1>,
    materialization_evidence_error: Option<String>,
}

impl F16ReferenceModelConstructionFailure {
    pub fn error(&self) -> &NnisError {
        &self.error
    }

    pub fn materialization_failure_evidence(
        &self,
    ) -> Option<&F16WeightMaterializationFailureEvidenceV1> {
        self.materialization_failure_evidence.as_ref()
    }

    pub fn materialization_evidence_error(&self) -> Option<&str> {
        self.materialization_evidence_error.as_deref()
    }

    pub fn into_error(self) -> NnisError {
        self.error
    }
}

'''
if 'pub struct F16ReferenceModelConstructionFailure' not in text:
    if model_marker not in text:
        raise SystemExit('model marker missing')
    text = text.replace(model_marker, internal_types + model_marker, 1)

old_sig = '''    fn from_f32(
        source: &ModelWeights,
        stream: &Stream,
        kernels: &F16ReferenceKernels,
        execution_plan: F16ReferenceExecutionPlan,
        projection_candidate: Option<&F16TransposedProjectionCandidate>,
        source_owned_allocation_bytes: u64,
    ) -> Result<(Self, F16WeightMaterializationTracker)> {
'''
new_sig = '''    fn from_f32(
        source: &ModelWeights,
        stream: &Stream,
        kernels: &F16ReferenceKernels,
        execution_plan: F16ReferenceExecutionPlan,
        projection_candidate: Option<&F16TransposedProjectionCandidate>,
        source_owned_allocation_bytes: u64,
    ) -> Result<F16WeightMaterializationBuildAttempt> {
'''
if old_sig not in text:
    raise SystemExit('from_f32 signature missing')
text = text.replace(old_sig, new_sig, 1)

start_marker = '        let mut tracker = F16WeightMaterializationTracker::new(source_owned_allocation_bytes)?;\n        let layout = execution_plan.projection_layout;\n'
if 'let build_result = (|| -> Result<Self>' not in text:
    if start_marker not in text:
        raise SystemExit('tracker start marker missing')
    text = text.replace(
        start_marker,
        '        let mut tracker = F16WeightMaterializationTracker::new(source_owned_allocation_bytes)?;\n'
        '        let build_result = (|| -> Result<Self> {\n'
        '        let layout = execution_plan.projection_layout;\n',
        1,
    )
old_tail = '''        Ok((
            Self {
                token_embedding,
                layers,
                final_norm,
                lm_head,
            },
            tracker,
        ))
    }
'''
new_tail = '''        Ok(Self {
            token_embedding,
            layers,
            final_norm,
            lm_head,
        })
        })();
        Ok(match build_result {
            Ok(weights) => F16WeightMaterializationBuildAttempt::Completed(weights, tracker),
            Err(error) => F16WeightMaterializationBuildAttempt::Failed { error, tracker },
        })
    }
'''
if old_tail not in text:
    raise SystemExit('from_f32 tail missing')
text = text.replace(old_tail, new_tail, 1)

old_general_header = '''    pub fn new_with_execution_and_attention_plan(
        config: ModelConfig,
        weights: ModelWeights,
        stream: &Stream,
        execution_plan: F16ReferenceExecutionPlan,
        attention_plan: F16AttentionPlan,
    ) -> Result<Self> {
'''
new_general_header = '''    pub fn new_with_execution_and_attention_plan(
        config: ModelConfig,
        weights: ModelWeights,
        stream: &Stream,
        execution_plan: F16ReferenceExecutionPlan,
        attention_plan: F16AttentionPlan,
    ) -> Result<Self> {
        Self::new_with_execution_and_attention_plan_impl(
            config,
            weights,
            stream,
            execution_plan,
            attention_plan,
            None,
        )
    }

    pub fn new_with_execution_plan_attempt(
        config: ModelConfig,
        weights: ModelWeights,
        stream: &Stream,
        execution_plan: F16ReferenceExecutionPlan,
    ) -> std::result::Result<Self, F16ReferenceModelConstructionFailure> {
        Self::new_with_execution_and_attention_plan_attempt(
            config,
            weights,
            stream,
            execution_plan,
            F16AttentionPlan::reference(),
        )
    }

    pub fn new_with_execution_and_attention_plan_attempt(
        config: ModelConfig,
        weights: ModelWeights,
        stream: &Stream,
        execution_plan: F16ReferenceExecutionPlan,
        attention_plan: F16AttentionPlan,
    ) -> std::result::Result<Self, F16ReferenceModelConstructionFailure> {
        let mut materialization_failure_evidence = None;
        let mut materialization_evidence_error = None;
        let mut sink = F16MaterializationFailureSink {
            evidence: &mut materialization_failure_evidence,
            evidence_error: &mut materialization_evidence_error,
        };
        match Self::new_with_execution_and_attention_plan_impl(
            config,
            weights,
            stream,
            execution_plan,
            attention_plan,
            Some(&mut sink),
        ) {
            Ok(model) => Ok(model),
            Err(error) => Err(F16ReferenceModelConstructionFailure {
                error,
                materialization_failure_evidence,
                materialization_evidence_error,
            }),
        }
    }

    fn new_with_execution_and_attention_plan_impl(
        config: ModelConfig,
        weights: ModelWeights,
        stream: &Stream,
        execution_plan: F16ReferenceExecutionPlan,
        attention_plan: F16AttentionPlan,
        materialization_failure_sink: Option<&mut F16MaterializationFailureSink<'_>>,
    ) -> Result<Self> {
'''
if old_general_header not in text:
    raise SystemExit('general constructor header missing')
text = text.replace(old_general_header, new_general_header, 1)

old_materialize = '''        let source_weight_allocations = weights.weight_allocation_summary_v1()?;
        let (resident_weights, materialization_tracker) = F16ModelWeights::from_f32(
            &weights,
            stream,
            &kernels,
            execution_plan,
            projection_candidate.as_ref(),
            source_weight_allocations.owned_device_allocation_bytes,
        )?;
        let steady_state_f16_weight_allocations =
            resident_weights.weight_allocation_summary_v1()?;
        let materialization_memory_evidence = materialization_tracker.finish(
            execution_plan,
            source_weight_allocations,
            steady_state_f16_weight_allocations,
        )?;
'''
new_materialize = '''        let source_weight_allocations = weights.weight_allocation_summary_v1()?;
        let materialization_attempt = F16ModelWeights::from_f32(
            &weights,
            stream,
            &kernels,
            execution_plan,
            projection_candidate.as_ref(),
            source_weight_allocations.owned_device_allocation_bytes,
        )?;
        let (resident_weights, materialization_tracker) = match materialization_attempt {
            F16WeightMaterializationBuildAttempt::Completed(weights, tracker) => (weights, tracker),
            F16WeightMaterializationBuildAttempt::Failed { error, tracker } => {
                if let Some(sink) = materialization_failure_sink {
                    match tracker.failure_evidence(
                        execution_plan,
                        source_weight_allocations,
                        &error,
                    ) {
                        Ok(evidence) => *sink.evidence = Some(evidence),
                        Err(evidence_error) => {
                            *sink.evidence_error = Some(evidence_error.to_string());
                        }
                    }
                }
                return Err(error);
            }
        };
        let steady_state_f16_weight_allocations =
            resident_weights.weight_allocation_summary_v1()?;
        let materialization_memory_evidence = materialization_tracker.finish(
            execution_plan,
            source_weight_allocations,
            steady_state_f16_weight_allocations,
        )?;
'''
if old_materialize not in text:
    raise SystemExit('materialization constructor block missing')
text = text.replace(old_materialize, new_materialize, 1)
runtime.write_text(text)

lib = Path('crates/nnis-model/src/lib.rs')
text = lib.read_text()
old_export = '''pub use f16_materialization_memory::{
    F16WeightMaterializationEventKindV1, F16WeightMaterializationEventV1,
    F16WeightMaterializationMemoryEvidenceV1,
    NNIS_F16_WEIGHT_MATERIALIZATION_MEMORY_EVIDENCE_VERSION,
};
'''
new_export = '''pub use f16_materialization_memory::{
    F16WeightMaterializationEventKindV1, F16WeightMaterializationEventV1,
    F16WeightMaterializationFailureEvidenceV1, F16WeightMaterializationMemoryEvidenceV1,
    NNIS_F16_WEIGHT_MATERIALIZATION_FAILURE_EVIDENCE_VERSION,
    NNIS_F16_WEIGHT_MATERIALIZATION_MEMORY_EVIDENCE_VERSION,
};
'''
if old_export not in text:
    raise SystemExit('materialization export block missing')
text = text.replace(old_export, new_export, 1)
old_runtime_export = '    F16ReferenceAccumulator, F16ReferenceLogits, F16ReferenceModel, F16ReferencePlan,\n'
new_runtime_export = '    F16ReferenceAccumulator, F16ReferenceLogits, F16ReferenceModel,\n    F16ReferenceModelConstructionFailure, F16ReferencePlan,\n'
if old_runtime_export not in text:
    raise SystemExit('runtime export anchor missing')
text = text.replace(old_runtime_export, new_runtime_export, 1)
lib.write_text(text)
