from pathlib import Path

path = Path('.github/scripts/apply_f16_failure_evidence.py')
text = path.read_text()

old = "old_runtime_export = '    F16ReferenceAccumulator, F16ReferenceLogits, F16ReferenceModel, F16ReferencePlan,\\n'"
new = "old_runtime_export = '    F16ReferenceAccumulator, F16ReferenceGenerationProfile, F16ReferenceLogits, F16ReferenceModel,\\n    F16ReferencePlan, F16ReferenceSession, F16ReferenceStorage, F16_REFERENCE_PLAN_VERSION,\\n'"
text = text.replace(old, new, 1)
old_new = "new_runtime_export = '    F16ReferenceAccumulator, F16ReferenceLogits, F16ReferenceModel,\\n    F16ReferenceModelConstructionFailure, F16ReferencePlan,\\n'"
new_new = "new_runtime_export = '    F16ReferenceAccumulator, F16ReferenceGenerationProfile, F16ReferenceLogits, F16ReferenceModel,\\n    F16ReferenceModelConstructionFailure, F16ReferencePlan, F16ReferenceSession,\\n    F16ReferenceStorage, F16_REFERENCE_PLAN_VERSION,\\n'"
text = text.replace(old_new, new_new, 1)

text = text.replace(
    '    error: NnisError,',
    '    error: Box<NnisError>,',
    1,
)
text = text.replace(
    '    materialization_failure_evidence: Option<F16WeightMaterializationFailureEvidenceV1>,',
    '    materialization_failure_evidence: Option<Box<F16WeightMaterializationFailureEvidenceV1>>,',
    1,
)
text = text.replace(
    '        &self.error',
    '        self.error.as_ref()',
    1,
)
text = text.replace(
    '        self.materialization_failure_evidence.as_ref()',
    '        self.materialization_failure_evidence.as_deref()',
    1,
)
text = text.replace(
    '        self.error\n    }',
    '        *self.error\n    }',
    1,
)
text = text.replace(
    '                error,\n                materialization_failure_evidence,\n                materialization_evidence_error,',
    '                error: Box::new(error),\n                materialization_failure_evidence: materialization_failure_evidence.map(Box::new),\n                materialization_evidence_error,',
    1,
)

path.write_text(text)
