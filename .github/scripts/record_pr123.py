from pathlib import Path

roadmap = Path('.agent/NNIS_SOVEREIGNTY_ROADMAP.yaml')
text = roadmap.read_text()
if 'f16_materialization_failure_evidence_v1:' not in text:
    marker = '\nreference_products:\n'
    entry = '''
  f16_materialization_failure_evidence_v1:
    pull_request: 123
    exact_head: a569ffda17822e980c4e61ede7963f2309dfedca
    merge_commit: 8982dce75e9ea68554f4320f427032fa3d66c075
    required_CI: CI_366_success_including_Rust_1_77
    smollm2_harness: harness_155_success
    guarantees:
      - failed_tracked_F16_materialization_can_retain_pre_cleanup_owned_allocation_lifetime_evidence
      - original_NnisError_retains_precedence_and_existing_constructors_preserve_their_Result_error_surface
      - opt_in_attempt_constructors_expose_optional_failure_evidence_without_claiming_post_return_liveness
      - CUDA_memory_introspection_audit_rejects_relabeling_global_free_total_or_pointer_metadata_as_physical_residency
    limits:
      - evidence_only_for_failures_inside_the_tracked_F16_weight_materialization_phase
      - not_physical_GPU_page_residency
      - not_process_wide_VRAM_attribution
      - no_performance_quality_compression_or_live_transition_claim
    next: qualify_a_semantically_correct_process_memory_source_before_any_process_wide_VRAM_claim
'''
    if marker not in text:
        raise SystemExit('sovereignty insertion marker missing')
    text = text.replace(marker, entry + marker, 1)
roadmap.write_text(text)

ml = Path('.agent/ML_MATURITY_5_OF_5.yaml')
text = ml.read_text()
text = text.replace(
    '  main_head: e86cf6ba9183bfa7b4c05848e2667469143eeade',
    '  main_head: 8982dce75e9ea68554f4320f427032fa3d66c075',
    1,
)
merge_anchor = '      - pr_121_F16_materialization_owned_peak_evidence_v1\n'
if 'pr_123_F16_materialization_failure_evidence_v1' not in text:
    if merge_anchor not in text:
        raise SystemExit('NNML2 merge anchor missing')
    text = text.replace(
        merge_anchor,
        merge_anchor + '      - pr_123_F16_materialization_failure_evidence_v1\n',
        1,
    )
cap_anchor = '      - failed_materialization_attempt_peak_is_not_persisted_by_v1\n'
if cap_anchor in text:
    text = text.replace(
        cap_anchor,
        '      - failed_F16_materialization_attempt_peak_is_persisted_at_failure_detection_before_RAII_cleanup\n'
        '      - historical_F16_constructors_preserve_the_original_NnisError_while_attempt_constructors_can_expose_failure_evidence\n'
        '      - CUDA_free_total_and_pointer_metadata_are_explicitly_not_treated_as_physical_residency_or_process_attribution\n',
        1,
    )
boundary_anchor = '    execution_transition_boundary:\n'
if 'f16_materialization_failure_boundary:' not in text:
    if boundary_anchor not in text:
        raise SystemExit('NNML2 boundary anchor missing')
    boundary = '''    f16_materialization_failure_boundary:
      contract_version: nnis_model_F16WeightMaterializationFailureEvidenceV1
      pull_request: 123
      exact_head: a569ffda17822e980c4e61ede7963f2309dfedca
      merge_commit: 8982dce75e9ea68554f4320f427032fa3d66c075
      required_CI: CI_366_success_on_exact_head_including_Rust_1_77
      exact_head_SmolLM2_harness: harness_155_success
      exact_scope: tracked_source_ModelWeights_plus_live_F16_materialization_DeviceBuffers_at_failure_detection
      pre_cleanup_failure_state_captured: true
      post_return_allocation_liveness_claimed: false
      physical_page_residency_claimed: false
      process_wide_VRAM_claimed: false
      performance_or_quality_claimed: false
'''
    text = text.replace(boundary_anchor, boundary + boundary_anchor, 1)
text = text.replace(
    '      - failed_F16_materialization_attempt_peak_trace_not_yet_persisted\n',
    '',
    1,
)
ml.write_text(text)
