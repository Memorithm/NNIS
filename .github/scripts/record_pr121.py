from pathlib import Path

roadmap = Path('.agent/NNIS_SOVEREIGNTY_ROADMAP.yaml')
text = roadmap.read_text()
if 'f16_materialization_owned_peak_evidence_v1:' not in text:
    marker = '\nreference_products:\n'
    entry = '''
  f16_materialization_owned_peak_evidence_v1:
    pull_request: 121
    exact_head: ccc844f3b08f4c7e06b4176fbeca442fe1b030b8
    merge_commit: e86cf6ba9183bfa7b4c05848e2667469143eeade
    required_CI: CI_363_success_including_Rust_1_77
    smollm2_harness: harness_153_success
    guarantees:
      - successful_F32_to_F16_weight_materialization_records_actual_DeviceBuffer_allocation_lifetimes
      - transposed_projection_KN_temporary_and_NK_resident_overlap_is_counted_from_live_buffer_size_bytes
      - final_tracked_F16_bytes_must_match_exact_resident_F16_WeightAllocationSummaryV1
      - pinned_SmolLM2_Thor_non_timing_harness_publishes_versioned_materialization_evidence
    limits:
      - successful_materialization_only_no_failed_attempt_trace_v1
      - not_physical_GPU_page_residency
      - not_process_wide_VRAM
      - not_CUDA_allocator_context_module_JIT_RoPE_KV_session_or_workspace_memory
      - no_performance_quality_compression_or_live_transition_claim
    next: audit_available_CUDA_memory_introspection_semantics_before_defining_any_physical_memory_evidence
'''
    if marker not in text:
        raise SystemExit('sovereignty insertion marker missing')
    text = text.replace(marker, entry + marker, 1)
roadmap.write_text(text)

ml = Path('.agent/ML_MATURITY_5_OF_5.yaml')
text = ml.read_text()
text = text.replace(
    '  main_head: 50a0a82be55e4ae60867efa483afea1495780cf4',
    '  main_head: e86cf6ba9183bfa7b4c05848e2667469143eeade',
    1,
)
merge_anchor = '      - pr_120_F16_steady_state_weight_allocation_accounting_v1\n'
if 'pr_121_F16_materialization_owned_peak_evidence_v1' not in text:
    if merge_anchor not in text:
        raise SystemExit('NNML2 merge anchor missing')
    text = text.replace(
        merge_anchor,
        merge_anchor + '      - pr_121_F16_materialization_owned_peak_evidence_v1\n',
        1,
    )
cap_anchor = '      - pinned_SmolLM2_F16_accounting_harness_emits_steady_state_owned_bytes_without_timing_or_generation\n'
if 'successful_F16_materialization_owned_allocation_peak_is_event_traced' not in text:
    if cap_anchor not in text:
        raise SystemExit('NNML2 capability anchor missing')
    text = text.replace(
        cap_anchor,
        cap_anchor
        + '      - successful_F16_materialization_owned_allocation_peak_is_event_traced_from_actual_DeviceBuffer_lifetimes\n'
        + '      - transposed_KN_temporary_overlap_with_resident_NK_is_accounted_and_final_state_reconciles_to_WeightAllocationSummaryV1\n'
        + '      - failed_materialization_attempt_peak_is_not_persisted_by_v1\n',
        1,
    )
boundary_anchor = '    execution_transition_boundary:\n'
if 'f16_materialization_owned_peak_boundary:' not in text:
    if boundary_anchor not in text:
        raise SystemExit('NNML2 boundary anchor missing')
    boundary = '''    f16_materialization_owned_peak_boundary:
      contract_version: nnis_model_F16WeightMaterializationMemoryEvidenceV1
      pull_request: 121
      exact_head: ccc844f3b08f4c7e06b4176fbeca442fe1b030b8
      merge_commit: e86cf6ba9183bfa7b4c05848e2667469143eeade
      required_CI: CI_363_success_on_exact_head_including_Rust_1_77
      exact_head_SmolLM2_harness: harness_153_success
      exact_scope: live_source_ModelWeights_plus_live_F16_weight_materialization_DeviceBuffers
      actual_buffer_lifetimes_instrumented: true
      derived_from_model_geometry: false
      failed_attempt_trace_claimed: false
      physical_page_residency_claimed: false
      process_wide_VRAM_claimed: false
      performance_or_quality_claimed: false
'''
    text = text.replace(boundary_anchor, boundary + boundary_anchor, 1)
text = text.replace(
    '      - physical_page_residency_and_F16_conversion_peak_evidence_not_yet_complete',
    '      - physical_page_residency_and_process_wide_VRAM_evidence_not_yet_complete\n      - failed_F16_materialization_attempt_peak_trace_not_yet_persisted',
    1,
)
ml.write_text(text)
