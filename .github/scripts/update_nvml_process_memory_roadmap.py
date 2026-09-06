from pathlib import Path

roadmap = Path('.agent/NNIS_SOVEREIGNTY_ROADMAP.yaml')
text = roadmap.read_text()
anchor = '''    next: qualify_a_semantically_correct_process_memory_source_before_any_process_wide_VRAM_claim

reference_products:
'''
replacement = '''    next: qualify_a_semantically_correct_process_memory_source_before_any_process_wide_VRAM_claim

  nvml_process_memory_snapshot_v1:
    pull_request: 125
    exact_head: b1c45d69a9d4196a008063e922ca64c77c3f5c7f
    merge_commit: 710d7c7466b4bee9d6accb42531e49fef2df4fb1
    required_CI: CI_378_success_including_Rust_1_77
    guarantees:
      - NVML_is_dynamically_loaded_without_a_link_time_dependency
      - CUDA_device_is_correlated_to_NVML_by_UUID_not_assumed_ordinal_equivalence
      - current_PID_usedGpuMemory_is_reported_as_a_versioned_process_scoped_snapshot
      - unavailable_missing_duplicate_or_unstable_process_records_fail_closed
      - process_list_growth_is_bounded
    limits:
      - software_contract_only_no_real_Thor_measurement_in_PR_125
      - not_physical_GPU_page_residency
      - not_per_allocation_residency
      - not_CUDA_allocator_or_page_table_overhead_attribution
      - not_weight_only_memory_attribution
      - no_performance_quality_compression_or_cross_runtime_memory_claim
    next: run_a_pinned_SmolLM2_Thor_lifecycle_campaign_reporting_NVML_process_memory_alongside_exact_NNIS_owned_allocation_summaries

reference_products:
'''
if anchor not in text:
    raise SystemExit('roadmap insertion anchor missing')
text = text.replace(anchor, replacement, 1)
roadmap.write_text(text)

ml = Path('.agent/ML_MATURITY_5_OF_5.yaml')
text = ml.read_text()
text = text.replace(
    '  main_head: 8982dce75e9ea68554f4320f427032fa3d66c075\n',
    '  main_head: 710d7c7466b4bee9d6accb42531e49fef2df4fb1\n',
    1,
)
merged_anchor = '      - pr_123_F16_materialization_failure_evidence_v1\n'
if '      - pr_125_NVML_process_memory_snapshot_v1\n' not in text:
    if merged_anchor not in text:
        raise SystemExit('NNML2 merged anchor missing')
    text = text.replace(
        merged_anchor,
        merged_anchor + '      - pr_125_NVML_process_memory_snapshot_v1\n',
        1,
    )
cap_anchor = '      - CUDA_free_total_and_pointer_metadata_are_explicitly_not_treated_as_physical_residency_or_process_attribution\n'
if 'NVML_current_PID_usedGpuMemory_is_available_as_a_fail_closed_process_scoped_software_contract' not in text:
    if cap_anchor not in text:
        raise SystemExit('NNML2 capability anchor missing')
    text = text.replace(
        cap_anchor,
        cap_anchor
        + '      - NVML_current_PID_usedGpuMemory_is_available_as_a_fail_closed_process_scoped_software_contract\n'
        + '      - CUDA_to_NVML_device_correlation_uses_UUID_and_does_not_assume_ordinal_equivalence\n'
        + '      - NVML_process_memory_is_not_relabelled_as_physical_page_residency_or_per_allocation_ownership\n',
        1,
    )
boundary_anchor = '    execution_transition_boundary:\n'
boundary = '''    nvml_process_memory_boundary:
      contract_version: nnis_rt_NvmlProcessMemorySnapshotV1
      pull_request: 125
      exact_head: b1c45d69a9d4196a008063e922ca64c77c3f5c7f
      merge_commit: 710d7c7466b4bee9d6accb42531e49fef2df4fb1
      required_CI: CI_378_success_on_exact_head_including_Rust_1_77
      exact_scope: NVML_usedGpuMemory_for_current_PID_on_UUID_correlated_target_GPU
      software_contract_qualified: true
      real_Thor_model_lifecycle_measurement_collected: false
      physical_page_residency_claimed: false
      per_allocation_attribution_claimed: false
      allocator_or_page_table_overhead_claimed: false
      performance_or_quality_claimed: false
'''
if '    nvml_process_memory_boundary:\n' not in text:
    if boundary_anchor not in text:
        raise SystemExit('NNML2 boundary anchor missing')
    text = text.replace(boundary_anchor, boundary + boundary_anchor, 1)
old_blocker = '      - physical_page_residency_and_process_wide_VRAM_evidence_not_yet_complete\n'
new_blockers = (
    '      - physical_page_residency_is_not_claimed_or_established_by_the_current_NVML_contract\n'
    '      - real_Thor_process_scoped_NVML_model_lifecycle_evidence_not_yet_collected\n'
)
if old_blocker in text:
    text = text.replace(old_blocker, new_blockers, 1)
ml.write_text(text)
