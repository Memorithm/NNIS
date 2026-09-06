from pathlib import Path

ROADMAP = Path('.agent/NNIS_SOVEREIGNTY_ROADMAP.yaml')
ML = Path('.agent/ML_MATURITY_5_OF_5.yaml')

roadmap = ROADMAP.read_text()
if 'smollm2_nvml_lifecycle_artifact_validation_v1:' not in roadmap:
    anchor = '\nreference_products:\n'
    if anchor not in roadmap:
        raise SystemExit('roadmap reference_products anchor missing')
    insertion = '''\n  smollm2_nvml_lifecycle_harness_v1:\n    pull_request: 126\n    exact_head: 15f23004dfe72b901cbdccd561efa026186d7730\n    merge_commit: 150ba5e5fbc1417695ef31533753c824baa1fcad\n    guarantees:\n      - pinned_SmolLM2_lifecycle_harness_records_post_context_source_F32_and_resident_F16_NVML_process_memory\n      - source_and_resident_weight_graphs_include_exact_WeightAllocationSummaryV1_alongside_NVML_observations\n      - F16_materialization_owned_peak_evidence_is_embedded_without_relabelling_NVML_as_owned_bytes\n    limits:\n      - software_harness_only_no_real_Thor_lifecycle_result_in_PR_126\n      - not_physical_GPU_page_residency\n      - no_NVML_minus_owned_bytes_overhead_attribution\n      - no_performance_quality_compression_or_live_transition_claim\n\n  smollm2_nvml_lifecycle_artifact_validation_v1:\n    pull_request: 127\n    exact_head: d058120bf0a5a0cd0686150de947fb941baf85db\n    merge_commit: 26f4d5ab200501568155d5165bcb05b0c9e7cf61\n    required_CI: CI_382_success_including_Rust_1_77\n    smollm2_harness: harness_163_success\n    guarantees:\n      - lifecycle_JSON_can_be_atomically_persisted_as_the_exact_stdout_bytes\n      - validator_fails_closed_on_checkpoint_head_Thor_run_context_PID_device_UUID_and_allocation_summary_drift\n      - source_and_resident_allocation_summaries_must_reconcile_byte_for_byte_with_materialization_evidence\n      - permanent_SmolLM2_CI_runs_validator_syntax_help_and_corruption_self_tests\n    limits:\n      - software_artifact_and_validator_only_no_real_Thor_measurement_in_PR_127\n      - not_physical_GPU_page_residency\n      - no_allocator_context_page_table_module_JIT_workspace_KV_session_or_RoPE_overhead_attribution\n      - no_performance_quality_compression_or_live_transition_claim\n    next: execute_the_persisted_validated_lifecycle_campaign_on_real_Jetson_AGX_Thor_at_an_exact_clean_green_head\n'''
    roadmap = roadmap.replace(anchor, insertion + anchor, 1)
    ROADMAP.write_text(roadmap)

ml = ML.read_text()
ml = ml.replace(
    '  main_head: 710d7c7466b4bee9d6accb42531e49fef2df4fb1\n',
    '  main_head: 26f4d5ab200501568155d5165bcb05b0c9e7cf61\n',
    1,
)
if '      - pr_126_SmolLM2_NVML_lifecycle_memory_harness_v1\n' not in ml:
    anchor = '      - pr_125_NVML_process_memory_snapshot_v1\n'
    if anchor not in ml:
        raise SystemExit('NNML2 merged PR125 anchor missing')
    ml = ml.replace(
        anchor,
        anchor
        + '      - pr_126_SmolLM2_NVML_lifecycle_memory_harness_v1\n'
        + '      - pr_127_SmolLM2_NVML_lifecycle_artifact_validation_v1\n',
        1,
    )
if '      - SmolLM2_NVML_lifecycle_artifacts_are_atomically_persisted_and_fail_closed_validated_before_physical_use\n' not in ml:
    anchor = '      - NVML_process_memory_is_not_relabelled_as_physical_page_residency_or_per_allocation_ownership\n'
    if anchor not in ml:
        raise SystemExit('NNML2 capability anchor missing')
    ml = ml.replace(
        anchor,
        anchor
        + '      - SmolLM2_NVML_lifecycle_harness_records_three_synchronized_process_memory_boundaries_alongside_exact_owned_weight_summaries\n'
        + '      - SmolLM2_NVML_lifecycle_artifacts_are_atomically_persisted_and_fail_closed_validated_before_physical_use\n',
        1,
    )
if '    smollm2_nvml_lifecycle_artifact_boundary:' not in ml:
    anchor = '    execution_transition_boundary:\n'
    if anchor not in ml:
        raise SystemExit('execution transition boundary anchor missing')
    insertion = '''    smollm2_nvml_lifecycle_artifact_boundary:\n      contract_version: nnis_smollm2_nvml_lifecycle_memory_v1\n      harness_pull_request: 126\n      artifact_validator_pull_request: 127\n      artifact_validator_exact_head: d058120bf0a5a0cd0686150de947fb941baf85db\n      artifact_validator_merge_commit: 26f4d5ab200501568155d5165bcb05b0c9e7cf61\n      required_CI: CI_382_success_on_exact_head_including_Rust_1_77\n      exact_head_SmolLM2_harness: harness_163_success\n      atomic_artifact_output_qualified: true\n      pinned_checkpoint_and_exact_head_validation_qualified: true\n      Thor_run_context_PID_device_UUID_and_allocation_reconciliation_fail_closed: true\n      real_Thor_model_lifecycle_measurement_collected: false\n      physical_page_residency_claimed: false\n      allocator_or_runtime_overhead_attribution_claimed: false\n      performance_or_quality_claimed: false\n'''
    ml = ml.replace(anchor, insertion + anchor, 1)
ML.write_text(ml)
