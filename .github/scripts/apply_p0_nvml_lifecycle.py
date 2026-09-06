from pathlib import Path

path = Path('tools/run_p0_physical_qualification_bundle.py')
text = path.read_text()

text = text.replace(
    'import json\nimport re\n',
    'import json\nimport os\nimport re\n',
    1,
)
text = text.replace(
    'SMOLLM2_DECODE_STEPS = 32\n',
    'SMOLLM2_DECODE_STEPS = 32\nRUN_CONTEXT_ENV = "NNIS_BENCH_RUN_CONTEXT_ID"\n',
    1,
)

helper_anchor = '''def parse_args() -> argparse.Namespace:\n'''
helper = '''def require_run_context_id(value: str | None) -> str:\n    if value is None or not value.strip():\n        raise QualificationError(\n            f"{RUN_CONTEXT_ENV} must be a non-empty explicit physical campaign id"\n        )\n    return value.strip()\n\n\n'''
if 'def require_run_context_id(' not in text:
    if helper_anchor not in text:
        raise SystemExit('parse_args anchor missing')
    text = text.replace(helper_anchor, helper + helper_anchor, 1)

run_anchor = '''    for command in ["git", "cargo", "curl", "sha256sum", "bash"]:\n        require_command(command)\n\n    root = repository_root()\n'''
run_replacement = '''    for command in ["git", "cargo", "curl", "sha256sum", "bash"]:\n        require_command(command)\n    run_context_id = require_run_context_id(os.environ.get(RUN_CONTEXT_ENV))\n\n    root = repository_root()\n'''
if run_anchor not in text:
    raise SystemExit('run preflight anchor missing')
text = text.replace(run_anchor, run_replacement, 1)

path_anchor = '''    smollm2_logit_report = work_dir / "smollm2-logit-report.log"\n    smollm2_reference_manifest = smollm2_fixture / "reference" / "reference.json"\n'''
path_replacement = '''    smollm2_logit_report = work_dir / "smollm2-logit-report.log"\n    smollm2_nvml_lifecycle = work_dir / "smollm2-nvml-lifecycle-memory.json"\n    smollm2_reference_manifest = smollm2_fixture / "reference" / "reference.json"\n'''
if path_anchor not in text:
    raise SystemExit('SmolLM2 path anchor missing')
text = text.replace(path_anchor, path_replacement, 1)

fixture_anchor = '''    run_stream(smollm2_fixture_command, cwd=root)\n    validate_smollm2_reference_contract(smollm2_reference_manifest)\n    if smollm2_parity.exists():\n'''
lifecycle_block = '''    run_stream(smollm2_fixture_command, cwd=root)\n    validate_smollm2_reference_contract(smollm2_reference_manifest)\n\n    if smollm2_nvml_lifecycle.exists():\n        smollm2_nvml_lifecycle.unlink()\n    run_stream(\n        [\n            "cargo",\n            "run",\n            "--locked",\n            "-p",\n            "nnis-bench",\n            "--example",\n            "smollm2_nvml_lifecycle_memory",\n            "--",\n            "--model",\n            str(smollm2_fixture / "model"),\n            "--device",\n            "0",\n            "--output",\n            str(smollm2_nvml_lifecycle),\n        ],\n        cwd=root,\n    )\n    run_stream(\n        [\n            sys.executable,\n            str(root / "tools" / "validate_smollm2_nvml_lifecycle_memory.py"),\n            str(smollm2_nvml_lifecycle),\n            "--expected-git-commit",\n            head,\n            "--require-thor",\n        ],\n        cwd=root,\n    )\n\n    if smollm2_parity.exists():\n'''
if fixture_anchor not in text:
    raise SystemExit('SmolLM2 fixture anchor missing')
text = text.replace(fixture_anchor, lifecycle_block, 1)

manifest_anchor = '''        "visible_device_ordinal": 0,\n        "device_selection_policy": "first_visible_device_for_all_physical_gates",\n'''
manifest_replacement = '''        "visible_device_ordinal": 0,\n        "device_selection_policy": "first_visible_device_for_all_physical_gates",\n        "run_context_id": run_context_id,\n'''
if manifest_anchor not in text:
    raise SystemExit('manifest device anchor missing')
text = text.replace(manifest_anchor, manifest_replacement, 1)

artifact_anchor = '''            "smollm2_reference_manifest": artifact_entry(smollm2_reference_manifest),\n            "smollm2_parity_record": artifact_entry(smollm2_parity),\n'''
artifact_replacement = '''            "smollm2_reference_manifest": artifact_entry(smollm2_reference_manifest),\n            "smollm2_nvml_lifecycle_memory": artifact_entry(smollm2_nvml_lifecycle),\n            "smollm2_parity_record": artifact_entry(smollm2_parity),\n'''
if artifact_anchor not in text:
    raise SystemExit('manifest artifact anchor missing')
text = text.replace(artifact_anchor, artifact_replacement, 1)

boundary_anchor = '''        "promotion_authorized": False,\n        "claim_boundary": (\n'''
boundary_replacement = '''        "promotion_authorized": False,\n        "nnml2_memory_claim_boundary": (\n            "the SmolLM2 lifecycle artifact is validated NVML current-PID process memory plus exact "\n            "NNIS-owned weight allocation evidence; it is not physical page residency and no "\n            "NVML-minus-owned-bytes difference is attributed to allocator, context, page tables, "\n            "modules, JIT, workspaces, KV, sessions, RoPE, or other runtime overhead"\n        ),\n        "claim_boundary": (\n'''
if boundary_anchor not in text:
    raise SystemExit('manifest claim boundary anchor missing')
text = text.replace(boundary_anchor, boundary_replacement, 1)

print_anchor = '''    print(f"smollm2_parity_record={smollm2_parity}")\n    print(f"smollm2_logit_report={smollm2_logit_report}")\n'''
print_replacement = '''    print(f"smollm2_nvml_lifecycle_memory={smollm2_nvml_lifecycle}")\n    print(f"smollm2_parity_record={smollm2_parity}")\n    print(f"smollm2_logit_report={smollm2_logit_report}")\n'''
if print_anchor not in text:
    raise SystemExit('manifest print anchor missing')
text = text.replace(print_anchor, print_replacement, 1)

selftest_anchor = '''def self_test() -> None:\n    smollm2 = validate_python_probe(\n'''
selftest_replacement = '''def self_test() -> None:\n    if require_run_context_id("  physical-test-run  ") != "physical-test-run":\n        raise AssertionError("run-context normalization failed")\n    try:\n        require_run_context_id("  ")\n    except QualificationError:\n        pass\n    else:\n        raise AssertionError("empty physical run-context unexpectedly passed")\n\n    smollm2 = validate_python_probe(\n'''
if selftest_anchor not in text:
    raise SystemExit('self-test anchor missing')
text = text.replace(selftest_anchor, selftest_replacement, 1)

path.write_text(text)
