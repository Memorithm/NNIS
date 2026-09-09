#!/usr/bin/env python3
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/cuda-rust-simt-qualification.json"
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def fail(message: str) -> None:
    print(f"qualification manifest invalid: {message}", file=sys.stderr)
    raise SystemExit(1)


def safe_repo_file(raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        fail(f"missing {label} path")
    candidate = (ROOT / raw).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        fail(f"{label} path escapes repository root")
    if not candidate.is_file():
        fail(f"{label} file is not accessible: {raw}")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_exact_mapping(actual: object, expected: dict, label: str) -> None:
    if actual != expected:
        fail(f"{label} does not match manifest")


def main() -> None:
    data = json.loads(MANIFEST.read_text())
    if data.get("schema") != "nnis-cuda-rust-simt-qualification-v1":
        fail("unexpected schema")
    if data.get("frontend_contract") != "CUDA_RUST_SIMT_PTX":
        fail("wrong frontend contract")
    if data.get("production_routing_authorized") is not False:
        fail("manifest must not authorize production routing")
    if data.get("performance_claim_authorized") is not False:
        fail("correction qualification must precede performance claims")

    boundary = data.get("scientific_boundary", {})
    required_boundary = {
        "external_prior_art_is_not_memorithm_evidence": True,
        "correction_qualification_precedes_performance_measurement": True,
        "successful_manifest_validation_does_not_promote_frontend": True,
    }
    if boundary != required_boundary:
        fail("scientific boundary changed")

    requirements = set(data.get("qualification_requirements", []))
    expected = {
        "exact_toolchain_source_commit",
        "exact_ptx_sha256",
        "exact_device_and_cuda_identity",
        "independent_vector_add_correction_oracle",
        "negative_fail_closed_tests",
        "content_addressed_evidence_bundle",
        "exact_nnis_revision_binding",
    }
    if requirements != expected:
        fail("qualification requirements changed")

    toolchain = data.get("toolchain", {})
    artifact = data.get("artifact", {})
    device = data.get("device", {})
    oracle = data.get("oracle", {})
    run = data.get("run", {})

    if toolchain.get("name") != "cuda-oxide":
        fail("unexpected toolchain")
    if artifact.get("kind") != "ptx":
        fail("unexpected artifact kind")
    if oracle.get("id") != "nnis-vector-add-reference-v1":
        fail("unexpected correction oracle")

    status = data.get("status")
    fields = [
        toolchain.get("version_or_channel"),
        toolchain.get("source_commit"),
        artifact.get("path"),
        artifact.get("sha256"),
        device.get("identity"),
        device.get("compute_capability"),
        device.get("driver_version"),
        device.get("cuda_version"),
        oracle.get("correction_passed"),
        oracle.get("negative_tests_passed"),
        run.get("id"),
        run.get("nnis_commit"),
        run.get("evidence_bundle_path"),
        run.get("evidence_bundle_sha256"),
    ]

    if status == "unresolved_blocking":
        if any(value is not None for value in fields):
            fail("unresolved manifest must not contain partial evidence")
        print("CUDA Rust SIMT qualification remains fail-closed and unresolved")
        return

    if status != "qualified_nonproduction":
        fail("unsupported status")
    if not HEX40.fullmatch(str(toolchain.get("source_commit", ""))):
        fail("toolchain source_commit must be an exact 40-hex Git commit")
    if not HEX40.fullmatch(str(run.get("nnis_commit", ""))):
        fail("run nnis_commit must be an exact 40-hex Git commit")
    if not HEX64.fullmatch(str(artifact.get("sha256", ""))):
        fail("PTX sha256 must be 64 hex characters")
    if not HEX64.fullmatch(str(run.get("evidence_bundle_sha256", ""))):
        fail("evidence bundle sha256 must be 64 hex characters")
    if not str(toolchain.get("version_or_channel", "")).strip():
        fail("missing toolchain version_or_channel")
    for name in ("identity", "compute_capability", "driver_version", "cuda_version"):
        if not str(device.get(name, "")).strip():
            fail(f"missing device {name}")
    if not str(run.get("id", "")).strip():
        fail("missing run id")
    if oracle.get("correction_passed") is not True:
        fail("correction oracle did not pass")
    if oracle.get("negative_tests_passed") is not True:
        fail("negative tests did not pass")

    expected_revision = os.environ.get("NNIS_EXPECTED_REVISION")
    if expected_revision:
        if not HEX40.fullmatch(expected_revision):
            fail("NNIS_EXPECTED_REVISION must be an exact 40-hex commit")
        if run.get("nnis_commit") != expected_revision:
            fail("evidence is not bound to the exact NNIS revision under validation")

    ptx_path = safe_repo_file(artifact.get("path"), "PTX artifact")
    actual_ptx_sha = sha256_file(ptx_path)
    if actual_ptx_sha.lower() != str(artifact.get("sha256")).lower():
        fail("PTX sha256 does not match accessible artifact")

    bundle_path = safe_repo_file(run.get("evidence_bundle_path"), "evidence bundle")
    actual_bundle_sha = sha256_file(bundle_path)
    if actual_bundle_sha.lower() != str(run.get("evidence_bundle_sha256")).lower():
        fail("evidence bundle sha256 does not match accessible bundle")

    bundle = json.loads(bundle_path.read_text())
    if bundle.get("schema") != "nnis-cuda-rust-simt-evidence-v1":
        fail("unexpected evidence bundle schema")
    if bundle.get("run_id") != run.get("id"):
        fail("evidence bundle run id does not match manifest")
    if bundle.get("nnis_commit") != run.get("nnis_commit"):
        fail("evidence bundle NNIS revision does not match manifest")
    require_exact_mapping(bundle.get("toolchain"), toolchain, "evidence toolchain")
    require_exact_mapping(
        bundle.get("artifact"),
        {"kind": artifact.get("kind"), "path": artifact.get("path"), "sha256": artifact.get("sha256")},
        "evidence artifact",
    )
    require_exact_mapping(bundle.get("device"), device, "evidence device")
    require_exact_mapping(bundle.get("oracle"), oracle, "evidence oracle")

    vector_add = bundle.get("vector_add", {})
    inputs_a = vector_add.get("a")
    inputs_b = vector_add.get("b")
    observed = vector_add.get("observed")
    if not all(isinstance(values, list) for values in (inputs_a, inputs_b, observed)):
        fail("vector-add evidence must contain list-valued a, b and observed")
    if not inputs_a or len(inputs_a) != len(inputs_b) or len(inputs_a) != len(observed):
        fail("vector-add evidence lengths are invalid")
    try:
        expected_values = [float(a) + float(b) for a, b in zip(inputs_a, inputs_b)]
        observed_values = [float(value) for value in observed]
    except (TypeError, ValueError):
        fail("vector-add evidence contains non-numeric values")
    if observed_values != expected_values:
        fail("recorded vector-add output does not satisfy independent correction oracle")

    negative_tests = bundle.get("negative_tests")
    if not isinstance(negative_tests, list) or not negative_tests:
        fail("evidence bundle must contain negative-test records")
    for case in negative_tests:
        if not isinstance(case, dict) or not str(case.get("id", "")).strip() or case.get("passed") is not True:
            fail("negative-test evidence is incomplete or failed")

    print("CUDA Rust SIMT evidence is inspectable, hash-verified, revision-bound, and non-production")


if __name__ == "__main__":
    main()
