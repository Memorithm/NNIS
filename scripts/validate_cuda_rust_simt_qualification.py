#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

MANIFEST = Path("docs/cuda-rust-simt-qualification.json")
HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def fail(message: str) -> None:
    print(f"qualification manifest invalid: {message}", file=sys.stderr)
    raise SystemExit(1)


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
        artifact.get("sha256"),
        device.get("identity"),
        device.get("compute_capability"),
        device.get("driver_version"),
        device.get("cuda_version"),
        oracle.get("correction_passed"),
        oracle.get("negative_tests_passed"),
        run.get("id"),
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
    if not HEX64.fullmatch(str(artifact.get("sha256", ""))):
        fail("PTX sha256 must be 64 hex characters")
    if not HEX64.fullmatch(str(run.get("evidence_bundle_sha256", ""))):
        fail("evidence bundle sha256 must be 64 hex characters")
    for name in ("version_or_channel",):
        if not str(toolchain.get(name, "")).strip():
            fail(f"missing toolchain {name}")
    for name in ("identity", "compute_capability", "driver_version", "cuda_version"):
        if not str(device.get(name, "")).strip():
            fail(f"missing device {name}")
    if not str(run.get("id", "")).strip():
        fail("missing run id")
    if oracle.get("correction_passed") is not True:
        fail("correction oracle did not pass")
    if oracle.get("negative_tests_passed") is not True:
        fail("negative tests did not pass")

    print("CUDA Rust SIMT evidence is complete, non-production, and content-addressed")


if __name__ == "__main__":
    main()
