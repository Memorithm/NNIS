#!/usr/bin/env python3
"""Fail-closed validator for SmolLM2 NVML lifecycle memory evidence."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

SOURCE_REPO = "HuggingFaceTB/SmolLM2-135M"
SOURCE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SOURCE_MODEL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
EVIDENCE = "nnis.smollm2-nvml-lifecycle-memory"
MEASUREMENT = (
    "nvml_process_used_gpu_memory_plus_exact_owned_weight_allocations_"
    "not_physical_page_residency"
)


class ValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty string")
    return value


def require_nonnegative_int(value: Any, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return value


def validate_weight_summary(value: Any, label: str) -> dict[str, Any]:
    summary = require_object(value, label)
    require(summary.get("schema_version") == 1, f"{label}.schema_version must be 1")
    owned = require_nonnegative_int(
        summary.get("owned_device_allocation_bytes"),
        f"{label}.owned_device_allocation_bytes",
    )
    require(owned > 0, f"{label}.owned_device_allocation_bytes must be > 0")
    segments = summary.get("segments")
    require(isinstance(segments, list) and segments, f"{label}.segments must be a non-empty list")
    require(
        summary.get("unique_device_allocations") == len(segments),
        f"{label}.unique_device_allocations must equal len(segments)",
    )
    segment_bytes = 0
    for index, segment_value in enumerate(segments):
        segment = require_object(segment_value, f"{label}.segments[{index}]")
        require(
            segment.get("allocation_index") == index,
            f"{label}.segments[{index}].allocation_index must be {index}",
        )
        segment_bytes += require_nonnegative_int(
            segment.get("bytes"), f"{label}.segments[{index}].bytes"
        )
    require(segment_bytes == owned, f"{label} segment byte sum must equal owned bytes")
    return summary


def validate_process_observation(value: Any, label: str) -> dict[str, Any]:
    observation = require_object(value, label)
    require(observation.get("schema_version") == 1, f"{label}.schema_version must be 1")
    require_nonnegative_int(observation.get("pid"), f"{label}.pid")
    require_nonnegative_int(observation.get("device_ordinal"), f"{label}.device_ordinal")
    require_nonempty_string(observation.get("device_uuid"), f"{label}.device_uuid")
    require_nonnegative_int(
        observation.get("used_gpu_memory_bytes"), f"{label}.used_gpu_memory_bytes"
    )
    return observation


def validate_report(
    report: Any,
    *,
    expected_git_commit: str | None = None,
    require_thor: bool = False,
) -> None:
    root = require_object(report, "report")
    require(root.get("schema_version") == 1, "report.schema_version must be 1")
    require(root.get("evidence") == EVIDENCE, "unexpected evidence contract")
    require(root.get("measurement") == MEASUREMENT, "unexpected measurement contract")
    require(root.get("source_repo") == SOURCE_REPO, "unexpected source_repo")
    require(root.get("source_revision") == SOURCE_REVISION, "unexpected source_revision")
    require(root.get("source_model_sha256") == SOURCE_MODEL_SHA256, "unexpected source_model_sha256")
    require(root.get("source_weight_dtype") == "bfloat16", "source_weight_dtype must be bfloat16")
    require(
        root.get("source_graph_execution_weight_dtype") == "f32",
        "source_graph_execution_weight_dtype must be f32",
    )

    metadata = require_object(root.get("metadata"), "metadata")
    git_commit = require_nonempty_string(metadata.get("git_commit"), "metadata.git_commit")
    if expected_git_commit is not None:
        require(git_commit == expected_git_commit, "metadata.git_commit does not match expected head")
    require_nonnegative_int(metadata.get("gpu_ordinal"), "metadata.gpu_ordinal")
    require_nonempty_string(metadata.get("gpu_name"), "metadata.gpu_name")
    fingerprint = require_object(
        metadata.get("environment_fingerprint"), "metadata.environment_fingerprint"
    )
    require(
        fingerprint.get("schema_version") == 1,
        "metadata.environment_fingerprint.schema_version must be 1",
    )
    if require_thor:
        require(metadata.get("git_dirty") is False, "Thor evidence requires git_dirty=false")
        require(metadata.get("host_arch") == "aarch64", "Thor evidence requires host_arch=aarch64")
        require_nonempty_string(
            fingerprint.get("run_context_id"),
            "metadata.environment_fingerprint.run_context_id",
        )
        platform_model = require_nonempty_string(
            fingerprint.get("platform_model"),
            "metadata.environment_fingerprint.platform_model",
        ).lower()
        require(
            "jetson" in platform_model and "thor" in platform_model,
            "Thor evidence requires a Jetson Thor platform model",
        )
        require_nonempty_string(
            fingerprint.get("jetson_power_mode"),
            "metadata.environment_fingerprint.jetson_power_mode",
        )
        require_nonempty_string(
            fingerprint.get("jetson_clock_state"),
            "metadata.environment_fingerprint.jetson_clock_state",
        )

    post = validate_process_observation(
        root.get("post_context_nvml_process_memory"),
        "post_context_nvml_process_memory",
    )
    source_stage = require_object(root.get("source_f32_weights"), "source_f32_weights")
    resident_stage = require_object(root.get("resident_f16_weights"), "resident_f16_weights")
    require(
        source_stage.get("stage") == "source_f32_weights_loaded",
        "unexpected source_f32_weights.stage",
    )
    require(
        resident_stage.get("stage") == "resident_f16_weights_materialized_source_f32_dropped",
        "unexpected resident_f16_weights.stage",
    )
    source_nvml = validate_process_observation(
        source_stage.get("nvml_process_memory"), "source_f32_weights.nvml_process_memory"
    )
    resident_nvml = validate_process_observation(
        resident_stage.get("nvml_process_memory"), "resident_f16_weights.nvml_process_memory"
    )

    identities = {
        (observation["pid"], observation["device_ordinal"], observation["device_uuid"])
        for observation in (post, source_nvml, resident_nvml)
    }
    require(len(identities) == 1, "all lifecycle NVML observations must share PID/device/UUID")
    require(
        post["device_ordinal"] == metadata.get("gpu_ordinal"),
        "NVML device ordinal must match benchmark metadata",
    )

    source_summary = validate_weight_summary(
        source_stage.get("exact_owned_weight_allocations"),
        "source_f32_weights.exact_owned_weight_allocations",
    )
    resident_summary = validate_weight_summary(
        resident_stage.get("exact_owned_weight_allocations"),
        "resident_f16_weights.exact_owned_weight_allocations",
    )
    materialization = require_object(root.get("materialization_memory"), "materialization_memory")
    require(
        materialization.get("schema_version") == 1,
        "materialization_memory.schema_version must be 1",
    )
    require(
        materialization.get("execution_plan") == root.get("execution_plan"),
        "materialization execution plan must match report execution_plan",
    )
    require(
        materialization.get("source_weight_allocations") == source_summary,
        "source allocation summary must reconcile with materialization evidence",
    )
    require(
        materialization.get("steady_state_f16_weight_allocations") == resident_summary,
        "resident F16 allocation summary must reconcile with materialization evidence",
    )
    peak = require_nonnegative_int(
        materialization.get("peak_scoped_owned_allocation_bytes"),
        "materialization_memory.peak_scoped_owned_allocation_bytes",
    )
    final = require_nonnegative_int(
        materialization.get("final_scoped_owned_allocation_bytes"),
        "materialization_memory.final_scoped_owned_allocation_bytes",
    )
    require(peak >= final, "materialization peak scoped bytes must be >= final scoped bytes")


def synthetic_summary(dtype: str, byte_count: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "logical_tensor_references": 1,
        "logical_element_references": byte_count // 2,
        "unique_device_allocations": 1,
        "unique_device_elements": byte_count // 2,
        "owned_device_allocation_bytes": byte_count,
        "segments": [
            {
                "allocation_index": 0,
                "dtype": dtype,
                "elements": byte_count // 2,
                "bytes": byte_count,
                "logical_names": ["synthetic"],
            }
        ],
    }


def synthetic_report() -> dict[str, Any]:
    source = synthetic_summary("f32", 400)
    resident = synthetic_summary("f16", 200)
    nvml = {
        "schema_version": 1,
        "pid": 4242,
        "device_ordinal": 0,
        "device_uuid": "GPU-synthetic",
        "used_gpu_memory_bytes": 1024,
    }
    execution_plan = {"schema_version": 1, "projection_layout": "nk_transposed_candidate"}
    return {
        "schema_version": 1,
        "evidence": EVIDENCE,
        "measurement": MEASUREMENT,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "source_weight_dtype": "bfloat16",
        "source_graph_execution_weight_dtype": "f32",
        "metadata": {
            "git_commit": "deadbeef",
            "git_dirty": False,
            "host_arch": "aarch64",
            "gpu_ordinal": 0,
            "gpu_name": "NVIDIA Jetson AGX Thor",
            "environment_fingerprint": {
                "schema_version": 1,
                "run_context_id": "synthetic-thor-run",
                "platform_model": "NVIDIA Jetson AGX Thor",
                "jetson_power_mode": "MAXN",
                "jetson_clock_state": "locked",
            },
        },
        "model_config": {},
        "execution_plan": execution_plan,
        "post_context_nvml_process_memory": copy.deepcopy(nvml),
        "source_f32_weights": {
            "stage": "source_f32_weights_loaded",
            "nvml_process_memory": copy.deepcopy(nvml),
            "exact_owned_weight_allocations": source,
        },
        "resident_f16_weights": {
            "stage": "resident_f16_weights_materialized_source_f32_dropped",
            "nvml_process_memory": copy.deepcopy(nvml),
            "exact_owned_weight_allocations": resident,
        },
        "materialization_memory": {
            "schema_version": 1,
            "execution_plan": execution_plan,
            "source_weight_allocations": copy.deepcopy(source),
            "steady_state_f16_weight_allocations": copy.deepcopy(resident),
            "peak_live_f16_allocation_bytes": 300,
            "peak_live_temporary_f16_allocation_bytes": 100,
            "peak_scoped_owned_allocation_bytes": 700,
            "final_scoped_owned_allocation_bytes": 600,
            "events": [],
        },
        "limitations": [],
    }


def expect_failure(
    report: dict[str, Any],
    *,
    expected_git_commit: str = "deadbeef",
    require_thor: bool = True,
) -> None:
    try:
        validate_report(
            report,
            expected_git_commit=expected_git_commit,
            require_thor=require_thor,
        )
    except ValidationError:
        return
    raise AssertionError("corrupted synthetic report unexpectedly validated")


def self_test() -> None:
    baseline = synthetic_report()
    validate_report(baseline, expected_git_commit="deadbeef", require_thor=True)

    bad = copy.deepcopy(baseline)
    bad["metadata"]["git_commit"] = "wrong"
    expect_failure(bad)

    bad = copy.deepcopy(baseline)
    bad["source_f32_weights"]["nvml_process_memory"]["pid"] += 1
    expect_failure(bad)

    bad = copy.deepcopy(baseline)
    bad["resident_f16_weights"]["nvml_process_memory"]["device_uuid"] = "GPU-other"
    expect_failure(bad)

    bad = copy.deepcopy(baseline)
    bad["source_f32_weights"]["exact_owned_weight_allocations"]["owned_device_allocation_bytes"] += 1
    expect_failure(bad)

    bad = copy.deepcopy(baseline)
    bad["metadata"]["environment_fingerprint"]["run_context_id"] = None
    expect_failure(bad)

    bad = copy.deepcopy(baseline)
    bad["metadata"]["environment_fingerprint"]["platform_model"] = "generic CUDA host"
    expect_failure(bad)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate versioned SmolLM2 NVML lifecycle memory evidence."
    )
    parser.add_argument("report", nargs="?", type=Path, help="lifecycle JSON artifact")
    parser.add_argument(
        "--expected-git-commit",
        help="require metadata.git_commit to equal this exact NNIS head",
    )
    parser.add_argument(
        "--require-thor",
        action="store_true",
        help="require clean aarch64 Jetson Thor run-context/power/clock evidence",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("smollm2 NVML lifecycle validator self-test: ok")
        return 0
    if args.report is None:
        parser.error("report is required unless --self-test is used")

    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        validate_report(
            report,
            expected_git_commit=args.expected_git_commit,
            require_thor=args.require_thor,
        )
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        print(f"invalid SmolLM2 NVML lifecycle evidence: {error}", file=sys.stderr)
        return 2

    print("SmolLM2 NVML lifecycle evidence: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
