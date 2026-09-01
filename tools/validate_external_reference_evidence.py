#!/usr/bin/env python3
"""Fail-closed validation for versioned NNIS external-reference evidence."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any

DEFAULT_EVIDENCE = Path("evidence/r1_tensorrt_edge_llm_v0_10_0_smollm2_thor.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_positive(value: Any, name: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    value = float(value)
    require(
        math.isfinite(value) and value > 0.0,
        f"{name} must be finite and positive",
    )
    return value


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-12)


def validate(data: dict[str, Any]) -> None:
    require(
        data.get("schema_version") == 1,
        "unsupported external reference evidence schema",
    )
    require(
        data.get("kind") == "nnis_external_reference_evidence",
        "unexpected evidence kind",
    )

    reference = data.get("reference_runtime")
    require(isinstance(reference, dict), "missing reference_runtime")
    require(bool(reference.get("name")), "reference runtime name is required")
    require(bool(reference.get("release")), "reference runtime release is required")
    require(
        isinstance(reference.get("source_commit"), str)
        and GIT_SHA_RE.fullmatch(reference["source_commit"]) is not None,
        "reference source_commit must be a full Git SHA",
    )
    finite_positive(reference.get("engine_bytes"), "reference_runtime.engine_bytes")
    require(
        isinstance(reference.get("runtime_checkpoint_bindings"), int)
        and reference["runtime_checkpoint_bindings"] > 0,
        "runtime checkpoint binding count must be positive",
    )

    model = data.get("model_identity")
    require(isinstance(model, dict), "missing model_identity")
    require(bool(model.get("source_repo")), "source repository is required")
    require(
        isinstance(model.get("source_revision"), str)
        and GIT_SHA_RE.fullmatch(model["source_revision"]) is not None,
        "source revision must be a full Git SHA",
    )
    for field in ("source_model_sha256", "tokenizer_sha256"):
        require(
            isinstance(model.get(field), str)
            and SHA256_RE.fullmatch(model[field]) is not None,
            f"{field} must be a lowercase SHA-256",
        )
    require(
        model.get("tokenizer_byte_identical_to_nnis_fixture") is True,
        "tokenizer identity gate did not pass",
    )

    workload = data.get("workload")
    require(isinstance(workload, dict), "missing workload")
    require(workload.get("batch_size") == 1, "R1 evidence must remain batch size 1")
    prompt_ids = workload.get("prompt_ids")
    require(isinstance(prompt_ids, list) and prompt_ids, "prompt_ids must be non-empty")
    require(
        all(isinstance(token, int) and token >= 0 for token in prompt_ids),
        "prompt_ids must be non-negative integers",
    )
    decode_steps = workload.get("decode_steps")
    require(isinstance(decode_steps, int) and decode_steps > 0, "decode_steps must be positive")
    require(
        workload.get("apply_chat_template") is False,
        "R1 workload must not apply a chat template",
    )
    require(
        workload.get("add_generation_prompt") is False,
        "R1 workload must not add a generation prompt",
    )
    require(
        workload.get("context_cache_lookup_policy") == "bypass",
        "R1 workload must bypass context reuse",
    )

    semantic = data.get("semantic_gate")
    require(isinstance(semantic, dict), "missing semantic_gate")
    require(semantic.get("status") == "passed", "semantic gate is not passed")
    reference_ids = semantic.get("reference_generated_ids")
    nnis_ids = semantic.get("qualified_nnis_generated_ids")
    require(
        isinstance(reference_ids, list) and isinstance(nnis_ids, list),
        "generated token trajectories are required",
    )
    require(
        len(reference_ids) == decode_steps,
        "reference generated-token count does not match decode_steps",
    )
    require(
        len(nnis_ids) == decode_steps,
        "NNIS generated-token count does not match decode_steps",
    )
    require(
        all(isinstance(token, int) for token in reference_ids + nnis_ids),
        "generated token ids must be integers",
    )
    actual_equal = reference_ids == nnis_ids
    require(
        semantic.get("exact_greedy_trajectory_equal") is actual_equal,
        "declared greedy trajectory equality does not match the recorded ids",
    )
    require(actual_equal, "R1 semantic trajectory gate is not exact")
    require(
        isinstance(semantic.get("nnis_reports_checked"), int)
        and semantic["nnis_reports_checked"] > 0,
        "semantic evidence must identify at least one qualified NNIS report",
    )

    precision = data.get("precision_and_comparability")
    require(isinstance(precision, dict), "missing precision_and_comparability")
    edge_dtype = precision.get("edge_llm_runtime_binding_dtype")
    nnis_dtype = precision.get("qualified_nnis_logical_execution_weight_dtype")
    require(
        isinstance(edge_dtype, str) and edge_dtype,
        "Edge-LLM runtime dtype is required",
    )
    require(
        isinstance(nnis_dtype, str) and nnis_dtype,
        "NNIS logical execution dtype is required",
    )
    if edge_dtype.lower() != nnis_dtype.lower():
        require(
            precision.get("cross_runtime_speed_comparison_allowed") is False,
            "cross-runtime speed comparison must fail closed when precision differs",
        )
    require(
        precision.get("cross_runtime_memory_comparison_allowed") is False,
        "current R1 memory metrics are not cross-runtime comparable",
    )

    campaign = data.get("reference_performance_campaign")
    require(isinstance(campaign, dict), "missing reference_performance_campaign")
    processes = campaign.get("processes")
    require(
        isinstance(processes, int) and processes > 0,
        "campaign process count must be positive",
    )
    require(
        isinstance(campaign.get("warmups_per_process"), int)
        and campaign["warmups_per_process"] >= 0,
        "warmup count must be non-negative",
    )
    require(
        campaign.get("all_semantic_outputs_equal") is True,
        "performance campaign semantic outputs drifted",
    )

    semantics = campaign.get("metric_semantics")
    require(isinstance(semantics, dict), "metric_semantics are required")
    for key in (
        "generation_tokens_per_second",
        "generation_stage_total_gpu_ms",
        "prefill_ms",
        "peak_unified_memory_bytes",
    ):
        require(
            isinstance(semantics.get(key), str) and semantics[key].strip(),
            f"missing metric semantics for {key}",
        )

    rows = campaign.get("raw_runs")
    require(
        isinstance(rows, list) and len(rows) == processes,
        "raw run count does not match processes",
    )
    require(
        [row.get("run") for row in rows] == list(range(1, processes + 1)),
        "raw run ordinals are incomplete or reordered",
    )

    metrics = (
        "prefill_ms",
        "generation_tokens_per_second_nvidia_definition",
        "generation_average_time_per_token_ms_nvidia_definition",
        "generation_stage_total_gpu_ms",
        "generation_stage_median_step_ms",
        "peak_unified_memory_bytes",
    )
    values: dict[str, list[float]] = {key: [] for key in metrics}
    for row in rows:
        require(isinstance(row, dict), "raw run must be an object")
        for key in metrics:
            values[key].append(finite_positive(row.get(key), f"raw_runs.{key}"))

        total_gpu_ms = float(row["generation_stage_total_gpu_ms"])
        expected_tps = decode_steps / (total_gpu_ms / 1000.0)
        expected_average_ms = total_gpu_ms / decode_steps
        require(
            close(
                float(row["generation_tokens_per_second_nvidia_definition"]),
                expected_tps,
            ),
            "NVIDIA generation throughput does not match its recorded metric definition",
        )
        require(
            close(
                float(row["generation_average_time_per_token_ms_nvidia_definition"]),
                expected_average_ms,
            ),
            "NVIDIA average generation time/token does not match its recorded metric definition",
        )

    medians = campaign.get("median")
    require(isinstance(medians, dict), "campaign median object is required")
    for key in metrics:
        expected = statistics.median(values[key])
        actual = finite_positive(medians.get(key), f"median.{key}")
        require(close(actual, expected), f"median.{key} does not match raw runs")

    ranges = campaign.get("range")
    require(isinstance(ranges, dict), "campaign range object is required")
    for key, recorded in ranges.items():
        require(key in values, f"range contains unknown metric {key}")
        require(
            isinstance(recorded, list) and len(recorded) == 2,
            f"range.{key} must contain min/max",
        )
        require(
            close(float(recorded[0]), min(values[key])),
            f"range.{key} minimum does not match raw runs",
        )
        require(
            close(float(recorded[1]), max(values[key])),
            f"range.{key} maximum does not match raw runs",
        )


def expect_rejected(data: dict[str, Any], expected_fragment: str) -> None:
    try:
        validate(data)
    except ValueError as error:
        require(
            expected_fragment in str(error),
            f"unexpected rejection: {error}; expected fragment {expected_fragment!r}",
        )
        return
    raise ValueError(f"negative self-test was accepted: {expected_fragment}")


def run_self_tests(data: dict[str, Any]) -> None:
    bad_schema = deepcopy(data)
    bad_schema["schema_version"] = 2
    expect_rejected(bad_schema, "unsupported external reference evidence schema")

    trajectory_drift = deepcopy(data)
    trajectory_drift["semantic_gate"]["reference_generated_ids"][0] += 1
    expect_rejected(trajectory_drift, "declared greedy trajectory equality")

    unsafe_speed_claim = deepcopy(data)
    unsafe_speed_claim["precision_and_comparability"][
        "cross_runtime_speed_comparison_allowed"
    ] = True
    expect_rejected(unsafe_speed_claim, "speed comparison must fail closed")

    aggregate_drift = deepcopy(data)
    aggregate_drift["reference_performance_campaign"]["median"]["prefill_ms"] += 1.0
    expect_rejected(aggregate_drift, "median.prefill_ms does not match raw runs")

    metric_semantics_drift = deepcopy(data)
    metric_semantics_drift["reference_performance_campaign"]["raw_runs"][0][
        "generation_tokens_per_second_nvidia_definition"
    ] += 1.0
    expect_rejected(
        metric_semantics_drift,
        "NVIDIA generation throughput does not match its recorded metric definition",
    )


def main() -> int:
    args = sys.argv[1:]
    self_test = False
    if "--self-test" in args:
        self_test = True
        args.remove("--self-test")
    if len(args) > 1:
        print("usage: validate_external_reference_evidence.py [--self-test] [path]", file=sys.stderr)
        return 2

    path = Path(args[0]) if args else DEFAULT_EVIDENCE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(data, dict), "evidence root must be an object")
        validate(data)
        if self_test:
            run_self_tests(data)
    except Exception as error:
        print(f"external reference evidence rejected: {error}", file=sys.stderr)
        return 1

    suffix = " with negative self-tests" if self_test else ""
    print(f"external reference evidence accepted{suffix}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
