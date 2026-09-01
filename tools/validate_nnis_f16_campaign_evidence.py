#!/usr/bin/env python3
"""Fail-closed validation for the R1 NNIS F16 Thor campaign evidence."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any

DEFAULT_EVIDENCE = Path("evidence/r1_nnis_f16_smollm2_thor.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_IDS = [
    260, 3075, 338, 6650, 260, 2591, 284, 260,
    8872, 1592, 30, 198, 198, 504, 8872, 314,
    253, 8304, 282, 260, 2591, 30, 657, 314,
    253, 19284, 1248, 338, 21837, 260, 2591, 30,
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_positive(value: Any, name: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    value = float(value)
    require(math.isfinite(value) and value > 0.0, f"{name} must be finite and positive")
    return value


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-10, abs_tol=1.0e-10)


def validate(data: dict[str, Any]) -> None:
    require(data.get("schema_version") == 1, "unsupported F16 campaign evidence schema")
    require(
        data.get("kind") == "nnis_f16_reference_campaign_evidence",
        "unexpected F16 campaign evidence kind",
    )

    nnis = data.get("nnis")
    require(isinstance(nnis, dict), "missing nnis identity")
    for field in ("merge_commit", "profile_pr_head"):
        require(
            isinstance(nnis.get(field), str) and GIT_SHA_RE.fullmatch(nnis[field]) is not None,
            f"nnis.{field} must be a full Git SHA",
        )
    require(nnis.get("profile_pr") == 72, "unexpected profiler PR")
    for field in ("profile_pr_ci_run", "profile_pr_harness_run", "post_merge_ci_run"):
        require(isinstance(nnis.get(field), int) and nnis[field] > 0, f"nnis.{field} must be positive")

    model = data.get("model_identity")
    require(isinstance(model, dict), "missing model_identity")
    require(model.get("source_repo") == "HuggingFaceTB/SmolLM2-135M", "unexpected model source")
    require(
        isinstance(model.get("source_revision"), str)
        and GIT_SHA_RE.fullmatch(model["source_revision"]) is not None,
        "source_revision must be a full Git SHA",
    )
    for field in ("source_model_sha256", "converted_model_manifest_sha256"):
        require(
            isinstance(model.get(field), str) and SHA256_RE.fullmatch(model[field]) is not None,
            f"{field} must be a lowercase SHA-256",
        )
    require(model.get("source_weight_dtype") == "bfloat16", "source weight dtype drifted")
    require(model.get("persisted_execution_weight_dtype") == "f32", "persisted dtype drifted")

    workload = data.get("workload")
    require(isinstance(workload, dict), "missing workload")
    require(workload.get("batch_size") == 1, "R1 workload must remain batch size 1")
    require(workload.get("prompt_ids") == [22007, 6463, 314], "prompt ids drifted")
    require(workload.get("decode_steps") == 32, "decode steps drifted")
    require(workload.get("sampling") == "greedy", "sampling mode drifted")
    require(workload.get("expected_generated_ids") == EXPECTED_IDS, "expected greedy trajectory drifted")

    plan = data.get("numeric_plan")
    require(isinstance(plan, dict), "missing numeric_plan")
    require(plan.get("schema_version") == 1, "unsupported F16 plan schema")
    expected_plan = {
        "resident_weight_dtype": "f16",
        "activation_dtype": "f16",
        "kv_dtype": "f16",
        "projection_accumulator_dtype": "f32",
        "attention_accumulator_dtype": "f32",
        "logits_dtype": "f32",
        "explicit_opt_in": True,
        "model_format_v1_unchanged": True,
    }
    for key, expected in expected_plan.items():
        require(plan.get(key) == expected, f"numeric_plan.{key} drifted")

    contract = data.get("metric_contract")
    require(isinstance(contract, dict), "missing metric_contract")
    require(contract.get("generated_tokens") == 32, "generated-token contract drifted")
    require(contract.get("generation_forward_runs") == 31, "generation forward count drifted")
    require(contract.get("first_generated_token_from_prefill") is True, "prefill token contract drifted")
    require(contract.get("sampling_in_generation_stage_gpu_time") is False, "sampling entered timed stage")
    require(contract.get("final_generated_token_consumed_by_model") is False, "final token was consumed")
    require(contract.get("metric_semantics_aligned_to_edge_reference") is True, "metric alignment gate failed")

    thor = data.get("thor_regime")
    require(isinstance(thor, dict), "missing Thor regime")
    require(thor.get("gpu") == "NVIDIA Thor", "unexpected GPU")
    require(thor.get("compute_capability") == "11.0", "unexpected compute capability")
    require(thor.get("power_mode") == "MAXN", "Thor power mode drifted")
    require(thor.get("competing_cuda_processes_before_each_run") == 0, "competing CUDA process recorded")
    for field in ("cpu_hz", "gpu_gpc_hz", "gpu_nvd_hz", "emc_hz"):
        require(isinstance(thor.get(field), int) and thor[field] > 0, f"thor_regime.{field} must be positive")

    campaign = data.get("campaign")
    require(isinstance(campaign, dict), "missing campaign")
    require(campaign.get("processes") == 5, "campaign must contain five independent processes")
    require(campaign.get("warmups_per_process") == 2, "campaign warmup count drifted")
    require(campaign.get("all_environment_fingerprints_identical") is True, "environment fingerprint drifted")
    require(campaign.get("all_exact_greedy_32_of_32") is True, "campaign semantic gate failed")

    rows = campaign.get("raw_runs")
    require(isinstance(rows, list) and len(rows) == 5, "raw run count must be five")
    require([row.get("run") for row in rows] == [1, 2, 3, 4, 5], "raw run ordinals drifted")

    metrics = (
        "prefill_gpu_ms",
        "generation_stage_total_gpu_ms",
        "generation_tokens_per_second_edge_definition",
        "generation_forward_median_ms",
        "request_wall_ms",
    )
    values: dict[str, list[float]] = {key: [] for key in metrics}
    for row in rows:
        require(isinstance(row, dict), "raw run must be an object")
        for key in metrics:
            values[key].append(finite_positive(row.get(key), f"raw_runs.{key}"))
        expected_tps = 32.0 / (float(row["generation_stage_total_gpu_ms"]) / 1000.0)
        require(
            close(float(row["generation_tokens_per_second_edge_definition"]), expected_tps),
            "NNIS generation throughput does not match the Edge-aligned metric definition",
        )

    medians = campaign.get("median")
    ranges = campaign.get("range")
    require(isinstance(medians, dict), "missing campaign median")
    require(isinstance(ranges, dict), "missing campaign range")
    for key in metrics:
        require(
            close(finite_positive(medians.get(key), f"median.{key}"), statistics.median(values[key])),
            f"median.{key} does not match raw runs",
        )
        recorded_range = ranges.get(key)
        require(isinstance(recorded_range, list) and len(recorded_range) == 2, f"range.{key} must be min/max")
        require(close(float(recorded_range[0]), min(values[key])), f"range.{key} minimum drifted")
        require(close(float(recorded_range[1]), max(values[key])), f"range.{key} maximum drifted")

    reference = data.get("trusted_external_reference")
    require(isinstance(reference, dict), "missing trusted_external_reference")
    require(reference.get("runtime") == "TensorRT Edge-LLM", "unexpected external reference")
    require(reference.get("release") == "v0.10.0", "external reference release drifted")
    require(
        isinstance(reference.get("source_commit"), str)
        and GIT_SHA_RE.fullmatch(reference["source_commit"]) is not None,
        "external reference source commit must be full SHA",
    )
    for field in ("same_model_source_sha256", "same_tokenizer_bytes", "same_prompt_ids", "same_decode_steps", "exact_greedy_trajectory_equal"):
        require(reference.get(field) is True, f"trusted_external_reference.{field} gate failed")
    require(reference.get("resident_weight_dtype") == "f16", "external weight dtype drifted")
    require(reference.get("activation_io_dtype") == "f16", "external activation dtype drifted")
    require(reference.get("kv_dtype") == "f16", "external KV dtype drifted")
    require(reference.get("logits_dtype") == "f32", "external logits dtype drifted")
    require(reference.get("generation_forward_runs") == 31, "external generation run count drifted")
    require(reference.get("generated_tokens") == 32, "external generated-token count drifted")
    finite_positive(reference.get("generation_tokens_per_second_median"), "external generation median TPS")
    finite_positive(reference.get("generation_stage_total_gpu_ms_median"), "external generation GPU median")

    audit = data.get("external_numeric_semantics_audit")
    require(isinstance(audit, dict), "missing external numeric-semantics audit")
    require(
        audit.get("edge_projection_accumulator_contract")
        == "not explicitly fixed to f32 by TensorRT Edge-LLM v0.10.0 source",
        "external projection accumulator audit drifted",
    )
    require(
        audit.get("edge_attention_accumulator_contract")
        == "not exposed as identical to the NNIS explicit f32 attention accumulator contract",
        "external attention accumulator audit drifted",
    )

    comparability = data.get("comparability")
    require(isinstance(comparability, dict), "missing comparability")
    for field in (
        "storage_precision_alignment",
        "semantic_trajectory_alignment",
        "generation_metric_accounting_alignment",
        "repeated_physical_campaign_complete",
    ):
        require(comparability.get(field) is True, f"comparability.{field} gate failed")
    require(
        comparability.get("cross_runtime_speed_comparison_allowed") is False,
        "cross-runtime speed comparison must remain fail-closed while accumulator semantics differ",
    )
    require(
        comparability.get("cross_runtime_memory_comparison_allowed") is False,
        "cross-runtime memory comparison must remain fail-closed",
    )
    require(comparability.get("quotient_claim_forbidden") is True, "external quotient claim must remain forbidden")


def expect_rejected(data: dict[str, Any], expected_fragment: str) -> None:
    try:
        validate(data)
    except ValueError as error:
        require(expected_fragment in str(error), f"unexpected rejection: {error}")
        return
    raise ValueError(f"negative self-test was accepted: {expected_fragment}")


def run_self_tests(data: dict[str, Any]) -> None:
    trajectory_drift = deepcopy(data)
    trajectory_drift["workload"]["expected_generated_ids"][0] += 1
    expect_rejected(trajectory_drift, "expected greedy trajectory drifted")

    metric_drift = deepcopy(data)
    metric_drift["campaign"]["raw_runs"][0]["generation_tokens_per_second_edge_definition"] += 1.0
    expect_rejected(metric_drift, "generation throughput does not match")

    aggregate_drift = deepcopy(data)
    aggregate_drift["campaign"]["median"]["generation_stage_total_gpu_ms"] += 1.0
    expect_rejected(aggregate_drift, "median.generation_stage_total_gpu_ms")

    unsafe_speed_claim = deepcopy(data)
    unsafe_speed_claim["comparability"]["cross_runtime_speed_comparison_allowed"] = True
    expect_rejected(unsafe_speed_claim, "speed comparison must remain fail-closed")

    accumulator_claim_drift = deepcopy(data)
    accumulator_claim_drift["external_numeric_semantics_audit"]["edge_projection_accumulator_contract"] = "f32"
    expect_rejected(accumulator_claim_drift, "projection accumulator audit drifted")


def main() -> int:
    args = sys.argv[1:]
    self_test = False
    if "--self-test" in args:
        self_test = True
        args.remove("--self-test")
    if len(args) > 1:
        print("usage: validate_nnis_f16_campaign_evidence.py [--self-test] [path]", file=sys.stderr)
        return 2

    path = Path(args[0]) if args else DEFAULT_EVIDENCE
    try:
        data = json.loads(path.read_text())
        require(isinstance(data, dict), "evidence root must be an object")
        validate(data)
        if self_test:
            run_self_tests(data)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"F16 campaign evidence validation failed: {error}", file=sys.stderr)
        return 1

    print(f"F16 campaign evidence valid: {path}")
    if self_test:
        print("F16 campaign negative self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
