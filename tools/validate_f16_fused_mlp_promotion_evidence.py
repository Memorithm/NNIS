#!/usr/bin/env python3
"""Validate the scoped F16 fused-MLP SmolLM2/Thor promotion evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

RUNTIME_COMMIT = "4101b8924f1e5400a7871259b9c1b732ae3c77bb"
TOOLING_COMMIT = "8d0720244a10624d3a75709700f9686e408ae2b8"
SOURCE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SOURCE_MODEL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
MIN_IMPROVEMENT = 0.03


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)


def validate_run(run: dict[str, Any], rounds: int) -> float:
    pairs = run.get("pairs")
    require(isinstance(pairs, list) and len(pairs) == rounds, "invalid pair count")
    improvements: list[float] = []
    parents: list[float] = []
    candidates: list[float] = []
    for index, pair in enumerate(pairs):
        require(pair.get("round") == index, "round index drifted")
        expected_order = "fused_then_fused_mlp" if index % 2 == 0 else "fused_mlp_then_fused"
        require(pair.get("order") == expected_order, "ABBA order drifted")
        parent = pair.get("fused_median_ms")
        candidate = pair.get("fused_mlp_median_ms")
        recorded = pair.get("paired_relative_improvement")
        require(isinstance(parent, (int, float)) and parent > 0.0, "invalid parent median")
        require(isinstance(candidate, (int, float)) and candidate > 0.0, "invalid candidate median")
        require(isinstance(recorded, (int, float)), "invalid paired improvement")
        recomputed = (float(parent) - float(candidate)) / float(parent)
        require(close(float(recorded), recomputed), "paired improvement does not match medians")
        require(recomputed > 0.0, "candidate did not win every pair")
        parents.append(float(parent))
        candidates.append(float(candidate))
        improvements.append(recomputed)

    require(run.get("candidate_round_wins") == rounds, "candidate win count drifted")
    require(run.get("parent_round_wins") == 0, "parent win count drifted")
    require(run.get("ties") == 0, "tie count drifted")
    checks = {
        "median_fused_generation_gpu_ms": statistics.median(parents),
        "median_fused_mlp_generation_gpu_ms": statistics.median(candidates),
        "median_paired_relative_improvement": statistics.median(improvements),
        "mean_paired_relative_improvement": statistics.mean(improvements),
        "min_paired_relative_improvement": min(improvements),
        "max_paired_relative_improvement": max(improvements),
    }
    for key, expected in checks.items():
        actual = run.get(key)
        require(isinstance(actual, (int, float)) and close(float(actual), expected), f"{key} drifted")
    median_improvement = statistics.median(improvements)
    require(median_improvement >= MIN_IMPROVEMENT, "run does not clear promotion floor")
    return median_improvement


def validate(data: dict[str, Any]) -> None:
    require(data.get("schema_version") == 1, "unexpected schema")
    require(data.get("kind") == "nnis_f16_smollm2_thor_min_latency_promotion_evidence", "unexpected kind")
    require(data.get("runtime_commit") == RUNTIME_COMMIT, "runtime commit drifted")
    require(data.get("tooling_commit") == TOOLING_COMMIT, "tooling commit drifted")
    require(data.get("candidate_plan") == "nk_transposed_fused_mlp_candidate", "candidate plan drifted")
    require(data.get("parent_plan") == "nk_transposed_fused_groups_candidate", "parent plan drifted")

    hardware = data.get("hardware")
    require(isinstance(hardware, dict), "missing hardware")
    require(hardware.get("gpu_name") == "NVIDIA Thor", "GPU identity drifted")
    require(hardware.get("compute_capability") == "11.0", "compute capability drifted")
    require(hardware.get("multiprocessor_count") == 20, "SM count drifted")

    model = data.get("model")
    require(isinstance(model, dict), "missing model identity")
    require(model.get("source_repo") == "HuggingFaceTB/SmolLM2-135M", "source repo drifted")
    require(model.get("source_revision") == SOURCE_REVISION, "source revision drifted")
    require(model.get("source_model_sha256") == SOURCE_MODEL_SHA256, "model hash drifted")
    require((model.get("hidden_size"), model.get("intermediate_size"), model.get("layers")) == (576, 1536, 30), "model geometry drifted")

    metric = data.get("metric")
    require(isinstance(metric, dict), "missing metric contract")
    require(metric.get("name") == "generation_stage_gpu", "metric identity drifted")
    require(metric.get("generated_tokens") == 32, "generated-token contract drifted")
    require(metric.get("generation_forward_runs") == 31, "forward-count contract drifted")
    require(metric.get("warmups_per_report") == 2, "warmup contract drifted")
    require(metric.get("iterations_per_report") == 5, "iteration contract drifted")
    rounds = metric.get("rounds_per_campaign")
    require(rounds == 8, "round count drifted")
    require(close(float(metric.get("minimum_median_paired_improvement")), MIN_IMPROVEMENT), "promotion floor drifted")

    runs = data.get("runs")
    require(isinstance(runs, list) and len(runs) == 2, "expected two independent campaigns")
    require([run.get("name") for run in runs] == ["KA13", "KA14"], "campaign identities drifted")
    contexts = [run.get("run_context_id") for run in runs]
    require(all(isinstance(value, str) and value for value in contexts), "missing run context")
    require(len(set(contexts)) == 2, "campaigns are not independent")
    run_medians = [validate_run(run, rounds) for run in runs]

    consensus = data.get("consensus")
    require(isinstance(consensus, dict), "missing consensus")
    require(consensus.get("independent_run_count") == 2, "independent-run count drifted")
    require(consensus.get("all_pairs_positive_in_all_runs") is True, "positive-pair gate failed")
    require(consensus.get("semantic_gate_passed_in_all_runs") is True, "semantic gate failed")
    require(consensus.get("environment_compatible_across_runs") is True, "environment gate failed")
    require(
        close(float(consensus.get("median_paired_relative_improvement_across_runs")), statistics.median(run_medians)),
        "cross-run median drifted",
    )
    require(
        close(float(consensus.get("minimum_run_median_paired_relative_improvement")), min(run_medians)),
        "minimum run median drifted",
    )
    require(min(run_medians) >= MIN_IMPROVEMENT, "consensus promotion floor failed")
    require(consensus.get("promotion_state") == "eligible_for_scoped_explicit_min_latency_plan", "promotion state drifted")

    boundary = data.get("claim_boundary")
    require(isinstance(boundary, dict), "missing claim boundary")
    require(boundary.get("generic_default_changed") is False, "generic default must remain unchanged")
    require(boundary.get("cross_device_portability_claimed") is False, "cross-device overclaim")
    require(boundary.get("other_model_speedup_claimed") is False, "cross-model overclaim")
    require(boundary.get("attention_staged_promoted") is False, "attention staged must remain unpromoted")


def self_test() -> None:
    pairs = []
    for index, improvement in enumerate((0.05, 0.06, 0.07, 0.08, 0.05, 0.06, 0.07, 0.08)):
        parent = 100.0
        candidate = parent * (1.0 - improvement)
        pairs.append({
            "round": index,
            "order": "fused_then_fused_mlp" if index % 2 == 0 else "fused_mlp_then_fused",
            "fused_median_ms": parent,
            "fused_mlp_median_ms": candidate,
            "paired_relative_improvement": improvement,
        })
    improvements = [pair["paired_relative_improvement"] for pair in pairs]
    run = {
        "pairs": pairs,
        "candidate_round_wins": 8,
        "parent_round_wins": 0,
        "ties": 0,
        "median_fused_generation_gpu_ms": 100.0,
        "median_fused_mlp_generation_gpu_ms": statistics.median(pair["fused_mlp_median_ms"] for pair in pairs),
        "median_paired_relative_improvement": statistics.median(improvements),
        "mean_paired_relative_improvement": statistics.mean(improvements),
        "min_paired_relative_improvement": min(improvements),
        "max_paired_relative_improvement": max(improvements),
    }
    require(validate_run(run, 8) >= MIN_IMPROVEMENT, "self-test positive case failed")
    run["pairs"][0]["paired_relative_improvement"] = -1.0
    try:
        validate_run(run, 8)
    except RuntimeError:
        return
    raise RuntimeError("self-test failed to reject corrupted pair evidence")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
        require(args.evidence is not None, "evidence path is required")
        data = json.loads(args.evidence.read_text(encoding="utf-8"))
        require(isinstance(data, dict), "evidence root must be an object")
        validate(data)
        print("F16_FUSED_MLP_PROMOTION_EVIDENCE_OK")
        return 0
    except Exception as error:  # noqa: BLE001 - validator must fail closed at CLI boundary.
        print(f"F16 fused-MLP promotion evidence validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
