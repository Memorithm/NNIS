#!/usr/bin/env python3
"""Run and verify the F16 fused-MLP SmolLM2 end-to-end ABBA gate.

The parent is the explicit `fused` F16 execution plan. The candidate is
`fused_mlp`, which keeps the same resident [N,K] representation and grouped QKV
path while replacing gate/up projection plus SiLU-multiply with one fused launch.

The tool is fail-closed: it validates exact Git identity, model provenance,
execution-plan identity, stable environment metadata, the pinned 32-token greedy
trajectory, and recomputes paired improvements from raw reports. It never changes
runtime defaults automatically.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any

SOURCE_REPO = "HuggingFaceTB/SmolLM2-135M"
SOURCE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SOURCE_MODEL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
INPUT_IDS = [22_007, 6_463, 314]
DECODE_STEPS = 32
EXPECTED_GREEDY_IDS = [
    260, 3_075, 338, 6_650, 260, 2_591, 284, 260, 8_872, 1_592, 30, 198,
    198, 504, 8_872, 314, 253, 8_304, 282, 260, 2_591, 30, 657, 314, 253,
    19_284, 1_248, 338, 21_837, 260, 2_591, 30,
]
MIN_PAIRED_IMPROVEMENT = 0.03
PLAN_LAYOUT = {
    "fused": "nk_transposed_fused_groups_candidate",
    "fused_mlp": "nk_transposed_fused_mlp_candidate",
}
NUMERIC_PLAN = {
    "schema_version": 1,
    "weight_storage": "f16",
    "activation_storage": "f16",
    "kv_storage": "f16",
    "projection_accumulator": "f32",
    "attention_accumulator": "f32",
    "logits_storage": "f32",
}
REPORT_RE = re.compile(r"^round_(\d+)_(fused|fused_mlp)\.json$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or verify the fail-closed F16 fused-MLP SmolLM2 ABBA gate."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--verify-dir",
        type=Path,
        help="recompute and validate one existing raw campaign directory",
    )
    modes.add_argument(
        "--consensus",
        nargs=2,
        type=Path,
        metavar=("RUN_A", "RUN_B"),
        help="recompute two raw campaigns and validate independent-run consensus",
    )
    modes.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rounds", type=positive_int, default=8)
    parser.add_argument("--warmups", type=positive_int, default=2)
    parser.add_argument("--iterations", type=positive_int, default=5)
    parser.add_argument(
        "--run-context",
        default=os.environ.get("NNIS_BENCH_RUN_CONTEXT_ID"),
        help="explicit campaign id (or NNIS_BENCH_RUN_CONTEXT_ID)",
    )
    parser.add_argument(
        "--environment-label",
        default=os.environ.get("NNIS_BENCH_ENVIRONMENT_LABEL"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be non-negative")
    if not (args.verify_dir or args.consensus or args.self_test):
        if args.model is None:
            parser.error("--model is required when running a campaign")
        if not args.run_context or not args.run_context.strip():
            parser.error("--run-context is required (or set NNIS_BENCH_RUN_CONTEXT_ID)")
    return args


def run_text(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({' '.join(command)}): {detail}")
    return completed.stdout.strip()


def repository_identity() -> tuple[str, bool]:
    head = run_text(["git", "rev-parse", "HEAD"])
    dirty = bool(run_text(["git", "status", "--porcelain", "--untracked-files=no"]))
    return head, dirty


def normalized_environment(metadata: dict[str, Any], *, keep_run_context: bool) -> dict[str, Any]:
    result = copy.deepcopy(metadata)
    result.pop("unix_timestamp_seconds", None)
    result.pop("git_commit", None)
    result.pop("git_dirty", None)
    fingerprint = result.get("environment_fingerprint")
    require(isinstance(fingerprint, dict), "missing environment_fingerprint")
    if not keep_run_context:
        fingerprint.pop("run_context_id", None)
    return result


def expected_execution_plan(plan: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "numeric": NUMERIC_PLAN,
        "projection_layout": PLAN_LAYOUT[plan],
    }


def validate_report(
    report: dict[str, Any],
    *,
    plan: str,
    warmups: int,
    iterations: int,
    expected_head: str | None,
    expected_run_context: str | None,
    expected_environment: dict[str, Any] | None,
) -> tuple[float, dict[str, Any], str]:
    require(report.get("schema_version") == 1, "unexpected report schema_version")
    require(
        report.get("benchmark") == "smollm2-135m-f16-projection-plan-e2e",
        "unexpected benchmark identity",
    )
    require(report.get("backend") == "nnis", "unexpected backend")
    require(report.get("plan_name") == plan, f"plan_name mismatch for {plan}")
    require(report.get("source_repo") == SOURCE_REPO, "unexpected source repository")
    require(report.get("source_revision") == SOURCE_REVISION, "unexpected source revision")
    require(report.get("source_model_sha256") == SOURCE_MODEL_SHA256, "unexpected source SHA-256")
    require(report.get("source_weight_dtype") == "bfloat16", "unexpected source dtype")
    require(report.get("persisted_execution_weight_dtype") == "f32", "unexpected persisted dtype")
    require(report.get("resident_weight_dtype") == "f16", "unexpected resident dtype")
    require(report.get("input_ids") == INPUT_IDS, "input token sequence drifted")
    require(report.get("decode_steps") == DECODE_STEPS, "decode-step contract drifted")
    require(report.get("warmup_iterations") == warmups, "warmup count drifted")
    require(report.get("iterations") == iterations, "iteration count drifted")
    require(
        report.get("execution_plan") == expected_execution_plan(plan),
        f"execution plan drifted for {plan}",
    )
    require(report.get("exact_greedy_32_of_32") is True, "32/32 greedy gate failed")
    require(report.get("generated_ids") == EXPECTED_GREEDY_IDS, "greedy trajectory drifted")
    require(
        report.get("generation_forward_runs_per_request") == DECODE_STEPS - 1,
        "generation forward-count drifted",
    )
    require(
        report.get("sampling_included_in_generation_stage_gpu_time") is False,
        "sampling entered the generation GPU interval",
    )
    require(
        report.get("final_generated_token_consumed_by_model") is False,
        "final-token consumption contract drifted",
    )

    metadata = report.get("metadata")
    require(isinstance(metadata, dict), "missing benchmark metadata")
    commit = metadata.get("git_commit")
    require(isinstance(commit, str) and commit, "missing metadata.git_commit")
    require(metadata.get("git_dirty") is False, "report came from a dirty tracked worktree")
    if expected_head is not None:
        require(commit == expected_head, "report is not from the exact requested Git head")
    fingerprint = metadata.get("environment_fingerprint")
    require(isinstance(fingerprint, dict), "missing environment fingerprint")
    run_context = fingerprint.get("run_context_id")
    require(isinstance(run_context, str) and run_context.strip(), "missing run_context_id")
    if expected_run_context is not None:
        require(run_context == expected_run_context, "run_context_id mismatch")
    environment = normalized_environment(metadata, keep_run_context=True)
    if expected_environment is not None:
        require(environment == expected_environment, "environment drifted within campaign")

    generation = report.get("generation_stage_gpu")
    require(isinstance(generation, dict), "missing generation_stage_gpu report")
    stats = generation.get("statistics")
    require(isinstance(stats, dict), "missing generation_stage_gpu statistics")
    median = stats.get("median_ms")
    require(
        isinstance(median, (int, float)) and math.isfinite(median) and median > 0.0,
        "invalid generation GPU median",
    )
    samples = generation.get("samples_ms")
    require(isinstance(samples, list) and len(samples) == iterations, "raw GPU sample count drifted")
    require(
        all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0 for value in samples),
        "invalid raw generation GPU sample",
    )
    recomputed_median = statistics.median(float(value) for value in samples)
    require(
        math.isclose(float(median), recomputed_median, rel_tol=0.0, abs_tol=1.0e-9),
        "reported generation GPU median does not match raw samples",
    )
    return float(median), environment, commit


def report_files(directory: Path) -> dict[tuple[int, str], Path]:
    require(directory.is_dir(), f"campaign directory does not exist: {directory}")
    result: dict[tuple[int, str], Path] = {}
    for path in directory.iterdir():
        match = REPORT_RE.match(path.name)
        if match is None:
            continue
        key = (int(match.group(1)), match.group(2))
        require(key not in result, f"duplicate raw report for {key}")
        result[key] = path
    require(result, f"no raw round reports found in {directory}")
    rounds = sorted({round_index for round_index, _ in result})
    require(rounds == list(range(len(rounds))), "round indices are not contiguous from zero")
    for round_index in rounds:
        require((round_index, "fused") in result, f"missing fused report for round {round_index}")
        require((round_index, "fused_mlp") in result, f"missing fused_mlp report for round {round_index}")
    require(len(result) == 2 * len(rounds), "unexpected raw report count")
    return result


def validate_recorded_order(directory: Path, rounds: int) -> None:
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return
    try:
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid existing summary JSON: {error}") from error
    pairs = recorded.get("pairs")
    if pairs is None:
        return
    require(isinstance(pairs, list) and len(pairs) == rounds, "recorded pair count drifted")
    for round_index, pair in enumerate(pairs):
        require(isinstance(pair, dict), "recorded pair is not an object")
        require(pair.get("round") == round_index, "recorded round index drifted")
        expected_order = (
            "fused_then_fused_mlp" if round_index % 2 == 0 else "fused_mlp_then_fused"
        )
        require(pair.get("order") == expected_order, "recorded ABBA order drifted")


def recompute_campaign(directory: Path, *, expected_head: str | None = None) -> dict[str, Any]:
    files = report_files(directory)
    rounds = len(files) // 2
    validate_recorded_order(directory, rounds)

    first = json.loads(files[(0, "fused")].read_text(encoding="utf-8"))
    warmups = first.get("warmup_iterations")
    iterations = first.get("iterations")
    require(isinstance(warmups, int) and warmups > 0, "invalid warmup count")
    require(isinstance(iterations, int) and iterations > 0, "invalid iteration count")
    first_metadata = first.get("metadata")
    require(isinstance(first_metadata, dict), "missing first-report metadata")
    fingerprint = first_metadata.get("environment_fingerprint")
    require(isinstance(fingerprint, dict), "missing first-report environment fingerprint")
    run_context = fingerprint.get("run_context_id")
    require(isinstance(run_context, str) and run_context.strip(), "missing campaign run_context_id")

    stable_environment: dict[str, Any] | None = None
    commit: str | None = None
    pairs: list[dict[str, Any]] = []
    for round_index in range(rounds):
        medians: dict[str, float] = {}
        for plan in ("fused", "fused_mlp"):
            report = json.loads(files[(round_index, plan)].read_text(encoding="utf-8"))
            median, environment, report_commit = validate_report(
                report,
                plan=plan,
                warmups=warmups,
                iterations=iterations,
                expected_head=expected_head,
                expected_run_context=run_context,
                expected_environment=stable_environment,
            )
            if stable_environment is None:
                stable_environment = environment
            if commit is None:
                commit = report_commit
            else:
                require(report_commit == commit, "Git commit drifted within campaign")
            medians[plan] = median
        parent = medians["fused"]
        candidate = medians["fused_mlp"]
        improvement = (parent - candidate) / parent
        order = "fused_then_fused_mlp" if round_index % 2 == 0 else "fused_mlp_then_fused"
        pairs.append(
            {
                "round": round_index,
                "order": order,
                "fused_median_ms": parent,
                "fused_mlp_median_ms": candidate,
                "paired_relative_improvement": improvement,
                "speedup_ratio": parent / candidate,
            }
        )

    require(stable_environment is not None and commit is not None, "campaign produced no evidence")
    improvements = [pair["paired_relative_improvement"] for pair in pairs]
    parent_medians = [pair["fused_median_ms"] for pair in pairs]
    candidate_medians = [pair["fused_mlp_median_ms"] for pair in pairs]
    parent_first = [
        pair["paired_relative_improvement"]
        for pair in pairs
        if pair["order"] == "fused_then_fused_mlp"
    ]
    candidate_first = [
        pair["paired_relative_improvement"]
        for pair in pairs
        if pair["order"] == "fused_mlp_then_fused"
    ]
    wins = sum(pair["fused_mlp_median_ms"] < pair["fused_median_ms"] for pair in pairs)
    losses = sum(pair["fused_mlp_median_ms"] > pair["fused_median_ms"] for pair in pairs)
    ties = rounds - wins - losses
    median_parent = statistics.median(parent_medians)
    median_candidate = statistics.median(candidate_medians)
    median_improvement = statistics.median(improvements)
    all_positive = all(value > 0.0 for value in improvements)
    passes_floor = median_improvement >= MIN_PAIRED_IMPROVEMENT and all_positive

    cross_run_environment = normalized_environment(first_metadata, keep_run_context=False)
    return {
        "schema_version": 1,
        "campaign": "f16_smollm2_fused_vs_fused_mlp_abba_v1",
        "promotion_state": "candidate-only; no automatic runtime promotion",
        "git_commit": commit,
        "run_context_id": run_context,
        "config": {
            "rounds": rounds,
            "warmups_per_report": warmups,
            "iterations_per_report": iterations,
            "decode_steps": DECODE_STEPS,
            "order_by_round_parity": ["fused_then_fused_mlp", "fused_mlp_then_fused"],
            "minimum_paired_improvement": MIN_PAIRED_IMPROVEMENT,
        },
        "semantic_gate": {
            "exact_greedy_32_of_32_all": True,
            "generated_ids": EXPECTED_GREEDY_IDS,
        },
        "environment_compatible_within_run": True,
        "cross_run_environment": cross_run_environment,
        "comparison": {
            "fused_mlp_round_wins": wins,
            "fused_round_wins": losses,
            "round_ties": ties,
            "median_fused_generation_gpu_ms": median_parent,
            "median_fused_mlp_generation_gpu_ms": median_candidate,
            "median_paired_relative_improvement": median_improvement,
            "mean_paired_relative_improvement": statistics.mean(improvements),
            "min_paired_relative_improvement": min(improvements),
            "max_paired_relative_improvement": max(improvements),
            "median_improvement_fused_then_fused_mlp": statistics.median(parent_first),
            "median_improvement_fused_mlp_then_fused": statistics.median(candidate_first),
            "aggregate_speedup_ratio": median_parent / median_candidate,
            "absolute_generation_gpu_ms_saved": median_parent - median_candidate,
            "all_pairs_positive": all_positive,
            "passes_three_percent_floor": passes_floor,
        },
        "pairs": pairs,
    }


def output_directory(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        requested = Path("artifacts") / f"f16-fused-mlp-e2e-abba-{stamp}"
    if requested.exists() and any(requested.iterdir()):
        raise RuntimeError(f"output directory is not empty: {requested}")
    requested.mkdir(parents=True, exist_ok=True)
    return requested


def build_benchmark() -> Path:
    completed = subprocess.run(
        [
            "cargo", "build", "--locked", "--release", "-p", "nnis-bench", "--example",
            "smollm2_f16_projection_plan_e2e",
        ],
        check=False,
    )
    require(completed.returncode == 0, "failed to build F16 projection-plan benchmark")
    binary = Path("target/release/examples/smollm2_f16_projection_plan_e2e")
    require(binary.is_file(), "benchmark binary is missing after cargo build")
    return binary


def run_variant(
    binary: Path,
    *,
    model: Path,
    device: int,
    plan: str,
    warmups: int,
    iterations: int,
    environment: dict[str, str],
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(binary), "--model", str(model), "--device", str(device), "--plan", plan,
            "--warmups", str(warmups), "--iterations", str(iterations),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"F16 projection-plan variant {plan} failed:\n{completed.stderr.strip()}\n{completed.stdout.strip()}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"variant {plan} emitted invalid JSON: {error}") from error
    require(isinstance(report, dict), f"variant {plan} emitted non-object JSON")
    return report


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    head, dirty = repository_identity()
    require(not dirty, "tracked worktree is dirty; ABBA requires a clean exact head")
    out_dir = output_directory(args.output_dir)
    run_context = args.run_context.strip()
    environment = os.environ.copy()
    environment["NNIS_BENCH_RUN_CONTEXT_ID"] = run_context
    if args.environment_label:
        environment["NNIS_BENCH_ENVIRONMENT_LABEL"] = args.environment_label.strip()
    binary = build_benchmark()

    stable_environment: dict[str, Any] | None = None
    for round_index in range(args.rounds):
        order = ("fused", "fused_mlp") if round_index % 2 == 0 else ("fused_mlp", "fused")
        for plan in order:
            print(
                f"F16 fused-MLP ABBA round {round_index + 1}/{args.rounds}: {plan}",
                file=sys.stderr,
                flush=True,
            )
            report = run_variant(
                binary,
                model=args.model,
                device=args.device,
                plan=plan,
                warmups=args.warmups,
                iterations=args.iterations,
                environment=environment,
            )
            _, observed_environment, _ = validate_report(
                report,
                plan=plan,
                warmups=args.warmups,
                iterations=args.iterations,
                expected_head=head,
                expected_run_context=run_context,
                expected_environment=stable_environment,
            )
            if stable_environment is None:
                stable_environment = observed_environment
            path = out_dir / f"round_{round_index}_{plan}.json"
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = recompute_campaign(out_dir, expected_head=head)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"summary_path": str(out_dir / "summary.json"), "summary": summary}


def consensus(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    left = recompute_campaign(left_dir)
    right = recompute_campaign(right_dir)
    require(left["run_context_id"] != right["run_context_id"], "independent runs require distinct run_context_id values")
    require(left["git_commit"] == right["git_commit"], "runtime Git commit differs between campaigns")
    require(left["config"] == right["config"], "campaign configuration differs between runs")
    require(
        left["cross_run_environment"] == right["cross_run_environment"],
        "stable hardware/software environment differs between runs",
    )
    require(
        left["semantic_gate"] == right["semantic_gate"],
        "semantic trajectory evidence differs between runs",
    )
    for label, summary in (("left", left), ("right", right)):
        comparison = summary["comparison"]
        require(comparison["all_pairs_positive"] is True, f"{label} run has a non-positive pair")
        require(
            comparison["passes_three_percent_floor"] is True,
            f"{label} run does not clear the 3 percent paired-improvement floor",
        )
    medians = [
        left["comparison"]["median_paired_relative_improvement"],
        right["comparison"]["median_paired_relative_improvement"],
    ]
    return {
        "schema_version": 1,
        "kind": "f16_smollm2_fused_mlp_e2e_consensus_v1",
        "compatible": True,
        "runtime_git_commit": left["git_commit"],
        "run_context_ids": [left["run_context_id"], right["run_context_id"]],
        "independent_run_count": 2,
        "minimum_required_independent_runs": 2,
        "minimum_paired_improvement": MIN_PAIRED_IMPROVEMENT,
        "median_paired_relative_improvement_across_runs": statistics.median(medians),
        "minimum_run_median_paired_relative_improvement": min(medians),
        "all_pairs_positive_in_all_runs": True,
        "semantic_gate_passed_in_all_runs": True,
        "environment_compatible_across_runs": True,
        "promotion_state": "eligible for explicit NNIS promotion review; not automatically promoted",
        "runs": [left, right],
    }


def self_test() -> None:
    values = [0.08, 0.10, 0.05, 0.09, 0.07, 0.08, 0.07, 0.07]
    require(statistics.median(values) > MIN_PAIRED_IMPROVEMENT, "self-test median gate failed")
    require(all(value > 0 for value in values), "self-test positivity gate failed")
    sample = {"unix_timestamp_seconds": 1, "git_commit": "abc", "git_dirty": False,
              "environment_fingerprint": {"run_context_id": "run-a", "schema_version": 1}}
    normalized = normalized_environment(sample, keep_run_context=False)
    require("unix_timestamp_seconds" not in normalized, "self-test timestamp normalization failed")
    require("git_commit" not in normalized, "self-test Git normalization failed")
    require(
        "run_context_id" not in normalized["environment_fingerprint"],
        "self-test run-context normalization failed",
    )


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            self_test()
            print("F16_FUSED_MLP_ABBA_SELF_TEST_OK")
            return 0
        if args.verify_dir:
            print(json.dumps(recompute_campaign(args.verify_dir), indent=2, sort_keys=True))
            return 0
        if args.consensus:
            print(json.dumps(consensus(args.consensus[0], args.consensus[1]), indent=2, sort_keys=True))
            return 0
        result = run_campaign(args)
        print(result["summary_path"])
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary must fail closed with context.
        print(f"F16 fused-MLP ABBA gate failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
