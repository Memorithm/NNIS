#!/usr/bin/env python3
"""Run the W1 SmolLM2 end-to-end ABBA promotion gate.

A is the promoted E1.1 all-f32 LM-head GEMV64 parent. B is the candidate-only
W1 runtime representation with only the LM-head resident as BF16 and GEMV32
execution selected independently through the projection plan.

The script verifies exact-head provenance, a stable hardware/environment
fingerprint, deterministic identical greedy token trajectories, and workload
identity before reporting latency ratios. It never promotes the candidate by
itself.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

SOURCE_REPO = "HuggingFaceTB/SmolLM2-135M"
SOURCE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SOURCE_MODEL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
EXPECTED_LM_HEAD_F32_BYTES = 113_246_208
EXPECTED_LM_HEAD_BF16_BYTES = 56_623_104

PARENT_PROJECTION = {
    "q_o": {"kernel": "gemm"},
    "k_v": {"kernel": "gemm"},
    "gate_up": {"kernel": "gemm"},
    "down": {"kernel": "gemm"},
    "lm_head": {"kernel": "gemv", "block_size": 64},
}
CANDIDATE_PROJECTION = {
    "q_o": {"kernel": "gemm"},
    "k_v": {"kernel": "gemm"},
    "gate_up": {"kernel": "gemm"},
    "down": {"kernel": "gemm"},
    "lm_head": {"kernel": "gemv", "block_size": 32},
}
PARENT_REPRESENTATION = {"schema_version": 1, "lm_head": "f32"}
CANDIDATE_REPRESENTATION = {"schema_version": 1, "lm_head": "bf16"}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fingerprint-gated W1 end-to-end ABBA evidence; no automatic promotion."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--decode-steps", type=positive_int, default=32)
    parser.add_argument("--rounds", type=positive_int, default=2)
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
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be non-negative")
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def stable_environment(metadata: dict[str, Any]) -> dict[str, Any]:
    fingerprint = metadata.get("environment_fingerprint")
    require(isinstance(fingerprint, dict), "report is missing environment_fingerprint")
    return {
        "nnis_version": metadata.get("nnis_version"),
        "host_arch": metadata.get("host_arch"),
        "host_os": metadata.get("host_os"),
        "gpu_ordinal": metadata.get("gpu_ordinal"),
        "gpu_name": metadata.get("gpu_name"),
        "gpu_uuid": metadata.get("gpu_uuid"),
        "compute_capability_major": metadata.get("compute_capability_major"),
        "compute_capability_minor": metadata.get("compute_capability_minor"),
        "multiprocessor_count": metadata.get("multiprocessor_count"),
        "driver_version": metadata.get("driver_version"),
        "nvrtc_version": metadata.get("nvrtc_version"),
        "environment_fingerprint": fingerprint,
    }


def expected_plans(variant: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if variant == "A":
        return PARENT_PROJECTION, PARENT_REPRESENTATION
    if variant == "B":
        return CANDIDATE_PROJECTION, CANDIDATE_REPRESENTATION
    raise RuntimeError(f"unknown variant {variant}")


def validate_report(
    report: dict[str, Any],
    *,
    variant: str,
    expected_head: str,
    run_context: str,
    expected_environment: dict[str, Any] | None,
    decode_steps: int,
    warmups: int,
    iterations: int,
    expected_generated_ids: list[int] | None,
) -> tuple[dict[str, Any], list[int]]:
    require(report.get("schema_version") == 2, "unexpected SmolLM2 e2e schema_version")
    require(report.get("benchmark") == "smollm2-135m-greedy-e2e", "unexpected benchmark")
    require(report.get("backend") == "nnis", "unexpected backend")
    require(report.get("source_repo") == SOURCE_REPO, "unexpected source repository")
    require(report.get("source_revision") == SOURCE_REVISION, "unexpected source revision")
    require(report.get("source_model_sha256") == SOURCE_MODEL_SHA256, "unexpected source SHA-256")
    require(report.get("execution_weight_dtype") == "f32", "unexpected logical execution dtype")
    require(report.get("decode_steps") == decode_steps, "decode length drifted")
    require(report.get("warmup_iterations") == warmups, "warmup count drifted")
    require(report.get("iterations") == iterations, "iteration count drifted")
    require(report.get("qualified_greedy_prefix_checked") is True, "qualified greedy prefix not checked")

    expected_projection, expected_representation = expected_plans(variant)
    require(report.get("projection_plan") == expected_projection, f"variant {variant} projection plan mismatch")
    require(
        report.get("representation_plan") == expected_representation,
        f"variant {variant} representation plan mismatch",
    )

    metadata = report.get("metadata")
    require(isinstance(metadata, dict), "missing benchmark metadata")
    require(metadata.get("git_commit") == expected_head, "report is not exact checkout HEAD")
    require(metadata.get("git_dirty") is False, "report has a dirty tracked worktree")
    fingerprint = metadata.get("environment_fingerprint")
    require(isinstance(fingerprint, dict), "missing environment fingerprint")
    require(fingerprint.get("run_context_id") == run_context, "run_context_id mismatch")
    environment = stable_environment(metadata)
    if expected_environment is not None:
        require(environment == expected_environment, "environment fingerprint drifted during ABBA")

    generated = report.get("generated_ids")
    require(isinstance(generated, list), "generated_ids missing")
    require(len(generated) == decode_steps, "generated token count mismatch")
    require(all(isinstance(token, int) for token in generated), "generated_ids contains non-integers")
    if expected_generated_ids is not None:
        require(generated == expected_generated_ids, "greedy token trajectory differs between A and B")

    generation = report.get("generation")
    request_total = report.get("request_total")
    memory = report.get("memory")
    require(isinstance(generation, dict) and isinstance(request_total, dict), "timing reports missing")
    require(isinstance(memory, dict), "memory report missing")
    generation_stats = generation.get("statistics")
    request_stats = request_total.get("statistics")
    require(isinstance(generation_stats, dict) and isinstance(request_stats, dict), "timing statistics missing")
    generation_median = generation_stats.get("median_ms")
    request_median = request_stats.get("median_ms")
    require(isinstance(generation_median, (int, float)) and generation_median > 0, "invalid generation median")
    require(isinstance(request_median, (int, float)) and request_median > 0, "invalid request median")

    model_bytes = memory.get("cuda_free_delta_after_model_bytes")
    session_bytes = memory.get("cuda_free_delta_after_session_bytes")
    require(model_bytes is None or isinstance(model_bytes, int), "invalid model memory observation")
    require(session_bytes is None or isinstance(session_bytes, int), "invalid session memory observation")

    return (
        {
            "environment": environment,
            "generation_median_ms": float(generation_median),
            "request_total_median_ms": float(request_median),
            "cuda_free_delta_after_model_bytes": model_bytes,
            "cuda_free_delta_after_session_bytes": session_bytes,
        },
        generated,
    )


def output_directory(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        requested = Path("artifacts") / f"w1-e2e-abba-{stamp}"
    if requested.exists() and any(requested.iterdir()):
        raise RuntimeError(f"output directory is not empty: {requested}")
    requested.mkdir(parents=True, exist_ok=True)
    return requested


def run_variant(
    *,
    variant: str,
    args: argparse.Namespace,
    environment: dict[str, str],
) -> dict[str, Any]:
    if variant == "A":
        projection = "thor-e1-1-lm-head"
        representation = "all-f32"
    else:
        projection = "w1-lm-head-gemv32"
        representation = "w1-lm-head-bf16"
    command = [
        "cargo",
        "run",
        "--locked",
        "--release",
        "-p",
        "nnis-bench",
        "--example",
        "smollm2_e2e",
        "--",
        "--model",
        str(args.model),
        "--device",
        str(args.device),
        "--decode-steps",
        str(args.decode_steps),
        "--warmups",
        str(args.warmups),
        "--iterations",
        str(args.iterations),
        "--projection-plan",
        projection,
        "--representation-plan",
        representation,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"W1 e2e variant {variant} failed:\n{completed.stderr.strip()}\n{completed.stdout.strip()}"
        )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"variant {variant} did not emit valid JSON: {error}") from error
    require(isinstance(report, dict), "SmolLM2 e2e emitted non-object JSON")
    return report


def main() -> int:
    args = parse_args()
    try:
        expected_head, dirty = repository_identity()
        if dirty:
            raise RuntimeError("tracked worktree is dirty; W1 ABBA requires a clean exact head")
        out_dir = output_directory(args.output_dir)
        run_context = args.run_context.strip()
        environment = os.environ.copy()
        environment["NNIS_BENCH_RUN_CONTEXT_ID"] = run_context
        if args.environment_label:
            environment["NNIS_BENCH_ENVIRONMENT_LABEL"] = args.environment_label.strip()

        stable_env: dict[str, Any] | None = None
        expected_generated_ids: list[int] | None = None
        samples: list[dict[str, Any]] = []
        by_variant: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}

        for round_index in range(args.rounds):
            for sequence_index, variant in enumerate(("A", "B", "B", "A"), start=1):
                print(
                    f"W1 e2e ABBA round {round_index + 1}/{args.rounds}: {variant}",
                    file=sys.stderr,
                    flush=True,
                )
                report = run_variant(variant=variant, args=args, environment=environment)
                validated, generated = validate_report(
                    report,
                    variant=variant,
                    expected_head=expected_head,
                    run_context=run_context,
                    expected_environment=stable_env,
                    decode_steps=args.decode_steps,
                    warmups=args.warmups,
                    iterations=args.iterations,
                    expected_generated_ids=expected_generated_ids,
                )
                if stable_env is None:
                    stable_env = validated["environment"]
                if expected_generated_ids is None:
                    expected_generated_ids = generated

                report_name = f"round-{round_index + 1:02d}-{sequence_index:02d}-{variant}.json"
                (out_dir / report_name).write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                sample = {
                    "round": round_index + 1,
                    "sequence_index": sequence_index,
                    "variant": variant,
                    "report_file": report_name,
                    "generation_median_ms": validated["generation_median_ms"],
                    "request_total_median_ms": validated["request_total_median_ms"],
                    "cuda_free_delta_after_model_bytes": validated[
                        "cuda_free_delta_after_model_bytes"
                    ],
                    "cuda_free_delta_after_session_bytes": validated[
                        "cuda_free_delta_after_session_bytes"
                    ],
                }
                samples.append(sample)
                by_variant[variant].append(sample)

        require(stable_env is not None, "ABBA produced no environment evidence")
        require(expected_generated_ids is not None, "ABBA produced no greedy trajectory")
        parent = by_variant["A"]
        candidate = by_variant["B"]
        require(len(parent) == 2 * args.rounds, "missing parent ABBA samples")
        require(len(candidate) == 2 * args.rounds, "missing candidate ABBA samples")

        parent_generation = statistics.median(sample["generation_median_ms"] for sample in parent)
        candidate_generation = statistics.median(
            sample["generation_median_ms"] for sample in candidate
        )
        parent_request = statistics.median(sample["request_total_median_ms"] for sample in parent)
        candidate_request = statistics.median(
            sample["request_total_median_ms"] for sample in candidate
        )

        summary = {
            "schema_version": 1,
            "campaign": "W1-smollm2-lm-head-bf16-e2e-abba",
            "promotion_state": "candidate-only; no automatic runtime promotion",
            "git_commit": expected_head,
            "git_dirty": False,
            "run_context_id": run_context,
            "environment": stable_env,
            "config": {
                "device": args.device,
                "decode_steps": args.decode_steps,
                "rounds": args.rounds,
                "warmups_per_run": args.warmups,
                "iterations_per_run": args.iterations,
                "order_per_round": ["A", "B", "B", "A"],
            },
            "parent": {
                "label": "E1.1 all-f32 LM-head GEMV64",
                "projection_plan": PARENT_PROJECTION,
                "representation_plan": PARENT_REPRESENTATION,
                "generation_median_ms_across_abba": parent_generation,
                "request_total_median_ms_across_abba": parent_request,
            },
            "candidate": {
                "label": "W1 BF16 LM-head GEMV32",
                "projection_plan": CANDIDATE_PROJECTION,
                "representation_plan": CANDIDATE_REPRESENTATION,
                "generation_median_ms_across_abba": candidate_generation,
                "request_total_median_ms_across_abba": candidate_request,
            },
            "comparison": {
                "candidate_over_parent_generation_latency_ratio": candidate_generation
                / parent_generation,
                "parent_over_candidate_generation_throughput_ratio": parent_generation
                / candidate_generation,
                "candidate_over_parent_request_latency_ratio": candidate_request / parent_request,
                "parent_over_candidate_request_throughput_ratio": parent_request
                / candidate_request,
            },
            "semantic_gate": {
                "identical_greedy_trajectory": True,
                "generated_ids": expected_generated_ids,
            },
            "exact_representation_evidence": {
                "scope": "SmolLM2 tied LM-head copy only",
                "f32_bytes": EXPECTED_LM_HEAD_F32_BYTES,
                "bf16_bytes": EXPECTED_LM_HEAD_BF16_BYTES,
                "bytes_saved": EXPECTED_LM_HEAD_F32_BYTES - EXPECTED_LM_HEAD_BF16_BYTES,
                "storage_reduction_fraction": 0.5,
                "note": "exact tensor-storage accounting from W1; not a whole-model CUDA-memory claim",
            },
            "decision_rule": (
                "Do not promote automatically. Interpret latency relative to ABBA variance and the "
                "selected objective; MinMemory may accept an explicit latency trade-off, while "
                "MinLatency requires a credible end-to-end win."
            ),
            "samples": samples,
        }
        summary_path = out_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(summary_path)
        return 0
    except Exception as error:  # noqa: BLE001 - CLI boundary must fail closed with context.
        print(f"W1 e2e ABBA failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
