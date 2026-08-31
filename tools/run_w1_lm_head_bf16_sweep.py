#!/usr/bin/env python3
"""Run the NNIS W1 LM-head BF16 representation sweep as one evidence campaign.

This driver does not promote a runtime or model-format change. It repeatedly runs
`smollm2_lm_head_weight_representation` across candidate block sizes, requires a
clean and stable exact-head/environment fingerprint, and emits an isolated winner
that still requires a separate end-to-end AB/ABBA promotion gate.
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

EXPERIMENT = "W1-smollm2-lm-head-f32-vs-bf16-weight-representation"
SOURCE_REPO = "HuggingFaceTB/SmolLM2-135M"
SOURCE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SOURCE_MODEL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
DEFAULT_BLOCKS = (32, 64, 128, 256, 512)
EXPECTED_LM_HEAD_ELEMENTS = 28_311_552
EXPECTED_F32_BYTES = 113_246_208
EXPECTED_BF16_BYTES = 56_623_104


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fail-closed physical W1 sweep. The output is candidate-only "
            "evidence and must not be treated as runtime promotion."
        )
    )
    parser.add_argument("--model", required=True, type=Path, help="pinned NNIS SmolLM2 fixture")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ordinal")
    parser.add_argument(
        "--blocks",
        nargs="+",
        type=positive_int,
        default=list(DEFAULT_BLOCKS),
        help="candidate CUDA block sizes (default: 32 64 128 256 512)",
    )
    parser.add_argument("--rounds", type=positive_int, default=2, help="number of sweep rounds")
    parser.add_argument("--warmups", type=positive_int, default=20)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument(
        "--run-context",
        default=os.environ.get("NNIS_BENCH_RUN_CONTEXT_ID"),
        help="explicit benchmark campaign id (or NNIS_BENCH_RUN_CONTEXT_ID)",
    )
    parser.add_argument(
        "--environment-label",
        default=os.environ.get("NNIS_BENCH_ENVIRONMENT_LABEL"),
        help="optional stable environment/container label",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="new directory for raw reports and summary JSON",
    )
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be non-negative")
    if not args.run_context or not args.run_context.strip():
        parser.error(
            "--run-context is required (or set NNIS_BENCH_RUN_CONTEXT_ID); "
            "cross-run comparison fails closed without it"
        )
    if len(set(args.blocks)) != len(args.blocks):
        parser.error("--blocks must not contain duplicates")
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


def stable_environment(metadata: dict[str, Any]) -> dict[str, Any]:
    fingerprint = metadata.get("environment_fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("report is missing environment_fingerprint")
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_report(
    report: dict[str, Any],
    *,
    block_size: int,
    expected_head: str,
    run_context: str,
    expected_environment: dict[str, Any] | None,
) -> dict[str, Any]:
    require(report.get("schema_version") == 1, "unexpected W1 report schema_version")
    require(report.get("experiment") == EXPERIMENT, "unexpected W1 experiment id")
    require(report.get("source_repo") == SOURCE_REPO, "unexpected W1 source repository")
    require(report.get("source_revision") == SOURCE_REVISION, "unexpected W1 source revision")
    require(
        report.get("source_model_sha256") == SOURCE_MODEL_SHA256,
        "unexpected W1 source model SHA-256",
    )
    require(report.get("bitwise_equivalent_all_logits") is True, "W1 bitwise gate did not pass")

    benchmark_config = report.get("benchmark_config")
    require(isinstance(benchmark_config, dict), "W1 report is missing benchmark_config")
    require(
        benchmark_config.get("candidate_block_size") == block_size,
        "W1 report block size does not match requested candidate",
    )

    representation = report.get("representation")
    require(isinstance(representation, dict), "W1 report is missing representation evidence")
    require(
        representation.get("lm_head_elements") == EXPECTED_LM_HEAD_ELEMENTS,
        "unexpected LM-head element count",
    )
    require(
        representation.get("baseline_storage_bytes") == EXPECTED_F32_BYTES,
        "unexpected f32 LM-head storage accounting",
    )
    require(
        representation.get("candidate_storage_bytes") == EXPECTED_BF16_BYTES,
        "unexpected BF16 LM-head storage accounting",
    )
    require(
        representation.get("storage_bytes_saved") == EXPECTED_F32_BYTES - EXPECTED_BF16_BYTES,
        "unexpected LM-head storage saving",
    )
    require(
        representation.get("exact_bf16_roundtrip_from_fixture_f32") is True,
        "fixture is not an exact widened-BF16 LM head",
    )
    require(
        representation.get("candidate_changes_representation") is True,
        "W1 report did not declare a physical representation change",
    )

    reference = report.get("reference")
    candidate = report.get("candidate")
    require(isinstance(reference, dict) and isinstance(candidate, dict), "missing benchmark reports")
    reference_metadata = reference.get("metadata")
    candidate_metadata = candidate.get("metadata")
    require(
        isinstance(reference_metadata, dict) and isinstance(candidate_metadata, dict),
        "missing benchmark metadata",
    )
    for side, metadata in (("reference", reference_metadata), ("candidate", candidate_metadata)):
        require(metadata.get("git_commit") == expected_head, f"{side} report is not exact HEAD")
        require(metadata.get("git_dirty") is False, f"{side} report has a dirty tracked worktree")
        fingerprint = metadata.get("environment_fingerprint")
        require(isinstance(fingerprint, dict), f"{side} report lacks environment fingerprint")
        require(
            fingerprint.get("run_context_id") == run_context,
            f"{side} report run_context_id does not match campaign",
        )

    reference_environment = stable_environment(reference_metadata)
    candidate_environment = stable_environment(candidate_metadata)
    require(
        reference_environment == candidate_environment,
        "reference and candidate environment fingerprints differ inside one W1 run",
    )
    if expected_environment is not None:
        require(
            reference_environment == expected_environment,
            "W1 environment fingerprint drifted across sweep runs",
        )

    reference_stats = reference.get("statistics")
    candidate_stats = candidate.get("statistics")
    require(
        isinstance(reference_stats, dict) and isinstance(candidate_stats, dict),
        "missing W1 timing statistics",
    )
    reference_median = reference_stats.get("median_ms")
    candidate_median = candidate_stats.get("median_ms")
    require(
        isinstance(reference_median, (int, float)) and reference_median > 0,
        "invalid reference median",
    )
    require(
        isinstance(candidate_median, (int, float)) and candidate_median > 0,
        "invalid candidate median",
    )
    return {
        "environment": reference_environment,
        "reference_median_ms": float(reference_median),
        "candidate_median_ms": float(candidate_median),
        "speedup_reference_over_candidate": float(reference_median) / float(candidate_median),
    }


def output_directory(requested: Path | None) -> Path:
    if requested is not None:
        path = requested
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = Path("artifacts") / f"w1-lm-head-bf16-sweep-{stamp}"
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    args = parse_args()
    try:
        expected_head, dirty = repository_identity()
        if dirty:
            raise RuntimeError("tracked worktree is dirty; W1 physical evidence requires a clean exact head")
        out_dir = output_directory(args.output_dir)
        run_context = args.run_context.strip()
        environment = os.environ.copy()
        environment["NNIS_BENCH_RUN_CONTEXT_ID"] = run_context
        environment["NNIS_PROFILE_WARMUPS"] = str(args.warmups)
        environment["NNIS_PROFILE_ITERATIONS"] = str(args.iterations)
        if args.environment_label:
            environment["NNIS_BENCH_ENVIRONMENT_LABEL"] = args.environment_label.strip()

        stable_env: dict[str, Any] | None = None
        per_block: dict[int, list[dict[str, Any]]] = {block: [] for block in args.blocks}
        raw_reports: list[dict[str, Any]] = []

        for round_index in range(args.rounds):
            order = list(args.blocks if round_index % 2 == 0 else reversed(args.blocks))
            for block_size in order:
                print(
                    f"W1 round {round_index + 1}/{args.rounds}: block={block_size}",
                    file=sys.stderr,
                    flush=True,
                )
                run_env = environment.copy()
                run_env["NNIS_BF16_WEIGHT_BLOCK_SIZE"] = str(block_size)
                command = [
                    "cargo",
                    "run",
                    "--locked",
                    "--release",
                    "-p",
                    "nnis-bench",
                    "--example",
                    "smollm2_lm_head_weight_representation",
                    "--",
                    "--model",
                    str(args.model),
                    "--device",
                    str(args.device),
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=run_env,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"W1 block {block_size} failed:\n{completed.stderr.strip()}\n{completed.stdout.strip()}"
                    )
                try:
                    report = json.loads(completed.stdout)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"W1 block {block_size} did not emit valid JSON: {error}"
                    ) from error
                require(isinstance(report, dict), "W1 example emitted non-object JSON")
                validated = validate_report(
                    report,
                    block_size=block_size,
                    expected_head=expected_head,
                    run_context=run_context,
                    expected_environment=stable_env,
                )
                if stable_env is None:
                    stable_env = validated["environment"]

                report_name = f"round-{round_index + 1:02d}-block-{block_size:04d}.json"
                (out_dir / report_name).write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                sample = {
                    "round": round_index + 1,
                    "block_size": block_size,
                    "report_file": report_name,
                    "reference_median_ms": validated["reference_median_ms"],
                    "candidate_median_ms": validated["candidate_median_ms"],
                    "speedup_reference_over_candidate": validated[
                        "speedup_reference_over_candidate"
                    ],
                }
                per_block[block_size].append(sample)
                raw_reports.append(sample)

        require(stable_env is not None, "W1 sweep produced no environment evidence")
        results: list[dict[str, Any]] = []
        for block_size in args.blocks:
            samples = per_block[block_size]
            require(len(samples) == args.rounds, f"missing W1 samples for block {block_size}")
            results.append(
                {
                    "block_size": block_size,
                    "rounds": len(samples),
                    "candidate_median_ms_across_rounds": statistics.median(
                        sample["candidate_median_ms"] for sample in samples
                    ),
                    "reference_median_ms_across_rounds": statistics.median(
                        sample["reference_median_ms"] for sample in samples
                    ),
                    "median_speedup_reference_over_candidate": statistics.median(
                        sample["speedup_reference_over_candidate"] for sample in samples
                    ),
                    "samples": samples,
                }
            )

        isolated_winner = min(results, key=lambda item: item["candidate_median_ms_across_rounds"])
        summary = {
            "schema_version": 1,
            "campaign": "W1-smollm2-lm-head-bf16-physical-sweep",
            "promotion_state": "candidate-only; nnis-model runtime and model format v1 remain unchanged",
            "git_commit": expected_head,
            "git_dirty": False,
            "run_context_id": run_context,
            "environment": stable_env,
            "config": {
                "device": args.device,
                "blocks": args.blocks,
                "rounds": args.rounds,
                "warmups_per_run": args.warmups,
                "iterations_per_run": args.iterations,
                "round_order_policy": "forward on odd rounds, reverse on even rounds",
            },
            "representation": {
                "scope": "SmolLM2 tied LM-head copy only",
                "f32_bytes": EXPECTED_F32_BYTES,
                "bf16_bytes": EXPECTED_BF16_BYTES,
                "bytes_saved": EXPECTED_F32_BYTES - EXPECTED_BF16_BYTES,
                "storage_reduction_fraction": 0.5,
            },
            "results": results,
            "isolated_winner": {
                "block_size": isolated_winner["block_size"],
                "candidate_median_ms_across_rounds": isolated_winner[
                    "candidate_median_ms_across_rounds"
                ],
                "median_speedup_reference_over_candidate": isolated_winner[
                    "median_speedup_reference_over_candidate"
                ],
            },
            "next_gate": (
                "Do not promote from this sweep. If the isolated result is worth pursuing, "
                "introduce an explicit representation plan without changing model format v1, "
                "then require fingerprint-compatible end-to-end AB/ABBA verification before promotion."
            ),
            "raw_reports": raw_reports,
        }
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(summary_path)
        return 0
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"W1 sweep failed closed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
