#!/usr/bin/env python3
"""One-command launcher and verifier for the trained TinyLlama massive F16 campaign.

The launcher creates/reuses the pinned fixture, builds the generic Rust ABBA harness,
runs multiple complete campaigns under distinct run_context_id values, validates
environment/Git compatibility, and emits raw plus consensus machine-readable evidence.
It never changes a runtime default or promotes a candidate automatically.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

BENCHMARK_KIND = "trained-llama-f16-massive-abba-v1"
SUITE_KIND = "nnis-trained-llama-reference-suite-v1"
SOURCE_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SOURCE_REVISION = "d9128824c0c80111be21424e68086f52413fb413"
SOURCE_MODEL_SHA256 = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
DEFAULT_CANDIDATES = "transposed,fused,fused_mlp,staged,fused_mlp_staged"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full pinned TinyLlama-1.1B massive NNIS F16 campaign in one invocation."
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=positive_int, default=2)
    parser.add_argument("--rounds", type=positive_int, default=4)
    parser.add_argument("--warmups", type=positive_int, default=1)
    parser.add_argument("--iterations", type=positive_int, default=3)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--environment-label",
        default="tinyllama-1p1b-trained-massive-v1",
        help="stable label recorded in every benchmark environment fingerprint",
    )
    parser.add_argument("--force-fixture", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be non-negative")
    if not args.environment_label.strip():
        parser.error("--environment-label must not be empty")
    return args


def run_text(command: list[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({' '.join(command)}): {detail}")
    return completed.stdout.strip()


def repo_root() -> Path:
    root = Path(run_text(["git", "rev-parse", "--show-toplevel"]))
    return root.resolve()


def repository_identity(root: Path) -> tuple[str, bool]:
    head = run_text(["git", "-C", str(root), "rev-parse", "HEAD"])
    dirty = bool(
        run_text(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]
        )
    )
    return head, dirty


def validate_existing_fixture(fixture: Path) -> bool:
    suite_path = fixture / "reference_suite.json"
    provenance_path = fixture / "model" / "provenance.json"
    model_path = fixture / "model" / "model.json"
    tokenizer_path = fixture / "tokenizer.json"
    if not all(path.is_file() for path in (suite_path, provenance_path, model_path, tokenizer_path)):
        return False
    try:
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        suite.get("schema_version") == 1
        and suite.get("kind") == SUITE_KIND
        and suite.get("source_repo") == SOURCE_REPO
        and suite.get("source_revision") == SOURCE_REVISION
        and suite.get("source_model_sha256") == SOURCE_MODEL_SHA256
        and isinstance(suite.get("cases"), list)
        and len(suite["cases"]) == 18
        and provenance.get("source_repo") == SOURCE_REPO
        and provenance.get("source_revision") == SOURCE_REVISION
        and provenance.get("source_model_sha256") == SOURCE_MODEL_SHA256
        and provenance.get("tokenizer_sha256") == suite.get("tokenizer_sha256")
    )


def ensure_fixture(root: Path, work_dir: Path, cache_dir: Path | None, force: bool) -> Path:
    fixture = work_dir / "fixture"
    if not force and validate_existing_fixture(fixture):
        return fixture
    command = [
        sys.executable,
        str(root / "tools" / "tinyllama_1p1b_chat_fixture.py"),
        "--output",
        str(fixture),
    ]
    if cache_dir is not None:
        command.extend(["--cache-dir", str(cache_dir)])
    run_text(command)
    if not validate_existing_fixture(fixture):
        raise RuntimeError("fixture generator completed but pinned fixture validation failed")
    return fixture


def build_benchmark(root: Path) -> Path:
    run_text(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "nnis-bench",
            "--example",
            "llama_f16_massive_abba",
        ]
    )
    binary = root / "target" / "release" / "examples" / "llama_f16_massive_abba"
    if not binary.is_file():
        raise RuntimeError(f"benchmark binary was not produced: {binary}")
    return binary


def normalized_environment(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(metadata)
    normalized.pop("unix_timestamp_seconds", None)
    normalized.pop("git_commit", None)
    normalized.pop("git_dirty", None)
    fingerprint = normalized.get("environment_fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("campaign metadata lacks environment_fingerprint")
    fingerprint.pop("run_context_id", None)
    return normalized


def validate_campaign(
    report: dict[str, Any],
    *,
    expected_head: str,
    expected_run_context: str | None = None,
) -> str:
    if report.get("schema_version") != 1 or report.get("benchmark") != BENCHMARK_KIND:
        raise RuntimeError("unexpected massive campaign report identity")
    if report.get("source_repo") != SOURCE_REPO:
        raise RuntimeError("campaign source_repo drifted")
    if report.get("source_revision") != SOURCE_REVISION:
        raise RuntimeError("campaign source_revision drifted")
    if report.get("source_model_sha256") != SOURCE_MODEL_SHA256:
        raise RuntimeError("campaign source model SHA256 drifted")
    metadata = report.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("campaign report lacks metadata")
    if metadata.get("git_commit") != expected_head:
        raise RuntimeError(
            f"campaign git_commit {metadata.get('git_commit')!r} != launcher HEAD {expected_head!r}"
        )
    if metadata.get("git_dirty") is not False:
        raise RuntimeError("campaign report came from a dirty tracked worktree")
    fingerprint = metadata.get("environment_fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("campaign report lacks environment fingerprint")
    run_context = fingerprint.get("run_context_id")
    if not isinstance(run_context, str) or not run_context.strip():
        raise RuntimeError("campaign report lacks run_context_id")
    if expected_run_context is not None and run_context != expected_run_context:
        raise RuntimeError(
            f"campaign run_context_id {run_context!r} != expected {expected_run_context!r}"
        )
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("campaign report has no candidates")
    candidate_reports = report.get("candidate_reports")
    if not isinstance(candidate_reports, list) or len(candidate_reports) != len(candidates):
        raise RuntimeError("campaign candidate report count drifted")
    return run_context


def campaign_path(run_dir: Path, repeat: int) -> Path:
    return run_dir / f"campaign_{repeat:02d}.json"


def load_resumable_campaign(path: Path, head: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        validate_campaign(report, expected_head=head)
        return report
    except (OSError, json.JSONDecodeError, RuntimeError):
        return None


def run_campaign(
    binary: Path,
    fixture: Path,
    args: argparse.Namespace,
    run_context: str,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["NNIS_BENCH_RUN_CONTEXT_ID"] = run_context
    environment["NNIS_BENCH_ENVIRONMENT_LABEL"] = args.environment_label
    command = [
        str(binary),
        "--model",
        str(fixture / "model"),
        "--suite",
        str(fixture / "reference_suite.json"),
        "--device",
        str(args.device),
        "--rounds",
        str(args.rounds),
        "--warmups",
        str(args.warmups),
        "--iterations",
        str(args.iterations),
        "--candidates",
        args.candidates,
    ]
    raw = run_text(command, env=environment)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"benchmark emitted invalid JSON: {error}") from error


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "observations": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "rounds_at_or_above_3pct": 0,
            "median_relative_improvement": None,
            "mean_relative_improvement": None,
            "min_relative_improvement": None,
            "max_relative_improvement": None,
            "sample_stdev_relative_improvement": None,
        }
    wins = sum(value > 0.0 for value in values)
    losses = sum(value < 0.0 for value in values)
    ties = len(values) - wins - losses
    return {
        "observations": len(values),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "rounds_at_or_above_3pct": sum(value >= 0.03 for value in values),
        "median_relative_improvement": statistics.median(values),
        "mean_relative_improvement": statistics.fmean(values),
        "min_relative_improvement": min(values),
        "max_relative_improvement": max(values),
        "sample_stdev_relative_improvement": statistics.stdev(values)
        if len(values) > 1
        else 0.0,
    }


def collect_case_observations(campaigns: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, list[float]]]:
    result: dict[tuple[str, str], dict[str, list[float]]] = {}
    for report in campaigns:
        for candidate_report in report["candidate_reports"]:
            candidate = candidate_report["candidate"]
            for round_report in candidate_report["rounds"]:
                for evidence in round_report["case_evidence"]:
                    key = (candidate, evidence["case_name"])
                    cell = result.setdefault(key, {"gpu": [], "wall": [], "request": []})
                    for output_key, source_key in [
                        ("gpu", "generation_stage_gpu_relative_improvement"),
                        ("wall", "generation_wall_relative_improvement"),
                        ("request", "request_total_wall_relative_improvement"),
                    ]:
                        value = finite_number(evidence.get(source_key))
                        if value is not None:
                            cell[output_key].append(value)
    return result


def build_consensus(
    campaigns: list[dict[str, Any]],
    head: str,
    run_dir: Path,
) -> dict[str, Any]:
    if not campaigns:
        raise RuntimeError("no campaigns available for consensus")
    contexts = [validate_campaign(report, expected_head=head) for report in campaigns]
    if len(set(contexts)) != len(contexts):
        raise RuntimeError("independent campaign run_context_id values are not distinct")

    environments = [normalized_environment(report["metadata"]) for report in campaigns]
    environment_compatible = all(environment == environments[0] for environment in environments[1:])
    exact_git_commit_equal = all(report["metadata"]["git_commit"] == head for report in campaigns)
    all_campaigns_complete = all(report.get("campaign_complete") is True for report in campaigns)

    cells = collect_case_observations(campaigns)
    case_summaries = []
    candidate_aggregate: dict[str, dict[str, list[float]]] = {}
    for (candidate, case_name), metrics in sorted(cells.items()):
        case_summaries.append(
            {
                "candidate": candidate,
                "case_name": case_name,
                "generation_stage_gpu": summarize_values(metrics["gpu"]),
                "generation_wall": summarize_values(metrics["wall"]),
                "request_total_wall": summarize_values(metrics["request"]),
            }
        )
        aggregate = candidate_aggregate.setdefault(candidate, {"gpu": [], "wall": [], "request": []})
        for metric in aggregate:
            aggregate[metric].extend(metrics[metric])

    aggregate_summaries = [
        {
            "candidate": candidate,
            "note": "descriptive aggregate across heterogeneous prompt/decode profiles; do not interpret as a workload-weighted serving speedup",
            "generation_stage_gpu": summarize_values(metrics["gpu"]),
            "generation_wall": summarize_values(metrics["wall"]),
            "request_total_wall": summarize_values(metrics["request"]),
        }
        for candidate, metrics in sorted(candidate_aggregate.items())
    ]

    consensus_valid = (
        len(campaigns) >= 2
        and environment_compatible
        and exact_git_commit_equal
        and all_campaigns_complete
    )
    return {
        "schema_version": 1,
        "kind": "nnis-tinyllama-trained-massive-f16-consensus-v1",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "git_commit": head,
        "campaign_count": len(campaigns),
        "run_context_ids": contexts,
        "environment_compatible_across_distinct_campaigns": environment_compatible,
        "exact_git_commit_equal": exact_git_commit_equal,
        "all_campaigns_complete": all_campaigns_complete,
        "consensus_valid": consensus_valid,
        "promotion_authorized": False,
        "claim_boundary": (
            "cross-model trained TinyLlama exploratory qualification for the exact pinned model, "
            "suite, F16 numeric contract, candidate plans, and compatible physical environment only; "
            "no default runtime change or general Llama-family support is authorized"
        ),
        "case_summaries": case_summaries,
        "candidate_aggregate_descriptive_only": aggregate_summaries,
        "raw_campaign_files": [str(campaign_path(run_dir, index + 1)) for index in range(len(campaigns))],
    }


def pct(value: Any) -> str:
    numeric = finite_number(value)
    return "n/a" if numeric is None else f"{numeric * 100.0:.3f}%"


def write_markdown_summary(consensus: dict[str, Any], path: Path) -> None:
    lines = [
        "# TinyLlama trained massive F16 campaign",
        "",
        f"- model: `{SOURCE_REPO}@{SOURCE_REVISION}`",
        f"- model SHA256: `{SOURCE_MODEL_SHA256}`",
        f"- Git commit: `{consensus['git_commit']}`",
        f"- independent campaigns: {consensus['campaign_count']}",
        f"- environment compatible: `{str(consensus['environment_compatible_across_distinct_campaigns']).lower()}`",
        f"- all campaigns complete: `{str(consensus['all_campaigns_complete']).lower()}`",
        f"- consensus valid: `{str(consensus['consensus_valid']).lower()}`",
        "- promotion authorized: `false`",
        "",
        "## Descriptive candidate aggregate",
        "",
        "These aggregates mix heterogeneous prompt/decode profiles and are descriptive only.",
        "",
        "| Candidate | GPU median improvement | GPU wins/obs | Request median improvement |",
        "|---|---:|---:|---:|",
    ]
    for item in consensus["candidate_aggregate_descriptive_only"]:
        gpu = item["generation_stage_gpu"]
        request = item["request_total_wall"]
        lines.append(
            f"| {item['candidate']} | {pct(gpu['median_relative_improvement'])} | "
            f"{gpu['wins']}/{gpu['observations']} | {pct(request['median_relative_improvement'])} |"
        )
    lines.extend(["", "## Claim boundary", "", consensus["claim_boundary"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = repo_root()
    head, dirty = repository_identity(root)
    if dirty:
        raise RuntimeError(
            "tracked worktree is dirty; commit or restore changes before a physical qualification campaign"
        )

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    fixture = ensure_fixture(root, work_dir, args.cache_dir, args.force_fixture)
    binary = build_benchmark(root)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = work_dir / f"runs-{head[:12]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    campaigns: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for repeat in range(1, args.repeats + 1):
        path = campaign_path(run_dir, repeat)
        existing = None if args.no_resume else load_resumable_campaign(path, head)
        if existing is not None:
            campaigns.append(existing)
            continue
        run_context = f"tinyllama-massive-{head[:12]}-{timestamp}-r{repeat:02d}"
        try:
            report = run_campaign(binary, fixture, args, run_context)
            validate_campaign(report, expected_head=head, expected_run_context=run_context)
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            campaigns.append(report)
        except Exception as error:
            failures.append({"repeat": repeat, "run_context_id": run_context, "error": str(error)})

    if failures:
        (run_dir / "launcher_failures.json").write_text(
            json.dumps({"failures": failures}, indent=2) + "\n", encoding="utf-8"
        )
    if not campaigns:
        raise RuntimeError(f"all campaign repeats failed; details: {run_dir / 'launcher_failures.json'}")

    consensus = build_consensus(campaigns, head, run_dir)
    if failures:
        consensus["launcher_failures"] = failures
        consensus["consensus_valid"] = False
    consensus_path = run_dir / "consensus.json"
    consensus_path.write_text(json.dumps(consensus, indent=2) + "\n", encoding="utf-8")
    summary_path = run_dir / "SUMMARY.md"
    write_markdown_summary(consensus, summary_path)

    print(f"fixture={fixture}")
    print(f"run_dir={run_dir}")
    print(f"consensus={consensus_path}")
    print(f"summary={summary_path}")
    print(f"campaigns_completed={len(campaigns)}/{args.repeats}")
    print(f"consensus_valid={str(consensus['consensus_valid']).lower()}")
    if not consensus["consensus_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
