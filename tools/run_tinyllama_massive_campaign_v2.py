#!/usr/bin/env python3
"""Run the TinyLlama massive campaign with the v2 F16 oracle and fail-closed repeats."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import run_tinyllama_massive_campaign as base
from validate_tinyllama_reference_artifact_v2 import validate_fixture

_original_validate_campaign = base.validate_campaign
_original_collect_semantic_validation = base.collect_semantic_validation


def validate_complete_campaign(
    report: dict[str, Any],
    *,
    expected_head: str,
    expected_run_context: str | None = None,
) -> str:
    run_context = _original_validate_campaign(
        report,
        expected_head=expected_head,
        expected_run_context=expected_run_context,
    )
    if report.get("campaign_complete") is not True:
        raise RuntimeError("campaign_complete must be true before a repeat is accepted or resumed")

    semantic = _original_collect_semantic_validation([report])
    if semantic.get("all_exact_oracle_greedy") is not True:
        raise RuntimeError(
            "campaign contains failed, incomplete, or non-exact greedy observations"
        )

    case_count = report.get("case_count")
    if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count <= 0:
        raise RuntimeError("campaign case_count is invalid")

    for candidate_report in report.get("candidate_reports", []):
        for round_report in candidate_report.get("rounds", []):
            evidence = round_report.get("case_evidence")
            if not isinstance(evidence, list) or len(evidence) != case_count:
                raise RuntimeError("campaign round case_evidence does not cover every case")
            if any(not isinstance(item, dict) or item.get("complete") is not True for item in evidence):
                raise RuntimeError("campaign contains incomplete round case_evidence")

    return run_context


def ensure_fixture(
    root: Path,
    work_dir: Path,
    cache_dir: Path | None,
    force: bool,
) -> Path:
    fixture = work_dir / "fixture"
    if not force:
        try:
            validate_fixture(fixture)
            return fixture
        except Exception:
            pass

    command = [
        sys.executable,
        str(root / "tools" / "tinyllama_1p1b_chat_fixture_v2.py"),
        "--output",
        str(fixture),
    ]
    if cache_dir is not None:
        command.extend(["--cache-dir", str(cache_dir)])
    base.run_text(command)
    validate_fixture(fixture)
    return fixture


def _self_test() -> None:
    good = {
        "schema_version": 1,
        "benchmark": base.BENCHMARK_KIND,
        "source_repo": base.SOURCE_REPO,
        "source_revision": base.SOURCE_REVISION,
        "source_model_sha256": base.SOURCE_MODEL_SHA256,
        "metadata": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "environment_fingerprint": {"run_context_id": "self-test"},
        },
        "candidates": ["transposed"],
        "rounds_per_candidate": 1,
        "case_count": 1,
        "campaign_complete": True,
        "candidate_reports": [
            {
                "candidate": "transposed",
                "rounds": [
                    {
                        "round": 0,
                        "blocks": [
                            {
                                "slot": slot,
                                "cases": [
                                    {
                                        "success": True,
                                        "exact_oracle_greedy": True,
                                        "generated_ids": [7, 8],
                                        "decode_steps": 2,
                                    }
                                ],
                            }
                            for slot in ("A1", "B1", "B2", "A2")
                        ],
                        "case_evidence": [{"case_name": "synthetic", "complete": True}],
                    }
                ],
            }
        ],
    }
    validate_complete_campaign(good, expected_head="a" * 40, expected_run_context="self-test")

    for mutator in (
        lambda d: d.__setitem__("campaign_complete", False),
        lambda d: d["candidate_reports"][0]["rounds"][0]["blocks"][0]["cases"][0].__setitem__("success", False),
        lambda d: d["candidate_reports"][0]["rounds"][0]["case_evidence"][0].__setitem__("complete", False),
    ):
        bad = json.loads(json.dumps(good))
        mutator(bad)
        try:
            validate_complete_campaign(bad, expected_head="a" * 40, expected_run_context="self-test")
        except RuntimeError:
            continue
        raise AssertionError("fail-closed negative self-test unexpectedly passed")
    print("TinyLlama massive campaign v2 fail-closed self-test passed")


def main() -> None:
    if sys.argv[1:] == ["--self-test-v2"]:
        _self_test()
        return

    base.validate_fixture = validate_fixture
    base.validate_campaign = validate_complete_campaign
    base.ensure_fixture = ensure_fixture
    base.main()


if __name__ == "__main__":
    main()
