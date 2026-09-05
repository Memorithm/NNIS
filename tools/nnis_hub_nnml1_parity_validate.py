#!/usr/bin/env python3
"""Stable process surface for validating NNML1 exact-checkpoint parity evidence.

This wrapper deliberately delegates all parity semantics to
``validate_nnml1_multi_model_parity_evidence``. It adds only a bounded,
deterministic process boundary suitable for orchestration systems such as
SciRust Hub. It does not execute CUDA, authorize model/runtime promotion, or
turn checkpoint-level parity evidence into serving/performance claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import validate_nnml1_multi_model_parity_evidence as parity

SCHEMA_VERSION = 1
CONTRACT = "nnis.nnml1.parity-validation@1.0.0"
MEDIA_TYPE = "application/vnd.nnis.nnml1.parity-validation.v1+json"
INPUT_MEDIA_TYPE = "application/vnd.nnis.nnml1.parity-evidence.v1+json"
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


class ProcessContractError(ValueError):
    pass


def _read_bounded_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        stat = path.stat()
    except OSError as error:
        raise ProcessContractError(f"cannot stat evidence {path}: {error}") from error
    if not path.is_file():
        raise ProcessContractError(f"evidence path is not a regular file: {path}")
    if stat.st_size <= 0 or stat.st_size > MAX_EVIDENCE_BYTES:
        raise ProcessContractError(
            f"evidence size must be 1..={MAX_EVIDENCE_BYTES} bytes; got {stat.st_size}"
        )
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ProcessContractError(f"cannot read evidence {path}: {error}") from error
    if len(raw) != stat.st_size:
        raise ProcessContractError("evidence size changed while reading")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProcessContractError(f"invalid UTF-8 JSON evidence: {error}") from error
    if not isinstance(value, dict):
        raise ProcessContractError("evidence root must be a JSON object")
    return raw, value


def _validated_summary(raw: bytes, document: dict[str, Any]) -> dict[str, Any]:
    kind = document.get("kind")
    if kind == parity.RECORD_KIND:
        validated = parity.validate_record(document)
        records = [validated]
        distinct_checkpoint_count = 1
    elif kind == parity.SUITE_KIND:
        validated = parity.validate_suite(document)
        records = validated["records"]
        distinct_checkpoint_count = validated["distinct_checkpoint_count"]
    else:
        raise ProcessContractError(f"unsupported evidence kind: {kind!r}")

    checkpoint_specs = sorted({record["checkpoint_spec_name"] for record in records})
    parity_levels = sorted({record["parity_level"] for record in records})
    reference_runtimes = sorted(
        {f"{record['reference_runtime']}@{record['reference_runtime_version']}" for record in records}
    )
    execution_backends = sorted({record["execution_backend"] for record in records})
    execution_commits = sorted({record["execution_git_commit"] for record in records})
    if len(execution_commits) != 1:
        raise ProcessContractError("validated evidence must resolve to one execution Git commit")

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "media_type": MEDIA_TYPE,
        "status": "validated",
        "validation_scope": "nnml1_exact_checkpoint_parity_contract_only",
        "source": {
            "media_type": INPUT_MEDIA_TYPE,
            "kind": kind,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "execution_git_commit": execution_commits[0],
        "distinct_checkpoint_count": distinct_checkpoint_count,
        "checkpoint_specs": checkpoint_specs,
        "parity_levels": parity_levels,
        "reference_runtimes": reference_runtimes,
        "execution_backends": execution_backends,
        "promotion_authorized": False,
        "serving_performance_verified": False,
        "general_model_family_support_verified": False,
        "claim_boundary": (
            "NNIS exact-checkpoint parity evidence contract validation only; this result does not "
            "authorize runtime/model-family promotion, establish general model-family support, or "
            "establish serving performance"
        ),
    }


def validate_path(path: Path) -> dict[str, Any]:
    raw, document = _read_bounded_json(path)
    try:
        return _validated_summary(raw, document)
    except parity.EvidenceError as error:
        raise ProcessContractError(str(error)) from error


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ProcessContractError(f"result path already exists: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        raise ProcessContractError(f"result path already exists: {path}") from error
    except OSError as error:
        raise RuntimeError(f"cannot write result {path}: {error}") from error


def self_test() -> None:
    commit = "c" * 40
    smollm2 = parity._synthetic_record(  # noqa: SLF001 - same-repository contract fixture
        "smollm2-135m-bf16", level=parity.LOGIT_AND_GENERATION, commit=commit
    )
    tinyllama = parity._synthetic_record(  # noqa: SLF001 - same-repository contract fixture
        "tinyllama-1.1b-chat-v1.0-bf16",
        level=parity.GENERATION_TRAJECTORY,
        commit=commit,
    )
    suite = parity.build_suite([smollm2, tinyllama])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        evidence = root / "suite.json"
        result = root / "result.json"
        evidence.write_text(json.dumps(suite, sort_keys=True) + "\n", encoding="utf-8")
        summary = validate_path(evidence)
        assert summary["status"] == "validated"
        assert summary["distinct_checkpoint_count"] == 2
        assert summary["execution_git_commit"] == commit
        assert summary["promotion_authorized"] is False
        _write_new_json(result, summary)
        reparsed = json.loads(result.read_text(encoding="utf-8"))
        assert reparsed == summary

        tampered = dict(suite)
        tampered["promotion_authorized"] = True
        bad = root / "tampered.json"
        bad.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            validate_path(bad)
        except ProcessContractError:
            pass
        else:
            raise AssertionError("promotion-authorizing evidence unexpectedly validated")

        oversized = root / "oversized.json"
        with oversized.open("wb") as handle:
            handle.truncate(MAX_EVIDENCE_BYTES + 1)
        try:
            validate_path(oversized)
        except ProcessContractError:
            pass
        else:
            raise AssertionError("oversized evidence unexpectedly validated")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("NNML1 Hub parity validation process self-test passed")
        return 0
    if args.evidence is None or args.result is None:
        parser.error("--evidence and --result are required unless --self-test is used")

    try:
        summary = validate_path(args.evidence)
        _write_new_json(args.result, summary)
    except ProcessContractError as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2
    except Exception as error:  # fail closed on unexpected process/runtime errors
        print(f"internal error: {error}", file=os.sys.stderr)
        return 3

    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
