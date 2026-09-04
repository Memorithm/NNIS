#!/usr/bin/env python3
"""Validate versioned NNML1 multi-model reference-parity evidence.

This module is deliberately standard-library only. It validates evidence identity,
checkpoint/tokenizer provenance, semantic trajectory gates, optional strict logit
parity, and same-head multi-model composition. It does not execute CUDA or infer
model-family support from individual checkpoint records.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

RECORD_KIND = "nnis-nnml1-reference-parity-record-v1"
SUITE_KIND = "nnis-nnml1-multi-model-parity-suite-v1"
SCHEMA_VERSION = 1
CHECKPOINT_SPEC_VERSION = 1
GENERATION_TRAJECTORY = "generation_trajectory"
LOGIT_AND_GENERATION = "logit_and_generation"
PARITY_LEVELS = {GENERATION_TRAJECTORY, LOGIT_AND_GENERATION}

KNOWN_CHECKPOINTS = {
    "smollm2-135m-bf16": {
        "source_repo": "HuggingFaceTB/SmolLM2-135M",
        "source_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "source_model_sha256": "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
    },
    "tinyllama-1.1b-chat-v1.0-bf16": {
        "source_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "source_revision": "d9128824c0c80111be21424e68086f52413fb413",
        "source_model_sha256": "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
    },
}

_HASH64 = re.compile(r"^[0-9a-f]{64}$")
_GIT40 = re.compile(r"^[0-9a-f]{40}$")


class EvidenceError(ValueError):
    pass


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object")
    return value


def _require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{label} must be a non-empty string")
    return value


def _require_non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    value = _require_non_negative_int(value, label)
    if value == 0:
        raise EvidenceError(f"{label} must be greater than zero")
    return value


def _require_finite_non_negative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise EvidenceError(f"{label} must be finite and non-negative")
    return numeric


def _validate_checkpoint_identity(record: dict[str, Any]) -> None:
    spec_name = _require_non_empty_string(record.get("checkpoint_spec_name"), "checkpoint_spec_name")
    if record.get("checkpoint_spec_version") != CHECKPOINT_SPEC_VERSION:
        raise EvidenceError("unexpected checkpoint_spec_version")
    expected = KNOWN_CHECKPOINTS.get(spec_name)
    if expected is None:
        raise EvidenceError(f"unknown exact checkpoint spec {spec_name!r}")
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise EvidenceError(
                f"{field} does not match exact checkpoint spec {spec_name}: "
                f"{record.get(field)!r} != {expected_value!r}"
            )
    tokenizer_sha = record.get("tokenizer_sha256")
    if not isinstance(tokenizer_sha, str) or _HASH64.fullmatch(tokenizer_sha) is None:
        raise EvidenceError("tokenizer_sha256 must be a lowercase 64-hex digest")


def _validate_semantic(semantic: Any) -> None:
    semantic = _require_dict(semantic, "semantic")
    case_count = _require_positive_int(semantic.get("case_count"), "semantic.case_count")
    observations = _require_positive_int(semantic.get("observations"), "semantic.observations")
    exact = _require_non_negative_int(
        semantic.get("exact_greedy_observations"), "semantic.exact_greedy_observations"
    )
    if observations < case_count:
        raise EvidenceError("semantic.observations must cover at least every reference case")
    if semantic.get("exact_greedy_all") is not True:
        raise EvidenceError("semantic.exact_greedy_all must be true for qualifying evidence")
    if exact != observations:
        raise EvidenceError(
            "semantic.exact_greedy_observations must equal semantic.observations"
        )


def _validate_logits(logits: Any) -> None:
    logits = _require_dict(logits, "logits")
    _require_finite_non_negative(logits.get("atol"), "logits.atol")
    _require_finite_non_negative(logits.get("rtol"), "logits.rtol")
    _require_positive_int(logits.get("stages"), "logits.stages")
    failures = _require_non_negative_int(logits.get("failures"), "logits.failures")
    non_finite = _require_non_negative_int(logits.get("non_finite"), "logits.non_finite")
    _require_finite_non_negative(logits.get("max_abs"), "logits.max_abs")
    _require_finite_non_negative(logits.get("max_rms"), "logits.max_rms")
    if failures != 0:
        raise EvidenceError("logits.failures must be zero for logit_and_generation evidence")
    if non_finite != 0:
        raise EvidenceError("logits.non_finite must be zero for logit_and_generation evidence")
    if logits.get("strict_tolerance_asserted") is not True:
        raise EvidenceError("logits.strict_tolerance_asserted must be true")


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    record = _require_dict(record, "record")
    if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != RECORD_KIND:
        raise EvidenceError("record schema_version/kind mismatch")
    _validate_checkpoint_identity(record)

    reference_runtime = _require_non_empty_string(
        record.get("reference_runtime"), "reference_runtime"
    )
    _require_non_empty_string(record.get("reference_runtime_version"), "reference_runtime_version")
    if reference_runtime != "transformers":
        raise EvidenceError("reference_runtime must be transformers for v1 evidence")

    git_commit = record.get("execution_git_commit")
    if not isinstance(git_commit, str) or _GIT40.fullmatch(git_commit) is None:
        raise EvidenceError("execution_git_commit must be a lowercase 40-hex commit")
    if record.get("execution_git_dirty") is not False:
        raise EvidenceError("qualifying parity evidence requires a clean tracked worktree")
    _require_non_empty_string(record.get("execution_backend"), "execution_backend")

    parity_level = record.get("parity_level")
    if parity_level not in PARITY_LEVELS:
        raise EvidenceError(f"unexpected parity_level {parity_level!r}")
    _validate_semantic(record.get("semantic"))
    if parity_level == LOGIT_AND_GENERATION:
        _validate_logits(record.get("logits"))
    elif record.get("logits") is not None:
        raise EvidenceError("generation_trajectory evidence must set logits to null")

    source_evidence = _require_dict(record.get("source_evidence"), "source_evidence")
    _require_non_empty_string(source_evidence.get("kind"), "source_evidence.kind")
    _require_non_empty_string(source_evidence.get("artifact"), "source_evidence.artifact")
    if record.get("promotion_authorized") is not False:
        raise EvidenceError("parity evidence must not authorize runtime/model-family promotion")
    _require_non_empty_string(record.get("claim_boundary"), "claim_boundary")
    return record


def build_record(
    *,
    checkpoint_spec_name: str,
    source_repo: str,
    source_revision: str,
    source_model_sha256: str,
    tokenizer_sha256: str,
    reference_runtime: str,
    reference_runtime_version: str,
    execution_git_commit: str,
    execution_backend: str,
    parity_level: str,
    case_count: int,
    observations: int,
    exact_greedy_observations: int,
    source_evidence_kind: str,
    source_evidence_artifact: str,
    logits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECORD_KIND,
        "checkpoint_spec_name": checkpoint_spec_name,
        "checkpoint_spec_version": CHECKPOINT_SPEC_VERSION,
        "source_repo": source_repo,
        "source_revision": source_revision,
        "source_model_sha256": source_model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "reference_runtime": reference_runtime,
        "reference_runtime_version": reference_runtime_version,
        "execution_git_commit": execution_git_commit,
        "execution_git_dirty": False,
        "execution_backend": execution_backend,
        "parity_level": parity_level,
        "semantic": {
            "case_count": case_count,
            "observations": observations,
            "exact_greedy_observations": exact_greedy_observations,
            "exact_greedy_all": observations > 0 and exact_greedy_observations == observations,
        },
        "logits": logits,
        "source_evidence": {
            "kind": source_evidence_kind,
            "artifact": source_evidence_artifact,
        },
        "promotion_authorized": False,
        "claim_boundary": (
            "exact-checkpoint reference parity evidence only; this record does not establish "
            "general model-family support, serving performance, or automatic runtime promotion"
        ),
    }
    return validate_record(record)


def validate_suite(suite: dict[str, Any]) -> dict[str, Any]:
    suite = _require_dict(suite, "suite")
    if suite.get("schema_version") != SCHEMA_VERSION or suite.get("kind") != SUITE_KIND:
        raise EvidenceError("suite schema_version/kind mismatch")
    git_commit = suite.get("execution_git_commit")
    if not isinstance(git_commit, str) or _GIT40.fullmatch(git_commit) is None:
        raise EvidenceError("suite execution_git_commit must be a lowercase 40-hex commit")
    records = suite.get("records")
    if not isinstance(records, list) or len(records) < 2:
        raise EvidenceError("multi-model parity suite requires at least two records")
    validated = [validate_record(record) for record in records]
    if any(record["execution_git_commit"] != git_commit for record in validated):
        raise EvidenceError("every parity record must come from the suite execution_git_commit")
    names = [record["checkpoint_spec_name"] for record in validated]
    if len(set(names)) < 2:
        raise EvidenceError("multi-model parity suite requires at least two distinct checkpoints")
    if suite.get("distinct_checkpoint_count") != len(set(names)):
        raise EvidenceError("suite distinct_checkpoint_count mismatch")
    if suite.get("promotion_authorized") is not False:
        raise EvidenceError("multi-model parity suite must not authorize model-family promotion")
    _require_non_empty_string(suite.get("claim_boundary"), "suite.claim_boundary")
    return suite


def build_suite(records: list[dict[str, Any]]) -> dict[str, Any]:
    validated = [validate_record(record) for record in records]
    if not validated:
        raise EvidenceError("cannot build an empty parity suite")
    commits = {record["execution_git_commit"] for record in validated}
    if len(commits) != 1:
        raise EvidenceError("cannot compose parity records from different Git commits")
    names = {record["checkpoint_spec_name"] for record in validated}
    suite = {
        "schema_version": SCHEMA_VERSION,
        "kind": SUITE_KIND,
        "execution_git_commit": next(iter(commits)),
        "distinct_checkpoint_count": len(names),
        "records": validated,
        "promotion_authorized": False,
        "claim_boundary": (
            "same-head exact-checkpoint parity composition only; multiple passing checkpoints do not "
            "constitute a general model-family admission or performance claim"
        ),
    }
    return validate_suite(suite)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EvidenceError(f"cannot read evidence {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise EvidenceError(f"invalid evidence JSON {path}: {error}") from error
    return _require_dict(value, str(path))


def _synthetic_record(spec_name: str, *, level: str, commit: str) -> dict[str, Any]:
    spec = KNOWN_CHECKPOINTS[spec_name]
    logits = None
    if level == LOGIT_AND_GENERATION:
        logits = {
            "atol": 1.0e-4,
            "rtol": 1.0e-3,
            "stages": 3,
            "failures": 0,
            "non_finite": 0,
            "max_abs": 2.0e-5,
            "max_rms": 3.0e-6,
            "strict_tolerance_asserted": True,
        }
    return build_record(
        checkpoint_spec_name=spec_name,
        source_repo=spec["source_repo"],
        source_revision=spec["source_revision"],
        source_model_sha256=spec["source_model_sha256"],
        tokenizer_sha256="a" * 64 if spec_name.startswith("smollm2") else "b" * 64,
        reference_runtime="transformers",
        reference_runtime_version="4.40.1" if spec_name.startswith("smollm2") else "4.43.3",
        execution_git_commit=commit,
        execution_backend="nnis-cuda",
        parity_level=level,
        case_count=1 if spec_name.startswith("smollm2") else 18,
        observations=1 if spec_name.startswith("smollm2") else 144,
        exact_greedy_observations=1 if spec_name.startswith("smollm2") else 144,
        source_evidence_kind="synthetic-self-test",
        source_evidence_artifact="self-test.json",
        logits=logits,
    )


def self_test() -> None:
    commit = "c" * 40
    smollm2 = _synthetic_record("smollm2-135m-bf16", level=LOGIT_AND_GENERATION, commit=commit)
    tinyllama = _synthetic_record(
        "tinyllama-1.1b-chat-v1.0-bf16", level=GENERATION_TRAJECTORY, commit=commit
    )
    validate_suite(build_suite([smollm2, tinyllama]))

    negatives: list[tuple[str, dict[str, Any], Any]] = []
    drifted_sha = copy.deepcopy(tinyllama)
    drifted_sha["source_model_sha256"] = "0" * 64
    negatives.append(("checkpoint SHA drift", drifted_sha, validate_record))

    dirty = copy.deepcopy(tinyllama)
    dirty["execution_git_dirty"] = True
    negatives.append(("dirty evidence", dirty, validate_record))

    semantic_failure = copy.deepcopy(tinyllama)
    semantic_failure["semantic"]["exact_greedy_all"] = False
    negatives.append(("semantic mismatch", semantic_failure, validate_record))

    logit_failure = copy.deepcopy(smollm2)
    logit_failure["logits"]["failures"] = 1
    negatives.append(("logit mismatch", logit_failure, validate_record))

    for label, document, validator in negatives:
        try:
            validator(document)
        except EvidenceError:
            pass
        else:
            raise AssertionError(f"negative self-test unexpectedly passed: {label}")

    duplicate_suite = {
        "schema_version": SCHEMA_VERSION,
        "kind": SUITE_KIND,
        "execution_git_commit": commit,
        "distinct_checkpoint_count": 1,
        "records": [smollm2, copy.deepcopy(smollm2)],
        "promotion_authorized": False,
        "claim_boundary": "self-test",
    }
    try:
        validate_suite(duplicate_suite)
    except EvidenceError:
        pass
    else:
        raise AssertionError("duplicate-checkpoint suite unexpectedly passed")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "suite.json"
        path.write_text(json.dumps(build_suite([smollm2, tinyllama]), indent=2) + "\n")
        validate_suite(_load_json(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--compose", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("NNML1 multi-model parity evidence self-test passed")
    if args.compose is not None:
        records = [_load_json(path) for path in args.paths]
        suite = build_suite(records)
        args.compose.parent.mkdir(parents=True, exist_ok=True)
        args.compose.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
        print(f"composed={args.compose}")
        return
    for path in args.paths:
        document = _load_json(path)
        if document.get("kind") == RECORD_KIND:
            validate_record(document)
        elif document.get("kind") == SUITE_KIND:
            validate_suite(document)
        else:
            raise EvidenceError(f"unsupported evidence kind in {path}: {document.get('kind')!r}")
        print(f"validated={path}")


if __name__ == "__main__":
    main()
