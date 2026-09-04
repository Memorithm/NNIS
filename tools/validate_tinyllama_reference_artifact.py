#!/usr/bin/env python3
"""Validate the exact pinned TinyLlama-1.1B NNIS reference artifact.

The validator is deliberately standard-library only. It validates artifact
identity and schema consistency; it does not execute Transformers, CUDA, or
claim hardware qualification.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

from tokenizer_reference_identity import (
    IdentityError,
    build_identity,
    validate_documents,
    validate_identity,
)

SOURCE_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SOURCE_REVISION = "d9128824c0c80111be21424e68086f52413fb413"
SOURCE_MODEL_SHA256 = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
CHECKPOINT_SPEC_NAME = "tinyllama-1.1b-chat-v1.0-bf16"
CHECKPOINT_SPEC_VERSION = 1
SUITE_KIND = "nnis-trained-llama-reference-suite-v1"
REFERENCE_RUNTIME = "transformers"
REFERENCE_TRANSFORMERS_VERSION = "4.43.3"
REFERENCE_ORACLE_SEMANTICS = (
    "Transformers CPU F32 greedy generation from the exact pinned checkpoint widened to F32"
)
IDENTITY_FILENAME = "tokenizer_reference_identity.json"
SOURCE_WEIGHT_DTYPE = "bfloat16"
EXECUTION_WEIGHT_DTYPE = "f32"
REFERENCE_ENV = {
    "MKL_CBWR": "COMPATIBLE",
    "MKL_DYNAMIC": "FALSE",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}
REFERENCE_EXECUTION_POLICY = {
    "torch_deterministic_algorithms": True,
    "torch_manual_seed": 0,
    "torch_num_threads": 1,
    "torch_num_interop_threads": 1,
    "mkl_cbwr": "COMPATIBLE",
    "mkl_dynamic": False,
    "omp_dynamic": False,
    "mkl_num_threads": 1,
    "omp_num_threads": 1,
}
EXPECTED_CONFIG = {
    "vocab_size": 32000,
    "eos_token_id": 2,
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "num_hidden_layers": 22,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "activation": "silu",
    "weight_dtype": "f32",
}
PROMPT_FAMILIES = ("prose", "code", "math")
TARGET_PROMPT_TOKENS = (8, 32, 128, 512, 1024)
STANDARD_DECODE_STEPS = 32
DEEP_DECODE_STEPS = 128


class ArtifactError(ValueError):
    pass


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ArtifactError(f"cannot read {label} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ArtifactError(f"invalid {label} JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactError(f"{label} must be a JSON object")
    return value


def _expected_case_names() -> set[str]:
    names: set[str] = set()
    for family in PROMPT_FAMILIES:
        for target in TARGET_PROMPT_TOKENS:
            names.add(f"{family}-p{target:04d}-d{STANDARD_DECODE_STEPS:03d}")
            if target == 32:
                names.add(f"{family}-p{target:04d}-d{DEEP_DECODE_STEPS:03d}")
    return names


def _validate_cases(cases: object) -> None:
    if not isinstance(cases, list):
        raise ArtifactError("reference suite cases must be a list")
    expected_names = _expected_case_names()
    if len(cases) != len(expected_names):
        raise ArtifactError(
            f"reference suite case count {len(cases)} != expected {len(expected_names)}"
        )
    names: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ArtifactError(f"case {index} must be an object")
        name = case.get("name")
        family = case.get("family")
        target = case.get("target_prompt_tokens")
        decode_steps = case.get("decode_steps")
        input_ids = case.get("input_ids")
        greedy_ids = case.get("greedy_ids")
        if not isinstance(name, str) or not name or name in names:
            raise ArtifactError(f"case {index} has empty or duplicated name {name!r}")
        names.add(name)
        if family not in PROMPT_FAMILIES:
            raise ArtifactError(f"case {name!r} has unexpected family {family!r}")
        if target not in TARGET_PROMPT_TOKENS:
            raise ArtifactError(f"case {name!r} has unexpected prompt length {target!r}")
        expected_decode = (
            DEEP_DECODE_STEPS if target == 32 and name.endswith("-d128") else STANDARD_DECODE_STEPS
        )
        if decode_steps != expected_decode:
            raise ArtifactError(
                f"case {name!r} decode_steps={decode_steps!r}, expected {expected_decode}"
            )
        if not isinstance(input_ids, list) or len(input_ids) != target:
            raise ArtifactError(f"case {name!r} input_ids length does not match target")
        if not isinstance(greedy_ids, list) or len(greedy_ids) != decode_steps:
            raise ArtifactError(f"case {name!r} greedy_ids length does not match decode_steps")
        tokens = [*input_ids, *greedy_ids]
        if any(isinstance(token, bool) or not isinstance(token, int) for token in tokens):
            raise ArtifactError(f"case {name!r} contains a non-integer token")
        if any(token < 0 or token >= EXPECTED_CONFIG["vocab_size"] for token in tokens):
            raise ArtifactError(f"case {name!r} contains a token outside the vocabulary")
        required_positions = target + decode_steps - 1
        if required_positions > EXPECTED_CONFIG["max_position_embeddings"]:
            raise ArtifactError(
                f"case {name!r} requires {required_positions} positions, exceeding the checkpoint"
            )
    if names != expected_names:
        raise ArtifactError(
            f"reference suite case names mismatch; missing={sorted(expected_names - names)}, "
            f"extra={sorted(names - expected_names)}"
        )


def validate_fixture(fixture: Path) -> dict:
    fixture = fixture.resolve()
    tokenizer_path = fixture / "tokenizer.json"
    identity_path = fixture / IDENTITY_FILENAME
    suite_path = fixture / "reference_suite.json"
    provenance_path = fixture / "model" / "provenance.json"
    model_path = fixture / "model" / "model.json"
    required = (tokenizer_path, identity_path, suite_path, provenance_path, model_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ArtifactError(f"TinyLlama reference artifact is incomplete; missing={missing}")

    identity = _load_json(identity_path, "identity")
    suite = _load_json(suite_path, "reference suite")
    provenance = _load_json(provenance_path, "model provenance")
    model = _load_json(model_path, "model manifest")

    try:
        validate_identity(
            identity,
            tokenizer_path=tokenizer_path,
            expected_checkpoint_spec_name=CHECKPOINT_SPEC_NAME,
            expected_source_repo=SOURCE_REPO,
            expected_source_revision=SOURCE_REVISION,
            expected_source_model_sha256=SOURCE_MODEL_SHA256,
            expected_reference_kind=SUITE_KIND,
            expected_reference_runtime=REFERENCE_RUNTIME,
            expected_reference_runtime_version=REFERENCE_TRANSFORMERS_VERSION,
        )
        validate_documents(identity, {"suite": suite, "provenance": provenance})
    except IdentityError as error:
        raise ArtifactError(str(error)) from error

    if identity.get("checkpoint_spec_version") != CHECKPOINT_SPEC_VERSION:
        raise ArtifactError("unexpected checkpoint_spec_version")
    if identity.get("source_weight_dtype") != SOURCE_WEIGHT_DTYPE:
        raise ArtifactError("unexpected source_weight_dtype")
    if identity.get("execution_weight_dtype") != EXECUTION_WEIGHT_DTYPE:
        raise ArtifactError("unexpected execution_weight_dtype")
    if identity.get("oracle_semantics") != REFERENCE_ORACLE_SEMANTICS:
        raise ArtifactError("unexpected oracle semantics")

    if suite.get("schema_version") != 1 or suite.get("kind") != SUITE_KIND:
        raise ArtifactError("reference suite schema/kind mismatch")
    if suite.get("expected_config") != EXPECTED_CONFIG:
        raise ArtifactError("reference suite expected_config mismatch")
    if suite.get("reference_execution_policy") != REFERENCE_EXECUTION_POLICY:
        raise ArtifactError("reference execution policy mismatch")
    case_policy = suite.get("case_policy")
    expected_case_policy = {
        "families": list(PROMPT_FAMILIES),
        "prompt_token_lengths": list(TARGET_PROMPT_TOKENS),
        "standard_decode_steps": STANDARD_DECODE_STEPS,
        "deep_decode_prompt_tokens": 32,
        "deep_decode_steps": DEEP_DECODE_STEPS,
        "oracle": REFERENCE_ORACLE_SEMANTICS,
    }
    if case_policy != expected_case_policy:
        raise ArtifactError("reference suite case_policy mismatch")
    _validate_cases(suite.get("cases"))

    if model.get("format") != "nnis-model" or model.get("version") != 1:
        raise ArtifactError("model manifest format/version mismatch")
    if model.get("config") != EXPECTED_CONFIG:
        raise ArtifactError("model manifest config mismatch")
    if provenance.get("transformers_version") != REFERENCE_TRANSFORMERS_VERSION:
        raise ArtifactError("model provenance Transformers version mismatch")

    return {
        "checkpoint_spec_name": CHECKPOINT_SPEC_NAME,
        "source_revision": SOURCE_REVISION,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "case_count": len(suite["cases"]),
    }


def _synthetic_cases() -> list[dict]:
    cases: list[dict] = []
    for family in PROMPT_FAMILIES:
        for target in TARGET_PROMPT_TOKENS:
            input_ids = [1, *([42] * (target - 1))]
            cases.append(
                {
                    "name": f"{family}-p{target:04d}-d{STANDARD_DECODE_STEPS:03d}",
                    "family": family,
                    "target_prompt_tokens": target,
                    "decode_steps": STANDARD_DECODE_STEPS,
                    "input_ids": input_ids,
                    "greedy_ids": [2] * STANDARD_DECODE_STEPS,
                }
            )
            if target == 32:
                cases.append(
                    {
                        "name": f"{family}-p{target:04d}-d{DEEP_DECODE_STEPS:03d}",
                        "family": family,
                        "target_prompt_tokens": target,
                        "decode_steps": DEEP_DECODE_STEPS,
                        "input_ids": input_ids,
                        "greedy_ids": [2] * DEEP_DECODE_STEPS,
                    }
                )
    return cases


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        (fixture / "model").mkdir()
        tokenizer = fixture / "tokenizer.json"
        tokenizer.write_bytes(b'{"version":"synthetic-tinyllama-tokenizer"}\n')
        identity = build_identity(
            checkpoint_spec_name=CHECKPOINT_SPEC_NAME,
            checkpoint_spec_version=CHECKPOINT_SPEC_VERSION,
            source_repo=SOURCE_REPO,
            source_revision=SOURCE_REVISION,
            source_model_sha256=SOURCE_MODEL_SHA256,
            tokenizer_path=tokenizer,
            reference_kind=SUITE_KIND,
            reference_runtime=REFERENCE_RUNTIME,
            reference_runtime_version=REFERENCE_TRANSFORMERS_VERSION,
            source_weight_dtype=SOURCE_WEIGHT_DTYPE,
            execution_weight_dtype=EXECUTION_WEIGHT_DTYPE,
            oracle_semantics=REFERENCE_ORACLE_SEMANTICS,
        )
        (fixture / IDENTITY_FILENAME).write_text(
            json.dumps(identity, indent=2) + "\n", encoding="utf-8"
        )
        provenance = {
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_model_sha256": SOURCE_MODEL_SHA256,
            "source_weight_dtype": SOURCE_WEIGHT_DTYPE,
            "execution_weight_dtype": EXECUTION_WEIGHT_DTYPE,
            "tokenizer_sha256": identity["tokenizer_sha256"],
            "transformers_version": REFERENCE_TRANSFORMERS_VERSION,
        }
        (fixture / "model" / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        model = {"format": "nnis-model", "version": 1, "config": EXPECTED_CONFIG, "tensors": []}
        (fixture / "model" / "model.json").write_text(
            json.dumps(model, indent=2) + "\n", encoding="utf-8"
        )
        suite = {
            "schema_version": 1,
            "kind": SUITE_KIND,
            "source_repo": SOURCE_REPO,
            "source_revision": SOURCE_REVISION,
            "source_model_sha256": SOURCE_MODEL_SHA256,
            "source_weight_dtype": SOURCE_WEIGHT_DTYPE,
            "execution_weight_dtype": EXECUTION_WEIGHT_DTYPE,
            "tokenizer_sha256": identity["tokenizer_sha256"],
            "transformers_version": REFERENCE_TRANSFORMERS_VERSION,
            "expected_config": EXPECTED_CONFIG,
            "reference_execution_policy": REFERENCE_EXECUTION_POLICY,
            "case_policy": {
                "families": list(PROMPT_FAMILIES),
                "prompt_token_lengths": list(TARGET_PROMPT_TOKENS),
                "standard_decode_steps": STANDARD_DECODE_STEPS,
                "deep_decode_prompt_tokens": 32,
                "deep_decode_steps": DEEP_DECODE_STEPS,
                "oracle": REFERENCE_ORACLE_SEMANTICS,
            },
            "cases": _synthetic_cases(),
        }
        (fixture / "reference_suite.json").write_text(
            json.dumps(suite, indent=2) + "\n", encoding="utf-8"
        )
        result = validate_fixture(fixture)
        if result["case_count"] != 18:
            raise AssertionError("valid synthetic artifact produced wrong case count")

        tampered = copy.deepcopy(suite)
        tampered["reference_execution_policy"]["mkl_num_threads"] = 2
        (fixture / "reference_suite.json").write_text(
            json.dumps(tampered, indent=2) + "\n", encoding="utf-8"
        )
        try:
            validate_fixture(fixture)
        except ArtifactError:
            pass
        else:
            raise AssertionError("tampered reference execution policy was accepted")

        (fixture / "reference_suite.json").write_text(
            json.dumps(suite, indent=2) + "\n", encoding="utf-8"
        )
        tokenizer.write_bytes(b'{"version":"tampered"}\n')
        try:
            validate_fixture(fixture)
        except ArtifactError:
            pass
        else:
            raise AssertionError("tampered tokenizer was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("TinyLlama reference artifact self-test passed")
    if args.fixture is not None:
        result = validate_fixture(args.fixture)
        print(json.dumps(result, sort_keys=True))
    if not args.self_test and args.fixture is None:
        parser.error("provide FIXTURE and/or --self-test")


if __name__ == "__main__":
    main()
