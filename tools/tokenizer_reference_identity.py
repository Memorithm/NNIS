#!/usr/bin/env python3
"""Versioned tokenizer/reference identity contract for NNIS qualification fixtures.

This module is standard-library only. It binds a real tokenizer artifact to an
exact checkpoint source identity and trusted reference-runtime identity without
hardcoding tokenizer hashes before the artifact has actually been downloaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Mapping

CONTRACT_VERSION = 1
KIND = "nnis-tokenizer-reference-identity-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class IdentityError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IdentityError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    value = _require_nonempty(value, name)
    if SHA256_RE.fullmatch(value) is None:
        raise IdentityError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


def build_identity(
    *,
    checkpoint_spec_name: str,
    checkpoint_spec_version: int,
    source_repo: str,
    source_revision: str,
    source_model_sha256: str,
    tokenizer_path: Path,
    reference_kind: str,
    reference_runtime: str,
    reference_runtime_version: str,
    source_weight_dtype: str,
    execution_weight_dtype: str,
    oracle_semantics: str,
) -> dict:
    if checkpoint_spec_version <= 0:
        raise IdentityError("checkpoint_spec_version must be positive")
    identity = {
        "contract_version": CONTRACT_VERSION,
        "kind": KIND,
        "checkpoint_spec_name": _require_nonempty(
            checkpoint_spec_name, "checkpoint_spec_name"
        ),
        "checkpoint_spec_version": checkpoint_spec_version,
        "source_repo": _require_nonempty(source_repo, "source_repo"),
        "source_revision": _require_nonempty(source_revision, "source_revision"),
        "source_model_sha256": _require_sha256(
            source_model_sha256, "source_model_sha256"
        ),
        "tokenizer_file": tokenizer_path.name,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "reference_kind": _require_nonempty(reference_kind, "reference_kind"),
        "reference_runtime": _require_nonempty(reference_runtime, "reference_runtime"),
        "reference_runtime_version": _require_nonempty(
            reference_runtime_version, "reference_runtime_version"
        ),
        "source_weight_dtype": _require_nonempty(
            source_weight_dtype, "source_weight_dtype"
        ),
        "execution_weight_dtype": _require_nonempty(
            execution_weight_dtype, "execution_weight_dtype"
        ),
        "oracle_semantics": _require_nonempty(oracle_semantics, "oracle_semantics"),
    }
    validate_identity(identity, tokenizer_path=tokenizer_path)
    return identity


def validate_identity(
    identity: object,
    *,
    tokenizer_path: Path | None = None,
    expected_checkpoint_spec_name: str | None = None,
    expected_source_repo: str | None = None,
    expected_source_revision: str | None = None,
    expected_source_model_sha256: str | None = None,
    expected_reference_kind: str | None = None,
    expected_reference_runtime: str | None = None,
    expected_reference_runtime_version: str | None = None,
) -> dict:
    if not isinstance(identity, dict):
        raise IdentityError("identity must be an object")
    expected_keys = {
        "contract_version",
        "kind",
        "checkpoint_spec_name",
        "checkpoint_spec_version",
        "source_repo",
        "source_revision",
        "source_model_sha256",
        "tokenizer_file",
        "tokenizer_sha256",
        "reference_kind",
        "reference_runtime",
        "reference_runtime_version",
        "source_weight_dtype",
        "execution_weight_dtype",
        "oracle_semantics",
    }
    actual_keys = set(identity)
    if actual_keys != expected_keys:
        raise IdentityError(
            f"identity keys mismatch; missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    if identity["contract_version"] != CONTRACT_VERSION:
        raise IdentityError(
            f"unsupported contract_version {identity['contract_version']!r}"
        )
    if identity["kind"] != KIND:
        raise IdentityError(f"unexpected identity kind {identity['kind']!r}")
    if isinstance(identity["checkpoint_spec_version"], bool) or not isinstance(
        identity["checkpoint_spec_version"], int
    ):
        raise IdentityError("checkpoint_spec_version must be an integer")
    if identity["checkpoint_spec_version"] <= 0:
        raise IdentityError("checkpoint_spec_version must be positive")

    string_fields = (
        "checkpoint_spec_name",
        "source_repo",
        "source_revision",
        "tokenizer_file",
        "reference_kind",
        "reference_runtime",
        "reference_runtime_version",
        "source_weight_dtype",
        "execution_weight_dtype",
        "oracle_semantics",
    )
    for field in string_fields:
        _require_nonempty(identity[field], field)
    model_sha = _require_sha256(identity["source_model_sha256"], "source_model_sha256")
    tokenizer_sha = _require_sha256(identity["tokenizer_sha256"], "tokenizer_sha256")

    expectations = {
        "checkpoint_spec_name": expected_checkpoint_spec_name,
        "source_repo": expected_source_repo,
        "source_revision": expected_source_revision,
        "source_model_sha256": expected_source_model_sha256,
        "reference_kind": expected_reference_kind,
        "reference_runtime": expected_reference_runtime,
        "reference_runtime_version": expected_reference_runtime_version,
    }
    for field, expected in expectations.items():
        if expected is not None and identity[field] != expected:
            raise IdentityError(
                f"{field} mismatch: got {identity[field]!r}, expected {expected!r}"
            )

    if tokenizer_path is not None:
        if tokenizer_path.name != identity["tokenizer_file"]:
            raise IdentityError(
                f"tokenizer filename mismatch: got {tokenizer_path.name!r}, "
                f"expected {identity['tokenizer_file']!r}"
            )
        actual = sha256_file(tokenizer_path)
        if actual != tokenizer_sha:
            raise IdentityError(
                f"tokenizer SHA-256 mismatch: actual {actual}, recorded {tokenizer_sha}"
            )

    if expected_source_model_sha256 is not None:
        _require_sha256(expected_source_model_sha256, "expected_source_model_sha256")
        if model_sha != expected_source_model_sha256:
            raise IdentityError(
                "source_model_sha256 does not match expected checkpoint source"
            )
    return identity


def canonical_record(identity: object) -> str:
    validated = validate_identity(identity)
    return "".join(
        [
            f"NNIS-TOKENIZER-REFERENCE-IDENTITY-V{validated['contract_version']}\n",
            f"checkpoint_spec_name={validated['checkpoint_spec_name']}\n",
            f"checkpoint_spec_version={validated['checkpoint_spec_version']}\n",
            f"source_repo={validated['source_repo']}\n",
            f"source_revision={validated['source_revision']}\n",
            f"source_model_sha256={validated['source_model_sha256']}\n",
            f"tokenizer_file={validated['tokenizer_file']}\n",
            f"tokenizer_sha256={validated['tokenizer_sha256']}\n",
            f"reference_kind={validated['reference_kind']}\n",
            f"reference_runtime={validated['reference_runtime']}\n",
            f"reference_runtime_version={validated['reference_runtime_version']}\n",
            f"source_weight_dtype={validated['source_weight_dtype']}\n",
            f"execution_weight_dtype={validated['execution_weight_dtype']}\n",
            f"oracle_semantics={validated['oracle_semantics']}\n",
        ]
    )


def validate_documents(
    identity: object,
    documents: Mapping[str, Mapping[str, object]],
) -> None:
    validated = validate_identity(identity)
    linked_fields = {
        "source_repo": validated["source_repo"],
        "source_revision": validated["source_revision"],
        "source_model_sha256": validated["source_model_sha256"],
        "tokenizer_sha256": validated["tokenizer_sha256"],
        "source_weight_dtype": validated["source_weight_dtype"],
        "execution_weight_dtype": validated["execution_weight_dtype"],
    }
    for label, document in documents.items():
        if not isinstance(document, Mapping):
            raise IdentityError(f"{label} must be an object")
        for field, expected in linked_fields.items():
            if document.get(field) != expected:
                raise IdentityError(
                    f"{label}.{field} mismatch: got {document.get(field)!r}, expected {expected!r}"
                )
        if document.get("transformers_version") != validated["reference_runtime_version"]:
            raise IdentityError(
                f"{label}.transformers_version mismatch: got "
                f"{document.get('transformers_version')!r}, expected "
                f"{validated['reference_runtime_version']!r}"
            )


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        tokenizer = Path(temporary) / "tokenizer.json"
        tokenizer.write_bytes(b'{"version":"synthetic"}\n')
        identity = build_identity(
            checkpoint_spec_name="synthetic-decoder",
            checkpoint_spec_version=1,
            source_repo="example/synthetic",
            source_revision="a" * 40,
            source_model_sha256="b" * 64,
            tokenizer_path=tokenizer,
            reference_kind="synthetic-reference-v1",
            reference_runtime="transformers",
            reference_runtime_version="4.0.0",
            source_weight_dtype="bfloat16",
            execution_weight_dtype="f32",
            oracle_semantics="synthetic CPU F32 oracle",
        )
        validate_identity(
            identity,
            tokenizer_path=tokenizer,
            expected_checkpoint_spec_name="synthetic-decoder",
            expected_source_repo="example/synthetic",
            expected_source_revision="a" * 40,
            expected_source_model_sha256="b" * 64,
            expected_reference_kind="synthetic-reference-v1",
            expected_reference_runtime="transformers",
            expected_reference_runtime_version="4.0.0",
        )
        document = {
            "source_repo": identity["source_repo"],
            "source_revision": identity["source_revision"],
            "source_model_sha256": identity["source_model_sha256"],
            "tokenizer_sha256": identity["tokenizer_sha256"],
            "source_weight_dtype": identity["source_weight_dtype"],
            "execution_weight_dtype": identity["execution_weight_dtype"],
            "transformers_version": identity["reference_runtime_version"],
        }
        validate_documents(identity, {"synthetic": document})
        record = canonical_record(identity)
        if not record.startswith("NNIS-TOKENIZER-REFERENCE-IDENTITY-V1\n"):
            raise AssertionError("canonical identity prefix drifted")

        mutations = [
            ("version", lambda value: value.__setitem__("contract_version", 2)),
            ("kind", lambda value: value.__setitem__("kind", "other")),
            ("model_sha", lambda value: value.__setitem__("source_model_sha256", "bad")),
            ("tokenizer_sha", lambda value: value.__setitem__("tokenizer_sha256", "bad")),
            ("runtime", lambda value: value.__setitem__("reference_runtime", "")),
            ("extra", lambda value: value.__setitem__("unexpected", True)),
        ]
        for name, mutate in mutations:
            candidate = copy.deepcopy(identity)
            mutate(candidate)
            try:
                validate_identity(candidate)
            except IdentityError:
                continue
            raise AssertionError(f"negative identity self-test {name!r} was accepted")

        tokenizer.write_bytes(b'{"version":"tampered"}\n')
        try:
            validate_identity(identity, tokenizer_path=tokenizer)
        except IdentityError:
            pass
        else:
            raise AssertionError("tampered tokenizer artifact was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("NNIS tokenizer/reference identity self-tests passed")
    if args.identity is not None:
        document = json.loads(args.identity.read_text(encoding="utf-8"))
        validate_identity(document, tokenizer_path=args.tokenizer)
        print(canonical_record(document), end="")
    elif not args.self_test:
        parser.error("provide --self-test and/or --identity FILE")


if __name__ == "__main__":
    main()
