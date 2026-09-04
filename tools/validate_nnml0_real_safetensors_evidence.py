#!/usr/bin/env python3
"""Validate NNML0 real-Safetensors physical qualification evidence."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path

KIND = "nnis-nnml0-real-safetensors"
SCHEMA_VERSION = 2
SOURCE_REPO = "HuggingFaceTB/SmolLM2-135M"
SOURCE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SOURCE_MODEL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

TOP_LEVEL_KEYS = {
    "schema_version",
    "kind",
    "result",
    "unix_timestamp_seconds",
    "nnis_git_commit",
    "nnis_git_dirty",
    "host_arch",
    "host_os",
    "source",
    "device",
    "model",
    "decoder_capabilities",
}
SOURCE_KEYS = {"repo", "revision", "model_sha256"}
DEVICE_KEYS = {
    "ordinal",
    "name",
    "uuid",
    "compute_capability_major",
    "compute_capability_minor",
    "multiprocessor_count",
}
MODEL_KEYS = {
    "vocab_size",
    "eos_token_id",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "activation",
    "weight_dtype",
    "head_dim",
    "key_value_width",
}
CAPABILITY_KEYS = {"profile", "canonical_record"}
CAPABILITY_PROFILE_KEYS = {
    "contract_version",
    "attention_topology",
    "rope_semantics",
    "mlp_semantics",
    "weight_dtype",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
}
EXPECTED_MODEL = {
    "vocab_size": 49152,
    "eos_token_id": 0,
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "max_position_embeddings": 8192,
    "activation": "silu",
    "weight_dtype": "bf16",
    "head_dim": 64,
    "key_value_width": 192,
}
EXPECTED_CAPABILITY_PROFILE = {
    "contract_version": 1,
    "attention_topology": "grouped_query",
    "rope_semantics": "llama_rotate_half_unscaled",
    "mlp_semantics": "swiglu_silu",
    "weight_dtype": "bf16",
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "head_dim": 64,
}
EXPECTED_CAPABILITY_RECORD = (
    "NNIS-DECODER-CAPABILITY-V1\n"
    "attention=grouped_query\n"
    "rope=llama_rotate_half_unscaled\n"
    "mlp=swiglu_silu\n"
    "weight_dtype=bf16\n"
    "q_heads=9\n"
    "kv_heads=3\n"
    "head_dim=64\n"
)


class EvidenceError(ValueError):
    pass


def require_dict(value: object, name: str, keys: set[str]) -> dict:
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} must be an object")
    actual = set(value)
    if actual != keys:
        raise EvidenceError(
            f"{name} keys mismatch; missing={sorted(keys - actual)}, extra={sorted(actual - keys)}"
        )
    return value


def require_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidenceError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise EvidenceError(f"{name} must be >= {minimum}; got {value}")
    return value


def require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{name} must be a non-empty string")
    return value


def validate_evidence(document: object, expected_commit: str) -> None:
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise EvidenceError(f"invalid expected commit {expected_commit!r}")

    root = require_dict(document, "evidence", TOP_LEVEL_KEYS)
    schema_version = require_int(root["schema_version"], "schema_version", minimum=1)
    if schema_version != SCHEMA_VERSION:
        raise EvidenceError(f"unsupported schema_version {schema_version!r}")
    if root["kind"] != KIND:
        raise EvidenceError(f"unexpected evidence kind {root['kind']!r}")
    if root["result"] != "pass":
        raise EvidenceError(f"qualification result is not pass: {root['result']!r}")
    require_int(root["unix_timestamp_seconds"], "unix_timestamp_seconds", minimum=1)

    commit = require_nonempty_string(root["nnis_git_commit"], "nnis_git_commit")
    if COMMIT_RE.fullmatch(commit) is None:
        raise EvidenceError(f"invalid nnis_git_commit {commit!r}")
    if commit != expected_commit:
        raise EvidenceError(
            f"evidence commit {commit} does not match expected commit {expected_commit}"
        )
    if root["nnis_git_dirty"] is not False:
        raise EvidenceError("nnis_git_dirty must be false")
    require_nonempty_string(root["host_arch"], "host_arch")
    require_nonempty_string(root["host_os"], "host_os")

    source = require_dict(root["source"], "source", SOURCE_KEYS)
    expected_source = {
        "repo": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "model_sha256": SOURCE_MODEL_SHA256,
    }
    if source != expected_source:
        raise EvidenceError(f"source identity mismatch: {source!r}")

    device = require_dict(root["device"], "device", DEVICE_KEYS)
    require_int(device["ordinal"], "device.ordinal", minimum=0)
    require_nonempty_string(device["name"], "device.name")
    require_nonempty_string(device["uuid"], "device.uuid")
    require_int(
        device["compute_capability_major"], "device.compute_capability_major", minimum=1
    )
    require_int(
        device["compute_capability_minor"], "device.compute_capability_minor", minimum=0
    )
    require_int(device["multiprocessor_count"], "device.multiprocessor_count", minimum=1)

    model = require_dict(root["model"], "model", MODEL_KEYS)
    for key, expected in EXPECTED_MODEL.items():
        actual = model[key]
        if isinstance(expected, int):
            actual = require_int(actual, f"model.{key}", minimum=0)
        elif isinstance(expected, str):
            actual = require_nonempty_string(actual, f"model.{key}")
        if actual != expected:
            raise EvidenceError(
                f"model.{key} mismatch: got {actual!r}, expected {expected!r}"
            )
    rms_norm_eps = model["rms_norm_eps"]
    if isinstance(rms_norm_eps, bool) or not isinstance(rms_norm_eps, (int, float)):
        raise EvidenceError("model.rms_norm_eps must be numeric")
    if not math.isfinite(float(rms_norm_eps)) or not math.isclose(
        float(rms_norm_eps), 1.0e-5, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise EvidenceError(f"unexpected model.rms_norm_eps {rms_norm_eps!r}")
    rope_theta = model["rope_theta"]
    if isinstance(rope_theta, bool) or not isinstance(rope_theta, (int, float)):
        raise EvidenceError("model.rope_theta must be numeric")
    if not math.isfinite(float(rope_theta)) or float(rope_theta) != 100000.0:
        raise EvidenceError(f"unexpected model.rope_theta {rope_theta!r}")

    capabilities = require_dict(
        root["decoder_capabilities"], "decoder_capabilities", CAPABILITY_KEYS
    )
    profile = require_dict(
        capabilities["profile"],
        "decoder_capabilities.profile",
        CAPABILITY_PROFILE_KEYS,
    )
    for key, expected in EXPECTED_CAPABILITY_PROFILE.items():
        actual = profile[key]
        if isinstance(expected, int):
            actual = require_int(
                actual, f"decoder_capabilities.profile.{key}", minimum=0
            )
        elif isinstance(expected, str):
            actual = require_nonempty_string(
                actual, f"decoder_capabilities.profile.{key}"
            )
        if actual != expected:
            raise EvidenceError(
                f"decoder_capabilities.profile.{key} mismatch: got {actual!r}, expected {expected!r}"
            )
    canonical_record = require_nonempty_string(
        capabilities["canonical_record"], "decoder_capabilities.canonical_record"
    )
    if canonical_record != EXPECTED_CAPABILITY_RECORD:
        raise EvidenceError("decoder capability canonical record mismatch")

    if profile["num_attention_heads"] != model["num_attention_heads"]:
        raise EvidenceError("decoder capability Q-head count disagrees with model evidence")
    if profile["num_key_value_heads"] != model["num_key_value_heads"]:
        raise EvidenceError("decoder capability KV-head count disagrees with model evidence")
    if profile["head_dim"] != model["head_dim"]:
        raise EvidenceError("decoder capability head_dim disagrees with model evidence")
    if profile["weight_dtype"] != model["weight_dtype"]:
        raise EvidenceError("decoder capability weight dtype disagrees with model evidence")


def synthetic_good_evidence() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "result": "pass",
        "unix_timestamp_seconds": 1788459000,
        "nnis_git_commit": "a" * 40,
        "nnis_git_dirty": False,
        "host_arch": "aarch64",
        "host_os": "linux",
        "source": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "model_sha256": SOURCE_MODEL_SHA256,
        },
        "device": {
            "ordinal": 0,
            "name": "NVIDIA Test GPU",
            "uuid": "CUuuid([1, 2, 3, 4])",
            "compute_capability_major": 11,
            "compute_capability_minor": 0,
            "multiprocessor_count": 14,
        },
        "model": {
            **EXPECTED_MODEL,
            "rms_norm_eps": 9.999999747378752e-06,
            "rope_theta": 100000.0,
        },
        "decoder_capabilities": {
            "profile": dict(EXPECTED_CAPABILITY_PROFILE),
            "canonical_record": EXPECTED_CAPABILITY_RECORD,
        },
    }


def negative_self_tests() -> None:
    good = synthetic_good_evidence()
    validate_evidence(good, expected_commit="a" * 40)

    mutations: list[tuple[str, object]] = [
        ("schema", lambda d: d.__setitem__("schema_version", 1)),
        ("schema_bool", lambda d: d.__setitem__("schema_version", True)),
        ("kind", lambda d: d.__setitem__("kind", "other")),
        ("result", lambda d: d.__setitem__("result", "fail")),
        ("dirty", lambda d: d.__setitem__("nnis_git_dirty", True)),
        ("commit", lambda d: d.__setitem__("nnis_git_commit", "bad")),
        ("source_sha", lambda d: d["source"].__setitem__("model_sha256", "0" * 64)),
        ("gpu_name", lambda d: d["device"].__setitem__("name", "")),
        ("gpu_uuid", lambda d: d["device"].__setitem__("uuid", None)),
        ("gpu_cc", lambda d: d["device"].__setitem__("compute_capability_major", 0)),
        ("gpu_cc_bool", lambda d: d["device"].__setitem__("compute_capability_major", True)),
        ("dtype", lambda d: d["model"].__setitem__("weight_dtype", "f32")),
        ("eos_bool", lambda d: d["model"].__setitem__("eos_token_id", False)),
        ("layers", lambda d: d["model"].__setitem__("num_hidden_layers", 29)),
        ("layers_bool", lambda d: d["model"].__setitem__("num_hidden_layers", True)),
        ("kv_width", lambda d: d["model"].__setitem__("key_value_width", 576)),
        ("eps", lambda d: d["model"].__setitem__("rms_norm_eps", float("nan"))),
        (
            "capability_version",
            lambda d: d["decoder_capabilities"]["profile"].__setitem__(
                "contract_version", 2
            ),
        ),
        (
            "attention_topology",
            lambda d: d["decoder_capabilities"]["profile"].__setitem__(
                "attention_topology", "multi_head"
            ),
        ),
        (
            "rope_semantics",
            lambda d: d["decoder_capabilities"]["profile"].__setitem__(
                "rope_semantics", "other"
            ),
        ),
        (
            "capability_record",
            lambda d: d["decoder_capabilities"].__setitem__(
                "canonical_record", EXPECTED_CAPABILITY_RECORD + "drift=true\n"
            ),
        ),
        (
            "capability_model_disagreement",
            lambda d: d["decoder_capabilities"]["profile"].__setitem__(
                "num_attention_heads", 3
            ),
        ),
        ("extra_key", lambda d: d.__setitem__("unexpected", True)),
    ]
    for name, mutate in mutations:
        candidate = copy.deepcopy(good)
        mutate(candidate)
        try:
            validate_evidence(candidate, expected_commit="a" * 40)
        except EvidenceError:
            continue
        raise AssertionError(f"negative self-test {name!r} was accepted")

    try:
        validate_evidence(good, expected_commit="b" * 40)
    except EvidenceError:
        pass
    else:
        raise AssertionError("expected-commit mismatch was accepted")

    try:
        validate_evidence(good, expected_commit="bad")
    except EvidenceError:
        pass
    else:
        raise AssertionError("malformed expected commit was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", nargs="?", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.expected_commit is not None and COMMIT_RE.fullmatch(args.expected_commit) is None:
        parser.error("--expected-commit must be a lowercase 40-hex commit SHA")
    if args.self_test:
        negative_self_tests()
        print("NNML0 real Safetensors evidence negative self-tests passed")
    if args.evidence is not None:
        if args.expected_commit is None:
            parser.error("--expected-commit is required when validating an evidence file")
        document = json.loads(args.evidence.read_text())
        validate_evidence(document, expected_commit=args.expected_commit)
        print(f"NNML0_REAL_SAFETENSORS_EVIDENCE_OK {args.evidence}")
    elif not args.self_test:
        parser.error("provide an evidence path and/or --self-test")


if __name__ == "__main__":
    main()
