#!/usr/bin/env python3
"""Validate the TinyLlama reference artifact under the F16 LM-head oracle contract.

Version 2 keeps the pinned checkpoint/model-format identity from the historical
artifact while changing the greedy oracle semantics explicitly: Transformers CPU
executes the checkpoint in F32, then each LM-head logit vector is rounded through
IEEE F16 before greedy argmax. This matches the documented NNIS F16 tensor boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import validate_tinyllama_reference_artifact as v1

CHECKPOINT_SPEC_VERSION = 2
REFERENCE_ORACLE_SEMANTICS = (
    "Transformers CPU F32 execution from the exact pinned checkpoint with each "
    "LM-head logit vector rounded F32 -> IEEE F16 -> F32 before deterministic greedy argmax"
)


def _activate_v2_contract() -> None:
    v1.CHECKPOINT_SPEC_VERSION = CHECKPOINT_SPEC_VERSION
    v1.REFERENCE_ORACLE_SEMANTICS = REFERENCE_ORACLE_SEMANTICS


def validate_fixture(fixture: Path) -> dict:
    _activate_v2_contract()
    result = v1.validate_fixture(fixture)
    result = dict(result)
    result["checkpoint_spec_version"] = CHECKPOINT_SPEC_VERSION
    result["oracle_semantics"] = REFERENCE_ORACLE_SEMANTICS
    return result


def self_test() -> None:
    _activate_v2_contract()
    v1.self_test()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("TinyLlama F16-oracle reference artifact v2 self-test passed")
    if args.fixture is not None:
        print(json.dumps(validate_fixture(args.fixture), sort_keys=True))
    if not args.self_test and args.fixture is None:
        parser.error("provide FIXTURE and/or --self-test")


if __name__ == "__main__":
    main()
