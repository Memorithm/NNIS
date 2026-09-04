#!/usr/bin/env python3
"""Run the remaining NNIS P0 physical qualification gates on one exact main commit.

This launcher is orchestration only. It does not weaken any underlying validator,
change runtime defaults, or promote a checkpoint/model family. It requires a clean
checkout at the fetched origin/main commit, uses separate pinned Python environments
for the SmolLM2 and TinyLlama reference generators, runs the existing NNML0 physical
loader gate, produces the two NNML1 parity records, composes the same-head parity
suite, and writes a manifest containing artifact SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

BUNDLE_KIND = "nnis-p0-physical-qualification-bundle-v1"
SCHEMA_VERSION = 1
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

SMOLLM2_ENV = {
    "torch": "2.4.0",
    "transformers": "4.40.1",
    "safetensors": "0.4.5",
    "huggingface_hub": "0.24.7",
}
TINYLLAMA_ENV = {
    "torch": "2.4.0",
    "transformers": "4.43.3",
    "safetensors": "0.4.5",
    "huggingface_hub": "0.24.7",
}


class QualificationError(RuntimeError):
    pass


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run NNML0 real-Safetensors plus same-head SmolLM2/TinyLlama NNML1 "
            "physical parity qualification on the fetched origin/main commit."
        )
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--smollm2-python",
        type=Path,
        required=True,
        help="Python executable with the exact SmolLM2 reference environment",
    )
    parser.add_argument(
        "--tinyllama-python",
        type=Path,
        required=True,
        help="Python executable with the exact TinyLlama reference environment",
    )
    parser.add_argument("--device", type=non_negative_int, default=0)
    parser.add_argument("--tinyllama-repeats", type=positive_int, default=2)
    parser.add_argument("--tinyllama-rounds", type=positive_int, default=4)
    parser.add_argument("--tinyllama-warmups", type=positive_int, default=1)
    parser.add_argument("--tinyllama-iterations", type=positive_int, default=3)
    parser.add_argument(
        "--expected-head",
        help=(
            "optional exact promoted main SHA; when supplied it must equal both HEAD "
            "and fetched origin/main"
        ),
    )
    parser.add_argument(
        "--no-resume-tinyllama",
        action="store_true",
        help="disable the TinyLlama launcher's exact-head resumable campaign reuse",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def run_capture(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise QualificationError(f"command failed ({' '.join(command)}): {detail}")
    return completed.stdout.strip()


def run_stream(command: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise QualificationError(
            f"command exited with status {completed.returncode}: {' '.join(command)}"
        )


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise QualificationError(f"required command not found in PATH: {name}")


def repository_root() -> Path:
    return Path(run_capture(["git", "rev-parse", "--show-toplevel"])).resolve()


def require_clean_repository(root: Path) -> None:
    status = run_capture(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root
    )
    if status:
        raise QualificationError(
            "refusing physical qualification from a dirty worktree, including untracked files"
        )


def require_work_dir_outside_repository(root: Path, work_dir: Path) -> None:
    if work_dir == root or work_dir.is_relative_to(root):
        raise QualificationError(
            f"--work-dir must be outside the repository so evidence does not dirty the checkout: {work_dir}"
        )


def resolve_exact_main(root: Path, expected_head: str | None) -> str:
    require_clean_repository(root)
    run_capture(["git", "fetch", "origin", "main"], cwd=root)
    head = run_capture(["git", "rev-parse", "HEAD"], cwd=root)
    origin_main = run_capture(["git", "rev-parse", "origin/main"], cwd=root)
    if COMMIT_RE.fullmatch(head) is None or COMMIT_RE.fullmatch(origin_main) is None:
        raise QualificationError("HEAD/origin/main did not resolve to lowercase 40-hex commits")
    if expected_head is not None:
        if COMMIT_RE.fullmatch(expected_head) is None:
            raise QualificationError("--expected-head must be a lowercase 40-hex commit")
        if head != expected_head or origin_main != expected_head:
            raise QualificationError(
                f"expected head {expected_head}, but HEAD={head} and origin/main={origin_main}"
            )
    elif head != origin_main:
        raise QualificationError(
            f"HEAD {head} is not the fetched promoted origin/main {origin_main}"
        )
    require_clean_repository(root)
    return head


def _probe_script() -> str:
    return """
import json
import torch
import transformers
import safetensors
import huggingface_hub
print(json.dumps({
    "python": __import__("sys").version.split()[0],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "safetensors": safetensors.__version__,
    "huggingface_hub": huggingface_hub.__version__,
}, sort_keys=True))
""".strip()


def normalize_base_version(value: str) -> str:
    return value.split("+", 1)[0]


def validate_python_probe(
    probe: dict[str, Any], expected: dict[str, str], label: str
) -> dict[str, str]:
    if not isinstance(probe, dict):
        raise QualificationError(f"{label} Python probe did not return an object")
    result: dict[str, str] = {}
    for field in ["python", "torch", "transformers", "safetensors", "huggingface_hub"]:
        value = probe.get(field)
        if not isinstance(value, str) or not value:
            raise QualificationError(f"{label} Python probe lacks {field}")
        result[field] = value
    for package, version in expected.items():
        actual = normalize_base_version(result[package])
        if actual != version:
            raise QualificationError(
                f"{label} requires {package} {version}; got {result[package]}"
            )
    return result


def probe_python(executable: Path, expected: dict[str, str], label: str) -> dict[str, str]:
    executable = executable.expanduser().resolve()
    if not executable.is_file():
        raise QualificationError(f"{label} Python executable does not exist: {executable}")
    raw = run_capture([str(executable), "-c", _probe_script()])
    try:
        probe = json.loads(raw.splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise QualificationError(f"{label} Python version probe emitted invalid JSON") from error
    return validate_python_probe(probe, expected, label)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_nonempty_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise QualificationError(f"{label} was not produced as a non-empty file: {path}")
    return path


def artifact_entry(path: Path) -> dict[str, Any]:
    require_nonempty_file(path, "artifact")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def validate_json_kind(path: Path, expected_kind: str, expected_head: str) -> dict[str, Any]:
    require_nonempty_file(path, expected_kind)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise QualificationError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(document, dict) or document.get("kind") != expected_kind:
        raise QualificationError(
            f"unexpected evidence kind in {path}: {document.get('kind') if isinstance(document, dict) else type(document)}"
        )
    commit = document.get("execution_git_commit")
    if expected_kind == "nnis-nnml1-multi-model-parity-suite-v1":
        if commit != expected_head:
            raise QualificationError(
                f"parity suite commit {commit!r} != exact qualification head {expected_head}"
            )
    elif expected_kind == "nnis-nnml1-reference-parity-record-v1":
        if commit != expected_head:
            raise QualificationError(
                f"parity record commit {commit!r} != exact qualification head {expected_head}"
            )
    return document


def run_bundle(args: argparse.Namespace) -> Path:
    for command in ["git", "cargo", "curl", "sha256sum", "bash"]:
        require_command(command)

    root = repository_root()
    work_dir = args.work_dir.expanduser().resolve()
    require_work_dir_outside_repository(root, work_dir)
    head = resolve_exact_main(root, args.expected_head)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    smollm2_python = args.smollm2_python.expanduser().resolve()
    tinyllama_python = args.tinyllama_python.expanduser().resolve()
    smollm2_probe = probe_python(smollm2_python, SMOLLM2_ENV, "SmolLM2")
    tinyllama_probe = probe_python(tinyllama_python, TINYLLAMA_ENV, "TinyLlama")
    if smollm2_python == tinyllama_python:
        raise QualificationError(
            "SmolLM2 and TinyLlama require different Transformers versions; provide distinct Python environments"
        )

    nnml0_model = work_dir / "nnml0-smollm2-source"
    nnml0_evidence = work_dir / "nnml0-real-safetensors-evidence.json"
    smollm2_fixture = work_dir / "smollm2-fixture"
    smollm2_parity = work_dir / "smollm2-parity-record.json"
    tinyllama_work = work_dir / "tinyllama"
    tinyllama_run_dir = tinyllama_work / f"runs-{head[:12]}"
    tinyllama_parity = tinyllama_run_dir / "parity_record.json"
    parity_suite = work_dir / "nnml1-multi-model-parity-suite.json"
    bundle_manifest = work_dir / "P0_PHYSICAL_QUALIFICATION.json"

    require_clean_repository(root)
    run_stream(
        [
            "bash",
            str(root / "tools" / "run_nnml0_real_safetensors_qualification.sh"),
            str(nnml0_model),
            str(nnml0_evidence),
        ],
        cwd=root,
    )
    run_stream(
        [
            sys.executable,
            str(root / "tools" / "validate_nnml0_real_safetensors_evidence.py"),
            str(nnml0_evidence),
            "--expected-commit",
            head,
        ],
        cwd=root,
    )

    smollm2_fixture_command = [
        str(smollm2_python),
        str(root / "tools" / "smollm2_135m_fixture.py"),
        "--output",
        str(smollm2_fixture),
    ]
    if cache_dir is not None:
        smollm2_fixture_command.extend(["--cache-dir", str(cache_dir)])
    require_clean_repository(root)
    run_stream(smollm2_fixture_command, cwd=root)
    if smollm2_parity.exists():
        smollm2_parity.unlink()
    run_stream(
        [
            "cargo",
            "run",
            "--locked",
            "-p",
            "nnis-model",
            "--example",
            "compare_smollm2_135m",
            "--",
            "--model",
            str(smollm2_fixture / "model"),
            "--reference",
            str(smollm2_fixture / "reference"),
            "--logit-policy",
            "strict",
            "--evidence-json",
            str(smollm2_parity),
        ],
        cwd=root,
    )
    run_stream(
        [
            sys.executable,
            str(root / "tools" / "validate_nnml1_multi_model_parity_evidence.py"),
            str(smollm2_parity),
        ],
        cwd=root,
    )
    smollm2_record = validate_json_kind(
        smollm2_parity, "nnis-nnml1-reference-parity-record-v1", head
    )
    if smollm2_record.get("parity_level") != "logit_and_generation":
        raise QualificationError("SmolLM2 physical record is not strict logit_and_generation evidence")

    tinyllama_command = [
        str(tinyllama_python),
        str(root / "tools" / "run_tinyllama_massive_campaign.py"),
        "--work-dir",
        str(tinyllama_work),
        "--device",
        str(args.device),
        "--repeats",
        str(args.tinyllama_repeats),
        "--rounds",
        str(args.tinyllama_rounds),
        "--warmups",
        str(args.tinyllama_warmups),
        "--iterations",
        str(args.tinyllama_iterations),
    ]
    if cache_dir is not None:
        tinyllama_command.extend(["--cache-dir", str(cache_dir)])
    if args.no_resume_tinyllama:
        tinyllama_command.append("--no-resume")
    require_clean_repository(root)
    run_stream(tinyllama_command, cwd=root)
    run_stream(
        [
            sys.executable,
            str(root / "tools" / "validate_nnml1_multi_model_parity_evidence.py"),
            str(tinyllama_parity),
        ],
        cwd=root,
    )
    tinyllama_record = validate_json_kind(
        tinyllama_parity, "nnis-nnml1-reference-parity-record-v1", head
    )
    if tinyllama_record.get("parity_level") != "generation_trajectory":
        raise QualificationError("TinyLlama physical record must remain generation_trajectory evidence")

    if parity_suite.exists():
        parity_suite.unlink()
    run_stream(
        [
            sys.executable,
            str(root / "tools" / "validate_nnml1_multi_model_parity_evidence.py"),
            "--compose",
            str(parity_suite),
            str(smollm2_parity),
            str(tinyllama_parity),
        ],
        cwd=root,
    )
    run_stream(
        [
            sys.executable,
            str(root / "tools" / "validate_nnml1_multi_model_parity_evidence.py"),
            str(parity_suite),
        ],
        cwd=root,
    )
    suite_document = validate_json_kind(
        parity_suite, "nnis-nnml1-multi-model-parity-suite-v1", head
    )
    if suite_document.get("distinct_checkpoint_count") != 2:
        raise QualificationError("composed parity suite does not contain exactly two registered checkpoints")

    require_clean_repository(root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "result": "pass",
        "nnis_git_commit": head,
        "nnis_git_dirty": False,
        "origin_main_commit": head,
        "device_ordinal_requested_for_tinyllama": args.device,
        "python_environments": {
            "smollm2": {
                "executable": str(smollm2_python),
                **smollm2_probe,
            },
            "tinyllama": {
                "executable": str(tinyllama_python),
                **tinyllama_probe,
            },
        },
        "artifacts": {
            "nnml0_real_safetensors": artifact_entry(nnml0_evidence),
            "smollm2_parity_record": artifact_entry(smollm2_parity),
            "tinyllama_parity_record": artifact_entry(tinyllama_parity),
            "nnml1_multi_model_parity_suite": artifact_entry(parity_suite),
        },
        "validated_checkpoint_records": [
            {
                "checkpoint_spec_name": smollm2_record["checkpoint_spec_name"],
                "parity_level": smollm2_record["parity_level"],
            },
            {
                "checkpoint_spec_name": tinyllama_record["checkpoint_spec_name"],
                "parity_level": tinyllama_record["parity_level"],
            },
        ],
        "promotion_authorized": False,
        "claim_boundary": (
            "P0 physical evidence bundle for exact registered checkpoints and NNML0 loader gate only; "
            "it does not establish multiple model-family admission, serving performance, or automatic runtime promotion"
        ),
    }
    bundle_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"P0_PHYSICAL_QUALIFICATION_OK head={head}")
    print(f"bundle_manifest={bundle_manifest}")
    print(f"nnml0_evidence={nnml0_evidence}")
    print(f"smollm2_parity_record={smollm2_parity}")
    print(f"tinyllama_parity_record={tinyllama_parity}")
    print(f"parity_suite={parity_suite}")
    return bundle_manifest


def self_test() -> None:
    smollm2 = validate_python_probe(
        {
            "python": "3.11.9",
            "torch": "2.4.0+test",
            "transformers": "4.40.1",
            "safetensors": "0.4.5",
            "huggingface_hub": "0.24.7",
        },
        SMOLLM2_ENV,
        "SmolLM2",
    )
    if smollm2["torch"] != "2.4.0+test":
        raise AssertionError("version probe normalization lost the recorded full torch version")
    try:
        validate_python_probe(
            {
                "python": "3.11.9",
                "torch": "2.4.1",
                "transformers": "4.40.1",
                "safetensors": "0.4.5",
                "huggingface_hub": "0.24.7",
            },
            SMOLLM2_ENV,
            "SmolLM2",
        )
    except QualificationError:
        pass
    else:
        raise AssertionError("drifted torch base version unexpectedly passed")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "repo"
        root.mkdir()
        outside = Path(temporary) / "evidence"
        require_work_dir_outside_repository(root, outside)
        try:
            require_work_dir_outside_repository(root, root / "evidence")
        except QualificationError:
            pass
        else:
            raise AssertionError("work directory inside repository unexpectedly passed")
        artifact = outside / "artifact.json"
        outside.mkdir()
        artifact.write_text('{"ok":true}\n', encoding="utf-8")
        entry = artifact_entry(artifact)
        if entry["bytes"] <= 0 or len(entry["sha256"]) != 64:
            raise AssertionError("artifact digest self-test failed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("P0 physical qualification bundle self-test passed")
        return
    try:
        run_bundle(args)
    except QualificationError as error:
        print(f"P0 physical qualification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
