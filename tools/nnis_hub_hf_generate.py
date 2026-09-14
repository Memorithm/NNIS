#!/usr/bin/env python3
"""Stable artifact-oriented process surface for local-HF NNIS generation.

The process delegates model loading, CUDA execution, tokenization and greedy
sampling to the native ``nnis-hf generate`` binary. It adds only a bounded,
versioned JSON artifact boundary suitable for orchestrators such as SciRust Hub.
It does not reinterpret model admission, authorize promotion, or create timing,
quality, memory, or cross-runtime equivalence evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import selectors
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "nnis.hf-generation@1.0.0"
MEDIA_TYPE = "application/vnd.nnis.hf-generation.v1+json"
MAX_PROMPT_BYTES = 1024 * 1024
MAX_NEW_TOKENS = 65_536
MAX_CAPTURE_BYTES = 16 * 1024 * 1024
MAX_DEVICE_ORDINAL = 2_147_483_647
DEFAULT_TIMEOUT_SECONDS = 3600.0
READ_CHUNK_BYTES = 64 * 1024
MAX_ERROR_BYTES = 4096


class ProcessContractError(ValueError):
    pass


def _require_model_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ProcessContractError(f"model path must be a regular directory, not a symlink: {path}")


def _require_tokenizer(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ProcessContractError(f"tokenizer path must be a regular file, not a symlink: {path}")


def _validate_request(
    *,
    model_dir: Path,
    tokenizer_file: Path,
    prompt: str,
    device_ordinal: int,
    max_new_tokens: int,
) -> None:
    _require_model_directory(model_dir)
    _require_tokenizer(tokenizer_file)
    if not isinstance(prompt, str) or "\0" in prompt:
        raise ProcessContractError("prompt must be a string without NUL bytes")
    try:
        prompt_bytes = prompt.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ProcessContractError("prompt must be valid UTF-8") from error
    if not prompt_bytes or len(prompt_bytes) > MAX_PROMPT_BYTES:
        raise ProcessContractError(
            f"prompt must encode to 1..={MAX_PROMPT_BYTES} UTF-8 bytes; got {len(prompt_bytes)}"
        )
    if type(device_ordinal) is not int or not 0 <= device_ordinal <= MAX_DEVICE_ORDINAL:
        raise ProcessContractError(f"device ordinal must be an integer in [0, {MAX_DEVICE_ORDINAL}]")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ProcessContractError(
            f"max-new-tokens must be an integer in [1, {MAX_NEW_TOKENS}]; got {max_new_tokens}"
        )


def _run_native_generation(
    *,
    nnis_hf_bin: str,
    model_dir: Path,
    tokenizer_file: Path,
    prompt: str,
    device_ordinal: int,
    max_new_tokens: int,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[bytes, bytes]:
    """Drain both POSIX pipes incrementally, enforcing limits before retaining bytes.

    The child stays in the caller's process group so Hub cancellation can reach
    the entire run. On failure we kill/reap the direct native child; this is not
    hostile-code isolation or independent descendant supervision.
    """
    if os.name != "posix":
        raise ProcessContractError("bounded native generation capture requires POSIX")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ProcessContractError("timeout-seconds must be positive and finite")
    command = [
        nnis_hf_bin,
        "generate",
        "--model",
        str(model_dir),
        "--tokenizer",
        str(tokenizer_file),
        "--prompt",
        prompt,
        "--device",
        str(device_ordinal),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    try:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            bufsize=0,
        )
    except OSError as error:
        raise RuntimeError(f"cannot execute native NNIS generator {nnis_hf_bin!r}: {error}") from error

    streams = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout_seconds
    try:
        with selectors.DefaultSelector() as selector:
            for label, pipe in (("stdout", child.stdout), ("stderr", child.stderr)):
                assert pipe is not None
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, label)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("native NNIS generation exceeded timeout-seconds")
                for key, _events in selector.select(timeout=min(remaining, 0.1)):
                    output = streams[key.data]
                    # Read at most one overflow byte; never retain it in the capture.
                    size = min(READ_CHUNK_BYTES, MAX_CAPTURE_BYTES - len(output) + 1)
                    try:
                        chunk = os.read(key.fd, size)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    if len(output) + len(chunk) > MAX_CAPTURE_BYTES:
                        raise RuntimeError(
                            f"native NNIS {key.data} exceeds the process-contract capture budget"
                        )
                    output.extend(chunk)
        # EOF alone is not process completion: a child can close both pipes and hang.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("native NNIS generation exceeded timeout-seconds")
        try:
            returncode = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("native NNIS generation exceeded timeout-seconds") from error
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        for pipe in (child.stdout, child.stderr):
            if pipe is not None:
                pipe.close()

    if returncode != 0:
        stderr = bytes(streams["stderr"][:MAX_ERROR_BYTES]).decode("utf-8", errors="replace").strip()
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"native nnis-hf generate failed with exit {returncode}{detail}")
    return bytes(streams["stdout"]), bytes(streams["stderr"])


def _decode_utf8(data: bytes, label: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"native NNIS {label} is not valid UTF-8: {error}") from error


def build_result(
    *,
    model_dir: Path,
    tokenizer_file: Path,
    prompt: str,
    device_ordinal: int,
    max_new_tokens: int,
    stdout: bytes,
    stderr: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "media_type": MEDIA_TYPE,
        "status": "generated",
        "execution_scope": "local_hf_f32_greedy_generation",
        "inputs": {
            "model_directory": str(model_dir),
            "tokenizer_file": str(tokenizer_file),
            "prompt": prompt,
            "device_ordinal": device_ordinal,
            "max_new_tokens": max_new_tokens,
        },
        "output": {
            "stdout_utf8": _decode_utf8(stdout, "stdout"),
            "stdout_bytes": len(stdout),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_utf8": _decode_utf8(stderr, "stderr"),
            "stderr_bytes": len(stderr),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        },
        "sampling": "greedy",
        "network_access": False,
        "promotion_authorized": False,
        "serving_performance_verified": False,
        "numerical_equivalence_verified": False,
        "general_model_family_support_verified": False,
        "claim_boundary": (
            "Executed local-HF F32 greedy generation through native nnis-hf only; this artifact "
            "does not establish numerical equivalence, model quality, serving performance, "
            "general model-family support, or runtime promotion eligibility"
        ),
    }


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    """Publish complete JSON with an atomic no-replace link in a trusted directory.

    Only our temporary file is cleaned up. A concurrent publisher's destination
    must never be overwritten or removed, including when it is a symlink.
    """
    if path.exists() or path.is_symlink():
        raise ProcessContractError(f"result path already exists: {path}")
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".nnis-result-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Unlike rename/replace, link fails atomically if another writer won.
        os.link(temporary, path)
    except FileExistsError as error:
        raise ProcessContractError(f"result path already exists: {path}") from error
    except OSError as error:
        raise RuntimeError(f"cannot write generation result {path}: {error}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                # Do not mask a primary failure or turn a published result into failure.
                pass


def execute(
    *,
    nnis_hf_bin: str,
    model_dir: Path,
    tokenizer_file: Path | None,
    prompt: str,
    device_ordinal: int,
    max_new_tokens: int,
    result_path: Path,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    resolved_tokenizer = tokenizer_file or model_dir / "tokenizer.json"
    _validate_request(
        model_dir=model_dir,
        tokenizer_file=resolved_tokenizer,
        prompt=prompt,
        device_ordinal=device_ordinal,
        max_new_tokens=max_new_tokens,
    )
    if result_path.exists() or result_path.is_symlink():
        raise ProcessContractError(f"result path already exists: {result_path}")
    stdout, stderr = _run_native_generation(
        nnis_hf_bin=nnis_hf_bin,
        model_dir=model_dir,
        tokenizer_file=resolved_tokenizer,
        prompt=prompt,
        device_ordinal=device_ordinal,
        max_new_tokens=max_new_tokens,
        timeout_seconds=timeout_seconds,
    )
    result = build_result(
        model_dir=model_dir,
        tokenizer_file=resolved_tokenizer,
        prompt=prompt,
        device_ordinal=device_ordinal,
        max_new_tokens=max_new_tokens,
        stdout=stdout,
        stderr=stderr,
    )
    _write_new_json(result_path, result)
    return result


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model = root / "model"
        model.mkdir()
        (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        fake = root / "nnis-hf"
        trace = root / "trace.json"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['NNIS_FAKE_TRACE']).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
            "if os.environ.get('NNIS_FAKE_FAIL') == '1':\n"
            "    print('synthetic failure', file=sys.stderr)\n"
            "    raise SystemExit(7)\n"
            "print('synthetic generation')\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)

        old_trace = os.environ.get("NNIS_FAKE_TRACE")
        old_fail = os.environ.get("NNIS_FAKE_FAIL")
        os.environ["NNIS_FAKE_TRACE"] = str(trace)
        os.environ.pop("NNIS_FAKE_FAIL", None)
        try:
            result_path = root / "result.json"
            result = execute(
                nnis_hf_bin=str(fake),
                model_dir=model,
                tokenizer_file=None,
                prompt="Hello from Hub",
                device_ordinal=2,
                max_new_tokens=7,
                result_path=result_path,
            )
            assert result["contract"] == CONTRACT
            assert result["status"] == "generated"
            assert result["output"]["stdout_utf8"] == "synthetic generation\n"
            assert result["output"]["stdout_sha256"] == hashlib.sha256(
                b"synthetic generation\n"
            ).hexdigest()
            assert result["promotion_authorized"] is False
            assert result["serving_performance_verified"] is False
            assert json.loads(result_path.read_text(encoding="utf-8")) == result
            assert json.loads(trace.read_text(encoding="utf-8")) == [
                "generate",
                "--model",
                str(model),
                "--tokenizer",
                str(model / "tokenizer.json"),
                "--prompt",
                "Hello from Hub",
                "--device",
                "2",
                "--max-new-tokens",
                "7",
            ]

            preexisting = root / "preexisting.json"
            preexisting.write_text("{}\n", encoding="utf-8")
            try:
                execute(
                    nnis_hf_bin=str(fake),
                    model_dir=model,
                    tokenizer_file=None,
                    prompt="x",
                    device_ordinal=0,
                    max_new_tokens=1,
                    result_path=preexisting,
                )
            except ProcessContractError:
                pass
            else:
                raise AssertionError("pre-existing result path unexpectedly accepted")

            os.environ["NNIS_FAKE_FAIL"] = "1"
            failed_result = root / "failed.json"
            try:
                execute(
                    nnis_hf_bin=str(fake),
                    model_dir=model,
                    tokenizer_file=None,
                    prompt="x",
                    device_ordinal=0,
                    max_new_tokens=1,
                    result_path=failed_result,
                )
            except RuntimeError as error:
                assert "exit 7" in str(error)
            else:
                raise AssertionError("failed native generation unexpectedly succeeded")
            assert not failed_result.exists()
        finally:
            if old_trace is None:
                os.environ.pop("NNIS_FAKE_TRACE", None)
            else:
                os.environ["NNIS_FAKE_TRACE"] = old_trace
            if old_fail is None:
                os.environ.pop("NNIS_FAKE_FAIL", None)
            else:
                os.environ["NNIS_FAKE_FAIL"] = old_fail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--nnis-hf-bin", default="nnis-hf")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("NNIS Hub local-HF generation process self-test passed")
        return 0
    if args.model is None or args.prompt is None or args.result is None:
        parser.error("--model, --prompt and --result are required unless --self-test is used")

    try:
        result = execute(
            nnis_hf_bin=args.nnis_hf_bin,
            model_dir=args.model,
            tokenizer_file=args.tokenizer,
            prompt=args.prompt,
            device_ordinal=args.device,
            max_new_tokens=args.max_new_tokens,
            result_path=args.result,
            timeout_seconds=args.timeout_seconds,
        )
    except ProcessContractError as error:
        print(f"error: {error}", file=os.sys.stderr)
        return 2
    except Exception as error:  # fail closed on unexpected process/runtime errors
        print(f"internal error: {error}", file=os.sys.stderr)
        return 3

    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
