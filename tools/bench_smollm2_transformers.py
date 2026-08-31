#!/usr/bin/env python3
"""Comparable end-to-end CUDA baseline for the pinned SmolLM2-135M probe.

This is a performance harness, not a correctness oracle. It verifies the exact
checkpoint bytes and the already-qualified two-token greedy prefix, then times a
cached autoregressive loop on CUDA. The generated token remains on device and
is fed back into the model without a per-step host observation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

SOURCE_REPO = "HuggingFaceTB/SmolLM2-135M"
SOURCE_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SOURCE_MODEL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
DEFAULT_INPUT_IDS = [22007, 6463, 314]
QUALIFIED_GREEDY_PREFIX = [260, 3075]


def parse_csv_ids(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid token id list: {error}") from error
    if not result or any(token < 0 for token in result):
        raise argparse.ArgumentTypeError("input IDs must be non-empty non-negative integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--input-ids", type=parse_csv_ids, default=DEFAULT_INPUT_IDS)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--attn-implementation",
        default="default",
        choices=["default", "eager", "sdpa", "flash_attention_2"],
        help="Transformers attention implementation. 'default' leaves model selection untouched.",
    )
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.decode_steps <= 0:
        parser.error("--decode-steps must be greater than zero")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be greater than zero")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def summarize(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms or any(not math.isfinite(value) or value < 0 for value in samples_ms):
        raise RuntimeError("timing samples must be finite, non-negative and non-empty")
    ordered = sorted(samples_ms)
    mean_ms = statistics.fmean(ordered)
    variance = statistics.fmean((value - mean_ms) ** 2 for value in ordered)
    return {
        "min_ms": ordered[0],
        "median_ms": percentile(ordered, 0.5),
        "mean_ms": mean_ms,
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "max_ms": ordered[-1],
        "stddev_ms": math.sqrt(variance),
    }


def memory_snapshot(torch: Any, device: Any) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def generate_once(torch: Any, model: Any, input_ids: list[int], decode_steps: int, device: Any) -> list[int]:
    prompt = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = []
    with torch.inference_mode():
        outputs = model(input_ids=prompt, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        for _ in range(decode_steps):
            token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(token)
            outputs = model(input_ids=token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
    return torch.cat(generated, dim=1).to("cpu").squeeze(0).tolist()


def require_qualified_prefix(input_ids: list[int], generated: list[int]) -> bool:
    if input_ids != DEFAULT_INPUT_IDS or len(generated) < len(QUALIFIED_GREEDY_PREFIX):
        return False
    actual = generated[: len(QUALIFIED_GREEDY_PREFIX)]
    if actual != QUALIFIED_GREEDY_PREFIX:
        raise RuntimeError(
            f"qualified SmolLM2 greedy prefix changed: actual={actual}, expected={QUALIFIED_GREEDY_PREFIX}"
        )
    return True


def main() -> None:
    args = parse_args()

    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Transformers performance baseline")
    if args.device >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {args.device} is out of range; visible devices={torch.cuda.device_count()}"
        )

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=SOURCE_REPO,
            filename="model.safetensors",
            revision=SOURCE_REVISION,
        )
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != SOURCE_MODEL_SHA256:
        raise RuntimeError(
            f"checkpoint hash mismatch: actual={checkpoint_sha256}, expected={SOURCE_MODEL_SHA256}"
        )

    gc.collect()
    torch.cuda.empty_cache()
    before_model = memory_snapshot(torch, device)

    load_kwargs: dict[str, Any] = {
        "revision": SOURCE_REVISION,
        "torch_dtype": torch.float32,
    }
    if args.attn_implementation != "default":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(SOURCE_REPO, **load_kwargs)
    model.eval()
    model.to(device)
    torch.cuda.synchronize(device)
    after_model = memory_snapshot(torch, device)

    resolved_attention = getattr(model.config, "_attn_implementation", None)
    warmup_generated: list[int] | None = None
    for _ in range(args.warmups):
        warmup_generated = generate_once(torch, model, args.input_ids, args.decode_steps, device)
        require_qualified_prefix(args.input_ids, warmup_generated)
    torch.cuda.synchronize(device)
    after_warmup = memory_snapshot(torch, device)

    samples_ms: list[float] = []
    expected_generated: list[int] | None = None
    qualified_prefix_checked = False
    peak_allocated_bytes = after_model["allocated_bytes"]
    peak_reserved_bytes = after_model["reserved_bytes"]

    for _ in range(args.iterations):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        generated = generate_once(torch, model, args.input_ids, args.decode_steps, device)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples_ms.append(elapsed_ms)
        qualified_prefix_checked |= require_qualified_prefix(args.input_ids, generated)
        if expected_generated is None:
            expected_generated = generated
        elif generated != expected_generated:
            raise RuntimeError(
                "non-deterministic greedy output across benchmark iterations: "
                f"expected={expected_generated}, actual={generated}"
            )
        peak_allocated_bytes = max(
            peak_allocated_bytes, int(torch.cuda.max_memory_allocated(device))
        )
        peak_reserved_bytes = max(
            peak_reserved_bytes, int(torch.cuda.max_memory_reserved(device))
        )

    timing = summarize(samples_ms)
    median_ms = timing["median_ms"]
    if median_ms <= 0:
        raise RuntimeError("end-to-end timer returned a non-positive median duration")

    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    report = {
        "schema_version": 1,
        "benchmark": "smollm2-135m-greedy-e2e",
        "backend": "transformers",
        "measurement": (
            "host-wall-clock bracketed by torch.cuda.synchronize; model load excluded; "
            "prefill plus fixed-length cached decode included"
        ),
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "source_weight_dtype": "bfloat16",
        "execution_weight_dtype": "f32",
        "input_ids": args.input_ids,
        "decode_steps": args.decode_steps,
        "warmup_iterations": args.warmups,
        "iterations": args.iterations,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_cuda": torch.version.cuda,
            "device_ordinal": args.device,
            "gpu_name": torch.cuda.get_device_name(device),
            "compute_capability_major": capability[0],
            "compute_capability_minor": capability[1],
            "multiprocessor_count": int(properties.multi_processor_count),
            "attention_implementation_requested": args.attn_implementation,
            "attention_implementation_resolved": resolved_attention,
        },
        "memory": {
            "before_model": before_model,
            "after_model": after_model,
            "after_warmup": after_warmup,
            "model_free_memory_delta_bytes": max(
                0, before_model["free_bytes"] - after_model["free_bytes"]
            ),
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
            "peak_generation_extra_allocated_bytes": max(
                0, peak_allocated_bytes - after_model["allocated_bytes"]
            ),
        },
        "generation": {
            "statistics": timing,
            "samples_ms": samples_ms,
        },
        "generated_tokens_per_second_median": args.decode_steps / (median_ms / 1000.0),
        "generated_ids": expected_generated,
        "qualified_greedy_prefix_checked": qualified_prefix_checked,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
