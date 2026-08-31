#!/usr/bin/env python3
"""Emit compact layerwise SmolLM2-135M reference vectors for NNIS diagnostics.

This is checkpoint-specific and diagnostic-only. It loads the same pinned
SmolLM2-135M checkpoint used by the NNIS trained-model qualification harness,
widens the persisted BF16 weights to f32, and records the last-token hidden
vector after every decoder layer plus the final RMSNorm output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

REPO_ID = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
TRANSFORMERS_VERSION = "4.40.1"
INPUT_IDS = [22007, 6463, 314]
HIDDEN_SIZE = 576
LAYERS = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    vector = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if vector.ndim != 1 or vector.numel() != HIDDEN_SIZE:
        raise RuntimeError(f"expected [{HIDDEN_SIZE}] vector, got {tuple(vector.shape)}")
    vector.numpy().astype("<f4", copy=False).tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"requires Transformers {TRANSFORMERS_VERSION}; got {transformers.__version__}"
        )

    checkpoint = Path(
        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            cache_dir=None if args.cache_dir is None else str(args.cache_dir),
            local_files_only=True,
        )
    )
    actual_hash = sha256(checkpoint / "model.safetensors")
    if actual_hash != MODEL_SHA256:
        raise RuntimeError(f"model SHA256 mismatch: {actual_hash} != {MODEL_SHA256}")

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map=None,
    ).cpu().eval()
    if len(model.model.layers) != LAYERS:
        raise RuntimeError(f"expected {LAYERS} layers, got {len(model.model.layers)}")
    if model.config.hidden_size != HIDDEN_SIZE:
        raise RuntimeError(
            f"expected hidden size {HIDDEN_SIZE}, got {model.config.hidden_size}"
        )

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        def capture(_module, _inputs, output, *, layer_index=index):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_index] = hidden[0, -1].detach().float().cpu().clone()

        handles.append(layer.register_forward_hook(capture))

    ids = torch.tensor([INPUT_IDS], dtype=torch.long)
    with torch.inference_mode():
        embedding = model.model.embed_tokens(ids)[0, -1].float().cpu().clone()
        result = model.model(input_ids=ids, use_cache=False, return_dict=True)
        final_norm = result.last_hidden_state[0, -1].float().cpu().clone()

    for handle in handles:
        handle.remove()
    if set(captured) != set(range(LAYERS)):
        raise RuntimeError(f"missing layer captures: {sorted(set(range(LAYERS)) - set(captured))}")

    args.output.mkdir(parents=True, exist_ok=True)
    stages = []

    def store(name: str, vector: torch.Tensor) -> None:
        file_name = f"{name}.f32le"
        write_f32(args.output / file_name, vector)
        stages.append({"name": name, "file": file_name, "elements": HIDDEN_SIZE})

    store("embedding", embedding)
    for index in range(LAYERS):
        store(f"layer{index:02d}.hidden", captured[index])
    store("final_norm", final_norm)

    manifest = {
        "format": "nnis-smollm2-layerwise",
        "version": 1,
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "source_model_sha256": MODEL_SHA256,
        "transformers_version": TRANSFORMERS_VERSION,
        "source_weight_dtype": "bfloat16",
        "execution_weight_dtype": "f32",
        "input_ids": INPUT_IDS,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": LAYERS,
        "stages": stages,
    }
    (args.output / "trace.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"trace_dir={args.output}")
    print(f"stages={len(stages)}")


if __name__ == "__main__":
    main()
