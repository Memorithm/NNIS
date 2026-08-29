#!/usr/bin/env python3
"""Generate a BOS-only layerwise Transformers trace for NNIS qualification.

This diagnostic is intentionally checkpoint-specific and reuses the exact pinned
source/provenance checks from tiny_random_llama_fixture.py. For a one-token BOS
sequence with equal query/KV head counts, causal attention has one admissible
position, so softmax is exactly one and the pre-o_proj attention value is the
flattened V projection. That lets us compare the arithmetic path stage by stage
without introducing a second attention implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_random_llama_fixture import (
    MODEL_SHA256,
    REPO_ID,
    REVISION,
    TRANSFORMERS_VERSION,
    download_checkpoint,
    write_f32,
)

TRACE_FORMAT = "nnis-tiny-llama-bos-layerwise"
TRACE_VERSION = 1


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    delta = (actual.float() - expected.float()).detach().cpu().reshape(-1)
    if delta.numel() == 0:
        return 0.0, 0.0
    max_abs = float(delta.abs().max().item())
    rms = float(torch.sqrt(torch.mean(delta.double() * delta.double())).item())
    return max_abs, rms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"trusted trace requires Transformers {TRANSFORMERS_VERSION}; "
            f"got {transformers.__version__}"
        )

    checkpoint = download_checkpoint(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.to(device="cpu", dtype=torch.float32)
    model.eval()

    encoded = tokenizer("", return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(dtype=torch.long, device="cpu")
    ids = input_ids[0].tolist()
    if ids != [1]:
        raise RuntimeError(f"BOS-only trace expected input_ids [1], got {ids}")

    config = model.config
    if config.num_attention_heads != config.num_key_value_heads:
        raise RuntimeError("BOS shortcut requires equal query and KV head counts")

    args.output.mkdir(parents=True, exist_ok=True)
    stages: list[dict] = []

    def record(name: str, tensor: torch.Tensor) -> None:
        flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1)
        filename = name.replace(".", "__") + ".f32le"
        write_f32(args.output / filename, flat)
        stages.append({"name": name, "file": filename, "elements": flat.numel()})

    with torch.inference_mode():
        hidden = model.model.embed_tokens(input_ids)
        record("embedding", hidden)

        for index, layer in enumerate(model.model.layers):
            normed = layer.input_layernorm(hidden)
            record(f"layer{index}.input_norm", normed)

            value = layer.self_attn.v_proj(normed)
            record(f"layer{index}.v_proj", value)

            # With one causal token and equal Q/KV head counts, the softmax has
            # one element equal to 1.0 and the head-unflattening/reflattening is
            # an identity permutation. The attention value entering o_proj is V.
            projected = layer.self_attn.o_proj(value)
            record(f"layer{index}.o_proj", projected)

            residual = hidden + projected
            record(f"layer{index}.residual", residual)

            post_norm = layer.post_attention_layernorm(residual)
            record(f"layer{index}.post_attention_norm", post_norm)

            gate = layer.mlp.gate_proj(post_norm)
            record(f"layer{index}.gate_proj", gate)

            up = layer.mlp.up_proj(post_norm)
            record(f"layer{index}.up_proj", up)

            activated = layer.mlp.act_fn(gate)
            record(f"layer{index}.silu", activated)

            gated = activated * up
            record(f"layer{index}.gated", gated)

            mlp = layer.mlp.down_proj(gated)
            record(f"layer{index}.down_proj", mlp)

            hidden = residual + mlp
            record(f"layer{index}.hidden", hidden)

        final_norm = model.model.norm(hidden)
        record("final_norm", final_norm)

        manual_logits = model.lm_head(final_norm)[0, -1]
        record("logits", manual_logits)

        full_logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float().cpu()
        full_max_abs, full_rms = metrics(manual_logits, full_logits)
        if full_max_abs > 1.0e-6:
            raise RuntimeError(
                "BOS manual shortcut does not reproduce full Transformers forward: "
                f"max_abs={full_max_abs:.8e} rms={full_rms:.8e}"
            )

    manifest = {
        "format": TRACE_FORMAT,
        "version": TRACE_VERSION,
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "source_model_sha256": MODEL_SHA256,
        "transformers_version": transformers.__version__,
        "input_ids": ids,
        "stages": stages,
        "full_forward_max_abs": full_max_abs,
        "full_forward_rms": full_rms,
    }
    (args.output / "trace.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"checkpoint={REPO_ID}@{REVISION}")
    print(f"source_model_sha256={MODEL_SHA256}")
    print(f"transformers={transformers.__version__}")
    print(f"input_ids={ids}")
    print(f"stages={len(stages)}")
    print(f"full_forward_max_abs={full_max_abs:.8e}")
    print(f"full_forward_rms={full_rms:.8e}")
    print(f"trace_dir={args.output}")


if __name__ == "__main__":
    main()
