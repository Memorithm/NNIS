#!/usr/bin/env python3
"""Build the one supported reference fixture for NNIS model validation.

This tool is intentionally checkpoint-specific. It does not implement a generic
Hugging Face loader. It downloads one pinned tiny random Llama revision,
converts its weights into NNIS's explicit internal matrix orientation, and
records trusted Transformers logits for the same tokenizer IDs and positions.

Dependencies:
    python -m pip install 'torch>=2.3' 'transformers==4.43.3' \
        'safetensors>=0.4' 'huggingface_hub>=0.24'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "amakhov/tiny-random-llama"
REVISION = "99160cb087861a1e3c54ff5d3f45fd9488d9c04e"
MODEL_SHA256 = "a4eb5dcdfc71d3a8f297bb1c2a672d3babe04f102480addde293210778805d30"
TRANSFORMERS_VERSION = "4.43.3"
DEFAULT_PROMPT = "Hello, NNIS!"
DEFAULT_DECODE_STEPS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(cache_dir: Path | None) -> Path:
    root = Path(
        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            cache_dir=None if cache_dir is None else str(cache_dir),
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
    )
    actual = sha256(root / "model.safetensors")
    if actual != MODEL_SHA256:
        raise RuntimeError(
            f"pinned model.safetensors SHA256 mismatch: {actual} != {MODEL_SHA256}"
        )
    return root


def expected_source_names(num_layers: int) -> set[str]:
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}"
        names.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        )
    return names


def check_source_config(config: dict) -> None:
    expected = {
        "model_type": "llama",
        "hidden_act": "silu",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "vocab_size": 32000,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "rope_scaling": None,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"pinned checkpoint contract changed: config[{key!r}]={config.get(key)!r}, expected {value!r}"
            )
    if not math.isclose(float(config["rms_norm_eps"]), 1.0e-5, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("unexpected rms_norm_eps")
    if not math.isclose(float(config["rope_theta"]), 10000.0, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("unexpected rope_theta")
    dtype = config.get("torch_dtype", config.get("dtype"))
    if dtype != "float32":
        raise RuntimeError(f"expected float32 checkpoint, got {dtype!r}")


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for value in contiguous.view(-1).tolist():
            handle.write(struct.pack("<f", value))


def tensor_entry(name: str, tensor: torch.Tensor, file_name: str) -> dict:
    return {
        "name": name,
        "dtype": "f32",
        "shape": list(tensor.shape),
        "file": file_name,
    }


def convert_weights(checkpoint: Path, output: Path) -> None:
    config = json.loads((checkpoint / "config.json").read_text())
    check_source_config(config)
    source = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    expected = expected_source_names(config["num_hidden_layers"])
    actual = set(source)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"unexpected Safetensors key set; missing={missing}, extra={extra}")
    if any(tensor.dtype != torch.float32 for tensor in source.values()):
        bad = sorted(name for name, tensor in source.items() if tensor.dtype != torch.float32)
        raise RuntimeError(f"non-f32 source tensors: {bad}")

    if output.exists():
        shutil.rmtree(output)
    tensor_dir = output / "tensors"
    tensor_dir.mkdir(parents=True)
    manifest_tensors: list[dict] = []

    def store(target: str, source_name: str, transpose: bool = False) -> None:
        tensor = source[source_name]
        if transpose:
            if tensor.ndim != 2:
                raise RuntimeError(f"cannot transpose non-matrix {source_name}")
            tensor = tensor.transpose(0, 1).contiguous()
        file_name = f"tensors/{target}.f32le"
        write_f32(output / file_name, tensor)
        manifest_tensors.append(tensor_entry(target, tensor, file_name))

    store("token_embedding", "model.embed_tokens.weight")
    for layer in range(config["num_hidden_layers"]):
        src = f"model.layers.{layer}"
        dst = f"layers.{layer}"
        store(f"{dst}.input_norm", f"{src}.input_layernorm.weight")
        store(f"{dst}.q_proj", f"{src}.self_attn.q_proj.weight", transpose=True)
        store(f"{dst}.k_proj", f"{src}.self_attn.k_proj.weight", transpose=True)
        store(f"{dst}.v_proj", f"{src}.self_attn.v_proj.weight", transpose=True)
        store(f"{dst}.o_proj", f"{src}.self_attn.o_proj.weight", transpose=True)
        store(
            f"{dst}.post_attention_norm",
            f"{src}.post_attention_layernorm.weight",
        )
        store(f"{dst}.gate_proj", f"{src}.mlp.gate_proj.weight", transpose=True)
        store(f"{dst}.up_proj", f"{src}.mlp.up_proj.weight", transpose=True)
        store(f"{dst}.down_proj", f"{src}.mlp.down_proj.weight", transpose=True)
    store("final_norm", "model.norm.weight")
    store("lm_head", "lm_head.weight", transpose=True)

    nnis_config = {
        "vocab_size": config["vocab_size"],
        "hidden_size": config["hidden_size"],
        "intermediate_size": config["intermediate_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "num_key_value_heads": config["num_key_value_heads"],
        "max_position_embeddings": config["max_position_embeddings"],
        "rms_norm_eps": config["rms_norm_eps"],
        "rope_theta": config["rope_theta"],
        "activation": "silu",
        "weight_dtype": "f32",
    }
    manifest = {
        "format": "nnis-model",
        "version": 1,
        "config": nnis_config,
        "tensors": manifest_tensors,
    }
    (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    provenance = {
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "source_model_sha256": MODEL_SHA256,
        "converter": Path(__file__).name,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def reference_logits(
    checkpoint: Path,
    output: Path,
    prompt: str,
    decode_steps: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(dtype=torch.long, device="cpu")
    if input_ids.shape[0] != 1:
        raise RuntimeError("reference harness supports one sequence")

    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "nnis-reference-logits",
        "version": 1,
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "source_model_sha256": MODEL_SHA256,
        "transformers_version": TRANSFORMERS_VERSION,
        "prompt": prompt,
        "input_ids": input_ids[0].tolist(),
        "decode_steps": decode_steps,
        "dtype": "f32",
        "logit_files": [],
        "greedy_ids": [],
    }

    with torch.inference_mode():
        result = model(input_ids=input_ids, use_cache=True)
        logits = result.logits[0, -1].float().cpu()
        file_name = "prefill_logits.f32le"
        write_f32(output / file_name, logits)
        metadata["logit_files"].append(file_name)
        past = result.past_key_values

        for step in range(decode_steps):
            token = int(torch.argmax(logits).item())
            metadata["greedy_ids"].append(token)
            current = torch.tensor([[token]], dtype=torch.long)
            result = model(input_ids=current, past_key_values=past, use_cache=True)
            logits = result.logits[0, -1].float().cpu()
            file_name = f"decode_{step:02d}_logits.f32le"
            write_f32(output / file_name, logits)
            metadata["logit_files"].append(file_name)
            past = result.past_key_values

    (output / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    args = parser.parse_args()
    if args.decode_steps < 0:
        parser.error("--decode-steps must be non-negative")

    checkpoint = download_checkpoint(args.cache_dir)
    model_dir = args.output / "model"
    reference_dir = args.output / "reference"
    convert_weights(checkpoint, model_dir)
    reference_logits(checkpoint, reference_dir, args.prompt, args.decode_steps)
    print(f"checkpoint={REPO_ID}@{REVISION}")
    print(f"model_dir={model_dir}")
    print(f"reference_dir={reference_dir}")


if __name__ == "__main__":
    main()
