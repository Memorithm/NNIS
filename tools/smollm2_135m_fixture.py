#!/usr/bin/env python3
"""Build the pinned trained SmolLM2-135M reference fixture for NNIS.

This is intentionally checkpoint-specific. It validates one exact upstream
revision, widens its persisted BF16 values to f32 for NNIS's current model
execution path, materializes the tied LM head in NNIS's internal matrix
orientation, and records trusted Transformers logits from the same widened
weights.

Dependencies:
    python -m pip install -r tools/requirements-smollm2-135m.txt
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_SHA256 = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
TRANSFORMERS_VERSION = "4.40.1"
DEFAULT_PROMPT = "Gravity is"
DEFAULT_DECODE_STEPS = 2


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
                "merges.txt",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
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
    # The upstream checkpoint ties lm_head.weight to model.embed_tokens.weight,
    # so Safetensors persists the shared storage once and omits lm_head.weight.
    names = {"model.embed_tokens.weight", "model.norm.weight"}
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
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_act": "silu",
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "max_position_embeddings": 8192,
        "vocab_size": 49152,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "rope_scaling": None,
        "rope_interleaved": False,
        "torch_dtype": "bfloat16",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"pinned checkpoint contract changed: config[{key!r}]={config.get(key)!r}, expected {value!r}"
            )
    if not math.isclose(
        float(config["rms_norm_eps"]), 1.0e-5, rel_tol=0.0, abs_tol=0.0
    ):
        raise RuntimeError("unexpected rms_norm_eps")
    if not math.isclose(
        float(config["rope_theta"]), 100000.0, rel_tol=0.0, abs_tol=0.0
    ):
        raise RuntimeError("unexpected rope_theta")
    if config["num_attention_heads"] % config["num_key_value_heads"] != 0:
        raise RuntimeError("query heads are not divisible by KV heads")
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    if head_dim != 64:
        raise RuntimeError(f"unexpected head_dim {head_dim}")


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    # NumPy exposes the contiguous f32 tensor without a per-scalar Python loop.
    array = contiguous.numpy().astype("<f4", copy=False)
    array.tofile(path)


def tensor_entry(name: str, tensor: torch.Tensor, file_name: str) -> dict:
    return {
        "name": name,
        "dtype": "f32",
        "shape": list(tensor.shape),
        "file": file_name,
    }


def convert_weights(checkpoint: Path, output: Path, tokenizer_sha256: str) -> None:
    config = json.loads((checkpoint / "config.json").read_text())
    check_source_config(config)
    source = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    expected = expected_source_names(config["num_hidden_layers"])
    actual = set(source)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"unexpected Safetensors key set; missing={missing}, extra={extra}")
    if any(tensor.dtype != torch.bfloat16 for tensor in source.values()):
        bad = sorted(
            f"{name}:{tensor.dtype}" for name, tensor in source.items() if tensor.dtype != torch.bfloat16
        )
        raise RuntimeError(f"non-BF16 source tensors: {bad}")

    if output.exists():
        shutil.rmtree(output)
    tensor_dir = output / "tensors"
    tensor_dir.mkdir(parents=True)
    manifest_tensors: list[dict] = []

    def store_tensor(target: str, tensor: torch.Tensor) -> None:
        file_name = f"tensors/{target}.f32le"
        write_f32(output / file_name, tensor)
        manifest_tensors.append(tensor_entry(target, tensor, file_name))

    def store(target: str, source_name: str, transpose: bool = False) -> None:
        tensor = source[source_name]
        if transpose:
            if tensor.ndim != 2:
                raise RuntimeError(f"cannot transpose non-matrix {source_name}")
            tensor = tensor.transpose(0, 1).contiguous()
        store_tensor(target, tensor)

    embedding = source["model.embed_tokens.weight"]
    store_tensor("token_embedding", embedding)
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
    # Hugging Face computes logits with the tied [vocab, hidden] embedding
    # matrix. NNIS's GEMM convention stores lm_head as [hidden, vocab].
    store_tensor("lm_head", embedding.transpose(0, 1).contiguous())

    nnis_config = {
        "vocab_size": config["vocab_size"],
        "eos_token_id": config["eos_token_id"],
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
        "source_weight_dtype": "bfloat16",
        "execution_weight_dtype": "f32",
        "tokenizer_sha256": tokenizer_sha256,
        "tied_lm_head_materialized": True,
        "converter": Path(__file__).name,
        "transformers_version": transformers.__version__,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def reference_logits(
    checkpoint: Path,
    output: Path,
    prompt: str,
    decode_steps: int,
    tokenizer_sha256: str,
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
    if model.model.embed_tokens.weight.dtype != torch.float32:
        raise RuntimeError("reference embedding was not widened to f32")
    if model.lm_head.weight.dtype != torch.float32:
        raise RuntimeError("reference lm_head was not widened to f32")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise RuntimeError("Transformers reference did not preserve tied LM-head storage")

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(dtype=torch.long, device="cpu")
    if input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise RuntimeError("reference harness requires one non-empty sequence")

    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "nnis-reference-logits",
        "version": 1,
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "source_model_sha256": MODEL_SHA256,
        "transformers_version": transformers.__version__,
        "source_weight_dtype": "bfloat16",
        "execution_weight_dtype": "f32",
        "tokenizer_sha256": tokenizer_sha256,
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
    del model
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    args = parser.parse_args()
    if args.decode_steps < 0:
        parser.error("--decode-steps must be non-negative")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            "trusted reference requires Transformers "
            f"{TRANSFORMERS_VERSION}; got {transformers.__version__}"
        )

    checkpoint = download_checkpoint(args.cache_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer_file = args.output / "tokenizer.json"
    shutil.copy2(checkpoint / "tokenizer.json", tokenizer_file)
    tokenizer_digest = sha256(tokenizer_file)
    model_dir = args.output / "model"
    reference_dir = args.output / "reference"
    convert_weights(checkpoint, model_dir, tokenizer_digest)
    gc.collect()
    reference_logits(
        checkpoint, reference_dir, args.prompt, args.decode_steps, tokenizer_digest
    )
    print(f"checkpoint={REPO_ID}@{REVISION}")
    print(f"transformers={transformers.__version__}")
    print(f"tokenizer_sha256={tokenizer_digest}")
    print(f"model_dir={model_dir}")
    print(f"tokenizer_file={tokenizer_file}")
    print(f"reference_dir={reference_dir}")


if __name__ == "__main__":
    main()
