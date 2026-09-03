#!/usr/bin/env python3
"""Build a pinned trained TinyLlama-1.1B-Chat-v1.0 fixture and greedy oracle suite.

The converter is intentionally checkpoint-specific. It verifies the exact upstream
revision and model SHA256, persists the NNIS model-format-v1 F32 base graph, and
records deterministic Transformers F32 greedy trajectories for a broad prompt suite.
The resulting suite is designed for `llama_f16_massive_abba` and is evidence, not a
claim of general Llama support.
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

REPO_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "d9128824c0c80111be21424e68086f52413fb413"
MODEL_SHA256 = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
TRANSFORMERS_VERSION = "4.43.3"
TARGET_PROMPT_TOKENS = (8, 32, 128, 512, 1024)
STANDARD_DECODE_STEPS = 32
DEEP_DECODE_STEPS = 128
PROMPT_FAMILIES = {
    "prose": (
        "Native inference runtimes should preserve numerical semantics while making "
        "memory layout, kernel selection, and measurement methodology explicit. "
        "A useful benchmark reports both successful optimizations and negative results. "
    ),
    "code": (
        "fn accumulate(values: &[f32]) -> f32 { values.iter().copied().sum() } "
        "In Rust, explain the ownership, iterator, and numerical behavior of this function. "
    ),
    "math": (
        "Consider a sequence defined by a_0 = 1 and a_{n+1} = (a_n + 3/a_n) / 2. "
        "Analyze its convergence and state which invariant or bound justifies each step. "
    ),
}


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


def check_source_config(config: dict) -> None:
    expected = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_hidden_layers": 22,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "max_position_embeddings": 2048,
        "vocab_size": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "rope_scaling": None,
        "torch_dtype": "bfloat16",
        "pretraining_tp": 1,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"pinned checkpoint contract changed: config[{key!r}]={config.get(key)!r}, expected {value!r}"
            )
    if not math.isclose(float(config["rms_norm_eps"]), 1.0e-5, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("unexpected rms_norm_eps")
    if not math.isclose(float(config["rope_theta"]), 10_000.0, rel_tol=0.0, abs_tol=0.0):
        raise RuntimeError("unexpected rope_theta")
    if config["hidden_size"] % config["num_attention_heads"] != 0:
        raise RuntimeError("hidden size is not divisible by query heads")
    if config["num_attention_heads"] % config["num_key_value_heads"] != 0:
        raise RuntimeError("query heads are not divisible by KV heads")
    if config["hidden_size"] // config["num_attention_heads"] != 64:
        raise RuntimeError("TinyLlama head_dim is no longer 64")


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


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    contiguous = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    contiguous.numpy().astype("<f4", copy=False).tofile(path)


def tensor_entry(name: str, tensor: torch.Tensor, file_name: str) -> dict:
    return {
        "name": name,
        "dtype": "f32",
        "shape": list(tensor.shape),
        "file": file_name,
    }


def nnis_config(source_config: dict) -> dict:
    return {
        "vocab_size": source_config["vocab_size"],
        "eos_token_id": source_config["eos_token_id"],
        "hidden_size": source_config["hidden_size"],
        "intermediate_size": source_config["intermediate_size"],
        "num_hidden_layers": source_config["num_hidden_layers"],
        "num_attention_heads": source_config["num_attention_heads"],
        "num_key_value_heads": source_config["num_key_value_heads"],
        "max_position_embeddings": source_config["max_position_embeddings"],
        "rms_norm_eps": source_config["rms_norm_eps"],
        "rope_theta": source_config["rope_theta"],
        "activation": "silu",
        "weight_dtype": "f32",
    }


def convert_weights(checkpoint: Path, output: Path, tokenizer_sha256: str) -> dict:
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    check_source_config(config)
    source = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    expected = expected_source_names(config["num_hidden_layers"])
    actual = set(source)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"unexpected Safetensors key set; missing={missing}, extra={extra}")
    bad_dtypes = sorted(
        f"{name}:{tensor.dtype}"
        for name, tensor in source.items()
        if tensor.dtype != torch.bfloat16
    )
    if bad_dtypes:
        raise RuntimeError(f"non-BF16 source tensors: {bad_dtypes}")

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

    store("token_embedding", "model.embed_tokens.weight")
    for layer in range(config["num_hidden_layers"]):
        src = f"model.layers.{layer}"
        dst = f"layers.{layer}"
        store(f"{dst}.input_norm", f"{src}.input_layernorm.weight")
        store(f"{dst}.q_proj", f"{src}.self_attn.q_proj.weight", transpose=True)
        store(f"{dst}.k_proj", f"{src}.self_attn.k_proj.weight", transpose=True)
        store(f"{dst}.v_proj", f"{src}.self_attn.v_proj.weight", transpose=True)
        store(f"{dst}.o_proj", f"{src}.self_attn.o_proj.weight", transpose=True)
        store(f"{dst}.post_attention_norm", f"{src}.post_attention_layernorm.weight")
        store(f"{dst}.gate_proj", f"{src}.mlp.gate_proj.weight", transpose=True)
        store(f"{dst}.up_proj", f"{src}.mlp.up_proj.weight", transpose=True)
        store(f"{dst}.down_proj", f"{src}.mlp.down_proj.weight", transpose=True)
    store("final_norm", "model.norm.weight")
    store("lm_head", "lm_head.weight", transpose=True)

    expected_tensor_count = 3 + config["num_hidden_layers"] * 9
    if len(manifest_tensors) != expected_tensor_count:
        raise RuntimeError(
            f"converted tensor count {len(manifest_tensors)} != expected {expected_tensor_count}"
        )

    converted_config = nnis_config(config)
    manifest = {
        "format": "nnis-model",
        "version": 1,
        "config": converted_config,
        "tensors": manifest_tensors,
    }
    (output / "model.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "source_model_sha256": MODEL_SHA256,
        "source_weight_dtype": "bfloat16",
        "execution_weight_dtype": "f32",
        "tokenizer_sha256": tokenizer_sha256,
        "tied_lm_head_materialized": False,
        "converter": Path(__file__).name,
        "transformers_version": transformers.__version__,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    del source
    gc.collect()
    return converted_config


def prompt_ids(tokenizer, family: str, target_tokens: int) -> list[int]:
    if target_tokens < 2:
        raise ValueError("target prompt length must be at least two tokens")
    seed_ids = tokenizer(PROMPT_FAMILIES[family], add_special_tokens=False)["input_ids"]
    if not seed_ids:
        raise RuntimeError(f"prompt family {family!r} tokenized to an empty sequence")
    bos = tokenizer.bos_token_id
    if bos is None:
        raise RuntimeError("pinned TinyLlama tokenizer has no BOS token")
    body_needed = target_tokens - 1
    repeated = (seed_ids * ((body_needed + len(seed_ids) - 1) // len(seed_ids)))[:body_needed]
    ids = [int(bos), *(int(token) for token in repeated)]
    if len(ids) != target_tokens:
        raise RuntimeError("failed to construct exact-length prompt")
    return ids


def greedy_tokens(model, input_ids: list[int], steps: int) -> list[int]:
    current = torch.tensor([input_ids], dtype=torch.long, device="cpu")
    generated: list[int] = []
    with torch.inference_mode():
        result = model(input_ids=current, use_cache=True)
        logits = result.logits[0, -1].float().cpu()
        past = result.past_key_values
        for step in range(steps):
            token = int(torch.argmax(logits).item())
            generated.append(token)
            if step + 1 == steps:
                break
            result = model(
                input_ids=torch.tensor([[token]], dtype=torch.long, device="cpu"),
                past_key_values=past,
                use_cache=True,
            )
            logits = result.logits[0, -1].float().cpu()
            past = result.past_key_values
    return generated


def build_reference_suite(
    checkpoint: Path,
    output: Path,
    converted_config: dict,
    tokenizer_sha256: str,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if tokenizer.bos_token_id != 1 or tokenizer.eos_token_id != 2:
        raise RuntimeError(
            f"unexpected tokenizer special ids: bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id}"
        )
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
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        raise RuntimeError("TinyLlama reference unexpectedly tied the LM head")

    cases: list[dict] = []
    for family in PROMPT_FAMILIES:
        for target in TARGET_PROMPT_TOKENS:
            ids = prompt_ids(tokenizer, family, target)
            max_decode = DEEP_DECODE_STEPS if target == 32 else STANDARD_DECODE_STEPS
            generated = greedy_tokens(model, ids, max_decode)
            cases.append(
                {
                    "name": f"{family}-p{target:04d}-d{STANDARD_DECODE_STEPS:03d}",
                    "family": family,
                    "target_prompt_tokens": target,
                    "decode_steps": STANDARD_DECODE_STEPS,
                    "input_ids": ids,
                    "greedy_ids": generated[:STANDARD_DECODE_STEPS],
                }
            )
            if target == 32:
                cases.append(
                    {
                        "name": f"{family}-p{target:04d}-d{DEEP_DECODE_STEPS:03d}",
                        "family": family,
                        "target_prompt_tokens": target,
                        "decode_steps": DEEP_DECODE_STEPS,
                        "input_ids": ids,
                        "greedy_ids": generated,
                    }
                )

    expected_case_count = len(PROMPT_FAMILIES) * (len(TARGET_PROMPT_TOKENS) + 1)
    if len(cases) != expected_case_count:
        raise RuntimeError(f"reference case count {len(cases)} != expected {expected_case_count}")

    suite = {
        "schema_version": 1,
        "kind": "nnis-trained-llama-reference-suite-v1",
        "source_repo": REPO_ID,
        "source_revision": REVISION,
        "source_model_sha256": MODEL_SHA256,
        "source_weight_dtype": "bfloat16",
        "execution_weight_dtype": "f32",
        "tokenizer_sha256": tokenizer_sha256,
        "transformers_version": transformers.__version__,
        "expected_config": converted_config,
        "case_policy": {
            "families": list(PROMPT_FAMILIES),
            "prompt_token_lengths": list(TARGET_PROMPT_TOKENS),
            "standard_decode_steps": STANDARD_DECODE_STEPS,
            "deep_decode_prompt_tokens": 32,
            "deep_decode_steps": DEEP_DECODE_STEPS,
            "oracle": "Transformers CPU F32 greedy generation from the exact pinned checkpoint widened to F32",
        },
        "cases": cases,
    }
    output.write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
    del model
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the pinned trained TinyLlama fixture and massive greedy oracle suite."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"trusted reference requires Transformers {TRANSFORMERS_VERSION}; got {transformers.__version__}"
        )

    checkpoint = download_checkpoint(args.cache_dir)
    source_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    check_source_config(source_config)
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer_file = args.output / "tokenizer.json"
    shutil.copy2(checkpoint / "tokenizer.json", tokenizer_file)
    tokenizer_digest = sha256(tokenizer_file)

    model_dir = args.output / "model"
    suite_file = args.output / "reference_suite.json"
    converted_config = convert_weights(checkpoint, model_dir, tokenizer_digest)
    gc.collect()
    build_reference_suite(checkpoint, suite_file, converted_config, tokenizer_digest)

    print(f"checkpoint={REPO_ID}@{REVISION}")
    print(f"model_sha256={MODEL_SHA256}")
    print(f"tokenizer_sha256={tokenizer_digest}")
    print(f"transformers={transformers.__version__}")
    print(f"model_dir={model_dir}")
    print(f"suite_file={suite_file}")
    print(f"cases={len(json.loads(suite_file.read_text(encoding='utf-8'))['cases'])}")


if __name__ == "__main__":
    main()
