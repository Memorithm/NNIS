# SOUP -> NNIS interoperability

This document defines the first explicit interoperability path between [SOUP](https://github.com/MakazhanAlpamys/Soup) and NNIS.

The boundary is artifact-based. NNIS does **not** depend on SOUP's Python runtime and does not absorb SOUP's training, data, or orchestration responsibilities. SOUP produces a local Hugging Face-style merged model; NNIS consumes the resulting `config.json`, Safetensors weights, and `tokenizer.json` through its existing strict local loader.

## Verified source boundary

The integration was designed against:

- SOUP `main` commit `1082db47f1297aed7e71e77410d90d61b4c153e3`.
- NNIS `main` commit `b9d2f1e74bb68dfa90ca499a24ce857d15e7fb02`.

At that SOUP revision, `soup merge` loads the base model at the requested `--dtype`, merges the LoRA adapter, calls `model.save_pretrained(...)`, and saves the tokenizer into the same output directory. The default `--dtype` is `float16`.

At that NNIS revision, the strict local Safetensors loader accepts F32 and BF16 source tensors, while the currently qualified direct decoder construction still requires an F32 logical base graph. Therefore the first executable SOUP -> NNIS path is intentionally restricted to a dense F32 merge.

## Produce a compatible SOUP artifact

From a SOUP environment:

```bash
soup merge \
  --adapter ./output \
  --output ./merged-nnis \
  --dtype float32
```

Do not use SOUP's default `float16` dtype for this direct path yet. Do not use the `4bit` or `4bit_forced` merged save formats for this path either. NNIS must fail closed rather than silently reinterpret an unqualified representation.

The resulting directory must satisfy NNIS's existing Hugging Face loader contract, including:

- `config.json` describing a decoder capability NNIS currently admits;
- `model.safetensors` or `model.safetensors.index.json` plus referenced shards;
- F32 source tensors matching the declared `torch_dtype`;
- `tokenizer.json` for the CLI path below.

The current loader remains deliberately narrow. A directory can still be rejected for unsupported architecture, bias tensors, RoPE semantics, EOS representation, tensor names/shapes, dtype, or other capability mismatches. SOUP provenance does not bypass those checks.

## Run the merged model with NNIS

```bash
cargo run --locked -p nnis-cli --bin nnis-hf -- \
  generate \
  --model ./merged-nnis \
  --prompt "Hello from SOUP to NNIS" \
  --max-new-tokens 16
```

`nnis-hf` defaults the tokenizer path to `MODEL_DIR/tokenizer.json`. Override it explicitly when needed:

```bash
cargo run --locked -p nnis-cli --bin nnis-hf -- \
  generate \
  --model ./merged-nnis \
  --tokenizer ./merged-nnis/tokenizer.json \
  --prompt "Hello"
```

No network access is performed by the NNIS loader.

## What this integration does not claim

This boundary does not establish:

- compatibility with every model SOUP can train;
- compatibility with SOUP's default FP16 merged output;
- compatibility with BNB 4-bit, GGUF, AWQ, GPTQ, ONNX, TensorRT, or other SOUP export targets;
- LoRA-adapter execution without first merging the adapter;
- numerical equivalence to Transformers merely because loading succeeds;
- serving-performance, memory, or quality equivalence to another runtime;
- a change to NNIS model-format v1 or NNIS's qualified runtime defaults.

Any future FP16 source admission, adapter-native execution, quantized representation, or broader architecture support requires its own explicit versioned semantics and qualification evidence.
