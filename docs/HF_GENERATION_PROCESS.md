# Local Hugging Face generation process contract

NNIS exposes an artifact-oriented process boundary for orchestrators that need to preserve the result of local Hugging Face/Safetensors generation as a machine-readable immutable artifact.

The process surface is:

```text
tools/nnis_hub_hf_generate.py
```

It delegates model loading, CUDA execution, tokenization, greedy sampling and decoded output to the native `nnis-hf generate` binary. The Python process does not implement decoder, tensor, sampling or CUDA semantics.

## Contract

- contract: `nnis.hf-generation@1.0.0`
- media type: `application/vnd.nnis.hf-generation.v1+json`
- schema version: `1`
- execution scope: `local_hf_f32_greedy_generation`

Example invocation:

```bash
python3 tools/nnis_hub_hf_generate.py \
  --model ./merged-nnis \
  --prompt "Hello from SOUP to NNIS" \
  --device 0 \
  --max-new-tokens 16 \
  --result ./artifacts/nnis-hf-generation.json
```

`--tokenizer` is optional and defaults to `MODEL/tokenizer.json`, matching the native CLI boundary. `--nnis-hf-bin` can select the installed native binary; its default is `nnis-hf` from `PATH`.

The process invokes the native command without a shell:

```text
nnis-hf generate
  --model MODEL
  --tokenizer TOKENIZER
  --prompt PROMPT
  --device DEVICE
  --max-new-tokens N
```

No network access is performed by the NNIS local-HF loader.

## Result semantics

On successful native generation the process writes one new JSON result and prints the same JSON summary to stdout. The result records:

- the requested model directory and resolved tokenizer path;
- prompt, CUDA device ordinal and maximum generated-token count;
- exact native stdout and stderr as UTF-8 strings;
- byte lengths and SHA-256 digests of both captured streams;
- the fixed greedy sampling identity;
- explicit negative claim flags for promotion, serving performance, numerical equivalence and general model-family support.

The process does not strip the newline printed by the native CLI or reconstruct generated text from logs. `output.stdout_utf8` is the captured native stdout exactly as decoded from UTF-8 bytes, and `output.stdout_sha256` binds the original bytes.

If native generation exits unsuccessfully, produces an oversized capture, cannot be executed, or emits invalid UTF-8 on a successful run, the process fails and does not publish the result artifact.

The result path must not already exist. This prevents a failed or retried process from silently replacing an earlier immutable result.

## Process bounds

Version 1 deliberately adds bounded orchestration limits without changing native generation semantics:

- prompt: 1 to 1 MiB of UTF-8;
- `max-new-tokens`: 1 to 65,536;
- stdout capture: at most 16 MiB;
- stderr capture: at most 16 MiB;
- CUDA device ordinal: non-negative.

The native NNIS model/session checks remain authoritative for model capacity, CUDA availability and all decoder execution failures.

## Relationship to SOUP and Hub

For the currently admitted SOUP path, the model input should be the dense F32 local-HF artifact produced by the explicit SOUP merge boundary and accepted by `nnis.hf-preflight@1`.

An orchestrator may therefore compose:

```text
SOUP training
  -> SOUP dense F32 merge
  -> NNIS HF preflight
  -> NNIS HF generation process
```

The generation process itself does not consume or reinterpret a Hub preflight artifact. Preflight admission remains a separate NNIS-owned contract and orchestration provenance should preserve both results independently.

## Non-claims

Successful execution of this process does not establish:

- numerical equivalence to Transformers or another runtime;
- model quality;
- serving latency, throughput or memory performance;
- promotion eligibility;
- compatibility with every model family;
- compatibility with FP16, BNB 4-bit or adapter-native SOUP artifacts;
- Jetson/ARM64 qualification merely because a CUDA execution succeeded;
- HML1 resource-aware placement, HML2 distributed orchestration or HML4 qualification-campaign maturity.

Physical qualification, performance comparison and promotion remain governed by the NNIS sovereignty roadmap and exact-head evidence requirements.
