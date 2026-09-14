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

If native generation exits unsuccessfully, produces an oversized capture, times out, cannot be executed, or emits invalid UTF-8 on a successful run, the process fails and does not publish the result artifact.

The result path must not already exist. This prevents a failed or retried process from silently replacing an earlier immutable result. JSON is written and synced to a private temporary file in the destination directory, then published by an atomic, no-replace hard link. A write, sync or link failure cannot expose a partial result at the destination. A concurrent publisher's file or symlink is neither overwritten nor removed. Filesystems without hard-link support fail closed; there is no non-atomic fallback. Temporary-file cleanup is best effort and cannot erase a competing destination. This requires trusted parent directories and does not promise directory-entry durability after power loss or hostile-filesystem isolation.

## Process bounds

Version 1 deliberately adds bounded orchestration limits without changing native generation semantics:

- prompt: 1 to 1 MiB of UTF-8, without NUL; operating-system argument-size limits may reject a smaller prompt;
- `max-new-tokens`: integer 1 to 65,536 (booleans are not integers in this contract);
- stdout capture: at most 16 MiB;
- stderr capture: at most 16 MiB;
- CUDA device ordinal: integer 0 to 2,147,483,647;
- native execution deadline: 3,600 seconds by default, configurable with positive finite `--timeout-seconds`.

The two POSIX pipes are drained concurrently and incrementally. Their limits are checked before appending bytes to each retained capture, not after `subprocess.run` has buffered all output. At most one overflow-detection byte is read beyond a stream's budget. The retained captures, UTF-8 decoding and JSON serialization still consume host memory; this is not a total-process RSS limit. JSON escaping can enlarge a report beyond its raw stream lengths. A consumer such as Hub may impose a separate serialized-report size limit.

The deadline covers pipe draining and child completion after process creation. It also handles a child that closes both streams without exiting. On overflow, timeout or read failure, the direct native child is killed and reaped and the pipes are closed. The child does not create a separate process group, preserving Hub's outer run-group cancellation. This is not standalone descendant supervision, network sandboxing, or a hard real-time guarantee for process creation/OS cleanup. Non-POSIX capture is explicitly rejected before spawning a child.

For an explicit shorter process budget:

```bash
python3 tools/nnis_hub_hf_generate.py \
  --model ./merged-nnis --prompt "Hello" \
  --timeout-seconds 120 \
  --result ./artifacts/generation-120s.json
```

The timeout is wrapper policy, not a decoder parameter or measured performance claim. Existing successful JSON fields, native argv, model admission, precision and sampling semantics are unchanged. The native NNIS model/session checks remain authoritative for model capacity, CUDA availability and all decoder execution failures.

CPU-only regression checks:

```bash
python3 tools/nnis_hub_hf_generate.py --self-test
python3 -m unittest tools/test_nnis_hub_hf_generate.py -v
```

These checks use synthetic child processes, not real models or CUDA evidence.

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
