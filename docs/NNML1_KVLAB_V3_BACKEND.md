# NNML1 KVLab v3 real-model backend

`nnis-kvlab-backend` is the concrete NNIS execution endpoint for KVLab request schema `kvlab.prospect-kv-backend-request/v3`.

It is intentionally a one-request/one-process correctness path. It loads a declared local Safetensors model, executes the exact request and writes one response to stdout. It does not download a model or tokenizer.

## Invocation

```text
nnis-kvlab-backend \
  --model /path/to/local/model \
  --model-id MODEL_ID \
  --model-revision MODEL_REVISION \
  --tokenizer-revision TOKENIZER_REVISION \
  --runtime-revision NNIS_GIT_REVISION \
  [--runtime-backend nnis-kvlab-v3] \
  [--device 0]
```

The model/runtime provenance carried by the request must equal the command-line declaration exactly. The request repository revision remains the producer/campaign provenance and is not rewritten by NNIS.

## Logical identities versus model tokens

KVLab v3 carries two aligned histories:

- `logical_input_token_ids`: unique identities used only to name KV rows and replay a selection;
- `model_input_token_ids`: actual model vocabulary ids, which may repeat.

NNIS maps each retained logical identity to its position in the logical history. Candidate requests pass those positions to `InferenceSession::compact_kv_cache_rows`. The compaction moves already-RoPE-encoded K/V vectors and therefore does not renumber their original positional phase. `InferenceSession.position` remains equal to the full prefill length.

Baseline requests retain the complete history and do not invoke candidate compaction.

## Teacher-forced evaluation

The backend requires at least two `evaluation_token_ids`.

The first is a **bridge token**. It is decoded after candidate compaction but is not scored. This is necessary because the logits returned by the initial prefill were produced before the selection was applied. Decoding the common bridge causes subsequent logits to be computed against the selected cache.

Every remaining evaluation token is then teacher-forced identically for baseline and candidate. The backend reports:

- `mean_nll` in natural-log units per scored token, lower is better;
- `token_accuracy`, exact top-1 teacher-forced accuracy, higher is better.

The opaque output artefact records the applied selection, physical row indexes, logical position after selection, bridge token and each scored target/top-1/NLL tuple. KVLab, not NNIS, computes the evidence envelope's artefact SHA-256 from the decoded response bytes.

## Validation

Before CUDA execution the backend rejects:

- unsupported schema/mode;
- model/runtime provenance drift;
- malformed Git/trace hashes;
- duplicate logical identities;
- retained identities that are not an ordered subset of the logical history;
- baseline/candidate policy inconsistencies;
- position misalignment between logical identities and actual model tokens;
- fewer than two evaluation tokens;
- an embedded trace whose canonical SHA-256 differs from `trace_sha256`.

After model load it also rejects vocabulary-out-of-range token ids and logical-position requests exceeding model capacity.

## Scientific boundary

This backend makes real model quality observations possible; its existence is not itself a representative model result. No fixture/unit test is promoted as observed model evidence.

KV compaction reduces the active rows scanned by cached attention but does not shrink NNIS's fixed cache allocation capacity. Logical bytes retained/evicted do not establish HBM release, DRAM traffic reduction, latency reduction, throughput gain, TTFT/TPOT improvement or energy reduction. Those require separate physical measurements and an appropriate benchmark protocol.
