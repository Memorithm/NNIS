# KVLab position-native backend v4

`nnis-kvlab-backend-v4` is the NNIS execution boundary for `kvlab.prospect-kv-backend-request/v4` / `...response/v4`.

The backend reads one canonical JSON request from stdin. It validates the exact model/runtime provenance supplied on the command line, recomputes the canonical position-trace SHA-256, validates a strictly increasing in-range `retained_positions` set, loads the requested local Safetensors model, and prefills the exact `model_input_token_ids` sequence. Duplicate vocabulary token ids are valid and remain distinct because KV row identity is the sequence position.

## Exact checkpoint admission

KVLab v4 real-model evidence is fail-closed to NNIS checkpoints that already have an `ExactDecoderCheckpointSpec`. The `--model-id` and `--model-revision` pair must identify one of those frozen specs. Before creating a CUDA device/context, the backend streams the local `model.safetensors` file through SHA-256 and requires an exact match with the spec's `source_model_sha256`. After loading, the parsed `ModelConfig` must also pass the same spec's exact geometry/capability validation before `Model::new` is called.

The currently admitted checkpoints are the frozen NNIS specs for `HuggingFaceTB/SmolLM2-135M` and `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, at the exact source revisions and model-file digests recorded by `ExactDecoderCheckpointSpec`. A caller-supplied provenance label is therefore not sufficient to substitute different local weights.

This attestation covers the model checkpoint and decoder configuration only. The v4 backend consumes already-tokenized integer ids and does not load a tokenizer, so tokenizer identity remains campaign provenance owned by KVLab/reference fixtures rather than a local tokenizer-artifact validation performed by this binary.

For a candidate request, NNIS calls `InferenceSession::compact_kv_cache_rows(retained_positions)`. The session logical/RoPE position must remain equal to the original input length. The first evaluation token is then decoded as a bridge under the selected history. Remaining evaluation tokens are teacher-forced; NNIS reports observed mean negative log-likelihood and top-1 token accuracy and returns a canonical response attesting the exact request SHA-256, mode, policy and retained positions.

The baseline retains every input position and does not compact the cache. A candidate equal to the full baseline is rejected.

This backend is an execution mechanism, not evidence that a particular campaign has run. It does not infer allocator release, HBM residency reduction, avoided memory traffic, latency or throughput improvements from row compaction. Those claims require separate measurements. Quality claims are limited to the explicit teacher-forced metrics returned for an actually executed request.