# KVLab position-native backend v4

`nnis-kvlab-backend-v4` is the NNIS execution boundary for `kvlab.prospect-kv-backend-request/v4` / `...response/v4`.

The backend reads one canonical JSON request from stdin. It validates the exact model/runtime provenance supplied on the command line, recomputes the canonical position-trace SHA-256, validates a strictly increasing in-range `retained_positions` set, loads the requested local Safetensors model, and prefills the exact `model_input_token_ids` sequence. Duplicate vocabulary token ids are valid and remain distinct because KV row identity is the sequence position.

For a candidate request, NNIS calls `InferenceSession::compact_kv_cache_rows(retained_positions)`. The session logical/RoPE position must remain equal to the original input length. The first evaluation token is then decoded as a bridge under the selected history. Remaining evaluation tokens are teacher-forced; NNIS reports observed mean negative log-likelihood and top-1 token accuracy and returns a canonical response attesting the exact request SHA-256, mode, policy and retained positions.

The baseline retains every input position and does not compact the cache. A candidate equal to the full baseline is rejected.

This backend is an execution mechanism, not evidence that a particular campaign has run. It does not infer allocator release, HBM residency reduction, avoided memory traffic, latency or throughput improvements from row compaction. Those claims require separate measurements. Quality claims are limited to the explicit teacher-forced metrics returned for an actually executed request.