# NNML1 sampled session batch

`SampledSessionBatch` is NNIS's first explicit multi-session correctness surface. It is intentionally narrower than production batching.

## Ownership contract

`Model::new_sampled_session_batch(n)` creates `n` independent `InferenceSession` values. Each session owns its own CUDA stream, KV cache, decode workspace, decoder position, and sampled-generation invocation. A zero-sized batch is rejected.

`SampledBatchRequest` carries an owned prompt plus its own `GenerationConfig` and `SamplingConfig`. Sampling RNG state is not shared across batch items.

## Execution contract

`SampledSessionBatch::generate_sampled` requires exactly one request per session. A request-count mismatch fails before any session is executed.

After shape validation, requests execute in deterministic batch-index order through the existing `InferenceSession::generate_sampled` implementation. Each item returns its own result. An item failure does not suppress later independent sessions and the API does not claim all-or-nothing GPU transaction semantics.

`positions()` exposes per-session decoder positions in stable batch order for continuation and tests.

## What this is not

The current implementation does not fuse prompts or decoder work into batched CUDA kernels. It does not overlap streams, dynamically combine requests, schedule continuous batches, provide backpressure, or claim a throughput/latency benefit. Those are separate NNML2/NNML4 gates and require physical evidence.

This design first freezes ownership and failure semantics so later concurrent execution can change scheduling without redefining request identity or cross-contaminating KV/RNG state.

## Qualification boundary

CUDA-optional tests verify:

- two independent sessions with the same seed reproduce the same frozen sampled sequence;
- session positions advance independently;
- request-count mismatch fails before mutation;
- an invalid item remains an item-local error while a later valid session still executes.

No multi-request performance claim follows from these tests.
