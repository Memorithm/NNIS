# NNML2 INT4 reference weight storage v1

Status: reference representation and exact owned-allocation accounting; **not an executable INT4 model runtime**.

This contract is the first NNIS-owned full-model fixed-4-bit representation step required by the frozen ElasticBitAllocation Stage-B protocol. It creates a real packed representation from the exact F32 execution-weight graph while preserving a strict boundary between representation evidence and runtime promotion.

## Contract

Version: `NNIS_INT4_REFERENCE_STORAGE_VERSION = 1`.

For each **unique live F32 source allocation** in the model weight graph:

1. copy the source values for reference quantization;
2. reject any non-finite value;
3. compute one symmetric F32 scale:
   `scale = max(abs(weight)) / 7` (or `1.0` for an all-zero allocation);
4. quantize deterministically with round-to-nearest and clamp to signed codes `[-7, 7]`;
5. reserve the signed nibble `-8` and never emit it;
6. pack two four-bit two's-complement values per byte, low nibble first;
7. allocate the packed payload and one F32 scale as real CUDA `DeviceBuffer` objects.

Source aliases are deduplicated by the live source allocation identity. Their logical names remain in evidence, but the physical INT4 payload and scale are counted once.

The scale scope is deliberately **one scale per unique source allocation**. This is a simple fixed baseline, not a claim that this quantizer is optimal.

## Canonical serialized accounting

Each unique allocation has one deterministic serialized record:

- 4 bytes: ASCII magic `NI41`;
- 8 bytes: little-endian logical element count;
- 4 bytes: little-endian IEEE-754 F32 scale bits;
- N bytes: packed INT4 payload.

The 16-byte header includes the scale. Tensor graph identity, ordering, logical shape and checkpoint identity remain owned by the exact NNIS model contract; they are not silently duplicated into an unversioned side format.

Reported serialized bits/value include the header and packed payload. Reported resident bits/value use the **actual requested bytes of the live CUDA payload and scale buffers**. Neither metric is inferred from nominal “4-bit” labeling.

## Reconstruction evidence

For every unique allocation NNIS records:

- maximum absolute reconstruction error;
- mean squared reconstruction error.

The model summary aggregates maximum absolute error and element-weighted MSE over unique logical values.

These are weight-reconstruction metrics only. They are **not** next-token NLL, perplexity, generation parity, model quality, or an acceptance threshold.

## SmolLM2 evidence harness

The operator harness is:

`cargo run -p nnis-bench --example smollm2_int4_reference_storage -- --model DIR --device 0`

It requires the existing pinned SmolLM2-135M NNIS model fixture:

- source repository: `HuggingFaceTB/SmolLM2-135M`;
- revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- source model SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`;
- source dtype: BF16;
- NNIS execution materialization used as the quantizer source: F32.

The harness emits both the established `WeightAllocationSummaryV1` of the source F32 graph and the new INT4 storage summary. The INT4 source byte count must reconcile exactly with the F32 owned-allocation summary.

## Deliberate non-claims

This v1 contract does **not** establish:

- an INT4 projection, embedding, MLP, attention or LM-head kernel;
- model execution from packed INT4 weights;
- next-token NLL, perplexity or generation parity;
- conversion/materialization peak memory;
- physical GPU page residency;
- process-wide VRAM attribution;
- latency, throughput or memory-bandwidth improvement;
- ElasticBitAllocation allocator/search authorization;
- final-test access.

The packed device buffers remain private to the storage object. A later isolated-projection slice may consume them only through an explicit versioned plan without exposing raw buffers. Full-model runtime semantics, comparison against the frozen Stage-B development split, and end-to-end execution remain separate qualification gates before this representation can satisfy the Stage-B fixed-4-bit baseline gate.

## Next gate

The next engineering step is an explicit INT4 execution plan that consumes this versioned representation without silently materializing a second full dense weight graph. It must retain exact resident accounting and report any transient dequantization workspace separately.

Only after real model execution is qualified may ElasticXxx Stage B collect NLL, perplexity and the frozen request-latency protocol for this candidate.


## Successor note

The isolated projection contract in `NNML2_INT4_PROJECTION_V1.md` consumes these private buffers through a shape/name-bound plan. It does not change this storage schema, does not set `execution_qualified` to true, and does not establish a full-model INT4 runtime.
