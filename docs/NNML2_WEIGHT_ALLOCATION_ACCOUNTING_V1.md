# NNML2 weight allocation accounting v1

This document defines the exact NNIS-owned accounting surface for model weight
allocations. It is evidence infrastructure for NNML2 and the ElasticBitAllocation
Stage-B program. It is not a compression, latency, quality, or physical-VRAM-
residency claim.

## Contract

`WeightAllocationSummaryV1` reports live `DeviceBuffer` allocations owned by one
model weight graph. Schema v1 is shared by the normal `ModelWeights` graph and
the explicit resident-F16 reference graph.

The contract version is:

```text
NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION = 1
```

For each unique live device allocation it records:

- a deterministic summary-local `allocation_index`;
- numeric storage dtype (`f32`, `bf16`, or `f16`);
- element count;
- exact allocation bytes;
- every logical model-weight name that references that same allocation.

The aggregate records:

- logical tensor references;
- logical element references;
- unique device allocations;
- unique device elements;
- exact owned device-allocation bytes;
- the complete unique-allocation segment list.

Both model graphs lower to the same crate-internal summarizer. The F16 path does
not maintain a second accounting algorithm.

## Why the byte count is exact

`nnis-rt::DeviceBuffer<T>` owns exactly `len * size_of::<T>()` bytes and passes
that checked byte count directly to `cuMemAlloc`. Therefore the sum in
`owned_device_allocation_bytes` is exact for the allocation scope defined here.

Aliasing is handled by the live CUDA allocation identity while the summary is
constructed. The raw device address is never serialized. If several logical
weight names reference the same live allocation, the allocation bytes are
counted once and all aliases remain attached to the same summary segment. An
alias that disagrees on dtype, element count, or byte count fails closed.

The F16 reference path enumerates the actual resident `DeviceBuffer<u16>` graph:
`token_embedding`, all nine named weight buffers for each decoder layer,
`final_norm`, and `lm_head`. Its total is therefore derived from live allocations,
not from model geometry or an assumed bytes-per-parameter ratio.

## What this does not mean

`owned_device_allocation_bytes` must not be renamed or interpreted as any of the
following:

- physical GPU page residency;
- CUDA allocator metadata or page-table cost;
- process-wide VRAM consumption;
- CUDA module/JIT/kernel storage;
- RoPE tables;
- session workspaces;
- KV-cache storage;
- temporary F32 -> F16/BF16 conversion memory;
- peak materialization memory;
- serialized checkpoint size.

`cuMemGetInfo` snapshots may remain useful telemetry, but their deltas are not a
replacement for this ownership accounting and are not exact allocation
provenance.

## Pinned SmolLM2 normal-graph accounting harness

The non-timing normal-graph harness is:

```bash
cargo run --locked --release -p nnis-bench \
  --example smollm2_weight_allocation_accounting -- \
  --model /tmp/smollm2-135m/model \
  --device 0
```

It refuses a model provenance that does not match the pinned
`HuggingFaceTB/SmolLM2-135M` checkpoint already used by NNIS qualification:

- revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`;
- `model.safetensors` SHA-256:
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`.

The JSON output binds the allocation summary to the model config and the normal
NNIS `BenchmarkMetadata` environment fingerprint. It intentionally performs no
latency measurement and no generation.

## Pinned SmolLM2 resident-F16 accounting harness

The corresponding steady-state F16 harness is:

```bash
cargo run --locked --release -p nnis-bench \
  --example smollm2_f16_weight_allocation_accounting -- \
  --model /tmp/smollm2-135m/model \
  --device 0
```

It validates the same pinned provenance and SmolLM2 geometry, constructs the
explicit `F16ReferencePlan::edge_llm_v0_10_0_alignment()` model, then reports the
live resident-F16 weight allocations after construction. It performs no token
generation and no timing.

The source model-format-v1 graph is F32 during construction. The report names
that source-graph execution dtype separately from the steady-state resident F16
weight dtype so the conversion boundary cannot be mistaken for an in-place
representation change.

## F16 construction-lifetime boundary

The steady-state F16 summary does **not** report peak conversion allocation.
During `F16ModelWeights::from_f32` the full source F32 graph remains live while
resident F16 buffers are accumulated. For transposed projection plans, a
narrowed F16 KN buffer can additionally coexist temporarily with its final F16 NK
buffer before the temporary is dropped.

Those lifetimes make a peak materially different from the final steady-state
weight total. A future peak contract must instrument the actual live allocation
set at construction transitions. It must not infer peak bytes by adding nominal
model sizes or by using a `cuMemGetInfo` delta.

## Current boundary

Schema v1 now closes exact owned-allocation accounting for:

1. the normal `ModelWeights` graph, including its supported F32/BF16 storage;
2. the explicit steady-state F16 reference weight graph.

Still open:

- physical GPU page residency;
- CUDA allocator/page-table overhead;
- process-wide VRAM attribution;
- exact F16 conversion/materialization peak;
- exact execution-state accounting beyond the weight graphs.

No low-bit, sparse, low-rank, codebook, residual, or elastic allocator capability
is authorized by this contract.
