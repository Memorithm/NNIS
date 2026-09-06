# NNML2 weight allocation accounting v1

This document defines the first exact NNIS-owned accounting surface for model
weight allocations. It is evidence infrastructure for NNML2 and the
ElasticBitAllocation Stage-B program. It is not a compression, latency, quality,
or physical-VRAM-residency claim.

## Contract

`WeightAllocationSummaryV1` reports the live `DeviceBuffer` allocations owned by
one `ModelWeights` graph.

The contract version is:

```text
NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION = 1
```

For each unique live device allocation it records:

- a deterministic summary-local `allocation_index`;
- numeric storage dtype (`f32`, `bf16`, with `f16` reserved for the separate F16
  reference-runtime follow-up);
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

## Why the byte count is exact

`nnis-rt::DeviceBuffer<T>` owns exactly `len * size_of::<T>()` bytes and passes
that checked byte count directly to `cuMemAlloc`. Therefore the sum in
`owned_device_allocation_bytes` is exact for the allocation scope defined here.

Aliasing is handled by the live CUDA allocation identity while the summary is
constructed. The raw device address is never serialized. If several logical
weight names reference the same live allocation, the allocation bytes are
counted once and all aliases remain attached to the same summary segment. An
alias that disagrees on dtype, element count, or byte count fails closed.

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

## Pinned SmolLM2 accounting harness

The non-timing harness is:

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

## Current boundary

Schema v1 closes only the exact owned-allocation accounting gap for the normal
`ModelWeights` graph, including the already-supported BF16 LM-head
representation.

The explicit F16 reference runtime owns a separate private `F16ModelWeights`
graph. Its exact steady-state allocation accounting and its conversion peak must
be added in its own follow-up and physically qualified on the pinned SmolLM2
path before using F16 evidence in the ElasticBitAllocation Stage-B baseline.

No low-bit, sparse, low-rank, codebook, residual, or elastic allocator capability
is authorized by this contract.
