# NNML2 F16 materialization memory evidence v1

This contract records exact NNIS-owned allocation lifetimes while a supported
F32 `ModelWeights` graph is materialized into the resident F16 reference graph.
It extends the steady-state ownership accounting added by PRs #119 and #120; it
is not a physical-VRAM or serving-memory metric.

## Contract

The public schema is `F16WeightMaterializationMemoryEvidenceV1` with:

```text
NNIS_F16_WEIGHT_MATERIALIZATION_MEMORY_EVIDENCE_VERSION = 1
```

For one successful materialization it records:

- the exact source `WeightAllocationSummaryV1` while the source graph is live;
- the exact final resident-F16 `WeightAllocationSummaryV1`;
- the F16 execution plan that determined physical projection layout;
- every successful F16 allocation and explicit temporary release in order;
- current live F16 bytes after every event;
- current live temporary-F16 bytes after every event;
- current scoped owned bytes after every event;
- peak live F16 bytes;
- peak live temporary-F16 bytes;
- peak scoped owned bytes;
- final scoped owned bytes at the end of weight materialization.

The event kinds are `allocate_resident`, `allocate_temporary`, and
`release_temporary`.

## Exactness boundary

The tracker is updated only after a `DeviceBuffer<u16>` allocation has succeeded.
Its byte value comes from that live buffer's `size_bytes()`. It does not infer an
allocation from model geometry, parameter count, a nominal bit width, or a
`cuMemGetInfo` delta.

The source graph's owned bytes are obtained from the existing
`WeightAllocationSummaryV1`. The source graph remains alive for the full
`F16ModelWeights::from_f32` call, so the scoped total for each event is:

```text
source ModelWeights owned bytes + currently live F16 materialization bytes
```

For the reference KN projection layout, the narrowed F16 projection allocation is
resident immediately. For transposed layouts, NNIS records the narrowed KN buffer
as temporary, then records the final NK allocation while both allocations are
live, synchronizes the transpose, records the temporary release, and explicitly
drops the KN buffer.

At successful completion the contract fails closed unless:

- no temporary F16 bytes remain live;
- tracked live F16 bytes exactly equal the resident F16 weight summary;
- the source summary still matches the source-byte value used to initialize the
  tracker;
- peak scoped bytes are at least the final scoped bytes.

## What the peak is not

`peak_scoped_owned_allocation_bytes` is not any of the following:

- physical GPU page residency;
- process-wide VRAM consumption;
- CUDA allocator/page-table overhead;
- CUDA context, module, JIT, or kernel storage;
- RoPE tables allocated after weight materialization;
- KV cache, session, or workspace storage;
- serialized checkpoint size;
- a compression ratio or effective-bits-per-parameter claim;
- a latency, throughput, or quality metric.

The schema currently represents successful materialization only. If construction
fails after one or more allocations, v1 does not emit a durable failed-attempt
trace and must not be used to claim a failure-path peak.

## Pinned SmolLM2 harness

The non-timing physical harness is:

```bash
cargo run --locked --release -p nnis-bench \
  --example smollm2_f16_materialization_memory -- \
  --model /tmp/smollm2-135m/model \
  --device 0
```

It requires the pinned widened-F32 fixture for
`HuggingFaceTB/SmolLM2-135M` revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, validates the known model geometry,
and selects `F16ReferenceExecutionPlan::smollm2_135m_thor_min_latency`. That plan
is fail-closed outside its qualified NVIDIA Thor domain.

The JSON binds the materialization evidence to `BenchmarkMetadata`, the exact
model identity, model config, and selected execution plan. It performs no
generation and no timing.

## Remaining NNML2 memory gap

This contract closes exact **owned allocation peak accounting for the weight
materialization scope** when a successful run is captured. It does not close the
separate physical-page-residency/process-wide VRAM evidence gap. Those metrics
require their own explicitly defined measurement semantics and must not be
substituted with this ownership evidence.
