# Design note: allocation pooling

Status: deferred design. Pooling is the largest measured remaining native-cost
opportunity in NNIS, but a safe design depends on ownership machinery that does
not exist yet. This note records the motivation, the candidate mechanisms, and
the decision criteria so implementation can resume without re-measuring.

## Motivation (measured)

Clean Wave-7 breakdown at commit `6dd485f` (Thor, CC 11.0):

| Operation (4 MiB f32) | Median |
| --- | --- |
| `cuMemAlloc` | 0.069487 ms |
| `cuMemFree` | 0.042208 ms |

A softmax or attention stage that allocates per call therefore pays roughly
0.11 ms of pure allocator overhead before any kernel runs - comparable to an
entire fused row-softmax launch at 16 Mi elements (1.019 ms). The flat
softmax safe wrapper allocates two scalars plus a workspace per call today.

## Candidate mechanism: CUDA memory pools

CUDA 11.2+ provides stream-ordered allocation natively:

- `cuMemPoolCreate` with `CU_MEMPOOL_ALLOC_RELEASE_THRESHOLD`
- `cuMemAllocFromPoolAsync` / `cuMemFreeAsync`, both ordered on a stream
- freeing enqueues rather than performs the release; reuse happens without
  host synchronization once outstanding work completes

Driver 580 / CUDA 13.0 fully supports this. It is the NVIDIA-native mechanism
and matches NNIS's direction; no third-party allocator is considered.

## Why it was deferred

Safe Rust requires that returning from an operation never leaves DMA entitled
to access a borrow. Today's contract is enforced by synchronizing wrappers and
explicitly unsafe `_async` variants (see ARCHITECTURE.md). Stream-ordered
freeing weakens that story in one specific way:

- A pooled `DeviceBuffer` dropped by the host while work is still queued is
  only *logically* freed; CUDA reuses its memory only after the ordering work
  completes. Correctness then depends on every future writer being properly
  stream-ordered with respect to the original use.
- Cross-stream consumers break that guarantee silently unless event
  record/wait dependencies are recorded when buffers change streams.

NNIS's current types do not encode "in-flight" state, so a naive pooled
`new` would promise more than the ownership model proves.

## Proposed design sketch

1. `StreamOrderedAllocator` bound to exactly one `Stream`; allocations carry
   their allocator's stream in their context identity checks.
2. `alloc` maps to `cuMemAllocFromPoolAsync(stream)`; Drop maps to
   `cuMemFreeAsync(stream)`, keeping the buffer's borrows alive until the
   enqueue returns (the free itself is stream-ordered).
3. Any cross-stream handoff must route through existing `Event` record/wait;
   expose this as an explicit `share_with(other_stream)` that records the
   dependency, so the safety obligation is visible at the call site.
4. Keep plain `DeviceBuffer::new` unchanged; pooling is opt-in per pipeline,
   mirroring how `enqueue_*` opt out of synchronization.
5. Validate with the benchmark harness: per-call alloc+kernel+free versus
   pre-allocated buffers across 1 KiB..64 MiB, plus a cross-stream case.

## Decision criteria

Implement only if a clean benchmark shows a reproducible end-to-end win for a
real multi-stage pipeline (e.g. flat softmax with pooled scalars/workspace),
not just an isolated allocator microbenchmark. Record negative results in the
continuation log either way.

## Alternatives considered

- Host-side free-list cache keyed by size class: avoids new FFI but cannot
  reuse memory still referenced by queued work without the same ordering
  problem, adds fragmentation policy for no native benefit.
- Arena/bump allocation over one large reservation: viable for fixed-shape
  inference graphs but changes lifetime rules more radically than pools.
