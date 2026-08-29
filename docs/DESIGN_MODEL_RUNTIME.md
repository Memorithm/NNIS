# NNIS model-runtime ownership and KV-cache design

Status: initial runtime foundation (owned async work + KV cache).

## Scope

This note records the ownership rules required before NNIS can safely compose
multiple CUDA operations into decoder-only transformer pipelines without a host
synchronization after every high-level primitive.

It does **not** define a model-file compatibility promise and it does not change
the existing numeric policy. NNIS currently supports native f32 paths and
packed bf16 storage with f32 compute where the bf16 kernel families implement
that policy.

## Existing constraint

NNIS kernel families deliberately expose two levels today:

- safe high-level methods that synchronize before returning; and
- `unsafe enqueue_*` methods that submit work without synchronization and put
  the completion-time lifetime obligation on the caller.

That split is sound for isolated operations, but a model runtime needs to keep
weights, activations, temporaries, modules, streams, and caches alive across a
long chain of asynchronous launches.

## Owned asynchronous work

`nnis_rt::PendingGpuWork<R>` is the low-level ownership guard for that chain.
It records an event at a stream tail and owns an arbitrary resource graph `R`
until that event completes.

Rules:

1. Creating a guard from already-enqueued work is `unsafe`, because Rust cannot
   inspect CUDA's queue and prove that `R` contains every referenced resource.
2. Safe higher layers may encapsulate that unsafe boundary only when they build
   the operation and the complete ownership graph together.
3. `wait()` releases ownership only after event completion.
4. Dropping unfinished work waits for completion. If CUDA cannot establish
   completion, NNIS leaks the resource graph instead of risking device
   use-after-free.
5. The guard retains a clone of the stream so the stream object also outlives
   the recorded work.

This is intentionally an ownership mechanism, not a scheduler. Cross-stream
ordering continues to use NNIS events and the existing stream/pool rules.

## KV-cache layout

`nnis_rt::KvCache<T>` uses one fixed K allocation and one fixed V allocation:

```text
[layer][head][capacity][head_dim]
```

Each `(layer, head)` therefore owns a stable contiguous capacity region. The
cache tracks a logical valid length independently for every layer.

Properties:

- device resident for its full lifetime;
- fixed explicit capacity;
- append copies only newly produced K/V rows;
- no whole-cache copy on token growth;
- no host round trip on append;
- explicit overflow errors before submission;
- reset changes logical lengths and reuses allocated storage;
- source and destination ownership is retained through an event-backed
  `KvAppend` completion handle;
- all transfer offsets/ranges are validated before the first CUDA copy is
  submitted, so an ordinary validation error cannot release resources after a
  partial batch has already reached the GPU.

The append input is packed as `[heads][tokens][head_dim]`. One D2D copy per head
places that packed suffix into the head's capacity-strided cache region. This
is preferable to the current checked scatter wrapper for the cache hot path:
scatter's safe API intentionally validates positions on the host, while decode
positions are already determined by the cache's validated logical length.

## Stream ownership

A cache is bound to exactly one stream. Appends are ordered on that stream and
logical length is advanced after a completion event has successfully been
recorded for the submitted append.

This avoids a hidden cross-stream race between host-side position accounting
and device-side writes. A future multi-stream model scheduler must introduce an
explicit event handoff before allowing cache consumers on another stream.

## Decoder integration constraint

The current attention APIs validate exact packed buffer lengths such as
`[heads][kv_rows][head_dim]`. A capacity-strided cache is intentionally larger
than its active prefix, so the decoder-runtime work must not fake a shorter
`DeviceBuffer` or copy the whole cache merely to satisfy those APIs.

The next runtime layer should add a validated device-buffer view / pointer-range
abstraction (or equivalent offset-aware kernel entry points) and teach the
attention/GEMM composition to consume active cache ranges directly. That change
must preserve the existing safe/unsafe lifetime split and must not expose raw
suballocation pointers as a safe lifetime-free API.

## Validation gates

Before this foundation is merged, run on an NNIS CUDA host:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets
```

No performance claim follows from this design. Performance work starts only
after a real decoder path is validated against a trusted reference model.
