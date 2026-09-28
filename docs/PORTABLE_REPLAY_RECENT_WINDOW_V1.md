# Portable bounded recent-window replay reference v1 (DSV41-1)

Status: software reference contract, CPU only. Roadmap item
`NNIS_DSV41_1_bounded_recent_window_replay_reference_path` of
`deepseek_v41_kv_runtime_program_2026_09_24` (off-main sovereignty roadmap).
Builds on the DSV41-0 identity contract (`nnis_core::replay_state`, PR #226/#227).

## Contract

### `nnis-core` (backend-neutral)

- `ReplaySourceIdentityV1::logical_items()` returns the exact declared item
  count, or `PositionOverflow` for a full `0..=u64::MAX` range.
- `ReplayWindowRequestV1::recent(source, window_items)` resolves a
  caller-supplied recent-window length into the inclusive range
  `end - (window_items - 1) ..= end`, where `end` is the source's last logical
  position.
  - `window_items == 0` fails with `ZeroWindowItems`.
  - `window_items` larger than the declared source fails with
    `RecentWindowExceedsSource { available, requested }`.
  - The window is never clamped and NNIS never chooses its length; the length
    is policy owned by the model/domain supplier.
  - Resolution is overflow-free for any source, including a full `u64` range.

### `nnis-cpu` (dense reference oracle)

- `CpuReplaySourceV1::new(identity, row_width, rows)` binds an immutable dense
  row-major F32 payload to an exact source identity. It requires a non-zero row
  width, exactly `logical_items * row_width` values, finite values only, and a
  source range representable in host `usize` indexing.
- `replay_window(&request)` validates the request against the provider's
  current identity through `ReplayStateProviderV1` (any provider, source,
  generation, representation, epoch or range drift fails closed with
  `SourceIdentityMismatch`) and returns a bit-exact copy of the requested rows.
- `replay_recent_window(window_items)` resolves the recent window against the
  current identity and returns the resolved request together with the rows.

## Evidence

Host-only unit tests in `crates/nnis-core/src/replay_state.rs` and
`crates/nnis-cpu/src/replay.rs` cover trailing-window offsets, full/single
windows, zero/oversized windows, full-`u64` ranges, bit preservation
(including signed zero and subnormals), interior windows, stale-generation
requests, and malformed payloads. No GPU, model, or physical run was performed.

## Claim boundary

- Dense reference only: no reconstruction, compression, FP4/INT4, paging,
  eviction, offload or cross-layer reuse.
- No WGPU or CUDA execution; a later WGPU replay path must match this oracle.
- Copying a logical window is not physical memory release and establishes no
  memory, latency, throughput or model-quality result.
- No equivalence claim with DeepSeek-V4.1 or any external bounded-replay
  implementation.
