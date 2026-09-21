# NNML2 NVML process-memory CLI surface

Status: **active** (selected next non-physical software slice after PR #154
KvCacheTelemetry schema v1 and PR #153 INT4 projection).

## Goal

Expose the existing `NvmlProcessMemorySnapshotV1` / `current_process_gpu_memory`
library contract through a fail-closed, read-only `nnis` CLI subcommand for
software observability. No facade churn (already re-exported from #138). No
physical Thor execution and no fake campaign results.

## Scope

- Add `nnis nvml-process-memory [--device N] [--json]` to `crates/nnis-cli`.
- Select CUDA device ordinal (default 0), retain a primary context so the PID is
  visible to NVML, call `current_process_gpu_memory`, print human text or
  versioned JSON with `schema_version`.
- Map library/NVML/CUDA errors to non-zero exit without panicking.
- CPU-only parse/help/formatter tests (no GPU required).
- Document CLI usage + claim boundary under `docs/memory/nvml-process-memory-v1.md`
  (and a short pointer in the lifecycle doc). Brief README CLI note.
- Skip redundant `nnis` facade re-exports.

## Required merge gates

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets --locked`
- strict Clippy under the repository workflow
- workspace tests (including host-only CLI parse/format tests)
- Rust 1.77 MSRV gate

## Claim boundary

Software observability CLI only. Does not claim physical residency, weight-only
attribution, allocator/page-table overhead, serving performance, or Thor parity.
