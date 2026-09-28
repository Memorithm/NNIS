# NNML2 KV logical cache telemetry surface

Status: **completed** (merged as PR #154 on main `c58ec1c`).

## Goal

Complete the PR #133 / #138 logical KV telemetry library surface as a
**versioned** public contract: schema version constant, `schema_version` field,
facade re-export of the version constant, and fail-closed documentation. No
physical Thor execution.

## Scope

- Add `NNIS_KV_CACHE_TELEMETRY_VERSION` and `schema_version` on `KvCacheTelemetry`.
- Re-export the version constant from `nnis-rt` and the `nnis` facade
  (`runtime` module + crate root) without duplicate `E0252` exports.
- Publish `docs/NNML2_KV_LOGICAL_CACHE_TELEMETRY_V1.md` and a short
  `docs/memory/` note.
- Mark `NNML1_CLI_SAMPLED_STREAMING.md` completed (PR #137) if still active.
- Do **not** re-land `InferenceSession::kv_cache_telemetry` (already on main
  from #138).

## Required merge gates

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets --locked`
- strict Clippy under the repository workflow
- workspace tests (including host-only telemetry version/invariant tests)
- Rust 1.77 MSRV gate

## Claim boundary

This slice does not claim physical residency, payload visibility, eviction
policy, process-wide VRAM attribution, serving performance, or Thor parity.
