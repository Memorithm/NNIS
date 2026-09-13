# NNML2 KV logical cache telemetry surface

Status: **active** (selected next non-physical software slice after PR #137).

## Goal

Complete the PR #133 logical KV telemetry library surface as a versioned public
contract: schema version, facade/model discoverability, session observer, and
fail-closed documentation. No physical Thor execution.

## Scope

- Add `NNIS_KV_CACHE_TELEMETRY_VERSION` and `schema_version` on `KvCacheTelemetry`.
- Re-export cache/telemetry types from the `nnis` facade `runtime` module (and
  crate root) plus `nnis_model` discoverability for the session return type.
- Add `InferenceSession::kv_cache_telemetry()` as a thin read-only wrapper.
- Publish `docs/NNML2_KV_LOGICAL_CACHE_TELEMETRY_V1.md` and a short
  `docs/memory/` note; link from `docs/MODEL_RUNTIME.md`.
- Mark `NNML1_CLI_SAMPLED_STREAMING.md` completed after that merge; keep physical
  P0/P1 gates untouched.

## Required merge gates

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets --locked`
- strict Clippy under the repository workflow
- workspace tests (including host-only telemetry version/invariant tests)
- Rust 1.77 MSRV gate

## Claim boundary

This slice does not claim physical residency, payload visibility, eviction
policy, process-wide VRAM attribution, serving performance, or Thor parity.
