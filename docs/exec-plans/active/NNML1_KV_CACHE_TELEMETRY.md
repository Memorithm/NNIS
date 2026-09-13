# NNML1 KV cache telemetry surface

## Current slice

- expose `InferenceSession::kv_cache_telemetry` as a synchronizing read-only observer;
- re-export `KvCacheTelemetry` / `observe_kv_cache` and NVML process-memory snapshot types through the `nnis` facade;
- document the claim boundary in `docs/NNML1_KV_CACHE_TELEMETRY.md`.

## Required merge gates

- `cargo fmt --all -- --check`;
- `cargo check --workspace --all-targets --locked`;
- strict Clippy;
- workspace tests;
- Rust 1.77 MSRV.

## Boundaries

No physical residency claim, no KV compression promotion, no serving performance claim.
