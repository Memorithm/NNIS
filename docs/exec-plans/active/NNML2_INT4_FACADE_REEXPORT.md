# NNML2 INT4 facade re-export surface

Status: **active** (selected next non-physical software slice after PR #154
KvCacheTelemetry schema v1 / PR #153 INT4 projection; may land alongside or
after in-flight PR #155 NVML process-memory CLI).

## Goal

Publish the already-merged INT4 reference storage and isolated projection
contracts through the public `nnis` facade so callers do not need to depend on
`nnis-model` / `nnis-kernels` directly for the versioned plan types. No physical
Thor execution and no fake campaign results.

## Scope

- Re-export from `nnis::model` and crate root:
  `Int4ReferenceModelStorageV1`, `Int4ReferenceProjectionPlanV1`,
  `Int4ReferenceStorageSummaryV1`, `Int4ReferenceAllocationSummaryV1`,
  `Int4ReferenceQuantizedTensorV1`,
  `quantize_int4_symmetric_reference_v1`, `dequantize_int4_symmetric_reference_v1`,
  and the `NNIS_INT4_REFERENCE_*` version/identity constants.
- Re-export `F32Int4Gemv` from `nnis::kernels` and crate root.
- Host-only facade test covering version constants and plan construction.
- Document the facade surface under `docs/NNML2_INT4_PROJECTION_V1.md` and a
  short `docs/memory/` note; keep claim boundary fail-closed.
- Mark `NNML2_KV_LOGICAL_CACHE_TELEMETRY.md` completed (PR #154) if still active.

## Required merge gates

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets --locked`
- strict Clippy under the repository workflow
- workspace tests (including host-only facade INT4 re-export test)
- Rust 1.77 MSRV gate

## Claim boundary

Software facade publicity only. Does not claim full-model INT4 execution,
equality to dense projections, model quality, serving performance, physical
residency, or Thor parity. `execution_qualified` on storage summaries remains
false.
