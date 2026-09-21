# NNML2 KV logical cache telemetry v1

This document freezes the read-only logical occupancy contract for NNIS
device-resident KV caches introduced by PR #133 (`c188c8c`), exposed on the
session/facade surface by PR #138, and completed as a **versioned** public
software contract in this NNML2 slice.

It is evidence/tooling infrastructure for comparing cache state across backends.
It is not a residency, bandwidth, eviction, tiering, latency, or quality claim.

## Contract

`KvCacheTelemetry` reports logical occupancy of one fixed-capacity NNIS
`KvCache` without reading K/V payload bytes back to the host.

The contract version is:

```text
NNIS_KV_CACHE_TELEMETRY_VERSION = 1
```

Every observer-produced snapshot sets:

```text
schema_version = NNIS_KV_CACHE_TELEMETRY_VERSION
```

For one cache it records:

- `layer_lengths`: valid token count for each layer, in layer order;
- `total_live_tokens`: checked sum of those per-layer lengths;
- `total_capacity_tokens`: checked product `layers * capacity`;
- `layers`, `heads`, `head_dim`: geometry retained by the CUDA cache config.

Observers:

- `nnis_rt::observe_kv_cache(&cache)` for any owned `KvCache<T: DevicePod>`;
- `InferenceSession::kv_cache_telemetry()` for the session-private decoder cache
  (already present from PR #138).

Both paths share the same observer implementation. Aggregate counters fail closed
on `usize` overflow instead of wrapping.

The version constant is re-exported from `nnis_rt` and the `nnis` facade
(`runtime` module and crate root). `KvCacheTelemetry` remains available from
both `nnis::runtime` and `nnis::model`; the version constant is exported only
via `runtime` / crate root to avoid duplicate `E0252` exports.

## Ownership and transfer boundary

NNIS remains owner of NVIDIA/CUDA allocation and movement semantics for this
cache. The telemetry view is metadata only: it does not alter placement, copy
K/V bytes to the host, or introduce an eviction/tiering policy.

Comparable logical metadata may be consumed by experiment harnesses (for
example FLAT/KVLab integration work) without either repository duplicating the
other's cache implementation.

## Relationship to other NNML2 memory contracts

This contract is intentionally distinct from:

- `WeightAllocationSummaryV1` — exact owned `DeviceBuffer` bytes for a weight graph;
- F16 materialization memory/failure evidence — scoped conversion lifetimes;
- `NvmlProcessMemorySnapshotV1` — process-scoped NVML `usedGpuMemory` for a PID/device;
- `cuMemGetInfo` free/total snapshots — global device environment telemetry.

Logical KV occupancy is not process-wide VRAM attribution and is not physical
page residency.

## Fail-closed behavior

Observation returns an error when:

- a layer index cannot be read from the cache;
- the live-token sum would overflow `usize`;
- the capacity-token product would overflow `usize`.

Callers must not invent occupancy values when observation fails.

## Explicit non-claims

V1 does **not** establish:

- physical GPU page residency of K or V storage;
- host-visible K/V payload contents;
- eviction, paging, or tiering policy;
- allocator/page-table/context overhead attribution;
- bandwidth, latency, throughput, or compression benefits;
- cross-runtime bit-identical cache layouts;
- physical Thor parity or model-family admission.

A physical campaign remains required before any performance or residency claim
that merely uses this metadata as a supporting field.
