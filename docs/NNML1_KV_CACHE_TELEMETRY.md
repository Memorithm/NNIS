# NNML1 KV cache telemetry

NNIS exposes read-only logical occupancy telemetry for device-resident KV caches.

## Surfaces

- `nnis_rt::observe_kv_cache` observes any `KvCache` without copying K/V payload bytes.
- `InferenceSession::kv_cache_telemetry` synchronizes the session stream, retires pending appends, then returns the same logical snapshot.
- The `nnis` facade re-exports `KvCacheTelemetry`, `observe_kv_cache`, and the NVML process-memory snapshot types for harness authors.

## Contract

`KvCacheTelemetry` reports:

- per-layer valid token lengths;
- aggregate live and capacity token counts;
- layer count, head count, and head dimension.

It does not report physical page residency, allocator overhead, eviction policy, or backend placement decisions. NVML process memory remains a separate process-scoped contract and must not be relabelled as per-allocation KV ownership.

## Qualification boundary

This is an NNML1/NNML2 observability software surface. Publishing telemetry types does not authorize compressed-KV promotion, multi-session concurrency claims, or physical Thor qualification.
