# KV logical cache telemetry v1

`KvCacheTelemetry` is the optional NNIS runtime metadata contract for logical
occupancy of one device-resident `KvCache`.

## Source

`observe_kv_cache` reads only public cache geometry and per-layer logical
lengths. It does not perform device-to-host copies of K/V payload bytes.

`InferenceSession::kv_cache_telemetry` (PR #138) is the session-facing wrapper
over the same observer for the decoder-owned cache.

Schema version:

```text
NNIS_KV_CACHE_TELEMETRY_VERSION = 1
```

Every observer-produced snapshot sets `schema_version` to that constant.

## Fail-closed behavior

Observation fails instead of wrapping when live or capacity aggregates overflow
`usize`, and it propagates cache indexing errors.

## Explicit non-claims

V1 does not claim physical residency, payload visibility, eviction policy,
process-wide VRAM attribution, or performance.
