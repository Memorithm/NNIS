# NNML1 transactional KV-cache compaction

NNIS exposes correctness-first physical KV-cache compaction through `nnis_rt::compact_kv_cache` and the compatibility wrapper `compact_kv_cache_layer`.

## Contract

- selections address physical rows in the current active cache prefix;
- retained positions must be strictly increasing, unique and in range;
- K/V rows are copied in retained-position order;
- RoPE/token logical position semantics are not rewritten by the runtime primitive;
- the source cache is synchronized before capture and is never mutated while the replacement cache is built;
- a failed allocation, transfer, append or synchronization leaves the source cache logically unchanged;
- `compact_kv_cache_layer` commits by replacing the original cache only after a complete replacement cache has been constructed successfully.

The implementation stages selected rows through separate device allocations before appending them into a new cache, so capture and destination regions do not overlap.

## Scientific boundary

This primitive establishes exact physical row retention semantics. It does not by itself establish memory-footprint reduction, HBM-traffic reduction, latency improvement, throughput improvement, or model-quality preservation. Those properties require separately captured device/model evidence.

The compaction API works on physical row positions, not vocabulary token values. A higher-level model adapter must maintain any mapping from experiment token identities or logical positions to the currently materialized physical rows.
