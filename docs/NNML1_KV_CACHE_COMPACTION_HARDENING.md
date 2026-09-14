# NNML1 KV-cache compaction hardening

NNIS compaction operates on physical active-row positions in the owned KV cache. It does not reinterpret token identities or RoPE positions.

## Failure semantics

`compact_kv_cache` and non-trivial `compact_kv_cache_layer` selections construct a replacement cache before committing the mutation. The original cache remains logically unchanged if validation, allocation, CUDA staging, append, or synchronization fails before the final swap.

All retained positions are required to be strictly increasing, unique, and inside the current active prefix. The implementation resolves and range-checks every complete K/V row region before the first `cuMemcpyAsync` for a staging operation.

A single-layer compaction preserves every non-target layer exactly. This currently requires rebuilding the fixed-capacity cache and is intentionally correctness-first rather than a performance optimization.

## Boundary

The fixed-capacity allocations remain fixed-capacity. Reducing the active row count therefore does not by itself establish reduced allocated HBM, lower memory traffic, lower latency, higher throughput, or preserved model quality. Those claims require separately observed runtime evidence.
