# NVML process GPU memory snapshot v1

`NvmlProcessMemorySnapshotV1` is an optional NNIS runtime telemetry contract for the current process on one CUDA device.

## Source

The value comes directly from NVML `nvmlDeviceGetComputeRunningProcesses_v3` and the matching `nvmlProcessInfo_t.usedGpuMemory` record for `std::process::id()`.

Device correlation is by CUDA UUID, not by ordinal. NNIS formats the CUDA UUID as the canonical `GPU-<uuid>` string and resolves the corresponding NVML device with `nvmlDeviceGetHandleByUUID`.

NVML is dynamically loaded from `libnvidia-ml.so.1`/`libnvidia-ml.so`, optionally overridden by `NNIS_NVML_PATH`. NNIS does not acquire a link-time dependency on NVML.

## Fail-closed behavior

The probe returns an error instead of inventing a value when:

- NVML or a required symbol is unavailable;
- the CUDA device has no UUID;
- NVML cannot resolve that UUID;
- the compute-process query is unsupported or denied;
- the process list cannot stabilize within a bounded retry budget;
- the current PID is absent;
- multiple records for the current PID make attribution ambiguous;
- `usedGpuMemory` is `NVML_VALUE_NOT_AVAILABLE`.

The process-list allocation is bounded to 65,536 records.

## Relationship to NNIS allocation accounting

This contract is intentionally distinct from `WeightAllocationSummaryV1` and the F16 materialization evidence contracts.

- `WeightAllocationSummaryV1` accounts exact bytes requested by live `DeviceBuffer` allocations owned by a specific NNIS weight graph.
- F16 materialization evidence accounts exact NNIS-owned buffer lifetimes in the scoped F32 -> F16 conversion path.
- `NvmlProcessMemorySnapshotV1` reports the process-scoped memory value attributed by NVML to the current PID on the target GPU.

A difference between NVML process memory and NNIS-owned buffer accounting is not automatically allocator overhead. It can include memory outside the weight graph and requires separate attribution.

## Explicit non-claims

V1 does **not** claim:

- physical GPU page residency;
- per-allocation page residency;
- CUDA allocator/page-table overhead attribution;
- that all process-scoped bytes are owned by NNIS weights;
- that RoPE, KV, session, workspace, JIT or module memory has been separated;
- cross-runtime memory equivalence;
- compression ratio, effective bits/parameter, latency, throughput or quality.

A physical Thor campaign is still required before this metric can be used as evidence for a concrete NNIS model run.
