# NNML2 NVML process GPU memory probe v1

## Purpose

NNIS needs a source whose semantics can support process-scoped GPU-memory evidence without relabeling CUDA device-wide free/total telemetry as process ownership or physical residency.

`Context::process_gpu_memory_probe_v1()` is a fail-closed capability probe built on the NVML compute-running-process query. It is deliberately optional and dynamically loaded. NNIS continues to build and run when NVML is absent.

## Identity binding

The probe binds the CUDA execution device to NVML by the exact CUDA device UUID. It does not assume that CUDA ordinal N maps to NVML ordinal N.

The NVML device handle is obtained from the UUID string `GPU-<cuda-uuid>` before querying compute processes.

The returned process entry must match `std::process::id()` exactly.

## Available state

`ProcessGpuMemoryProbeV1::Available` is produced only when all of the following are true:

- `libnvidia-ml` can be loaded dynamically;
- NVML initialization succeeds;
- CUDA exposed a device UUID;
- NVML resolves that exact UUID;
- `nvmlDeviceGetComputeRunningProcesses_v3` is supported and succeeds;
- the current PID is present in the returned compute-process set; and
- `usedGpuMemory` is not `NVML_VALUE_NOT_AVAILABLE`.

The snapshot records the current PID, CUDA device ordinal, exact UUID and the `usedGpuMemory` byte value returned by NVML.

## Unavailable state

No estimate or CUDA fallback is synthesized. The probe reports an explicit unavailable reason for cases including:

- NVML library unavailable;
- NVML initialization failure;
- CUDA UUID unavailable;
- NVML device lookup failure;
- query not supported;
- permission denied;
- current process not reported;
- `usedGpuMemory` unavailable;
- process-list instability across bounded retries; or
- another native query failure.

`NVML_ERROR_INSUFFICIENT_SIZE` is handled with a bounded resize-and-retry loop because the process list can change between the size query and the data query.

## Jetson / Thor boundary

NNIS does not assume that NVML process-memory reporting is available on Jetson AGX Thor. A physical target must run the probe and persist whether the capability is `Available` or `Unavailable`.

An unavailable result is valid negative capability evidence. It must not be replaced with `cuMemGetInfo`, `tegrastats`, model geometry, or an inferred allocation delta.

## Machine-readable harness

`cargo run -p nnis-bench --example process_gpu_memory_probe -- --device 0`

prints a JSON report with:

- device identity;
- probe state;
- source and process memory bytes when available; or
- unavailable reason, native operation/status and detail when unavailable.

Set `NNIS_REQUIRE_NVML_PROCESS_MEMORY=1` when a qualification environment explicitly promises this capability. In that mode an unavailable probe exits non-zero after emitting the JSON evidence.

The harness performs no timing and no model execution.

## Explicit non-claims

Even an `Available` NVML result does **not** by itself establish:

- physical GPU page residency of individual NNIS allocations;
- exact equality between NVML process memory and the sum of NNIS-owned `DeviceBuffer` allocations;
- allocator, driver, module, JIT, context, KV, workspace, or other component attribution;
- a clean before/after allocation delta;
- memory compression or effective bits per parameter; or
- latency, throughput or quality.

The value is retained under the narrower semantic label: NVML-attributed compute-process GPU memory.

A later model-level evidence contract may compare this process-level observation with NNIS allocation accounting, but it must preserve the distinction between the two metrics.
