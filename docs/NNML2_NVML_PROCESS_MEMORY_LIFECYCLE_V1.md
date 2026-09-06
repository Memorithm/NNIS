# NNML2 SmolLM2 NVML process-memory lifecycle evidence v1

This harness is the physical follow-up to `NvmlProcessMemorySnapshotV1`. It does not change model execution and does not promote a memory optimization.

## Pinned workload

The executable `smollm2_nvml_lifecycle_memory` requires the existing widened-F32 SmolLM2-135M fixture:

- repository: `HuggingFaceTB/SmolLM2-135M`
- revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- model SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- source dtype recorded by provenance: `bfloat16`
- loaded source graph dtype: `f32`

The F16 target must be admitted by `F16ReferenceExecutionPlan::smollm2_135m_thor_min_latency`, so the model/device capability gate remains owned by `nnis-model`.

## Lifecycle observations

The harness synchronizes the NNIS stream immediately before each NVML observation and emits three points:

1. `post_context_nvml_process_memory`
   - after CUDA device/context/stream construction;
   - before loading the model weights.
2. `source_f32_weights`
   - after the pinned source model is loaded and synchronized;
   - includes both NVML process `usedGpuMemory` and `WeightAllocationSummaryV1` for the exact live F32 weight graph.
3. `resident_f16_weights`
   - after the qualified F16 model is materialized, the source F32 graph has been consumed/dropped, and the stream is synchronized;
   - includes both NVML process `usedGpuMemory` and the exact resident F16 `WeightAllocationSummaryV1`.

The report also embeds `F16WeightMaterializationMemoryEvidenceV1` so the already-qualified NNIS-owned conversion peak remains available beside the process-scoped observations.

## Durable artifact output

The harness always emits the versioned JSON report to stdout. `--output FILE` additionally publishes those exact bytes atomically through a same-directory temporary file, `sync_all`, and rename. A partially written JSON file is therefore not a valid campaign artifact.

A physical campaign should bind every process to an explicit run context and retain the exact clean Git head:

```bash
export NNIS_BENCH_RUN_CONTEXT_ID=nnml2-smollm2-nvml-$(date -u +%Y%m%dT%H%M%SZ)
cargo run --locked -p nnis-bench --example smollm2_nvml_lifecycle_memory -- \
  --model /path/to/pinned/smollm2-135m-f32 \
  --device 0 \
  --output evidence/nnml2_smollm2_nvml_lifecycle_thor.json
```

The artifact is then validated independently:

```bash
python3 tools/validate_smollm2_nvml_lifecycle_memory.py \
  evidence/nnml2_smollm2_nvml_lifecycle_thor.json \
  --expected-git-commit "$(git rev-parse HEAD)" \
  --require-thor
```

`--require-thor` fails closed unless the report records a clean `aarch64` Git execution, a non-empty run-context id, a Jetson Thor platform identity, and Jetson power/clock evidence. It also requires all three NVML observations to share PID/device/UUID and reconciles the F32/F16 allocation summaries byte-for-byte with the embedded materialization evidence.

The validator deliberately performs no `NVML - owned allocations` subtraction and assigns no allocator, context, page-table, module, JIT, workspace, KV, session, or RoPE attribution.

## P0 physical bundle integration

`tools/run_p0_physical_qualification_bundle.py` consumes this contract as part of the existing exact-head physical campaign. A physical bundle now requires `NNIS_BENCH_RUN_CONTEXT_ID`, generates the pinned SmolLM2 fixture, runs `smollm2_nvml_lifecycle_memory --output`, validates that artifact with the exact bundle head and `--require-thor`, then records the artifact path, byte count and SHA-256 in `P0_PHYSICAL_QUALIFICATION.json`.

This keeps the memory evidence on the same clean `origin/main` head as the NNML0 loader gate and NNML1 SmolLM2/TinyLlama parity records. The bundle still sets `promotion_authorized` to false and records an explicit NNML2 memory claim boundary. Bundle inclusion is provenance/orchestration evidence; it does not by itself establish physical page residency or an allocator/runtime-overhead decomposition.

## Interpretation boundary

The following quantities are intentionally different:

- `WeightAllocationSummaryV1`: exact `DeviceBuffer` bytes owned by the named NNIS weight graph;
- `F16WeightMaterializationMemoryEvidenceV1`: exact NNIS-owned source/resident/temporary weight-buffer lifetimes during conversion;
- `NvmlProcessMemorySnapshotV1.used_gpu_memory_bytes`: NVML's process-scoped GPU-memory attribution for the current PID on the UUID-correlated device.

The harness does **not** define subtraction between these quantities as allocator overhead. A difference may include context, modules, JIT state, RoPE, workspaces, caches, runtime bookkeeping or other process allocations and requires separate evidence before attribution.

## Physical campaign rules

A result is admissible as NNIS evidence only when:

- the report is produced on NVIDIA Jetson AGX Thor under the existing SmolLM2 qualification environment;
- checkpoint/provenance validation succeeds;
- the F16 execution plan capability gate succeeds;
- NVML resolves the CUDA device by UUID;
- NVML reports exactly one record for the harness PID at every lifecycle point;
- none of the three process-memory values is `NVML_VALUE_NOT_AVAILABLE`;
- the exact Git head, device metadata, driver/runtime identity and report artifact are retained;
- the persisted artifact passes `validate_smollm2_nvml_lifecycle_memory.py --require-thor` for that same exact head;
- competing GPU activity is checked and recorded by the campaign operator rather than inferred from a process-memory delta.

## Explicit non-claims

This v1 evidence does not establish:

- physical GPU page residency;
- per-allocation residency;
- CUDA allocator/page-table/context/module/JIT overhead attribution;
- that all process-scoped bytes belong to NNIS weights;
- latency, throughput, numerical quality or compression ratio;
- effective bits/parameter;
- live F16 plan transition support;
- a low-bit representation baseline.

The purpose is to create a reproducible physical process-memory observation surface that can later be consumed alongside the exact NNIS-owned allocation evidence by the preregistered ElasticBitAllocation Stage B protocol. The software artifact/validator contract by itself is not a Thor measurement.
