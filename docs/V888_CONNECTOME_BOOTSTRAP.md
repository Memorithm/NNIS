# BANC V888 and portable native-runtime bootstrap for NNIS

Status: strategic bootstrap.

## Direction change

For this programme NNIS must not depend on NVIDIA products.

The repository currently implements an NVIDIA/CUDA-native stack. That code remains historical/reference evidence until separately removed, but no new V888/SML/FLAT work is to depend on CUDA, NVRTC, TensorRT, TensorRT-LLM, NVML, CUBIN or NVIDIA-specific runtime facilities.

The acronym NNIS is retained. The target meaning for the portable line is Native Neural Inference Stack. A repository-wide rename/documentation migration is a separate mechanical task after the portable core exists.

## V888 scope

Only FlyWire BANC v888 is used as the connectome source for this research programme. Raw data remains external.

Codex currently identifies BANC v888 as Female Adult Fly Brain and Nerve Cord, snapshot 2026-05-20, with 158,262 neurons and 3,037,361 aggregated connections.

Sources:
- https://codex.flywire.ai/?dataset=banc
- https://codex.flywire.ai/faq

## Runtime mission

NNIS should become the thin native execution layer between model semantics and heterogeneous hardware:

Rust model/runtime contracts
  -> portable execution planning
  -> CPU SIMD and WGPU/open GPU kernels
  -> FLAT-ATTENTION / SML / sparse recurrent kernels

NNIS does not own model science. SciRust owns generic numerical primitives, FLAT owns attention semantics, SML owns its model, and TDI owns scientific evaluation.

## Bootstrap sequence

### NNIS-P0 — sovereignty inventory

Current implementation inventory: [`PORTABLE_RUNTIME_INVENTORY_V1.md`](PORTABLE_RUNTIME_INVENTORY_V1.md).

Classify every public and internal subsystem as:
- semantic/model-neutral;
- CUDA-coupled;
- NVIDIA-observability-coupled;
- reusable with abstraction;
- legacy-only.

Freeze a migration matrix before moving code.

### NNIS-P1 — backend-neutral core contracts

Introduce or extract Rust traits/types for:
- Device;
- Buffer;
- CommandQueue/Stream equivalent;
- Event/Fence;
- Kernel;
- KernelModule;
- Dispatch;
- CapabilitySet;
- MemoryClass;
- Timestamp/benchmark source.

The contracts must not contain CUDA ordinals, PTX/CUBIN concepts or NVIDIA capability numbers.

### NNIS-P2 — CPU reference backend

Provide a deterministic CPU backend for:
- buffer ownership/copies;
- elementwise kernels;
- reductions;
- sparse gather/scatter;
- model scheduling.

It is the runtime oracle and CI fallback, not a performance claim.

### NNIS-P3 — WGPU backend

Implement portable GPU execution through WGPU/WGSL:
- adapter/device selection;
- storage buffers;
- command submission;
- timestamp queries where available;
- explicit capability detection;
- deterministic validation against CPU.

No project-authored C/C++ bridge and no vendor SDK requirement.

### NNIS-P4 — portable kernel package contract

Replace runtime compilation assumptions with a versioned kernel artifact description:
- WGSL source or validated portable IR;
- entry point;
- binding schema;
- workgroup geometry;
- capability requirements;
- numerical policy;
- source/evidence fingerprint.

Keep JIT/specialization optional and backend-neutral.

### NNIS-P5 — FLAT-ATTENTION integration

Consume FLAT through a stable portable contract:
- resident Q/K/V/O/LSE buffers;
- dense and sparse candidate forms;
- causal metadata;
- kernel variant/capability reporting.

NNIS schedules; FLAT defines correctness.

### NNIS-P6 — sparse recurrent graph runtime

Add scheduling primitives needed for V888-derived/SML research:
- CSR/CSC adjacency buffers;
- active-node/active-edge frontiers;
- segmented sparse propagation;
- delayed event queues;
- deterministic event-window batching;
- exact active-edge and queue telemetry.

Reference behavior comes from SciRust.

### NNIS-P7 — event-driven execution planner

Select among:
- fixed-step sparse propagation;
- event-driven sparse propagation;
- batched event windows;
- CPU or WGPU.

Selection is capability- and workload-based and must be observable. Static baselines remain available.

### NNIS-P8 — SML-GENIUS runtime surface

Support SML-specific but model-neutral execution needs:
- packed Boolean/ternary/low-bit pages;
- direct page routing;
- bounded recurrent cycles;
- sparse graph state;
- bounded memory operations;
- exact active parameter/page/edge counters.

No attention primitive may be silently inserted into the SML candidate path.

### NNIS-P9 — ElasticXxx actuation boundary

Expose measurements and versioned candidate plans to ElasticXxx:
- active-edge ratio;
- queue pressure;
- buffer residency;
- dispatch occupancy proxies;
- latency;
- memory use.

ElasticXxx may select a plan only through invariant-checked contracts. NNIS remains authoritative for whether a device action is executable.

### NNIS-P10 — vendor-neutral benchmark qualification

Qualify at least:
- x86_64 CPU where available;
- ARM64 CPU;
- one non-NVIDIA WGPU adapter or software Vulkan adapter for CI;
- additional GPUs only as optional hardware evidence.

Reports must identify the real adapter/backend and never generalize across devices.

### NNIS-P11 — legacy NVIDIA quarantine/deprecation decision

After CPU+WGPU parity:
- stop routing new features through CUDA-only abstractions;
- move NVIDIA-only examples/docs under an explicit legacy/reference boundary or remove them in a separately reviewed migration;
- update README/project expansion to Native Neural Inference Stack;
- preserve historical benchmark provenance without presenting NVIDIA as the product direction.

## V888-specific runtime experiments

NNIS itself does not ingest the biological dataset. It consumes compact graph artifacts produced through SciRust/SML contracts.

Required experiments:
- whole-graph sparse propagation microbenchmark;
- active-front propagation at multiple activity densities;
- fixed-step versus event-driven;
- random/degree-matched/V888-derived synthetic topology with identical dimensions;
- FLAT sparse-attention candidate routing;
- SML recurrent graph path.

## Exit criteria

Portable NNIS is established when:
1. core model/runtime tests pass without CUDA libraries installed;
2. CPU reference executes a complete small inference graph;
3. WGPU executes the same graph within declared numerical tolerances;
4. FLAT attention runs through the portable runtime;
5. sparse recurrent/event workloads run through the same runtime;
6. SML has a non-attention portable execution path;
7. all new V888 work passes on a machine with no NVIDIA software.

Until these gates pass, existing CUDA support is legacy capability, not evidence of the new architecture.
