# NNIS-P0 — Portable Runtime Sovereignty Inventory

Status: implementation inventory for the V888 / SML / FLAT portable-runtime programme.

This document records the current coupling visible in the repository before any backend-neutral refactor. It is an engineering inventory, not a performance result and not authorization to delete historical NVIDIA code.

## Programme constraint

For new BANC v888, SML-GENIUS sparse-recurrent, event-driven and FLAT sparse-routing work, NNIS targets the **Native Neural Inference Stack** direction defined in `docs/V888_CONNECTOME_BOOTSTRAP.md`:

- Rust host implementation;
- CPU reference backend;
- WGPU/open portable GPU backend;
- no CUDA, NVRTC, TensorRT, TensorRT-LLM, NVML or CUBIN dependency in the new portable path.

Existing NVIDIA-specific code and evidence remain valid historical/reference material until a separately reviewed migration moves, quarantines or removes them.

## Current workspace dependency graph

The root workspace currently contains:

```text
nnis-sys
   ↑
nnis-rt
   ↑  ↖
nnis-jit
   ↑   ↑
nnis-kernels
   ↑    ↑
nnis-model
   ↑
nnis
   ↑
nnis-cli

nnis-bench → nnis-rt + nnis-sys
           → dev: nnis-jit + nnis-kernels + nnis-model
```

The actual Cargo manifests establish the following direct dependencies:

| Crate | Direct NNIS dependencies | Current classification |
| --- | --- | --- |
| `nnis-sys` | none | NVIDIA-specific leaf |
| `nnis-rt` | `nnis-sys` | NVIDIA-specific implementation containing some reusable semantic concepts |
| `nnis-jit` | `nnis-sys`, `nnis-rt` | CUDA/NVRTC-specific implementation |
| `nnis-kernels` | `nnis-jit`, `nnis-rt`, `nnis-sys` | CUDA-kernel implementation |
| `nnis-model` | `nnis-jit`, `nnis-kernels`, `nnis-rt` | mixed: portable model/storage semantics plus CUDA-bound execution |
| `nnis-bench` | `nnis-rt`, `nnis-sys` | mixed: report semantics plus CUDA timing/environment acquisition |
| `nnis` | `nnis-jit`, `nnis-kernels`, `nnis-model`, `nnis-rt` | CUDA-bound facade |
| `nnis-cli` | `nnis`, `nnis-model` | mixed CLI, currently CUDA-bound for execution |

## Crate-by-crate inventory

### nnis-sys — legacy/backend-specific

Observed source boundary:

- raw dynamically loaded CUDA Driver API;
- NVRTC;
- NVML;
- CUDA handles and device pointers.

Decision:

- do not generalize these raw types into the portable contract;
- retain as a backend-specific historical/reference implementation during migration;
- no new V888/SML/FLAT portable code may depend on `nnis-sys`.

### nnis-rt — split required

Current public surface includes:

- `Device` / `DeviceProps`;
- `Context`;
- `Stream` / `Event`;
- `DeviceBuffer` / `PinnedBuffer`;
- stream-ordered allocation;
- KV cache and KV telemetry;
- NVML process-memory telemetry;
- BF16 conversion helpers.

Its crate manifest directly depends on `nnis-sys`, and its module documentation identifies it as a CUDA/NVIDIA driver runtime.

Portable candidates to extract as backend-neutral semantics:

- capability descriptors that do not encode CUDA compute capability;
- buffer size/usage contracts;
- queue/fence semantics;
- portable memory-class vocabulary;
- KV logical layout and accounting when independent from CUDA handles;
- BF16 bit conversion helpers;
- generic error categories that do not expose CUDA result codes.

Backend-specific remnants:

- CUDA device/context ownership;
- CUDA streams/events;
- CUDA device/pinned allocation implementation;
- NVML telemetry;
- CUDA memory-pool implementation.

### nnis-jit — legacy/backend-specific implementation

Current module explicitly owns runtime CUDA compilation, module loading and launch. Tests compile CUDA C source into PTX/CUBIN through NVRTC and load it on a CUDA context.

Decision:

- retain as a backend-specific reference during migration;
- do not make PTX, CUBIN, NVRTC compile options or CUDA occupancy part of the portable API;
- replace the portable-facing concept with a versioned kernel package/dispatch contract in NNIS-P4.

Potentially reusable ideas, after renaming and decoupling:

- deterministic kernel artifact cache identity;
- validated launch geometry;
- explicit capability requirements;
- kernel metadata hashing.

### nnis-kernels — semantic definitions reusable, implementation backend-specific

The crate currently imports `Context`, `DeviceBuffer`, `Stream`, JIT kernels and CUDA-system functionality. Its embedded kernel sources use CUDA C syntax and CUDA execution identifiers.

Decision:

- current executable kernel implementations remain backend-specific;
- mathematical operation contracts and CPU-oracle expectations may be reused;
- portable kernels must be re-expressed through a backend-neutral operation contract and WGSL/WGPU implementation;
- FLAT-ATTENTION remains owner of FLAT attention semantics instead of copying them into this crate.

### nnis-model — mixed; highest-value extraction target

The manifest describes this crate as model-neutral but its execution dependency chain reaches `nnis-jit`, `nnis-kernels` and `nnis-rt`.

Portable material to preserve/extract where source review confirms no device ownership:

- model/config/manifest schemas;
- safetensors loading and source validation;
- logical tensor/weight descriptors;
- INT4 reference representation and accounting;
- generation/sampling configuration semantics;
- logical KV metadata.

Execution surfaces to isolate behind a backend contract:

- device tensor allocation;
- inference session;
- kernel dispatch;
- resident KV realization;
- GPU generation loop.

P1 must separate these without changing model-format semantics silently.

### nnis-bench — mixed

The crate depends directly on `nnis-rt` and `nnis-sys`; current evidence captures NVIDIA/CUDA identity and event-based timing.

Portable candidates:

- versioned benchmark report schema;
- exact source/environment identity rules;
- compatibility/fingerprint comparison principles;
- latency distribution and workload metadata.

Backend-specific collectors:

- CUDA events;
- CUDA device properties;
- driver/NVRTC versions;
- NVIDIA process-memory observations.

P1/P10 should make acquisition pluggable while retaining old reports as historically scoped evidence.

### nnis facade — currently CUDA-bound

The facade currently reexports CUDA-derived `Context`, `Device`, `DeviceBuffer`, `Stream`, JIT types and native kernel families. `Session` constructs a CUDA context/stream/compiler/kernel set.

Decision:

- do not mutate `Session` in place before portable contracts exist;
- introduce a new backend-neutral surface first;
- retain legacy `Session` compatibility until migration evidence permits deprecation;
- new V888/SML portable consumers must target the new surface, not legacy `Session`.

### nnis-cli — mixed

Argument parsing/tokenization/generation configuration is mostly model/frontend logic, but current commands select a CUDA ordinal and instantiate CUDA execution paths. The NVML command is explicitly NVIDIA-only.

Decision:

- preserve parsing/model frontend where reusable;
- add backend-neutral device selection only after NNIS-P1/P2/P3 exist;
- move NVML-specific diagnostics under an explicit legacy/backend-specific command boundary rather than pretending they are portable.

## P1 extraction boundary

The smallest acceptable backend-neutral runtime contract should be dependency-light and must compile/test on a machine without NVIDIA libraries.

Provisional responsibilities:

```text
PortableDevice
CapabilitySet
BufferDesc / BufferUsage
MemoryClass
Queue
Fence
KernelPackage
BindingSchema
Dispatch
TimestampProvider
BackendId
```

The names are provisional until code review fixes the concrete Rust surface.

The P1 layer must not contain:

- CUDA ordinals;
- CUDA compute capability;
- PTX/CUBIN;
- NVRTC options;
- CUDA streams/events/contexts;
- raw device pointers;
- NVML;
- TensorRT concepts.

## Migration sequence

1. Add a small backend-neutral core crate or module without changing legacy execution.
2. Implement a deterministic CPU backend against that contract.
3. Qualify CPU operation semantics with tiny known-answer kernels.
4. Add WGPU backend and CPU differential tests.
5. Adapt FLAT-ATTENTION through its portable contract.
6. Add sparse recurrent/event primitives required by V888/SML.
7. Migrate model execution incrementally behind the portable backend.
8. Change public facade defaults only after end-to-end parity.
9. Quarantine/deprecate NVIDIA-specific paths only after portable replacement evidence exists.

## Non-regression rules

During migration:

- historical CUDA/NVIDIA evidence remains labelled with its original environment;
- portable code may not depend transitively on `nnis-sys`;
- a CPU fallback must never be reported as WGPU execution;
- model-format changes require a separate versioned migration;
- precision-policy changes are separate from backend migration;
- performance promotion requires exact-device evidence;
- no V888 scientific claim is owned by NNIS.

## NNIS-P0 completion criterion

P0 is complete when this inventory is reviewed and the P1 implementation starts from a backend-neutral dependency boundary that can compile without `nnis-sys`.

The next code slice is NNIS-P1: introduce the minimal portable runtime contract while leaving current CUDA execution behavior unchanged.
