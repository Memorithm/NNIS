# Pascal Vector Projection (PVP) — NNIS portable runtime bootstrap

Status: portable-runtime research programme.

For this programme, NNIS means **Native Neural Inference Stack**.

## Role

NNIS is an optional execution and qualification carrier for the versioned PVP
layout produced by the SML/SciRust/FLAT programme.

It may:

- execute a CPU reference through the portable session contract;
- execute a WGPU realization through the portable session contract;
- retain exact cross-session parity/evidence;
- measure end-to-end runtime cost on real hardware;
- verify that a qualified component remains useful once embedded in an
  inference/session lifecycle.

It does not own SML model semantics and it is not a required dependency of the
final SML model.

## Hardware sovereignty

For all new SML/PVP work, the target path is vendor-neutral.

Allowed vendor-specific software layer:

- the installed GPU driver required by WGPU/open GPU access.

Forbidden as required dependencies of this programme:

- CUDA;
- NVRTC;
- cuDNN;
- TensorRT / TensorRT-LLM;
- NVML;
- CUTLASS;
- CUBIN-specific execution contracts;
- project-authored C/C++ vendor-SDK bridges.

NNIS already contains historical/current CUDA/NVRTC/NVML-oriented facilities.
They remain factual legacy/reference capabilities, but they are not selected by
this PVP programme and cannot provide portable PVP qualification evidence.

A machine having an NVIDIA GPU does not authorize use of CUDA/NVRTC for PVP.

## Inputs

From SML-GENIUS:

- frozen Pascal/ANF semantics;
- model-side task contract;
- final acceptance boundary.

From SciRust:

- versioned bitplane layout;
- scalar/SIMD reference and pack/unpack/transpose oracles;
- generic hardware capability metadata where appropriate.

From FLAT-ATTENTION:

- qualified portable WGPU PVP kernel contract;
- device capability requirements;
- exact kernel evidence/identity.

## Runtime programme

1. NNIS-PVP0: versioned adapter for the shared address-major bitplane layout.
2. NNIS-PVP1: CPU `PortableSessionV1` execution against the shared oracle.
3. NNIS-PVP2: WGPU portable session execution over the same immutable inputs.
4. NNIS-PVP3: CPU/WGPU exact parity, lifecycle/reset/reuse and resource
   accounting.
5. NNIS-PVP4: real-device end-to-end measurements including upload/residency,
   dispatch, synchronization and output-consumption costs.
6. NNIS-PVP5: return a bounded evidence envelope to SML-GENIUS for its
   internalization decision.

## SML and harness boundary

The final SML model must remain sufficient to itself. NNIS may help qualify and
prototype runtime realizations but must not become hidden model infrastructure.

The future `SML-HARNESS` is a separate orchestration/training/evaluation/
serving surface. NNIS does not own that harness and the model must not require
the harness to supply semantic behavior.

## Evidence boundary

Legacy NVIDIA-specific benchmark results cannot be transferred to PVP.
Component microbenchmarks cannot establish model throughput. WGPU/software
adapter execution without physical hardware evidence proves only code-path
correctness.
