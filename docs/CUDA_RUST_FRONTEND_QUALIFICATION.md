# CUDA Rust frontend qualification gate

Status: non-performance, non-production qualification scaffold.

Date: 2026-09-08.

## External trigger

NVIDIA published "Introducing CUDA Rust: Two Tracks for Writing GPU Kernels" on 2026-09-08. It describes two distinct native-Rust paths:

- `cuda-oxide`: SIMT-style Rust compiled to PTX through a custom rustc codegen backend;
- `cutile-rs`: tile-oriented Rust whose compiler/runtime uses CUDA Tile IR JIT compilation.

Primary source: https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/

NVIDIA Research separately describes cuTile Rust and an end-to-end Rust inference engine in "Fearless Concurrency on the GPU" (2026-06-16, arXiv:2606.15991). Those external measurements are prior-art evidence only; they are not NNIS performance evidence.

Primary source: https://research.nvidia.com/publication/2026-06_fearless-concurrency-gpu

## Why NNIS needs an explicit boundary

NNIS currently owns a qualified NVRTC compilation path and CUDA module loading. A native-Rust frontend must not be silently treated as equivalent merely because it eventually executes on CUDA.

`nnis-jit::KernelFrontendContract` therefore separates:

1. frontend identity;
2. artifact hand-off boundary (`PTX`, `CUBIN`, or runtime-managed);
3. qualification state.

Recognition is not promotion. Both native-Rust contracts remain `Experimental`, so `production_routing_allowed()` is false.

## Qualification sequence

A native-Rust frontend may become qualified only after a dedicated PR provides all applicable evidence below on an exact revision:

1. pinned frontend/toolchain identity and license/provenance;
2. deterministic or explicitly characterized compile/JIT inputs;
3. artifact-boundary validation without mislabeling runtime-managed Tile artifacts as PTX/CUBIN;
4. correctness parity against an existing qualified NNIS kernel on fixed inputs;
5. negative tests for malformed shapes/arguments and unsupported devices;
6. device/runtime provenance including CUDA, driver, GPU and compiler versions;
7. measured compile/JIT cost and steady-state runtime separately;
8. no performance claim until repeated physical-device measurements exist.

## First candidate experiments

### SIMT / PTX

A future isolated adapter may accept PTX emitted by a pinned `cuda-oxide` toolchain and load it through NNIS' existing module boundary. The first experiment should be vector-add or another already-qualified kernel, not a new optimized kernel, so frontend correctness is separated from optimization.

### Tile / runtime-managed

`cutile-rs` is modeled as `RuntimeManaged` until an actual adapter is implemented and its ownership/JIT contract is understood from executable evidence. It must not be routed through `Module::load` merely by assuming an undocumented artifact shape.

## Ecosystem transfer

- **SciRust / FLAT-ATTENTION:** potential future Rust-native kernel authoring backend after NNIS qualification evidence exists. Do not duplicate frontend-specific safety semantics independently.
- **Forge:** may search kernel parameters only after a destination-owned qualification contract identifies a valid frontend/backend and correctness oracle; frontend choice itself is not a scientific verdict.
- **ElasticXxx:** may later select among already-qualified execution capabilities, but must treat experimental CUDA-Rust frontends as unsupported rather than available.

No dependency on `cuda-oxide` or `cutile-rs` is added by this scaffold, and no external benchmark result is imported as a Memorithm result.
