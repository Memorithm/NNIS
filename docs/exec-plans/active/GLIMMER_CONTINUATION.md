# GLIMMER CONTINUATION - NNIS

## Current objective
Turn validated CUDA foundation into usable NVIDIA-native inference substrate.

## Baseline
- Commit: d086ec2d69f8a9ea388ec3a505038737f0bf0539
- Tests: 9 passed (7 nnis-rt GPU tests + 2 nnis-sys)
- cargo check/test/fmt all ok
- GPU: Thor (aarch64, CUDA 13.0, driver 580.00)

## Completed waves
- Baseline verified: device enumeration, context, mem, H2D/D2H, D2D, zero
- Wave 1: NVRTC lifecycle, diagnostics, PTX/CUBIN extraction, deterministic
  cache keys, and a process-local compiled-code cache.
- Wave 2: owned CUDA modules/functions plus validated grid/block/shared-memory,
  context, buffer, and typed argument packing for `cuLaunchKernel`.
- Real Thor validation: runtime-compiled vector add passed for 1,025 elements;
  PTX and CUBIN paths, cache reuse, missing functions, and compiler failures
  are exercised. Workspace total: 15 tests passed with GPU required.

## Architectural decisions
- nnis-sys: dynamic dlopen CUDA + NVRTC, no link-time dep
- nnis-rt: safe wrappers, primary context, streams, events, DeviceBuffer
- Preserve process-lifetime ownership of dynamically loaded libraries; cached
  function pointers must never outlive their `libloading::Library`.
- Kernel functions must retain their CUDA module; raw function handles may not
  outlive `cuModuleUnload`.
- CUDA kernel parameters are pointers to aligned host storage containing each
  argument value; device addresses must not be passed as parameter-array
  entries directly.
- JIT cache entries are immutable and keyed by source, ordered options,
  architecture, and output kind. Compilation occurs outside the cache mutex.

## Next task
Wave 3: turn `nnis-kernels` into a reusable runtime-specialized kernel library,
starting with vector add and fused affine across zero/boundary/non-multiple
sizes with CPU oracles and explicit floating-point tolerances.

## Commands that passed
Takeover run (2026-08-22):
- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` (15 passed)
- `git diff --check`

## Measured benchmarks
- None yet; benchmark infrastructure is Wave 4.

## Blockers
- No implementation blocker. Sandbox device isolation requires GPU commands to
  run with direct hardware access.

## Recent changes
Protected baseline `d086ec2` and pushed bootstrap `f7b39c6` remain intact.
The initial raw launch crashed in `cuLaunchKernel`; GDB proved that device
addresses had been supplied where CUDA expects host pointers to argument
values. The validated typed launcher fixes that root cause. Next command:
`cargo test -p nnis-kernels --lib -- --nocapture` after implementing Wave 3.
