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
- Wave 3: `F32Elementwise` runtime-specialized CUBIN kernels for vector add,
  scale, and fused affine, with safe synchronizing and unsafe enqueue APIs.
- Wave 4: CUDA-event benchmark harness with warmups, full samples,
  min/median/mean/p95/p99/max/stddev, derived throughput, JSON reports, and
  git/GPU/driver/NVRTC metadata. A release elementwise benchmark validates its
  output after timing.
- Real Thor validation: runtime-compiled vector add passed for 1,025 elements;
  PTX and CUBIN paths, cache reuse, missing functions, and compiler failures
  are exercised. Elementwise kernels passed CPU-oracle checks for 0, 1, 31,
  32, 255, 256, 257, 1,025, and 4,097 elements. Workspace total: 20 tests
  passed with GPU required.

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
- Safe high-level kernel calls synchronize before returning; enqueue variants
  remain `unsafe` until device-buffer ownership can cover asynchronous work.
- Benchmarks record start/end events on the operation stream, synchronize the
  end event, and derive latency from `cuEventElapsedTime`; host enqueue time is
  never reported as GPU latency.

## Next task
Wave 5: expose a deliberate top-level `nnis` facade over runtime, JIT, and
kernels, then add the Wave 6 end-to-end example using event timing.

## Commands that passed
Takeover run (2026-08-22):
- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` (15 passed)
- `git diff --check`
- `NNIS_REQUIRE_GPU=1 cargo test -p nnis-kernels --all-targets -- --nocapture`
  (2 new GPU tests passed; workspace total now 17)
- `NNIS_REQUIRE_GPU=1 cargo test -p nnis-bench --all-targets -- --nocapture`
  (3 benchmark tests passed, including real event-timed GPU execution)

## Measured benchmarks
- Thor, CC 11.0, driver/NVRTC 13.0, release `nnis_scale_f32`, 16,777,216
  elements (134,217,728 read+write bytes), 20 warmups, 100 iterations:
  0.642272 ms median, 0.711525 ms p95, 0.728031 ms p99, 0.627520-0.731072 ms
  range, 208.973 GB/s derived median throughput. Result validation passed.
  This first run identified commit `00f9d20` with `git_dirty=true` because the
  benchmark harness itself was the uncommitted change; rerun after its commit.

## Blockers
- No implementation blocker. Sandbox device isolation requires GPU commands to
  run with direct hardware access.

## Recent changes
Protected baseline `d086ec2`; pushed milestones: `f7b39c6`, `34c8168`, and
Wave 3 kernel commit `00f9d20`.
The initial raw launch crashed in `cuLaunchKernel`; GDB proved that device
addresses had been supplied where CUDA expects host pointers to argument
values. The validated typed launcher fixes that root cause. Next command:
`cargo check -p nnis --all-targets` after implementing the facade.
