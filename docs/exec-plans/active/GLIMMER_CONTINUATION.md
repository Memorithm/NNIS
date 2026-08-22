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
- Wave 5: top-level `nnis` facade with deliberate runtime/JIT/kernel modules,
  common root exports, a prelude, and a ready-to-use `Session`.
- Wave 6: explicit custom-CUDA end-to-end example covering Device -> Context
  -> buffers -> NVRTC CUBIN -> Module -> Kernel -> launch -> events -> CPU
  validation.
- Wave 7: reproducible JSON performance breakdown for NVRTC load/compile/cache,
  module load/unload, allocation/free, argument packing, host launch submission,
  event overhead, kernels, and H2D/D2H/D2D transfers. Inline aligned launch
  arguments replace one heap allocation and dynamic dispatch per parameter.
- Post-wave safety hardening: ordinary zero/H2D/D2H/D2D methods now retain all
  borrows through stream synchronization; explicitly unsafe `_async` variants
  preserve native overlap. Host transfers require `DevicePod`, and pinned
  allocations are initialized with valid empty/ZST slice handling.
- CUDA 13.0 header audit: driver/NVRTC/module/launch/copy/event/occupancy
  signatures match except the corrected `cuMemsetD8Async` fill argument
  (`unsigned char`, not `unsigned int`). An odd-byte GPU zero test covers it.
- Project-level usage and architecture documentation now covers the facade,
  custom JIT path, ownership graph, blocking/async lifetime contract,
  `DevicePod`, validation, and reproducible CUDA-event benchmark commands.
- JIT kernel introspection exposes cached typed function attributes, maximum
  active blocks per SM, and occupancy-based launch recommendations. Launch
  validation now applies function-specific thread/dynamic-shared-memory limits.
- Real Thor validation: runtime-compiled vector add passed for 1,025 elements;
  PTX and CUBIN paths, cache reuse, missing functions, and compiler failures
  are exercised. Elementwise kernels passed CPU-oracle checks for 0, 1, 31,
  32, 255, 256, 257, 1,025, and 4,097 elements. Workspace total: 26 tests
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
- The facade does not expose `nnis-sys`; raw FFI remains an implementation
  layer while custom-kernel users receive the validated JIT launch surface.
- Kernel arguments use one aligned contiguous store. The A/B baseline showed
  fewer allocations improve packing materially, while total launch submission
  remains driver-dominated; no deeper launch micro-optimization is justified.
- Safe borrowed-memory APIs never return with DMA still entitled to access the
  borrow. Async copies/zero/D2D are `unsafe` and document the completion-time
  lifetime obligation. `DevicePod` gates bytewise host representations.

## Next task
Benchmark the elementwise kernels at 128/256/512/768/1024 threads on Thor and
compare against CUDA's occupancy recommendation before changing dispatch.

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
- `NNIS_BENCH_ELEMENTS=16777216 NNIS_BENCH_WARMUPS=20
  NNIS_BENCH_ITERATIONS=100 cargo run --release -p nnis-bench --example
  elementwise` (clean commit `85420bd`, output validation passed)
- `cargo run --release -p nnis --example end_to_end` (1,000,003 results
  validated on clean commit `75ca7d6`: 21.591 ms JIT, 0.020736 ms GPU)
- `cargo run --release -p nnis-bench --example performance_breakdown`
  (all transfer/kernel outputs validated; JSON report emitted)
- `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` (25 passed after
  synchronous-safe / explicit-async memory hardening)
- `NNIS_REQUIRE_GPU=1 cargo test -p nnis-sys -p nnis-rt --all-targets`
  (14 passed, including corrected D8 ABI execution; workspace total 26)
- Documentation milestone: `cargo doc --workspace --no-deps`, full workspace
  check/clippy/fmt/diff checks, and `NNIS_REQUIRE_GPU=1 cargo test --workspace
  --all-targets` all passed (26 tests on the Thor).
- `cargo run --release -p nnis-jit --example inspect_kernel` executed a real
  NVRTC CUBIN/function query on Thor: 8 registers/thread, no local/static shared
  memory, 49,152-byte dynamic-shared limit, PTX/binary 11.0; CUDA recommended
  768 threads, 40 minimum blocks, and 2 active blocks/SM.
- Introspection milestone: workspace fmt/check/clippy/doc passed, followed by
  `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` (26 tests, including
  real attribute/occupancy queries and a launch using the recommended width).

## Measured benchmarks
- Thor, CC 11.0, driver/NVRTC 13.0, release `nnis_scale_f32`, 16,777,216
  elements (134,217,728 read+write bytes), 20 warmups, 100 iterations:
  0.623600 ms median, 0.683651 ms p95, 0.708187 ms p99, 0.615296-0.710816 ms
  range, 215.230 GB/s derived median throughput. Result validation passed;
  report provenance is commit `85420bd` with `git_dirty=false`.
- Wave 7 clean run at `6dd485f` (`git_dirty=false`): NVRTC library load
  5.793 ms; cold CUBIN compile 10.211 ms median; cache lookup 0.001371 ms;
  module load/unload 0.023477/0.001551 ms; 4 MiB allocation/free
  0.069487/0.042208 ms; inline argument packing 0.000121 ms; host launch
  submission 0.002120 ms; empty event pair 0.001088 ms; one-element kernel
  0.004576 ms; and 16,777,216-element scale 0.615392 ms (218.101 GB/s).
  The 64 MiB transfer medians were 0.621296 ms H2D, 0.611408 ms D2H, and
  0.599744 ms D2D; every data path passed validation.
- Optimization A/B: boxed argument packing was 0.000166 ms median and total
  host submission 0.002148-0.002194 ms. Inline aligned storage measured
  0.000120-0.000121 ms packing and 0.002084-0.002130 ms total submission across
  three runs. Retained: ~27% faster packing and ~2-4% total submission with
  lower allocator exposure. The ~7,000x cold-JIT/cache gap validates retaining
  the process-local compilation cache. Allocation pooling is the next measured
  optimization opportunity; it is deferred until stream-ordered ownership can
  be designed safely.

## Blockers
- No implementation blocker. Sandbox device isolation requires GPU commands to
  run with direct hardware access.

## Recent changes
Protected baseline `d086ec2`; pushed milestones: `f7b39c6`, `34c8168`,
Wave 3 `00f9d20`, Wave 4 `85420bd`, benchmark record `1282912`, facade /
end-to-end `75ca7d6`, Wave 7 `6dd485f`, performance record `8413eb0`, memory
safety hardening `bee938f`, and CUDA ABI audit `ef04ffb`.
The root guide and architecture/safety contract are the current documentation
milestone.
The initial raw launch crashed in `cuLaunchKernel`; GDB proved that device
addresses had been supplied where CUDA expects host pointers to argument
values. The validated typed launcher fixes that root cause. Next task: make
the already-bound CUDA function/occupancy queries available through safe JIT
introspection and validate the current kernel dispatch choice.
