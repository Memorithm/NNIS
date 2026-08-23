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
- Kernel tuning infrastructure can load the real `F32Elementwise` family with
  a validated explicit width, report per-operation occupancy, and sweep widths
  through the existing CUDA-event benchmark harness with result validation.
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
- Occupancy is treated as a candidate generator, not a performance oracle. The
  256-thread elementwise default stays because the clean ordered sweeps did not
  show a repeatable improvement from any alternative.

- Arbitrary-length `f32` sum reduction shipped: multi-pass tree reduction
  kernel (two-elements-per-lane load + shared-memory stride halving), reusable
  context-bound workspace, safe synchronizing `sum`/`sum_into` plus unsafe
  `enqueue_sum`, exact ordered-tree bit-for-bit CPU oracle, forward-error-bound
  tolerance checks over 17 sizes (0..=1,000,003), invalid-shape/workspace
  rejection tests, CUDA-event benchmark example with pass count, traffic model,
  and post-timing result validation.
- `F32Reduction` extended with a second tree kernel `nnis_reduce_max_f32`
  sharing one workspace: `max`, `max_into`, unsafe `enqueue_max`; fmaxf is
  exact so tests assert bit-for-bit equality against a CPU max across all
  boundary sizes; empty input leaves destination untouched (sum zeroes it).
- Numerically stable `F32Softmax` shipped as a four-stage native async
  pipeline on one stream with zero host roundtrips: device-side max ->
  exp(input - max) -> device-side sum -> in-place normalize by the
  device-resident total. Safe wrappers synchronize once; scalar intermediates
  stay on device between stages. f64-oracle tests across sizes 0..=4,097,
  singleton -> exactly 1.0, constant input -> uniform, invalid-shape and
  undersized-workspace rejection, distinct-scratch validation; Session gained
  a validated softmax path; CUDA-event benchmark example validates after
  timing.

- Row-batched `F32Softmax2D` shipped (attention-shaped primitive): one block
  per row strided max/sum reductions plus per-element exp-shift and in-place
  normalize, all four stages enqueued on one stream with device-resident row
  scalars; reusable per-row workspace; f64-oracle tests over 9 shapes
  (1x1..17x4097), uniform-row check, invalid-shape/capacity rejection.
  Debugging note: a stage initially produced NaN because the normalize launch
  pushed its data buffer twice, shifting every later kernel argument; the
  probe-isolate-per-stage workflow found the root cause quickly. CUDA-event
  benchmark example validates after timing.

- Fused single-kernel row softmax shipped (`nnis_softmax_row_fused_f32`):
  one block stages its whole row in dynamic shared memory and performs max,
  exp, sum, normalize in place - two matrix streams instead of six, no
  workspace, no host roundtrips. Launch validation rejects rows where
  `(cols + block_size) * 4` exceeds the function dynamic-shared limit.
  f64-oracle tests across all shapes plus oversized-row rejection; benchmark
  example gained an `NNIS_BENCH_FUSED=1` A/B mode.

- Automatic row-softmax dispatch shipped: `softmax_rows_dispatched` picks
  the fused kernel when `fused_available(cols)` (row + partials fit dynamic
  shared memory) and falls back to the staged pipeline with freshly allocated
  scratch otherwise; tested fused-selected, narrow, and oversized-row paths.
  `Session` now loads and exposes `softmax_2d`. Workspace total: 36 GPU tests.

- `F32Gemv` shipped: `y = A*x` for row-major matrices, one block per row
  with strided explicit-FMA accumulation and shared-memory tree reduction.
  The CPU oracle replays the identical evaluation order via `mul_add`, so
  GPU results are bit-exact regardless of compiler FMA contraction; an
  independent f64 sum cross-checks every row. Ten boundary shapes,
  invalid-shape rejection, facade/Session integration (`session.gemv()`),
  and a validated event-timed benchmark example.

- RMSNorm milestone (2026-08-23, branch `feature/rmsnorm`, PR workflow):
  staged/fused/dispatched f32 RMS normalization following the LayerNorm
  template; fmt/clippy/check clean; 50 GPU tests passed on the branch;
  clean benchmark at `873ca3d`: 4096x4096 fused, 0.670544 ms median,
  0.638912 min, 200.162 GB/s, max element error 4.89e-7 - faster than
  fused layer norm (0.707 ms) because one reduction pass replaces two.

- Cross-stream pooling milestone (2026-08-23, branch
  `feature/pooled-cross-stream`, PR workflow): `PooledBuffer::share_with`
  grants a second stream access via an explicit producer-side event chain,
  registers the consumer, and orders the drop-time free after all work the
  consumer had enqueued; `PoolOptions` exposes the three CUDA reuse flags
  (defaults match the driver). Strict-pool tests cycle shared buffers eight
  times and verify no consumer write leaks into recycled blocks; repeated
  grants deduplicate; same-stream grants are no-ops. Mutation experiments
  showed CUDA's pool already repairs ordering whenever any event chain
  exists, so tests assert the observable contract rather than one side of
  the implementation. fmt/clippy/check clean; 52 GPU tests passed.

- Pooled-pipeline milestone (2026-08-23, branch `feature/pooled-softmax`,
  PR workflow): `F32Softmax::softmax_pooled` runs the four-stage pipeline
  with reduction workspace AND both scalars taken stream-ordered from a
  `StreamOrderedAllocator`; `F32Reduction::pooled_workspace` +
  pointer-form `enqueue_sum_ptr`/`enqueue_max_ptr` back it. Clean A/B at
  three sizes (5 warmups, 50 iterations), all paths oracle-validated:
  - 65,536 elements: sync-alloc 0.0528 ms vs pooled 0.0479 vs owned floor
    0.0295 (pooled 1.10x over sync)
  - 1,048,576: 0.3181 vs 0.1521 vs 0.1319 (pooled **2.09x** over sync)
  - 16,777,216: 2.9602 vs 2.4959 vs 2.4756 (pooled 1.19x; within 0.8% of
    the allocator-free floor)
  Verdict per DESIGN_ALLOCATION_POOLING.md decision criteria: pooling
  delivers a reproducible end-to-end win on the real multi-stage softmax
  pipeline and lands within measurement noise of full pre-allocation.
  fmt/clippy/check clean; 53 GPU tests passed.

- Pooled-norms milestone (2026-08-23, branch `feature/pooled-norms`,
  stacked on PR #3): `F32RmsNorm::normalize_rows_dispatched_pooled` and
  `F32LayerNorm::normalize_rows_dispatched_pooled` run fused when the row
  fits shared memory and pooled staged workspaces otherwise; per-row
  statistic columns live behind an internal RowColumn enum with pointer
  launches and capacity validation. Oracle tests cover fused-selected,
  staged-fallback (20,000 columns), and undersized-pooled-workspace
  rejection for both families. fmt/clippy/check clean; 55 GPU tests.
  No dedicated benchmark: the mechanism and economics are identical to the
  softmax A/B in PR #3 (pool overhead ~1 microsecond per buffer).

- bf16 milestone (2026-08-23, branch `feature/bf16-elementwise`, stacked on
  #3/#4): numeric policy fixed as **bf16 storage, f32 compute**; RNE
  conversion implemented bit-identically in host helpers
  (`nnis_rt::f32_to_bf16_rne` / `bf16_bits_to_f32`) and device bit-math
  (no CUDA headers needed under NVRTC). `Bf16Elementwise` family:
  widen/narrow conversions, vector_add, scale, affine (explicit FMA) -
  all oracle tests assert BIT-EXACT equality against host replay.
  Facade/Session wired (`session.bf16_elementwise()`). Clean benchmark at
  16M elements: 0.611616 ms median, 164.586 GB/s derived (6 bytes per
  element), bit-exact post-timing validation. Debugging note: kernels with
  different arities must not share one argument pack - extra entries are
  ignored by the driver but a kernel reading its count from entry N picks
  up whatever scalar occupies that slot (found twice, fixed by dedicated
  per-signature packs). fmt/clippy/check clean; 58 GPU tests passed.

## Workflow rule (2026-08-23, owner decision)
Pull requests are mandatory from now on: every wave lands on a
`feature/<name>` branch and reaches `main` only through a GitHub PR after
the full validation loop passes on the branch. Direct pushes to `main` are
no longer allowed for code changes.

## Next task
All planned waves are complete. Remaining candidates in priority order:
1. Cross-stream pooled-buffer handoff via explicit event record/wait
   (`share_with`) per step 3 of DESIGN_ALLOCATION_POOLING.md.
2. Wire `StreamOrderedAllocator` into a real pipeline (flat softmax scratch)
   behind an API that keeps today's safe defaults untouched.
3. bf16/f16 elementwise + reduction kernels once a numeric policy is chosen.

## Measured benchmarks (continued)
- Clean pooling A/B at `4a51645` (`git_dirty=false`), Thor CC 11.0,
  5 warmups, 50 iterations per size; per iteration: 3 pool allocs ->
  vector-add kernel -> 3 stream-ordered frees vs pre-allocated buffers:
  - 256 elements: 0.007360 vs 0.004480 ms (1.64x)
  - 4,096: 0.007392 vs 0.004416 ms (1.67x)
  - 65,536: 0.007616 vs 0.006016 ms (1.27x)
  - 1,048,576: 0.022080 vs 0.020352 ms (1.08x)
  - 16,777,216: 0.883552 vs 0.863120 ms (1.02x)
  Verdict recorded per the design note's decision criteria: async-pool
  overhead is ~1 microsecond per buffer (3 buffers ~2.9 microseconds),
  versus ~37 microseconds per buffer for synchronous cuMemAlloc+cuMemFree
  from Wave 7 - roughly a 12x allocator-cost reduction for per-call
  scratch. Pre-allocation still wins when shapes are static because it
  pays zero allocator cost; pooling is the correct default for pipelines
  whose shapes vary per call. Both paths validated bit-exact after timing.
- Clean fused LayerNorm at `00dc763` (`git_dirty=false`), Thor CC 11.0,
  4096x4096 f32 (16,777,216 elements), block 256, fused shared-memory path,
  20 warmups, 100 iterations: 0.707136 ms median, 0.681984 min,
  stddev 0.023695 ms; 189.805 GB/s derived; max element error 6.17e-7 vs the
  f64 oracle, validated after timing. Dirty pre-commit run agreed
  (0.731152 ms median, 183.570 GB/s).
- Clean GEMV benchmark at `3e2893f` (`git_dirty=false`), Thor CC 11.0,
  4096x4096 f32 matrix, block 256, 20 warmups, 100 iterations:
  0.377824 ms median, 0.375520 min, stddev 0.009716 ms; 177.706 GB/s derived;
  max absolute error 0.0 against the f64 reference on this data distribution
  (the ordered oracle additionally matched bit-for-bit in tests).
- Pinned vs pageable 64 MiB f32 transfers, clean tree, Thor CC 11.0,
  20 warmups, 100 iterations (`nnis-bench` example `transfers`, event-timed,
  both paths data-validated): H2D pinned 0.627344 ms vs pageable 0.711184 ms
  (pinned 11.8% faster); D2H pinned 0.613056 ms vs pageable 0.636144 ms
  (pinned 3.6% faster). Positive result: pinned staging benefits both
  directions; largest win on H2D. Pooled/pinned staging is worth exposing to
  pipelines that stream host data.
- Fused-vs-staged row-softmax A/B, Thor CC 11.0, 8192x2048 f32 (16,777,216
  elements), block 256, 20 warmups, 100 iterations:
  - Staged four-kernel pipeline at clean `51fe244`: 2.507344 ms median,
    stddev 0.0358 ms.
  - Fused single kernel at clean `a3a3cd5` (`git_dirty=false`): 1.018608 ms
    median, 1.015744 min, 1.049754 p95, stddev 0.0126 ms; max element error
    4.09e-8 vs f64 oracle, validated.
  - Decision: fused retained - 2.46x faster end-to-end and lower variance.
    The staged path remains required for rows exceeding the shared-memory
    budget. Dirty-tree smoke at 1024x2048 agreed (0.1266 vs 0.2069 ms).
- Clean row-softmax benchmark at `ab0da37` (`git_dirty=false`), Thor CC 11.0,
  8192x2048 f32 (16,777,216 elements), block 256, four-stage pipeline,
  20 warmups, 100 iterations: 2.507344 ms median, 2.485664 min, 2.587187 p95,
  stddev 0.035823 ms; 160.590 GB/s derived; max element error 4.09e-8,
  validated. Observation: slower per byte than flat softmax (218 GB/s)
  because the staged pipeline moves six full-matrix streams; motivates the
  fused shared-memory row kernel (one read + one write).
- Clean softmax benchmark at `0459ee4` (`git_dirty=false`), Thor CC 11.0,
  16,777,216 f32 elements, block 256, four-stage pipeline, 20 warmups,
  100 iterations: 2.462560 ms median, 2.434816 min, 2.541634 p95,
  2.557857 p99, stddev 0.033968 ms; 218.227 GB/s derived traffic.
  Post-timing validation: max element error 2.21e-12 against f64 oracle;
  f32 probability-sum 0.9993 (expected f32 accumulation drift over 16.7M
  elements). The pipeline is bandwidth-bound like elementwise scale.
- Clean reduction benchmark at `4a22f3c` (`git_dirty=false`), Thor CC 11.0,
  16,777,216 f32 elements, block 256, 20 warmups, 100 iterations: 3 passes;
  0.477168 ms median, 0.465536 min, 0.541933 p95, 0.570264 p99, 141.190 GB/s
  derived from the multi-pass traffic model (values moved x 4 bytes).
  Post-timing validation passed: absolute error 1.94e-6 against an
  f64 reference, forward-error bound 99.31.

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
- Dirty-tree smoke run: `NNIS_BENCH_ELEMENTS=1048576 NNIS_BENCH_WARMUPS=3
  NNIS_BENCH_ITERATIONS=10 cargo run --release -p nnis-bench --example
  block_size_sweep` tested 128/256/512/768/1024-thread launches and validated
  every result. It is functional evidence only, not a retained performance
  result; clean full-size measurements are the exact next task.
- Sweep-infrastructure milestone: workspace fmt/check/clippy/doc and
  `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` passed (26 tests).
- Reduction takeover (2026-08-23): inherited uncommitted reduction work from a
  context-exhausted session, verified it, and passed `cargo fmt/check/clippy`,
  `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` (28 tests), plus a
  dirty-tree smoke run `NNIS_BENCH_ELEMENTS=1048576 NNIS_BENCH_WARMUPS=3
  NNIS_BENCH_ITERATIONS=10 cargo run --release -p nnis-bench --example
  reduction` (validated output; 3 passes; median 0.0342 ms).
- Softmax milestone (2026-08-23): workspace fmt/clippy/check clean;
  `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` = 31 passed
  (softmax oracle + rejection, max-reduction bit-exactness, Session softmax);
  dirty-tree smoke `NNIS_BENCH_ELEMENTS=1048576 NNIS_BENCH_WARMUPS=3
  NNIS_BENCH_ITERATIONS=10 cargo run --release -p nnis-bench --example
  softmax`: median 0.132 ms, probability sum 0.99998, max element error
  2.4e-11, validated.
- Row-softmax milestone (2026-08-23): workspace fmt/clippy/check clean;
  `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` = 33 passed;
  dirty-tree smoke `NNIS_BENCH_ELEMENTS=2097152 NNIS_BENCH_COLS=2048
  NNIS_BENCH_WARMUPS=3 NNIS_BENCH_ITERATIONS=10 cargo run --release -p
  nnis-bench --example row_softmax`: median 0.2069 ms at 1024x2048,
  243.25 GB/s derived, max element error 4.09e-8, validated.
- Dispatch milestone (2026-08-23): fmt/clippy/check clean;
  `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` = 36 passed;
  committed `ee5f646`, pushed to origin main.
- Documentation closeout (2026-08-23): ARCHITECTURE.md gained a "Kernel
  families" section (reduction tree semantics, flat softmax pipeline pattern,
  row softmax staged/fused paths and dispatch boundary); README quick start
  now covers reductions and dispatched row softmax with benchmark commands;
  `docs/DESIGN_ALLOCATION_POOLING.md` records the deferred pooling design
  (stream-ordered pools, safety rationale, decision criteria); new event-timed
  `transfers` benchmark example measured pinned vs pageable. fmt/clippy/check
  clean, 36 GPU tests passed, all committed and pushed.
- GEMV milestone (2026-08-23): fmt/clippy/check clean;
  `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` = 38 passed;
  committed `3e2893f`, pushed to origin main with clean benchmark record.
- Pooling milestone (2026-08-23): nnis-sys pool FFI + nnis-rt
  StreamOrderedAllocator/PooledBuffer shipped; fmt/clippy/check clean;
  `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` = 45 passed;
  committed `4a51645`, pushed with clean A/B benchmark record and verdict.
- LayerNorm milestone (2026-08-23): fmt/clippy/check clean;
  `NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets` = 43 passed
  (staged/fused/dispatched f64-oracle suites, constant-row stability,
  oversized-row and invalid-shape rejection); committed `00dc763`, pushed
  with clean benchmark record.
- Clean `50e6d96` full-size block sweeps ran 20 warmups + 100 measurements per
  width in forward and reverse order; all 1,000 measured kernel outputs were
  validated. The command was `NNIS_BENCH_ELEMENTS=16777216
  NNIS_BENCH_WARMUPS=20 NNIS_BENCH_ITERATIONS=100 cargo run --release -p
  nnis-bench --example block_size_sweep` with the two explicit candidate orders.

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
- Block-size investigation at clean `50e6d96`, 16,777,216 elements, 20 warmups,
  100 iterations per width, `git_dirty=false` (forward / reverse medians):
  128 threads (12 active blocks/SM) 0.627552 / 0.630720 ms; 256 (6) 0.644560 /
  0.625568 ms; 512 (3) 0.681936 / 0.662432 ms; 768 (2) 0.707584 / 0.691680 ms;
  1024 (1) 0.950224 / 0.941504 ms. CUDA recommended 768 threads, but it was
  9.8-10.6% slower than 256. The 128/256 winner reversed with order, so there
  is no reproducible evidence to replace the existing 256-thread default.

## Blockers
- No implementation blocker. Sandbox device isolation requires GPU commands to
  run with direct hardware access.

## Recent changes
Protected baseline `d086ec2`; pushed milestones include Wave 7 `6dd485f`,
memory hardening `bee938f`, ABI audit `ef04ffb`, docs `b688e26`,
introspection `9656323`, sweep `50e6d96`/`f3480f7`, reduction
`4a22f3c`/`85c27a2`, softmax `0459ee4`/`9924380`, row softmax
`ab0da37`/`51fe244`, fused row softmax `a3a3cd5`/`0f200b6`, and auto-dispatch
`ee5f646`. All pushed to origin main.
The initial raw launch crashed in `cuLaunchKernel`; GDB proved that device
addresses had been supplied where CUDA expects host pointers to argument
values. The validated typed launcher fixes that root cause. Next task: add and
validate a reusable multi-pass `f32` sum reduction.
