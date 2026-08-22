# NNIS — AUTONOMOUS CONTINUATION FROM OX ALPHA

You are taking over an existing NVIDIA Native Inference Stack (NNIS) implementation after another coding agent stopped because its context window was exhausted.

This is a CONTINUATION task.

Do NOT restart the project.
Do NOT redesign from scratch.
Do NOT merely produce a plan.

Your job is to inspect the existing implementation, preserve its validated foundation, and continue implementing NNIS autonomously for as long as useful work remains.

Repository:

`/root/NNIS`

GitHub:

`Memorithm/NVIDIA-Native-Inference-Stack`

Baseline commit that MUST be preserved:

`d086ec2d69f8a9ea388ec3a505038737f0bf0539`

Baseline commit message:

`feat: bootstrap NNIS native CUDA runtime foundation`

The baseline has already been pushed to `origin/main`.

---

# HARDWARE

Current development machine:

* Linux aarch64
* NVIDIA Thor GPU
* NVIDIA driver 580.00
* CUDA 13.0
* nvcc 13.0.48
* approximately 122 GiB system RAM
* 14 CPU cores

Use the actual machine to execute GPU tests.

Never claim GPU functionality works merely because it compiles.

---

# VERIFIED BASELINE

Before this takeover, the following were already executed successfully:

```text
cargo check --workspace --all-targets
cargo test --workspace --all-targets
cargo fmt --all -- --check
git diff --check
```

Nine tests passed.

Existing GPU tests include real CUDA operations such as:

* device enumeration
* GPU context creation
* memory information
* host-to-device / device-to-host roundtrip
* device-to-device copy
* zeroing GPU memory

The baseline is therefore a working CUDA foundation.

Do not weaken or remove those tests.

---

# EXISTING ARCHITECTURE

The workspace already contains:

```text
crates/nnis-sys
crates/nnis-rt
crates/nnis-jit
crates/nnis-kernels
crates/nnis-bench
crates/nnis
```

Existing substantial implementation:

## nnis-sys

Low-level dynamically loaded CUDA/NVRTC bindings.

Existing areas include:

* CUDA Driver API
* NVRTC
* CUDA constants
* raw CUDA types

## nnis-rt

Safe runtime layer.

Existing areas include:

* Device
* DeviceProps
* primary CUDA Context
* streams
* events
* GPU buffers
* memory copies
* memory information
* error types

## Currently unfinished

These crates were intentionally created but are still essentially placeholders:

* `nnis-jit`
* `nnis-kernels`
* `nnis-bench`
* high-level facade `nnis`

Do not mistake these placeholders for completed functionality.

---

# FIRST ACTION

Start by running:

```bash
cd /root/NNIS
git status --short --branch
git log --oneline --decorate -5
cargo check --workspace --all-targets
NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets
cargo fmt --all -- --check
```

The worktree should initially be clean.

If it is not clean, inspect changes carefully before touching them.

Never discard existing user work.

---

# CONTEXT SURVIVAL RULE

You are a local model with a finite context window.

The previous agent died because its context was exhausted.

Do NOT repeat that failure.

Maintain durable project state on disk.

Create:

`docs/exec-plans/active/GLIMMER_CONTINUATION.md`

Keep it concise.

Record:

* current objective
* completed implementation waves
* important architectural decisions
* commands/tests that actually passed
* measured benchmark results
* discovered blockers
* exact next task

Update it after every substantial implementation wave.

After a coherent validated milestone:

1. format
2. check
3. test
4. run relevant GPU tests
5. inspect diff
6. commit it
7. continue

This allows another fresh agent/session to resume if your context ends.

Do not postpone all commits until the end.

---

# PRIMARY OBJECTIVE

Turn the existing low-level CUDA foundation into a genuinely usable NVIDIA-native inference substrate.

Work through the following waves unless repository evidence shows that a prerequisite must be changed.

---

# WAVE 1 — COMPLETE THE JIT PATH

Implement `nnis-jit` as the first major continuation.

Use the existing NVRTC support in `nnis-sys`.

Build a reusable JIT compilation path capable of taking CUDA source and producing executable GPU code.

At minimum investigate and implement, where supported by the existing driver bindings:

* NVRTC program lifecycle
* compile options
* architecture targeting derived from `DeviceProps`
* compilation logs
* PTX and/or CUBIN retrieval
* module loading
* function lookup
* safe lifetime ownership
* compilation errors with useful diagnostics
* deterministic cache keys
* optional in-memory compilation cache where justified

Do not shell out to `nvcc` as the primary JIT implementation if NVRTC already provides the correct native mechanism.

The JIT must have an actual GPU execution test.

A minimal acceptance test is something equivalent to:

1. create CUDA source at runtime
2. compile it through NNIS
3. load it
4. obtain a kernel function
5. allocate buffers through `nnis-rt`
6. launch the kernel
7. synchronize correctly
8. copy result to host
9. compare against CPU oracle

Use a simple operation first, such as vector addition or SAXPY.

Do not fake this test.

---

# WAVE 2 — KERNEL LAUNCH INFRASTRUCTURE

If the current runtime lacks the necessary CUDA Driver API functions, extend `nnis-sys` and `nnis-rt` cleanly.

Implement the minimum reusable infrastructure necessary for kernels, potentially including:

* `cuModuleLoadData`
* `cuModuleUnload`
* `cuModuleGetFunction`
* `cuLaunchKernel`
* typed or validated launch parameters
* grid dimensions
* block dimensions
* dynamic shared memory
* optional stream selection

Keep unsafe code narrow.

Document every important safety invariant.

Prevent common size/pointer/lifetime mistakes where practical.

Add failure-path tests.

---

# WAVE 3 — NNIS KERNEL LIBRARY

Turn `nnis-kernels` into a real reusable component.

Do NOT attempt to implement every inference kernel.

Start with a small set of primitives that exercise the architecture end-to-end and are useful building blocks.

Candidates include:

* vector add
* elementwise scaling
* fused affine operation
* reductions
* softmax building blocks

Choose based on technical value and reusable architecture.

For each kernel:

* define a CPU/reference oracle
* test multiple sizes
* test boundary sizes
* test non-multiple-of-block-size sizes
* validate floating-point tolerance explicitly
* execute on the actual Thor GPU

Prefer runtime-specialized CUDA source through `nnis-jit` where that advances the project architecture.

Do not hard-code Thor assumptions into generic APIs unless unavoidable.

Architecture-specific optimization belongs behind dispatch/specialization boundaries.

---

# WAVE 4 — BENCHMARK INFRASTRUCTURE

Turn `nnis-bench` into a real benchmark facility.

GPU measurements must use correct synchronization or CUDA events.

Do NOT measure asynchronous kernel launches with naive host wall-clock timing and call that GPU latency.

Capture useful metadata such as:

* git commit
* GPU name
* compute capability
* CUDA/driver information where available
* dtype
* dimensions
* warmup count
* iteration count
* median or appropriate distribution summary

Provide machine-readable output if practical.

Use it immediately on at least one real NNIS kernel.

---

# WAVE 5 — HIGH-LEVEL `nnis` FACADE

Turn the top-level `nnis` crate into a useful public entry point.

It should make common workflows possible without forcing clients to import every internal crate manually.

Re-export only APIs that belong in the stable public surface.

Do not expose raw unsafe implementation details unnecessarily.

Keep layering clear:

```text
nnis
  |
  +-- nnis-kernels
  +-- nnis-jit
  +-- nnis-rt
          |
          +-- nnis-sys
```

Adjust this only if repository evidence demonstrates a better dependency direction.

Avoid dependency cycles.

---

# WAVE 6 — REAL END-TO-END DEMONSTRATION

Create at least one end-to-end example that demonstrates the complete NNIS stack.

For example:

```text
Device
  -> Context
  -> Buffer allocation
  -> JIT compilation
  -> Module
  -> Kernel
  -> Launch
  -> CUDA event timing
  -> Result validation
```

The example must execute successfully on the Thor.

---

# WAVE 7 — PERFORMANCE INVESTIGATION

Only after correctness exists, benchmark the implementation.

Investigate measurable costs such as:

* JIT compilation latency
* first-run versus cached compilation
* kernel launch latency
* allocation overhead
* transfer overhead
* event overhead
* kernel execution throughput

Then identify one or more justified optimization opportunities.

Use:

baseline → hypothesis → implementation → correctness → benchmark → decision

Retain an optimization only if evidence supports it or if it provides necessary functionality.

Record important negative results.

---

# NATIVE NVIDIA PRINCIPLE

NNIS exists to provide NVIDIA-native mechanisms.

Do not avoid native CUDA implementation simply because a generic framework would be easier.

At the same time:

Do NOT add PyTorch, TensorFlow, CUDA wrappers, wgpu, Candle, Burn, or another large runtime merely to get functionality working.

The existing architecture intentionally starts from CUDA Driver API + NVRTC.

Preserve that direction unless there is overwhelming technical evidence otherwise.

---

# RELATION TO OTHER PROJECTS

NNIS is lower-level infrastructure intended to become reusable by projects such as:

* ADA
* FLAT / FLAT-ATTENTION
* SciRust

Do NOT make NNIS dependent on those projects.

You MAY inspect them read-only if a concrete implementation question requires reference material.

In particular:

`/root/FLAT-ATTENTION`

may contain useful GPU implementation experience.

However:

ADA-A1 generic development is frozen.

Do not modify FLAT/ADA as part of this continuation unless absolutely required.

Implement reusable mechanisms in NNIS itself.

---

# CORRECTNESS RULES

Never:

* fabricate tests
* fabricate benchmark numbers
* weaken tolerances merely to pass tests
* comment out failing tests
* convert GPU tests into mocks
* silently skip GPU execution when the GPU is available
* claim CUDA code ran when it only compiled

For GPU-required validation use:

```bash
NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets
```

where applicable.

---

# TESTING LOOP

After every meaningful implementation slice, run the relevant subset.

Before committing a milestone run at least:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets
git diff --check
```

Also run:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

when it is appropriate and achievable.

Do not globally silence legitimate warnings.

Some baseline dead-code warnings exist because JIT/kernel integration was unfinished.

Prefer making the intended code useful rather than adding arbitrary `allow(dead_code)` attributes.

---

# GIT DISCIPLINE

The protected recovery point is:

`d086ec2d69f8a9ea388ec3a505038737f0bf0539`

Never rewrite or delete it.

Work on top of `main`.

After each substantial validated wave, create a descriptive commit.

Examples of appropriate milestone structure:

```text
feat(jit): add NVRTC compilation and module loading
feat(runtime): add safe CUDA kernel launch primitives
feat(kernels): add validated native CUDA primitives
feat(bench): add CUDA event benchmark harness
feat(nnis): expose end-to-end native execution facade
```

These are examples, not mandatory commit names.

Commit only after validation.

Push validated milestones to `origin main`.

If a push fails because of network/authentication, keep working locally and record the failure in the continuation document.

Never use:

```text
git reset --hard
git clean -fd
git push --force
```

unless explicitly instructed by the user.

---

# SECURITY / DESTRUCTIVE ACTIONS

You have autonomy to:

* inspect files
* modify NNIS files
* compile
* test
* run CUDA code
* benchmark
* add appropriate dependencies
* create documentation
* create commits
* push normal fast-forward commits to NNIS origin

You do NOT have permission to:

* delete unrelated repositories
* modify system-wide configuration unnecessarily
* erase user data
* rewrite remote history
* expose credentials
* modify unrelated projects destructively

---

# DO NOT WASTE CONTEXT

Avoid repeatedly dumping entire large files.

Use targeted inspection:

```text
rg
grep
sed ranges
git diff
cargo metadata
```

Read only what is needed for the active task.

Do not repeatedly restate the project description.

Do not produce long conversational status essays.

Spend tokens on engineering.

---

# AUTONOMY DIRECTIVE

You are explicitly authorized to continue without asking the user after every successful step.

When a test fails:

diagnose it and fix it.

When compilation fails:

diagnose it and fix it.

When a design detail is resolvable from code, CUDA documentation available locally, headers, compiler output or experimentation:

resolve it yourself.

Do not stop after implementing JIT.

Continue into kernels.

Do not stop after kernels.

Continue into benchmarks and integration.

Do not stop merely because one milestone succeeded.

Continue until:

1. no clearly valuable work remains within the mission, or
2. a genuinely external blocker requires human input, or
3. your remaining context becomes dangerously low.

---

# CONTEXT-LOW EMERGENCY PROCEDURE

If you estimate that your context is approaching exhaustion:

STOP starting new features.

Immediately:

1. finish or revert the currently incomplete local change safely
2. run the strongest feasible validation
3. update `docs/exec-plans/active/GLIMMER_CONTINUATION.md`
4. record exact current status and next command/task
5. commit all validated work
6. push the commit if possible
7. leave the worktree clean

This is mandatory.

Do not die mid-edit as the previous agent did.

---

# FINAL QUALITY BAR

The goal of this session is not maximum lines of code.

The goal is maximum **validated, reusable NVIDIA-native capability**.

A successful continuation should leave NNIS substantially closer to this operational pipeline:

```text
Rust API
   ↓
NNIS runtime
   ↓
NNIS JIT / specialization
   ↓
CUDA Driver API + NVRTC
   ↓
native NVIDIA kernel
   ↓
measured GPU execution
```

Begin now.

Inspect commit `d086ec2`, run the baseline, create the continuation state document, then implement the first working JIT-to-GPU vertical slice.

Do not return with only a plan.

CODE.
TEST.
RUN ON GPU.
COMMIT.
PUSH.
CONTINUE.
