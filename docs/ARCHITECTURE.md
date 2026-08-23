# NNIS architecture and safety

This document describes the ownership and execution contracts behind NNIS's
public API. It is intentionally explicit about asynchronous CUDA work: a Rust
borrow ending does not make queued device access stop.

## Layering

```text
                         nnis-bench
                             |
nnis facade -> nnis-kernels  |
      |             |        |
      +---------- nnis-jit ---+
                    |
                 nnis-rt
                    |
                 nnis-sys
                    |
          CUDA Driver API + NVRTC
```

- `nnis-sys` transcribes the required CUDA 13.0 Driver API and NVRTC ABI and
  dynamically resolves it. Its function pointers retain their owning dynamic
  libraries for the process lifetime.
- `nnis-rt` owns devices, primary-context references, streams, events, device
  allocations, and pinned host allocations. Operations establish their
  context as current before native calls.
- `nnis-jit` compiles source into PTX or architecture-specific CUBIN, caches
  immutable results, owns loaded modules, retains modules through function
  handles, and validates launch shape/context relationships.
- `nnis-kernels` packages tested kernel source behind operation-level APIs.
  Its explicit block-size constructor and operation-level occupancy reports
  form a tuning boundary without exposing the underlying CUDA functions.
- `nnis-bench` is a separate measurement layer so the application facade does
  not acquire serialization or benchmarking dependencies.
- `nnis` re-exports the intended application surface without exposing raw
  `nnis-sys` handles.

Dependencies point down this list. Downstream projects are consumers of NNIS,
not dependencies of it.

## Context and native ownership

`Context::new` retains the selected device's CUDA primary context. Streams,
events, buffers, modules, and kernels retain or refer to that context. NNIS
checks context identity before copies, event waits, and launches so objects
from unrelated contexts cannot be accidentally combined.

The key ownership chains are:

```text
process DriverApi/NvrtcApi -> loaded native library
JitCompiler cache          -> Arc<CompiledCode>
Module                     -> Arc<ModuleInner> -> CUDA module
Kernel                     -> Arc<ModuleInner> -> CUDA module
DeviceBuffer / Stream      -> Arc<Context>     -> primary context reference
```

A `Kernel` therefore keeps its defining module loaded even after the original
`Module` value is dropped. A cached native function pointer cannot outlive the
dynamic library from which it was resolved.

## JIT compilation and launch

The custom-kernel path is:

1. `CompileOptions::for_device` converts `DeviceProps` into an NVRTC target.
2. `JitCompiler::compile_cubin` or `compile_ptx` hashes the source, target,
   ordered options, and output kind into a deterministic cache key.
3. A cache miss creates and compiles an NVRTC program. Compiler diagnostics
   are preserved on both success and failure.
4. `Module::load` gives the code to `cuModuleLoadDataEx` and owns the result.
5. `Module::get_function` resolves a named function and returns a module-owning
   `Kernel`.
6. `KernelArgs` copies primitive argument values into aligned host storage and
   retains the contexts of borrowed buffers.
7. `Kernel::attributes` and `Kernel::recommend_occupancy` expose typed,
   context-correct Driver API introspection without leaking raw handles.
8. `KernelLaunch` validates nonzero grid/block axes, function/device block and
   shared-memory limits, and context relationships before calling
   `cuLaunchKernel`.

The final launch remains `unsafe` because neither CUDA nor Rust reflection can
prove that the selected function's parameter order and widths match the values
in `KernelArgs`. The caller must also keep the kernel, stream, argument pack,
and referenced allocations alive until the stream completes.

CUDA expects its launch parameter array to contain host pointers to storage
holding each argument value. NNIS's packer provides exactly that representation;
a device address is stored as a `u64` value inside aligned host storage rather
than being misinterpreted as the host pointer to the parameter.

## Kernel families

Standard kernels live in `nnis-kernels` behind operation-level APIs. Each
family compiles its CUDA source through the JIT cache on first use, validates
launch shapes before any native call, and offers a safe synchronizing method
plus an explicitly unsafe enqueue counterpart.

### Reductions (`F32Reduction`)

Sum and max share one tree structure: each block halves `2 * block_size`
input elements into shared-memory partials through stride-halving passes,
writing one partial per block; multi-pass invocations reduce those partials
recursively until a single device scalar remains. Consequences:

- A `F32ReductionWorkspace` holds two scratch buffers sized by the largest
  input and is reusable for any input up to that capacity, but never by
  overlapping asynchronous operations.
- The sum tree defines a deterministic evaluation order. The CPU oracle
  replays that order, so GPU sums match it bit for bit; agreement with an
  f64 reference is additionally checked inside an explicit forward
  error bound rather than a silently loosened tolerance.
- `fmaxf` is exact, so max results are asserted bit-for-bit against the CPU.
- An empty sum input schedules a zeroed output; an empty max input enqueues
  nothing and leaves the destination untouched.

### Flat softmax (`F32Softmax`)

A numerically stable four-stage pipeline over one flat buffer: device-side
max, `exp(input - max)`, device-side sum of exponentials, in-place normalize.
The maximum and total stay in one-element device buffers between stages, so
the whole pipeline enqueues without host roundtrips and safe wrappers
synchronize exactly once. This is the pattern for multi-stage NNIS pipelines:
stage boundaries are stream ordering, not host synchronization.

### Row-batched softmax (`F32Softmax2D`)

For row-major `rows x cols` matrices, two execution paths exist behind one
operation family:

- Staged: one thread block per row performs strided max/sum reductions into
  a device-resident per-row scalar column (`F32Softmax2DWorkspace`); exp
  shift and in-place normalize index that column per element. Six full-matrix
  streams move through global memory.
- Fused: when `(cols + block_size) * 4` bytes fit the kernel's dynamic
  shared-memory limit, one block stages its entire row in shared memory and
  computes max, exponentials, total, and normalized output with one matrix
  read and one matrix write.

A clean Thor A/B at 8192x2048 measured the fused path 2.46x faster than the
staged pipeline; the staged path remains required for rows exceeding the
shared-memory budget. `softmax_rows_dispatched` chooses automatically via
`fused_available`, keeping the architecture-specific optimization behind a
dispatch boundary instead of hard-coding Thor assumptions.

## Blocking and asynchronous APIs

The default memory and standard-kernel methods are safe and synchronizing:

- `DeviceBuffer::{zero,copy_from_host,copy_to_host,copy_from_buffer}`
- `DeviceBuffer::{from_host,to_vec}`
- `F32Elementwise::{vector_add,scale,affine}`

They enqueue work and wait for the stream before returning. This keeps every
Rust borrow live for the entire interval in which CUDA may access it.

The `_async` memory methods and `enqueue_*` kernel methods return immediately
and are `unsafe`. For every such call, all referenced host memory, device
buffers, kernels, and streams must remain alive and otherwise untouched until
completion is established by `Stream::synchronize`, a synchronized event, or
another valid CUDA dependency. In particular:

- do not drop or reallocate a source/destination while DMA may use it;
- do not read a D2H destination before completion;
- do not mutate the same allocation concurrently without an ordered CUDA
  dependency;
- do not drop a buffer that an enqueued kernel may still access.

The current types do not encode an in-flight operation in their ownership
state. Keeping that boundary explicitly `unsafe` prevents a safe API from
promising more than Rust's lifetimes can enforce.

## Byte-copyable host types

Host/device transfers require `DevicePod`. Its implementations must be
`Copy`, contain no uninitialized padding, and accept every possible bit pattern
as a valid value. NNIS implements it for integer and IEEE floating-point
primitives and arrays of `DevicePod` elements.

Implementing `DevicePod` for an application type is `unsafe`. A type containing
references, invalid enum discriminants, padding that may be uninitialized, or
other restricted representations does not satisfy the contract.

Device allocations themselves are initially uninitialized unless constructed
with `new_zeroed`, copied from a host slice, or written by a kernel. Reading an
uninitialized allocation back and interpreting it as meaningful data remains a
logic error even when the element type is `DevicePod`.

## Event timing

CUDA work is asynchronous with respect to the host. `nnis-bench::benchmark_gpu`
therefore performs warmups, records start and end events on the same stream as
the operation, synchronizes the end event, and uses `cuEventElapsedTime` for
each sample. Reports include the raw sample distribution and execution
metadata.

The closure passed to `benchmark_gpu` must enqueue the complete operation
without synchronizing it. Standard safe kernels are intentionally unsuitable
inside that closure because their synchronization would distort the event
interval; benchmark code uses the corresponding unsafe enqueue method and
keeps every captured object alive through the harness.

Occupancy recommendations describe active-warp resource limits, not observed
latency or bandwidth. Dispatch changes require clean correctness-preserving
benchmarks; the Thor elementwise sweep retained 256 threads after the driver's
768-thread recommendation measured slower in both candidate orders.

## Failure behavior

CUDA and NVRTC return codes become structured `NnisError` values with operation
context. NVRTC compilation failures include the compiler log and target
architecture. Invalid dimensions, length mismatches, integer overflows, and
cross-context combinations are rejected before the native operation where
practical.

Destructors cannot report errors. They make the owning context current and
best-effort release the native resource. Applications should explicitly
synchronize work before teardown so asynchronous failures are observed at an
operation boundary rather than during cleanup.
