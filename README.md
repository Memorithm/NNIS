# NVIDIA Native Inference Stack (NNIS)

NNIS is an early-stage Rust substrate for executing native NVIDIA GPU kernels
without a framework runtime. It builds directly on the CUDA Driver API and
NVRTC:

```text
Rust API
   -> NNIS runtime and kernel library
   -> runtime CUDA compilation and specialization
   -> CUDA Driver API + NVRTC
   -> native NVIDIA kernel execution
```

The current implementation provides dynamically loaded CUDA/NVRTC bindings,
primary-context ownership, streams and events, device and pinned memory,
runtime PTX/CUBIN compilation, owned modules and functions, validated launch
configuration, reusable `f32` elementwise kernels, and CUDA-event benchmarks.
It has been exercised end to end on an NVIDIA Thor GPU; the tests perform real
allocations, copies, compilation, launches, synchronization, and result
validation.

NNIS is not yet a complete model runtime. The public API is version `0.1`, the
standard kernel library is intentionally small, and compatibility beyond the
hardware/software configurations exercised by the tests is not yet guaranteed.

## Requirements

- Linux with a working NVIDIA driver exposing `libcuda.so.1`
- an NVRTC library from an installed CUDA toolkit
- Rust 1.77 or newer

NNIS resolves the native libraries at runtime, so applications do not link to
the CUDA libraries at build time. If they are outside the dynamic loader's
search path, set `NNIS_CUDA_DRIVER_PATH` and/or `NNIS_NVRTC_PATH` to their full
paths.

The validated development configuration is Linux `aarch64`, NVIDIA Thor,
driver 580.00, CUDA 13.0, and NVRTC 13.0. Compilation options are derived from
the selected device's compute capability rather than hard-coded for Thor.

## Quick start

The `nnis` facade is the normal application entry point. A `Session` owns one
context-bound stream, a process-local JIT cache, and the standard kernels:

```rust
use nnis::prelude::*;

fn main() -> Result<()> {
    let session = Session::first()?;
    let host = vec![1.0_f32, 2.0, 3.0, 4.0];
    let input = DeviceBuffer::from_host(session.context(), session.stream(), &host)?;
    let output = DeviceBuffer::<f32>::new(session.context(), host.len())?;

    session
        .elementwise()
        .affine(session.stream(), &input, &output, 2.0, -1.0)?;

    let actual = output.to_vec(session.stream())?;
    assert_eq!(actual, vec![1.0, 3.0, 5.0, 7.0]);
    Ok(())
}
```

The safe `affine`, `scale`, and `vector_add` methods wait for their stream
before returning. Their explicitly unsafe `enqueue_*` counterparts are
available when an application needs asynchronous overlap and can uphold the
lifetime contract described in [Architecture and safety](docs/ARCHITECTURE.md).

To compile and launch custom CUDA source through the complete stack, run:

```bash
cargo run --release -p nnis --example end_to_end
```

The example derives its NVRTC architecture from the active device, compiles a
CUBIN, verifies a cache hit, loads a module, resolves and launches a function,
measures it with CUDA events, and checks every result against a CPU oracle.

Inspect a compiled kernel's registers, local/static/dynamic memory limits,
code-generation versions, and CUDA occupancy recommendation with:

```bash
cargo run --release -p nnis-jit --example inspect_kernel
```

## Workspace

| Crate | Responsibility |
| --- | --- |
| `nnis` | Stable facade, common re-exports, and `Session` |
| `nnis-kernels` | Reusable native kernel families and CPU-oracle tests |
| `nnis-jit` | NVRTC compilation/cache, modules, functions, and launches |
| `nnis-rt` | Devices, contexts, streams, events, and memory ownership |
| `nnis-sys` | Narrow, dynamically loaded CUDA Driver/NVRTC FFI |
| `nnis-bench` | CUDA-event timing and machine-readable benchmark reports |

See [Architecture and safety](docs/ARCHITECTURE.md) for the dependency and
ownership model.

## Validation

Run the complete test suite with GPU availability mandatory:

```bash
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
NNIS_REQUIRE_GPU=1 cargo test --workspace --all-targets
git diff --check
```

Without `NNIS_REQUIRE_GPU=1`, GPU-dependent tests may report that they were
skipped when no CUDA device is available. The required form is therefore the
one used to establish GPU functionality.

## Benchmarks

NNIS records start/end CUDA events on the operation stream and synchronizes
the end event before calculating device time. It does not report asynchronous
host enqueue time as kernel latency.

Run the standard elementwise benchmark:

```bash
NNIS_BENCH_ELEMENTS=16777216 \
NNIS_BENCH_WARMUPS=20 \
NNIS_BENCH_ITERATIONS=100 \
cargo run --release -p nnis-bench --example elementwise
```

The command emits JSON containing every sample, distribution statistics,
throughput, git state, GPU identity, compute capability, driver/NVRTC
versions, dimensions, dtype, and iteration counts.

Run the broader native-cost investigation with:

```bash
cargo run --release -p nnis-bench --example performance_breakdown
```

For reference, a clean Thor run at commit `85420bd` measured the 16,777,216
element `f32` scale kernel at 0.623600 ms median and 215.230 decimal GB/s over
100 iterations. A later clean run at `6dd485f` measured 0.615392 ms and 218.101
GB/s. These are observed results on that machine, not portable performance
claims. Full measurements and methodology are recorded in the active
[continuation log](docs/exec-plans/active/GLIMMER_CONTINUATION.md).

## Current scope

Implemented standard kernels are `f32` vector addition, scaling, and fused
affine transformation. Near-term extension points include launch
introspection/dispatch, reductions and softmax building blocks, and safe owned
abstractions for longer asynchronous pipelines. NNIS deliberately does not
depend on PyTorch, TensorFlow, Candle, Burn, wgpu, or downstream projects.
