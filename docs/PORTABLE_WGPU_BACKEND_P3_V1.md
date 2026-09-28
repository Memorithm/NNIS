# NNIS-P3 portable WGPU backend, first slice

`crates/nnis-wgpu` is a new, separate crate. It is the only NNIS crate that
depends on `wgpu`. `nnis-core` and `nnis-cpu` stay dependency-free, and
`scripts/check-portable-core.sh` / `check-portable-cpu.sh` now reject `wgpu`
and `nnis-wgpu` in their dependency trees.

## wgpu version

`wgpu = "23.0.1"` (caret requirement, so versions >=23.0.1 and <24 are
allowed; `Cargo.lock` pins 23.0.1, naga 23.1.0, wgpu-core/hal 23.0.1).

wgpu 23 is the newest release whose declared MSRV is at or below the
workspace MSRV. Its README declares 1.76 for the whole workspace. wgpu 24
declares 1.83 for `wgpu`, and 25 and later declare 1.84 to 1.92 in
`rust-version`. `cargo +1.77.0 check --workspace --all-targets --locked`
builds cleanly. The new lock entries were resolved with
`CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback` and only add packages.
No existing locked version changed.

## Scope

- `WgpuDevice::discover()`: requests the default high-performance adapter on
  the primary native backends (`WGPU_BACKEND` overrides this) and creates a
  device with the adapter's full limits plus `SHADER_F16` when available. It
  returns `Ok(None)` when there is no adapter.
- `WgpuAdapterReportV1` / `classify_adapter`: an adapter is classed as
  `Software` if its device type is CPU or its name or driver matches
  llvmpipe, lavapipe, SwiftShader, softpipe or WARP. Software adapters are
  never hardware evidence.
- `capabilities_from_limits`: maps the granted limits and features to the
  NNIS `CapabilitySet` consumed by `KernelArtifactV1::bind`:
  - `max_buffer_bytes` = min(`max_buffer_size`, `max_storage_buffer_binding_size`)
  - `max_workgroup_invocations` / `max_workgroup_size` = the compute limits
  - `max_bindings` = min(per-stage storage + uniform buffers,
    bindings per group × bind groups)
  - `supports_f16` = `SHADER_F16`, `supports_timestamps` = `TIMESTAMP_QUERY`
    (not requested, so false)

  The result goes through `CapabilitySet::validate`, so limit tables with no
  compute support (WebGL2 downlevel) are rejected.
- `check_wgpu_binding_limits`: checks per-binding WGPU limits that
  `CapabilitySet` does not express: bind-group index, binding index,
  storage and uniform counts per stage, and uniform binding size.
  `WgpuDevice::bind` runs the P4 bind first, then this check.
- The WGSL kernel `nnis.wgpu.add_f32` (F32 elementwise add, workgroup 64) is a
  P4 `KernelArtifactV1`. `WgpuDevice::add_f32` builds and binds the artifact
  before creating any GPU object. It rejects unequal lengths, empty inputs,
  non-finite inputs and dispatches over the limit, and captures validation
  and out-of-memory errors.

## Numerical tolerance

Policy `wgsl-f32-add-correctly-rounded-normal-range-v1`: WGSL requires
correctly rounded binary32 addition but allows subnormals to be flushed. The
declared tolerance is **bit-exact** equality with the `nnis-cpu`
`CpuF32KernelsV1` add, for finite inputs whose operands and exact sums are
normal or zero. The test covers signed zeros, ties to even, `2^24 + 1`,
cancellation of large values, and a partial last workgroup (1000 elements).

## CI and adapter policy

CI has no GPU. `nnis-wgpu` is a normal workspace member, so the existing
`cargo check/clippy/test --workspace --all-targets` steps and the
`Rust 1.77 check` job build and test it. No new CI job was needed.
`tests/adapter_add_f32.rs` calls `discover()`. If there is no adapter, it
writes `SKIP nnis-wgpu adapter_add_f32: no WGPU adapter available; ...`
straight to the stderr handle, which libtest output capture does not
intercept, so the line appears in the CI test log. If the runner exposes a
software adapter, the test does run, and it logs the adapter class with
"(not hardware evidence)". The limit-mapping and binding tests
(`tests/capabilities.rs`) need no adapter.

`scripts/check-portable-wgpu.sh` checks the dependency boundary: no
vendor-specific NNIS crates and no CUDA/NVML bindings. It then runs the
crate checks, tests and rustdoc. It is not wired into
`.github/workflows/ci.yml` yet, because that change needs a push with
workflow scope. The proposed step is:

```yaml
      - name: Verify portable WGPU dependency boundary (no GPU; adapter test logs SKIP)
        if: always()
        run: bash scripts/check-portable-wgpu.sh
```

## Portable buffer, queue and fence contracts (P3 slice 2)

`WgpuDevice` implements `PortableDevice`, with `WgpuBuffer`, `WgpuQueue` and
`WgpuFence` implementing `PortableBuffer`, `PortableQueue` and
`PortableFence`. `WgpuQueue::copy_buffer` mirrors `CpuQueue::copy_buffer`.
Semantics follow `nnis-cpu`:

- Descriptors are revalidated at allocation, so literals that skip
  `BufferDesc::new` are still checked. Sizes above
  `CapabilitySet::max_buffer_bytes` are rejected as `Unsupported`.
- Buffers are zero-initialized.
- Usage bits are enforced at the portable API. Writes need `COPY_DST`.
  Reads and copy sources need `COPY_SRC`. Copy destinations need
  `COPY_DST`.
- Every range is checked against the logical size before any mutation.
  Empty ranges are allowed at the end of a buffer and rejected beyond it.
  A failed call leaves the payload unchanged.
- Arbitrary byte offsets and sizes work. Allocations are padded to WGPU's
  4-byte copy alignment. Unaligned writes read, modify and write back the
  aligned span. Unaligned copies are staged through the host. The padding is
  never observable through the API.
- `write_buffer` stages the caller's bytes before returning (WGPU copies
  the slice) and submits. The fence completes when the queue reports that the
  submitted work is done. `read_buffer` blocks until all previously
  submitted work has finished.
- `MemoryClass::Host` is rejected. `Shared` and `DeviceLocal` are accepted
  as declared intent only: WGPU does not report physical placement, so no
  residency is claimed.
- A buffer is tied to the device that created it. Using it with another
  device's queue fails closed.

Concurrency notes from local runs on Mesa llvmpipe (software; these are
code-path findings, not evidence):

- Creating and tearing down several devices concurrently crashed the
  process (SIGSEGV). `WgpuDevice::discover` is therefore serialized
  process-wide. The tests in each binary share devices that are never
  dropped.
- A work-done callback can run on another thread's poll, after this thread's
  blocking wait has returned. `WgpuFence::wait` therefore records
  completion once `Maintain::wait_for(index)` returns, and does not depend
  on the callback. The fixed version passed 140 repeated runs under Rust
  1.77.0 and stable.

`tests/portable_memory.rs` runs the same scripted sequence through the
generic traits on `CpuDevice` and on a WGPU adapter and requires identical
results. Error text is normalized, but error variants and out-of-bounds
fields must match. The test also covers WGPU-specific admission rules,
fences, and aligned and unaligned `copy_buffer` against `CpuQueue`. Without
an adapter every test logs an explicit SKIP.

## WGSL F32 reference kernels (P3 slice 3)

`nnis_wgpu::numerical::WgpuF32KernelsV1` mirrors `nnis_cpu::numerical::CpuF32KernelsV1`
for `binary` (add, multiply), `relu`, `sum`, `project_kn`, `gather` and
`scatter_add`. Each operation, and the finiteness validator it uses, is a P4
`KernelArtifactV1`. The artifact is bound with `WgpuDevice::bind`
(fingerprint, backend family, capabilities, WGPU binding limits) before any
pipeline is created. The report records the artifact fingerprint.

Semantics follow the CPU reference:

- Every buffer needs `STORAGE` and a non-zero multiple of four bytes.
- Every input value, including unselected gather inputs and the whole old
  scatter destination, must be finite. So must every arithmetic
  intermediate. Finiteness is tested on raw bits, so it cannot be optimized
  away.
- Output goes to a scratch buffer. It is copied to the destination only when
  no non-finite value was seen, so any error leaves the destination
  unchanged. This includes a scatter-add that fails partway.
- Sum and scatter-add run serially in the CPU order, in a single
  invocation. They are reference paths, not performance paths.
- When several errors apply at once, the reported variant can differ from
  the CPU reference. Structural checks run on the host first; finiteness
  checks run on the device.

| operation | policy | declared tolerance |
|---|---|---|
| ReLU, gather | `wgsl-f32-finite-bitwise-exact-v1` | bit-exact, subnormals included (bit manipulation only) |
| add, multiply, sum, scatter-add | `wgsl-f32-finite-serial-correctly-rounded-normal-range-v1` | bit-exact for normal-range operands and results (WGSL may flush subnormals) |
| project_kn | `wgsl-f32-finite-fma-inherited-bound-3k-plus-1-ulp-v1` | `|wgpu - cpu| <= (3k + 1) * 2^-24 * sum_r |x_r * w_rj|`; sign of zero not compared |

The projection bound exists because WGSL `fma` accuracy is "inherited from
`a * b + c`", so a device may round twice where the CPU reference uses a
fused `mul_add`. Locally on llvmpipe (software, code path only), 51 of 70
outputs were bit-identical to the fused CPU result, and all 70 were within
the bound.

`tests/numerical_parity.rs` covers:

- bit-exact add and multiply over 1000 elements, including signed zeros and
  ties;
- ReLU and gather, including subnormals and repeated indices;
- serial sum, including `-0` inputs, cancellation, and large-magnitude
  terms that cancel;
- serial scatter-add with repeated indices;
- the projection bound;
- error variants matching the CPU reference, with the destination checked
  unchanged after each error.

Without an adapter every test logs an explicit SKIP.

## Portable F32 graph execution (P3 slice 4)

`nnis_wgpu::graph::execute_f32_graph` runs a `ValidatedF32GraphV1` through
`WgpuF32KernelsV1`, mirroring `nnis_cpu::graph::execute_f32_graph`:

- Before any node buffer is allocated, it checks the schema and policy, the
  binding count, per-buffer device limits, input `STORAGE` usage and byte
  lengths, and the finiteness of every input, including unused ones.
- Caller buffers are never mutated. Scatter-add writes into a fresh copy of
  its base.
- Intermediates stay private and are dropped on any error. Only the final
  value escapes.

The report records the bound artifact fingerprint of every node.
`execute_f32_graph_traced` returns every node output.

Declared tolerance:

- Each node meets the policy of the kernel it runs.
- A graph with **no `ProjectKn` node** is **bit-exact** end-to-end with the
  CPU graph reference for normal-range values.
- For graphs **with projections**, the contract is per node. Every
  non-projection node must be bit-exact with the CPU kernel applied to the
  same WGPU node inputs. Every projection node must be within the declared
  projection bound for those inputs. End-to-end equality with the CPU graph
  is not claimed, because projection differences propagate into later nodes.

`tests/graph_parity.rs` covers:

- projection-free graphs, including a branched graph and a six-node
  300-element graph using add, multiply, ReLU, gather, scatter-add and sum,
  checked bit-exact end-to-end;
- a seven-node two-projection graph checked node by node through the trace;
- errors (late overflow, non-finite used and unused inputs, binding count,
  byte length, usage) matching the CPU error variants, with inputs checked
  unchanged.

Without an adapter every test logs an explicit SKIP.

## Claim boundary

This slice adds WGPU compilation plus adapter discovery, limit mapping and a
single-kernel parity check. It contains no performance claim and no hardware
parity claim. A pass without an adapter, or on a software adapter, is not
WGPU hardware evidence. `PortableDevice`/`PortableQueue` for WGPU, graph
execution, and DSV41 WGPU counterparts are still open.
