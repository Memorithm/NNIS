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

## Claim boundary

This slice adds WGPU compilation plus adapter discovery, limit mapping and a
single-kernel parity check. It contains no performance claim and no hardware
parity claim. A pass without an adapter, or on a software adapter, is not
WGPU hardware evidence. `PortableDevice`/`PortableQueue` for WGPU, graph
execution, and DSV41 WGPU counterparts are still open.
