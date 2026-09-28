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

## Claim boundary

This slice adds WGPU compilation plus adapter discovery, limit mapping and a
single-kernel parity check. It contains no performance claim and no hardware
parity claim. A pass without an adapter, or on a software adapter, is not
WGPU hardware evidence. `PortableDevice`/`PortableQueue` for WGPU, graph
execution, and DSV41 WGPU counterparts are still open.
