# Portable CPU memory v1

Scope: NNIS-P2a memory ownership and synchronous data movement. This is not a
numerical-kernel, model-inference, WGPU or adaptive-policy qualification.

## Public surface

`nnis-cpu` depends only on `nnis-core`. The production code forbids unsafe Rust.
`CpuDevice` implements `PortableDevice`; `CpuQueue` implements `PortableQueue`;
`CpuFence` reports completion after successful synchronous operations.

`CpuDevice::with_max_buffer_bytes(limit)` sets a positive per-buffer admission
ceiling bounded by the host `isize::MAX`. `CpuDevice::new()` uses that
representability ceiling, not an available-RAM estimate. A successful allocation
is zero-initialized. Host and Shared are backed by host memory; DeviceLocal is
explicitly unsupported rather than relabelled.

Public `BufferDesc` values are revalidated immediately before allocation even
when a caller constructs them as literals or mutates them after construction.
Zero-size descriptors, empty usage sets and sizes exceeding the configured
ceiling are rejected before payload allocation.

Creation and readback reserve fallibly with `Vec::try_reserve_exact` before
initializing or copying bytes. Capacity/allocation errors returned by the Rust
allocator become `PortableError::Backend`. This does not guarantee survival of
OS overcommit termination or make the derived `Clone` allocation fallible.

## Checked data movement

Writes require COPY_DST; reads require COPY_SRC. `CpuQueue::copy_buffer` requires
both the source COPY_SRC and destination COPY_DST permissions. Every source and
destination range is validated with checked arithmetic before changing bytes.

An invalid range or permission leaves the destination unchanged. Buffer-to-buffer
copy does not allocate a temporary payload and does not resize the destination.
An empty copy at a valid endpoint is allowed; an empty copy beyond the endpoint
is rejected. Rust borrows prevent using the same owned buffer as both operands.
This is a synchronous single-operation guarantee, not crash durability or a
multi-operation transaction protocol.

## Accounting boundary

`CpuBuffer::len()` reports initialized payload bytes.
`CpuBuffer::capacity_bytes()` reports retained Vec capacity, which can exceed the
payload length. Neither includes allocator bookkeeping nor measures process RSS,
physical residency, available RAM, or a global live-allocation budget.

The configured maximum limits one requested buffer size. Multiple buffers,
readback copies and independently allocated clones require separate caller-side
accounting. No total-budget, memory-savings or performance claim is implied.

## Executable qualification

```bash
cargo test --locked -p nnis-core -p nnis-cpu --all-targets
cargo run --locked -p nnis-cpu --example portable_buffers
bash scripts/check-portable-cpu.sh
```

The example uses three small real host buffers: it writes a current state, copies
an explicit checkpoint, applies candidate bytes, rejects an out-of-bounds copy
without mutation, and restores the checkpoint exactly. It emits
`CPU_BUFFER_SMOKE_OK` only after these checks and the per-buffer admission test
succeed. It has no CUDA or biological dataset prerequisite and reports no timing.

The permanent dependency guard covers normal, build, test and example
dependencies. CPU tests cover constructor bypass, invalid limits, deterministic
capacity overflow, usage restrictions, range overflow, unchanged destinations,
copy correctness, capacity preservation and immediate fences.

## Ecosystem and next step

This provides a portable memory foundation for future ElasticXxx admission and
rollback adapters, SML page operations and FLAT buffer transport. It does not
itself implement any of those cross-repository integrations. Numerical kernels,
complete graph execution and the WGPU backend remain separate phases.

Historical CUDA INT4 measurements/contracts remain legacy evidence. This CPU
memory qualification neither runs the frozen ElasticBitAllocation Stage-B
protocol nor changes its checkpoint, backend, acceptance thresholds, final-test
lock or allocator/search NO-GO gates. A portable experiment requires its own
versioned, reviewed protocol before collecting or comparing model evidence.

No registry publication is authorized by this slice.
