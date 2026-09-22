# Portable CPU numerical reference v1

Scope: NNIS-P2b finite F32 reference operations and one fixed tiny graph.
This is not a general graph scheduler, trained-model evaluation, WGPU backend,
ElasticXxx actuator, INT4 runtime or performance promotion.

## Ownership and reuse review

SciRust remains the owner of general numerical/tensor algorithms. Before this
slice, the following existing surfaces were inspected at immutable SciRust
revision `571154e122c53aa1d8fcbb01d3654526b9ffd459`:

- `scirust-core/Cargo.toml`: edition 2024, Rust 1.89, broader numerical stack;
- `scirust-retrieval/Cargo.toml`: Rust 1.89, retrieval-specific ownership;
- `scirust-core/src/nn/latent_kv_kernels.rs`: dispatched contiguous dot entry;
- `scirust-retrieval/src/vector.rs`: ordered dot entry with debug-only length check.

Neither inspected crate is a drop-in Rust 1.77 dependency. NNIS does not silently
raise its MSRV, copy SciRust's optimized kernels or claim that no reusable SciRust
primitive exists anywhere. A compatible shared numerical contract remains a
separate extraction/reuse decision. This small scalar executor owns only the
NNIS byte-layout, numerical-order and failure contract needed as a CPU reference.
No new crate dependency is added. The previously qualified CPU memory buffers,
usage validation and fallible reservation helper are reused directly.

## Numerical contract

Version: `CPU_F32_CONTRACT_VERSION = 1`.
Identity: `finite-f32-le-ordered-fma-v1`.

All input/output buffers must have STORAGE usage and a positive payload length
that is a multiple of four. Each element is little-endian IEEE binary32; decoding
uses byte conversion, not pointer casts or alignment assumptions.

Operations:

- Binary Add/Multiply: equal-length vectors, no broadcasting.
- ReLU: positive values retained, all negative values and both zeros become +0.
- Sum: increasing-index F32 addition starting at +0.
- ProjectKn: `[1,K] x [K,N] -> [1,N]`; row-major K,N weights, positive dimensions,
  no implicit transpose or bias, increasing-K `f32::mul_add` starting at +0.
- Gather: caller-owned index list, preserving order and duplicates.
- ScatterAdd: sequential additions to existing output in index-list order;
  repeated indices accumulate deterministically, unselected values are retained.

Every input is finite, including unselected gather elements and existing scatter
outputs. Every sum/FMA/scatter intermediate must remain finite. Later
cancellation does not rescue an overflowing intermediate. Underflow, subnormals
and signed zeros otherwise follow the documented Rust F32 arithmetic; ReLU's
zero canonicalization is explicit. No epsilon, relaxed tolerance, fast-math,
parallel reduction or lower-precision substitution is introduced.

Rust reference semantics:
- https://doc.rust-lang.org/std/primitive.f32.html
- https://doc.rust-lang.org/std/primitive.f32.html#method.mul_add
- https://doc.rust-lang.org/std/vec/struct.Vec.html#method.try_reserve_exact

Cross-hardware parity is not inferred from this policy identifier. Each backend
must be independently tested, including subnormal and fused-rounding behavior.

## Failure and temporary memory

`CpuF32KernelsV1::new(max_scratch_bytes)` requires a positive representable
ceiling for the requested output-staging payload. Oversized outputs are rejected
before allocating scratch. One fallibly reserved byte Vec stages the output;
scatter stages the prior destination before adding. No full input or matrix
copy is allocated by these operations.

Only a complete successful result is copied into the destination. Any returned
error leaves its bytes and retained capacity unchanged. The original output may
contain arbitrary bits when an operation fully overwrites it; ScatterAdd requires
its previous values to be finite. Source/output aliasing is excluded by safe
Rust borrows. This is per-operation in-process failure atomicity, not a durable
or multi-operation transaction.

A successful `CpuF32ReportV1` records operation identity, output count, scratch
payload and actual retained Vec capacity. Scratch has been released on return.
The ceiling bounds requested payload, not allocator-rounded capacity, allocator
metadata, stack use, caller buffers, total RSS or physical residency. OS
termination/overcommit is outside the Result guarantee. Admission scanning time
and model performance are not claimed to be optimized.

## Fixed graph

`portable_f32_graph` executes:

`project -> bias add -> ReLU -> project -> bias add -> gather -> sum`.

The fixture uses explicitly chosen dyadic weights, not a trained checkpoint.
Expected hidden values `[5,0]`, outputs `[11,-3]`, gathered values `[-3,11,-3]`
and final sum `5` are derived analytically and compared byte-for-byte. CPU-only
CI executes this example in addition to testing it; success is not a skipped
GPU test or compilation-only result.

```bash
cargo test --locked -p nnis-cpu --all-targets
cargo run --locked -p nnis-cpu --example portable_f32_graph
bash scripts/check-portable-cpu.sh
```

Regression coverage includes rectangular layout, FMA versus split rounding,
ordered cancellation, NaN/infinity, finite overflow after a valid output prefix,
usage/shape/index violations, repeated gather/scatter indices, scratch boundaries,
unchanged destination and capacity, and preservation of source buffers.

## Ecosystem boundary

These reference operations are groundwork for SML numerical/page execution,
FLAT backend comparisons and future ElasticXxx plan validation. No such consumer
integration is claimed by this PR. It leaves all historical CUDA evidence and
the frozen ElasticBitAllocation Stage-B model/backend/data protocol unchanged.
Allocator/search, final-test and registry publication remain unauthorized.

Next: a versioned portable kernel/graph dispatch contract, followed by independently
qualified WGPU parity and real consumer adapters. The fixed graph here does not
complete those remaining P1/P2/P3/P4/P9 requirements.
