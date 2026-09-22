# Portable built-in F32 graph v1

Scope: P2c metadata validation and synchronous scalar CPU graph execution.
This is a typed in-memory builtin dispatch contract, not a serialized model
format, shader ABI, kernel compilation package or general inference scheduler.

## Ownership and reuse

`nnis-core::graph` has no dependency on a hardware API or numerical library.
`nnis-cpu::graph` reuses every existing P2b arithmetic operation and the P2a
memory implementation. It does not duplicate SciRust numerical kernels or change
`finite-f32-le-ordered-fma-v1`. The earlier SciRust MSRV/reuse audit remains in
`PORTABLE_CPU_NUMERICAL_V1.md`.

This provides a bounded admission surface for later ElasticXxx/FLAT/SML consumers;
no cross-repository actuator or model integration is implemented by this slice.

## Plan and validation

`F32GraphV1` borrows input shapes, nodes and index arrays. Input slots are numbered
first; each node appends exactly one new value. Every operand must refer to an
input or an earlier node. All nodes execute in declaration order, even unused
ones. The last node is the sole published output.

There is no implicit transpose, broadcasting, in-place write, cycle, loop,
conditional routing, node elimination, parallel reassociation or dynamic shader.
Vectors and matrices are distinct shapes even when byte sizes agree. Projection
uses `[1,K] x [K,N] -> [1,N]`, with the matrix orientation carried by its shape.
ScatterAdd takes an immutable base and source and creates a fresh output.

`validate` checks the schema and exact policy identity; positive shape sizes;
checked u64 products/sums; input/node count; every operand and output shape;
every gather/scatter index; and the caller's limits. It allocates no heap memory.
`ValidatedF32GraphV1` has private fields and retains immutable borrows, so graph
metadata cannot be changed behind a still-used validation handle. There is no
unchecked public constructor or deserialization bypass.

This does not authenticate a model or weight payload. Validation is structural
and resource admission only; data-dependent finite-arithmetic checks remain an
execution responsibility.

## Explicit budgets

The caller supplies positive limits for inputs, nodes, one tensor's bytes, live
graph payload, scratch payload and logical work. Zero never means unlimited.

The v1 CPU schedule deliberately retains every node output until completion.
For that schedule, portable validation computes the conservative bound:

`input binding bytes + all node output bytes + largest node output bytes`.

The final term is the largest temporary result staging payload used by P2b.
Aliases among input bindings are charged per binding, not deduplicated by
address. This overestimate is deliberate and does not claim physical allocation
identity. Limits are inclusive: a graph exactly at a limit is admissible.

The logical-work counter includes one initial scan of all input values, per-node
input validation visits, scalar arithmetic/selection, output writes, and index
scans. Checked accounting is enforced before scanning each index list. It is a
logical work admission measure, NOT an instruction count, timing bound or
performance model.

Payload limits exclude retained allocator capacity above length, allocator
bookkeeping, stacks, caller metadata, control vectors and process RSS. The CPU
report separately records retained node-buffer capacities, maximum scratch
capacity, and node-handle-vector capacity bytes. These are not physical page
residency. No process-wide memory budget or successful OS allocation is promised.

The retain-all-output bound is intentionally simple rather than optimal. Liveness
analysis, pooling and scratch reuse require a separately qualified planner.

## CPU execution and failure semantics

`execute_f32_graph` accepts only a validated handle and immutable buffer bindings.
Before node allocation it checks actual binding counts/lengths/STORAGE usages,
all input values (including unused inputs), host representability, device
per-buffer capabilities and the supported numerical policy.

Node storage is newly allocated and zero initialized. The executor invokes P2b
kernels without copying entire inputs or weight matrices. ScatterAdd initializes
its fresh destination from the immutable base; it never updates the caller's
base buffer. Buffer-handle reservation is fallible and node counts are bounded.

Each operation stages its result as in P2b. On any returned error, all private
intermediates and the current destination drop; no output escapes and all
caller-owned inputs retain their bytes and capacity. A later arithmetic overflow
cannot publish a successful prefix of the graph. Only full success returns the
last tensor plus the completed execution report. This is in-process visibility
atomicity, not crash durability, a callback transaction, a rollback of external
effects, or a guarantee against OS overcommit termination.

## Executable qualification

The same seven-operation analytical fixture from P2b is now represented by a
runtime plan and dispatched through this executor, rather than a hardcoded
sequence of kernel calls. Its five input bindings total 68 bytes, seven output
payloads total 56 bytes, and maximum scratch payload is 12 bytes. The conservative
live-payload bound is therefore 136 bytes. A 135-byte limit must reject it.
Expected output is the exact F32 value 5. Rebinding a different valid input to the
same immutable plan produces a separately checked result of 15 in tests.

```bash
cargo test --locked -p nnis-core --test graph_contract
cargo test --locked -p nnis-cpu --test graph_execution
cargo run --locked -p nnis-cpu --example portable_graph_plan
bash scripts/check-portable-cpu.sh
```

The permanent gate tests structural and runtime regressions in debug and release,
builds strict rustdoc, preserves previous memory/numerical tests and executes
all three portable examples. Its dependency guard includes normal, build and
dev dependencies. No new tests conditionally skip for lack of a GPU.

## Remaining gates

No WGPU implementation, cross-hardware numerical parity, trained model, compressed
weight graph, serving benchmark or scientific result is asserted. This is not
completion of the portable shader/kernel package, WGPU, FLAT/SML integration or
ElasticXxx actuation phases. The historical ElasticBitAllocation Stage-B protocol
and its allocator/search/final-test locks are unchanged. No publication to a
package registry is authorized.

Next: explicit portable kernel artifact/binding/capability identity and WGPU
execution with independently checked numerical semantics, then real consumers.
