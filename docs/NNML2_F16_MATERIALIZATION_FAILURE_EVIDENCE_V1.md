# NNML2 F16 materialization failure evidence v1

## Purpose

`F16WeightMaterializationFailureEvidenceV1` records the allocation-lifetime state that NNIS had observed when an F32 -> F16 weight materialization failed.

This contract complements `F16WeightMaterializationMemoryEvidenceV1`, which is produced only after a successful materialization. It does not replace the successful evidence contract and it does not change the F16 execution plan or numerical semantics.

## Observation boundary

Failure evidence is captured at the point where the materialization operation returns an error, before the stack unwinds and Rust RAII drops the F16 buffers created by that materialization attempt.

The fields ending in `_at_failure_detection` therefore describe the tracked live allocations at failure detection. They do **not** claim that those allocations remain live after the constructor returns the error.

The scoped byte count is limited to:

- the source `ModelWeights` owned device allocations, which are live during the materialization call; and
- F16 resident or temporary `DeviceBuffer<u16>` allocations that the materialization tracker has observed as live at that point.

The evidence is derived from actual successful `DeviceBuffer` allocations and their `size_bytes()`. It is not reconstructed from model geometry or nominal precision.

## Versioned contract

The public schema version is:

`NNIS_F16_WEIGHT_MATERIALIZATION_FAILURE_EVIDENCE_VERSION = 1`

`F16WeightMaterializationFailureEvidenceV1` records:

- the exact F16 execution plan;
- the exact source `WeightAllocationSummaryV1`;
- the failing operation when `NnisError` exposes one;
- the CUDA driver code when the root error is a CUDA driver error;
- the full NNIS error message, including attached context;
- live F16 allocation bytes at failure detection;
- live temporary F16 allocation bytes at failure detection;
- scoped owned allocation bytes at failure detection;
- the peak live F16 allocation bytes observed before the failure;
- the peak live temporary allocation bytes observed before the failure;
- the peak scoped owned allocation bytes observed before the failure; and
- the ordered materialization events recorded before failure detection.

The source allocation summary is reconciled with the tracker before evidence is accepted. Evidence construction fails closed if that invariant is violated.

## Constructor surface

Existing constructors preserve their historical `Result<_, NnisError>` surface and return the original runtime/model error:

- `F16ReferenceModel::new`
- `F16ReferenceModel::new_with_execution_plan`
- `F16ReferenceModel::new_with_execution_and_attention_plan`

The opt-in attempt constructors expose failure evidence:

- `F16ReferenceModel::new_with_execution_plan_attempt`
- `F16ReferenceModel::new_with_execution_and_attention_plan_attempt`

They return `F16ReferenceModelConstructionFailure` on failure. The wrapper provides:

- `error()` for the original `NnisError`;
- `materialization_failure_evidence()` for optional v1 failure evidence;
- `materialization_evidence_error()` if evidence conversion itself failed; and
- `into_error()` to recover the original `NnisError` by value.

The original runtime error retains precedence. A failure to build auxiliary evidence does not replace it.

## Scope of failures

V1 can attach materialization evidence only after the F16 materialization tracker has been created and the failure occurs inside the tracked `F16ModelWeights::from_f32` materialization phase.

Errors before that phase, such as execution-plan validation or kernel/candidate setup failures, still return through `F16ReferenceModelConstructionFailure` when an attempt constructor is used, but have no materialization trace because no tracked F16 materialization had begun.

Errors after successful F16 weight materialization, such as later model-construction allocations, are also outside this materialization-failure schema.

## Explicit non-claims

This contract is **not** evidence of:

- physical GPU page residency;
- process-wide VRAM usage;
- CUDA allocator metadata or page-table overhead;
- JIT/module/kernel memory;
- RoPE, KV-cache, session, workspace, or serving memory;
- memory remaining allocated after the failed constructor returns;
- latency, throughput, quality, compression, or effective bits per parameter; or
- live/in-place F16 execution-plan switching.

`cuMemGetInfo` free/total snapshots and CUDA pointer/address metadata have different semantics and are not treated as physical-residency attribution for this contract.

## Validation

Host tests exercise both a failure after resident/temporary allocation overlap and a failure before the first F16 allocation. The implementation is additionally gated by the repository Rust checks, Clippy with warnings denied, Rust 1.77 MSRV, and the existing SmolLM2 harness workflow before merge.
