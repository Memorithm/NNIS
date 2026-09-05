# Elastic execution transition boundary v1

NNIS owns the semantics and physical materialization of its execution plans. An external adaptive controller must not infer that a versioned NNIS plan is live-switchable merely because multiple plans exist.

`nnis-model` therefore exposes `F16ExecutionTransitionRequirementsV1` and `F16ReferenceExecutionPlan::transition_requirements()`.

## Current F16 contract

All currently implemented `F16ReferenceExecutionPlan` layouts declare:

- schema version `NNIS_F16_EXECUTION_TRANSITION_REQUIREMENTS_VERSION = 1`;
- transition mode `model_rebuild_required`;
- source logical weights are required to construct the target resident layout;
- active sessions are not preserved;
- KV state is not preserved;
- live in-place transition is not authorized.

These statements follow the current implementation. `F16ReferenceModel::new_with_execution_and_attention_plan()` selects candidate kernels and materializes resident F16 weights according to the execution plan during model construction. The resulting model stores its chosen execution plan and exposes it read-only.

## Consequence for ElasticXxx

This contract is deliberately not an `ElasticXxx` dependency and does not make NNIS depend on a newer Rust toolchain. NNIS retains its Rust 1.77 MSRV and owns the backend semantics.

A cross-repository adapter may consume the versioned requirement and must fail closed if it requires a live transactional transition. In particular, the current NNIS F16 path must not be presented as an implementation of an in-place `apply_profile` operation.

A future controller could potentially orchestrate a full model rebuild, but that is a different transition class. It would need explicit ownership and evidence for:

- source-weight availability and identity;
- construction of the target resident representation;
- active-request quiescence;
- session recreation;
- KV-state handling or intentional reset;
- verification of the newly constructed model;
- rollback by reconstructing a prior plan when possible;
- failure semantics when reconstruction or rollback cannot complete.

None of those behaviors are authorized by this v1 requirements contract alone.

## Resource telemetry

NNIS already exposes real CUDA capacity telemetry independently through `nnis_rt::Context::mem_info()`, which returns CUDA-driver-reported free and total device memory in bytes after making the owned primary context current.

That measurement can support an external observer boundary without implying that NNIS execution plans are live-switchable.

## Scope

This contract does not:

- add a dependency on ElasticXxx;
- implement a live execution-plan switch;
- migrate or preserve active sessions;
- migrate or preserve KV state;
- authorize a rebuild-based rollback protocol;
- define MoE expert-count, expert-width, or activation-budget semantics;
- make a new latency, throughput, memory, or quality claim.

The qualified SmolLM2/Thor execution-plan selector remains a narrowly scoped NNIS capability. It must not be generalized to other models, devices, or adaptive dimensions without separate qualification evidence.
