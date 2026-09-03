# DAL0 — NNIS binding to the FLAT DA-LUC research contract

Status: research-only contract audit. This document preserves the original DAL0 ownership and prior-art boundary while reconciling it with the DA-LUC implementation currently present on NNIS `main`. It makes no quality, compression, latency, throughput, novelty, or runtime-promotion claim.

## Original audited repository heads

The original PR #85 audit was performed against:

- NNIS: `4101b8924f1e5400a7871259b9c1b732ae3c77bb`
- FLAT-ATTENTION: `c35b044c5324963a300ff50da0f7ec10dcc6db71`
- SLHAv2: `b2cee0d0f30ff0fc752c03193cb9ed93dc91be53`

These are historical audit identities, not dependency pins.

## Reconciliation with current NNIS

Since PR #85 was opened, its original implementation has been superseded on `main` by the current `da_luc_plan` and `da_luc_evidence` modules. The current implementation keeps DA-LUC research-only and fail-closed, exposes a dense-reference default execution policy, separates FLAT-facing semantic identity from NVIDIA-local physical identity, and includes deterministic SHA-256 plan and semantic fingerprints.

The current plan surface also narrows the first admitted CUDA-facing subset instead of carrying the broader capability-intent structure from the original PR. PR #85 therefore must not restore its historical `da_luc_plan.rs`, `da_luc_fingerprint.rs`, `lib.rs`, or dependency edits over the newer implementation.

## Ownership boundary

FLAT-ATTENTION owns the attention-facing DA-LUC semantic/view contract. NNIS owns NVIDIA-local validation, execution policy, physical realization identity, evidence binding, and future CUDA execution. NNIS must not redefine FLAT semantic identity through a physical packing choice.

SLHAv2 remains the owner of its own tile, codec, latent, residual, scale, and dynamic-allocation semantics. Those payload semantics are not imported into the NNIS plan identity. Any future SLHAv2-to-NNIS path must explicitly map a compatible representation through the FLAT attention-facing contract.

## Runtime boundary

DA-LUC remains separate from the existing F16/F32 attention plans and runtime defaults. Constructing or validating a DA-LUC candidate does not authorize compressed KV execution by itself.

The dense NNIS KV path remains the correctness reference unless an explicit candidate path is selected and all required capability/evidence gates succeed. Missing capability, unknown schema, incompatible geometry, unsupported representation metadata, or evidence mismatch must fail closed.

## Current admitted semantic surface

The current NNIS DA-LUC plan binds and validates the following classes of metadata:

- FLAT DA-LUC view schema version;
- Q/KV head geometry and K/V head dimensions;
- K subspace/codebook/index semantics;
- independent V representation semantics;
- residual semantics;
- row order, topology, alignment and padding;
- direct compressed-consumption intent;
- an NNIS-local physical layout identity that is deliberately excluded from the FLAT-facing semantic fingerprint.

The current implementation also validates a deliberately narrow FDAL3-compatible subset before any future compressed CUDA path may be considered.

## Identity and evidence

NNIS maintains two distinct identities:

1. a complete plan fingerprint, which includes NNIS plan/physical identity; and
2. a semantic fingerprint, which covers only FLAT-facing logical/view semantics plus the direct-consumption requirement.

This separation prevents an NNIS packing change from redefining the semantic contract.

The current `da_luc_evidence` module binds reconstruction/storage evidence to the plan and rejects schema, semantic-view, storage, or provenance drift. Evidence metadata contains no representation payload, CUDA pointer, secret, or measured-performance surface.

## Prior-art boundary

DAL0 inherits the existing DA-LUC prior-art boundary. Asymmetric K/V quantization, sparse outlier preservation, low-bit KV caches, codebooks/PQ, lookup-based key scoring, and related compressed-attention ingredients are established research/engineering directions. This audit makes no novelty claim.

## Historical PR #85 disposition

The code originally proposed by PR #85 is intentionally not replayed because newer DA-LUC plan/evidence code is already present on `main` and is stricter in several important areas. The remaining mergeable value of PR #85 is this audited ownership, provenance, and research-claim boundary.
