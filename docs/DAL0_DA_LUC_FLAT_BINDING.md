# DAL0 — NNIS binding to the FLAT DA-LUC view contract

Status: research-only contract/audit slice. This document makes no novelty, model-quality, compression-ratio, latency, throughput, VRAM, energy, or runtime-promotion claim.

## Exact audited repository heads

- NNIS: `4101b8924f1e5400a7871259b9c1b732ae3c77bb`
- FLAT-ATTENTION: `c35b044c5324963a300ff50da0f7ec10dcc6db71`
- SLHAv2: `b2cee0d0f30ff0fc752c03193cb9ed93dc91be53`

These commits identify the source trees inspected for DAL0. They are evidence/provenance references, not dependency pins and not semantic-version substitutes.

## Ownership boundary

FLAT-ATTENTION owns the attention-facing DA-LUC logical/view contract. Its research-only `flat_attention::api::research_da_luc` schema v1 defines head geometry, K codebook/index semantics, independent V representation semantics, residual semantics, physical row/topology/alignment metadata, and fail-closed backend capability validation.

NNIS owns NVIDIA-specific storage realization, CUDA Driver/NVRTC execution, device capability checks, kernel mapping, explicit execution-plan selection, and physical evidence. DAL0 therefore adds only an NNIS adapter/selection plan that binds to FLAT schema v1. The NNIS plan must not become a competing ecosystem DA-LUC semantic contract.

SLHAv2 owns its specialized serialized tile/codec semantics. At the audited head, its 128-byte tile includes latent bytes, residual bitmap, scalar metadata, token/position/head identity, codec flags and group scales. Those offsets/flags are not copied into the NNIS DAL0 plan. A later SLHAv2 adapter must explicitly map a compatible representation into the FLAT view instead of treating an SLHA tile as the FLAT contract by similarity.

## Current NNIS dense/reference boundary

NNIS already has versioned F32/F16 cached-attention execution plans and a model configuration that validates MHA/GQA/MQA geometry through exact divisibility of query heads by KV heads. Existing F16/F32 plans and constructors remain unchanged by DAL0.

`NnisKvExecutionPolicy::default()` is `dense_reference`. The DA-LUC plan is research-only explicit opt-in metadata and is not wired into the current runtime allocator, append path, cached-attention kernel dispatch, model format, or default execution plan.

## FLAT FDAL3-compatible subset admitted by DAL0

The first NNIS candidate intentionally maps only the already-qualified narrow FLAT FDAL3 v1 direct-compressed correctness subset:

- `flat_view_schema_version = 1`;
- exact NNIS model head geometry with integral Q-head to KV-head grouping;
- key/value head dimensions in `1..=128`;
- contiguous storage;
- `BatchHeadToken` row order;
- zero-filled power-of-two plane alignment of at least 4 bytes;
- F32 K codebooks, shared across KV heads or per KV head;
- 8-bit LSB0 K indices;
- no K sparse residual;
- query-local `subspaces * codebook_entries <= 2048`;
- groupwise-affine 8-bit LSB0 V;
- F32 V scales;
- U8 V zero points;
- no V sparse residual;
- direct compressed consumption with no dense K/V materialization.

Scalar/register conversion is still allowed and expected. The property is therefore **no dense K/V materialization**, not "zero dequantization".

Paged storage, dense V, sub-byte K/V, F16/BF16 representation planes, MSB0 streams, sparse residuals and broader physical layouts fail closed in DAL0. They require later explicit evidence and schema-compatible extension; they are never implicit fallbacks.

## Semantic identity versus NVIDIA physical identity

`NnisDalucCandidatePlan::semantic_fingerprint()` hashes only the FLAT-facing schema/version, geometry, representation/view metadata and direct-consumption requirement.

`NnisDalucCandidatePlan::fingerprint()` hashes the complete NNIS plan, including the NVIDIA-local physical-layout identity.

Both are deterministic SHA-256 provenance/cache identities, not authentication. No token ids, codebook payload values, residual payload values, CUDA pointers, stream handles, credentials or secrets are fields of the plan.

This separation allows a future CUDA packing to change implementation identity without silently redefining the validated FLAT logical contract.

## Backend capability boundary

DAL0 never infers support merely from a device name, compute capability, compiler success, or the presence of CUDA. A caller must provide `NnisDalucBackendCapabilities`; missing direct-compressed support, incompatible head/LUT limits, unsupported packed representation, or incompatible alignment rejects the candidate.

A later runtime adapter must derive those capabilities from real implementation/device checks. DAL0 does not claim that any current NNIS CUDA kernel already consumes this representation.

## Prior-art / novelty boundary

The FLAT FDAL0 audit already records relevant established ingredients and DAL0 inherits that non-novelty boundary rather than rebranding them:

| Reference | Relevant established ingredient | DAL0 implication |
| --- | --- | --- |
| KIVI, arXiv:2402.02750 | low-bit asymmetric K/V quantization | K/V asymmetry is not a novelty claim |
| KVQuant, arXiv:2401.18079 | low-bit KV with key-specific treatment and sparse outliers | sparse residual/outlier ideas are prior-art ingredients |
| QServe, arXiv:2405.04532 | fused serving around low-bit KV | direct kernel consumption of low-bit KV is not sufficient for novelty |
| PQCache, arXiv:2407.12820 | product-quantized keys/codebooks for retrieval | PQ/codebook/LUT concepts require explicit comparison, not renaming |
| TensorRT-LLM quantized/paged KV documentation | production quantized/paged KV modes | external production baseline when regimes are reproducibly comparable |

No combination-level novelty conclusion is made by this implementation audit.

## Validation requirements

DAL0 tests require:

- dense execution policy remains the default;
- the exact narrow FDAL3-compatible metadata mapping validates;
- unknown FLAT schema versions fail closed;
- model/GQA/MQA geometry mismatch fails closed;
- unsupported K/V representation, residual, packing, paging or alignment fails closed;
- missing backend capability fails closed;
- canonical serialization/fingerprints are deterministic;
- semantic identity changes on semantically relevant metadata changes;
- the semantic fingerprint excludes NNIS plan/physical identity;
- serialized identity metadata has no payload/secret surface;
- unknown JSON fields fail closed.

Required repository CI remains `cargo fmt`, workspace check, Clippy `-D warnings`, tests, and Rust 1.77 MSRV check on the exact PR head.

## Next gate

The smallest safe successor is DAL1 host/reference mapping: construct a host-side adapter from a validated FLAT-v1-compatible descriptor/payload into NNIS-owned reference metadata and exact storage accounting, without CUDA promotion and while retaining dense KV as the correctness comparator.

DAL2 may only follow after DAL1: implement the first CUDA q_len=1 direct-compressed key-scoring/value-consumption candidate against the same semantic fingerprint, with no dense K/V materialization and no runtime-default change. Any performance statement then requires separate physical and end-to-end evidence.
