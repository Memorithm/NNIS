# NNML2 isolated INT4 projection v1

Status: isolated executable projection contract; **not a full-model INT4 runtime**.

This slice advances the NNIS fixed-4-bit reference from storage/accounting into one explicitly bound execution primitive while preserving the scientific boundary required by ElasticBitAllocation Stage B.

## Execution contract

The kernel primitive is `F32Int4Gemv`.

It executes:

`[1,K] F32 activation × [K,N] packed signed-INT4 weight -> [1,N] F32 output`.

The weight payload is read directly from the live `DeviceBuffer<u8>` created by `Int4ReferenceModelStorageV1`. One live `DeviceBuffer<f32>` scale is read from the same storage allocation.

For each weight element the kernel:

1. extracts the low or high four-bit nibble from the packed byte;
2. sign-extends the nibble to an integer;
3. converts that integer to F32 and multiplies by the resident scale in a register;
4. accumulates with explicit `fmaf` in increasing-K order.

No dense weight tensor, dequantization workspace, or second full-precision weight allocation is created by the projection path.

The kernel safely decodes the full signed nibble domain `[-8,7]`. The stricter NNIS storage contract remains authoritative and never emits the reserved `-8` code.

## Versioned plan

`Int4ReferenceProjectionPlanV1` binds:

- projection-plan schema version;
- exact INT4 storage version;
- exact logical weight name;
- exact matrix rows and columns;
- `signed-int4-to-f32-register-v1` dequantization semantics;
- `increasing-k-f32-fma-v1` accumulation semantics;
- `dense_weight_materialization = false`.

Execution fails closed when the logical name does not exist, names a vector rather than a matrix, or the requested matrix orientation differs from the shape recorded when the source graph was quantized.

The logical binding table is internal to `Int4ReferenceModelStorageV1`; raw packed/scale device buffers remain private.

## Storage schema boundary

`Int4ReferenceStorageSummaryV1` is unchanged.

Its existing `execution_qualified = false` continues to state that the reference storage is **not qualified as a full-model execution format**. Qualification of this isolated projection primitive does not mutate that meaning and does not promote the model runtime.

## SmolLM2 smoke

The operator harness is:

`cargo run -p nnis-bench --example smollm2_int4_projection_smoke -- --model DIR --device 0`

It requires the pinned SmolLM2-135M fixture and executes the actual packed `layers.0.q_proj` matrix through the INT4 storage binding.

The oracle is independently formed by:

- copying the original F32 `q_proj` source;
- applying the public deterministic INT4 reference quantizer;
- reconstructing quantized F32 values;
- evaluating the same increasing-K F32 `mul_add` projection on CPU.

The smoke fails unless the GPU primitive bit-matches that **quantized-weight** ordered oracle. This is a representation/kernel consistency check, not equivalence to the original dense model.

## Explicit non-claims

This slice does not establish:

- equality to the original dense projection;
- model NLL, perplexity, generation parity or model quality;
- embedding, norm, attention, MLP or LM-head integration;
- full-model INT4 execution;
- latency or throughput improvement;
- conversion/materialization peak memory;
- physical page residency or process-wide VRAM attribution;
- ElasticBitAllocation search/allocator authority;
- final-test access.

## Next gate

The next engineering gate is a full decoder execution graph that consumes packed INT4 weights for every required weight operation, or explicitly documents and accounts any tensor family that cannot yet use the representation.

That runtime must not silently retain a complete F32 weight graph during steady-state execution. Only after the pinned SmolLM2 development split runs end-to-end may the Stage-B fixed-4-bit execution baseline be considered for closure.
