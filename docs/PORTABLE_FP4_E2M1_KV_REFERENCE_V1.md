# Portable FP4 E2M1 group-scaled KV reference and accounting v1 (DSV41-3)

Status: software reference contract. The layout and accounting are backend-neutral (`nnis-core`), with a CPU encode/decode oracle (`nnis-cpu`).
This is roadmap item `NNIS_DSV41_3_portable_FP4_E2M1_group_scaled_KV_reference_and_accounting`
of `deepseek_v41_kv_runtime_program_2026_09_24` (off-main sovereignty roadmap).
The DA-LUC overlay's exact-storage rules apply.

## Format (`nnis_core::kv_fp4`, layout version 1)

- Representation id: `nnis.kv.fp4-e2m1.group-scaled`.
- Element: 4-bit E2M1 (1 sign, 2 exponent with bias 1, 1 mantissa). This is the OCP MX FP4 element encoding.
  - Magnitudes: `0, 0.5, 1, 1.5, 2, 3, 4, 6`. Code `0x8` is `-0.0`.
  - There are no infinities or NaNs.
  - This is a non-uniform floating-point grid. It is **distinct from the NNIS INT4 formats**, and the two must never share evidence.
- Groups: `group_size` consecutive values inside one row. The size must be even, from 2 to 4096. Groups never cross rows.
  - The final group of a row is padded with zero codes.
- Packing: in each row's padded code stream, value `2i` is the low nibble of byte `i` and value `2i+1` is the high nibble.
- Scale per group, explicit in the layout:
  - `F32`: little-endian IEEE F32, finite and non-negative (4 bytes). The CPU encoder uses `amax / 6`.
  - `E8M0`: one byte `e`, giving scale `2^(e-127)` with `e` from 0 to 254. `255` is invalid.
    - The CPU encoder picks the smallest `k` from -127 to 127 with `6 * 2^k >= amax`.
- Descriptor metadata: 24 bytes, charged to every block.
  - Fields: version `u32`, rows `u64`, row width `u32`, group size `u32`, scale tag `u8`, and 3 reserved bytes.

## Exact storage accounting (`Fp4KvStorageV1`)

`total_bytes = code_bytes (incl. padding codes) + scale_bytes + 24`

Reported separately:
- `logical_values`
- `groups`
- `code_bytes`
- `scale_bytes`
- `padding_values` / `padding_bits`
- `metadata_bytes`
- `codebook_bytes` (always 0)
- `residual_bytes` (always 0 in v1)

`effective_bits_per_value = total_bits / logical_values` is available as an exact rational and as `f64`. `compression_ratio_against(F32|F16|BF16)` is `dense_bits / total_bits`, also as an exact rational.

Examples:
- 128×64, group 32, E8M0: 4.2734375 bits/value.
- 3×10, group 4, F32: 20.8 bits/value.

Nominal 4-bit width is never a compression result.

## CPU reference (`nnis_cpu::fp4_kv`)

- `CpuFp4E2M1KvBlockV1::encode(layout, &values)` requires exactly `rows × row_width` finite F32 values.
  - Scaled magnitudes round to the nearest E2M1 value, with ties going to the even code.
  - The sign is preserved, including `-0.0`.
  - Any group that would decode to a non-finite F32 fails with `DecodeOverflow`.
- `from_parts(layout, codes, scales)` rejects wrong lengths, non-zero padding nibbles, invalid scales, and values that would decode to non-finite F32.
- `decode()` forms `magnitude × scale` exactly in F64 and rounds once to F32. Padding is excluded.
- The dense F32 input remains the reference. `fp4_reconstruction_error` reports the maximum, sum, and mean absolute error in index order.

## Evidence

Host-only unit tests cover:
- the E2M1 table derived from sign/exponent/mantissa;
- exact round trips of representable values, with packed nibble order;
- ties-to-even behaviour;
- smallest-covering E8M0 scales and F32 `amax/6` scales;
- signed zero and all-zero groups;
- zeroed padding that is excluded from decode;
- the per-value error bound `|x - x̂| <= scale`;
- tamper rejection;
- overflow rejection;
- agreement between the storage report and the encoded bytes.

No GPU, model, or physical run was performed.

## Claim boundary

- This is an isolated reference-error oracle only. It makes no model-quality, KV memory, latency, or throughput claim.
  - Compressed-KV runtime promotion requires quality, memory, decode latency, and tokens/s reported together from a real-model campaign (DSV41-5).
- An opt-in portable-session FP4 *shadow* storage mode exists (`PortableKvStorageModeV1::Fp4E2M1`
  on `CpuPortableSession` / `WgpuPortableSession`) for exact accounting only; the dense
  F32 reference stays the default and the semantic path. This is still not an attention
  or CUDA runtime KV path and authorizes no quality/memory/latency claim.
- There is no WGPU or CUDA execution. A later portable GPU path must match this oracle.
- This is not INT4, and no INT4 evidence carries over.
- Semantic ownership: the DSV41 programme assigns compressed-KV representation semantics to SLHAv2.
  - This is the NNIS runtime reference named by the roadmap step.
  - If SLHAv2 publishes a canonical FP4 KV contract, NNIS must bind to it and be verified against it, under a new layout version if the contracts differ.
- There is no novelty claim and no DeepSeek-V4.1 equivalence claim.
