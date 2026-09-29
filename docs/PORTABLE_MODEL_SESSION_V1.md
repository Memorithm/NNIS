# Portable model session v1

Scope: backend-neutral encode / decode_one / KV-advance / truncate session
surface, host-dense F32 KV layout, a CPU reference implementation, and a WGPU
parity implementation over a tiny analytical synthetic model. This is not a
trained-model runtime, not a checkpoint loader, and not a performance claim.

## Ownership

- `nnis-core::session` defines `PortableSessionV1`, `PortableKvCacheV1`,
  `SyntheticPortableModelSpecV1` and fail-closed `PortableSessionError`.
- `nnis-cpu::session::CpuPortableSession` implements the trait with the existing
  P2b F32 kernels (`gather`, `binary` Add, `project_kn`) and can optionally
  project logits through the P2c `execute_f32_graph` plan for parity checks.
- `nnis-wgpu::session::WgpuPortableSession` mirrors the CPU session on the same
  synthetic model through WGSL F32 kernels. `session_parity` tests SKIP without
  an adapter; software adapters are code-path only.
- The CUDA `nnis_rt::KvCache` and `nnis_model::InferenceSession` paths are
  unchanged. Portable work must not depend on NVIDIA facilities.

## Session operations

1. `stage_token(token)` gathers the synthetic embedding, forms
   `running_sum + row`, projects logits, and stores a pending KV row without
   changing `kv_len` / `position`.
2. `advance_kv()` commits the pending row into `PortableKvCacheV1`, updates the
   running sum and increments `position`.
3. `encode(tokens)` requires a fresh session and stages+advances each token.
4. `decode_one(token)` stages+advances one token.
5. `truncate(length)` clears any pending row, shortens every layer
   all-or-nothing, recomputes the running sum from committed rows, and refreshes
   logits (or clears them when `length == 0`).

## Synthetic model

`SyntheticPortableModelSpecV1::tiny()` uses vocab=4, hidden=4, capacity=8,
layers=1. Embeddings are one-hot (modulo hidden) and the LM head is an identity
projection when dimensions allow. Exact dyadic F32 results make CPU unit tests
analytical. Synthetic fixtures are never model-quality or hardware evidence.

## KV layout and accounting

`PortableKvCacheV1` stores `[layer][capacity][row_width]` host F32 values under
identity `nnis.kv.host-dense-f32.v1`. `logical_payload_bytes` and
`capacity_payload_bytes` count only addressed f32 payload. They are not process
RSS, physical pages, or a compression ratio.

## Claim boundary

- No latency, throughput, acceptance-rate or quality claim.
- No physical, Thor, ARM64 or hardware result.
- No change to the CUDA generation path.
- WGPU software-adapter runs are code-path only, never hardware evidence.
- FP4 KV storage is not authorized by this surface; it remains blocked until
  DSV41-5 campaign evidence is reviewed on top of the portable session.

## Numerical policy

`finite-f32-le-ordered-fma-v1`, identical to the portable F32 graph and CPU
kernel contract (`PORTABLE_SESSION_POLICY`).
