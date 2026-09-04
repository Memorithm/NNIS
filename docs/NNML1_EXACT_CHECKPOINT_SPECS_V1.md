# NNML1 exact decoder checkpoint specifications v1

`ExactDecoderCheckpointSpec` records one exact external decoder checkpoint identity
and the exact NNIS `ModelConfig` plus decoder-capability record that may be associated
with it.

The contract is intentionally narrower than model-family support. A matching spec
proves only that source identity and decoder configuration agree with a frozen NNIS
software contract. It does not establish tokenizer identity, reference-logit parity,
generation quality, CUDA execution success, performance, or compatibility with any
sibling checkpoint.

## Contract version

- `NNIS_EXACT_DECODER_CHECKPOINT_SPEC_VERSION = 1`
- validation first requires the current NNIS decoder execution contract to accept the
  configuration
- all model geometry, EOS policy, RMSNorm epsilon, RoPE theta, activation and weight
  dtype must match exactly
- the derived `DecoderExecutionCapabilities::canonical_record()` must match the
  checkpoint spec's frozen capability record
- `canonical_identity()` provides a deterministic provenance/cache record; it is not
  an authentication primitive

## Frozen specifications

### SmolLM2-135M BF16

- repository: `HuggingFaceTB/SmolLM2-135M`
- revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- `model.safetensors` SHA-256:
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- decoder geometry: hidden 576, intermediate 1536, 30 layers, 9 Q heads, 3 KV
  heads, head dimension 64, vocabulary 49152
- capability profile: grouped-query attention, Llama rotate-half unscaled RoPE,
  SwiGLU/SiLU, BF16 weights

NNML0's real-Safetensors qualification now consumes this shared spec instead of
maintaining a second hand-written copy of the same geometry and capability checks.
The physical NNML0 gate remains open until exact-main CUDA evidence exists.

### TinyLlama-1.1B-Chat-v1.0 BF16

- repository: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- revision: `d9128824c0c80111be21424e68086f52413fb413`
- `model.safetensors` SHA-256:
  `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`
- decoder geometry: hidden 2048, intermediate 5632, 22 layers, 32 Q heads, 4 KV
  heads, head dimension 64, vocabulary 32000
- capability profile: grouped-query attention, Llama rotate-half unscaled RoPE,
  SwiGLU/SiLU, BF16 weights

These values are the same pinned source identity and geometry already used by the
committed TinyLlama massive-campaign fixture. Registering them in the shared exact
checkpoint contract does not promote TinyLlama or claim that the native Safetensors
path, generation path, or any candidate kernel has passed physical qualification for
this checkpoint.

## Deliberately not covered by v1

Tokenizer and chat-template identity remain owned by the corresponding reference or
campaign fixture until a versioned tokenizer identity contract is introduced. Raw
`config.json` bytes are not hashed by this type; the supported semantic fields are
validated exactly after parsing. Physical checkpoint loading, reference parity,
quality, latency, throughput, VRAM, concurrency and model-family admission remain
separate evidence gates.
