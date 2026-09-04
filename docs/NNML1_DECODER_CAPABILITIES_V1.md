# NNML1 decoder capabilities v1

NNIS now exposes a versioned decoder execution-capability profile derived from a validated `ModelConfig`.

## Purpose

`ModelConfig::decoder_capabilities()` answers a narrow runtime question: which already-implemented decoder semantics and attention topology will NNIS execute for this validated configuration?

The v1 profile records:

- capability contract version;
- MHA, GQA, or MQA topology derived from Q/KV head counts;
- exact current RoPE family: Llama rotate-half, unscaled and non-interleaved;
- exact current MLP family: SwiGLU with SiLU;
- persisted/device weight dtype;
- Q-head count, KV-head count, and head dimension.

`canonical_record()` provides deterministic text suitable for evidence fixtures and cache/provenance joins. It is not an authentication primitive.

## Important separation

A decoder capability profile is not a model-family support claim.

For example, the Safetensors loader currently accepts only the explicitly validated `LlamaForCausalLM` / `model_type=llama` source contract and rejects Mistral or other architectures. A model from another family is not admitted merely because its head counts happen to classify as MHA, GQA, or MQA.

Likewise, capability compatibility does not establish model-quality parity, tokenizer compatibility, numerical equivalence to another runtime, or a performance claim.

## Current topology semantics

- MHA: `num_key_value_heads == num_attention_heads`;
- MQA: `num_key_value_heads == 1` and fewer KV heads than Q heads;
- GQA: all other already-validated cases where Q heads are an integer multiple of KV heads.

The existing `ModelConfig` validation remains authoritative for non-zero dimensions, head divisibility, even rotary head dimension, positive finite RMSNorm epsilon and RoPE theta, and supported activation policy.

## Follow-up gate

A new Hugging Face architecture or RoPE variant must receive its own parser/semantic checks and real-model parity evidence. Extending this enum alone must never be treated as support.
