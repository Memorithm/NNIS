# NNML1 seeded sampling

NNIS now has a correctness-first seeded sampling path for decoder logits. This
is an NNML1 capability surface, not a claim that sampling is device-resident or
serving-optimized.

## Public policy

`nnis_model::SamplingConfig` freezes sampling-policy semantics at
`NNIS_SAMPLING_POLICY_VERSION = 1`.

The policy fields are:

- `temperature: f32`, finite and strictly positive;
- `top_k: Option<usize>`, either absent or in `1..=vocab_size`;
- `top_p: Option<f32>`, either absent or finite in `(0, 1]`;
- `seed: u64`.

`SamplingConfig::seeded(seed)` selects temperature 1.0 with no top-k or top-p
truncation. Builder-style `with_temperature`, `with_top_k`, and `with_top_p`
methods narrow the policy explicitly.

`GenerationConfig` remains the strategy-neutral length/EOS envelope. New
`fixed` and `until_eos` constructors are the preferred names for sampling;
existing `greedy` and `greedy_until_eos` constructors remain compatible aliases
for current callers.

## Version-1 selection semantics

For one full-vocabulary `f32` logit vector:

1. reject any non-finite logit;
2. rank tokens by descending logit, with lower token ID winning exact ties;
3. apply top-k truncation when requested;
4. compute temperature-scaled softmax weights in `f64`, subtracting the maximum
   scaled logit before exponentiation;
5. apply top-p truncation in that same descending order, retaining the first
   token whose cumulative mass reaches or exceeds the requested threshold;
6. draw once from the retained mass with NNIS-owned SplitMix64 state.

Positive temperature scaling does not change rank, so selecting top-k before
computing scaled weights is equivalent to ranking the temperature-scaled
logits.

The SplitMix64 transition and constants are owned by NNIS rather than delegated
to a third-party RNG crate. This freezes RNG state evolution. Floating-point
`exp` remains a platform math operation, so cross-platform bit-identical token
streams are not claimed for adversarial probability boundaries. Exact same-seed
reproducibility is tested on equal-logit and runtime smoke cases where that
ambiguity is absent.

## Runtime path

`InferenceSession::generate_sampled(input_ids, generation, sampling)` executes:

- prompt prefill through the normal NNIS decoder;
- full logit copy to host;
- host-side version-1 sampling;
- sampled-token copy back to the device;
- normal NNIS decode for that token;
- repeat until the requested length or explicit EOS.

Every emitted token, including the final token or EOS, is executed through the
decoder before return. Session position and KV state therefore remain ready for
continuation, matching the existing greedy generation state convention.

The current implementation intentionally copies a full vocabulary logit vector
to the host after prefill and after every decoded sampled token. The last copy
is also retained for simple state semantics even when no further sample will be
drawn. These transfers are known and explicit; eliminating avoidable host
roundtrips is an NNML2 goal.

## Failure behavior

The sampling path fails closed for:

- zero vocabulary;
- invalid temperature/top-k/top-p;
- vocabulary wider than `u32::MAX` token IDs;
- non-finite logits;
- logit vector width drift;
- invalid EOS token IDs;
- prompt plus generation capacity overflow;
- any underlying CUDA/session error.

The existing fixed-length greedy path is unchanged and remains the preferred
path when device-resident top-1 generation is required.

## Qualification boundary

Unit tests freeze policy validation, top-k tie behavior, SplitMix64 sequence,
top-p retention, support restriction, and malformed-logit rejection. A GPU
smoke test builds a tiny decoder whose LM head emits equal logits, then verifies
same-seed token identity and EOS handling through `generate_sampled` when CUDA
is available.

This establishes a seeded NNML1 sampling path. It does not satisfy NNML1's full
exit criterion, which still requires multiple real model families, broader
versioned decoder/RoPE capability coverage, streaming generation, and batched
sessions.
