# SmolLM2-135M validation evidence

This document records measured NNIS evidence for the pinned trained-model probe introduced with the SmolLM2-135M reference harness. It is an evidence record, not a performance claim and not a declaration of bitwise equivalence with PyTorch/Transformers.

## Pinned reference

- upstream repository: `HuggingFaceTB/SmolLM2-135M`
- upstream revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- `model.safetensors` SHA-256: `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- source weights: BF16
- NNIS execution weights: persisted BF16 values widened to f32
- trusted oracle: Transformers `4.40.1`, CPU, f32 execution
- prompt: `Gravity is`
- input IDs: `[22007, 6463, 314]`
- reference greedy IDs for the two-step probe: `[260, 3075]`

## Physical CUDA environment

The diagnostic evidence below was measured on an NVIDIA Thor system with compute capability 11.0 and CUDA 13.0. The repository's hosted CI does not reproduce this GPU environment.

## End-to-end observation

The original strict cross-backend comparison used harness defaults `atol=1e-4`, `rtol=1e-3`. Those thresholds were explicitly draft defaults, not validated SmolLM2 acceptance bounds. On physical Thor they fail for the pinned prefill logits:

- prefill max absolute difference: `8.19277763e-2`
- prefill RMS difference: `2.55335605e-2`
- logits outside the draft tolerance: `40315 / 49152`

A diagnostic run that disabled numeric rejection, solely to observe the subsequent trajectory, measured:

- prefill: max abs `8.19277763e-2`, RMS `2.55335605e-2`
- decode step 0: max abs `1.62665844e-1`, RMS `4.91878280e-2`
- decode step 1: max abs `1.30074501e-1`, RMS `5.87158042e-2`
- NNIS greedy IDs: `[260, 3075]`

The large permissive tolerance used for that diagnostic is not an acceptance threshold and must not be treated as qualification evidence by itself. The useful semantic observation is only that the pinned two-token greedy trajectory matched the trusted reference.

## Numerical localization

Subsequent diagnostics localized the cross-backend difference instead of relaxing the threshold.

### Final LM head

Using the exact Transformers final hidden state as input to the persisted NNIS LM-head weights, a sequential ordered f32 FMA oracle differed from the PyTorch CPU linear result by:

- max abs `6.105661392e-2`
- RMS `2.374590667e-2`

This demonstrates that LM-head reduction order alone can account for most of the prefill-logit RMS difference. It does not, by itself, account for the full end-to-end difference because the NNIS hidden state also differs slightly before the LM head.

### Decoder hidden state

Comparing the NNIS final normalized hidden state against Transformers for the final prompt token measured:

- max abs `6.91680908e-2`
- RMS `6.99631732e-3`

Layerwise tracing showed gradual f32 divergence through the decoder with a visible amplification around layer 24. The layer-24 input RMS difference was `3.40535357e-2`; after the complete layer it was `4.96102529e-2`.

### Layer-24 attention

Layer-24 tracing measured:

- input RMS difference: `3.40535357e-2`
- input RMSNorm RMS difference: `8.01257685e-4`
- Q projection RMS difference: `3.74305327e-3`
- K projection RMS difference: `3.55185549e-3`
- V projection RMS difference: `6.04696755e-3`
- attention output before `O_proj`: RMS difference `1.81436326e-3`
- attention output after `O_proj`: RMS difference `3.03349858e-2`

The `O_proj` increase was then decomposed using the exact same NNIS input and weights:

- NNIS GPU `O_proj` versus ordered same-input f32 oracle: max abs `0`, RMS `0`, bitwise mismatches `0 / 576`
- ordered projection of NNIS input versus ordered projection of reference input: max abs `6.66961670e-1`, RMS `3.08260058e-2`
- ordered projection of the reference input versus Transformers `O_proj`: max abs `1.40838623e-2`, RMS `1.98501783e-3`

For this probe the NNIS `O_proj` GEMM therefore reproduces its ordered same-input oracle bit-for-bit. The large post-projection difference is primarily amplification of a much smaller difference already present at the attention output, not evidence of an `O_proj` kernel error.

### Same-input GQA / KV-cache / attention oracle

The NNIS cached-attention CUDA result was compared with a host oracle implementing the same GQA mapping, cache layout, ordered f32 dot products and online softmax while consuming the exact same NNIS Q-RoPE and K/V cache values.

Measured result:

- GPU versus same-input host oracle max abs: `8.94069672e-8`
- GPU versus same-input host oracle RMS: `8.78021828e-9`
- bitwise mismatches: `58 / 576`
- same-input host oracle versus Transformers attention RMS: `1.81436478e-3`

The remaining GPU/host difference is at near-f32-rounding scale for this diagnostic. These measurements provide evidence that the implemented GQA head mapping, device KV-cache indexing and cached online-softmax attention are behaving consistently with the NNIS same-input oracle for this pinned three-token probe. They do not establish correctness for every sequence length, model, GPU or numerical environment.

## Qualification policy

Two distinct questions must not be conflated:

1. **NNIS internal correctness:** kernels should be checked against same-input host or high-precision oracles where such an oracle is meaningful. Exact/near-exact checks are appropriate here.
2. **Cross-backend model agreement:** PyTorch CPU and NNIS CUDA can use materially different f32 reduction orders and transcendental implementations. A small universal elementwise `atol`/`rtol` must not be asserted without evidence.

The SmolLM2 comparator therefore keeps strict elementwise checking as the default. A separate explicit report mode may be used to collect cross-backend metrics while still failing closed on non-finite outputs and greedy-token mismatches. Report mode is diagnostic/semantic evidence and does not claim numeric equivalence.

No relaxed SmolLM2 logit tolerance is declared by this evidence record. A future numeric acceptance envelope should be based on broader measurements across prompts, sequence lengths, decode steps, GPU architectures and trusted reference implementations before it becomes a release gate.
