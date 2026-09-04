# NNML1 sampled streaming

NNIS exposes `InferenceSession::generate_sampled_streaming` as the first token-by-token generation surface for the NNML1 broader-decoder-runtime program.

## Contract

The API uses the same `GenerationConfig` envelope and `SamplingConfig` policy as `generate_sampled`. A caller receives each emitted token through a callback and returns `GenerationStreamControl::Continue` or `GenerationStreamControl::Stop`.

The callback runs only after the emitted token has been executed through the decoder. Therefore, after EOS, caller-requested stop, or normal completion, `InferenceSession::position()` and the KV cache include every token that was delivered to the caller. The same session can be continued with `decode_one` without replaying the streamed token.

EOS remains authoritative: when the configured EOS token is emitted, the callback observes that token once and generation then stops. Caller-requested stop is graceful and occurs after the current token is part of session state.

## Reproducibility

Sampling uses the same NNIS-owned sampling-policy version and SplitMix64 sequence as `generate_sampled`. Streaming does not introduce a second RNG or a second candidate-ranking rule. With the same prompt, generation envelope and sampling configuration, the emitted prefix follows the same deterministic sampling sequence until the callback stops it.

## Host/device boundary

This is an NNML1 correctness and API surface, not an NNML2 device-residency claim. The current sampled path materializes full vocabulary logits on the host, performs top-k/top-p/temperature sampling on the host, and sends the selected token back to CUDA for decoder execution. Streaming does not remove those transfers.

The existing fixed-length greedy path remains the device-resident path and is unchanged by this feature.

## Failure and capacity behavior

The streaming path fails closed for the same malformed sampling policy, non-finite logits, invalid EOS token, invalid prompt token, and vocabulary constraints as sampled generation. It also checks `prompt_length + max_new_tokens` against session capacity before prefill.

A callback cannot inject a token or alter model state. Its only control is continue or graceful stop.

## Qualification boundary

The CUDA-optional integration tests cover:

- frozen seeded token delivery;
- graceful callback stop after two emitted tokens;
- continuation-ready session state after callback stop;
- EOS delivery before termination.

This feature does not claim dynamic batching, request scheduling, backpressure, cancellation across concurrent requests, network streaming, device-resident sampling, or serving-grade latency/throughput. Those remain NNML2/NNML4 work.
