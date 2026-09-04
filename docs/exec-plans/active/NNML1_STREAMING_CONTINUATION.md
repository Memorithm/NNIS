# NNML1 streaming continuation

This branch advances the NNML1 broader-decoder-runtime program with a narrowly scoped sampled streaming surface.

## Current slice

- add token-by-token delivery for the existing reproducible sampled generation path;
- preserve the existing sampling policy and RNG sequence;
- make callback stop graceful only after the emitted token has been executed;
- keep session position and KV state continuation-ready;
- preserve the existing fixed-length greedy device-resident path unchanged.

## Required merge gates

- `cargo fmt --all -- --check`;
- `cargo check --workspace --all-targets --locked`;
- strict Clippy under the repository workflow;
- workspace tests;
- Rust 1.77 MSRV gate;
- CUDA-optional sampled streaming tests when hardware is present.

## Boundaries

This slice does not claim dynamic batching, concurrent request scheduling, network transport streaming, backpressure, device-resident sampling, serving-grade performance, or multiple-model-family qualification.

After merge, the next non-physical NNML1 slice should address batched-session contracts only if the existing single-session ownership invariants can be preserved explicitly. Physical P0/P1 gates for real Safetensors qualification and the `fused_mlp + parallel-score` composition remain separate and must not be inferred from this work.
