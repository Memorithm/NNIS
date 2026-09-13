# NNML1 CLI sampled / streaming surface

Status: **active** (selected next non-physical software slice).

## Goal

Expose the already-merged NNML1 library contracts through the public `nnis`
facade and opt-in CLI flags without changing default greedy behavior.

## Scope

- Re-export `SamplingConfig`, `NNIS_SAMPLING_POLICY_VERSION`,
  `GenerationStreamControl`, `SampledBatchRequest`, and `SampledSessionBatch`
  from the `nnis` facade (`model` module and top-level `pub use`).
- Extend `nnis generate` and `nnis-hf generate`:
  - default remains greedy;
  - `--sample` requires `--seed U64` (fail-closed);
  - optional `--temperature` / `--top-k` / `--top-p` builders with `--sample`;
  - `--stream` only with `--sample`, printing each decoded token piece as emitted;
  - fail closed on invalid flag combinations;
  - parse tests without CUDA.
- Update README generate docs briefly; preserve `nnis.hf-generation@1.0.0`
  greedy-default process identity (no schema version bump required).
- Mark `NNML1_STREAMING_CONTINUATION.md` completed after this merge lands.

## Required merge gates

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets --locked`
- strict Clippy under the repository workflow
- workspace tests (including CLI parse tests without CUDA)
- Rust 1.77 MSRV gate

## Claim boundary

This slice does not claim model quality, serving latency/throughput, device-resident
sampling, physical Thor parity, or general model-family support. Physical P0/P1
gates remain open and must not be inferred from CLI flag availability.
