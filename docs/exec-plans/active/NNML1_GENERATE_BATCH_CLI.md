# NNML1 SampledSessionBatch generate-batch CLI

Status: **active** (selected next non-physical software slice after PR #155
NVML process-memory CLI on main `a9a17ef`; deliberately avoids conflict with
in-flight PR #156 INT4 facade re-exports by not touching `crates/nnis/src/lib.rs`).

## Goal

Expose the already-merged `SampledSessionBatch` / `SampledBatchRequest` APIs
(PR #106 library; facade publicity via PR #137) through a fail-closed
`nnis generate-batch` CLI. No facade churn. No physical Thor execution and no
fake campaign results.

## Scope

- Add `nnis generate-batch --model DIR --tokenizer FILE --prompt TEXT
  [--prompt TEXT ...] --seed U64 [--seed U64 ...] [--device N]
  [--max-new-tokens N] [--temperature F] [--top-k N] [--top-p F] [--json]`.
- Fail-closed: require one `--seed` per `--prompt` (refuse silent seed reuse);
  map library/CUDA/tokenize errors to non-zero exit without panicking.
- Keep `nnis generate` greedy-by-default unchanged.
- CPU-only parse/help/formatter tests (no GPU required).
- Docs under `docs/memory/sampled-session-batch-cli-v1.md` + brief README note.
- Mark `NNML2_NVML_PROCESS_MEMORY_CLI.md` completed (PR #155) if still active.

## Required merge gates

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets --locked`
- strict Clippy under the repository workflow
- workspace tests (including host-only CLI parse/format tests)
- Rust 1.77 MSRV gate

## Claim boundary

Software CLI only. Does **not** claim fused batched kernels, overlapping CUDA
streams, concurrent multi-session overlap, device-resident sampling, serving
performance, model quality, or physical Thor parity. Host-orchestrated
independent sessions in deterministic index order only.
