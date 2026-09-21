# SampledSessionBatch CLI v1

Thin fail-closed CLI over the existing NNML1 `SampledSessionBatch` /
`SampledBatchRequest` library surface (already re-exported by the `nnis`
facade from PR #137).

## Source

- Library: `nnis_model::session_batch` (`SampledBatchRequest`,
  `SampledSessionBatch`, `Model::new_sampled_session_batch`)
- Facade: `nnis::{SampledBatchRequest, SampledSessionBatch}` (unchanged by this
  CLI slice)
- Sampling policy: `SamplingConfig` / `NNIS_SAMPLING_POLICY_VERSION`

## CLI surface

```bash
cargo run --release -p nnis-cli --bin nnis -- generate-batch \
  --model DIR --tokenizer FILE \
  --prompt TEXT --seed U64 \
  [--prompt TEXT --seed U64 ...] \
  [--device N] [--max-new-tokens N] \
  [--temperature F] [--top-k N] [--top-p F] [--json]
```

Fail-closed parsing rules:

- at least one `--prompt` is required;
- the number of `--seed` values must equal the number of `--prompt` values
  (silent seed reuse / derivation is refused);
- shared optional `--temperature` / `--top-k` / `--top-p` apply to every
  request;
- `--json` emits a versioned CLI envelope (`schema_version` independent of
  `NNIS_SAMPLING_POLICY_VERSION`).

Runtime behavior:

- tokenizes each prompt, loads one model, allocates
  `SampledSessionBatch` with `session_count == prompt_count`;
- builds one `SampledBatchRequest` per prompt with an independent
  `SamplingConfig` seed;
- calls `generate_sampled` (deterministic index order; per-item failure
  domains as documented on the library API);
- human text prints indexed results; any item error yields non-zero exit after
  printing remaining outcomes.

`nnis generate` remains greedy-by-default; this command is inherently sampled
because the batch API is `generate_sampled`.

## Explicit non-claims

This CLI does **not** claim:

- fused batched kernels;
- overlapping / concurrent CUDA streams;
- concurrent multi-session overlap throughput;
- device-resident sampling;
- serving performance, latency, or quality;
- physical Thor parity or residency.

See `docs/exec-plans/active/NNML1_GENERATE_BATCH_CLI.md`.
