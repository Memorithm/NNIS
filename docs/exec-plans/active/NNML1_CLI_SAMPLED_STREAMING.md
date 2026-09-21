# NNML1 CLI sampled / streaming surface

Status: **completed** (merged as PR #137 on main `faab2ea`).

## Delivered software

- `nnis` facade re-exports `SamplingConfig`, `NNIS_SAMPLING_POLICY_VERSION`,
  `GenerationStreamControl`, `SampledBatchRequest`, and `SampledSessionBatch`.
- Opt-in `--sample` / `--stream` flags on `nnis generate` and `nnis-hf generate`
  (greedy default preserved; `--sample` requires `--seed`; `--stream` only with
  `--sample`).
- README generate docs updated; CLI parse tests without CUDA.

## Follow-on

Library + single-session CLI shipped in #137. Later non-physical software
selections included NVML process-memory CLI (#155). Next selected:
`docs/exec-plans/active/NNML1_GENERATE_BATCH_CLI.md` (SampledSessionBatch thin
CLI).

## Claim boundary (unchanged)

CLI opt-in sampled/streaming does not claim model quality, serving performance,
device-resident sampling, physical Thor parity, or general model-family support.
Physical P0/P1 gates remain open.
