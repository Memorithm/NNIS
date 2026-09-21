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

Next non-physical software selection:
`docs/exec-plans/active/NNML2_KV_LOGICAL_CACHE_TELEMETRY.md`.

## Claim boundary (unchanged)

CLI opt-in sampled/streaming does not claim model quality, serving performance,
device-resident sampling, physical Thor parity, or general model-family support.
Physical P0/P1 gates remain open.
