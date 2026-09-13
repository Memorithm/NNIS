# NNML1 streaming continuation

Status: **completed** (library surface merged; CLI follow-on selected).

## Merged software

- PR #104 — reproducible seeded sampling (`InferenceSession::generate_sampled`, `SamplingConfig`)
- PR #105 — sampled token streaming (`generate_sampled_streaming`, `GenerationStreamControl`)
- PR #106 — sampled multi-session batch (`SampledSessionBatch`)

Active follow-on software slice: `docs/exec-plans/active/NNML1_CLI_SAMPLED_STREAMING.md`
(`NNML1_CLI_SAMPLED_STREAMING_SURFACE`) — expose sampling/streaming through the
`nnis` facade and opt-in CLI flags. That slice does not reopen this library plan.

## Boundaries (unchanged)

This work does not claim dynamic batching as a scheduler, concurrent request
serving, network transport streaming, backpressure, device-resident sampling,
serving-grade performance, physical Thor parity, or multiple-model-family
qualification. Physical P0/P1 gates remain separate.
