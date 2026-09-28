# NNIS execution plans

There is currently **no active execution plan** in this directory.

The current non-physical software selection, the ordered portable (V888)
phases, and the open physical gates are recorded in the `.agent/` ledger on the
`agent/sovereignty-roadmap` branch (`CURRENT_EXECUTION_STATUS.yaml`,
`NNIS_SOVEREIGNTY_ROADMAP.yaml`, `PORTABLE_RUNTIME_PROGRESS.yaml`). Each merged
slice is recorded there with its PR and merge commit.

## Completed plans

All plans below are retained for chronology only. They describe software that
has merged; none of them records physical Thor, parity, residency or
performance results beyond what the linked PRs state.

| Plan | Outcome |
|---|---|
| [GLIMMER_CONTINUATION](completed/GLIMMER_CONTINUATION.md) | Historical bootstrap continuation log (CUDA foundation, early Thor measurements as recorded there). |
| [NNML1_STREAMING_CONTINUATION](completed/NNML1_STREAMING_CONTINUATION.md) | Seeded sampling, streaming and sampled batch library surface (PRs #104–#106). |
| [NNML1_CLI_SAMPLED_STREAMING](completed/NNML1_CLI_SAMPLED_STREAMING.md) | `--sample` / `--stream` CLI surface (PR #137, `faab2ea`). |
| [NNML1_KV_CACHE_TELEMETRY](completed/NNML1_KV_CACHE_TELEMETRY.md) | Session KV cache telemetry on the facade (PR #138, `8d19ca6`). |
| [NNML2_KV_LOGICAL_CACHE_TELEMETRY](completed/NNML2_KV_LOGICAL_CACHE_TELEMETRY.md) | Versioned `KvCacheTelemetry` schema v1 (PR #154, `c58ec1c`). |
| [NNML2_NVML_PROCESS_MEMORY_CLI](completed/NNML2_NVML_PROCESS_MEMORY_CLI.md) | Read-only `nnis nvml-process-memory` CLI (PR #155, `a9a17ef`). |
| [NNML2_INT4_FACADE_REEXPORT](completed/NNML2_INT4_FACADE_REEXPORT.md) | INT4 projection contracts on the `nnis` facade (PR #156, `2e99408`). |
| [NNML1_GENERATE_BATCH_CLI](completed/NNML1_GENERATE_BATCH_CLI.md) | Fail-closed `nnis generate-batch` CLI (PR #157, `1f8aeaa`; rustfmt fix PR #168, `cd7aa97`). |

`PROMPT_GLIMMER_CONTINUE.md` at the repository root is a historical prompt and
still names the original `docs/exec-plans/active/` path; the log it refers to
now lives under `completed/`.

New plans, if any, go under `docs/exec-plans/active/` and move to `completed/`
with a status line naming the merged PR and commit.
