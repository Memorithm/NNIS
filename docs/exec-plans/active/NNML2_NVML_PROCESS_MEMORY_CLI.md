# NNML2 NVML process-memory CLI surface

Status: **completed** (merged as PR #155 on main `a9a17ef`).

## Delivered software

- Fail-closed, read-only `nnis nvml-process-memory [--device N] [--json]` over
  `NvmlProcessMemorySnapshotV1` / `current_process_gpu_memory` (facade already
  re-exported from #138).
- Human text by default; `--json` emits versioned JSON with `schema_version`.
- CPU-only parse/help/formatter tests; docs under
  `docs/memory/nvml-process-memory-v1.md` + README note.

## Follow-on

Next non-physical software selection:
`docs/exec-plans/active/NNML1_GENERATE_BATCH_CLI.md`
(SampledSessionBatch thin CLI; avoids conflict with in-flight #156 INT4 facade).

## Claim boundary (unchanged)

Software observability CLI only. Does not claim physical residency, weight-only
attribution, allocator/page-table overhead, serving performance, or Thor parity.
