# R2 CUDA launch/submission control

R2 requires explicit evidence about host-side CUDA launch/submission cost before
NNIS introduces new decode fusions or changes kernel-selection policy.

`launch_submission_control` is deliberately a control experiment rather than a
model benchmark. It repeatedly enqueues the existing one-element
`nnis_scale_f32` kernel so useful GPU work is minimized while the same NNIS
Driver-API/JIT launch machinery is exercised.

The default launch counts are `1, 8, 32, 128, 211, 512`. The value `211` is
included because the existing SmolLM2 projection-stream diagnostic contains
`30 * 7 + 1 = 211` linear-projection launches per token. The control does **not**
claim that 211 tiny scale launches reproduce those projection kernels or the
complete decoder.

## Physical run

Run from a clean exact NNIS checkout under one explicit benchmark context:

```bash
cd /root/NNIS && \
test -z "$(git status --porcelain --untracked-files=no)" && \
export NNIS_BENCH_RUN_CONTEXT_ID="r2-launch-$(date -u +%Y%m%dT%H%M%SZ)" && \
export NNIS_BENCH_ENVIRONMENT_LABEL="native-thor" && \
export NNIS_PROFILE_WARMUPS=20 && \
export NNIS_PROFILE_ITERATIONS=100 && \
cargo run --locked --release -p nnis-bench --example launch_submission_control \
  > /tmp/nnis-r2-launch-submission.json
```

An alternate sequence may be requested with comma-separated positive unique
counts, for example:

```bash
NNIS_LAUNCH_COUNTS=1,16,64,211,256,512 \
cargo run --locked --release -p nnis-bench --example launch_submission_control
```

## Measurements

For every launch count the report contains two distinct distributions:

- `gpu_timeline`: CUDA-event time covering the complete queued sequence;
- `host_submission`: host monotonic-clock time spent issuing only the NNIS
  enqueue calls. The required stream synchronization occurs after this timed
  interval and is therefore excluded from the sample.

The report also computes
`host_submission_sequence_average_us_per_launch`, which is only the measured
sequence median divided by its launch count. It is not a separately measured
per-launch latency and must not be extrapolated into an end-to-end decoder
speedup.

All CUDA-event reports inside one process are required to share a compatible
environment fingerprint and exact git commit. A clean tracked worktree and
explicit `NNIS_BENCH_RUN_CONTEXT_ID` are mandatory.

## Interpretation boundary

This experiment can establish whether host submission is large enough to merit
further investigation under one measured environment. It cannot establish that
submission overhead dominates NNIS, cannot rank decoder kernels, and cannot
justify fusion by itself.

A fusion candidate still needs a concrete removed-work/data-movement hypothesis,
correctness validation, isolated measurement when useful, and a
fingerprint-compatible end-to-end verification before promotion.
