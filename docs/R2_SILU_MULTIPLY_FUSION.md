# R2 SiLU-multiply fusion candidate

R2 now has physical evidence that CUDA launch/submission overhead is measurable on
Jetson AGX Thor, but that control does not justify arbitrary decoder fusion. This
candidate targets a concrete repeated sequence in the current SmolLM2 path:

1. `activated = SiLU(gate)`;
2. `gated = activated * up`.

The candidate computes `gated = SiLU(gate) * up` in one kernel.

## Structural hypothesis

For the pinned SmolLM2-135M shape (`intermediate_size = 1536`, 30 layers), an
integrated candidate would remove exactly:

- one CUDA launch per decoder layer, or 30 launches per token;
- one explicit write and one explicit read of the 1536-element `activated`
  intermediate per layer;
- 12,288 logical intermediate bytes per layer and 368,640 logical intermediate
  bytes per token (`1536 * 4 bytes * 2 * 30`).

The byte count is logical buffer traffic, not measured DRAM traffic. Cache behavior
may change the physical memory traffic.

## Current scope

This PR only adds the fused primitive and an isolated benchmark. It does **not**:

- change the decoder execution path;
- add or select a runtime fusion plan;
- remove the existing `activated` workspace allocation;
- claim an end-to-end speedup;
- change weight representation or model format v1.

## Physical isolated gate

Run from a clean exact checkout on the target GPU with no competing CUDA work:

```bash
cd /root/NNIS && \
test -z "$(git status --porcelain --untracked-files=no)" && \
export NNIS_BENCH_RUN_CONTEXT_ID="r2-silu-fusion-$(date -u +%Y%m%dT%H%M%SZ)" && \
export NNIS_BENCH_ENVIRONMENT_LABEL="native-thor" && \
export NNIS_PROFILE_WARMUPS=20 && \
export NNIS_PROFILE_ITERATIONS=100 && \
cargo run --locked --release -p nnis-bench --example silu_multiply_fusion
```

The benchmark compares the existing two-launch sequence against the fused
one-launch candidate on 1536 elements. It fails closed unless:

- both reports come from the same exact clean git commit;
- the environment fingerprints are compatible under one explicit run context;
- all 1536 output f32 values are bitwise identical;
- both measured medians are positive.

## Promotion boundary

Even a strong isolated result does not promote this candidate. If the physical
result is worth pursuing, the next step is an explicit opt-in runtime fusion axis
followed by fingerprint-compatible SmolLM2 end-to-end A/B/B/A verification with
an unchanged greedy trajectory. The candidate is rejected if end-to-end evidence
does not beat its parent by a credible margin relative to observed variance.
