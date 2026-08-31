# W1 LM-head BF16 end-to-end ABBA gate

W1 has isolated physical evidence for storing only the pinned SmolLM2-135M tied
LM-head copy as BF16 while keeping f32 activations, accumulation and logits. The
isolated sweep is not sufficient for runtime promotion.

This gate compares the current promoted E1.1 parent against the W1 candidate on
the complete NNIS greedy generation path:

- **A / parent:** all resident weights f32; E1.1 LM-head GEMV64;
- **B / candidate:** only the resident LM-head is BF16; LM-head GEMV32;
- every other projection remains on the same f32 GEMM path;
- model format v1 and the fixture files remain unchanged;
- the representation choice and kernel choice are serialized as separate plans.

The BF16 loader is fail-closed: every f32 LM-head value must already be exactly
BF16-representable. NNIS does not round arbitrary f32 model weights merely to run
this candidate.

## Physical execution

Run from a clean exact checkout on the target GPU, with no competing CUDA work,
and keep the intended power/clock regime stable for the complete campaign.
Use one explicit run-context id:

```bash
cd /root/NNIS && \
test -z "$(git status --porcelain --untracked-files=no)" && \
export NNIS_BENCH_RUN_CONTEXT_ID="w1-e2e-thor-$(date -u +%Y%m%dT%H%M%SZ)" && \
export NNIS_BENCH_ENVIRONMENT_LABEL="native-thor" && \
python3 tools/run_w1_e2e_abba.py \
  --model /tmp/smollm2-135m/model \
  --device 0 \
  --decode-steps 32 \
  --rounds 2 \
  --warmups 2 \
  --iterations 5
```

The driver executes `A B B A` in every round and prints the path to
`summary.json`. Raw reports remain beside the summary.

## Fail-closed gates

The campaign rejects evidence when any of the following occurs:

- the tracked worktree is dirty;
- a report does not identify the exact checkout HEAD;
- the explicit run-context id is absent or changes;
- GPU, UUID, driver, NVRTC, host kernel, Jetson power/clock state or other
  environment fingerprint evidence changes during ABBA;
- checkpoint provenance is not the pinned SmolLM2-135M fixture;
- workload parameters change across runs;
- A does not serialize the E1.1 all-f32/GEMV64 plans;
- B does not serialize the W1 LM-head-BF16/GEMV32 plans;
- the qualified greedy prefix fails;
- any generated token differs between A and B;
- timing medians are missing or non-positive.

The generic `compare_smollm2_reports` tool deliberately rejects reports whose
representation plans differ. W1 uses this dedicated driver because the
representation difference is the explicit experimental variable, not an
accidental benchmark mismatch.

## Interpretation

The ABBA summary is evidence, not an automatic promotion decision.

For **MinLatency**, the candidate must beat E1.1 end to end by a margin credible
relative to the observed ABBA variation. For **MinMemory**, exact storage savings
may justify an explicit candidate even without a latency win, but any latency
trade-off must remain visible. A balanced objective requires an explicit policy.

The exact W1 storage accounting applies only to the tied LM-head copy:

- f32: 113,246,208 bytes;
- BF16: 56,623,104 bytes;
- saved: 56,623,104 bytes (50%).

These values are tensor-storage accounting, not a whole-model CUDA memory claim.
Observed CUDA free-memory deltas are reported separately and must not be
substituted for the exact tensor accounting.
