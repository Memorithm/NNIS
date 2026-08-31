# R2 parallel-value cached-attention end-to-end gate

This gate tests whether the physically qualified cached-attention primitive
translates into a credible SmolLM2 end-to-end MinLatency improvement.

It does not promote automatically.

## Compared plans

A and B are identical except for the explicit `F32AttentionPlan` axis.

A, parent:

- projection: promoted E1.1 all-f32 LM-head GEMV64;
- representation: all-f32;
- fusion: baseline separate SiLU/multiply;
- attention: historical serial single-thread cached decode.

B, candidate:

- projection: identical E1.1 all-f32 LM-head GEMV64;
- representation: identical all-f32;
- fusion: identical baseline;
- attention: parallel-value cached decode with 64 threads/query-head.

The default runtime remains A unless a caller explicitly supplies the candidate
attention plan.

## Physical prerequisites

Run on the same physical Thor regime used for qualifying R2 evidence:

- clean exact tracked checkout;
- pinned `HuggingFaceTB/SmolLM2-135M` fixture and provenance;
- prompt IDs `[22007, 6463, 314]`;
- 32 greedy decode steps by default;
- explicit run-context ID;
- stable environment fingerprint;
- no competing CUDA workload;
- same MAXN/fixed-clock regime across all A/B runs.

The candidate has already passed isolated bitwise-equality gates at KV rows
`1, 2, 4, 8, 16, 24, 35` and was faster at every tested length. This is only the
entry condition for end-to-end measurement.

## ABBA driver

```bash
export NNIS_BENCH_RUN_CONTEXT_ID="r2-attention-e2e-$(date -u +%Y%m%dT%H%M%SZ)"
export NNIS_BENCH_ENVIRONMENT_LABEL="native-thor"
python3 tools/run_r2_attention_parallel_value_e2e_abba.py \
  --model /tmp/smollm2-135m/model \
  --device 0 \
  --decode-steps 32 \
  --rounds 2 \
  --warmups 2 \
  --iterations 5
```

The default order per round is `A, B, B, A`.

The driver fails closed if any of the following drifts:

- exact Git HEAD or tracked worktree cleanliness;
- pinned model provenance;
- workload or decode length;
- projection, representation or fusion plan;
- expected attention plan for A/B;
- environment fingerprint;
- qualified greedy prefix;
- complete generated-token trajectory.

It writes every raw SmolLM2 report plus `summary.json` under
`artifacts/r2-attention-e2e-abba-<timestamp>/`.

## Promotion rule

A physical campaign may justify `MinLatency` promotion only if:

1. the fingerprint remains compatible for the entire campaign;
2. the exact greedy trajectory is unchanged;
3. the candidate wins end-to-end by a margin credible relative to the observed
   ABBA variation;
4. the result is recorded in the off-main sovereignty roadmap before any
   performance-promotion merge decision.

A positive aggregate median alone is not sufficient. The summary reports both
`all_candidate_generation_medians_below_all_parent_medians` and
`generation_separation_min_parent_minus_max_candidate_ms` to expose overlap.

No memory saving is claimed by this kernel substitution. The launch count is
unchanged; this candidate changes intra-block parallelism only.
