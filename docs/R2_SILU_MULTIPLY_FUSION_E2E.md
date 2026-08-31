# R2 SiLU-multiply fusion end-to-end ABBA gate

The isolated Thor result for `SiLU(gate) * up` is strong enough to justify a
runtime candidate, but isolated evidence cannot promote a decoder optimization.
This campaign changes exactly one runtime axis.

## A and B

- **A / parent:** E1.1 all-f32 LM-head GEMV64, all-f32 representation, separate
  SiLU and multiply launches.
- **B / candidate:** identical projection and representation plans, but the
  explicit fusion plan selects the fused SiLU-multiply block-256 kernel.

The candidate therefore removes one launch per decoder layer and the explicit
`activated` buffer write/read while leaving model weights, projection choices,
model format v1 and greedy decoding semantics unchanged.

## Physical execution

Run from a clean exact checkout on the target GPU with no competing CUDA work.
Keep MAXN and the fixed clock regime stable for the complete campaign:

```bash
cd /root/NNIS && \
test -z "$(git status --porcelain --untracked-files=no)" && \
export NNIS_BENCH_RUN_CONTEXT_ID="r2-silu-e2e-$(date -u +%Y%m%dT%H%M%SZ)" && \
export NNIS_BENCH_ENVIRONMENT_LABEL="native-thor" && \
python3 tools/run_r2_silu_fusion_e2e_abba.py \
  --model /tmp/smollm2-135m/model \
  --device 0 \
  --decode-steps 32 \
  --rounds 2 \
  --warmups 2 \
  --iterations 5
```

The driver executes `A B B A` in each round and prints the path to
`summary.json`. Raw reports remain beside the summary.

## Fail-closed gates

The campaign rejects evidence if any of these conditions fails:

- tracked worktree is clean;
- every report identifies the exact checkout HEAD;
- one explicit run-context id is present throughout the campaign;
- GPU identity, UUID, CUDA driver, NVRTC, host kernel and Jetson power/clock
  fingerprint remain compatible;
- source fixture provenance is the pinned SmolLM2-135M checkpoint;
- input IDs, decode length, warmups and iterations are unchanged;
- both A and B use E1.1 all-f32 LM-head GEMV64;
- both A and B keep the all-f32 representation plan;
- A serializes the baseline separate fusion plan;
- B serializes the fused block-256 fusion plan;
- the qualified greedy prefix passes;
- every generated token is identical between all A and B reports;
- generation and request medians are positive.

The generic `compare_smollm2_reports` comparator deliberately rejects reports
with different fusion plans. This dedicated driver is the only comparator for
this intentional fusion-axis experiment.

## Interpretation

The summary is evidence, not an automatic promotion decision. A MinLatency
promotion requires an end-to-end advantage credible relative to ABBA variation,
with an identical greedy trajectory. The isolated 33.04% sequence reduction must
not be extrapolated into decoder throughput.

If end-to-end evidence is not credible, keep the fused primitive and isolated
result as rejected evidence and leave the runtime default unchanged. If it is
credible, record the exact physical campaign in the sovereignty roadmap before
any default or promoted-plan decision.
