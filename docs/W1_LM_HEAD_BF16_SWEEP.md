# W1 LM-head BF16 physical sweep

W1 is a candidate-only representation experiment for the pinned
`HuggingFaceTB/SmolLM2-135M` checkpoint. It keeps f32 activations and outputs,
changes only the physical LM-head weight storage from f32 to BF16, and widens
BF16 values to f32 inside the candidate GEMV before the existing increasing-K
`fmaf` accumulation.

This document is an execution gate, not a promotion record. `nnis-model`, model
format v1, generation semantics, and the promoted E1.1 LM-head GEMV64 runtime
remain unchanged by the sweep.

## Preconditions

Run from a clean exact NNIS checkout on the physical target GPU. The pinned
SmolLM2 fixture must already exist and must retain the provenance enforced by
`smollm2_lm_head_weight_representation`.

On Jetson AGX Thor, select the intended power/clock regime before starting the
campaign. The benchmark fingerprint records the resulting state and the sweep
driver rejects drift across runs.

## One-command campaign

Use one explicit run-context id for the complete sweep:

```bash
cd /root/NNIS && \
test -z "$(git status --porcelain --untracked-files=no)" && \
export NNIS_BENCH_RUN_CONTEXT_ID="w1-thor-$(date -u +%Y%m%dT%H%M%SZ)" && \
export NNIS_BENCH_ENVIRONMENT_LABEL="native-thor" && \
python3 tools/run_w1_lm_head_bf16_sweep.py \
  --model /tmp/smollm2-135m/model \
  --device 0 \
  --blocks 32 64 128 256 512 \
  --rounds 2 \
  --warmups 20 \
  --iterations 100
```

The command prints the path to `summary.json`. Raw per-round/per-block reports
are kept beside it.

## Fail-closed checks

The driver rejects the campaign when any of the following occurs:

- tracked worktree is dirty;
- a subprocess does not report the exact checkout HEAD;
- a report omits the explicit run-context id;
- GPU, driver, NVRTC, host kernel, Jetson power/clock, or other fingerprint
  evidence drifts across sweep runs;
- checkpoint provenance differs from the pinned SmolLM2 source;
- LM-head storage accounting differs from 28,311,552 elements,
  113,246,208 f32 bytes, or 56,623,104 BF16 bytes;
- any of the 49,152 candidate logits differs bit-for-bit from the f32 GEMV64
  reference;
- timing output is missing or invalid.

Odd rounds traverse block sizes forward; even rounds traverse them in reverse to
reduce simple monotonic order bias. The summary selects the lowest isolated
candidate median only as `isolated_winner`.

## Promotion rule

Do **not** promote a representation plan, runtime default, model-format change,
or speed claim from this sweep alone.

If the isolated winner is worth pursuing, the next change must keep kernel
selection and weight representation as separate explicit plan axes, preserve
model format v1, and then pass fingerprint-compatible end-to-end AB/ABBA
verification with unchanged greedy semantics before any latency promotion.

For a memory objective, the exact LM-head storage saving is 56,623,104 bytes
(50% for that one tensor copy). This is not a whole-model memory claim and does
not imply a latency improvement.
