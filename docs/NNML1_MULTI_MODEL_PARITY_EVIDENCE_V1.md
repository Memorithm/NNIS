# NNML1 multi-model parity evidence v1

This contract records reference-comparison evidence for exact decoder checkpoints without turning an individual checkpoint result into a model-family claim.

## Evidence levels

`generation_trajectory` means that the NNIS execution produced the exact trusted greedy token trajectory for every qualifying observation in the record. It is semantic evidence only. It does not assert numerical equivalence of logits.

`logit_and_generation` additionally requires strict logit comparison against the trusted reference runtime. The record is valid only when every compared stage has zero values outside the supplied `atol + rtol * |reference|` tolerance and zero non-finite NNIS logits. The supplied tolerances remain part of the evidence and are not universal model tolerances.

Both levels bind the exact checkpoint specification, source revision and model SHA-256, tokenizer SHA-256, reference runtime/version, clean NNIS Git commit, execution backend, and source evidence artifact. Neither level authorizes runtime promotion or general model-family support.

## Same-head multi-model suite

A `nnis-nnml1-multi-model-parity-suite-v1` contains at least two distinct exact checkpoint records. Every record must have been produced from the same exact NNIS Git commit. Mixing records from different commits fails closed.

Version 1 recognizes only the exact checkpoint identities already registered by NNIS:

- `smollm2-135m-bf16`
- `tinyllama-1.1b-chat-v1.0-bf16`

Adding another checkpoint requires an explicit versioned checkpoint/specification audit. Passing these two checkpoints does not by itself establish general Llama-family support.

## TinyLlama producer

`tools/run_tinyllama_massive_campaign.py` retains the existing physical massive ABBA campaign. Its consensus now recounts every reference/candidate ABBA case observation and requires all observations to be successful, complete, and `exact_oracle_greedy=true` before the consensus is valid.

When that physical consensus is valid, the launcher writes `parity_record.json` at level `generation_trajectory`. It deliberately sets `logits` to null because the massive F16 campaign verifies the greedy oracle trajectory rather than full-logit numerical equivalence.

Example invocation on the supported CUDA target:

```bash
python3 tools/run_tinyllama_massive_campaign.py --work-dir /tmp/nnis-tinyllama-parity
```

The launcher still requires a clean tracked worktree and binds reports to the exact current Git commit. Running this command is required to obtain physical TinyLlama evidence; this software contract alone is not such evidence.

## SmolLM2 producer

The existing `compare_smollm2_135m` comparator can emit a machine-readable record with `--evidence-json`.

Given an already generated pinned SmolLM2 fixture:

```bash
cargo run --locked -p nnis-model --example compare_smollm2_135m -- \
  --model /path/to/fixture/model \
  --reference /path/to/fixture/reference \
  --logit-policy strict \
  --evidence-json /tmp/smollm2-parity.json
```

With `strict`, evidence is written only after the comparator has passed prefill and every decode-stage tolerance check plus the exact greedy generation check. The record level is `logit_and_generation`.

With `--logit-policy report`, the comparator may report numerical differences without asserting numeric equivalence; any emitted record is therefore only `generation_trajectory`.

The comparator refuses to emit qualifying evidence from a dirty tracked worktree and records the exact Git `HEAD`.

## Compose and validate

After real records for multiple checkpoints have been produced from the same NNIS commit, compose them with:

```bash
python3 tools/validate_nnml1_multi_model_parity_evidence.py \
  --compose /tmp/nnml1-parity-suite.json \
  /tmp/smollm2-parity.json \
  /path/to/tinyllama/parity_record.json
```

Validate an existing record or suite with:

```bash
python3 tools/validate_nnml1_multi_model_parity_evidence.py /tmp/nnml1-parity-suite.json
```

The CI self-test uses synthetic records only to exercise the schema and negative tamper cases. Synthetic self-test records are not model-quality, CUDA, parity, or performance evidence.

## Claim boundary

This contract prepares and validates evidence production. It does not itself prove SmolLM2 or TinyLlama parity on a new hardware run, does not qualify Jetson Thor for TinyLlama, does not establish serving performance, and does not authorize a default runtime or model-family promotion.
