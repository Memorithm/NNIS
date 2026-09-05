# NNML1 Hub parity validation process v1

NNIS publishes a narrow process boundary for orchestration systems that need to validate already-produced NNML1 exact-checkpoint parity evidence without reimplementing NNIS parity semantics.

## Ownership and claim boundary

NNIS remains authoritative for:

- `nnis-nnml1-reference-parity-record-v1`;
- `nnis-nnml1-multi-model-parity-suite-v1`;
- exact checkpoint identities;
- semantic greedy-trajectory requirements;
- strict logit tolerance requirements;
- same-head composition rules.

The process wrapper delegates those checks to `tools/validate_nnml1_multi_model_parity_evidence.py`. It does not execute CUDA, generate new parity observations, infer model-family admission, authorize runtime promotion, or establish serving performance.

## Contract

Contract identifier:

```text
nnis.nnml1.parity-validation@1.0.0
```

Input media type:

```text
application/vnd.nnis.nnml1.parity-evidence.v1+json
```

The input may contain either one qualified `nnis-nnml1-reference-parity-record-v1` or one `nnis-nnml1-multi-model-parity-suite-v1`.

Output media type:

```text
application/vnd.nnis.nnml1.parity-validation.v1+json
```

The process reads at most 16 MiB, requires a regular non-empty file, validates the document with the existing NNIS validator, binds the exact input bytes by SHA-256, and writes a deterministic JSON result.

A successful output contains:

- `schema_version = 1`;
- the exact contract and media type above;
- `status = "validated"`;
- `validation_scope = "nnml1_exact_checkpoint_parity_contract_only"`;
- the input evidence kind and SHA-256;
- the single validated NNIS execution Git commit;
- exact checkpoint specification names;
- observed parity levels;
- reference-runtime identities;
- NNIS execution backend identities;
- `promotion_authorized = false`;
- `serving_performance_verified = false`;
- `general_model_family_support_verified = false`.

Validation success means the supplied artifact conforms to the NNML1 parity evidence contract. It does not independently rerun the model or reference runtime.

## Process invocation

Repository invocation:

```text
python3 tools/nnis_hub_nnml1_parity_validate.py \
  --evidence /path/to/parity-evidence.json \
  --result /path/to/result.json
```

For a Hub deployment, install the two NNIS-owned validator scripts together, for example under `/opt/nnis/libexec/`, and invoke the wrapper through an absolute Python interpreter path. The wrapper imports `validate_nnml1_multi_model_parity_evidence.py`; deployment must not replace that validator with a Hub-owned copy carrying different semantics.

Recommended process binding:

```text
/usr/bin/python3
  /opt/nnis/libexec/nnis_hub_nnml1_parity_validate.py
  --evidence {input:parity_evidence}
  --result {output:validation}
```

## Exit semantics

- `0`: the NNIS parity evidence contract validated and the result artifact was written;
- `2`: invalid, unsupported, oversized, or otherwise non-conforming input/process contract;
- `3`: unexpected process/runtime failure.

Exit `0` is not a model-family promotion decision. `promotion_authorized` remains false by construction.

## CI qualification

`--self-test` exercises the wrapper against the existing NNIS synthetic contract fixtures, deterministic output, promotion tampering, and the input-size bound. Synthetic fixtures remain contract tests only and are not physical CUDA/model-quality evidence.
