# NNML0 real Safetensors qualification

This gate qualifies one exact external decoder checkpoint through NNIS's native
Rust Safetensors loader on a real CUDA device. It is a model-loading evidence
gate, not a performance benchmark and not a claim of general Llama-family
support.

## Frozen source identity

- repository: `HuggingFaceTB/SmolLM2-135M`
- revision: `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`
- `model.safetensors` SHA-256:
  `80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`
- expected persisted weight dtype: BF16

The qualification executable rejects source hash drift, architecture/config
drift, unsupported dtype, dirty NNIS worktrees, malformed tensor geometry, and
CUDA upload failures.

## Physical run

Run from the exact NNIS commit being considered for promotion:

```bash
cd /root/NNIS
git status --short --branch
git rev-parse HEAD
tools/run_nnml0_real_safetensors_qualification.sh \
  /tmp/nnis-nnml0-smollm2-135m \
  /tmp/nnis-nnml0-real-safetensors-evidence.json
```

The launcher downloads only `config.json` and `model.safetensors` from the
frozen upstream revision, verifies the model SHA-256 before execution, and
invokes the Rust loader with `cargo --locked`. Python, PyTorch, and Transformers
are not runtime dependencies of the model load.

A successful run must end with both:

```text
NNML0_REAL_SAFETENSORS_LOAD_OK
NNML0_EVIDENCE_PATH=/tmp/nnis-nnml0-real-safetensors-evidence.json
```

## Evidence validation

Validate the generated JSON against the same exact NNIS commit:

```bash
python3 tools/validate_nnml0_real_safetensors_evidence.py \
  /tmp/nnis-nnml0-real-safetensors-evidence.json \
  --expected-commit "$(git rev-parse HEAD)"
```

The validator is fail-closed. It requires schema version 1, a clean 40-hex NNIS
commit identity, the frozen source identity, a concrete CUDA UUID and device
geometry, BF16 SmolLM2-135M model geometry, and `result = "pass"`. Unknown
fields and altered contract values are rejected. CI runs negative self-tests of
the validator independently of physical GPU evidence.

## Promotion boundary

NNML0's real-checkpoint loading sub-gate is not closed merely because the
qualification executable compiles or because GitHub CI is green. Closure
requires physical evidence generated from the exact promoted NNIS commit and
accepted by the validator above.

This gate also does not by itself establish the separate SciRust interoperability
goal. At the SciRust master revision audited while this gate was prepared
(`f6bdadb6234129e14e9ea4d69f46901c6dcecbd0`), backend-neutral `Tensor` and
`DType` contracts exist, but the relevant `scirust-compute` and
`scirust-tensor-runtime` crates declare Rust 1.89 while NNIS preserves Rust
1.77, and no public canonical `Model` trait was verified. NNIS therefore does
not add a nominal SciRust dependency or claim that interoperability contract
until a compatible, versioned boundary is actually available.
