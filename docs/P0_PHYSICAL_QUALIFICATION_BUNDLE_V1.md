# P0 physical qualification bundle v1

`tools/run_p0_physical_qualification_bundle.py` orchestrates the remaining physical P0 evidence gates on one exact promoted NNIS `main` commit.

It does not promote a runtime, a checkpoint, or a model family. A successful bundle means only that the existing validators accepted the generated artifacts for that exact commit.

## What the bundle executes

In order, the launcher:

1. requires a completely clean repository, including no untracked files;
2. fetches `origin/main` and requires local `HEAD` to equal the fetched commit;
3. optionally requires that both commits also equal `--expected-head`;
4. checks the two supplied Python reference environments against the pinned package versions;
5. runs `run_nnml0_real_safetensors_qualification.sh` for the pinned SmolLM2 Safetensors loader gate;
6. validates the resulting schema-v2 NNML0 evidence against the exact Git commit;
7. generates the pinned SmolLM2 CPU reference fixture and runs the direct NNIS CUDA comparator in strict logit mode;
8. validates the emitted SmolLM2 `logit_and_generation` parity record;
9. runs the existing trained TinyLlama massive F16 campaign and requires its valid exact-greedy consensus;
10. validates the emitted TinyLlama `generation_trajectory` parity record;
11. composes the two records into `nnis-nnml1-multi-model-parity-suite-v1` and validates it;
12. writes `P0_PHYSICAL_QUALIFICATION.json` with the exact commit, Python environment versions, artifact paths, sizes, and SHA-256 digests.

Any failed command or validator aborts the bundle. No partial result is promoted.

## Separate Python environments are mandatory

The two trusted reference generators intentionally use different pinned Transformers versions:

- SmolLM2: `torch 2.4.0`, `transformers 4.40.1`, `safetensors 0.4.5`, `huggingface_hub 0.24.7`
- TinyLlama: `torch 2.4.0`, `transformers 4.43.3`, `safetensors 0.4.5`, `huggingface_hub 0.24.7`

The launcher therefore requires two distinct Python executables. A local or vendor PyTorch build may contain a `+...` build suffix, but its base version must remain `2.4.0`; the other pinned package versions must match exactly.

On systems where the pinned wheels are available, conventional environments can be prepared with:

```bash
python3 -m venv /tmp/nnis-smollm2-ref
/tmp/nnis-smollm2-ref/bin/python -m pip install --upgrade pip
/tmp/nnis-smollm2-ref/bin/python -m pip install -r tools/requirements-smollm2-135m.txt

python3 -m venv /tmp/nnis-tinyllama-ref
/tmp/nnis-tinyllama-ref/bin/python -m pip install --upgrade pip
/tmp/nnis-tinyllama-ref/bin/python -m pip install -r tools/requirements-tinyllama-1p1b.txt
```

Architecture-specific NVIDIA/PyTorch installations may require their supported package source instead. Do not bypass the launcher's version checks merely to make an environment run.

## Physical run

The evidence directory and optional cache directory must both be outside the repository. From a clean checkout of promoted `main`:

```bash
git fetch origin main
git checkout main
git reset --hard origin/main

HEAD_SHA="$(git rev-parse HEAD)"
python3 tools/run_p0_physical_qualification_bundle.py \
  --work-dir "/tmp/nnis-p0-${HEAD_SHA:0:12}" \
  --cache-dir /tmp/nnis-hf-cache \
  --smollm2-python /tmp/nnis-smollm2-ref/bin/python \
  --tinyllama-python /tmp/nnis-tinyllama-ref/bin/python \
  --expected-head "$HEAD_SHA"
```

The TinyLlama campaign retains its existing exact-head resume behavior. To force all TinyLlama repeats to execute again, add `--no-resume-tinyllama`.

## Expected artifacts

For a successful run, the bundle directory contains at least:

- `nnml0-real-safetensors-evidence.json`
- `smollm2-parity-record.json`
- `tinyllama/runs-<head-prefix>/parity_record.json`
- `nnml1-multi-model-parity-suite.json`
- `P0_PHYSICAL_QUALIFICATION.json`

The bundle manifest records SHA-256 digests for these evidence artifacts and for the SmolLM2 `reference.json` and TinyLlama `consensus.json` source-evidence files. The individual validators remain authoritative for their semantics.

## Claim boundary

A passing bundle does not satisfy the NNML1 exit criterion of multiple independently admitted real model families. SmolLM2 and TinyLlama are currently exact registered checkpoint identities inside the audited Llama-like execution semantics. The bundle also makes no serving-performance or automatic-promotion claim.

The checked-in CI runs only `--self-test` for the orchestration guardrails. CI does not run this physical bundle and its synthetic/self-test success is not CUDA evidence.
