# NNML1 TinyLlama reference artifact v1

## Status

This contract prepares the exact pinned TinyLlama-1.1B-Chat-v1.0 checkpoint for reproducible NNIS reference generation. It is an artifact-integrity contract, not a hardware qualification or a claim of general Llama-family support.

The exact checkpoint remains:

- repository: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- revision: `d9128824c0c80111be21424e68086f52413fb413`
- `model.safetensors` SHA-256: `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`
- source weights: BF16
- persisted NNIS execution weights: F32
- trusted reference runtime: Transformers 4.43.3 on CPU F32

## Persisted identity

A generated fixture must contain `tokenizer_reference_identity.json` beside `tokenizer.json`, `reference_suite.json`, and the converted `model/` directory.

The identity is produced by `tools/tokenizer_reference_identity.py` from the bytes of the resolved `tokenizer.json`. The campaign launcher reads and validates that persisted identity; it does not reconstruct a replacement identity and silently accept a missing record.

The identity binds the exact checkpoint spec, immutable source revision and model digest, tokenizer digest, reference runtime/version, source and execution dtypes, and oracle semantics. `reference_suite.json` and `model/provenance.json` must agree with those linked fields.

## CPU reference execution policy

Reference generation uses an explicit deterministic CPU policy:

- Torch deterministic algorithms enabled
- manual seed `0`
- one intra-op Torch thread
- one inter-op Torch thread
- `MKL_CBWR=COMPATIBLE`
- `MKL_DYNAMIC=FALSE`
- `OMP_DYNAMIC=FALSE`
- `MKL_NUM_THREADS=1`
- `OMP_NUM_THREADS=1`

The policy is recorded in `reference_suite.json`. This policy is for reference reproducibility, not performance measurement.

## Evidence boundary

A valid artifact proves that the fixture's recorded source, tokenizer and trusted-reference identities are internally consistent and that the declared CPU reference policy was used by the generator. It does not prove:

- CUDA correctness on Jetson Thor,
- NNIS/Transformers logit parity for this checkpoint,
- performance or memory targets,
- quality beyond the recorded greedy reference cases,
- compatibility with checkpoints other than the exact pinned TinyLlama artifact.

Those require separate executed evidence and their existing qualification gates.
