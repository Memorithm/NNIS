# NNML1 tokenizer/reference identity v1

NNIS qualification must distinguish checkpoint identity from tokenizer and trusted
reference identity. A model SHA alone is insufficient when prompts are converted to
token IDs by an external tokenizer or when oracle outputs come from a versioned
reference runtime.

`tools/tokenizer_reference_identity.py` defines the standard-library-only contract
`nnis-tokenizer-reference-identity-v1`.

## Bound identity

The contract records:

- exact checkpoint spec name and version;
- source repository and immutable revision;
- source `model.safetensors` SHA-256;
- the actual SHA-256 computed from the downloaded/copied `tokenizer.json` artifact;
- reference kind;
- trusted reference runtime and exact version;
- source and persisted execution weight dtypes;
- a concise oracle-semantics description.

A deterministic canonical record is available for provenance/cache keys. It is not an
authentication primitive.

## No fabricated tokenizer hashes

Tokenizer SHA-256 values are deliberately not hardcoded before the tokenizer artifact
has actually been resolved from the pinned upstream revision. The fixture computes the
hash from bytes on disk and downstream validation must recompute the file hash instead
of trusting two metadata fields merely because they contain the same string.

This matters for the existing TinyLlama massive campaign: its fixture already stores a
`tokenizer_sha256`, but the launcher previously only checked equality between the model
provenance and reference-suite fields. The v1 contract additionally verifies that the
actual `tokenizer.json` bytes hash to the recorded value.

SmolLM2's trained reference fixture now records the copied tokenizer SHA-256 in both
model provenance and reference-logit metadata so the same contract can be applied to
future exact reference-parity gates.

## Claim boundary

A valid tokenizer/reference identity proves provenance consistency only. It does not
prove tokenizer behavioral parity across libraries, correct chat-template application,
model-logit parity, generation quality, CUDA correctness, latency, throughput, VRAM, or
model-family support. Those remain separate evidence gates.

CI runs negative self-tests of the identity contract without network or GPU access.
