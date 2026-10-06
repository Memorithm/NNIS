# NNIS PVP explicit ANF-bank qualification protocol

Frozen before implementation, 2026-10-06. Correctness qualification only.

## Source and ownership

Reuse the frozen synthetic corpus of Memorithm/FLAT-ATTENTION PR #341,
merged at 50c0230cc645bd44f1520e2c7194878a4b13009c.
SciRust owns reusable SIMD; SML-GENIUS owns model semantics. NNIS is an
optional CPU/WGPU runtime carrier. No new runtime dependency is introduced.

## Frozen corpus

Eight (addresses K, gates G) geometries:
(1,1), (2,31), (4,65), (8,129), (64,257), (256,513),
(2048,257), (256,2048). K = 2^n.

Each geometry has three banks, for exactly 24 cases:
- boundary: gate modulo six selects [], [0], [K>1 ? 1 : 0],
  [K-1], [0, K>1 ? 1 : 0], or all singleton variable masks.
- sparse4: min(4,K) masks (257*g + 73*t) modulo K.
- dense32: min(32,K) masks from the same formula.
Sparse/dense masks must be distinct. Mask zero is the constant monomial.
Repeated terms cancel under XOR, including the K=1 boundary case.

Input coefficients are packed directly from explicit monomial masks into
address-major u64 words. Expected truth is computed independently as XOR
of ((assignment & mask) == mask), with independent address-major packing.
No butterfly, subset enumeration, or layout conversion generates expected truth.

## Acceptance

A CPU-only test must run all 24 cases without needing a WGPU adapter.
The existing mandatory WGPU parity target must run all 24 cases on its actual
adapter, comparing independently with the direct oracle. Compare every word,
including zero padding. A second transform must recover the exact original
coefficients in each session. Check execution counts, stage/logical-XOR
accounting and declared algorithmic scratch, CPU storage and packed updates.
The WGPU u32 projection must round-trip exactly before execution.

Golden boundary truth: K=8,G=6 yields
[18,38,50,6,50,6,18,46]; K=1,G=6 yields [14], including cancellation.
Log completion separately for CPU and WGPU, exactly 24 cases each.

No adapter: explicit skip outside the required gate; failure when
NNIS_REQUIRE_WGPU_PVP is set. The existing CI target includes all new tests,
so the required gate cannot silently qualify without WGPU execution.

## Evidence limits

No timings or selection changes. Software adapters qualify correctness only.
This does not qualify native GPU performance, model quality, trained banks,
SML internalization, allocation-free sessions, or physical register residency.
New execution remains Rust CPU/WGPU with no vendor SDK dependency.
