# Portable adapter evidence record and WGPU harness (v1)

Status: software-only record format, validator and harness. No hardware record exists yet. This slice makes no hardware, parity, performance or promotion claim.

## Why

CI has no GPU. Every WGPU parity test in CI logs an explicit SKIP, and local runs on llvmpipe exercise only the code path. Hardware WGPU evidence has to be produced by a person on a real adapter. This slice fixes, ahead of time:

- what that person runs;
- what gets recorded;
- how a record is judged,

so that a later hardware run cannot be over-read.

## Record: `nnis.portable.adapter-evidence`, schema 1

`nnis_core::adapter_evidence::PortableAdapterEvidenceV1` has three parts:

- `source`:
  - `git_commit`: 40 lowercase hex characters;
  - `worktree_clean`;
  - `crate_version`;
  - `toolchain`: operator-supplied, for example `rustc -V`;
  - `target`.
- `adapter`, as reported by the driver:
  - `backend_family`, `api_backend`, `name`, `device_type`;
  - `vendor_id`, `device_id`;
  - `driver`, `driver_info`;
  - `class`: `hardware`, `software` or `unknown`.
- `suites`: an ordered list. Each entry has `suite_id`, `outcome` (`pass`, `fail` or `skip`), `checks`, `mismatches`, `tolerance` (the declared policy id) and `detail`.

The record has **no timing, throughput, memory or energy field**, so no performance can be recorded through it.

JSON I/O is strict. `adapter_evidence_from_json` rejects:

- syntax errors;
- a wrong `kind` or schema version;
- missing or unknown fields (a `latency_ms` field is rejected);
- wrong types;
- non-integer or out-of-range integers;
- unknown enum values.

Parsing does not validate.

## Validator

`validate_adapter_evidence(record, required_suites)` rejects malformed records with an error:

- a malformed commit;
- empty required text;
- text over 512 bytes, or with surrounding whitespace or control characters;
- no suites, or more than 64;
- invalid or duplicate suite ids;
- more mismatches than checks;
- a pass that has mismatches or has no checks;
- a skip that has checks;
- a failure or skip without detail;
- a **hardware class claimed for a `Cpu` device type or a known software implementation** (llvmpipe, lavapipe, SwiftShader, softpipe, WARP).

For a structurally valid record it returns the first verdict that applies, in this order:

1. `Failed { suite_ids }`: any suite failed.
2. `Incomplete { suite_ids }`: a required suite is missing or skipped.
3. `DirtyWorktree`: the commit does not identify the code.
4. `CodePathOnlySoftwareAdapter` or `UnclassifiedAdapter`, depending on the class.
5. `HardwareParityObserved`: a hardware adapter where every required suite passed with zero mismatches.

`HardwareParityObserved` is scoped to the exact adapter, driver, commit and toolchain in the record. It is not a performance result, not a claim about other devices and not a promotion decision. A person must still review it.

## WGPU harness

`nnis_wgpu::evidence::WGPU_QUALIFICATION_SUITES_V1` fixes the required suites. Each one uses the tolerance declared by the P3 slice it covers.

| Suite | Compared against CPU | Tolerance |
| --- | --- | --- |
| `wgpu.add_f32` | 4 sizes, including 4097 elements | bit-exact (normal range) |
| `wgpu.portable_memory` | zero-init, unaligned writes, unaligned copies | byte-exact |
| `wgpu.f32_kernels` | add, multiply, ReLU, gather, sum, scatter-add, projection | bit-exact, except projection within `(3k+1)·2⁻²⁴·Σ\|x·w\|` |
| `wgpu.f32_graph` | 6-node projection-free graph, input preservation | bit-exact |
| `wgpu.dsv41_replay_kv` | every window of a 16-position source | bit-exact |
| `wgpu.dsv41_fp4_decode` | all E8M0 exponents, 605 F32 scales × 16 codes | bit-exact |
| `wgpu.dsv41_speculative` | 64 randomized drafts | identical outcomes |

Run it on the machine whose adapter is being qualified, from a clean checkout of the exact commit:

```text
cargo run --release --locked -p nnis-wgpu --example wgpu_adapter_evidence -- \
    --commit "$(git rev-parse HEAD)" \
    --worktree-clean "$(test -z "$(git status --porcelain)" && echo true || echo false)" \
    --toolchain "$(rustc -V)" \
    --out adapter-evidence.json
```

Behaviour:

- Without an adapter it prints `SKIP nnis-wgpu adapter evidence: ...`, writes nothing and exits with status 2.
- Backend errors and panics become failed suites.
- It writes the record only if the record is structurally valid, then prints the verdict.
- It exits non-zero for `Failed`.

Reviewers can re-check a record without running any suite:

```text
cargo run --locked -p nnis-wgpu --example wgpu_adapter_evidence -- --validate adapter-evidence.json
```

`tests/adapter_evidence_harness.rs` runs the same suites (shared source file) in CI. There it logs SKIP, because CI has no GPU. When an adapter is present, it requires every suite to pass, round-trips the record through JSON, and checks that the verdict matches the adapter class.

## Local run on this box (code path only)

On llvmpipe (Mesa 25.0.7, Vulkan), every suite passes with zero mismatches:

| Suite | Checks |
| --- | --- |
| `wgpu.add_f32` | 4225 |
| `wgpu.portable_memory` | 8 |
| `wgpu.f32_kernels` | 2616 |
| `wgpu.f32_graph` | 301 |
| `wgpu.dsv41_replay_kv` | 4896 |
| `wgpu.dsv41_fp4_decode` | 13760 |
| `wgpu.dsv41_speculative` | 64 |

The validator correctly gives `CodePathOnlySoftwareAdapter` on a clean tree, or `DirtyWorktree`. Editing the record's class to `hardware` is rejected. This is software-adapter evidence of the code path only.

## Claim boundary

- No hardware record exists. Producing one needs a person with a real adapter.
- CI passes are SKIP-only and are not evidence.
- Software adapters never give a hardware verdict.
- No timing is recorded or implied.
- Wiring `scripts/check-portable-wgpu.sh` into CI still needs a workflow-scope push.
