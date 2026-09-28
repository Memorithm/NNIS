# Portable CPU host evidence record and harness (v1)

Status: software-only record format, validator and harness for NNIS-P10 CPU host qualification. No reviewed host record is committed. This slice makes no ARM64, performance or promotion claim.

## Why

P10 asks for qualification on x86_64 CPU where available and on ARM64 CPU, with reports that name the real host and never generalize across devices. CI runs on x86_64 only. The CPU reference is the oracle for every other backend, so it needs its own per-host check. That check has to be something a person can run on an ARM64 (or any other) machine, producing a record that cannot be over-read.

## Record: `nnis.portable.cpu-host-evidence`, schema 1

`nnis_core::host_evidence::PortableCpuHostEvidenceV1` reuses the `source` and `suites` parts of the [adapter evidence record](PORTABLE_ADAPTER_EVIDENCE_V1.md) and replaces the adapter with a host:

- `host.arch`, `host.os`: from the compilation target (`std::env::consts`);
- `host.endian`: `little` or `big`;
- `host.pointer_width`: 16, 32 or 64;
- `host.cpu_model`: operator-supplied, may be empty.

Like adapter records, host records have **no timing, throughput, memory or energy field**. JSON I/O is equally strict: wrong kind or version, missing or unknown fields, wrong types and invalid integers are all rejected. An adapter record is not accepted as a host record, and a host record is not accepted as an adapter record.

## Validator

`validate_cpu_host_evidence(record, required_suites)` applies the same source and suite checks as the adapter validator. It also rejects:

- an empty `arch` or `os`;
- a non-canonical `cpu_model`;
- an `endian` other than `little`/`big`;
- a `pointer_width` other than 16/32/64;
- a `source.target` that does not name `host.arch`.

For a structurally valid record it returns the first verdict that applies, in this order:

1. `Failed { suite_ids }`
2. `Incomplete { suite_ids }`
3. `DirtyWorktree`
4. `ReferenceOracleAgreementObserved`: clean worktree, every required suite passed with zero mismatches.

`ReferenceOracleAgreementObserved` is scoped to the exact host, toolchain and commit in the record. It is not a performance result. It does not extend to other hosts, and it does not qualify any GPU backend.

## Suites and the integer oracle

`nnis_cpu::evidence::CPU_HOST_QUALIFICATION_SUITES_V1` fixes the required suites. Each suite runs the `nnis-cpu` reference and compares every output bit with an **integer-only oracle**:

- each binary32 is decomposed into an exact integer mantissa and exponent;
- sums and products are computed exactly in 128-bit integers;
- each result is rounded once to nearest-even binary32, with gradual underflow.

A term more than 60 binary orders below the other is replaced by a same-sign one bit 61 orders below. It rounds identically, because it stays strictly inside the same rounding interval.

The oracle never uses the host FPU, `mul_add` or float conversion. A host whose binary32 addition, multiplication, fused multiply-add (`f32::mul_add`, which is a libm call on some targets), float-to-float conversion or subnormal handling (flush-to-zero) departs from IEEE-754 therefore fails.

| Suite | Content |
| --- | --- |
| `cpu.f32_binary` | add and multiply on 2000 random normal pairs and all 144 ordered pairs of 12 edge values (signed zeros, subnormals, smallest normals, 2²⁴, 2⁶³) |
| `cpu.f32_relu_gather` | ReLU of edge and random values (both zeros to +0); gather with repeated indices |
| `cpu.f32_serial_accumulation` | serial sum including order-sensitive and subnormal fixtures; scatter-add with repeated destinations |
| `cpu.f32_fused_projection` | `project_kn` with one rounding per step, including the `(1+2⁻²³)(1−2⁻²³)−1 = −2⁻⁴⁶` fixture, up to K = 256 |
| `cpu.f32_graph` | 6-node graph (add, ReLU, multiply, gather, scatter-add, sum) on 5 input pairs, plus input preservation |
| `cpu.dsv41_fp4_decode` | all 16 codes for every E8M0 exponent and 606 F32 scales, including subnormal results |

`tests/cpu_host_evidence_harness.rs` runs the suites in CI on x86_64 and requires every suite to pass. It also checks the oracle itself against hand-derived cases:

- ties to even;
- signed-zero rules;
- gradual underflow;
- sticky-bit tie breaking;
- overflow;
- the fused fixture.

Mutating the oracle's rounding, sticky placement or zero sign makes both tests fail. That was checked locally before this change.

## Harness

On the host being qualified, from a clean checkout of the exact commit:

```text
cargo run --release --locked -p nnis-cpu --example cpu_host_evidence -- \
    --commit "$(git rev-parse HEAD)" \
    --worktree-clean "$(test -z "$(git status --porcelain)" && echo true || echo false)" \
    --toolchain "$(rustc -V)" \
    --target "$(rustc -vV | sed -n 's/^host: //p')" \
    --cpu-model "<CPU model, or empty>" \
    --out cpu-host-evidence.json
```

It writes the record only when the record is structurally valid, prints the verdict, and exits non-zero for `Failed`. Reviewers re-check a record without running anything:

```text
cargo run --locked -p nnis-cpu --example cpu_host_evidence -- --validate cpu-host-evidence.json
```

The harness touches no GPU and measures no time. `scripts/check-portable-cpu.sh` also runs the harness test in release mode.

The suites and oracle live in the library (`nnis_cpu::evidence::run_cpu_host_suites`, `nnis_cpu::evidence::oracle`, with `cpu_host_evidence_record` assembling a record). The example, the CLI and the test therefore run the same code. The same run is available as:

```text
nnis evidence cpu-host --commit SHA --worktree-clean true|false --toolchain TEXT [--target TEXT] [--cpu-model TEXT] [--out FILE] [--json]
nnis evidence validate --input FILE [--json]
```

`--json` prints the record on stdout. `validate` detects the record kind and prints the verdict; `--json` prints a versioned `{schema_version, record_kind, verdict}` envelope.

## Local run on this box

Development box: x86_64 Linux, virtualized "Intel(R) Xeon(R) Processor", `rustc 1.98.1` and `1.77.0`, uncommitted worktree, so the verdict was `DirtyWorktree`. Every suite passed with zero mismatches:

| Suite | Checks |
| --- | --- |
| `cpu.f32_binary` | 4288 |
| `cpu.f32_relu_gather` | 1212 |
| `cpu.f32_serial_accumulation` | 48 |
| `cpu.f32_fused_projection` | 83 |
| `cpu.f32_graph` | 3005 |
| `cpu.dsv41_fp4_decode` | 13776 |

That run is not a committed record. `cargo check --target aarch64-unknown-linux-gnu` of `nnis-cpu` (all targets) succeeds, but nothing was executed on ARM64.

## Claim boundary

- Format, validator and harness only. No reviewed host record exists.
- No ARM64 execution was performed. ARM64 qualification needs a person to run the harness on an ARM64 host.
- The CPU verdict says nothing about WGPU or any GPU adapter. That is covered separately by the adapter evidence harness.
- No latency, throughput, memory or energy claim.
