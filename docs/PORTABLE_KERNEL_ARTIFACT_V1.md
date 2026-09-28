# Portable kernel artifact binding and capability identity v1 (NNIS-P4)

Status: software contract (`nnis-core`), with a CPU binding test against the declared `CpuDevice` limits. It covers the portable-runtime `next` item `define_versioned_portable_kernel_artifact_binding_and_capability_identity` (see `.agent/PORTABLE_RUNTIME_PROGRESS.yaml`), which is NNIS-P4 in `docs/V888_CONNECTOME_BOOTSTRAP.md`.

## Artifact (`nnis_core::kernel_artifact::KernelArtifactV1`)

An artifact replaces the assumption that kernels are compiled at runtime. Each field below is validated, and any failure rejects the artifact (fail closed).

- **Identity:** a canonical `artifact_id` of at most 128 bytes, and a non-zero `artifact_revision`.
- **Source:** 1 byte to 16 MiB. The source kind must match a backend family:
  - `Wgsl`: UTF-8 WGSL, for WGPU;
  - `PortableIr`: pre-validated portable IR bytes, for WGPU;
  - `CpuBuiltin`: the UTF-8 name of a CPU reference operation, for CPU.
- **Entry point:** an ASCII identifier.
- **Binding schema:** 1 to 64 bindings.
  - Entries are strictly increasing by `(group, binding)`.
  - Each entry has a kind (read-only storage, read-write storage, or uniform), an element type (`F32`, `F16`, `U32`), and a non-zero, element-aligned minimum size.
  - At least one binding must be read-write storage.
- **Workgroup geometry:** non-zero dimensions whose product does not overflow.
- **Numerical policy:** a policy id, e.g. `finite-f32-le-ordered-fma-v1`.
- **Qualification evidence:** an optional SHA-256 of the qualification evidence. If absent, the artifact is **unqualified**.

## Fingerprints

- `source_sha256`: SHA-256 of the exact source bytes.
- `artifact_fingerprint`: SHA-256 of the canonical v1 descriptor.
  - The descriptor starts with the domain tag `nnis.portable-kernel-artifact.v1\0` and the contract version.
  - It then contains every field in a fixed order: little-endian integers, `u32` length-prefixed strings, the source length, and the source digest.
  - Changing any field changes the fingerprint.
  - A golden test value pins the encoding. It was independently reproduced with Python `hashlib`.
- SHA-256 is implemented without dependencies in `nnis_core::sha256`. It is checked against the FIPS 180 known-answer vectors, including one million `a` characters.

## Binding and capability identity

`KernelArtifactV1::bind(backend, capabilities, expected_fingerprint)` succeeds only if all of these hold:
- the expected fingerprint matches exactly;
- the backend family can consume the source kind;
- every derived requirement fits the backend's `CapabilitySet`: workgroup size per axis, invocations per workgroup, binding count, per-buffer bytes, and F16 support.

On success it returns a `BoundKernelArtifactV1`: the artifact id, revision, and fingerprint, plus the backend id and the capabilities it was checked against. `check_capabilities` reports the first missing capability. Kernel placement is rejected when a required capability is missing.

With today's declared `CpuDevice` limits (1 binding, workgroup `[1,1,1]`), a single-binding CPU built-in binds, and wider artifacts are rejected.

## Claim boundary

- Binding compiles, dispatches, and executes nothing.
- A bound artifact is not evidence of numerical correctness, cross-backend parity, or performance.
- There is no WGPU backend or WGSL validation in this slice. WGSL is carried as opaque UTF-8, and `PortableIr` is trusted as already validated by its producer.
- No new dependencies. There are no vendor (CUDA/NVRTC/NVML) facilities, and no physical run was performed.
