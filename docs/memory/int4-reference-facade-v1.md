# INT4 reference facade v1

The `nnis` facade re-exports the NNML2 INT4 reference storage and isolated
projection contracts that already live in `nnis-model` / `nnis-kernels`
(PRs #152 / #153).

## Source

- Storage + plan types and version constants: `nnis_model::int4_reference`
- Kernel primitive: `nnis_kernels::F32Int4Gemv`

Facade modules:

- `nnis::model` and crate root for storage/plan/quantize helpers and
  `NNIS_INT4_REFERENCE_*` constants;
- `nnis::kernels` and crate root for `F32Int4Gemv`.

## Explicit non-claims

Facade publicity does not claim full-model INT4 execution, dense-weight
equivalence, model quality, serving performance, physical residency, or Thor
parity. See `docs/NNML2_INT4_PROJECTION_V1.md` and
`docs/NNML2_INT4_REFERENCE_STORAGE_V1.md`.
