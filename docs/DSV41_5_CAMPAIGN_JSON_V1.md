# DSV41-5 campaign JSON, schema version 1

`nnis_core::dsv41_campaign_json` reads and writes the DSV41-5
preregistration and evidence records as versioned JSON. The parser and writer
are hand-written and crate-private (`nnis_core::json`), so `nnis-core` stays
dependency-free and within the Rust 1.77 MSRV.

## Records

| kind tag | Rust type | functions |
|---|---|---|
| `nnis.dsv41.campaign-preregistration` | `Dsv41CampaignPreregistrationV1` | `preregistration_to_json`, `preregistration_from_json` |
| `nnis.dsv41.campaign-evidence` | `Dsv41CampaignEvidenceV1` | `evidence_to_json`, `evidence_from_json` |

Every record has `"kind"` and `"schema_version": 1` at the top level. Field
names match the Rust fields. Enumerations are snake_case strings:

- backend: `cpu`, `wgpu`, `legacy_cuda_cross_check`
- primitives: `bounded_replay`, `cross_layer_kv_reuse`, `fp4_e2m1_kv`, `speculative_verification`
- memory metric: `peak_process_rss_bytes`, `peak_device_allocated_bytes`, `peak_nnis_owned_allocation_bytes`
- quality direction: `higher_is_better`, `lower_is_better`
- evidence `record_kind`: `physical_execution`, `synthetic_fixture`

Optional evidence cell metrics (`quality`, `memory_metric`, `memory_bytes`,
`decode_latency`, `tokens_per_second`) are required keys whose value may be
`null`.

## Fail-closed parsing

Parsing is rejected on: invalid JSON (strict RFC 8259, duplicate keys,
trailing content, nesting deeper than 64, input over 16 MiB); a wrong `kind`;
a `schema_version` other than the integer `1`; missing or unknown fields at
any level; wrong JSON types; integer fields with a sign, fraction, exponent
or out-of-range value; non-finite floats; unknown enumeration values.
A parsed preregistration also goes through
`Dsv41CampaignPreregistrationV1::new`, so contract violations are rejected.

Parsing an evidence record does **not** validate it. Callers must still run
`validate_dsv41_evidence` against the preregistration, which rejects
synthetic fixtures, `null` metrics, missing cells and provenance failures.

Encoding writes integers exactly and floats using Rust's shortest round-trip
representation, so `from_json(to_json(x)) == x` and re-encoding is
byte-identical. Non-finite floats cannot be encoded.

## Claim boundary

This is a serialization format only. No campaign has been run; the tests use
synthetic placeholder values that are not measurements.
