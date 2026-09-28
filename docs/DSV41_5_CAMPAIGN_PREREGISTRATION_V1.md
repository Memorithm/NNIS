# DSV41-5 campaign preregistration and evidence validation v1

Status: software contract only. **No campaign has been executed and no result is recorded.**
This supports roadmap step `NNIS_DSV41_5_real_model_quality_memory_latency_and_throughput_campaign`
of `deepseek_v41_kv_runtime_program_2026_09_24`. That step requires physical execution.

The module is `nnis_core::dsv41_campaign` (zero-dependency, contract version 1).

## Preregistration (`Dsv41CampaignPreregistrationV1`)

The preregistration is declared before execution. It is rejected (fails closed) if any field is malformed. It declares:

- **Campaign and model identity:** a canonical campaign id, the model id, an exact 40-hex source revision, and the SHA-256 of the weight bytes.
- **Prompt suite:** the suite id, its SHA-256, and a prompt count of at least 1.
- **Backend:** `Cpu`, `Wgpu`, or `LegacyCudaCrossCheck`. The programme order is CPU → WGPU → optional CUDA cross-check.
- **Arms:** exactly one dense baseline with no primitives, plus at least one candidate.
  - Candidate arm ids are unique.
  - Each candidate's DSV41 primitives are listed in strictly increasing order: `BoundedReplay`, `CrossLayerKvReuse`, `Fp4E2M1Kv`, `SpeculativeVerification`.
- **Grid:** context and decode lengths, each non-zero and strictly increasing.
- **Repetitions:** at least 2, so latency is a distribution.
- **Memory metric:** one metric shared by every arm: peak process RSS, peak device-allocated bytes, or peak NNIS-owned allocation bytes.
- **Quality metric:** its id, its direction, and a caller-declared finite, non-negative regression budget.

## Evidence (`Dsv41CampaignEvidenceV1`)

`validate_dsv41_evidence` rejects a record unless all of the following hold:

- **Record kind:** the record is `PhysicalExecution`. `SyntheticFixture` records are **never evidence**, even when structurally complete.
- **Campaign:** the record names the preregistered campaign.
- **Provenance:** an exact 40-hex Git head, a clean worktree, and a non-zero CI run id with CI green on the exact head.
- **Hardware:** the executed backend equals the preregistered one, and the device name, driver/runtime version, and host OS are all present.
- **Grid coverage:** every arm × context × decode cell appears exactly once, and nothing else appears.
- **Per-cell metrics:** each cell reports all of these together:
  - the preregistered repetitions;
  - a finite quality value;
  - the preregistered memory metric, with a non-zero byte value;
  - a decode-latency distribution: at least `repetitions` samples, finite positive values, and `p50 ≤ p90 ≤ p99 ≤ max`;
  - finite positive tokens/s.
- A missing metric is reported as `MissingMetric`. It is never defaulted.

`validate_dsv41_evidence_structure` checks everything except the record kind. It exists so software tests can exercise the rules with synthetic fixtures.

## Claim boundary

- A validated record only means it is complete and well-formed. It does not authorize promotion, comparison, or a speed-up. The module computes no ratios between arms.
- The tests use clearly synthetic placeholder values that exist only in test code. The validator rejects those fixtures as non-evidence.
- No model, GPU, Thor, or other physical run was performed. No numbers are recorded anywhere in this contract.
- JSON/file serialization and hashing of preregistrations are out of scope for v1.
