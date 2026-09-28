# Portable confidence-scheduled speculative verification surface v1 (DSV41-4)

Status: software surface contract. The contract is backend-neutral (`nnis-core`), and there is a deterministic CPU greedy reference verifier (`nnis-cpu`).
This is roadmap item `NNIS_DSV41_4_confidence_scheduled_speculative_verification_runtime_surface`
of `deepseek_v41_kv_runtime_program_2026_09_24` (off-main sovereignty roadmap).

## Ownership

The caller (a model, draft source, or domain policy) supplies:
- the draft tokens;
- their confidences;
- the schedule parameters.

NNIS does not draft tokens, estimate confidence, or choose thresholds or caps. NNIS verifies the draft and records counts.

## `nnis_core::speculative_verify`

- `SpeculativeDraftV1::new(tokens, confidences)`: requires 1 to 1024 tokens and exactly one finite confidence in `0..=1` per token.
- `ConfidenceScheduleV1::new(policy_id, policy_schema_version, min_confidence, max_verify_tokens)`:
  - requires a canonical policy id of at most 128 bytes, a non-zero schema version, a threshold in `0..=1`, and a cap of at least 1;
  - v1 has a single mechanism: verify the **leading** draft tokens whose confidence is `>= min_confidence`, stopping at the first token below it, and never more than `max_verify_tokens`.
- `verify_scheduled_draft(draft, schedule, target_tokens)`:
  - requires exactly `scheduled + 1` target tokens under the target's declared semantics;
  - accepts the longest prefix of scheduled tokens that equals the target tokens;
  - emits the accepted tokens plus exactly one target token: the correction at the first mismatch, or the next token after a full accept;
  - when zero tokens are scheduled, the step is a plain target step;
  - the result is a pure function of its inputs.
- `SpeculativeAcceptanceStatsV1` holds checked cumulative counts:
  - steps, drafted, scheduled, accepted, emitted, skipped-by-schedule, and steps with a rejection;
  - an exact `accepted / scheduled` ratio;
  - a failed record leaves the stats unchanged.

## `nnis_cpu::speculative`

- `CpuGreedySpeculativeVerifierV1::new(vocab_size)` builds the verifier.
- `verify(draft, schedule, target_logits)`:
  - takes `(scheduled + 1) × vocab` finite logits, where row `i` is the target's logits at position `i` conditioned on the draft prefix;
  - derives the greedy tokens, with the lowest index winning ties;
  - applies the core rule;
  - rejects draft tokens outside the vocabulary.
- `verify_and_record` updates the stats only on success.

## Evidence

Host-only unit tests cover:
- the schedule's threshold, cap, and equality behaviour;
- full accept with the extra token;
- correction on the first mismatch, with later matches ignored;
- zero-scheduled steps;
- target-length checks;
- count accumulation, the exact ratio, and overflow atomicity;
- rejection of malformed drafts and schedules;
- greedy tie-break;
- how many logits rows the schedule requires;
- determinism across repeats;
- rejection of out-of-vocabulary tokens and non-finite logits.

No model, GPU, or physical run was performed.

## Claim boundary

- Counts only. Acceptance counts are **not** a speed-up. Any throughput claim needs measured acceptance and real runtime cost, from both draft and target, on the exact environment (DSV41-5).
- The surface is not wired into `InferenceSession` generation. No draft model and no confidence estimator ship with it.
- Sampling-based (non-greedy) acceptance rules are out of scope for v1.
- No WGPU or CUDA execution. No novelty claim and no DeepSeek-V4.1 equivalence claim.
