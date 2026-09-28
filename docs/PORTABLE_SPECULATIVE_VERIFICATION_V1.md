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

## Opt-in session API

`InferenceSession::verify_greedy_draft(&draft, &schedule, &mut stats)` (re-exported with the surface types from the `nnis` facade) is the caller-driven runtime entry point. The CUDA decoder session is the only real-model session today.

1. It requires a prefilled session and checks every draft token against the vocabulary. It also checks that `position + scheduled + 1` fits the session capacity.
2. It reads back the current logits.
3. It teacher-forces each scheduled draft token through the unchanged `decode_one` path.
4. It judges the collected rows with `CpuGreedySpeculativeVerifierV1` (same greedy semantics as the reference).
5. It rewinds the KV cache (new `KvCache::truncate`, all layers, all-or-nothing) and the position to the accepted prefix.
6. It decodes the one emitted target token and returns `SessionSpeculativeStepV1 { verification, start_position, next_logits }`.

`stats` changes only when the whole step succeeds. On failure the session is rewound to `start_position`. If that rewind also fails, the error says so and the caller must `reset`.

Nothing runs unless the method is called, and `generate`, `prefill`, `decode_one` and batch paths are unchanged. Verification goes one token at a time, so it is a correctness surface, not a speed-up.

Tests:

- **On CPU:** the teacher-forcing, judging and counter steps (scheduled-token decoding only, schedule cap, reject/accept outcomes, logits-length, decode, non-finite, out-of-vocabulary and counter-overflow failures) are unit-tested with a fake decoder. `KvCache` truncation length logic is unit-tested.
- **GPU-gated:** a truncate test is added. It SKIPs without CUDA.
- **Not run:** the session method itself has not been executed on a GPU. CI has no GPU and no GPU run is claimed.

## Claim boundary

- Counts only. Acceptance counts are **not** a speed-up. Any throughput claim needs measured acceptance and real runtime cost, from both draft and target, on the exact environment (DSV41-5).
- The surface is reachable from `InferenceSession` only through the opt-in, caller-driven `verify_greedy_draft`. Default generation does not use it. No draft model and no confidence estimator ship with it. The session method has not been executed on a GPU.
- Sampling-based (non-greedy) acceptance rules are out of scope for v1.
- No WGPU or CUDA execution was performed. No novelty claim and no DeepSeek-V4.1 equivalence claim.
