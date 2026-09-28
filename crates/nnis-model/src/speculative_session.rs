//! Opt-in, caller-driven greedy speculative verification for sessions.
//!
//! [`InferenceSession::verify_greedy_draft`](crate::InferenceSession::verify_greedy_draft)
//! consumes a caller-supplied DSV41-4 draft (tokens plus confidences) under a
//! caller-supplied [`ConfidenceScheduleV1`]. It teacher-forces every
//! scheduled draft token through the ordinary one-token decode path, judges
//! the collected target logits with the `nnis-cpu`
//! [`CpuGreedySpeculativeVerifierV1`] (finite logits, lowest index wins
//! ties), rewinds the KV cache to the accepted prefix and decodes the one
//! emitted target token. Acceptance counts go to a caller-owned
//! [`SpeculativeAcceptanceStatsV1`].
//!
//! NNIS does not produce drafts or confidences and chooses no schedule. The
//! path verifies one token at a time and is **not** a speed-up: counts are not
//! throughput, and no latency or quality claim is made. Default generation is
//! unchanged; nothing here runs unless called.

use nnis_core::speculative_verify::{
    ConfidenceScheduleV1, SpeculativeAcceptanceStatsV1, SpeculativeDraftV1,
    SpeculativeVerificationV1,
};
use nnis_cpu::speculative::CpuGreedySpeculativeVerifierV1;
use nnis_rt::{NnisError, Result};

/// Outcome of one session speculative verification step.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSpeculativeStepV1 {
    /// Verification outcome; `emitted_tokens` is the accepted draft prefix
    /// followed by exactly one target token.
    pub verification: SpeculativeVerificationV1,
    /// Session position before the step.
    pub start_position: usize,
    /// Logits after the last emitted token, as returned by `decode_one`.
    pub next_logits: Vec<f32>,
}

/// Teacher-forced target logits for a scheduled draft: the current logits
/// followed by the logits after each scheduled draft token, row-major.
pub(crate) fn teacher_forced_logits<D>(
    current_logits: Vec<f32>,
    draft: &SpeculativeDraftV1,
    schedule: &ConfidenceScheduleV1,
    mut decode: D,
) -> Result<Vec<f32>>
where
    D: FnMut(u32) -> Result<Vec<f32>>,
{
    let vocab = current_logits.len();
    let scheduled = schedule.scheduled_len(draft) as usize;
    let mut rows = current_logits;
    rows.reserve(scheduled.saturating_mul(vocab));
    for &token in &draft.tokens()[..scheduled] {
        let logits = decode(token)?;
        if logits.len() != vocab {
            return Err(NnisError::invalid_input(format!(
                "decode returned {} logits; expected {vocab}",
                logits.len()
            )));
        }
        rows.extend_from_slice(&logits);
    }
    Ok(rows)
}

/// Judge teacher-forced logits with the CPU greedy reference semantics.
pub(crate) fn judge_greedy(
    vocab_size: usize,
    draft: &SpeculativeDraftV1,
    schedule: &ConfidenceScheduleV1,
    target_logits: &[f32],
) -> Result<SpeculativeVerificationV1> {
    let vocab = u32::try_from(vocab_size)
        .map_err(|_| NnisError::invalid_input("vocabulary size exceeds u32"))?;
    CpuGreedySpeculativeVerifierV1::new(vocab)
        .and_then(|verifier| verifier.verify(draft, schedule, target_logits))
        .map_err(|error| NnisError::invalid_input(format!("speculative verification: {error}")))
}

/// Record one outcome into a copy of `stats`, so the caller's counters change
/// only when the whole step succeeds.
pub(crate) fn recorded(
    stats: &SpeculativeAcceptanceStatsV1,
    outcome: &SpeculativeVerificationV1,
) -> Result<SpeculativeAcceptanceStatsV1> {
    let mut next = *stats;
    next.record(outcome)
        .map_err(|error| NnisError::invalid_input(format!("speculative counters: {error}")))?;
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic fake target: after token `t` the argmax is `(t * 3 + 1) % 5`.
    fn fake_logits(previous: u32) -> Vec<f32> {
        let mut logits = vec![0.0; 5];
        logits[((previous * 3 + 1) % 5) as usize] = 1.0;
        logits
    }

    fn schedule(max: u32) -> ConfidenceScheduleV1 {
        ConfidenceScheduleV1::new("test.schedule", 1, 0.5, max).unwrap()
    }

    #[test]
    fn teacher_forcing_decodes_only_scheduled_tokens_and_judges_greedily() {
        // Target chain from token 0: 1, 4, 3, 0, 1, ...
        let draft = SpeculativeDraftV1::new(vec![1, 4, 2, 0], vec![0.9, 0.9, 0.9, 0.1]).unwrap();
        let mut decoded = Vec::new();
        let rows = teacher_forced_logits(fake_logits(0), &draft, &schedule(8), |token| {
            decoded.push(token);
            Ok(fake_logits(token))
        })
        .unwrap();
        // Confidence 0.1 < 0.5 stops the schedule after three tokens.
        assert_eq!(decoded, [1, 4, 2]);
        assert_eq!(rows.len(), 4 * 5);
        let outcome = judge_greedy(5, &draft, &schedule(8), &rows).unwrap();
        assert_eq!(outcome.scheduled, 3);
        assert_eq!(outcome.accepted, 2);
        assert_eq!(outcome.rejected_at, Some(2));
        assert_eq!(outcome.emitted_tokens, [1, 4, 3]);

        let stats = recorded(&SpeculativeAcceptanceStatsV1::default(), &outcome).unwrap();
        assert_eq!((stats.steps, stats.accepted, stats.emitted), (1, 2, 3));
        assert_eq!(stats.skipped_by_schedule, 1);
        assert_eq!(stats.acceptance_ratio(), Some((2, 3)));
    }

    #[test]
    fn full_accept_emits_the_next_target_token_and_cap_limits_decodes() {
        let draft = SpeculativeDraftV1::new(vec![1, 4, 3], vec![1.0; 3]).unwrap();
        let mut decoded = 0;
        let rows = teacher_forced_logits(fake_logits(0), &draft, &schedule(2), |token| {
            decoded += 1;
            Ok(fake_logits(token))
        })
        .unwrap();
        assert_eq!(decoded, 2);
        let outcome = judge_greedy(5, &draft, &schedule(2), &rows).unwrap();
        assert_eq!((outcome.accepted, outcome.rejected_at), (2, None));
        assert_eq!(outcome.emitted_tokens, [1, 4, 3]);
    }

    #[test]
    fn failures_are_explicit() {
        let draft = SpeculativeDraftV1::new(vec![1], vec![1.0]).unwrap();
        let error =
            teacher_forced_logits(fake_logits(0), &draft, &schedule(4), |_| Ok(vec![0.0; 3]))
                .unwrap_err();
        assert!(error.to_string().contains("expected 5"), "{error}");
        let error = teacher_forced_logits(fake_logits(0), &draft, &schedule(4), |_| {
            Err(NnisError::invalid_input("decode failed"))
        })
        .unwrap_err();
        assert!(error.to_string().contains("decode failed"), "{error}");
        let mut logits = fake_logits(0);
        logits.extend(fake_logits(1));
        logits[7] = f32::NAN;
        assert!(judge_greedy(5, &draft, &schedule(4), &logits).is_err());
        let out_of_vocab = SpeculativeDraftV1::new(vec![9], vec![1.0]).unwrap();
        let mut two_rows = fake_logits(0);
        two_rows.extend(fake_logits(0));
        assert!(judge_greedy(5, &out_of_vocab, &schedule(4), &two_rows).is_err());
        let saturated = SpeculativeAcceptanceStatsV1 {
            steps: u64::MAX,
            ..SpeculativeAcceptanceStatsV1::default()
        };
        let outcome = judge_greedy(5, &draft, &schedule(4), &{
            let mut rows = fake_logits(0);
            rows.extend(fake_logits(1));
            rows
        })
        .unwrap();
        assert!(recorded(&saturated, &outcome).is_err());
    }
}
