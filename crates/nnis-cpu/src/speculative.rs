//! DSV41-4 deterministic CPU reference verifier for speculative drafts.
//!
//! [`CpuGreedySpeculativeVerifierV1`] consumes a caller-supplied
//! [`SpeculativeDraftV1`] and [`ConfidenceScheduleV1`] together with target
//! logits for the scheduled positions, derives the target's greedy tokens
//! (finite logits only, lowest index wins ties), and applies the
//! backend-neutral [`verify_scheduled_draft`] rule.
//!
//! Row `i` of the logits must be the target model's logits for draft position
//! `i` when conditioned on the draft prefix `tokens[..i]` (teacher-forced
//! parallel verification); row `scheduled` gives the token after a full
//! accept. Rows after the first mismatch do not influence the outcome.
//!
//! This is a correctness oracle. It records acceptance counts only and makes
//! no latency, cost, or throughput claim; it neither drafts tokens nor
//! chooses confidence policy.

use core::fmt;

use nnis_core::speculative_verify::{
    verify_scheduled_draft, ConfidenceScheduleV1, SpeculativeAcceptanceStatsV1, SpeculativeDraftV1,
    SpeculativeError, SpeculativeVerificationV1,
};

/// Greedy-semantics CPU reference verifier for one vocabulary size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuGreedySpeculativeVerifierV1 {
    vocab_size: u32,
}

impl CpuGreedySpeculativeVerifierV1 {
    /// Construct a verifier for a non-zero vocabulary size.
    pub fn new(vocab_size: u32) -> Result<Self, CpuSpeculativeError> {
        if vocab_size == 0 {
            return Err(CpuSpeculativeError::ZeroVocabulary);
        }
        Ok(Self { vocab_size })
    }

    /// Vocabulary size of each logits row.
    pub const fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// Verify one draft step from row-major target logits.
    ///
    /// Requires every draft token to be inside the vocabulary and exactly
    /// `(scheduled + 1) * vocab_size` finite logits.
    pub fn verify(
        &self,
        draft: &SpeculativeDraftV1,
        schedule: &ConfidenceScheduleV1,
        target_logits: &[f32],
    ) -> Result<SpeculativeVerificationV1, CpuSpeculativeError> {
        if let Some(index) = draft
            .tokens()
            .iter()
            .position(|&token| token >= self.vocab_size)
        {
            return Err(CpuSpeculativeError::TokenOutOfVocabulary {
                index,
                token: draft.tokens()[index],
            });
        }
        let rows = schedule.scheduled_len(draft) as usize + 1;
        let vocab = self.vocab_size as usize;
        let expected = rows
            .checked_mul(vocab)
            .ok_or(CpuSpeculativeError::HostIndexOverflow)?;
        if target_logits.len() != expected {
            return Err(CpuSpeculativeError::LogitsLengthMismatch {
                expected,
                actual: target_logits.len(),
            });
        }
        let mut target_tokens = Vec::with_capacity(rows);
        for (row, logits) in target_logits.chunks_exact(vocab).enumerate() {
            target_tokens.push(greedy_argmax(logits).map_err(|column| {
                CpuSpeculativeError::NonFiniteLogit {
                    row,
                    column: column as u32,
                }
            })?);
        }
        Ok(verify_scheduled_draft(draft, schedule, &target_tokens)?)
    }

    /// Verify one draft step and add its counts to `stats`.
    ///
    /// `stats` is unchanged when verification or recording fails.
    pub fn verify_and_record(
        &self,
        stats: &mut SpeculativeAcceptanceStatsV1,
        draft: &SpeculativeDraftV1,
        schedule: &ConfidenceScheduleV1,
        target_logits: &[f32],
    ) -> Result<SpeculativeVerificationV1, CpuSpeculativeError> {
        let outcome = self.verify(draft, schedule, target_logits)?;
        stats.record(&outcome)?;
        Ok(outcome)
    }
}

/// Deterministic greedy argmax: finite values only, lowest index wins ties.
///
/// Returns the index of the first non-finite value as the error.
fn greedy_argmax(logits: &[f32]) -> Result<u32, usize> {
    let mut best = 0usize;
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(index);
        }
        if value > logits[best] {
            best = index;
        }
    }
    Ok(best as u32)
}

/// Fail-closed CPU speculative-verification errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuSpeculativeError {
    /// Backend-neutral contract validation failed.
    Contract(SpeculativeError),
    /// Vocabulary size was zero.
    ZeroVocabulary,
    /// A draft token was outside the vocabulary.
    TokenOutOfVocabulary { index: usize, token: u32 },
    /// Logits length was not `(scheduled + 1) * vocab_size`.
    LogitsLengthMismatch { expected: usize, actual: usize },
    /// A logit was NaN or infinite.
    NonFiniteLogit { row: usize, column: u32 },
    /// A size did not fit host indexing.
    HostIndexOverflow,
}

impl From<SpeculativeError> for CpuSpeculativeError {
    fn from(error: SpeculativeError) -> Self {
        Self::Contract(error)
    }
}

impl fmt::Display for CpuSpeculativeError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(output, "speculative contract: {error}"),
            Self::ZeroVocabulary => output.write_str("verifier vocabulary size must be non-zero"),
            Self::TokenOutOfVocabulary { index, token } => {
                write!(
                    output,
                    "draft token {index} ({token}) is outside the vocabulary"
                )
            }
            Self::LogitsLengthMismatch { expected, actual } => {
                write!(
                    output,
                    "target logits have {actual} values, expected {expected}"
                )
            }
            Self::NonFiniteLogit { row, column } => {
                write!(
                    output,
                    "target logit at row {row} column {column} is not finite"
                )
            }
            Self::HostIndexOverflow => output.write_str("logits size does not fit host indexing"),
        }
    }
}

impl std::error::Error for CpuSpeculativeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VOCAB: u32 = 4;

    fn verifier() -> CpuGreedySpeculativeVerifierV1 {
        CpuGreedySpeculativeVerifierV1::new(VOCAB).unwrap()
    }

    fn schedule(min_confidence: f32) -> ConfidenceScheduleV1 {
        ConfidenceScheduleV1::new("caller.threshold", 1, min_confidence, 8).unwrap()
    }

    /// One-hot style logits whose greedy token is `token`.
    fn rows(tokens: &[u32]) -> Vec<f32> {
        tokens
            .iter()
            .flat_map(|&token| {
                (0..VOCAB).map(move |column| if column == token { 2.0 } else { -1.0 })
            })
            .collect()
    }

    #[test]
    fn greedy_logits_accept_matching_prefix_and_emit_correction() {
        let draft = SpeculativeDraftV1::new(vec![1, 2, 3], vec![0.9, 0.9, 0.9]).unwrap();
        let outcome = verifier()
            .verify(&draft, &schedule(0.5), &rows(&[1, 0, 3, 2]))
            .unwrap();
        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.rejected_at, Some(1));
        assert_eq!(outcome.emitted_tokens, vec![1, 0]);
    }

    #[test]
    fn full_accept_uses_bonus_row() {
        let draft = SpeculativeDraftV1::new(vec![1, 2], vec![1.0, 1.0]).unwrap();
        let outcome = verifier()
            .verify(&draft, &schedule(0.5), &rows(&[1, 2, 3]))
            .unwrap();
        assert_eq!(outcome.accepted, 2);
        assert_eq!(outcome.emitted_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn argmax_ties_choose_lowest_index() {
        assert_eq!(greedy_argmax(&[0.5, 1.0, 1.0, -2.0]), Ok(1));
        assert_eq!(greedy_argmax(&[3.0, 3.0]), Ok(0));
        assert_eq!(greedy_argmax(&[-0.0, 0.0]), Ok(0));
        let draft = SpeculativeDraftV1::new(vec![2], vec![1.0]).unwrap();
        let outcome = verifier()
            .verify(
                &draft,
                &schedule(0.5),
                &[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            .unwrap();
        // Row 0 ties between tokens 1 and 2; token 1 wins and rejects draft 2.
        assert_eq!(outcome.rejected_at, Some(0));
        assert_eq!(outcome.emitted_tokens, vec![1]);
    }

    #[test]
    fn schedule_controls_logits_rows_required() {
        let draft = SpeculativeDraftV1::new(vec![1, 2, 3], vec![0.9, 0.3, 0.9]).unwrap();
        let outcome = verifier()
            .verify(&draft, &schedule(0.5), &rows(&[1, 3]))
            .unwrap();
        assert_eq!(outcome.scheduled, 1);
        assert_eq!(outcome.emitted_tokens, vec![1, 3]);
        assert_eq!(
            verifier().verify(&draft, &schedule(0.5), &rows(&[1, 3, 3])),
            Err(CpuSpeculativeError::LogitsLengthMismatch {
                expected: 8,
                actual: 12
            })
        );
    }

    #[test]
    fn verification_is_deterministic_across_repeats() {
        let draft = SpeculativeDraftV1::new(vec![3, 3, 0, 1], vec![0.7, 0.6, 0.9, 0.8]).unwrap();
        let logits = rows(&[3, 3, 0, 2, 1]);
        let first = verifier().verify(&draft, &schedule(0.5), &logits).unwrap();
        for _ in 0..8 {
            assert_eq!(
                verifier().verify(&draft, &schedule(0.5), &logits).unwrap(),
                first
            );
        }
        assert_eq!(first.accepted, 3);
        assert_eq!(first.emitted_tokens, vec![3, 3, 0, 2]);
    }

    #[test]
    fn verify_and_record_accumulates_only_on_success() {
        let mut stats = SpeculativeAcceptanceStatsV1::default();
        let draft = SpeculativeDraftV1::new(vec![1, 2], vec![1.0, 1.0]).unwrap();
        verifier()
            .verify_and_record(&mut stats, &draft, &schedule(0.5), &rows(&[1, 0, 0]))
            .unwrap();
        let before = stats;
        assert!(verifier()
            .verify_and_record(&mut stats, &draft, &schedule(0.5), &rows(&[1]))
            .is_err());
        assert_eq!(stats, before);
        assert_eq!(stats.steps, 1);
        assert_eq!(stats.acceptance_ratio(), Some((1, 2)));
        assert_eq!(stats.emitted, 2);
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert_eq!(
            CpuGreedySpeculativeVerifierV1::new(0),
            Err(CpuSpeculativeError::ZeroVocabulary)
        );
        let draft = SpeculativeDraftV1::new(vec![1, 4], vec![1.0, 1.0]).unwrap();
        assert_eq!(
            verifier().verify(&draft, &schedule(0.5), &rows(&[1, 1, 1])),
            Err(CpuSpeculativeError::TokenOutOfVocabulary { index: 1, token: 4 })
        );
        let draft = SpeculativeDraftV1::new(vec![1], vec![1.0]).unwrap();
        let mut logits = rows(&[1, 2]);
        logits[6] = f32::NAN;
        assert_eq!(
            verifier().verify(&draft, &schedule(0.5), &logits),
            Err(CpuSpeculativeError::NonFiniteLogit { row: 1, column: 2 })
        );
    }
}
