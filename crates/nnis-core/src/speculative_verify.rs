//! Backend-neutral confidence-scheduled speculative verification surface.
//!
//! DSV41-4 defines how NNIS consumes a caller-supplied speculative draft
//! (token ids plus per-token confidences) under a caller-supplied confidence
//! schedule, and how the verified outcome and acceptance counts are recorded.
//!
//! NNIS does not produce drafts, choose confidence values, or choose schedule
//! parameters. The only v1 schedule mechanism verifies the leading draft
//! tokens whose confidence meets the caller's threshold, capped by the
//! caller's maximum. Verification compares the scheduled tokens with the
//! target model's tokens under its declared semantics (for example greedy
//! argmax, as in the CPU reference) and accepts the longest matching prefix.
//!
//! Records contain counts only. Acceptance counts are not a speed-up: any
//! throughput claim requires measured acceptance together with real runtime
//! cost on the exact environment.

use core::fmt;

/// Version of the NNIS speculative-verification surface contract.
pub const NNIS_SPECULATIVE_VERIFICATION_VERSION: u32 = 1;

/// Maximum draft tokens accepted in one verification step.
pub const MAX_SPECULATIVE_DRAFT_TOKENS: usize = 1024;

/// Maximum UTF-8 bytes accepted for schedule policy identities.
pub const MAX_SPECULATIVE_POLICY_ID_BYTES: usize = 128;

/// Caller-supplied speculative draft for one verification step.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeDraftV1 {
    tokens: Vec<u32>,
    confidences: Vec<f32>,
}

impl SpeculativeDraftV1 {
    /// Validate a non-empty draft with one finite confidence in `0..=1` per token.
    pub fn new(tokens: Vec<u32>, confidences: Vec<f32>) -> Result<Self, SpeculativeError> {
        if tokens.is_empty() {
            return Err(SpeculativeError::EmptyDraft);
        }
        if tokens.len() > MAX_SPECULATIVE_DRAFT_TOKENS {
            return Err(SpeculativeError::DraftTooLong {
                tokens: tokens.len(),
            });
        }
        if confidences.len() != tokens.len() {
            return Err(SpeculativeError::ConfidenceCountMismatch {
                tokens: tokens.len(),
                confidences: confidences.len(),
            });
        }
        if let Some(index) = confidences
            .iter()
            .position(|confidence| !(0.0..=1.0).contains(confidence))
        {
            return Err(SpeculativeError::InvalidConfidence { index });
        }
        Ok(Self {
            tokens,
            confidences,
        })
    }

    /// Draft token ids in proposal order.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Caller-supplied confidence for each draft token.
    pub fn confidences(&self) -> &[f32] {
        &self.confidences
    }

    /// Number of draft tokens (at most [`MAX_SPECULATIVE_DRAFT_TOKENS`]).
    pub fn len(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// Always false for a validated draft.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Caller-supplied confidence schedule.
///
/// v1 verifies the leading draft tokens whose confidence is at least
/// `min_confidence`, stopping at the first token below it, and never more than
/// `max_verify_tokens`. The parameters belong to the caller's policy.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceScheduleV1 {
    policy_id: String,
    policy_schema_version: u32,
    min_confidence: f32,
    max_verify_tokens: u32,
}

impl ConfidenceScheduleV1 {
    /// Validate caller-owned schedule parameters.
    pub fn new(
        policy_id: impl Into<String>,
        policy_schema_version: u32,
        min_confidence: f32,
        max_verify_tokens: u32,
    ) -> Result<Self, SpeculativeError> {
        let policy_id = policy_id.into();
        let trimmed = policy_id.trim();
        if trimmed.is_empty() {
            return Err(SpeculativeError::EmptyPolicyId);
        }
        if trimmed != policy_id {
            return Err(SpeculativeError::NonCanonicalPolicyId);
        }
        if policy_id.len() > MAX_SPECULATIVE_POLICY_ID_BYTES {
            return Err(SpeculativeError::PolicyIdTooLong {
                bytes: policy_id.len(),
            });
        }
        if policy_schema_version == 0 {
            return Err(SpeculativeError::ZeroPolicySchemaVersion);
        }
        if !(0.0..=1.0).contains(&min_confidence) {
            return Err(SpeculativeError::InvalidThreshold);
        }
        if max_verify_tokens == 0 {
            return Err(SpeculativeError::ZeroMaxVerifyTokens);
        }
        Ok(Self {
            policy_id,
            policy_schema_version,
            min_confidence,
            max_verify_tokens,
        })
    }

    /// Caller policy identity.
    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    /// Caller policy schema version.
    pub const fn policy_schema_version(&self) -> u32 {
        self.policy_schema_version
    }

    /// Caller confidence threshold.
    pub const fn min_confidence(&self) -> f32 {
        self.min_confidence
    }

    /// Caller cap on verified draft tokens per step.
    pub const fn max_verify_tokens(&self) -> u32 {
        self.max_verify_tokens
    }

    /// Number of leading draft tokens this schedule sends to verification.
    pub fn scheduled_len(&self, draft: &SpeculativeDraftV1) -> u32 {
        let leading = draft
            .confidences
            .iter()
            .take_while(|&&confidence| confidence >= self.min_confidence)
            .count() as u32;
        leading.min(self.max_verify_tokens)
    }
}

/// Outcome of one speculative verification step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeVerificationV1 {
    /// Draft tokens supplied.
    pub drafted: u32,
    /// Draft tokens sent to verification by the schedule.
    pub scheduled: u32,
    /// Leading scheduled tokens equal to the target tokens.
    pub accepted: u32,
    /// Scheduled index of the first mismatch, if any.
    pub rejected_at: Option<u32>,
    /// Accepted draft tokens followed by exactly one target token (the
    /// correction at the mismatch, or the next token after a full accept).
    pub emitted_tokens: Vec<u32>,
}

impl SpeculativeVerificationV1 {
    /// Draft tokens never verified because of the schedule.
    pub const fn skipped_by_schedule(&self) -> u32 {
        self.drafted - self.scheduled
    }
}

/// Verify a scheduled draft against the target model's tokens.
///
/// `target_tokens[i]` is the target's token at draft position `i` given the
/// accepted prefix before it, under the target's declared semantics. Exactly
/// `scheduled + 1` target tokens are required so that one target token is
/// always emitted. The outcome is a pure function of its inputs.
pub fn verify_scheduled_draft(
    draft: &SpeculativeDraftV1,
    schedule: &ConfidenceScheduleV1,
    target_tokens: &[u32],
) -> Result<SpeculativeVerificationV1, SpeculativeError> {
    let scheduled = schedule.scheduled_len(draft);
    let expected = scheduled as usize + 1;
    if target_tokens.len() != expected {
        return Err(SpeculativeError::TargetLengthMismatch {
            expected,
            actual: target_tokens.len(),
        });
    }
    let accepted = draft.tokens[..scheduled as usize]
        .iter()
        .zip(target_tokens)
        .take_while(|(draft_token, target_token)| draft_token == target_token)
        .count();
    let rejected_at = if accepted < scheduled as usize {
        Some(accepted as u32)
    } else {
        None
    };
    let mut emitted_tokens = draft.tokens[..accepted].to_vec();
    emitted_tokens.push(target_tokens[accepted]);
    Ok(SpeculativeVerificationV1 {
        drafted: draft.len(),
        scheduled,
        accepted: accepted as u32,
        rejected_at,
        emitted_tokens,
    })
}

/// Cumulative acceptance counts over verification steps.
///
/// Counts only: no timing, cost, or throughput is recorded or implied.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeculativeAcceptanceStatsV1 {
    /// Verification steps recorded.
    pub steps: u64,
    /// Draft tokens supplied.
    pub drafted: u64,
    /// Draft tokens scheduled for verification.
    pub scheduled: u64,
    /// Draft tokens accepted.
    pub accepted: u64,
    /// Tokens emitted (accepted plus one target token per step).
    pub emitted: u64,
    /// Draft tokens skipped by the schedule.
    pub skipped_by_schedule: u64,
    /// Steps with at least one rejected scheduled token.
    pub steps_with_rejection: u64,
}

impl SpeculativeAcceptanceStatsV1 {
    /// Add one verification outcome with checked arithmetic.
    pub fn record(&mut self, outcome: &SpeculativeVerificationV1) -> Result<(), SpeculativeError> {
        let mut next = *self;
        next.steps = add(next.steps, 1)?;
        next.drafted = add(next.drafted, u64::from(outcome.drafted))?;
        next.scheduled = add(next.scheduled, u64::from(outcome.scheduled))?;
        next.accepted = add(next.accepted, u64::from(outcome.accepted))?;
        next.emitted = add(next.emitted, outcome.emitted_tokens.len() as u64)?;
        next.skipped_by_schedule = add(
            next.skipped_by_schedule,
            u64::from(outcome.skipped_by_schedule()),
        )?;
        if outcome.rejected_at.is_some() {
            next.steps_with_rejection = add(next.steps_with_rejection, 1)?;
        }
        *self = next;
        Ok(())
    }

    /// Exact acceptance ratio `accepted / scheduled`, or `None` if nothing
    /// was scheduled.
    pub fn acceptance_ratio(&self) -> Option<(u64, u64)> {
        if self.scheduled == 0 {
            None
        } else {
            Some((self.accepted, self.scheduled))
        }
    }
}

fn add(left: u64, right: u64) -> Result<u64, SpeculativeError> {
    left.checked_add(right)
        .ok_or(SpeculativeError::CounterOverflow)
}

/// Fail-closed speculative-verification errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculativeError {
    /// Draft contained no tokens.
    EmptyDraft,
    /// Draft exceeded [`MAX_SPECULATIVE_DRAFT_TOKENS`].
    DraftTooLong { tokens: usize },
    /// Token and confidence counts differed.
    ConfidenceCountMismatch { tokens: usize, confidences: usize },
    /// A confidence was NaN or outside `0..=1`.
    InvalidConfidence { index: usize },
    /// Policy id was blank.
    EmptyPolicyId,
    /// Policy id had leading or trailing whitespace.
    NonCanonicalPolicyId,
    /// Policy id exceeded [`MAX_SPECULATIVE_POLICY_ID_BYTES`].
    PolicyIdTooLong { bytes: usize },
    /// Policy schema version zero is not valid.
    ZeroPolicySchemaVersion,
    /// Threshold was NaN or outside `0..=1`.
    InvalidThreshold,
    /// Schedule allowed zero verified tokens.
    ZeroMaxVerifyTokens,
    /// Target token count was not `scheduled + 1`.
    TargetLengthMismatch { expected: usize, actual: usize },
    /// Cumulative counter overflowed.
    CounterOverflow,
}

impl fmt::Display for SpeculativeError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDraft => output.write_str("speculative draft must not be empty"),
            Self::DraftTooLong { tokens } => write!(
                output,
                "speculative draft has {tokens} tokens, maximum is {MAX_SPECULATIVE_DRAFT_TOKENS}"
            ),
            Self::ConfidenceCountMismatch {
                tokens,
                confidences,
            } => write!(
                output,
                "speculative draft has {tokens} tokens but {confidences} confidences"
            ),
            Self::InvalidConfidence { index } => {
                write!(output, "draft confidence {index} is not a finite value in 0..=1")
            }
            Self::EmptyPolicyId => output.write_str("schedule policy id must not be empty"),
            Self::NonCanonicalPolicyId => {
                output.write_str("schedule policy id must not have leading or trailing whitespace")
            }
            Self::PolicyIdTooLong { bytes } => write!(
                output,
                "schedule policy id uses {bytes} bytes, maximum is {MAX_SPECULATIVE_POLICY_ID_BYTES}"
            ),
            Self::ZeroPolicySchemaVersion => {
                output.write_str("schedule policy schema version must be non-zero")
            }
            Self::InvalidThreshold => {
                output.write_str("schedule confidence threshold must be a finite value in 0..=1")
            }
            Self::ZeroMaxVerifyTokens => {
                output.write_str("schedule must allow at least one verified token")
            }
            Self::TargetLengthMismatch { expected, actual } => write!(
                output,
                "verification needs {expected} target tokens, got {actual}"
            ),
            Self::CounterOverflow => output.write_str("speculative acceptance counter overflow"),
        }
    }
}

impl std::error::Error for SpeculativeError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(min_confidence: f32, max: u32) -> ConfidenceScheduleV1 {
        ConfidenceScheduleV1::new("caller.threshold", 1, min_confidence, max).unwrap()
    }

    fn draft(tokens: &[u32], confidences: &[f32]) -> SpeculativeDraftV1 {
        SpeculativeDraftV1::new(tokens.to_vec(), confidences.to_vec()).unwrap()
    }

    #[test]
    fn schedule_takes_leading_tokens_meeting_threshold_up_to_cap() {
        let draft = draft(&[5, 6, 7, 8], &[0.9, 0.8, 0.4, 0.95]);
        assert_eq!(schedule(0.5, 8).scheduled_len(&draft), 2);
        assert_eq!(schedule(0.5, 1).scheduled_len(&draft), 1);
        assert_eq!(schedule(0.0, 8).scheduled_len(&draft), 4);
        assert_eq!(schedule(0.95, 8).scheduled_len(&draft), 0);
        // Exact threshold equality is scheduled.
        assert_eq!(schedule(0.9, 8).scheduled_len(&draft), 1);
    }

    #[test]
    fn full_accept_emits_bonus_target_token() {
        let draft = draft(&[5, 6, 7], &[1.0, 1.0, 1.0]);
        let outcome = verify_scheduled_draft(&draft, &schedule(0.5, 8), &[5, 6, 7, 9]).unwrap();
        assert_eq!(outcome.scheduled, 3);
        assert_eq!(outcome.accepted, 3);
        assert_eq!(outcome.rejected_at, None);
        assert_eq!(outcome.emitted_tokens, vec![5, 6, 7, 9]);
        assert_eq!(outcome.skipped_by_schedule(), 0);
    }

    #[test]
    fn first_mismatch_emits_target_correction_and_stops() {
        let draft = draft(&[5, 6, 7, 8], &[1.0, 1.0, 1.0, 0.1]);
        let outcome = verify_scheduled_draft(&draft, &schedule(0.5, 8), &[5, 4, 7, 2]).unwrap();
        assert_eq!(outcome.drafted, 4);
        assert_eq!(outcome.scheduled, 3);
        assert_eq!(outcome.accepted, 1);
        assert_eq!(outcome.rejected_at, Some(1));
        // Later matches after a mismatch are never accepted.
        assert_eq!(outcome.emitted_tokens, vec![5, 4]);
        assert_eq!(outcome.skipped_by_schedule(), 1);
    }

    #[test]
    fn zero_scheduled_tokens_is_a_plain_target_step() {
        let draft = draft(&[5, 6], &[0.1, 0.9]);
        let outcome = verify_scheduled_draft(&draft, &schedule(0.5, 8), &[3]).unwrap();
        assert_eq!(outcome.scheduled, 0);
        assert_eq!(outcome.accepted, 0);
        assert_eq!(outcome.rejected_at, None);
        assert_eq!(outcome.emitted_tokens, vec![3]);
        assert_eq!(outcome.skipped_by_schedule(), 2);
    }

    #[test]
    fn target_length_must_be_scheduled_plus_one() {
        let draft = draft(&[5, 6], &[1.0, 1.0]);
        assert_eq!(
            verify_scheduled_draft(&draft, &schedule(0.5, 8), &[5, 6]),
            Err(SpeculativeError::TargetLengthMismatch {
                expected: 3,
                actual: 2
            })
        );
    }

    #[test]
    fn stats_accumulate_counts_and_exact_ratio() {
        let mut stats = SpeculativeAcceptanceStatsV1::default();
        assert_eq!(stats.acceptance_ratio(), None);
        let plan = schedule(0.5, 8);
        let first = draft(&[1, 2, 3], &[1.0, 1.0, 1.0]);
        stats
            .record(&verify_scheduled_draft(&first, &plan, &[1, 2, 3, 4]).unwrap())
            .unwrap();
        let second = draft(&[1, 2, 3, 4], &[1.0, 1.0, 0.2, 1.0]);
        stats
            .record(&verify_scheduled_draft(&second, &plan, &[9, 2, 3]).unwrap())
            .unwrap();
        assert_eq!(
            stats,
            SpeculativeAcceptanceStatsV1 {
                steps: 2,
                drafted: 7,
                scheduled: 5,
                accepted: 3,
                emitted: 5,
                skipped_by_schedule: 2,
                steps_with_rejection: 1,
            }
        );
        assert_eq!(stats.acceptance_ratio(), Some((3, 5)));

        let mut saturated = SpeculativeAcceptanceStatsV1 {
            steps: u64::MAX,
            ..SpeculativeAcceptanceStatsV1::default()
        };
        let before = saturated;
        assert_eq!(
            saturated.record(&verify_scheduled_draft(&first, &plan, &[1, 2, 3, 4]).unwrap()),
            Err(SpeculativeError::CounterOverflow)
        );
        assert_eq!(saturated, before);
    }

    #[test]
    fn malformed_drafts_and_schedules_fail_closed() {
        assert_eq!(
            SpeculativeDraftV1::new(Vec::new(), Vec::new()),
            Err(SpeculativeError::EmptyDraft)
        );
        assert_eq!(
            SpeculativeDraftV1::new(
                vec![0; MAX_SPECULATIVE_DRAFT_TOKENS + 1],
                vec![1.0; MAX_SPECULATIVE_DRAFT_TOKENS + 1]
            ),
            Err(SpeculativeError::DraftTooLong {
                tokens: MAX_SPECULATIVE_DRAFT_TOKENS + 1
            })
        );
        assert_eq!(
            SpeculativeDraftV1::new(vec![1, 2], vec![1.0]),
            Err(SpeculativeError::ConfidenceCountMismatch {
                tokens: 2,
                confidences: 1
            })
        );
        for bad in [f32::NAN, -0.1, 1.5, f32::INFINITY] {
            assert_eq!(
                SpeculativeDraftV1::new(vec![1, 2], vec![0.5, bad]),
                Err(SpeculativeError::InvalidConfidence { index: 1 })
            );
            assert_eq!(
                ConfidenceScheduleV1::new("p", 1, bad, 4),
                Err(SpeculativeError::InvalidThreshold)
            );
        }
        assert_eq!(
            ConfidenceScheduleV1::new("", 1, 0.5, 4),
            Err(SpeculativeError::EmptyPolicyId)
        );
        assert_eq!(
            ConfidenceScheduleV1::new("p ", 1, 0.5, 4),
            Err(SpeculativeError::NonCanonicalPolicyId)
        );
        assert_eq!(
            ConfidenceScheduleV1::new("p".repeat(129), 1, 0.5, 4),
            Err(SpeculativeError::PolicyIdTooLong { bytes: 129 })
        );
        assert_eq!(
            ConfidenceScheduleV1::new("p", 0, 0.5, 4),
            Err(SpeculativeError::ZeroPolicySchemaVersion)
        );
        assert_eq!(
            ConfidenceScheduleV1::new("p", 1, 0.5, 0),
            Err(SpeculativeError::ZeroMaxVerifyTokens)
        );
    }
}
