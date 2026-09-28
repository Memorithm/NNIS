//! DSV41-1 CPU reference path for bounded recent-window replay.
//!
//! [`CpuReplaySourceV1`] owns an immutable dense row-major F32 payload for one
//! exact [`ReplaySourceIdentityV1`]. A replay copies the requested inclusive
//! logical window bit-for-bit after the request has been validated against the
//! provider's current source identity. This is the dense reference oracle for
//! later portable (WGPU) replay paths; it performs no reconstruction,
//! compression, eviction, paging, or model-specific interpretation, and says
//! nothing about physical memory release, latency, or throughput.

use core::fmt;

use nnis_core::replay_state::{
    ReplayIdentityError, ReplaySourceIdentityV1, ReplayStateProviderV1, ReplayWindowRequestV1,
};

/// Immutable dense CPU replay source bound to one exact source identity.
///
/// Row `i` of the payload holds the `row_width` values for logical position
/// `identity.logical_start_position() + i`.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuReplaySourceV1 {
    identity: ReplaySourceIdentityV1,
    row_width: usize,
    rows: Vec<f32>,
}

impl CpuReplaySourceV1 {
    /// Bind a dense finite row-major payload to an exact source identity.
    ///
    /// The payload must contain exactly one row of `row_width` finite values
    /// for every logical position declared by `identity`.
    pub fn new(
        identity: ReplaySourceIdentityV1,
        row_width: usize,
        rows: Vec<f32>,
    ) -> Result<Self, CpuReplayError> {
        if row_width == 0 {
            return Err(CpuReplayError::ZeroRowWidth);
        }
        let items = host_index(identity.logical_items()?)?;
        let expected = items
            .checked_mul(row_width)
            .ok_or(CpuReplayError::HostIndexOverflow)?;
        if rows.len() != expected {
            return Err(CpuReplayError::PayloadLengthMismatch {
                expected,
                actual: rows.len(),
            });
        }
        if let Some(index) = rows.iter().position(|value| !value.is_finite()) {
            return Err(CpuReplayError::NonFiniteValue { index });
        }
        Ok(Self {
            identity,
            row_width,
            rows,
        })
    }

    /// Number of F32 values stored per logical position.
    pub const fn row_width(&self) -> usize {
        self.row_width
    }

    /// Dense row-major payload for the full declared source range.
    pub fn rows(&self) -> &[f32] {
        &self.rows
    }

    /// Copy the rows of a validated replay window.
    ///
    /// Fails closed if the request was built for a different source identity,
    /// including any provider, source, generation, representation, epoch, or
    /// range drift.
    pub fn replay_window(
        &self,
        request: &ReplayWindowRequestV1,
    ) -> Result<Vec<f32>, CpuReplayError> {
        self.validate_replay_window(request)?;
        let first_row = request
            .logical_start_position()
            .checked_sub(self.identity.logical_start_position())
            .ok_or(ReplayIdentityError::PositionOverflow)?;
        let first_row = host_index(first_row)?;
        let row_count = host_index(request.logical_items()?)?;
        let start = first_row
            .checked_mul(self.row_width)
            .ok_or(CpuReplayError::HostIndexOverflow)?;
        let len = row_count
            .checked_mul(self.row_width)
            .ok_or(CpuReplayError::HostIndexOverflow)?;
        let end = start
            .checked_add(len)
            .ok_or(CpuReplayError::HostIndexOverflow)?;
        let window = self
            .rows
            .get(start..end)
            .ok_or(CpuReplayError::HostIndexOverflow)?;
        Ok(window.to_vec())
    }

    /// Resolve and copy the most recent `window_items` logical positions.
    ///
    /// `window_items` is caller-supplied policy; it is neither chosen nor
    /// clamped here. The resolved request is returned with the copied rows so
    /// the caller can bind the replay to its exact logical range.
    pub fn replay_recent_window(
        &self,
        window_items: u64,
    ) -> Result<(ReplayWindowRequestV1, Vec<f32>), CpuReplayError> {
        let request = ReplayWindowRequestV1::recent(self.identity.clone(), window_items)?;
        let rows = self.replay_window(&request)?;
        Ok((request, rows))
    }
}

impl ReplayStateProviderV1 for CpuReplaySourceV1 {
    fn replay_source_identity(&self) -> &ReplaySourceIdentityV1 {
        &self.identity
    }
}

/// Fail-closed CPU replay errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuReplayError {
    /// Backend-neutral identity or window validation failed.
    Identity(ReplayIdentityError),
    /// A replay source must store at least one value per logical position.
    ZeroRowWidth,
    /// Payload length did not match declared items times row width.
    PayloadLengthMismatch { expected: usize, actual: usize },
    /// Payload contained a NaN or infinity at this flat index.
    NonFiniteValue { index: usize },
    /// A logical count or offset does not fit host `usize` indexing.
    HostIndexOverflow,
}

impl From<ReplayIdentityError> for CpuReplayError {
    fn from(error: ReplayIdentityError) -> Self {
        Self::Identity(error)
    }
}

impl fmt::Display for CpuReplayError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity(error) => write!(output, "replay identity: {error}"),
            Self::ZeroRowWidth => output.write_str("CPU replay row width must be non-zero"),
            Self::PayloadLengthMismatch { expected, actual } => write!(
                output,
                "CPU replay payload has {actual} values, expected {expected}"
            ),
            Self::NonFiniteValue { index } => {
                write!(output, "CPU replay payload value {index} is not finite")
            }
            Self::HostIndexOverflow => {
                output.write_str("CPU replay logical range does not fit host indexing")
            }
        }
    }
}

impl std::error::Error for CpuReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Identity(error) => Some(error),
            _ => None,
        }
    }
}

fn host_index(value: u64) -> Result<usize, CpuReplayError> {
    usize::try_from(value).map_err(|_| CpuReplayError::HostIndexOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_core::replay_state::ReplayRepresentationIdentityV1;

    const WIDTH: usize = 3;

    fn identity(generation: u64, start: u64, end: u64) -> ReplaySourceIdentityV1 {
        ReplaySourceIdentityV1::new(
            "nnis-cpu-reference",
            "session-kv",
            generation,
            ReplayRepresentationIdentityV1::new("dense.f32.rows", 1, 0).unwrap(),
            start,
            end,
        )
        .unwrap()
    }

    fn payload(items: usize) -> Vec<f32> {
        (0..items * WIDTH)
            .map(|value| value as f32 * 0.5 - 1.0)
            .collect()
    }

    #[test]
    fn recent_window_copies_exact_trailing_rows() {
        let source = CpuReplaySourceV1::new(identity(1, 10, 17), WIDTH, payload(8)).unwrap();
        let (request, rows) = source.replay_recent_window(3).unwrap();
        assert_eq!(request.logical_start_position(), 15);
        assert_eq!(request.logical_end_position(), 17);
        assert_eq!(rows, source.rows()[5 * WIDTH..].to_vec());
    }

    #[test]
    fn replay_preserves_bits_including_signed_zero() {
        let mut values = payload(4);
        values[WIDTH * 3] = -0.0;
        values[WIDTH * 3 + 1] = f32::MIN_POSITIVE / 2.0;
        let source = CpuReplaySourceV1::new(identity(1, 0, 3), WIDTH, values.clone()).unwrap();
        let (_, rows) = source.replay_recent_window(1).unwrap();
        let expected: Vec<u32> = values[WIDTH * 3..].iter().map(|v| v.to_bits()).collect();
        let actual: Vec<u32> = rows.iter().map(|v| v.to_bits()).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn explicit_interior_window_matches_row_offsets() {
        let source = CpuReplaySourceV1::new(identity(2, 100, 109), WIDTH, payload(10)).unwrap();
        let request = ReplayWindowRequestV1::new(identity(2, 100, 109), 103, 105).unwrap();
        let rows = source.replay_window(&request).unwrap();
        assert_eq!(rows, source.rows()[3 * WIDTH..6 * WIDTH].to_vec());
    }

    #[test]
    fn full_recent_window_equals_whole_payload() {
        let source = CpuReplaySourceV1::new(identity(1, 0, 4), WIDTH, payload(5)).unwrap();
        let (_, rows) = source.replay_recent_window(5).unwrap();
        assert_eq!(rows, source.rows().to_vec());
    }

    #[test]
    fn empty_or_oversized_recent_windows_fail_closed() {
        let source = CpuReplaySourceV1::new(identity(1, 0, 4), WIDTH, payload(5)).unwrap();
        assert_eq!(
            source.replay_recent_window(0),
            Err(CpuReplayError::Identity(
                ReplayIdentityError::ZeroWindowItems
            ))
        );
        assert_eq!(
            source.replay_recent_window(6),
            Err(CpuReplayError::Identity(
                ReplayIdentityError::RecentWindowExceedsSource {
                    available: 5,
                    requested: 6,
                }
            ))
        );
    }

    #[test]
    fn stale_request_fails_closed_on_generation_drift() {
        let stale = ReplayWindowRequestV1::recent(identity(1, 0, 4), 2).unwrap();
        let current = CpuReplaySourceV1::new(identity(2, 0, 4), WIDTH, payload(5)).unwrap();
        assert_eq!(
            current.replay_window(&stale),
            Err(CpuReplayError::Identity(
                ReplayIdentityError::SourceIdentityMismatch
            ))
        );
    }

    #[test]
    fn malformed_payloads_are_rejected() {
        assert_eq!(
            CpuReplaySourceV1::new(identity(1, 0, 4), 0, Vec::new()),
            Err(CpuReplayError::ZeroRowWidth)
        );
        assert_eq!(
            CpuReplaySourceV1::new(identity(1, 0, 4), WIDTH, payload(4)),
            Err(CpuReplayError::PayloadLengthMismatch {
                expected: 15,
                actual: 12,
            })
        );
        let mut values = payload(5);
        values[7] = f32::NAN;
        assert_eq!(
            CpuReplaySourceV1::new(identity(1, 0, 4), WIDTH, values),
            Err(CpuReplayError::NonFiniteValue { index: 7 })
        );
        let mut values = payload(5);
        values[14] = f32::NEG_INFINITY;
        assert_eq!(
            CpuReplaySourceV1::new(identity(1, 0, 4), WIDTH, values),
            Err(CpuReplayError::NonFiniteValue { index: 14 })
        );
    }

    #[test]
    fn full_u64_source_range_cannot_be_materialized_on_host() {
        assert_eq!(
            CpuReplaySourceV1::new(identity(1, 0, u64::MAX), WIDTH, Vec::new()),
            Err(CpuReplayError::Identity(
                ReplayIdentityError::PositionOverflow
            ))
        );
    }
}
