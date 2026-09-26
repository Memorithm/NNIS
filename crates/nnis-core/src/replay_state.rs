//! Backend-neutral replayable-state provider identity contract.
//!
//! DSV41-0 defines only source identity and bounded-window validation. It does
//! not define model/KV reconstruction semantics, move bytes, allocate memory,
//! or claim equivalence with an external bounded-replay implementation.

use core::fmt;

/// Version of the NNIS replay-state provider contract.
pub const NNIS_REPLAY_STATE_PROVIDER_VERSION: u32 = 1;

/// Maximum UTF-8 bytes accepted for provider/source/representation identities.
pub const MAX_REPLAY_ID_BYTES: usize = 128;

/// Exact representation/materialization identity of a replay source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReplayRepresentationIdentityV1 {
    id: String,
    schema_version: u32,
    materialization_epoch: u64,
}

impl ReplayRepresentationIdentityV1 {
    /// Construct a representation identity with a non-zero schema version.
    pub fn new(
        id: impl Into<String>,
        schema_version: u32,
        materialization_epoch: u64,
    ) -> Result<Self, ReplayIdentityError> {
        let id = id.into();
        validate_id("representation_id", &id)?;
        if schema_version == 0 {
            return Err(ReplayIdentityError::ZeroRepresentationSchemaVersion);
        }
        Ok(Self {
            id,
            schema_version,
            materialization_epoch,
        })
    }

    /// Stable representation contract id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Representation contract schema version.
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Materialization epoch bound to the replay source.
    pub const fn materialization_epoch(&self) -> u64 {
        self.materialization_epoch
    }
}

/// Exact logical source range made available by one runtime provider.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReplaySourceIdentityV1 {
    provider_id: String,
    source_id: String,
    source_generation: u64,
    representation: ReplayRepresentationIdentityV1,
    logical_start_position: u64,
    logical_end_position: u64,
}

impl ReplaySourceIdentityV1 {
    /// Construct an exact source identity.
    pub fn new(
        provider_id: impl Into<String>,
        source_id: impl Into<String>,
        source_generation: u64,
        representation: ReplayRepresentationIdentityV1,
        logical_start_position: u64,
        logical_end_position: u64,
    ) -> Result<Self, ReplayIdentityError> {
        let provider_id = provider_id.into();
        let source_id = source_id.into();
        validate_id("provider_id", &provider_id)?;
        validate_id("source_id", &source_id)?;
        if logical_start_position > logical_end_position {
            return Err(ReplayIdentityError::InvalidSourceRange {
                start: logical_start_position,
                end: logical_end_position,
            });
        }
        Ok(Self {
            provider_id,
            source_id,
            source_generation,
            representation,
            logical_start_position,
            logical_end_position,
        })
    }

    /// Stable provider identity.
    pub fn provider_id(&self) -> &str {
        &self.provider_id
    }

    /// Stable source identity within the provider.
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    /// Monotonic source generation supplied by the provider.
    pub const fn source_generation(&self) -> u64 {
        self.source_generation
    }

    /// Representation/materialization identity.
    pub const fn representation(&self) -> &ReplayRepresentationIdentityV1 {
        &self.representation
    }

    /// Inclusive first logical position available from this source.
    pub const fn logical_start_position(&self) -> u64 {
        self.logical_start_position
    }

    /// Inclusive last logical position available from this source.
    pub const fn logical_end_position(&self) -> u64 {
        self.logical_end_position
    }
}

/// One bounded replay-window request tied to an exact source identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayWindowRequestV1 {
    source: ReplaySourceIdentityV1,
    logical_start_position: u64,
    logical_end_position: u64,
}

impl ReplayWindowRequestV1 {
    /// Construct a non-empty window contained by the declared source range.
    pub fn new(
        source: ReplaySourceIdentityV1,
        logical_start_position: u64,
        logical_end_position: u64,
    ) -> Result<Self, ReplayIdentityError> {
        if logical_start_position > logical_end_position {
            return Err(ReplayIdentityError::InvalidWindowRange {
                start: logical_start_position,
                end: logical_end_position,
            });
        }
        if logical_start_position < source.logical_start_position
            || logical_end_position > source.logical_end_position
        {
            return Err(ReplayIdentityError::WindowOutsideSource {
                source_start: source.logical_start_position,
                source_end: source.logical_end_position,
                requested_start: logical_start_position,
                requested_end: logical_end_position,
            });
        }
        Ok(Self {
            source,
            logical_start_position,
            logical_end_position,
        })
    }

    /// Exact source identity expected by this request.
    pub const fn source(&self) -> &ReplaySourceIdentityV1 {
        &self.source
    }

    /// Inclusive requested start position.
    pub const fn logical_start_position(&self) -> u64 {
        self.logical_start_position
    }

    /// Inclusive requested end position.
    pub const fn logical_end_position(&self) -> u64 {
        self.logical_end_position
    }

    /// Exact number of logical items in the replay window.
    pub fn logical_items(&self) -> Result<u64, ReplayIdentityError> {
        self.logical_end_position
            .checked_sub(self.logical_start_position)
            .and_then(|delta| delta.checked_add(1))
            .ok_or(ReplayIdentityError::PositionOverflow)
    }
}

/// Runtime-neutral provider of replay-source identity.
///
/// The trait intentionally exposes no payload type in DSV41-0. Later CPU/WGPU
/// execution slices may add explicit data-movement contracts after the source
/// identity boundary is qualified.
pub trait ReplayStateProviderV1 {
    /// Exact source identity currently exposed by this provider.
    fn replay_source_identity(&self) -> &ReplaySourceIdentityV1;

    /// Validate that a request still targets the exact current source.
    fn validate_replay_window(
        &self,
        request: &ReplayWindowRequestV1,
    ) -> Result<(), ReplayIdentityError> {
        if self.replay_source_identity() != request.source() {
            return Err(ReplayIdentityError::SourceIdentityMismatch);
        }
        Ok(())
    }
}

/// Fail-closed replay-provider identity errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayIdentityError {
    /// One opaque identity was blank.
    EmptyId { field: &'static str },
    /// One opaque identity exceeded the bounded contract.
    IdTooLong { field: &'static str, bytes: usize },
    /// One opaque identity was not in canonical trimmed form.
    NonCanonicalId { field: &'static str },
    /// Representation schema version zero is not valid.
    ZeroRepresentationSchemaVersion,
    /// Source range had start greater than end.
    InvalidSourceRange { start: u64, end: u64 },
    /// Requested window had start greater than end.
    InvalidWindowRange { start: u64, end: u64 },
    /// Requested window was not fully contained by the declared source.
    WindowOutsideSource {
        source_start: u64,
        source_end: u64,
        requested_start: u64,
        requested_end: u64,
    },
    /// Window-length arithmetic overflowed.
    PositionOverflow,
    /// Provider identity changed after the request was constructed.
    SourceIdentityMismatch,
}

impl fmt::Display for ReplayIdentityError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId { field } => write!(output, "{field} must not be empty"),
            Self::IdTooLong { field, bytes } => {
                write!(output, "{field} uses {bytes} bytes, maximum is {MAX_REPLAY_ID_BYTES}")
            }
            Self::NonCanonicalId { field } => {
                write!(output, "{field} must not have leading or trailing whitespace")
            }
            Self::ZeroRepresentationSchemaVersion => {
                output.write_str("replay representation schema version must be non-zero")
            }
            Self::InvalidSourceRange { start, end } => {
                write!(output, "invalid replay source range {start}..={end}")
            }
            Self::InvalidWindowRange { start, end } => {
                write!(output, "invalid replay window {start}..={end}")
            }
            Self::WindowOutsideSource {
                source_start,
                source_end,
                requested_start,
                requested_end,
            } => write!(
                output,
                "replay window {requested_start}..={requested_end} is outside source {source_start}..={source_end}"
            ),
            Self::PositionOverflow => output.write_str("replay logical-position arithmetic overflow"),
            Self::SourceIdentityMismatch => {
                output.write_str("replay request source identity no longer matches provider")
            }
        }
    }
}

impl std::error::Error for ReplayIdentityError {}

fn validate_id(field: &'static str, value: &str) -> Result<(), ReplayIdentityError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ReplayIdentityError::EmptyId { field });
    }
    if trimmed != value {
        return Err(ReplayIdentityError::NonCanonicalId { field });
    }
    if value.len() > MAX_REPLAY_ID_BYTES {
        return Err(ReplayIdentityError::IdTooLong {
            field,
            bytes: value.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(generation: u64, epoch: u64) -> ReplaySourceIdentityV1 {
        ReplaySourceIdentityV1::new(
            "cpu-reference",
            "session-prefix",
            generation,
            ReplayRepresentationIdentityV1::new("slha.replay-source", 1, epoch).unwrap(),
            64,
            255,
        )
        .unwrap()
    }

    struct Provider {
        identity: ReplaySourceIdentityV1,
    }

    impl ReplayStateProviderV1 for Provider {
        fn replay_source_identity(&self) -> &ReplaySourceIdentityV1 {
            &self.identity
        }
    }

    #[test]
    fn bounded_window_is_tied_to_exact_source_identity() {
        let identity = source(3, 7);
        let request = ReplayWindowRequestV1::new(identity.clone(), 128, 255).unwrap();
        assert_eq!(request.logical_items().unwrap(), 128);

        let provider = Provider { identity };
        provider.validate_replay_window(&request).unwrap();
    }

    #[test]
    fn generation_or_materialization_epoch_drift_fails_closed() {
        let request = ReplayWindowRequestV1::new(source(3, 7), 192, 255).unwrap();

        let generation_drift = Provider {
            identity: source(4, 7),
        };
        assert_eq!(
            generation_drift.validate_replay_window(&request),
            Err(ReplayIdentityError::SourceIdentityMismatch)
        );

        let epoch_drift = Provider {
            identity: source(3, 8),
        };
        assert_eq!(
            epoch_drift.validate_replay_window(&request),
            Err(ReplayIdentityError::SourceIdentityMismatch)
        );
    }

    #[test]
    fn window_must_be_contained_by_source_range() {
        let identity = source(1, 1);
        assert!(matches!(
            ReplayWindowRequestV1::new(identity, 63, 127),
            Err(ReplayIdentityError::WindowOutsideSource { .. })
        ));
    }

    #[test]
    fn replay_ids_must_be_canonical_trimmed_strings() {
        assert!(matches!(
            ReplayRepresentationIdentityV1::new(" rep", 1, 1),
            Err(ReplayIdentityError::NonCanonicalId {
                field: "representation_id"
            })
        ));
        assert!(matches!(
            ReplaySourceIdentityV1::new(
                "cpu ",
                "source",
                0,
                ReplayRepresentationIdentityV1::new("rep", 1, 1).unwrap(),
                0,
                1,
            ),
            Err(ReplayIdentityError::NonCanonicalId {
                field: "provider_id"
            })
        ));
        assert!(matches!(
            ReplaySourceIdentityV1::new(
                "cpu",
                "\tsource",
                0,
                ReplayRepresentationIdentityV1::new("rep", 1, 1).unwrap(),
                0,
                1,
            ),
            Err(ReplayIdentityError::NonCanonicalId {
                field: "source_id"
            })
        ));
    }

    #[test]
    fn malformed_id_schema_and_ranges_are_rejected() {
        assert!(matches!(
            ReplayRepresentationIdentityV1::new("rep", 0, 1),
            Err(ReplayIdentityError::ZeroRepresentationSchemaVersion)
        ));
        assert!(matches!(
            ReplaySourceIdentityV1::new(
                " ",
                "source",
                0,
                ReplayRepresentationIdentityV1::new("rep", 1, 1).unwrap(),
                0,
                1,
            ),
            Err(ReplayIdentityError::EmptyId {
                field: "provider_id"
            })
        ));
        assert!(matches!(
            ReplaySourceIdentityV1::new(
                "cpu",
                "source",
                0,
                ReplayRepresentationIdentityV1::new("rep", 1, 1).unwrap(),
                9,
                8,
            ),
            Err(ReplayIdentityError::InvalidSourceRange { start: 9, end: 8 })
        ));
    }
}
