//! Versioned record format and fail-closed validator for portable adapter evidence.
//!
//! A [`PortableAdapterEvidenceV1`] records one run of a backend's parity
//! suites against the CPU reference on one exactly identified adapter, at one
//! exact source commit. It is meant to be produced later by a person on real
//! hardware with the `nnis-wgpu` `wgpu_adapter_evidence` example and reviewed
//! before anything is cited.
//!
//! The record stores only what was run and what was observed: suite outcomes,
//! the number of compared checks and the number of mismatches against each
//! suite's declared tolerance. It has no timing, throughput, memory or energy
//! fields, so no performance can be recorded through it.
//!
//! [`validate_adapter_evidence`] rejects malformed records with an error and
//! otherwise returns a verdict:
//!
//! - any failed suite gives [`AdapterEvidenceVerdictV1::Failed`];
//! - a required suite that is missing or skipped gives
//!   [`AdapterEvidenceVerdictV1::Incomplete`];
//! - a dirty worktree gives [`AdapterEvidenceVerdictV1::DirtyWorktree`];
//! - a software adapter gives
//!   [`AdapterEvidenceVerdictV1::CodePathOnlySoftwareAdapter`];
//! - an unclassified adapter gives [`AdapterEvidenceVerdictV1::UnclassifiedAdapter`];
//! - only a hardware adapter with every required suite passing and zero
//!   mismatches gives [`AdapterEvidenceVerdictV1::HardwareParityObserved`].
//!
//! A record that claims a hardware class for a CPU device type or a known
//! software implementation (llvmpipe, lavapipe, SwiftShader, softpipe, WARP)
//! is rejected. Even `HardwareParityObserved` is scoped to the exact adapter,
//! driver, commit and toolchain recorded. It is not a performance result, not
//! a cross-device generalization and not a promotion decision.

use core::fmt;

use crate::json::{self, Json};

/// `kind` tag of an adapter evidence record.
pub const PORTABLE_ADAPTER_EVIDENCE_KIND: &str = "nnis.portable.adapter-evidence";

/// Schema version written and accepted.
pub const PORTABLE_ADAPTER_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Maximum byte length of any free-text field.
pub const MAX_EVIDENCE_TEXT_BYTES: usize = 512;

/// Maximum number of suites in one record.
pub const MAX_EVIDENCE_SUITES: usize = 64;

/// Adapter names or drivers containing these markers are software adapters.
pub const SOFTWARE_ADAPTER_MARKERS: [&str; 5] =
    ["llvmpipe", "lavapipe", "swiftshader", "softpipe", "warp"];

/// Adapter classification recorded by the harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdapterClassV1 {
    /// Discrete, integrated or virtual GPU reported by the driver.
    Hardware,
    /// CPU device type or a known software implementation.
    Software,
    /// Device type not reported.
    Unknown,
}

impl AdapterClassV1 {
    /// Stable JSON name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Hardware => "hardware",
            Self::Software => "software",
            Self::Unknown => "unknown",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "hardware" => Some(Self::Hardware),
            "software" => Some(Self::Software),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }
}

/// Outcome of one parity suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuiteOutcomeV1 {
    /// Every check met the declared tolerance.
    Pass,
    /// At least one check missed the tolerance, or execution failed.
    Fail,
    /// The suite did not run; a skip is never evidence.
    Skip,
}

impl SuiteOutcomeV1 {
    /// Stable JSON name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
            Self::Skip => "skip",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "pass" => Some(Self::Pass),
            "fail" => Some(Self::Fail),
            "skip" => Some(Self::Skip),
            _ => None,
        }
    }
}

/// Exact source and environment identity of the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceSourceV1 {
    /// Full 40-character lowercase hexadecimal Git commit.
    pub git_commit: String,
    /// Whether the worktree had no uncommitted changes.
    pub worktree_clean: bool,
    /// Version of the crate that produced the record.
    pub crate_version: String,
    /// Toolchain identity as reported by the operator (for example `rustc -V`).
    pub toolchain: String,
    /// Target triple or `os-arch` of the build.
    pub target: String,
}

/// Adapter identity as reported by the backend driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterIdentityV1 {
    /// Portable backend family, for example `wgpu`.
    pub backend_family: String,
    /// Graphics API behind the backend (Vulkan, Metal, Dx12, Gl, ...).
    pub api_backend: String,
    /// Adapter name.
    pub name: String,
    /// Reported device type (DiscreteGpu, IntegratedGpu, VirtualGpu, Cpu, Other).
    pub device_type: String,
    /// PCI vendor id, or 0 when unknown.
    pub vendor_id: u32,
    /// PCI device id, or 0 when unknown.
    pub device_id: u32,
    /// Driver name (may be empty when unreported).
    pub driver: String,
    /// Driver information (may be empty when unreported).
    pub driver_info: String,
    /// Harness classification.
    pub class: AdapterClassV1,
}

/// Result of one parity suite against the CPU reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuiteResultV1 {
    /// Stable suite id (`[a-z0-9._-]`, at most 64 bytes).
    pub suite_id: String,
    pub outcome: SuiteOutcomeV1,
    /// Number of compared values or outcomes.
    pub checks: u64,
    /// Checks that missed the declared tolerance.
    pub mismatches: u64,
    /// Declared tolerance identity (for example a numerical policy id).
    pub tolerance: String,
    /// Operator-readable detail; required for failures and skips.
    pub detail: String,
}

/// One adapter evidence record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableAdapterEvidenceV1 {
    pub source: EvidenceSourceV1,
    pub adapter: AdapterIdentityV1,
    pub suites: Vec<SuiteResultV1>,
}

/// Verdict of a structurally valid record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterEvidenceVerdictV1 {
    /// Hardware adapter, clean worktree, every required suite passed with
    /// zero mismatches. Scoped to the exact recorded environment.
    HardwareParityObserved,
    /// Suites passed on a software adapter: code path only, never hardware
    /// evidence.
    CodePathOnlySoftwareAdapter,
    /// Suites passed but the adapter class is unknown.
    UnclassifiedAdapter,
    /// Suites passed on a dirty worktree; the commit does not identify the code.
    DirtyWorktree,
    /// These required suites were missing or skipped.
    Incomplete { suite_ids: Vec<String> },
    /// These suites failed.
    Failed { suite_ids: Vec<String> },
}

/// Structural errors of an adapter evidence record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterEvidenceError {
    /// Git commit was not 40 lowercase hexadecimal characters.
    InvalidGitCommit,
    /// A required text field was empty.
    EmptyField { field: &'static str },
    /// A text field exceeded [`MAX_EVIDENCE_TEXT_BYTES`].
    FieldTooLong { field: &'static str },
    /// A text field had surrounding whitespace or control characters.
    NonCanonicalField { field: &'static str },
    /// The record had no suites.
    NoSuites,
    /// The record had more than [`MAX_EVIDENCE_SUITES`] suites.
    TooManySuites,
    /// A suite id was empty, too long or used characters outside `[a-z0-9._-]`.
    InvalidSuiteId { suite_id: String },
    /// A suite id appeared more than once.
    DuplicateSuite { suite_id: String },
    /// A required suite id was not a valid suite id.
    InvalidRequiredSuite { suite_id: String },
    /// More mismatches than checks.
    MismatchesExceedChecks { suite_id: String },
    /// A passing suite reported mismatches.
    PassWithMismatches { suite_id: String },
    /// A passing suite reported no checks.
    PassWithoutChecks { suite_id: String },
    /// A skipped suite reported checks or mismatches.
    SkipWithChecks { suite_id: String },
    /// A failed or skipped suite had no detail.
    MissingDetail { suite_id: String },
    /// Hardware class claimed for a CPU device type or a software adapter.
    HardwareClassConflict,
}

impl fmt::Display for AdapterEvidenceError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGitCommit => {
                output.write_str("git_commit must be 40 lowercase hexadecimal characters")
            }
            Self::EmptyField { field } => write!(output, "field {field} is empty"),
            Self::FieldTooLong { field } => write!(output, "field {field} is too long"),
            Self::NonCanonicalField { field } => {
                write!(
                    output,
                    "field {field} has surrounding whitespace or control characters"
                )
            }
            Self::NoSuites => output.write_str("record has no suites"),
            Self::TooManySuites => output.write_str("record has too many suites"),
            Self::InvalidSuiteId { suite_id } => write!(output, "invalid suite id {suite_id:?}"),
            Self::DuplicateSuite { suite_id } => write!(output, "duplicate suite {suite_id}"),
            Self::InvalidRequiredSuite { suite_id } => {
                write!(output, "invalid required suite id {suite_id:?}")
            }
            Self::MismatchesExceedChecks { suite_id } => {
                write!(output, "suite {suite_id} has more mismatches than checks")
            }
            Self::PassWithMismatches { suite_id } => {
                write!(output, "suite {suite_id} passed with mismatches")
            }
            Self::PassWithoutChecks { suite_id } => {
                write!(output, "suite {suite_id} passed without checks")
            }
            Self::SkipWithChecks { suite_id } => {
                write!(output, "skipped suite {suite_id} reports checks")
            }
            Self::MissingDetail { suite_id } => {
                write!(output, "failed or skipped suite {suite_id} has no detail")
            }
            Self::HardwareClassConflict => output.write_str(
                "hardware class claimed for a CPU device type or a known software adapter",
            ),
        }
    }
}

impl std::error::Error for AdapterEvidenceError {}

/// Whether an adapter name or driver identifies a known software implementation.
pub fn is_known_software_adapter(name: &str, driver: &str) -> bool {
    let name = name.to_ascii_lowercase();
    let driver = driver.to_ascii_lowercase();
    SOFTWARE_ADAPTER_MARKERS
        .iter()
        .any(|marker| name.contains(marker) || driver.contains(marker))
}

/// Validate a record against the suites that must pass, returning its verdict.
pub fn validate_adapter_evidence(
    record: &PortableAdapterEvidenceV1,
    required_suites: &[&str],
) -> Result<AdapterEvidenceVerdictV1, AdapterEvidenceError> {
    let source = &record.source;
    if source.git_commit.len() != 40
        || !source
            .git_commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(AdapterEvidenceError::InvalidGitCommit);
    }
    required_text("crate_version", &source.crate_version)?;
    required_text("toolchain", &source.toolchain)?;
    required_text("target", &source.target)?;
    let adapter = &record.adapter;
    required_text("backend_family", &adapter.backend_family)?;
    required_text("api_backend", &adapter.api_backend)?;
    required_text("adapter_name", &adapter.name)?;
    required_text("device_type", &adapter.device_type)?;
    optional_text("driver", &adapter.driver)?;
    optional_text("driver_info", &adapter.driver_info)?;
    if adapter.class == AdapterClassV1::Hardware
        && (adapter.device_type.eq_ignore_ascii_case("cpu")
            || is_known_software_adapter(&adapter.name, &adapter.driver))
    {
        return Err(AdapterEvidenceError::HardwareClassConflict);
    }
    if record.suites.is_empty() {
        return Err(AdapterEvidenceError::NoSuites);
    }
    if record.suites.len() > MAX_EVIDENCE_SUITES {
        return Err(AdapterEvidenceError::TooManySuites);
    }
    for (index, suite) in record.suites.iter().enumerate() {
        let id = || suite.suite_id.clone();
        if !valid_suite_id(&suite.suite_id) {
            return Err(AdapterEvidenceError::InvalidSuiteId { suite_id: id() });
        }
        if record.suites[..index]
            .iter()
            .any(|other| other.suite_id == suite.suite_id)
        {
            return Err(AdapterEvidenceError::DuplicateSuite { suite_id: id() });
        }
        required_text("tolerance", &suite.tolerance)?;
        optional_text("detail", &suite.detail)?;
        if suite.mismatches > suite.checks {
            return Err(AdapterEvidenceError::MismatchesExceedChecks { suite_id: id() });
        }
        match suite.outcome {
            SuiteOutcomeV1::Pass if suite.mismatches != 0 => {
                return Err(AdapterEvidenceError::PassWithMismatches { suite_id: id() })
            }
            SuiteOutcomeV1::Pass if suite.checks == 0 => {
                return Err(AdapterEvidenceError::PassWithoutChecks { suite_id: id() })
            }
            SuiteOutcomeV1::Skip if suite.checks != 0 || suite.mismatches != 0 => {
                return Err(AdapterEvidenceError::SkipWithChecks { suite_id: id() })
            }
            SuiteOutcomeV1::Fail | SuiteOutcomeV1::Skip if suite.detail.is_empty() => {
                return Err(AdapterEvidenceError::MissingDetail { suite_id: id() })
            }
            _ => {}
        }
    }
    for required in required_suites {
        if !valid_suite_id(required) {
            return Err(AdapterEvidenceError::InvalidRequiredSuite {
                suite_id: (*required).to_owned(),
            });
        }
    }

    let failed: Vec<String> = record
        .suites
        .iter()
        .filter(|suite| suite.outcome == SuiteOutcomeV1::Fail)
        .map(|suite| suite.suite_id.clone())
        .collect();
    if !failed.is_empty() {
        return Ok(AdapterEvidenceVerdictV1::Failed { suite_ids: failed });
    }
    let incomplete: Vec<String> = required_suites
        .iter()
        .filter(|required| {
            !record
                .suites
                .iter()
                .any(|suite| suite.suite_id == **required && suite.outcome == SuiteOutcomeV1::Pass)
        })
        .map(|required| (*required).to_owned())
        .collect();
    if !incomplete.is_empty() {
        return Ok(AdapterEvidenceVerdictV1::Incomplete {
            suite_ids: incomplete,
        });
    }
    if !source.worktree_clean {
        return Ok(AdapterEvidenceVerdictV1::DirtyWorktree);
    }
    Ok(match adapter.class {
        AdapterClassV1::Hardware => AdapterEvidenceVerdictV1::HardwareParityObserved,
        AdapterClassV1::Software => AdapterEvidenceVerdictV1::CodePathOnlySoftwareAdapter,
        AdapterClassV1::Unknown => AdapterEvidenceVerdictV1::UnclassifiedAdapter,
    })
}

fn valid_suite_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|byte| matches!(byte, b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-'))
}

fn optional_text(field: &'static str, value: &str) -> Result<(), AdapterEvidenceError> {
    if value.len() > MAX_EVIDENCE_TEXT_BYTES {
        return Err(AdapterEvidenceError::FieldTooLong { field });
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(AdapterEvidenceError::NonCanonicalField { field });
    }
    Ok(())
}

fn required_text(field: &'static str, value: &str) -> Result<(), AdapterEvidenceError> {
    if value.is_empty() {
        return Err(AdapterEvidenceError::EmptyField { field });
    }
    optional_text(field, value)
}

/// Encode a record as compact JSON. Encoding does not validate.
pub fn adapter_evidence_to_json(record: &PortableAdapterEvidenceV1) -> String {
    let source = &record.source;
    let adapter = &record.adapter;
    let value = Json::Object(vec![
        member("kind", text(PORTABLE_ADAPTER_EVIDENCE_KIND)),
        member(
            "schema_version",
            json::unsigned(u64::from(PORTABLE_ADAPTER_EVIDENCE_SCHEMA_VERSION)),
        ),
        member(
            "source",
            Json::Object(vec![
                member("git_commit", text(&source.git_commit)),
                member("worktree_clean", Json::Bool(source.worktree_clean)),
                member("crate_version", text(&source.crate_version)),
                member("toolchain", text(&source.toolchain)),
                member("target", text(&source.target)),
            ]),
        ),
        member(
            "adapter",
            Json::Object(vec![
                member("backend_family", text(&adapter.backend_family)),
                member("api_backend", text(&adapter.api_backend)),
                member("name", text(&adapter.name)),
                member("device_type", text(&adapter.device_type)),
                member("vendor_id", json::unsigned(u64::from(adapter.vendor_id))),
                member("device_id", json::unsigned(u64::from(adapter.device_id))),
                member("driver", text(&adapter.driver)),
                member("driver_info", text(&adapter.driver_info)),
                member("class", text(adapter.class.name())),
            ]),
        ),
        member(
            "suites",
            Json::Array(
                record
                    .suites
                    .iter()
                    .map(|suite| {
                        Json::Object(vec![
                            member("suite_id", text(&suite.suite_id)),
                            member("outcome", text(suite.outcome.name())),
                            member("checks", json::unsigned(suite.checks)),
                            member("mismatches", json::unsigned(suite.mismatches)),
                            member("tolerance", text(&suite.tolerance)),
                            member("detail", text(&suite.detail)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]);
    let mut output = String::new();
    json::write(&value, &mut output);
    output
}

/// Parse a record strictly. Parsing does not validate; call
/// [`validate_adapter_evidence`] on the result.
pub fn adapter_evidence_from_json(
    input: &str,
) -> Result<PortableAdapterEvidenceV1, AdapterEvidenceJsonError> {
    let value = json::parse(input).map_err(|error| AdapterEvidenceJsonError::Syntax {
        offset: error.offset,
        reason: error.reason,
    })?;
    let mut root = Object::new("root", value)?;
    if root.string("kind")? != PORTABLE_ADAPTER_EVIDENCE_KIND {
        return Err(AdapterEvidenceJsonError::WrongKind);
    }
    if root.u64("schema_version").ok() != Some(u64::from(PORTABLE_ADAPTER_EVIDENCE_SCHEMA_VERSION))
    {
        return Err(AdapterEvidenceJsonError::UnsupportedSchemaVersion);
    }
    let mut source_object = root.object("source")?;
    let source = EvidenceSourceV1 {
        git_commit: source_object.string("git_commit")?,
        worktree_clean: source_object.bool("worktree_clean")?,
        crate_version: source_object.string("crate_version")?,
        toolchain: source_object.string("toolchain")?,
        target: source_object.string("target")?,
    };
    source_object.finish()?;
    let mut adapter_object = root.object("adapter")?;
    let adapter = AdapterIdentityV1 {
        backend_family: adapter_object.string("backend_family")?,
        api_backend: adapter_object.string("api_backend")?,
        name: adapter_object.string("name")?,
        device_type: adapter_object.string("device_type")?,
        vendor_id: adapter_object.u32("vendor_id")?,
        device_id: adapter_object.u32("device_id")?,
        driver: adapter_object.string("driver")?,
        driver_info: adapter_object.string("driver_info")?,
        class: AdapterClassV1::parse(&adapter_object.string("class")?)
            .ok_or(AdapterEvidenceJsonError::UnknownVariant { field: "class" })?,
    };
    adapter_object.finish()?;
    let mut suites = Vec::new();
    for item in root.array("suites")? {
        let mut suite = Object::new("suites", item)?;
        suites.push(SuiteResultV1 {
            suite_id: suite.string("suite_id")?,
            outcome: SuiteOutcomeV1::parse(&suite.string("outcome")?)
                .ok_or(AdapterEvidenceJsonError::UnknownVariant { field: "outcome" })?,
            checks: suite.u64("checks")?,
            mismatches: suite.u64("mismatches")?,
            tolerance: suite.string("tolerance")?,
            detail: suite.string("detail")?,
        });
        suite.finish()?;
    }
    root.finish()?;
    Ok(PortableAdapterEvidenceV1 {
        source,
        adapter,
        suites,
    })
}

/// Fail-closed JSON errors of adapter evidence records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterEvidenceJsonError {
    /// Input was not valid JSON.
    Syntax { offset: usize, reason: &'static str },
    /// `kind` tag did not match.
    WrongKind,
    /// `schema_version` was not supported.
    UnsupportedSchemaVersion,
    /// A required field was absent.
    MissingField { field: &'static str },
    /// An object contained a field not defined by the schema.
    UnknownField { field: String },
    /// A field had the wrong JSON type.
    WrongType { field: &'static str },
    /// An integer field was fractional, negative or out of range.
    InvalidInteger { field: &'static str },
    /// An enumerated string had an unknown value.
    UnknownVariant { field: &'static str },
}

impl fmt::Display for AdapterEvidenceJsonError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax { offset, reason } => {
                write!(output, "JSON syntax error at byte {offset}: {reason}")
            }
            Self::WrongKind => output.write_str("record kind tag does not match"),
            Self::UnsupportedSchemaVersion => write!(
                output,
                "unsupported schema_version, expected {PORTABLE_ADAPTER_EVIDENCE_SCHEMA_VERSION}"
            ),
            Self::MissingField { field } => write!(output, "missing field {field}"),
            Self::UnknownField { field } => write!(output, "unknown field {field}"),
            Self::WrongType { field } => write!(output, "field {field} has the wrong type"),
            Self::InvalidInteger { field } => {
                write!(output, "field {field} is not a valid integer")
            }
            Self::UnknownVariant { field } => write!(output, "field {field} has an unknown value"),
        }
    }
}

impl std::error::Error for AdapterEvidenceJsonError {}

struct Object {
    members: Vec<(String, Json)>,
}

impl Object {
    fn new(field: &'static str, value: Json) -> Result<Self, AdapterEvidenceJsonError> {
        match value {
            Json::Object(members) => Ok(Self { members }),
            _ => Err(AdapterEvidenceJsonError::WrongType { field }),
        }
    }

    fn take(&mut self, field: &'static str) -> Result<Json, AdapterEvidenceJsonError> {
        let index = self
            .members
            .iter()
            .position(|(key, _)| key == field)
            .ok_or(AdapterEvidenceJsonError::MissingField { field })?;
        Ok(self.members.remove(index).1)
    }

    fn string(&mut self, field: &'static str) -> Result<String, AdapterEvidenceJsonError> {
        match self.take(field)? {
            Json::String(text) => Ok(text),
            _ => Err(AdapterEvidenceJsonError::WrongType { field }),
        }
    }

    fn bool(&mut self, field: &'static str) -> Result<bool, AdapterEvidenceJsonError> {
        match self.take(field)? {
            Json::Bool(value) => Ok(value),
            _ => Err(AdapterEvidenceJsonError::WrongType { field }),
        }
    }

    fn u64(&mut self, field: &'static str) -> Result<u64, AdapterEvidenceJsonError> {
        match self.take(field)? {
            Json::Number(text) => {
                if !text.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err(AdapterEvidenceJsonError::InvalidInteger { field });
                }
                text.parse::<u64>()
                    .map_err(|_| AdapterEvidenceJsonError::InvalidInteger { field })
            }
            _ => Err(AdapterEvidenceJsonError::WrongType { field }),
        }
    }

    fn u32(&mut self, field: &'static str) -> Result<u32, AdapterEvidenceJsonError> {
        u32::try_from(self.u64(field)?)
            .map_err(|_| AdapterEvidenceJsonError::InvalidInteger { field })
    }

    fn object(&mut self, field: &'static str) -> Result<Self, AdapterEvidenceJsonError> {
        Self::new(field, self.take(field)?)
    }

    fn array(&mut self, field: &'static str) -> Result<Vec<Json>, AdapterEvidenceJsonError> {
        match self.take(field)? {
            Json::Array(items) => Ok(items),
            _ => Err(AdapterEvidenceJsonError::WrongType { field }),
        }
    }

    fn finish(self) -> Result<(), AdapterEvidenceJsonError> {
        match self.members.into_iter().next() {
            None => Ok(()),
            Some((field, _)) => Err(AdapterEvidenceJsonError::UnknownField { field }),
        }
    }
}

fn member(name: &str, value: Json) -> (String, Json) {
    (name.to_owned(), value)
}

fn text(value: &str) -> Json {
    Json::String(value.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUIRED: [&str; 2] = ["wgpu.add_f32", "wgpu.f32_kernels"];

    fn suite(id: &str, outcome: SuiteOutcomeV1, checks: u64, mismatches: u64) -> SuiteResultV1 {
        SuiteResultV1 {
            suite_id: id.to_owned(),
            outcome,
            checks,
            mismatches,
            tolerance: "bit-exact".to_owned(),
            detail: if outcome == SuiteOutcomeV1::Pass {
                String::new()
            } else {
                "see log".to_owned()
            },
        }
    }

    fn record(class: AdapterClassV1) -> PortableAdapterEvidenceV1 {
        PortableAdapterEvidenceV1 {
            source: EvidenceSourceV1 {
                git_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
                worktree_clean: true,
                crate_version: "0.1.0".to_owned(),
                toolchain: "rustc 1.77.0 (aedd173a2 2024-03-17)".to_owned(),
                target: "x86_64-unknown-linux-gnu".to_owned(),
            },
            adapter: AdapterIdentityV1 {
                backend_family: "wgpu".to_owned(),
                api_backend: "Vulkan".to_owned(),
                name: "Example GPU".to_owned(),
                device_type: "DiscreteGpu".to_owned(),
                vendor_id: 0x1002,
                device_id: 0x73bf,
                driver: "radv".to_owned(),
                driver_info: "Mesa 24.0".to_owned(),
                class,
            },
            suites: vec![
                suite("wgpu.add_f32", SuiteOutcomeV1::Pass, 1024, 0),
                suite("wgpu.f32_kernels", SuiteOutcomeV1::Pass, 4000, 0),
            ],
        }
    }

    #[test]
    fn verdicts_follow_class_and_outcomes() {
        use AdapterEvidenceVerdictV1 as V;
        assert_eq!(
            validate_adapter_evidence(&record(AdapterClassV1::Hardware), &REQUIRED),
            Ok(V::HardwareParityObserved)
        );
        let mut software = record(AdapterClassV1::Software);
        software.adapter.name = "llvmpipe (LLVM 19.1.7, 256 bits)".to_owned();
        software.adapter.device_type = "Cpu".to_owned();
        assert_eq!(
            validate_adapter_evidence(&software, &REQUIRED),
            Ok(V::CodePathOnlySoftwareAdapter)
        );
        assert_eq!(
            validate_adapter_evidence(&record(AdapterClassV1::Unknown), &REQUIRED),
            Ok(V::UnclassifiedAdapter)
        );
        let mut dirty = record(AdapterClassV1::Hardware);
        dirty.source.worktree_clean = false;
        assert_eq!(
            validate_adapter_evidence(&dirty, &REQUIRED),
            Ok(V::DirtyWorktree)
        );
        let mut skipped = record(AdapterClassV1::Hardware);
        skipped.suites[1] = suite("wgpu.f32_kernels", SuiteOutcomeV1::Skip, 0, 0);
        assert_eq!(
            validate_adapter_evidence(&skipped, &REQUIRED),
            Ok(V::Incomplete {
                suite_ids: vec!["wgpu.f32_kernels".to_owned()]
            })
        );
        let missing = record(AdapterClassV1::Hardware);
        assert_eq!(
            validate_adapter_evidence(&missing, &["wgpu.add_f32", "wgpu.f32_graph"]),
            Ok(V::Incomplete {
                suite_ids: vec!["wgpu.f32_graph".to_owned()]
            })
        );
        let mut failed = record(AdapterClassV1::Hardware);
        failed.suites[0] = suite("wgpu.add_f32", SuiteOutcomeV1::Fail, 1024, 3);
        failed.source.worktree_clean = false;
        assert_eq!(
            validate_adapter_evidence(&failed, &REQUIRED),
            Ok(V::Failed {
                suite_ids: vec!["wgpu.add_f32".to_owned()]
            })
        );
    }

    #[test]
    fn malformed_records_are_rejected() {
        use AdapterEvidenceError as E;
        let check = |edit: &dyn Fn(&mut PortableAdapterEvidenceV1), expected: E| {
            let mut value = record(AdapterClassV1::Hardware);
            edit(&mut value);
            assert_eq!(validate_adapter_evidence(&value, &REQUIRED), Err(expected));
        };
        check(
            &|r| r.source.git_commit = "0123".into(),
            E::InvalidGitCommit,
        );
        check(
            &|r| r.source.git_commit = "0123456789ABCDEF0123456789abcdef01234567".into(),
            E::InvalidGitCommit,
        );
        check(
            &|r| r.source.toolchain.clear(),
            E::EmptyField { field: "toolchain" },
        );
        check(
            &|r| r.adapter.name = " GPU".into(),
            E::NonCanonicalField {
                field: "adapter_name",
            },
        );
        check(
            &|r| r.adapter.driver_info = "a\nb".into(),
            E::NonCanonicalField {
                field: "driver_info",
            },
        );
        check(
            &|r| r.adapter.driver = "x".repeat(MAX_EVIDENCE_TEXT_BYTES + 1),
            E::FieldTooLong { field: "driver" },
        );
        check(
            &|r| r.adapter.device_type = "Cpu".into(),
            E::HardwareClassConflict,
        );
        check(
            &|r| r.adapter.driver = "llvmpipe".into(),
            E::HardwareClassConflict,
        );
        check(
            &|r| r.adapter.name = "SwiftShader Device".into(),
            E::HardwareClassConflict,
        );
        check(&|r| r.suites.clear(), E::NoSuites);
        check(
            &|r| r.suites = vec![suite("x", SuiteOutcomeV1::Pass, 1, 0); MAX_EVIDENCE_SUITES + 1],
            E::TooManySuites,
        );
        let owned = |id: &str| id.to_owned();
        check(
            &|r| r.suites[0].suite_id = "WGPU.add".into(),
            E::InvalidSuiteId {
                suite_id: owned("WGPU.add"),
            },
        );
        check(
            &|r| r.suites[1].suite_id = "wgpu.add_f32".into(),
            E::DuplicateSuite {
                suite_id: owned("wgpu.add_f32"),
            },
        );
        check(
            &|r| r.suites[0].mismatches = 1,
            E::PassWithMismatches {
                suite_id: owned("wgpu.add_f32"),
            },
        );
        check(
            &|r| r.suites[0] = suite("wgpu.add_f32", SuiteOutcomeV1::Pass, 0, 0),
            E::PassWithoutChecks {
                suite_id: owned("wgpu.add_f32"),
            },
        );
        check(
            &|r| r.suites[0] = suite("wgpu.add_f32", SuiteOutcomeV1::Skip, 3, 0),
            E::SkipWithChecks {
                suite_id: owned("wgpu.add_f32"),
            },
        );
        check(
            &|r| r.suites[0] = suite("wgpu.add_f32", SuiteOutcomeV1::Fail, 3, 4),
            E::MismatchesExceedChecks {
                suite_id: owned("wgpu.add_f32"),
            },
        );
        check(
            &|r| {
                r.suites[0] = suite("wgpu.add_f32", SuiteOutcomeV1::Fail, 3, 1);
                r.suites[0].detail.clear();
            },
            E::MissingDetail {
                suite_id: owned("wgpu.add_f32"),
            },
        );
        check(
            &|r| r.suites[0].tolerance.clear(),
            E::EmptyField { field: "tolerance" },
        );
        let value = record(AdapterClassV1::Hardware);
        assert_eq!(
            validate_adapter_evidence(&value, &["Bad Id"]),
            Err(E::InvalidRequiredSuite {
                suite_id: owned("Bad Id")
            })
        );
    }

    #[test]
    fn json_round_trips_and_rejects_drift() {
        let value = record(AdapterClassV1::Hardware);
        let text = adapter_evidence_to_json(&value);
        assert_eq!(adapter_evidence_from_json(&text), Ok(value.clone()));

        use AdapterEvidenceJsonError as E;
        let replace = |from: &str, to: &str| {
            assert_eq!(text.matches(from).count(), 1, "{from}");
            adapter_evidence_from_json(&text.replacen(from, to, 1))
        };
        assert_eq!(
            replace(PORTABLE_ADAPTER_EVIDENCE_KIND, "nnis.other"),
            Err(E::WrongKind)
        );
        assert_eq!(
            replace("\"schema_version\":1", "\"schema_version\":2"),
            Err(E::UnsupportedSchemaVersion)
        );
        assert_eq!(
            replace("\"class\":\"hardware\"", "\"class\":\"gpu\""),
            Err(E::UnknownVariant { field: "class" })
        );
        assert_eq!(
            replace("\"worktree_clean\":true", "\"worktree_clean\":1"),
            Err(E::WrongType {
                field: "worktree_clean"
            })
        );
        assert_eq!(
            replace("\"vendor_id\":4098", "\"vendor_id\":4294967296"),
            Err(E::InvalidInteger { field: "vendor_id" })
        );
        assert_eq!(
            replace("\"checks\":1024", "\"checks\":1.5"),
            Err(E::InvalidInteger { field: "checks" })
        );
        assert_eq!(
            replace("\"target\":", "\"latency_ms\":1,\"target\":"),
            Err(E::UnknownField {
                field: "latency_ms".to_owned()
            })
        );
        assert_eq!(
            replace(",\"toolchain\":\"rustc 1.77.0 (aedd173a2 2024-03-17)\"", ""),
            Err(E::MissingField { field: "toolchain" })
        );
        assert!(matches!(
            adapter_evidence_from_json("{"),
            Err(E::Syntax { .. })
        ));
    }
}
