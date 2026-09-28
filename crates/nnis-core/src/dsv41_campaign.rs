//! DSV41-5 campaign preregistration and fail-closed evidence validation.
//!
//! This module defines what a real-model DSV41 campaign must declare before it
//! runs and what its evidence must contain afterwards. It records **no
//! results**. A validated evidence record is only complete and well-formed; it
//! is not a promotion, a comparison, or a speed-up, and this module computes no
//! ratios between arms.
//!
//! Evidence is rejected unless, for every preregistered arm, context length,
//! and decode length, it reports quality, exact memory (one declared memory
//! metric shared by all arms), a decode-latency distribution, and tokens/s
//! together, with the declared repetitions, a clean exact Git head with green
//! CI, and a hardware identity for the declared backend. Records marked as
//! synthetic fixtures are never evidence.

use core::fmt;

/// Version of the DSV41-5 preregistration/evidence contract.
pub const NNIS_DSV41_CAMPAIGN_CONTRACT_VERSION: u32 = 1;

/// Maximum UTF-8 bytes accepted for campaign identities and labels.
pub const MAX_DSV41_CAMPAIGN_ID_BYTES: usize = 128;

/// DSV41 runtime primitive exercised by a candidate arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dsv41PrimitiveV1 {
    /// DSV41-1 bounded recent-window replay.
    BoundedReplay,
    /// DSV41-2 model-supplied cross-layer KV reuse.
    CrossLayerKvReuse,
    /// DSV41-3 FP4 E2M1 group-scaled KV.
    Fp4E2M1Kv,
    /// DSV41-4 confidence-scheduled speculative verification.
    SpeculativeVerification,
}

/// Execution backend of a campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CampaignBackendV1 {
    /// Portable CPU reference.
    Cpu,
    /// Portable WGPU backend.
    Wgpu,
    /// Existing CUDA path, secondary cross-check only for this programme.
    LegacyCudaCrossCheck,
}

/// Exact memory metric used for every arm of one campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CampaignMemoryMetricV1 {
    /// Peak process resident set size in bytes.
    PeakProcessRssBytes,
    /// Peak device-allocated bytes reported by the backend.
    PeakDeviceAllocatedBytes,
    /// Peak NNIS-owned allocation bytes from explicit accounting.
    PeakNnisOwnedAllocationBytes,
}

/// Direction of the preregistered quality metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualityDirectionV1 {
    /// Larger values are better.
    HigherIsBetter,
    /// Smaller values are better.
    LowerIsBetter,
}

/// Exact model identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CampaignModelV1 {
    /// Model identifier.
    pub model_id: String,
    /// Exact 40-character lowercase hexadecimal source revision.
    pub revision: String,
    /// Lowercase hexadecimal SHA-256 of the weight bytes.
    pub weights_sha256: String,
}

/// Exact prompt-suite identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CampaignPromptSuiteV1 {
    /// Prompt-suite identifier.
    pub suite_id: String,
    /// Lowercase hexadecimal SHA-256 of the suite bytes.
    pub suite_sha256: String,
    /// Number of prompts in the suite.
    pub prompt_count: u32,
}

/// One preregistered campaign arm.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CampaignArmV1 {
    /// Unique arm identifier.
    pub arm_id: String,
    /// Primitives enabled; empty only for the dense baseline.
    pub primitives: Vec<Dsv41PrimitiveV1>,
}

/// Preregistered quality metric and caller-declared budget.
#[derive(Debug, Clone, PartialEq)]
pub struct CampaignQualityMetricV1 {
    /// Metric identifier.
    pub metric_id: String,
    /// Metric direction.
    pub direction: QualityDirectionV1,
    /// Caller-declared maximum tolerated regression (finite, non-negative).
    pub max_regression: f64,
}

/// Validated DSV41-5 campaign preregistration.
#[derive(Debug, Clone, PartialEq)]
pub struct Dsv41CampaignPreregistrationV1 {
    campaign_id: String,
    model: CampaignModelV1,
    prompt_suite: CampaignPromptSuiteV1,
    backend: CampaignBackendV1,
    dense_baseline_arm: String,
    arms: Vec<CampaignArmV1>,
    context_lengths: Vec<u32>,
    decode_lengths: Vec<u32>,
    repetitions: u32,
    memory_metric: CampaignMemoryMetricV1,
    quality_metric: CampaignQualityMetricV1,
}

/// Unvalidated preregistration fields.
#[derive(Debug, Clone, PartialEq)]
pub struct Dsv41CampaignPreregistrationFieldsV1 {
    /// Campaign identifier.
    pub campaign_id: String,
    /// Exact model identity.
    pub model: CampaignModelV1,
    /// Exact prompt suite.
    pub prompt_suite: CampaignPromptSuiteV1,
    /// Execution backend.
    pub backend: CampaignBackendV1,
    /// Arms; exactly one must be the dense baseline with no primitives.
    pub arms: Vec<CampaignArmV1>,
    /// Strictly increasing context lengths.
    pub context_lengths: Vec<u32>,
    /// Strictly increasing decode lengths.
    pub decode_lengths: Vec<u32>,
    /// Repetitions per cell (at least 2 so latency is a distribution).
    pub repetitions: u32,
    /// Memory metric shared by all arms.
    pub memory_metric: CampaignMemoryMetricV1,
    /// Quality metric and budget.
    pub quality_metric: CampaignQualityMetricV1,
}

impl Dsv41CampaignPreregistrationV1 {
    /// Validate preregistration fields, fail closed.
    pub fn new(fields: Dsv41CampaignPreregistrationFieldsV1) -> Result<Self, Dsv41CampaignError> {
        validate_id("campaign_id", &fields.campaign_id)?;
        validate_id("model_id", &fields.model.model_id)?;
        validate_hex("revision", &fields.model.revision, 40)?;
        validate_hex("weights_sha256", &fields.model.weights_sha256, 64)?;
        validate_id("suite_id", &fields.prompt_suite.suite_id)?;
        validate_hex("suite_sha256", &fields.prompt_suite.suite_sha256, 64)?;
        if fields.prompt_suite.prompt_count == 0 {
            return Err(Dsv41CampaignError::EmptyPromptSuite);
        }
        validate_lengths("context_lengths", &fields.context_lengths)?;
        validate_lengths("decode_lengths", &fields.decode_lengths)?;
        if fields.repetitions < 2 {
            return Err(Dsv41CampaignError::TooFewRepetitions {
                repetitions: fields.repetitions,
            });
        }
        validate_id("quality_metric_id", &fields.quality_metric.metric_id)?;
        let budget = fields.quality_metric.max_regression;
        if !budget.is_finite() || budget < 0.0 {
            return Err(Dsv41CampaignError::InvalidQualityBudget);
        }
        let mut dense_baseline_arm = None;
        let mut candidates = 0usize;
        for (index, arm) in fields.arms.iter().enumerate() {
            validate_id("arm_id", &arm.arm_id)?;
            if fields.arms[..index]
                .iter()
                .any(|other| other.arm_id == arm.arm_id)
            {
                return Err(Dsv41CampaignError::DuplicateArm {
                    arm_id: arm.arm_id.clone(),
                });
            }
            if arm.primitives.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(Dsv41CampaignError::UnsortedOrDuplicatePrimitives {
                    arm_id: arm.arm_id.clone(),
                });
            }
            if arm.primitives.is_empty() {
                if dense_baseline_arm.is_some() {
                    return Err(Dsv41CampaignError::MultipleDenseBaselines);
                }
                dense_baseline_arm = Some(arm.arm_id.clone());
            } else {
                candidates += 1;
            }
        }
        let dense_baseline_arm =
            dense_baseline_arm.ok_or(Dsv41CampaignError::MissingDenseBaseline)?;
        if candidates == 0 {
            return Err(Dsv41CampaignError::MissingCandidateArm);
        }
        Ok(Self {
            campaign_id: fields.campaign_id,
            model: fields.model,
            prompt_suite: fields.prompt_suite,
            backend: fields.backend,
            dense_baseline_arm,
            arms: fields.arms,
            context_lengths: fields.context_lengths,
            decode_lengths: fields.decode_lengths,
            repetitions: fields.repetitions,
            memory_metric: fields.memory_metric,
            quality_metric: fields.quality_metric,
        })
    }

    /// Campaign identifier.
    pub fn campaign_id(&self) -> &str {
        &self.campaign_id
    }

    /// Exact model identity.
    pub const fn model(&self) -> &CampaignModelV1 {
        &self.model
    }

    /// Exact prompt suite.
    pub const fn prompt_suite(&self) -> &CampaignPromptSuiteV1 {
        &self.prompt_suite
    }

    /// Declared backend.
    pub const fn backend(&self) -> CampaignBackendV1 {
        self.backend
    }

    /// Dense baseline arm identifier.
    pub fn dense_baseline_arm(&self) -> &str {
        &self.dense_baseline_arm
    }

    /// All arms in declaration order.
    pub fn arms(&self) -> &[CampaignArmV1] {
        &self.arms
    }

    /// Context lengths.
    pub fn context_lengths(&self) -> &[u32] {
        &self.context_lengths
    }

    /// Decode lengths.
    pub fn decode_lengths(&self) -> &[u32] {
        &self.decode_lengths
    }

    /// Repetitions per cell.
    pub const fn repetitions(&self) -> u32 {
        self.repetitions
    }

    /// Declared memory metric.
    pub const fn memory_metric(&self) -> CampaignMemoryMetricV1 {
        self.memory_metric
    }

    /// Declared quality metric.
    pub const fn quality_metric(&self) -> &CampaignQualityMetricV1 {
        &self.quality_metric
    }

    /// Number of required (arm, context length, decode length) cells.
    pub fn required_cells(&self) -> usize {
        self.arms.len() * self.context_lengths.len() * self.decode_lengths.len()
    }
}

/// Whether a record is claimed to come from a real execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CampaignRecordKindV1 {
    /// Claimed output of an actual campaign execution.
    PhysicalExecution,
    /// Synthetic fixture for software tests; never evidence.
    SyntheticFixture,
}

/// Exact source provenance of a campaign record.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CampaignProvenanceV1 {
    /// Exact 40-character lowercase hexadecimal Git head executed.
    pub git_head: String,
    /// Tracked worktree was clean at execution.
    pub worktree_clean: bool,
    /// CI run identifier for the exact head.
    pub ci_run_id: u64,
    /// Required CI was green on the exact head.
    pub ci_green_on_exact_head: bool,
}

/// Hardware and software identity of the execution environment.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CampaignHardwareV1 {
    /// Backend actually executed.
    pub backend: CampaignBackendV1,
    /// Device or CPU model name.
    pub device_name: String,
    /// Driver or runtime version (for CPU, the host runtime identity).
    pub driver_version: String,
    /// Host operating-system identity.
    pub host_os: String,
}

/// Decode-latency distribution in milliseconds per generated token.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecodeLatencyDistributionV1 {
    /// Latency samples summarized.
    pub samples: u64,
    /// Median.
    pub p50_ms: f64,
    /// 90th percentile.
    pub p90_ms: f64,
    /// 99th percentile.
    pub p99_ms: f64,
    /// Maximum.
    pub max_ms: f64,
}

/// One measured cell. Every metric is optional so absent data is rejected
/// explicitly rather than defaulted.
#[derive(Debug, Clone, PartialEq)]
pub struct CampaignCellV1 {
    /// Arm identifier.
    pub arm_id: String,
    /// Context length.
    pub context_length: u32,
    /// Decode length.
    pub decode_length: u32,
    /// Repetitions actually executed.
    pub repetitions: u32,
    /// Quality metric value.
    pub quality: Option<f64>,
    /// Memory metric actually used.
    pub memory_metric: Option<CampaignMemoryMetricV1>,
    /// Exact memory value in bytes.
    pub memory_bytes: Option<u64>,
    /// Decode-latency distribution.
    pub decode_latency: Option<DecodeLatencyDistributionV1>,
    /// Decode throughput in generated tokens per second.
    pub tokens_per_second: Option<f64>,
}

/// Unvalidated campaign evidence record.
#[derive(Debug, Clone, PartialEq)]
pub struct Dsv41CampaignEvidenceV1 {
    /// Record kind.
    pub kind: CampaignRecordKindV1,
    /// Campaign identifier the record claims to satisfy.
    pub campaign_id: String,
    /// Source provenance.
    pub provenance: CampaignProvenanceV1,
    /// Execution environment.
    pub hardware: CampaignHardwareV1,
    /// Measured cells.
    pub cells: Vec<CampaignCellV1>,
}

/// Which metric a cell failed to report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CampaignMetricV1 {
    /// Quality value.
    Quality,
    /// Memory metric kind.
    MemoryMetric,
    /// Memory bytes.
    MemoryBytes,
    /// Decode-latency distribution.
    DecodeLatency,
    /// Tokens per second.
    TokensPerSecond,
}

/// Check everything except the record kind.
///
/// Use [`validate_dsv41_evidence`] for evidence; this entry point exists so
/// software tests can exercise structural rules with synthetic fixtures.
pub fn validate_dsv41_evidence_structure(
    preregistration: &Dsv41CampaignPreregistrationV1,
    evidence: &Dsv41CampaignEvidenceV1,
) -> Result<(), Dsv41CampaignError> {
    if evidence.campaign_id != preregistration.campaign_id {
        return Err(Dsv41CampaignError::CampaignMismatch);
    }
    validate_hex("git_head", &evidence.provenance.git_head, 40)?;
    if !evidence.provenance.worktree_clean {
        return Err(Dsv41CampaignError::DirtyWorktree);
    }
    if evidence.provenance.ci_run_id == 0 || !evidence.provenance.ci_green_on_exact_head {
        return Err(Dsv41CampaignError::ExactHeadCiNotGreen);
    }
    if evidence.hardware.backend != preregistration.backend {
        return Err(Dsv41CampaignError::BackendMismatch);
    }
    validate_id("device_name", &evidence.hardware.device_name)?;
    validate_id("driver_version", &evidence.hardware.driver_version)?;
    validate_id("host_os", &evidence.hardware.host_os)?;
    if evidence.cells.len() != preregistration.required_cells() {
        return Err(Dsv41CampaignError::CellCountMismatch {
            expected: preregistration.required_cells(),
            actual: evidence.cells.len(),
        });
    }
    for arm in &preregistration.arms {
        for &context_length in &preregistration.context_lengths {
            for &decode_length in &preregistration.decode_lengths {
                let mut matches = evidence.cells.iter().filter(|cell| {
                    cell.arm_id == arm.arm_id
                        && cell.context_length == context_length
                        && cell.decode_length == decode_length
                });
                let cell = matches
                    .next()
                    .ok_or_else(|| Dsv41CampaignError::MissingCell {
                        arm_id: arm.arm_id.clone(),
                        context_length,
                        decode_length,
                    })?;
                if matches.next().is_some() {
                    return Err(Dsv41CampaignError::DuplicateCell {
                        arm_id: arm.arm_id.clone(),
                        context_length,
                        decode_length,
                    });
                }
                validate_cell(preregistration, cell)?;
            }
        }
    }
    Ok(())
}

/// Validate a campaign evidence record, fail closed.
///
/// Synthetic fixtures are rejected even when structurally complete. Success
/// means only that the record is complete and well-formed for the
/// preregistration; it authorizes no promotion or comparison claim.
pub fn validate_dsv41_evidence(
    preregistration: &Dsv41CampaignPreregistrationV1,
    evidence: &Dsv41CampaignEvidenceV1,
) -> Result<(), Dsv41CampaignError> {
    validate_dsv41_evidence_structure(preregistration, evidence)?;
    if evidence.kind != CampaignRecordKindV1::PhysicalExecution {
        return Err(Dsv41CampaignError::SyntheticFixtureIsNotEvidence);
    }
    Ok(())
}

fn validate_cell(
    preregistration: &Dsv41CampaignPreregistrationV1,
    cell: &CampaignCellV1,
) -> Result<(), Dsv41CampaignError> {
    let missing = |metric| Dsv41CampaignError::MissingMetric {
        arm_id: cell.arm_id.clone(),
        context_length: cell.context_length,
        decode_length: cell.decode_length,
        metric,
    };
    let invalid = |metric| Dsv41CampaignError::InvalidMetric {
        arm_id: cell.arm_id.clone(),
        context_length: cell.context_length,
        decode_length: cell.decode_length,
        metric,
    };
    if cell.repetitions != preregistration.repetitions {
        return Err(Dsv41CampaignError::RepetitionMismatch {
            arm_id: cell.arm_id.clone(),
            expected: preregistration.repetitions,
            actual: cell.repetitions,
        });
    }
    let quality = cell
        .quality
        .ok_or_else(|| missing(CampaignMetricV1::Quality))?;
    if !quality.is_finite() {
        return Err(invalid(CampaignMetricV1::Quality));
    }
    let metric = cell
        .memory_metric
        .ok_or_else(|| missing(CampaignMetricV1::MemoryMetric))?;
    if metric != preregistration.memory_metric {
        return Err(invalid(CampaignMetricV1::MemoryMetric));
    }
    let bytes = cell
        .memory_bytes
        .ok_or_else(|| missing(CampaignMetricV1::MemoryBytes))?;
    if bytes == 0 {
        return Err(invalid(CampaignMetricV1::MemoryBytes));
    }
    let latency = cell
        .decode_latency
        .ok_or_else(|| missing(CampaignMetricV1::DecodeLatency))?;
    let ordered = [
        latency.p50_ms,
        latency.p90_ms,
        latency.p99_ms,
        latency.max_ms,
    ];
    let min_samples = u64::from(preregistration.repetitions);
    if latency.samples < min_samples
        || ordered
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || ordered.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(invalid(CampaignMetricV1::DecodeLatency));
    }
    let tokens_per_second = cell
        .tokens_per_second
        .ok_or_else(|| missing(CampaignMetricV1::TokensPerSecond))?;
    if !tokens_per_second.is_finite() || tokens_per_second <= 0.0 {
        return Err(invalid(CampaignMetricV1::TokensPerSecond));
    }
    Ok(())
}

fn validate_id(field: &'static str, value: &str) -> Result<(), Dsv41CampaignError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Dsv41CampaignError::EmptyField { field });
    }
    if trimmed != value || value.len() > MAX_DSV41_CAMPAIGN_ID_BYTES {
        return Err(Dsv41CampaignError::InvalidField { field });
    }
    Ok(())
}

fn validate_hex(field: &'static str, value: &str, len: usize) -> Result<(), Dsv41CampaignError> {
    if value.len() != len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(Dsv41CampaignError::InvalidField { field });
    }
    Ok(())
}

fn validate_lengths(field: &'static str, values: &[u32]) -> Result<(), Dsv41CampaignError> {
    if values.is_empty() || values[0] == 0 || values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(Dsv41CampaignError::InvalidField { field });
    }
    Ok(())
}

/// Fail-closed DSV41-5 campaign errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dsv41CampaignError {
    /// A required field was empty.
    EmptyField { field: &'static str },
    /// A field was malformed, non-canonical, or out of range.
    InvalidField { field: &'static str },
    /// Prompt suite declared zero prompts.
    EmptyPromptSuite,
    /// Fewer than two repetitions were preregistered.
    TooFewRepetitions { repetitions: u32 },
    /// Quality budget was not finite and non-negative.
    InvalidQualityBudget,
    /// Two arms shared an identifier.
    DuplicateArm { arm_id: String },
    /// Arm primitives were not strictly increasing.
    UnsortedOrDuplicatePrimitives { arm_id: String },
    /// No dense baseline arm was declared.
    MissingDenseBaseline,
    /// More than one dense baseline arm was declared.
    MultipleDenseBaselines,
    /// No candidate arm was declared.
    MissingCandidateArm,
    /// Evidence named a different campaign.
    CampaignMismatch,
    /// Evidence came from a dirty worktree.
    DirtyWorktree,
    /// Exact-head CI was absent or not green.
    ExactHeadCiNotGreen,
    /// Evidence backend differed from the preregistration.
    BackendMismatch,
    /// Evidence cell count differed from the preregistered grid.
    CellCountMismatch { expected: usize, actual: usize },
    /// A preregistered cell was absent.
    MissingCell {
        arm_id: String,
        context_length: u32,
        decode_length: u32,
    },
    /// A preregistered cell appeared more than once.
    DuplicateCell {
        arm_id: String,
        context_length: u32,
        decode_length: u32,
    },
    /// A cell ran a different number of repetitions.
    RepetitionMismatch {
        arm_id: String,
        expected: u32,
        actual: u32,
    },
    /// A cell omitted a required metric.
    MissingMetric {
        arm_id: String,
        context_length: u32,
        decode_length: u32,
        metric: CampaignMetricV1,
    },
    /// A cell reported an invalid or incomparable metric.
    InvalidMetric {
        arm_id: String,
        context_length: u32,
        decode_length: u32,
        metric: CampaignMetricV1,
    },
    /// The record is a synthetic fixture, not evidence.
    SyntheticFixtureIsNotEvidence,
}

impl fmt::Display for Dsv41CampaignError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField { field } => write!(output, "{field} must not be empty"),
            Self::InvalidField { field } => write!(output, "{field} is malformed"),
            Self::EmptyPromptSuite => output.write_str("prompt suite must contain prompts"),
            Self::TooFewRepetitions { repetitions } => {
                write!(output, "{repetitions} repetitions preregistered, at least 2 required")
            }
            Self::InvalidQualityBudget => {
                output.write_str("quality regression budget must be finite and non-negative")
            }
            Self::DuplicateArm { arm_id } => write!(output, "arm {arm_id} is declared twice"),
            Self::UnsortedOrDuplicatePrimitives { arm_id } => {
                write!(output, "arm {arm_id} primitives must be strictly increasing")
            }
            Self::MissingDenseBaseline => output.write_str("a dense baseline arm is required"),
            Self::MultipleDenseBaselines => output.write_str("only one dense baseline arm is allowed"),
            Self::MissingCandidateArm => output.write_str("at least one candidate arm is required"),
            Self::CampaignMismatch => output.write_str("evidence names a different campaign"),
            Self::DirtyWorktree => output.write_str("evidence was produced from a dirty worktree"),
            Self::ExactHeadCiNotGreen => output.write_str("exact-head CI is missing or not green"),
            Self::BackendMismatch => output.write_str("evidence backend differs from preregistration"),
            Self::CellCountMismatch { expected, actual } => {
                write!(output, "evidence has {actual} cells, preregistration requires {expected}")
            }
            Self::MissingCell {
                arm_id,
                context_length,
                decode_length,
            } => write!(
                output,
                "missing cell arm={arm_id} context={context_length} decode={decode_length}"
            ),
            Self::DuplicateCell {
                arm_id,
                context_length,
                decode_length,
            } => write!(
                output,
                "duplicate cell arm={arm_id} context={context_length} decode={decode_length}"
            ),
            Self::RepetitionMismatch {
                arm_id,
                expected,
                actual,
            } => write!(output, "arm {arm_id} ran {actual} repetitions, expected {expected}"),
            Self::MissingMetric {
                arm_id,
                context_length,
                decode_length,
                metric,
            } => write!(
                output,
                "cell arm={arm_id} context={context_length} decode={decode_length} is missing {metric:?}"
            ),
            Self::InvalidMetric {
                arm_id,
                context_length,
                decode_length,
                metric,
            } => write!(
                output,
                "cell arm={arm_id} context={context_length} decode={decode_length} has invalid {metric:?}"
            ),
            Self::SyntheticFixtureIsNotEvidence => {
                output.write_str("synthetic fixture records are never campaign evidence")
            }
        }
    }
}

impl std::error::Error for Dsv41CampaignError {}

#[cfg(test)]
mod tests {
    //! Every numeric value below is a synthetic placeholder used only to
    //! exercise validation rules. None is a measurement.

    use super::*;

    const HEAD: &str = "0123456789abcdef0123456789abcdef01234567";
    const SHA: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

    fn fields() -> Dsv41CampaignPreregistrationFieldsV1 {
        Dsv41CampaignPreregistrationFieldsV1 {
            campaign_id: "synthetic.dsv41-5.fixture".into(),
            model: CampaignModelV1 {
                model_id: "synthetic/model".into(),
                revision: HEAD.into(),
                weights_sha256: SHA.into(),
            },
            prompt_suite: CampaignPromptSuiteV1 {
                suite_id: "synthetic.prompts".into(),
                suite_sha256: SHA.into(),
                prompt_count: 4,
            },
            backend: CampaignBackendV1::Cpu,
            arms: vec![
                CampaignArmV1 {
                    arm_id: "dense".into(),
                    primitives: Vec::new(),
                },
                CampaignArmV1 {
                    arm_id: "fp4".into(),
                    primitives: vec![Dsv41PrimitiveV1::Fp4E2M1Kv],
                },
            ],
            context_lengths: vec![256, 1024],
            decode_lengths: vec![32],
            repetitions: 3,
            memory_metric: CampaignMemoryMetricV1::PeakNnisOwnedAllocationBytes,
            quality_metric: CampaignQualityMetricV1 {
                metric_id: "synthetic.quality".into(),
                direction: QualityDirectionV1::HigherIsBetter,
                max_regression: 0.0,
            },
        }
    }

    fn prereg() -> Dsv41CampaignPreregistrationV1 {
        Dsv41CampaignPreregistrationV1::new(fields()).unwrap()
    }

    fn synthetic_cell(arm_id: &str, context_length: u32) -> CampaignCellV1 {
        CampaignCellV1 {
            arm_id: arm_id.into(),
            context_length,
            decode_length: 32,
            repetitions: 3,
            quality: Some(1.0),
            memory_metric: Some(CampaignMemoryMetricV1::PeakNnisOwnedAllocationBytes),
            memory_bytes: Some(1),
            decode_latency: Some(DecodeLatencyDistributionV1 {
                samples: 3,
                p50_ms: 1.0,
                p90_ms: 1.0,
                p99_ms: 1.0,
                max_ms: 1.0,
            }),
            tokens_per_second: Some(1.0),
        }
    }

    fn synthetic_record() -> Dsv41CampaignEvidenceV1 {
        Dsv41CampaignEvidenceV1 {
            kind: CampaignRecordKindV1::SyntheticFixture,
            campaign_id: "synthetic.dsv41-5.fixture".into(),
            provenance: CampaignProvenanceV1 {
                git_head: HEAD.into(),
                worktree_clean: true,
                ci_run_id: 1,
                ci_green_on_exact_head: true,
            },
            hardware: CampaignHardwareV1 {
                backend: CampaignBackendV1::Cpu,
                device_name: "synthetic-device".into(),
                driver_version: "synthetic-runtime".into(),
                host_os: "synthetic-os".into(),
            },
            cells: vec![
                synthetic_cell("dense", 256),
                synthetic_cell("dense", 1024),
                synthetic_cell("fp4", 256),
                synthetic_cell("fp4", 1024),
            ],
        }
    }

    #[test]
    fn synthetic_fixture_is_never_evidence_even_when_complete() {
        let prereg = prereg();
        let record = synthetic_record();
        validate_dsv41_evidence_structure(&prereg, &record).unwrap();
        assert_eq!(
            validate_dsv41_evidence(&prereg, &record),
            Err(Dsv41CampaignError::SyntheticFixtureIsNotEvidence)
        );
    }

    #[test]
    fn each_missing_metric_is_rejected() {
        let prereg = prereg();
        type Strip = fn(&mut CampaignCellV1);
        let cases: [(CampaignMetricV1, Strip); 5] = [
            (CampaignMetricV1::Quality, |cell| cell.quality = None),
            (CampaignMetricV1::MemoryMetric, |cell| {
                cell.memory_metric = None
            }),
            (CampaignMetricV1::MemoryBytes, |cell| {
                cell.memory_bytes = None
            }),
            (CampaignMetricV1::DecodeLatency, |cell| {
                cell.decode_latency = None
            }),
            (CampaignMetricV1::TokensPerSecond, |cell| {
                cell.tokens_per_second = None
            }),
        ];
        for (metric, strip) in cases {
            let mut record = synthetic_record();
            strip(&mut record.cells[3]);
            let expected = Dsv41CampaignError::MissingMetric {
                arm_id: "fp4".into(),
                context_length: 1024,
                decode_length: 32,
                metric,
            };
            assert_eq!(
                validate_dsv41_evidence_structure(&prereg, &record),
                Err(expected.clone())
            );
            let mut physical = record;
            physical.kind = CampaignRecordKindV1::PhysicalExecution;
            assert_eq!(validate_dsv41_evidence(&prereg, &physical), Err(expected));
        }
    }

    #[test]
    fn incomparable_or_invalid_metrics_are_rejected() {
        let prereg = prereg();
        let invalid = |metric| {
            Err(Dsv41CampaignError::InvalidMetric {
                arm_id: "dense".into(),
                context_length: 256,
                decode_length: 32,
                metric,
            })
        };
        let mut record = synthetic_record();
        record.cells[0].memory_metric = Some(CampaignMemoryMetricV1::PeakProcessRssBytes);
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            invalid(CampaignMetricV1::MemoryMetric)
        );
        let mut record = synthetic_record();
        record.cells[0].quality = Some(f64::NAN);
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            invalid(CampaignMetricV1::Quality)
        );
        let mut record = synthetic_record();
        record.cells[0].memory_bytes = Some(0);
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            invalid(CampaignMetricV1::MemoryBytes)
        );
        for latency in [
            DecodeLatencyDistributionV1 {
                samples: 2,
                p50_ms: 1.0,
                p90_ms: 1.0,
                p99_ms: 1.0,
                max_ms: 1.0,
            },
            DecodeLatencyDistributionV1 {
                samples: 3,
                p50_ms: 2.0,
                p90_ms: 1.0,
                p99_ms: 3.0,
                max_ms: 4.0,
            },
            DecodeLatencyDistributionV1 {
                samples: 3,
                p50_ms: 0.0,
                p90_ms: 1.0,
                p99_ms: 1.0,
                max_ms: 1.0,
            },
            DecodeLatencyDistributionV1 {
                samples: 3,
                p50_ms: 1.0,
                p90_ms: 1.0,
                p99_ms: 1.0,
                max_ms: f64::INFINITY,
            },
        ] {
            let mut record = synthetic_record();
            record.cells[0].decode_latency = Some(latency);
            assert_eq!(
                validate_dsv41_evidence_structure(&prereg, &record),
                invalid(CampaignMetricV1::DecodeLatency)
            );
        }
        let mut record = synthetic_record();
        record.cells[0].tokens_per_second = Some(-1.0);
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            invalid(CampaignMetricV1::TokensPerSecond)
        );
        let mut record = synthetic_record();
        record.cells[0].repetitions = 2;
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            Err(Dsv41CampaignError::RepetitionMismatch {
                arm_id: "dense".into(),
                expected: 3,
                actual: 2,
            })
        );
    }

    #[test]
    fn provenance_hardware_and_grid_are_enforced() {
        let prereg = prereg();
        let mut record = synthetic_record();
        record.provenance.worktree_clean = false;
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            Err(Dsv41CampaignError::DirtyWorktree)
        );
        let mut record = synthetic_record();
        record.provenance.ci_green_on_exact_head = false;
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            Err(Dsv41CampaignError::ExactHeadCiNotGreen)
        );
        let mut record = synthetic_record();
        record.provenance.git_head = "main".into();
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            Err(Dsv41CampaignError::InvalidField { field: "git_head" })
        );
        let mut record = synthetic_record();
        record.hardware.backend = CampaignBackendV1::LegacyCudaCrossCheck;
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            Err(Dsv41CampaignError::BackendMismatch)
        );
        let mut record = synthetic_record();
        record.hardware.device_name = String::new();
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            Err(Dsv41CampaignError::EmptyField {
                field: "device_name"
            })
        );
        let mut record = synthetic_record();
        record.campaign_id = "other".into();
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            Err(Dsv41CampaignError::CampaignMismatch)
        );
        let mut record = synthetic_record();
        record.cells.pop();
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            Err(Dsv41CampaignError::CellCountMismatch {
                expected: 4,
                actual: 3
            })
        );
        let mut record = synthetic_record();
        record.cells[3] = synthetic_cell("fp4", 256);
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            Err(Dsv41CampaignError::DuplicateCell {
                arm_id: "fp4".into(),
                context_length: 256,
                decode_length: 32,
            })
        );
        let mut record = synthetic_record();
        record.cells[3] = synthetic_cell("fp4", 512);
        assert_eq!(
            validate_dsv41_evidence_structure(&prereg, &record),
            Err(Dsv41CampaignError::MissingCell {
                arm_id: "fp4".into(),
                context_length: 1024,
                decode_length: 32,
            })
        );
    }

    #[test]
    fn preregistration_rules_fail_closed() {
        let check = |edit: fn(&mut Dsv41CampaignPreregistrationFieldsV1), expected| {
            let mut fields = fields();
            edit(&mut fields);
            assert_eq!(Dsv41CampaignPreregistrationV1::new(fields), Err(expected));
        };
        check(
            |f| {
                f.arms.remove(0);
            },
            Dsv41CampaignError::MissingDenseBaseline,
        );
        check(
            |f| {
                f.arms.remove(1);
            },
            Dsv41CampaignError::MissingCandidateArm,
        );
        check(
            |f| f.arms[1].primitives.clear(),
            Dsv41CampaignError::MultipleDenseBaselines,
        );
        check(
            |f| f.arms[1].arm_id = "dense".into(),
            Dsv41CampaignError::DuplicateArm {
                arm_id: "dense".into(),
            },
        );
        check(
            |f| {
                f.arms[1].primitives =
                    vec![Dsv41PrimitiveV1::Fp4E2M1Kv, Dsv41PrimitiveV1::BoundedReplay]
            },
            Dsv41CampaignError::UnsortedOrDuplicatePrimitives {
                arm_id: "fp4".into(),
            },
        );
        check(
            |f| f.repetitions = 1,
            Dsv41CampaignError::TooFewRepetitions { repetitions: 1 },
        );
        check(
            |f| f.prompt_suite.prompt_count = 0,
            Dsv41CampaignError::EmptyPromptSuite,
        );
        check(
            |f| f.quality_metric.max_regression = f64::NAN,
            Dsv41CampaignError::InvalidQualityBudget,
        );
        check(
            |f| f.context_lengths = vec![1024, 256],
            Dsv41CampaignError::InvalidField {
                field: "context_lengths",
            },
        );
        check(
            |f| f.decode_lengths.clear(),
            Dsv41CampaignError::InvalidField {
                field: "decode_lengths",
            },
        );
        check(
            |f| f.model.weights_sha256 = SHA.to_uppercase(),
            Dsv41CampaignError::InvalidField {
                field: "weights_sha256",
            },
        );
        check(
            |f| f.model.revision = "v1".into(),
            Dsv41CampaignError::InvalidField { field: "revision" },
        );
        check(
            |f| f.campaign_id = " c".into(),
            Dsv41CampaignError::InvalidField {
                field: "campaign_id",
            },
        );
        let prereg = prereg();
        assert_eq!(prereg.required_cells(), 4);
        assert_eq!(prereg.dense_baseline_arm(), "dense");
    }
}
