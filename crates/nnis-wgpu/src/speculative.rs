//! DSV41-4 WGPU greedy verification of speculative drafts.
//!
//! [`WgpuGreedySpeculativeVerifierV1`] is the WGPU counterpart of
//! `nnis_cpu::speculative::CpuGreedySpeculativeVerifierV1`. Target logits stay
//! in a device buffer. One WGSL kernel, bound as a P4 artifact, derives each
//! scheduled row's greedy token: finite logits only, lowest index wins ties,
//! and the first non-finite column of a row is reported. The host then applies
//! the backend-neutral `verify_scheduled_draft` rule to those tokens.
//!
//! The comparison runs on raw bits mapped to a monotonic `u32` key, with
//! `-0.0` folded onto `+0.0`, so it is exactly the CPU's F32 `>` over finite
//! values, including subnormals, on any device. Rows are scanned serially from
//! column 0, like the CPU. The kernel is a reference path, not a performance
//! path.
//!
//! Declared tolerance: target tokens, verification outcomes and error variants
//! are identical to the CPU reference. It records acceptance counts only and
//! makes no latency, cost or throughput claim. It neither drafts tokens nor
//! chooses confidence policy.

use core::fmt;

use nnis_core::kernel_artifact::{KernelBindingKindV1, KernelElementTypeV1};
use nnis_core::speculative_verify::{
    verify_scheduled_draft, ConfidenceScheduleV1, SpeculativeAcceptanceStatsV1, SpeculativeDraftV1,
    SpeculativeError, SpeculativeVerificationV1,
};
use nnis_core::{
    BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableError, PortableFence,
    PortableQueue,
};

use crate::memory::{require_usage, WgpuBuffer};
use crate::numerical::{pop_scopes, Binding, WgpuF32KernelsV1};
use crate::WgpuDevice;

/// Contract version of the WGSL greedy-argmax kernel.
pub const WGSL_GREEDY_ARGMAX_CONTRACT_VERSION: u32 = 1;

/// Numerical policy of the WGSL greedy-argmax artifact.
pub const WGSL_GREEDY_ARGMAX_NUMERICAL_POLICY: &str =
    "wgsl-f32-finite-greedy-argmax-bitwise-ordered-lowest-index-ties-v1";

const WORKGROUP: u32 = 64;
const NO_NON_FINITE: u32 = u32::MAX;

const ARGMAX_SOURCE: &str = "
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> logits: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;

// Monotonic key of a finite F32 with -0.0 folded onto +0.0.
fn order_key(raw: u32) -> u32 {
    var bits = raw;
    if ((bits & 0x7fffffffu) == 0u) { bits = 0u; }
    if ((bits & 0x80000000u) != 0u) { return ~bits; }
    return bits | 0x80000000u;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let rows = params[0];
    let vocab = params[1];
    if (row >= rows) { return; }
    let base = row * vocab;
    var best = 0u;
    var best_key = 0u;
    var non_finite = 0xffffffffu;
    for (var column = 0u; column < vocab; column = column + 1u) {
        let bits = logits[base + column];
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            non_finite = column;
            break;
        }
        let key = order_key(bits);
        if (column == 0u || key > best_key) {
            best = column;
            best_key = key;
        }
    }
    result[2u * row] = best;
    result[2u * row + 1u] = non_finite;
}
";

/// Verification outcome plus the device-derived greedy tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuSpeculativeOutcomeV1 {
    /// Outcome of `verify_scheduled_draft` over `target_tokens`.
    pub verification: SpeculativeVerificationV1,
    /// Greedy target token of every scheduled row plus the bonus row.
    pub target_tokens: Vec<u32>,
    pub schema_version: u32,
    /// Fingerprint of the bound greedy-argmax artifact.
    pub artifact_fingerprint: [u8; 32],
}

/// Greedy-semantics WGPU verifier for one vocabulary size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuGreedySpeculativeVerifierV1 {
    vocab_size: u32,
}

impl WgpuGreedySpeculativeVerifierV1 {
    /// Construct a verifier for a non-zero vocabulary size.
    pub fn new(vocab_size: u32) -> Result<Self, WgpuSpeculativeError> {
        if vocab_size == 0 {
            return Err(WgpuSpeculativeError::ZeroVocabulary);
        }
        Ok(Self { vocab_size })
    }

    /// Vocabulary size of each logits row.
    pub const fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// Verify one draft step from row-major target logits in a device buffer.
    ///
    /// Checks run in the CPU order: draft tokens inside the vocabulary, then
    /// exactly `(scheduled + 1) * vocab_size` F32 values (`STORAGE` usage,
    /// owned by `device`), then finiteness row by row, then the contract.
    pub fn verify(
        &self,
        device: &WgpuDevice,
        draft: &SpeculativeDraftV1,
        schedule: &ConfidenceScheduleV1,
        target_logits: &WgpuBuffer,
    ) -> Result<WgpuSpeculativeOutcomeV1, WgpuSpeculativeError> {
        let (rows, expected) = self.expected_values(draft, schedule)?;
        let expected_bytes = (expected as u64)
            .checked_mul(4)
            .ok_or(WgpuSpeculativeError::HostIndexOverflow)?;
        if target_logits.len() != expected_bytes {
            return Err(if target_logits.len() % 4 == 0 {
                WgpuSpeculativeError::LogitsLengthMismatch {
                    expected,
                    actual: usize::try_from(target_logits.len() / 4)
                        .map_err(|_| WgpuSpeculativeError::HostIndexOverflow)?,
                }
            } else {
                WgpuSpeculativeError::LogitsByteLengthMismatch {
                    expected_bytes,
                    actual_bytes: target_logits.len(),
                }
            });
        }
        self.run(device, draft, schedule, target_logits, rows, expected)
    }

    /// Upload host logits and verify them on `device`.
    ///
    /// Length errors are reported before any upload, exactly as the CPU
    /// reference reports them.
    pub fn verify_host(
        &self,
        device: &WgpuDevice,
        draft: &SpeculativeDraftV1,
        schedule: &ConfidenceScheduleV1,
        target_logits: &[f32],
    ) -> Result<WgpuSpeculativeOutcomeV1, WgpuSpeculativeError> {
        let (rows, expected) = self.expected_values(draft, schedule)?;
        if target_logits.len() != expected {
            return Err(WgpuSpeculativeError::LogitsLengthMismatch {
                expected,
                actual: target_logits.len(),
            });
        }
        let bytes: Vec<u8> = target_logits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut buffer = device.create_buffer(BufferDesc::new(
            bytes.len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            MemoryClass::DeviceLocal,
        )?)?;
        device
            .create_queue()?
            .write_buffer(&mut buffer, 0, &bytes)?
            .wait()?;
        self.run(device, draft, schedule, &buffer, rows, expected)
    }

    /// Verify one draft step and add its counts to `stats`.
    ///
    /// `stats` is unchanged when verification or recording fails.
    pub fn verify_and_record(
        &self,
        device: &WgpuDevice,
        stats: &mut SpeculativeAcceptanceStatsV1,
        draft: &SpeculativeDraftV1,
        schedule: &ConfidenceScheduleV1,
        target_logits: &WgpuBuffer,
    ) -> Result<WgpuSpeculativeOutcomeV1, WgpuSpeculativeError> {
        let outcome = self.verify(device, draft, schedule, target_logits)?;
        stats.record(&outcome.verification)?;
        Ok(outcome)
    }

    fn expected_values(
        &self,
        draft: &SpeculativeDraftV1,
        schedule: &ConfidenceScheduleV1,
    ) -> Result<(usize, usize), WgpuSpeculativeError> {
        if let Some(index) = draft
            .tokens()
            .iter()
            .position(|&token| token >= self.vocab_size)
        {
            return Err(WgpuSpeculativeError::TokenOutOfVocabulary {
                index,
                token: draft.tokens()[index],
            });
        }
        let rows = schedule.scheduled_len(draft) as usize + 1;
        let expected = rows
            .checked_mul(self.vocab_size as usize)
            .ok_or(WgpuSpeculativeError::HostIndexOverflow)?;
        Ok((rows, expected))
    }

    fn run(
        &self,
        device: &WgpuDevice,
        draft: &SpeculativeDraftV1,
        schedule: &ConfidenceScheduleV1,
        target_logits: &WgpuBuffer,
        rows: usize,
        values: usize,
    ) -> Result<WgpuSpeculativeOutcomeV1, WgpuSpeculativeError> {
        if u32::try_from(values).is_err() {
            return Err(WgpuSpeculativeError::ExceedsDeviceIndexing);
        }
        let kernels = WgpuF32KernelsV1::new(device)?;
        let queue = kernels.queue();
        queue.check_owner(target_logits)?;
        require_usage(target_logits, BufferUsages::STORAGE, "WGPU greedy argmax")?;
        let raw = queue.device();
        let param_bytes: Vec<u8> = [rows as u32, self.vocab_size]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let result_bytes = (rows as u64) * 8;
        raw.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        raw.push_error_scope(wgpu::ErrorFilter::Validation);
        let params = raw.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nnis.wgpu.greedy_argmax.params"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.queue().write_buffer(&params, 0, &param_bytes);
        let result = raw.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nnis.wgpu.greedy_argmax.result"),
            size: result_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        pop_scopes(raw, "WGPU greedy argmax allocation")?;
        let bindings = [
            Binding {
                buffer: &params,
                kind: KernelBindingKindV1::StorageReadOnly,
                element: KernelElementTypeV1::U32,
                bytes: 8,
            },
            Binding {
                buffer: target_logits.raw(),
                kind: KernelBindingKindV1::StorageReadOnly,
                element: KernelElementTypeV1::F32,
                bytes: target_logits.len(),
            },
            Binding {
                buffer: &result,
                kind: KernelBindingKindV1::StorageReadWrite,
                element: KernelElementTypeV1::U32,
                bytes: result_bytes,
            },
        ];
        let artifact = kernels.artifact(
            "nnis.wgpu.greedy_argmax",
            WGSL_GREEDY_ARGMAX_NUMERICAL_POLICY,
            ARGMAX_SOURCE,
            &bindings,
            [WORKGROUP, 1, 1],
        )?;
        let groups = kernels.groups(rows, WORKGROUP)?;
        raw.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        raw.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = raw.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nnis.wgpu.greedy_argmax"),
        });
        kernels.dispatch(&mut encoder, &artifact, &bindings, groups);
        queue.queue().submit(Some(encoder.finish()));
        pop_scopes(raw, "WGPU greedy argmax")?;
        let words: Vec<u32> = queue
            .read_wgpu(&result, 0, result_bytes)?
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect();
        let mut target_tokens = Vec::with_capacity(rows);
        for (row, pair) in words.chunks_exact(2).enumerate() {
            if pair[1] != NO_NON_FINITE {
                return Err(WgpuSpeculativeError::NonFiniteLogit {
                    row,
                    column: pair[1],
                });
            }
            target_tokens.push(pair[0]);
        }
        let verification = verify_scheduled_draft(draft, schedule, &target_tokens)?;
        Ok(WgpuSpeculativeOutcomeV1 {
            verification,
            target_tokens,
            schema_version: WGSL_GREEDY_ARGMAX_CONTRACT_VERSION,
            artifact_fingerprint: *artifact.artifact_fingerprint(),
        })
    }
}

/// Fail-closed WGPU speculative-verification errors.
///
/// Variants shared with `CpuSpeculativeError` carry identical payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgpuSpeculativeError {
    /// Backend-neutral contract validation failed.
    Contract(SpeculativeError),
    /// Vocabulary size was zero.
    ZeroVocabulary,
    /// A draft token was outside the vocabulary.
    TokenOutOfVocabulary { index: usize, token: u32 },
    /// Logits length was not `(scheduled + 1) * vocab_size` values.
    LogitsLengthMismatch { expected: usize, actual: usize },
    /// Device logits buffer is not a whole number of F32 values.
    LogitsByteLengthMismatch {
        expected_bytes: u64,
        actual_bytes: u64,
    },
    /// A logit was NaN or infinite.
    NonFiniteLogit { row: usize, column: u32 },
    /// A size did not fit host indexing.
    HostIndexOverflow,
    /// Logit count exceeds `u32` device indexing.
    ExceedsDeviceIndexing,
    /// Portable WGPU allocation, binding, dispatch or readback failed.
    Portable(PortableError),
}

impl From<SpeculativeError> for WgpuSpeculativeError {
    fn from(error: SpeculativeError) -> Self {
        Self::Contract(error)
    }
}

impl From<PortableError> for WgpuSpeculativeError {
    fn from(error: PortableError) -> Self {
        Self::Portable(error)
    }
}

impl fmt::Display for WgpuSpeculativeError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(output, "speculative contract: {error}"),
            Self::ZeroVocabulary => output.write_str("verifier vocabulary size must be non-zero"),
            Self::TokenOutOfVocabulary { index, token } => write!(
                output,
                "draft token {index} ({token}) is outside the vocabulary"
            ),
            Self::LogitsLengthMismatch { expected, actual } => write!(
                output,
                "target logits have {actual} values, expected {expected}"
            ),
            Self::LogitsByteLengthMismatch {
                expected_bytes,
                actual_bytes,
            } => write!(
                output,
                "target logits buffer has {actual_bytes} bytes, expected {expected_bytes}"
            ),
            Self::NonFiniteLogit { row, column } => write!(
                output,
                "target logit at row {row} column {column} is not finite"
            ),
            Self::HostIndexOverflow => output.write_str("logits size does not fit host indexing"),
            Self::ExceedsDeviceIndexing => {
                output.write_str("logits size exceeds u32 WGPU device indexing")
            }
            Self::Portable(error) => write!(output, "WGPU speculative verification: {error}"),
        }
    }
}

impl std::error::Error for WgpuSpeculativeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            Self::Portable(error) => Some(error),
            _ => None,
        }
    }
}
