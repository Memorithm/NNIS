//! Synchronous finite-F32 CPU reference execution over portable byte buffers.
//!
//! All buffers use little-endian binary32 and require STORAGE. Each operation
//! stages its complete output before writing: any returned error preserves the
//! destination bytes and capacity. This is not multi-operation crash atomicity.
//! The scratch limit covers requested payload bytes, not allocator overhead.

use nnis_core::{BufferUsages, PortableError, Result};

use crate::{require_usage, reserve_bytes, CpuBuffer};

/// Numerical contract version; independent from memory and model schemas.
pub const CPU_F32_CONTRACT_VERSION: u32 = 1;
/// Ordered binary32 arithmetic, finite inputs/intermediates, explicit fused dot.
pub const CPU_F32_NUMERICAL_POLICY: &str = "finite-f32-le-ordered-fma-v1";

/// Elementwise binary operation; no implicit broadcasting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuF32BinaryOp {
    Add,
    Multiply,
}

/// Operation identity recorded after successful synchronous execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuF32Operation {
    Add,
    Multiply,
    Relu,
    Sum,
    ProjectKn,
    Gather,
    ScatterAdd,
}

/// Successful software execution accounting, not time or physical residency.
///
/// Scratch is already released on return. Its peak retained Vec capacity can
/// exceed its requested payload; allocator bookkeeping and input/output buffers
/// are excluded. No process-wide memory or performance claim is implied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuF32ReportV1 {
    pub schema_version: u32,
    pub operation: CpuF32Operation,
    pub output_values: usize,
    pub scratch_payload_bytes: usize,
    pub scratch_capacity_bytes: usize,
}

/// Scalar reference kernels with an explicit temporary-output payload ceiling.
///
/// Uses no vendor library, unsafe code, parallel reduction or implicit dtype
/// conversion. It is a reference executor, not a general tensor/BLAS library.
#[derive(Debug, Clone, Copy)]
pub struct CpuF32KernelsV1 {
    max_scratch_bytes: usize,
}

impl CpuF32KernelsV1 {
    /// Set a positive representable limit for one operation's scratch payload.
    pub fn new(max_scratch_bytes: usize) -> Result<Self> {
        if max_scratch_bytes == 0 || max_scratch_bytes > isize::MAX as usize {
            return Err(PortableError::InvalidDescriptor(
                "CPU F32 scratch limit must be positive and fit isize::MAX",
            ));
        }
        Ok(Self { max_scratch_bytes })
    }

    /// Return the requested scratch payload ceiling, not a total memory quota.
    #[must_use]
    pub const fn max_scratch_bytes(&self) -> usize {
        self.max_scratch_bytes
    }

    /// Add or multiply equal-length vectors without broadcasting.
    pub fn binary(
        &self,
        operation: CpuF32BinaryOp,
        left: &CpuBuffer,
        right: &CpuBuffer,
        output: &mut CpuBuffer,
    ) -> Result<CpuF32ReportV1> {
        let left = Input::new(left)?;
        let right = Input::new(right)?;
        require_equal(left.len(), right.len())?;
        let identity = match operation {
            CpuF32BinaryOp::Add => CpuF32Operation::Add,
            CpuF32BinaryOp::Multiply => CpuF32Operation::Multiply,
        };
        self.evaluate(output, left.len(), identity, |index| {
            Ok(match operation {
                CpuF32BinaryOp::Add => left.at(index) + right.at(index),
                CpuF32BinaryOp::Multiply => left.at(index) * right.at(index),
            })
        })
    }

    /// ReLU maps negative values and both signed zeros to positive zero.
    pub fn relu(&self, input: &CpuBuffer, output: &mut CpuBuffer) -> Result<CpuF32ReportV1> {
        let input = Input::new(input)?;
        self.evaluate(output, input.len(), CpuF32Operation::Relu, |index| {
            let value = input.at(index);
            Ok(if value > 0.0 { value } else { 0.0 })
        })
    }

    /// Reduce in increasing index order from +0 using binary32 addition.
    ///
    /// Output must contain exactly one F32. Every intermediate must be finite;
    /// an overflowing partial sum is not rescued by later cancellation.
    pub fn sum(&self, input: &CpuBuffer, output: &mut CpuBuffer) -> Result<CpuF32ReportV1> {
        let input = Input::new(input)?;
        self.evaluate(output, 1, CpuF32Operation::Sum, |_| {
            let mut sum = 0.0_f32;
            for index in 0..input.len() {
                sum = finite(sum + input.at(index))?;
            }
            Ok(sum)
        })
    }

    /// Project [1,K] x row-major [K,N] -> [1,N], without bias or transposition.
    ///
    /// Each output starts at +0 and calls F32 mul_add in increasing K order.
    /// Every intermediate must be finite. K and N must be non-zero.
    pub fn project_kn(
        &self,
        input: &CpuBuffer,
        weights: &CpuBuffer,
        output: &mut CpuBuffer,
        k: usize,
        n: usize,
    ) -> Result<CpuF32ReportV1> {
        if k == 0 || n == 0 {
            return Err(PortableError::InvalidDescriptor(
                "CPU F32 matrix dimensions must be positive",
            ));
        }
        let elements = k.checked_mul(n).ok_or(PortableError::InvalidDescriptor(
            "CPU F32 matrix shape overflows usize",
        ))?;
        let input = Input::new(input)?;
        let weights = Input::new(weights)?;
        require_equal(input.len(), k)?;
        require_equal(weights.len(), elements)?;
        self.evaluate(output, n, CpuF32Operation::ProjectKn, |column| {
            let mut sum = 0.0_f32;
            for row in 0..k {
                sum = finite(input.at(row).mul_add(weights.at(row * n + column), sum))?;
            }
            Ok(sum)
        })
    }

    /// Gather selected elements in index-list order. Repeated indices are valid.
    ///
    /// The whole input is validated, including unselected values. Empty results
    /// are unsupported because portable buffers are non-empty.
    pub fn gather(
        &self,
        input: &CpuBuffer,
        indices: &[usize],
        output: &mut CpuBuffer,
    ) -> Result<CpuF32ReportV1> {
        let input = Input::new(input)?;
        validate_indices(indices, input.len())?;
        self.evaluate(output, indices.len(), CpuF32Operation::Gather, |index| {
            Ok(input.at(indices[index]))
        })
    }

    /// Add source[i] to output[indices[i]] in index-list order.
    ///
    /// Repeated indices accumulate serially, not in parallel. Unselected output
    /// values are preserved. The complete old destination must be finite.
    /// An empty list is invalid rather than an ambiguous no-op.
    pub fn scatter_add(
        &self,
        source: &CpuBuffer,
        indices: &[usize],
        output: &mut CpuBuffer,
    ) -> Result<CpuF32ReportV1> {
        let source = Input::new(source)?;
        require_equal(source.len(), indices.len())?;
        let old = Input::new(output)?;
        validate_indices(indices, old.len())?;
        let mut scratch = self.scratch(output)?;
        scratch.extend_from_slice(&output.bytes);
        for (index, &destination) in indices.iter().enumerate() {
            let value = finite(value_at(&scratch, destination) + source.at(index))?;
            let offset = destination * 4;
            scratch[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        Ok(commit(output, scratch, CpuF32Operation::ScatterAdd))
    }

    fn evaluate(
        &self,
        output: &mut CpuBuffer,
        values: usize,
        operation: CpuF32Operation,
        mut compute: impl FnMut(usize) -> Result<f32>,
    ) -> Result<CpuF32ReportV1> {
        require_equal(word_count(output)?, values)?;
        let mut scratch = self.scratch(output)?;
        for index in 0..values {
            scratch.extend_from_slice(&finite(compute(index)?)?.to_le_bytes());
        }
        Ok(commit(output, scratch, operation))
    }

    fn scratch(&self, output: &CpuBuffer) -> Result<Vec<u8>> {
        word_count(output)?;
        if output.len() > self.max_scratch_bytes {
            return Err(PortableError::Unsupported(
                "CPU F32 output exceeds scratch payload limit".to_string(),
            ));
        }
        reserve_bytes(output.len())
    }
}

struct Input<'a> {
    buffer: &'a CpuBuffer,
    values: usize,
}

impl<'a> Input<'a> {
    fn new(buffer: &'a CpuBuffer) -> Result<Self> {
        let values = word_count(buffer)?;
        let input = Self { buffer, values };
        for index in 0..values {
            finite(input.at(index))?;
        }
        Ok(input)
    }

    fn len(&self) -> usize {
        self.values
    }

    fn at(&self, index: usize) -> f32 {
        value_at(&self.buffer.bytes, index)
    }
}

fn word_count(buffer: &CpuBuffer) -> Result<usize> {
    require_usage(buffer, BufferUsages::STORAGE, "CPU F32 execution")?;
    if buffer.is_empty() || buffer.len() % 4 != 0 {
        return Err(PortableError::InvalidDescriptor(
            "CPU F32 buffer must contain a non-zero multiple of four bytes",
        ));
    }
    Ok(buffer.len() / 4)
}

// Callers validate buffer lengths and every index before reaching this helper.
fn value_at(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 4;
    f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn finite(value: f32) -> Result<f32> {
    if !value.is_finite() {
        return Err(PortableError::InvalidDescriptor(
            "CPU F32 input or arithmetic intermediate is non-finite",
        ));
    }
    Ok(value)
}

fn require_equal(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(PortableError::InvalidDescriptor("CPU F32 shape mismatch"));
    }
    Ok(())
}

fn validate_indices(indices: &[usize], limit: usize) -> Result<()> {
    if indices.is_empty() || indices.iter().any(|&index| index >= limit) {
        return Err(PortableError::InvalidDescriptor(
            "CPU F32 index list is empty or out of range",
        ));
    }
    Ok(())
}

fn commit(output: &mut CpuBuffer, scratch: Vec<u8>, operation: CpuF32Operation) -> CpuF32ReportV1 {
    let report = CpuF32ReportV1 {
        schema_version: CPU_F32_CONTRACT_VERSION,
        operation,
        output_values: output.len() / 4,
        scratch_payload_bytes: scratch.len(),
        scratch_capacity_bytes: scratch.capacity(),
    };
    output.bytes.copy_from_slice(&scratch);
    report
}
