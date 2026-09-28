//! DSV41-3 CPU reference encode/decode for FP4 E2M1 group-scaled KV rows.
//!
//! Implements the [`Fp4E2M1KvLayoutV1`] contract from `nnis-core` on the host:
//!
//! - scale per group: `F32` uses `amax / 6`; `E8M0` uses the smallest
//!   power of two `2^k` with `6 * 2^k >= amax` (`k` in `-127..=127`);
//! - codes: the scaled magnitude is rounded to the nearest E2M1 magnitude with
//!   ties to the even code; the sign bit is preserved, including `-0.0`;
//! - packing: value `2i` of a row's padded code stream is the low nibble of
//!   byte `i`, value `2i + 1` the high nibble; padding codes are zero;
//! - decode: `magnitude * scale` is formed exactly in F64 and rounded once to
//!   F32.
//!
//! The dense F32 input remains the reference: [`fp4_reconstruction_error`]
//! reports the isolated reconstruction error against it. This is a
//! correctness oracle only. It makes no model-quality, memory, latency, or
//! throughput claim, and FP4 E2M1 is distinct from the NNIS INT4 formats.

use core::fmt;

use nnis_core::kv_fp4::{
    Fp4E2M1KvLayoutV1, Fp4KvStorageV1, Fp4LayoutError, Fp4ScaleEncodingV1, FP4_E2M1_MAGNITUDES,
    FP4_E2M1_MAX_MAGNITUDE,
};

/// FP4 E2M1 group-scaled KV block encoded by the CPU reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuFp4E2M1KvBlockV1 {
    layout: Fp4E2M1KvLayoutV1,
    codes: Vec<u8>,
    scales: Vec<u8>,
}

impl CpuFp4E2M1KvBlockV1 {
    /// Encode dense finite row-major F32 values under `layout`.
    pub fn encode(layout: Fp4E2M1KvLayoutV1, values: &[f32]) -> Result<Self, CpuFp4Error> {
        let storage = layout.storage()?;
        let expected = host_len(storage.logical_values)?;
        if values.len() != expected {
            return Err(CpuFp4Error::ValueCountMismatch {
                expected,
                actual: values.len(),
            });
        }
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(CpuFp4Error::NonFiniteValue { index });
        }
        let mut codes = zeroed(host_len(storage.code_bytes)?)?;
        let mut scales = zeroed(host_len(storage.scale_bytes)?)?;
        let row_width = layout.row_width() as usize;
        let group_size = layout.group_size() as usize;
        let groups_per_row = layout.groups_per_row() as usize;
        let code_bytes_per_row = host_len(layout.code_bytes_per_row())?;
        let scale_bytes = layout.scale_encoding().bytes() as usize;
        for (row, row_values) in values.chunks_exact(row_width).enumerate() {
            let row_codes = &mut codes[row * code_bytes_per_row..(row + 1) * code_bytes_per_row];
            for group in 0..groups_per_row {
                let start = group * group_size;
                let end = row_width.min(start + group_size);
                let group_values = &row_values[start..end];
                let group_index = row * groups_per_row + group;
                let (scale, scale_bytes_le) =
                    choose_scale(layout.scale_encoding(), group_values, group_index)?;
                scales[group_index * scale_bytes..(group_index + 1) * scale_bytes]
                    .copy_from_slice(&scale_bytes_le[..scale_bytes]);
                for (offset, &value) in group_values.iter().enumerate() {
                    let code = quantize(value, scale);
                    if !decode_code(code, scale).is_finite() {
                        return Err(CpuFp4Error::DecodeOverflow { group: group_index });
                    }
                    let position = start + offset;
                    let shift = if position % 2 == 0 { 0 } else { 4 };
                    row_codes[position / 2] |= code << shift;
                }
            }
        }
        Ok(Self {
            layout,
            codes,
            scales,
        })
    }

    /// Reassemble a block from packed codes and raw scale bytes, fail closed.
    ///
    /// Lengths must match the layout, padding nibbles must be zero, every
    /// scale must be valid for its encoding, and every decoded value finite.
    pub fn from_parts(
        layout: Fp4E2M1KvLayoutV1,
        codes: Vec<u8>,
        scales: Vec<u8>,
    ) -> Result<Self, CpuFp4Error> {
        let storage = layout.storage()?;
        let expected_codes = host_len(storage.code_bytes)?;
        if codes.len() != expected_codes {
            return Err(CpuFp4Error::CodeBytesMismatch {
                expected: expected_codes,
                actual: codes.len(),
            });
        }
        let expected_scales = host_len(storage.scale_bytes)?;
        if scales.len() != expected_scales {
            return Err(CpuFp4Error::ScaleBytesMismatch {
                expected: expected_scales,
                actual: scales.len(),
            });
        }
        let block = Self {
            layout,
            codes,
            scales,
        };
        let row_width = layout.row_width() as usize;
        let padded = host_len(layout.padded_row_width())?;
        let code_bytes_per_row = host_len(layout.code_bytes_per_row())?;
        let group_size = layout.group_size() as usize;
        let groups_per_row = layout.groups_per_row() as usize;
        for row in 0..host_len(layout.rows())? {
            let row_codes = &block.codes[row * code_bytes_per_row..(row + 1) * code_bytes_per_row];
            for position in 0..padded {
                let code = nibble(row_codes, position);
                if position >= row_width {
                    if code != 0 {
                        return Err(CpuFp4Error::NonZeroPadding { row, position });
                    }
                    continue;
                }
                let group_index = row * groups_per_row + position / group_size;
                let scale = block.scale(group_index)?;
                if !decode_code(code, scale).is_finite() {
                    return Err(CpuFp4Error::DecodeOverflow { group: group_index });
                }
            }
        }
        Ok(block)
    }

    /// Validated layout.
    pub const fn layout(&self) -> &Fp4E2M1KvLayoutV1 {
        &self.layout
    }

    /// Packed code bytes (low nibble first).
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Raw little-endian scale bytes, one scale per group.
    pub fn scales(&self) -> &[u8] {
        &self.scales
    }

    /// Exact storage breakdown for this block's layout.
    pub fn storage(&self) -> Fp4KvStorageV1 {
        self.layout
            .storage()
            .expect("layout storage was validated at construction")
    }

    /// Decoded scale of one group as an exact F64 value.
    pub fn group_scale(&self, group: usize) -> Result<f64, CpuFp4Error> {
        self.scale(group)
    }

    /// Decode the block to dense row-major F32 values (padding excluded).
    pub fn decode(&self) -> Result<Vec<f32>, CpuFp4Error> {
        let row_width = self.layout.row_width() as usize;
        let rows = host_len(self.layout.rows())?;
        let total = rows
            .checked_mul(row_width)
            .ok_or(CpuFp4Error::HostIndexOverflow)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(total)
            .map_err(|_| CpuFp4Error::AllocationFailed)?;
        let code_bytes_per_row = host_len(self.layout.code_bytes_per_row())?;
        let group_size = self.layout.group_size() as usize;
        let groups_per_row = self.layout.groups_per_row() as usize;
        for row in 0..rows {
            let row_codes = &self.codes[row * code_bytes_per_row..(row + 1) * code_bytes_per_row];
            for position in 0..row_width {
                let group_index = row * groups_per_row + position / group_size;
                let scale = self.scale(group_index)?;
                output.push(decode_code(nibble(row_codes, position), scale));
            }
        }
        Ok(output)
    }

    fn scale(&self, group: usize) -> Result<f64, CpuFp4Error> {
        match self.layout.scale_encoding() {
            Fp4ScaleEncodingV1::F32 => {
                let bytes = self
                    .scales
                    .get(group * 4..group * 4 + 4)
                    .ok_or(CpuFp4Error::HostIndexOverflow)?;
                let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                if !scale.is_finite() || scale.is_sign_negative() {
                    return Err(CpuFp4Error::InvalidScale { group });
                }
                Ok(f64::from(scale))
            }
            Fp4ScaleEncodingV1::E8M0 => {
                let exponent = *self
                    .scales
                    .get(group)
                    .ok_or(CpuFp4Error::HostIndexOverflow)?;
                if exponent == u8::MAX {
                    return Err(CpuFp4Error::InvalidScale { group });
                }
                Ok(pow2(i32::from(exponent) - 127))
            }
        }
    }
}

/// Isolated reconstruction error of decoded values against the dense reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fp4ReconstructionErrorV1 {
    /// Values compared.
    pub values: u64,
    /// Largest absolute error.
    pub max_abs_error: f64,
    /// Sum of absolute errors in increasing index order.
    pub sum_abs_error: f64,
}

impl Fp4ReconstructionErrorV1 {
    /// Mean absolute error.
    pub fn mean_abs_error(&self) -> f64 {
        self.sum_abs_error / self.values as f64
    }
}

/// Compare decoded values with the dense F32 reference, in index order.
pub fn fp4_reconstruction_error(
    reference: &[f32],
    decoded: &[f32],
) -> Result<Fp4ReconstructionErrorV1, CpuFp4Error> {
    if reference.len() != decoded.len() || reference.is_empty() {
        return Err(CpuFp4Error::ValueCountMismatch {
            expected: reference.len(),
            actual: decoded.len(),
        });
    }
    let mut max_abs_error = 0.0f64;
    let mut sum_abs_error = 0.0f64;
    for (index, (&expected, &actual)) in reference.iter().zip(decoded).enumerate() {
        if !expected.is_finite() || !actual.is_finite() {
            return Err(CpuFp4Error::NonFiniteValue { index });
        }
        let error = (f64::from(expected) - f64::from(actual)).abs();
        max_abs_error = max_abs_error.max(error);
        sum_abs_error += error;
    }
    Ok(Fp4ReconstructionErrorV1 {
        values: reference.len() as u64,
        max_abs_error,
        sum_abs_error,
    })
}

/// Fail-closed CPU FP4 errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuFp4Error {
    /// Layout validation failed.
    Layout(Fp4LayoutError),
    /// Value count did not match the layout (or comparison inputs differ).
    ValueCountMismatch { expected: usize, actual: usize },
    /// A value at this flat index was NaN or infinite.
    NonFiniteValue { index: usize },
    /// Packed code length did not match the layout.
    CodeBytesMismatch { expected: usize, actual: usize },
    /// Scale byte length did not match the layout.
    ScaleBytesMismatch { expected: usize, actual: usize },
    /// A group scale was invalid for its encoding.
    InvalidScale { group: usize },
    /// A padding nibble was non-zero.
    NonZeroPadding { row: usize, position: usize },
    /// A value in this group would decode to a non-finite F32.
    DecodeOverflow { group: usize },
    /// A size does not fit host indexing.
    HostIndexOverflow,
    /// Host allocation failed.
    AllocationFailed,
}

impl From<Fp4LayoutError> for CpuFp4Error {
    fn from(error: Fp4LayoutError) -> Self {
        Self::Layout(error)
    }
}

impl fmt::Display for CpuFp4Error {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(error) => write!(output, "FP4 layout: {error}"),
            Self::ValueCountMismatch { expected, actual } => {
                write!(output, "FP4 value count {actual} does not match {expected}")
            }
            Self::NonFiniteValue { index } => {
                write!(output, "FP4 input value {index} is not finite")
            }
            Self::CodeBytesMismatch { expected, actual } => {
                write!(output, "FP4 code bytes {actual} do not match {expected}")
            }
            Self::ScaleBytesMismatch { expected, actual } => {
                write!(output, "FP4 scale bytes {actual} do not match {expected}")
            }
            Self::InvalidScale { group } => write!(output, "FP4 group {group} scale is invalid"),
            Self::NonZeroPadding { row, position } => {
                write!(
                    output,
                    "FP4 padding code at row {row} position {position} is non-zero"
                )
            }
            Self::DecodeOverflow { group } => {
                write!(
                    output,
                    "FP4 group {group} would decode to a non-finite value"
                )
            }
            Self::HostIndexOverflow => output.write_str("FP4 size does not fit host indexing"),
            Self::AllocationFailed => output.write_str("FP4 host allocation failed"),
        }
    }
}

impl std::error::Error for CpuFp4Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Layout(error) => Some(error),
            _ => None,
        }
    }
}

fn host_len(value: u64) -> Result<usize, CpuFp4Error> {
    usize::try_from(value).map_err(|_| CpuFp4Error::HostIndexOverflow)
}

fn zeroed(len: usize) -> Result<Vec<u8>, CpuFp4Error> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| CpuFp4Error::AllocationFailed)?;
    bytes.resize(len, 0);
    Ok(bytes)
}

/// Exact `2^k` for `k` in the normal F64 exponent range.
fn pow2(k: i32) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

fn nibble(row_codes: &[u8], position: usize) -> u8 {
    let byte = row_codes[position / 2];
    if position % 2 == 0 {
        byte & 0x0F
    } else {
        byte >> 4
    }
}

fn choose_scale(
    encoding: Fp4ScaleEncodingV1,
    values: &[f32],
    group: usize,
) -> Result<(f64, [u8; 4]), CpuFp4Error> {
    let amax = values
        .iter()
        .fold(0.0f32, |max, value| max.max(value.abs()));
    match encoding {
        Fp4ScaleEncodingV1::F32 => {
            let mut scale = amax / FP4_E2M1_MAX_MAGNITUDE;
            if amax > 0.0 && scale == 0.0 {
                // Subnormal amax underflowed; use the smallest positive scale.
                scale = f32::from_bits(1);
            }
            Ok((f64::from(scale), scale.to_le_bytes()))
        }
        Fp4ScaleEncodingV1::E8M0 => {
            if amax == 0.0 {
                return Ok((pow2(-127), [0, 0, 0, 0]));
            }
            let amax = f64::from(amax);
            let max = f64::from(FP4_E2M1_MAX_MAGNITUDE);
            let mut k = ((amax / max).log2().ceil() as i32).clamp(-127, 127);
            while k > -127 && max * pow2(k - 1) >= amax {
                k -= 1;
            }
            while max * pow2(k) < amax {
                if k == 127 {
                    return Err(CpuFp4Error::DecodeOverflow { group });
                }
                k += 1;
            }
            Ok((pow2(k), [(k + 127) as u8, 0, 0, 0]))
        }
    }
}

fn quantize(value: f32, scale: f64) -> u8 {
    let sign = if value.is_sign_negative() { 0x8 } else { 0x0 };
    if scale == 0.0 {
        return sign;
    }
    let x = f64::from(value.abs()) / scale;
    let mut code = 7u8;
    for index in 0..7 {
        let low = f64::from(FP4_E2M1_MAGNITUDES[index]);
        let high = f64::from(FP4_E2M1_MAGNITUDES[index + 1]);
        if x < high {
            let mid = (low + high) / 2.0;
            code = if x < mid {
                index as u8
            } else if x > mid {
                index as u8 + 1
            } else if index % 2 == 0 {
                index as u8
            } else {
                index as u8 + 1
            };
            break;
        }
    }
    sign | code
}

fn decode_code(code: u8, scale: f64) -> f32 {
    let magnitude = (f64::from(FP4_E2M1_MAGNITUDES[usize::from(code & 0x7)]) * scale) as f32;
    if code & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_core::kv_fp4::DenseKvBaselineV1;

    fn layout(rows: u64, width: u32, group: u32, scale: Fp4ScaleEncodingV1) -> Fp4E2M1KvLayoutV1 {
        Fp4E2M1KvLayoutV1::new(rows, width, group, scale).unwrap()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn representable_values_round_trip_bit_exactly() {
        let values = [0.0, 0.5, -1.0, 1.5, 2.0, -3.0, 4.0, 6.0];
        let block = CpuFp4E2M1KvBlockV1::encode(layout(1, 8, 8, Fp4ScaleEncodingV1::E8M0), &values)
            .unwrap();
        assert_eq!(block.scales(), &[127]);
        assert_eq!(block.codes(), &[0x10, 0x3A, 0xD4, 0x76]);
        assert_eq!(bits(&block.decode().unwrap()), bits(&values));
    }

    #[test]
    fn ties_round_to_even_code_and_large_values_use_top_code() {
        let values = [6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
        let block = CpuFp4E2M1KvBlockV1::encode(layout(1, 8, 8, Fp4ScaleEncodingV1::E8M0), &values)
            .unwrap();
        assert_eq!(
            block.decode().unwrap(),
            vec![6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
        );
    }

    #[test]
    fn e8m0_scale_is_smallest_covering_power_of_two() {
        let block = CpuFp4E2M1KvBlockV1::encode(
            layout(1, 4, 2, Fp4ScaleEncodingV1::E8M0),
            &[12.0, -1.0, 12.5, 0.1],
        )
        .unwrap();
        // amax 12 -> 2^1 exactly; amax 12.5 -> 2^2.
        assert_eq!(block.scales(), &[128, 129]);
        assert_eq!(block.group_scale(0).unwrap(), 2.0);
        assert_eq!(block.group_scale(1).unwrap(), 4.0);
        assert_eq!(block.decode().unwrap(), vec![12.0, -1.0, 12.0, 0.0]);
    }

    #[test]
    fn f32_scale_maps_group_amax_to_six() {
        let values = [3.0, -1.5, 0.75, 0.0];
        let block =
            CpuFp4E2M1KvBlockV1::encode(layout(1, 4, 4, Fp4ScaleEncodingV1::F32), &values).unwrap();
        assert_eq!(block.scales(), &0.5f32.to_le_bytes());
        assert_eq!(bits(&block.decode().unwrap()), bits(&values));
    }

    #[test]
    fn signed_zero_and_zero_groups_are_preserved() {
        for encoding in [Fp4ScaleEncodingV1::F32, Fp4ScaleEncodingV1::E8M0] {
            let values = [0.0, -0.0, 0.0, -0.0];
            let block = CpuFp4E2M1KvBlockV1::encode(layout(1, 4, 2, encoding), &values).unwrap();
            assert_eq!(bits(&block.decode().unwrap()), bits(&values));
        }
    }

    #[test]
    fn row_padding_is_zero_and_excluded_from_decode() {
        let values = [1.0, 2.0, 3.0, 4.0, 6.0, -6.0, 0.5, 1.0, 1.5, 2.0];
        let block = CpuFp4E2M1KvBlockV1::encode(layout(2, 5, 4, Fp4ScaleEncodingV1::E8M0), &values)
            .unwrap();
        assert_eq!(block.codes().len(), 8);
        assert_eq!(block.scales().len(), 4);
        // Row 0 position 5..8 and row 1 position 5..8 are padding.
        assert_eq!(block.codes()[2] >> 4, 0);
        assert_eq!(block.codes()[3], 0);
        assert_eq!(block.codes()[6] >> 4, 0);
        assert_eq!(block.codes()[7], 0);
        assert_eq!(block.decode().unwrap(), values.to_vec());
        let storage = block.storage();
        assert_eq!(storage.padding_values, 6);
        assert_eq!(storage.total_bytes, 8 + 4 + 24);
    }

    #[test]
    fn reference_error_is_bounded_by_half_top_spacing_times_scale() {
        let mut state = 0x2545_f491u32;
        let values: Vec<f32> = (0..256)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 24) as f32 * 8.0 - 4.0
            })
            .collect();
        for encoding in [Fp4ScaleEncodingV1::F32, Fp4ScaleEncodingV1::E8M0] {
            let block = CpuFp4E2M1KvBlockV1::encode(layout(8, 32, 16, encoding), &values).unwrap();
            let decoded = block.decode().unwrap();
            for (index, (&expected, &actual)) in values.iter().zip(&decoded).enumerate() {
                let scale = block.group_scale(index / 16).unwrap();
                // Largest E2M1 spacing is 2 (between 4 and 6), so half is 1.
                assert!((f64::from(expected) - f64::from(actual)).abs() <= scale);
            }
            let error = fp4_reconstruction_error(&values, &decoded).unwrap();
            assert_eq!(error.values, 256);
            assert!(error.max_abs_error > 0.0);
            assert!(error.mean_abs_error() <= error.max_abs_error);
        }
    }

    #[test]
    fn from_parts_round_trips_and_rejects_tampering() {
        let values = [1.0, -2.0, 3.0, 0.5, 6.0];
        let lay = layout(1, 5, 4, Fp4ScaleEncodingV1::E8M0);
        let block = CpuFp4E2M1KvBlockV1::encode(lay, &values).unwrap();
        let rebuilt =
            CpuFp4E2M1KvBlockV1::from_parts(lay, block.codes().to_vec(), block.scales().to_vec())
                .unwrap();
        assert_eq!(rebuilt, block);

        let mut padded = block.codes().to_vec();
        padded[2] |= 0x10;
        assert_eq!(
            CpuFp4E2M1KvBlockV1::from_parts(lay, padded, block.scales().to_vec()),
            Err(CpuFp4Error::NonZeroPadding {
                row: 0,
                position: 5
            })
        );
        let mut scales = block.scales().to_vec();
        scales[1] = u8::MAX;
        assert_eq!(
            CpuFp4E2M1KvBlockV1::from_parts(lay, block.codes().to_vec(), scales),
            Err(CpuFp4Error::InvalidScale { group: 1 })
        );
        assert_eq!(
            CpuFp4E2M1KvBlockV1::from_parts(lay, vec![0; 3], block.scales().to_vec()),
            Err(CpuFp4Error::CodeBytesMismatch {
                expected: 4,
                actual: 3
            })
        );
        assert_eq!(
            CpuFp4E2M1KvBlockV1::from_parts(lay, block.codes().to_vec(), vec![0; 1]),
            Err(CpuFp4Error::ScaleBytesMismatch {
                expected: 2,
                actual: 1
            })
        );

        let f32_layout = layout(1, 2, 2, Fp4ScaleEncodingV1::F32);
        for bad in [-1.0f32, f32::NAN, f32::INFINITY, -0.0] {
            assert_eq!(
                CpuFp4E2M1KvBlockV1::from_parts(f32_layout, vec![0x11], bad.to_le_bytes().to_vec()),
                Err(CpuFp4Error::InvalidScale { group: 0 })
            );
        }
        assert_eq!(
            CpuFp4E2M1KvBlockV1::from_parts(
                f32_layout,
                vec![0x77],
                f32::MAX.to_le_bytes().to_vec()
            ),
            Err(CpuFp4Error::DecodeOverflow { group: 0 })
        );
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        let lay = layout(1, 4, 4, Fp4ScaleEncodingV1::E8M0);
        assert_eq!(
            CpuFp4E2M1KvBlockV1::encode(lay, &[1.0, 2.0, 3.0]),
            Err(CpuFp4Error::ValueCountMismatch {
                expected: 4,
                actual: 3
            })
        );
        assert_eq!(
            CpuFp4E2M1KvBlockV1::encode(lay, &[1.0, f32::NAN, 3.0, 4.0]),
            Err(CpuFp4Error::NonFiniteValue { index: 1 })
        );
        assert_eq!(
            CpuFp4E2M1KvBlockV1::encode(lay, &[f32::MAX, 0.0, 0.0, 0.0]),
            Err(CpuFp4Error::DecodeOverflow { group: 0 })
        );
        assert!(fp4_reconstruction_error(&[], &[]).is_err());
    }

    #[test]
    fn storage_report_matches_encoded_bytes() {
        let lay = layout(16, 64, 32, Fp4ScaleEncodingV1::E8M0);
        let values = vec![1.0f32; 16 * 64];
        let block = CpuFp4E2M1KvBlockV1::encode(lay, &values).unwrap();
        let storage = block.storage();
        assert_eq!(storage.code_bytes as usize, block.codes().len());
        assert_eq!(storage.scale_bytes as usize, block.scales().len());
        assert_eq!(
            storage.total_bytes,
            (block.codes().len() + block.scales().len()) as u64 + storage.metadata_bytes
        );
        let (dense, total) = storage.compression_ratio_against(DenseKvBaselineV1::BF16);
        assert_eq!((dense, total), (16 * 64 * 16, (512 + 32 + 24) * 8));
    }
}
