//! Deterministic <=2-bit ternary reference weight codec for NNIS.
//!
//! The fixed baseline stores four 2-bit codes per byte with the mapping
//! `0 -> 0`, `1 -> +1`, `2 -> -1`, while code `3` is reserved.
//! One positive finite F32 scale applies to the whole logical tensor.
//!
//! This module is host-side representation evidence only. Runtime execution is
//! added separately so storage, execution and qualification claims remain
//! independently versioned.

use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of the deterministic ternary INT2 reference storage contract.
pub const NNIS_INT2_REFERENCE_STORAGE_VERSION: u32 = 1;
/// Canonical serialized header: magic + element count + F32 scale.
pub const NNIS_INT2_REFERENCE_SERIALIZED_HEADER_BYTES: u64 = 16;
/// Code representing zero.
pub const NNIS_INT2_REFERENCE_CODE_ZERO: u8 = 0;
/// Code representing positive one.
pub const NNIS_INT2_REFERENCE_CODE_POSITIVE: u8 = 1;
/// Code representing negative one.
pub const NNIS_INT2_REFERENCE_CODE_NEGATIVE: u8 = 2;
/// Reserved 2-bit code. Canonical payloads never emit it.
pub const NNIS_INT2_REFERENCE_CODE_RESERVED: u8 = 3;

const SERIALIZED_MAGIC: [u8; 4] = *b"NI21";

/// Host-side deterministic quantization result for one logical tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Int2ReferenceQuantizedTensorV1 {
    pub element_count: u64,
    pub scale: f32,
    pub packed_values: Vec<u8>,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
}

impl Int2ReferenceQuantizedTensorV1 {
    /// Canonical representation bytes.
    ///
    /// Layout:
    /// - 4 bytes: ASCII magic `NI21`;
    /// - 8 bytes: little-endian logical element count;
    /// - 4 bytes: little-endian IEEE-754 F32 scale bits;
    /// - remaining bytes: four 2-bit codes per byte, least-significant lane
    ///   first.
    pub fn canonical_serialized_bytes(&self) -> Result<Vec<u8>> {
        validate_quantized_tensor(self)?;
        let header_len = usize::try_from(NNIS_INT2_REFERENCE_SERIALIZED_HEADER_BYTES)
            .map_err(|_| NnisError::invalid_input("INT2 serialized header does not fit usize"))?;
        let capacity = header_len
            .checked_add(self.packed_values.len())
            .ok_or_else(|| NnisError::invalid_input("INT2 serialized size overflows usize"))?;
        let mut encoded = Vec::with_capacity(capacity);
        encoded.extend_from_slice(&SERIALIZED_MAGIC);
        encoded.extend_from_slice(&self.element_count.to_le_bytes());
        encoded.extend_from_slice(&self.scale.to_bits().to_le_bytes());
        encoded.extend_from_slice(&self.packed_values);
        Ok(encoded)
    }
}

/// Quantize one F32 slice to the fixed ternary <=2-bit baseline.
///
/// The scale is the maximum absolute source value, except that an all-zero
/// tensor uses scale 1.0 so the contract remains finite and positive. Values
/// with magnitude strictly below half the scale map to zero; values at or
/// beyond the threshold map to the corresponding signed unit level.
pub fn quantize_int2_ternary_reference_v1(
    values: &[f32],
) -> Result<Int2ReferenceQuantizedTensorV1> {
    if values.is_empty() {
        return Err(NnisError::invalid_input(
            "INT2 reference quantization requires at least one value",
        ));
    }

    let mut max_abs = 0.0_f32;
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(NnisError::invalid_input(format!(
                "INT2 reference source value {index} is not finite"
            )));
        }
        max_abs = max_abs.max(value.abs());
    }

    let scale = if max_abs == 0.0 { 1.0 } else { max_abs };
    if !scale.is_finite() || scale <= 0.0 {
        return Err(NnisError::invalid_input(
            "INT2 reference scale is not a finite positive F32 value",
        ));
    }
    let threshold = scale * 0.5;

    let packed_len = values
        .len()
        .checked_add(3)
        .ok_or_else(|| NnisError::invalid_input("INT2 element count overflows usize"))?
        / 4;
    let mut packed_values = vec![0_u8; packed_len];
    let mut max_abs_error = 0.0_f32;
    let mut squared_error_sum = 0.0_f64;

    for (index, value) in values.iter().copied().enumerate() {
        let (code, level) = if value >= threshold {
            (NNIS_INT2_REFERENCE_CODE_POSITIVE, 1_i8)
        } else if value <= -threshold {
            (NNIS_INT2_REFERENCE_CODE_NEGATIVE, -1_i8)
        } else {
            (NNIS_INT2_REFERENCE_CODE_ZERO, 0_i8)
        };
        packed_values[index / 4] |= code << ((index % 4) * 2);

        let reconstructed = f32::from(level) * scale;
        let error = (value - reconstructed).abs();
        max_abs_error = max_abs_error.max(error);
        let error_f64 = f64::from(value) - f64::from(reconstructed);
        squared_error_sum += error_f64 * error_f64;
    }

    let element_count = u64::try_from(values.len())
        .map_err(|_| NnisError::invalid_input("INT2 element count exceeds u64"))?;
    let quantized = Int2ReferenceQuantizedTensorV1 {
        element_count,
        scale,
        packed_values,
        max_abs_error,
        mean_squared_error: squared_error_sum / values.len() as f64,
    };
    validate_quantized_tensor(&quantized)?;
    Ok(quantized)
}

/// Reconstruct F32 values from the fixed ternary reference payload.
pub fn dequantize_int2_ternary_reference_v1(
    quantized: &Int2ReferenceQuantizedTensorV1,
) -> Result<Vec<f32>> {
    validate_quantized_tensor(quantized)?;
    let element_count = usize::try_from(quantized.element_count)
        .map_err(|_| NnisError::invalid_input("INT2 element count does not fit usize"))?;
    let mut values = Vec::with_capacity(element_count);
    for index in 0..element_count {
        let code = (quantized.packed_values[index / 4] >> ((index % 4) * 2)) & 0x03;
        let level = match code {
            NNIS_INT2_REFERENCE_CODE_ZERO => 0_i8,
            NNIS_INT2_REFERENCE_CODE_POSITIVE => 1_i8,
            NNIS_INT2_REFERENCE_CODE_NEGATIVE => -1_i8,
            NNIS_INT2_REFERENCE_CODE_RESERVED => {
                return Err(NnisError::invalid_input(
                    "INT2 reference payload contains reserved code 3",
                ));
            }
            _ => unreachable!("2-bit code is masked"),
        };
        values.push(f32::from(level) * quantized.scale);
    }
    Ok(values)
}

fn validate_quantized_tensor(quantized: &Int2ReferenceQuantizedTensorV1) -> Result<()> {
    if quantized.element_count == 0 {
        return Err(NnisError::invalid_input(
            "INT2 reference tensor has zero logical elements",
        ));
    }
    if !quantized.scale.is_finite() || quantized.scale <= 0.0 {
        return Err(NnisError::invalid_input(
            "INT2 reference tensor scale must be finite and positive",
        ));
    }
    if !quantized.max_abs_error.is_finite()
        || !quantized.mean_squared_error.is_finite()
        || quantized.max_abs_error < 0.0
        || quantized.mean_squared_error < 0.0
    {
        return Err(NnisError::invalid_input(
            "INT2 reference reconstruction evidence is invalid",
        ));
    }

    let element_count = usize::try_from(quantized.element_count)
        .map_err(|_| NnisError::invalid_input("INT2 element count does not fit usize"))?;
    let expected_payload = element_count
        .checked_add(3)
        .ok_or_else(|| NnisError::invalid_input("INT2 payload size overflows usize"))?
        / 4;
    if quantized.packed_values.len() != expected_payload {
        return Err(NnisError::invalid_input(format!(
            "INT2 packed payload has {} bytes; expected {expected_payload}",
            quantized.packed_values.len()
        )));
    }
    for index in 0..element_count {
        let code = (quantized.packed_values[index / 4] >> ((index % 4) * 2)) & 0x03;
        if code == NNIS_INT2_REFERENCE_CODE_RESERVED {
            return Err(NnisError::invalid_input(
                "INT2 reference payload contains reserved code 3",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_int2_known_values_pack_and_reconstruct_deterministically() {
        let source = [-1.0_f32, -0.6, -0.49, 0.0, 0.49, 0.6, 1.0];
        let quantized = quantize_int2_ternary_reference_v1(&source).unwrap();
        assert_eq!(quantized.scale.to_bits(), 1.0_f32.to_bits());
        assert_eq!(quantized.packed_values.len(), 2);
        assert_eq!(quantized.packed_values[0], 10);
        assert_eq!(quantized.packed_values[1], 20);

        let reconstructed = dequantize_int2_ternary_reference_v1(&quantized).unwrap();
        assert_eq!(reconstructed, vec![-1.0_f32, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn zero_tensor_uses_finite_scale_and_roundtrips_exactly() {
        let quantized = quantize_int2_ternary_reference_v1(&[0.0_f32; 9]).unwrap();
        assert_eq!(quantized.scale.to_bits(), 1.0_f32.to_bits());
        assert_eq!(quantized.max_abs_error.to_bits(), 0.0_f32.to_bits());
        assert_eq!(quantized.mean_squared_error, 0.0);
        assert_eq!(
            dequantize_int2_ternary_reference_v1(&quantized).unwrap(),
            vec![0.0_f32; 9]
        );
    }

    #[test]
    fn canonical_serialization_accounts_header_and_payload_exactly() {
        let quantized =
            quantize_int2_ternary_reference_v1(&[1.0_f32, -1.0, 0.0, 0.25, 0.75]).unwrap();
        let serialized = quantized.canonical_serialized_bytes().unwrap();
        assert_eq!(
            serialized.len() as u64,
            NNIS_INT2_REFERENCE_SERIALIZED_HEADER_BYTES + quantized.packed_values.len() as u64
        );
        assert_eq!(&serialized[..4], b"NI21");
    }

    #[test]
    fn invalid_sources_and_reserved_code_fail_closed() {
        assert!(quantize_int2_ternary_reference_v1(&[]).is_err());
        assert!(quantize_int2_ternary_reference_v1(&[f32::NAN]).is_err());
        assert!(quantize_int2_ternary_reference_v1(&[f32::INFINITY]).is_err());

        let mut quantized = quantize_int2_ternary_reference_v1(&[1.0_f32]).unwrap();
        quantized.packed_values[0] = NNIS_INT2_REFERENCE_CODE_RESERVED;
        assert!(dequantize_int2_ternary_reference_v1(&quantized).is_err());
        assert!(quantized.canonical_serialized_bytes().is_err());
    }
}
