//! Deterministic magnitude-threshold sparse reference codec for NNIS.
//!
//! The representation stores one occupancy bit per logical F32 value plus the
//! retained F32 values in logical order. Values whose absolute magnitude is
//! less than or equal to the configured threshold reconstruct as exact zero.
//! This is a structural research baseline with exact accounting; it does not
//! imply that a dense checkpoint compresses beneficially.

use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of the sparse reference storage contract.
pub const NNIS_SPARSE_REFERENCE_STORAGE_VERSION: u32 = 1;
/// Canonical header: magic + element count + threshold + retained count.
pub const NNIS_SPARSE_REFERENCE_SERIALIZED_HEADER_BYTES: u64 = 24;

const SERIALIZED_MAGIC: [u8; 4] = *b"NSP1";

/// Host-side sparse representation for one logical tensor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseReferenceTensorV1 {
    pub element_count: u64,
    pub threshold: f32,
    pub occupancy_bitmap: Vec<u8>,
    pub retained_values: Vec<f32>,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
}

impl SparseReferenceTensorV1 {
    /// Canonical little-endian serialization.
    pub fn canonical_serialized_bytes(&self) -> Result<Vec<u8>> {
        validate_sparse_tensor(self)?;
        let retained_count = u64::try_from(self.retained_values.len())
            .map_err(|_| NnisError::invalid_input("sparse retained count exceeds u64"))?;
        let value_bytes = self
            .retained_values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| NnisError::invalid_input("sparse value bytes overflow usize"))?;
        let header_len = usize::try_from(NNIS_SPARSE_REFERENCE_SERIALIZED_HEADER_BYTES)
            .map_err(|_| NnisError::invalid_input("sparse header does not fit usize"))?;
        let capacity = header_len
            .checked_add(self.occupancy_bitmap.len())
            .and_then(|value| value.checked_add(value_bytes))
            .ok_or_else(|| NnisError::invalid_input("sparse serialized size overflows usize"))?;
        let mut encoded = Vec::with_capacity(capacity);
        encoded.extend_from_slice(&SERIALIZED_MAGIC);
        encoded.extend_from_slice(&self.element_count.to_le_bytes());
        encoded.extend_from_slice(&self.threshold.to_bits().to_le_bytes());
        encoded.extend_from_slice(&retained_count.to_le_bytes());
        encoded.extend_from_slice(&self.occupancy_bitmap);
        for value in &self.retained_values {
            encoded.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        Ok(encoded)
    }

    #[must_use]
    pub fn retained_count(&self) -> usize {
        self.retained_values.len()
    }
}

/// Build a deterministic sparse tensor by pruning values with
/// `abs(value) <= threshold`.
pub fn sparsify_magnitude_reference_v1(
    values: &[f32],
    threshold: f32,
) -> Result<SparseReferenceTensorV1> {
    if values.is_empty() {
        return Err(NnisError::invalid_input(
            "sparse reference requires at least one value",
        ));
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(NnisError::invalid_input(
            "sparse threshold must be finite and non-negative",
        ));
    }

    let bitmap_len = values
        .len()
        .checked_add(7)
        .ok_or_else(|| NnisError::invalid_input("sparse element count overflows usize"))?
        / 8;
    let mut occupancy_bitmap = vec![0_u8; bitmap_len];
    let mut retained_values = Vec::new();
    let mut max_abs_error = 0.0_f32;
    let mut squared_error_sum = 0.0_f64;

    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(NnisError::invalid_input(format!(
                "sparse source value {index} is not finite"
            )));
        }
        let reconstructed = if value.abs() > threshold {
            occupancy_bitmap[index / 8] |= 1_u8 << (index % 8);
            retained_values.push(value);
            value
        } else {
            0.0
        };
        let error = (value - reconstructed).abs();
        max_abs_error = max_abs_error.max(error);
        let error_f64 = f64::from(value) - f64::from(reconstructed);
        squared_error_sum += error_f64 * error_f64;
    }

    let tensor = SparseReferenceTensorV1 {
        element_count: u64::try_from(values.len())
            .map_err(|_| NnisError::invalid_input("sparse element count exceeds u64"))?,
        threshold,
        occupancy_bitmap,
        retained_values,
        max_abs_error,
        mean_squared_error: squared_error_sum / values.len() as f64,
    };
    validate_sparse_tensor(&tensor)?;
    Ok(tensor)
}

/// Reconstruct the dense F32 logical tensor.
pub fn densify_sparse_reference_v1(sparse: &SparseReferenceTensorV1) -> Result<Vec<f32>> {
    validate_sparse_tensor(sparse)?;
    let element_count = usize::try_from(sparse.element_count)
        .map_err(|_| NnisError::invalid_input("sparse element count does not fit usize"))?;
    let mut retained_index = 0_usize;
    let mut output = Vec::with_capacity(element_count);
    for index in 0..element_count {
        let occupied = sparse.occupancy_bitmap[index / 8] & (1_u8 << (index % 8)) != 0;
        if occupied {
            let value = sparse.retained_values.get(retained_index).ok_or_else(|| {
                NnisError::invalid_input("sparse bitmap consumes more values than retained")
            })?;
            output.push(*value);
            retained_index += 1;
        } else {
            output.push(0.0);
        }
    }
    if retained_index != sparse.retained_values.len() {
        return Err(NnisError::invalid_input(
            "sparse retained values exceed bitmap occupancy",
        ));
    }
    Ok(output)
}

fn validate_sparse_tensor(sparse: &SparseReferenceTensorV1) -> Result<()> {
    if sparse.element_count == 0 {
        return Err(NnisError::invalid_input(
            "sparse reference tensor has zero logical elements",
        ));
    }
    if !sparse.threshold.is_finite() || sparse.threshold < 0.0 {
        return Err(NnisError::invalid_input(
            "sparse threshold must be finite and non-negative",
        ));
    }
    if !sparse.max_abs_error.is_finite()
        || !sparse.mean_squared_error.is_finite()
        || sparse.max_abs_error < 0.0
        || sparse.mean_squared_error < 0.0
    {
        return Err(NnisError::invalid_input(
            "sparse reconstruction evidence is invalid",
        ));
    }
    if sparse.retained_values.iter().any(|value| !value.is_finite()) {
        return Err(NnisError::invalid_input(
            "sparse retained values must all be finite",
        ));
    }

    let element_count = usize::try_from(sparse.element_count)
        .map_err(|_| NnisError::invalid_input("sparse element count does not fit usize"))?;
    let expected_bitmap = element_count
        .checked_add(7)
        .ok_or_else(|| NnisError::invalid_input("sparse bitmap size overflows usize"))?
        / 8;
    if sparse.occupancy_bitmap.len() != expected_bitmap {
        return Err(NnisError::invalid_input(format!(
            "sparse bitmap has {} bytes; expected {expected_bitmap}",
            sparse.occupancy_bitmap.len()
        )));
    }

    let occupied = (0..element_count)
        .filter(|&index| sparse.occupancy_bitmap[index / 8] & (1_u8 << (index % 8)) != 0)
        .count();
    if occupied != sparse.retained_values.len() {
        return Err(NnisError::invalid_input(format!(
            "sparse bitmap contains {occupied} occupied values but {} values are retained",
            sparse.retained_values.len()
        )));
    }
    if element_count % 8 != 0 {
        let used_bits = element_count % 8;
        let padding_mask = !((1_u8 << used_bits) - 1);
        if sparse.occupancy_bitmap[expected_bitmap - 1] & padding_mask != 0 {
            return Err(NnisError::invalid_input(
                "sparse bitmap padding bits must be zero",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnitude_sparse_reference_reconstructs_in_logical_order() {
        let source = [-1.0_f32, -0.25, 0.0, 0.5, 1.5, -0.75, 0.1, 2.0, -3.0];
        let sparse = sparsify_magnitude_reference_v1(&source, 0.5).unwrap();
        assert_eq!(sparse.occupancy_bitmap, vec![0b1011_0001, 0b0000_0001]);
        assert_eq!(sparse.retained_values, vec![-1.0_f32, 1.5, -0.75, 2.0, -3.0]);
        assert_eq!(
            densify_sparse_reference_v1(&sparse).unwrap(),
            vec![-1.0_f32, 0.0, 0.0, 0.0, 1.5, -0.75, 0.0, 2.0, -3.0]
        );
    }

    #[test]
    fn canonical_serialization_accounts_bitmap_and_values_exactly() {
        let sparse = sparsify_magnitude_reference_v1(&[1.0_f32, 0.0, -2.0, 0.1], 0.25)
            .unwrap();
        let serialized = sparse.canonical_serialized_bytes().unwrap();
        let expected = NNIS_SPARSE_REFERENCE_SERIALIZED_HEADER_BYTES
            + sparse.occupancy_bitmap.len() as u64
            + sparse.retained_values.len() as u64 * 4;
        assert_eq!(serialized.len() as u64, expected);
        assert_eq!(&serialized[..4], b"NSP1");
    }

    #[test]
    fn zero_threshold_is_lossless_for_exact_zeros() {
        let source = [0.0_f32, 1.25, -2.5, 0.0];
        let sparse = sparsify_magnitude_reference_v1(&source, 0.0).unwrap();
        assert_eq!(densify_sparse_reference_v1(&sparse).unwrap(), source);
        assert_eq!(sparse.max_abs_error.to_bits(), 0.0_f32.to_bits());
        assert_eq!(sparse.mean_squared_error, 0.0);
    }

    #[test]
    fn invalid_sources_and_malformed_bitmap_fail_closed() {
        assert!(sparsify_magnitude_reference_v1(&[], 0.0).is_err());
        assert!(sparsify_magnitude_reference_v1(&[1.0], -1.0).is_err());
        assert!(sparsify_magnitude_reference_v1(&[f32::NAN], 0.0).is_err());

        let mut sparse = sparsify_magnitude_reference_v1(&[1.0_f32; 9], 0.0).unwrap();
        sparse.occupancy_bitmap[1] |= 0b1000_0000;
        assert!(densify_sparse_reference_v1(&sparse).is_err());
        assert!(sparse.canonical_serialized_bytes().is_err());
    }
}
