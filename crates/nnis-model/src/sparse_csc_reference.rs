//! Deterministic sparse CSC reference representation for projection matrices.
//!
//! Dense source matrices are row-major `[K,N]`. The canonical sparse payload
//! is column-compressed so the CUDA projection primitive can consume retained
//! weights without scanning pruned entries. Rows inside each column are stored
//! in strictly increasing order.

use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of the sparse CSC reference contract.
pub const NNIS_SPARSE_CSC_REFERENCE_VERSION: u32 = 1;
/// Canonical fixed header: magic + rows + cols + threshold + NNZ.
pub const NNIS_SPARSE_CSC_SERIALIZED_HEADER_BYTES: u64 = 32;

const SERIALIZED_MAGIC: [u8; 4] = *b"NSC1";

/// Host-side deterministic CSC representation of one row-major F32 matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseCscReferenceMatrixV1 {
    pub rows: u64,
    pub cols: u64,
    pub threshold: f32,
    pub column_offsets: Vec<u64>,
    pub row_indices: Vec<u32>,
    pub values: Vec<f32>,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
}

impl SparseCscReferenceMatrixV1 {
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Canonical little-endian serialization consumed by evidence tooling.
    pub fn canonical_serialized_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let nnz = u64::try_from(self.values.len())
            .map_err(|_| NnisError::invalid_input("sparse CSC NNZ exceeds u64"))?;
        let offsets_bytes = self
            .column_offsets
            .len()
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| NnisError::invalid_input("sparse CSC offset bytes overflow usize"))?;
        let indices_bytes = self
            .row_indices
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| NnisError::invalid_input("sparse CSC index bytes overflow usize"))?;
        let values_bytes = self
            .values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| NnisError::invalid_input("sparse CSC value bytes overflow usize"))?;
        let header = usize::try_from(NNIS_SPARSE_CSC_SERIALIZED_HEADER_BYTES)
            .map_err(|_| NnisError::invalid_input("sparse CSC header does not fit usize"))?;
        let capacity = header
            .checked_add(offsets_bytes)
            .and_then(|value| value.checked_add(indices_bytes))
            .and_then(|value| value.checked_add(values_bytes))
            .ok_or_else(|| {
                NnisError::invalid_input("sparse CSC serialized size overflows usize")
            })?;

        let mut encoded = Vec::with_capacity(capacity);
        encoded.extend_from_slice(&SERIALIZED_MAGIC);
        encoded.extend_from_slice(&self.rows.to_le_bytes());
        encoded.extend_from_slice(&self.cols.to_le_bytes());
        encoded.extend_from_slice(&self.threshold.to_bits().to_le_bytes());
        encoded.extend_from_slice(&nnz.to_le_bytes());
        for offset in &self.column_offsets {
            encoded.extend_from_slice(&offset.to_le_bytes());
        }
        for row in &self.row_indices {
            encoded.extend_from_slice(&row.to_le_bytes());
        }
        for value in &self.values {
            encoded.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        Ok(encoded)
    }

    /// Validate canonical CSC shape/order and finite evidence.
    pub fn validate(&self) -> Result<()> {
        if self.rows == 0 || self.cols == 0 {
            return Err(NnisError::invalid_input(
                "sparse CSC matrix dimensions must be non-zero",
            ));
        }
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            return Err(NnisError::invalid_input(
                "sparse CSC threshold must be finite and non-negative",
            ));
        }
        if !self.max_abs_error.is_finite()
            || !self.mean_squared_error.is_finite()
            || self.max_abs_error < 0.0
            || self.mean_squared_error < 0.0
        {
            return Err(NnisError::invalid_input(
                "sparse CSC reconstruction evidence is invalid",
            ));
        }
        let cols = usize::try_from(self.cols)
            .map_err(|_| NnisError::invalid_input("sparse CSC cols do not fit usize"))?;
        let expected_offsets = cols
            .checked_add(1)
            .ok_or_else(|| NnisError::invalid_input("sparse CSC offset count overflows usize"))?;
        if self.column_offsets.len() != expected_offsets {
            return Err(NnisError::invalid_input(format!(
                "sparse CSC has {} column offsets; expected {expected_offsets}",
                self.column_offsets.len()
            )));
        }
        if self.row_indices.len() != self.values.len() {
            return Err(NnisError::invalid_input(
                "sparse CSC row-index and value lengths disagree",
            ));
        }
        if self.column_offsets.first().copied() != Some(0) {
            return Err(NnisError::invalid_input(
                "sparse CSC first column offset must be zero",
            ));
        }
        let nnz = u64::try_from(self.values.len())
            .map_err(|_| NnisError::invalid_input("sparse CSC NNZ exceeds u64"))?;
        if self.column_offsets.last().copied() != Some(nnz) {
            return Err(NnisError::invalid_input(
                "sparse CSC final column offset must equal NNZ",
            ));
        }
        let rows = u32::try_from(self.rows)
            .map_err(|_| NnisError::unsupported("sparse CSC row count exceeds u32 index domain"))?;

        for col in 0..cols {
            let begin = usize::try_from(self.column_offsets[col])
                .map_err(|_| NnisError::invalid_input("sparse CSC begin offset exceeds usize"))?;
            let end = usize::try_from(self.column_offsets[col + 1])
                .map_err(|_| NnisError::invalid_input("sparse CSC end offset exceeds usize"))?;
            if begin > end || end > self.values.len() {
                return Err(NnisError::invalid_input(
                    "sparse CSC column offsets are not monotonic and bounded",
                ));
            }
            let mut previous = None;
            for index in begin..end {
                let row = self.row_indices[index];
                if row >= rows {
                    return Err(NnisError::invalid_input(
                        "sparse CSC row index exceeds matrix row count",
                    ));
                }
                if previous.is_some_and(|value| row <= value) {
                    return Err(NnisError::invalid_input(
                        "sparse CSC rows must be strictly increasing within each column",
                    ));
                }
                if !self.values[index].is_finite() {
                    return Err(NnisError::invalid_input(
                        "sparse CSC retained values must be finite",
                    ));
                }
                previous = Some(row);
            }
        }
        Ok(())
    }
}

/// Convert one row-major dense F32 matrix to canonical magnitude-pruned CSC.
pub fn sparsify_matrix_csc_reference_v1(
    dense: &[f32],
    rows: usize,
    cols: usize,
    threshold: f32,
) -> Result<SparseCscReferenceMatrixV1> {
    if rows == 0 || cols == 0 {
        return Err(NnisError::invalid_input(
            "sparse CSC source dimensions must be non-zero",
        ));
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return Err(NnisError::invalid_input(
            "sparse CSC threshold must be finite and non-negative",
        ));
    }
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| NnisError::invalid_input("sparse CSC source shape overflows usize"))?;
    if dense.len() != elements {
        return Err(NnisError::invalid_input(format!(
            "sparse CSC source has {} values; shape ({rows}, {cols}) requires {elements}",
            dense.len()
        )));
    }
    let rows_u64 =
        u64::try_from(rows).map_err(|_| NnisError::invalid_input("sparse CSC rows exceed u64"))?;
    let cols_u64 =
        u64::try_from(cols).map_err(|_| NnisError::invalid_input("sparse CSC cols exceed u64"))?;
    u32::try_from(rows)
        .map_err(|_| NnisError::unsupported("sparse CSC rows exceed u32 index domain"))?;

    let mut column_offsets = Vec::with_capacity(cols + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    let mut max_abs_error = 0.0_f32;
    let mut squared_error_sum = 0.0_f64;
    column_offsets.push(0);

    for col in 0..cols {
        for row in 0..rows {
            let value = dense[row * cols + col];
            if !value.is_finite() {
                return Err(NnisError::invalid_input(format!(
                    "sparse CSC source value ({row}, {col}) is not finite"
                )));
            }
            let reconstructed = if value.abs() > threshold {
                row_indices.push(u32::try_from(row).map_err(|_| {
                    NnisError::unsupported("sparse CSC row exceeds u32 index domain")
                })?);
                values.push(value);
                value
            } else {
                0.0
            };
            let error = (value - reconstructed).abs();
            max_abs_error = max_abs_error.max(error);
            let error_f64 = f64::from(value) - f64::from(reconstructed);
            squared_error_sum += error_f64 * error_f64;
        }
        column_offsets.push(
            u64::try_from(values.len())
                .map_err(|_| NnisError::invalid_input("sparse CSC NNZ exceeds u64"))?,
        );
    }

    let matrix = SparseCscReferenceMatrixV1 {
        rows: rows_u64,
        cols: cols_u64,
        threshold,
        column_offsets,
        row_indices,
        values,
        max_abs_error,
        mean_squared_error: squared_error_sum / elements as f64,
    };
    matrix.validate()?;
    Ok(matrix)
}

/// Reconstruct the row-major dense logical matrix.
pub fn densify_matrix_csc_reference_v1(sparse: &SparseCscReferenceMatrixV1) -> Result<Vec<f32>> {
    sparse.validate()?;
    let rows = usize::try_from(sparse.rows)
        .map_err(|_| NnisError::invalid_input("sparse CSC rows do not fit usize"))?;
    let cols = usize::try_from(sparse.cols)
        .map_err(|_| NnisError::invalid_input("sparse CSC cols do not fit usize"))?;
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| NnisError::invalid_input("sparse CSC dense shape overflows usize"))?;
    let mut dense = vec![0.0_f32; elements];
    for col in 0..cols {
        let begin = usize::try_from(sparse.column_offsets[col])
            .map_err(|_| NnisError::invalid_input("sparse CSC begin offset exceeds usize"))?;
        let end = usize::try_from(sparse.column_offsets[col + 1])
            .map_err(|_| NnisError::invalid_input("sparse CSC end offset exceeds usize"))?;
        for index in begin..end {
            let row = sparse.row_indices[index] as usize;
            dense[row * cols + col] = sparse.values[index];
        }
    }
    Ok(dense)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_dense_matrix_converts_to_canonical_csc() {
        let dense = [1.0_f32, 0.1, -2.0, 0.0, 3.0, 0.2, -4.0, 0.0, 5.0];
        let sparse = sparsify_matrix_csc_reference_v1(&dense, 3, 3, 0.25).unwrap();
        assert_eq!(sparse.column_offsets, vec![0, 2, 3, 5]);
        assert_eq!(sparse.row_indices, vec![0, 2, 1, 0, 2]);
        assert_eq!(sparse.values, vec![1.0_f32, -4.0, 3.0, -2.0, 5.0]);
        assert_eq!(
            densify_matrix_csc_reference_v1(&sparse).unwrap(),
            vec![1.0_f32, 0.0, -2.0, 0.0, 3.0, 0.0, -4.0, 0.0, 5.0]
        );
    }

    #[test]
    fn canonical_serialization_accounts_every_owned_array() {
        let dense = [1.0_f32, 0.0, -2.0, 3.0, 0.0, 4.0];
        let sparse = sparsify_matrix_csc_reference_v1(&dense, 2, 3, 0.0).unwrap();
        let encoded = sparse.canonical_serialized_bytes().unwrap();
        let expected = NNIS_SPARSE_CSC_SERIALIZED_HEADER_BYTES
            + sparse.column_offsets.len() as u64 * 8
            + sparse.row_indices.len() as u64 * 4
            + sparse.values.len() as u64 * 4;
        assert_eq!(encoded.len() as u64, expected);
        assert_eq!(&encoded[..4], b"NSC1");
    }

    #[test]
    fn malformed_csc_contract_fails_closed() {
        assert!(sparsify_matrix_csc_reference_v1(&[], 0, 1, 0.0).is_err());
        assert!(sparsify_matrix_csc_reference_v1(&[1.0], 1, 1, -0.1).is_err());
        assert!(sparsify_matrix_csc_reference_v1(&[f32::NAN], 1, 1, 0.0).is_err());
        assert!(sparsify_matrix_csc_reference_v1(&[1.0, 2.0], 1, 1, 0.0).is_err());

        let mut sparse =
            sparsify_matrix_csc_reference_v1(&[1.0_f32, 2.0, 3.0, 4.0], 2, 2, 0.0).unwrap();
        sparse.row_indices[1] = sparse.row_indices[0];
        assert!(sparse.validate().is_err());
    }
}
