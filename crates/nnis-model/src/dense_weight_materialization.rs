//! Exact accounting for compact-weight representations materialized to dense F32.
//!
//! Issue #151 explicitly permits dense execution materialization when NNIS is
//! honest about the boundary. This contract keeps compact representation bytes,
//! dense execution bytes, materialization time and peak scoped device ownership
//! separate and machine-readable. It never labels such a path low-bit compute.

use crate::WeightRepresentationFamilyV1;
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of dense-materialization evidence for fixed weight baselines.
pub const NNIS_DENSE_WEIGHT_MATERIALIZATION_EVIDENCE_VERSION: u32 = 1;

/// Exact device-ownership evidence for a compact representation that executes
/// through a materialized dense-F32 model graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseWeightMaterializationEvidenceV1 {
    pub schema_version: u32,
    pub family: WeightRepresentationFamilyV1,
    pub unique_logical_values: u64,
    pub source_f32_weight_bytes: u64,
    pub representation_resident_bytes: u64,
    pub dense_f32_execution_weight_bytes: u64,
    pub peak_additional_temporary_device_bytes: u64,
    pub final_scoped_owned_device_bytes: u64,
    pub peak_scoped_owned_device_bytes: u64,
    pub materialization_duration_ns: u64,
    pub representation_resident_bits_per_unique_logical_value: f64,
    pub dense_execution_bits_per_unique_logical_value: f64,
    pub final_resident_bits_per_unique_logical_value: f64,
    pub execution_storage: String,
    pub low_bit_compute: bool,
}

impl DenseWeightMaterializationEvidenceV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        family: WeightRepresentationFamilyV1,
        unique_logical_values: u64,
        source_f32_weight_bytes: u64,
        representation_resident_bytes: u64,
        dense_f32_execution_weight_bytes: u64,
        peak_additional_temporary_device_bytes: u64,
        materialization_duration_ns: u64,
    ) -> Result<Self> {
        if unique_logical_values == 0 {
            return Err(NnisError::invalid_input(
                "dense materialization requires at least one unique logical value",
            ));
        }
        if source_f32_weight_bytes == 0
            || representation_resident_bytes == 0
            || dense_f32_execution_weight_bytes == 0
        {
            return Err(NnisError::invalid_input(
                "dense materialization source, representation and execution byte counts must be non-zero",
            ));
        }

        let expected_dense_bytes = unique_logical_values.checked_mul(4).ok_or_else(|| {
            NnisError::invalid_input("dense F32 execution byte count overflows u64")
        })?;
        if source_f32_weight_bytes != expected_dense_bytes {
            return Err(NnisError::invalid_input(format!(
                "source F32 weight bytes {source_f32_weight_bytes} disagree with canonical unique-value denominator {unique_logical_values} (expected {expected_dense_bytes})"
            )));
        }
        if dense_f32_execution_weight_bytes != expected_dense_bytes {
            return Err(NnisError::invalid_input(format!(
                "dense F32 execution bytes {dense_f32_execution_weight_bytes} disagree with canonical unique-value denominator {unique_logical_values} (expected {expected_dense_bytes})"
            )));
        }

        let final_scoped_owned_device_bytes = representation_resident_bytes
            .checked_add(dense_f32_execution_weight_bytes)
            .ok_or_else(|| {
                NnisError::invalid_input("final dense-materialization device bytes overflow u64")
            })?;
        let peak_scoped_owned_device_bytes = source_f32_weight_bytes
            .checked_add(final_scoped_owned_device_bytes)
            .and_then(|value| value.checked_add(peak_additional_temporary_device_bytes))
            .ok_or_else(|| {
                NnisError::invalid_input("peak dense-materialization device bytes overflow u64")
            })?;

        let values = unique_logical_values as f64;
        let representation_resident_bits_per_unique_logical_value =
            representation_resident_bytes as f64 * 8.0 / values;
        let dense_execution_bits_per_unique_logical_value =
            dense_f32_execution_weight_bytes as f64 * 8.0 / values;
        let final_resident_bits_per_unique_logical_value =
            final_scoped_owned_device_bytes as f64 * 8.0 / values;
        for (label, value) in [
            (
                "representation resident bits/value",
                representation_resident_bits_per_unique_logical_value,
            ),
            (
                "dense execution bits/value",
                dense_execution_bits_per_unique_logical_value,
            ),
            (
                "final resident bits/value",
                final_resident_bits_per_unique_logical_value,
            ),
        ] {
            if !value.is_finite() {
                return Err(NnisError::invalid_input(format!(
                    "dense materialization {label} is not finite"
                )));
            }
        }
        if dense_execution_bits_per_unique_logical_value.to_bits() != 32.0_f64.to_bits() {
            return Err(NnisError::invalid_input(
                "dense F32 execution must account exactly 32 bits per unique logical value",
            ));
        }

        let evidence = Self {
            schema_version: NNIS_DENSE_WEIGHT_MATERIALIZATION_EVIDENCE_VERSION,
            family,
            unique_logical_values,
            source_f32_weight_bytes,
            representation_resident_bytes,
            dense_f32_execution_weight_bytes,
            peak_additional_temporary_device_bytes,
            final_scoped_owned_device_bytes,
            peak_scoped_owned_device_bytes,
            materialization_duration_ns,
            representation_resident_bits_per_unique_logical_value,
            dense_execution_bits_per_unique_logical_value,
            final_resident_bits_per_unique_logical_value,
            execution_storage: "dense_f32_materialized".to_string(),
            low_bit_compute: false,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_DENSE_WEIGHT_MATERIALIZATION_EVIDENCE_VERSION {
            return Err(NnisError::unsupported(format!(
                "dense weight materialization evidence schema {}; supported version is {}",
                self.schema_version, NNIS_DENSE_WEIGHT_MATERIALIZATION_EVIDENCE_VERSION
            )));
        }
        if self.execution_storage != "dense_f32_materialized" {
            return Err(NnisError::unsupported(
                "dense materialization evidence has an unsupported execution-storage identity",
            ));
        }
        if self.low_bit_compute {
            return Err(NnisError::invalid_input(
                "dense materialization evidence must never claim low-bit compute",
            ));
        }
        if self.unique_logical_values == 0 {
            return Err(NnisError::invalid_input(
                "dense materialization requires at least one unique logical value",
            ));
        }
        let expected_dense_bytes = self
            .unique_logical_values
            .checked_mul(4)
            .ok_or_else(|| {
                NnisError::invalid_input("dense F32 execution byte count overflows u64")
            })?;
        if self.source_f32_weight_bytes != expected_dense_bytes
            || self.dense_f32_execution_weight_bytes != expected_dense_bytes
        {
            return Err(NnisError::invalid_input(
                "dense materialization F32 byte counts disagree with the canonical denominator",
            ));
        }
        if self.representation_resident_bytes == 0 {
            return Err(NnisError::invalid_input(
                "dense materialization representation resident bytes must be non-zero",
            ));
        }

        let expected_final = self
            .representation_resident_bytes
            .checked_add(self.dense_f32_execution_weight_bytes)
            .ok_or_else(|| {
                NnisError::invalid_input("final dense-materialization device bytes overflow u64")
            })?;
        let expected_peak = self
            .source_f32_weight_bytes
            .checked_add(expected_final)
            .and_then(|value| value.checked_add(self.peak_additional_temporary_device_bytes))
            .ok_or_else(|| {
                NnisError::invalid_input("peak dense-materialization device bytes overflow u64")
            })?;
        let values = self.unique_logical_values as f64;
        let expected_representation_bits =
            self.representation_resident_bytes as f64 * 8.0 / values;
        let expected_dense_bits = self.dense_f32_execution_weight_bytes as f64 * 8.0 / values;
        let expected_final_bits = expected_final as f64 * 8.0 / values;
        if !expected_representation_bits.is_finite()
            || !expected_dense_bits.is_finite()
            || !expected_final_bits.is_finite()
            || expected_dense_bits.to_bits() != 32.0_f64.to_bits()
            || self.final_scoped_owned_device_bytes != expected_final
            || self.peak_scoped_owned_device_bytes != expected_peak
            || self
                .representation_resident_bits_per_unique_logical_value
                .to_bits()
                != expected_representation_bits.to_bits()
            || self.dense_execution_bits_per_unique_logical_value.to_bits()
                != expected_dense_bits.to_bits()
            || self.final_resident_bits_per_unique_logical_value.to_bits()
                != expected_final_bits.to_bits()
        {
            return Err(NnisError::invalid_input(
                "dense materialization cached accounting disagrees with exact integer byte totals",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_materialization_counts_compact_and_dense_residence() {
        let evidence = DenseWeightMaterializationEvidenceV1::new(
            WeightRepresentationFamilyV1::Int4Symmetric,
            100,
            400,
            60,
            400,
            32,
            123_456,
        )
        .unwrap();
        assert_eq!(evidence.final_scoped_owned_device_bytes, 460);
        assert_eq!(evidence.peak_scoped_owned_device_bytes, 892);
        assert_eq!(
            evidence
                .dense_execution_bits_per_unique_logical_value
                .to_bits(),
            32.0_f64.to_bits()
        );
        assert_eq!(
            evidence
                .final_resident_bits_per_unique_logical_value
                .to_bits(),
            36.8_f64.to_bits()
        );
        assert!(!evidence.low_bit_compute);
        evidence.validate().unwrap();
    }

    #[test]
    fn denominator_and_cached_accounting_fail_closed() {
        assert!(DenseWeightMaterializationEvidenceV1::new(
            WeightRepresentationFamilyV1::Int2Ternary,
            100,
            399,
            30,
            400,
            0,
            1,
        )
        .is_err());
        assert!(DenseWeightMaterializationEvidenceV1::new(
            WeightRepresentationFamilyV1::Int2Ternary,
            100,
            400,
            30,
            396,
            0,
            1,
        )
        .is_err());

        let mut evidence = DenseWeightMaterializationEvidenceV1::new(
            WeightRepresentationFamilyV1::MagnitudeSparse,
            8,
            32,
            10,
            32,
            0,
            1,
        )
        .unwrap();
        evidence.low_bit_compute = true;
        assert!(evidence.validate().is_err());
    }

    #[test]
    fn integer_overflow_fails_closed() {
        assert!(DenseWeightMaterializationEvidenceV1::new(
            WeightRepresentationFamilyV1::Int4Symmetric,
            u64::MAX,
            u64::MAX,
            1,
            u64::MAX,
            1,
            0,
        )
        .is_err());
    }
}
