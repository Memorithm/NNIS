//! Evidence required to promote a fixed weight representation to FullModel.
//!
//! A full-model claim is stronger than "the model object constructed". This
//! contract binds one exact NNIS commit and checkpoint to a concrete runtime
//! entrypoint, physical execution, verified generated-token output, dense
//! materialization accounting, and the final resident weight footprint.

use crate::{
    CanonicalWeightDenominatorV1, DenseWeightMaterializationEvidenceV2,
    WeightExecutionQualificationLevelV1, WeightRepresentationAccountingV1,
    WeightRepresentationFamilyV1, WeightRepresentationQualificationRecordV1,
    NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION,
};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of full-model weight execution evidence.
pub const NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION: u32 = 1;

/// Physical full-model execution evidence for one fixed representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightFullModelExecutionEvidenceV1 {
    pub schema_version: u32,
    pub family: WeightRepresentationFamilyV1,
    pub representation_version: u32,
    pub exact_checkpoint: String,
    pub nnis_commit: String,
    pub runtime_entrypoint: String,
    pub physical_execution_observed: bool,
    pub generated_token_count: u64,
    pub generated_token_ids_sha256: String,
    pub non_finite_output_observed: bool,
    pub serialized_representation_bytes: u64,
    pub logical_tensor_references: u64,
    pub logical_element_references: u64,
    pub unique_source_allocations: u64,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
    pub materialization: DenseWeightMaterializationEvidenceV2,
}

impl WeightFullModelExecutionEvidenceV1 {
    /// Validate every claim-bearing field before a FullModel record is emitted.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight full-model evidence schema {}; supported version is {}",
                self.schema_version, NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION
            )));
        }
        if self.representation_version == 0 {
            return Err(NnisError::invalid_input(
                "full-model representation version must be non-zero",
            ));
        }
        validate_trimmed("exact checkpoint", &self.exact_checkpoint)?;
        validate_trimmed("runtime entrypoint", &self.runtime_entrypoint)?;
        validate_lower_hex("NNIS commit", &self.nnis_commit, 40)?;
        validate_lower_hex(
            "generated token ids sha256",
            &self.generated_token_ids_sha256,
            64,
        )?;
        if !self.physical_execution_observed {
            return Err(NnisError::invalid_input(
                "full-model evidence requires physical execution",
            ));
        }
        if self.generated_token_count == 0 {
            return Err(NnisError::invalid_input(
                "full-model evidence requires at least one verified generated token",
            ));
        }
        if self.non_finite_output_observed {
            return Err(NnisError::invalid_input(
                "full-model evidence rejects non-finite runtime output",
            ));
        }
        if self.serialized_representation_bytes == 0
            || self.logical_tensor_references == 0
            || self.logical_element_references == 0
            || self.unique_source_allocations == 0
        {
            return Err(NnisError::invalid_input(
                "full-model evidence contains a zero required accounting count",
            ));
        }
        if !self.max_abs_error.is_finite()
            || !self.mean_squared_error.is_finite()
            || self.max_abs_error < 0.0
            || self.mean_squared_error < 0.0
        {
            return Err(NnisError::invalid_input(
                "full-model reconstruction metrics are invalid",
            ));
        }

        self.materialization.validate()?;
        let device = &self.materialization.device_ownership;
        if device.family != self.family {
            return Err(NnisError::invalid_input(
                "full-model family disagrees with materialization family",
            ));
        }
        if device.low_bit_compute {
            return Err(NnisError::invalid_input(
                "dense-materialized full-model evidence cannot claim low-bit compute",
            ));
        }
        if device.execution_storage != "dense_f32_materialized" {
            return Err(NnisError::unsupported(
                "full-model evidence requires the registered dense-F32 materialization boundary",
            ));
        }
        if self.logical_element_references < device.unique_logical_values {
            return Err(NnisError::invalid_input(
                "logical element references cannot be smaller than unique logical values",
            ));
        }
        Ok(())
    }

    /// Build the canonical FullModel qualification record.
    ///
    /// Serialized bytes remain the compact representation. Resident bytes are
    /// the final compact + dense-F32 execution graph owned after materialization.
    pub fn qualification_record(&self) -> Result<WeightRepresentationQualificationRecordV1> {
        self.validate()?;
        let device = &self.materialization.device_ownership;
        let accounting = WeightRepresentationAccountingV1::new(
            CanonicalWeightDenominatorV1 {
                schema_version: NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION,
                logical_tensor_references: self.logical_tensor_references,
                logical_element_references: self.logical_element_references,
                unique_source_allocations: self.unique_source_allocations,
                unique_logical_values: device.unique_logical_values,
                source_owned_bytes: device.source_f32_weight_bytes,
            },
            self.serialized_representation_bytes,
            device.final_scoped_owned_device_bytes,
        )?;
        WeightRepresentationQualificationRecordV1::new(
            self.family,
            self.representation_version,
            self.exact_checkpoint.clone(),
            WeightExecutionQualificationLevelV1::FullModel,
            true,
            true,
            self.max_abs_error,
            self.mean_squared_error,
            accounting,
        )
    }
}

fn validate_trimmed(label: &str, value: &str) -> Result<()> {
    if value.is_empty() || value.trim() != value {
        return Err(NnisError::invalid_input(format!(
            "{label} must be non-empty and trimmed"
        )));
    }
    Ok(())
}

fn validate_lower_hex(label: &str, value: &str, expected_len: usize) -> Result<()> {
    if value.len() != expected_len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NnisError::invalid_input(format!(
            "{label} must contain exactly {expected_len} lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DenseWeightMaterializationEvidenceV1, NNIS_DENSE_WEIGHT_MATERIALIZATION_EVIDENCE_V2_VERSION,
    };

    fn evidence() -> WeightFullModelExecutionEvidenceV1 {
        let device = DenseWeightMaterializationEvidenceV1::new(
            WeightRepresentationFamilyV1::Int4Symmetric,
            16,
            64,
            12,
            64,
            0,
            1_000,
        )
        .unwrap();
        let materialization = DenseWeightMaterializationEvidenceV2::new(device, 80).unwrap();
        assert_eq!(
            materialization.schema_version,
            NNIS_DENSE_WEIGHT_MATERIALIZATION_EVIDENCE_V2_VERSION
        );
        WeightFullModelExecutionEvidenceV1 {
            schema_version: NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION,
            family: WeightRepresentationFamilyV1::Int4Symmetric,
            representation_version: 1,
            exact_checkpoint: "smollm2-135m-bf16@sha256:test".to_string(),
            nnis_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            runtime_entrypoint: "Int4DenseMaterializedModelV1::model".to_string(),
            physical_execution_observed: true,
            generated_token_count: 4,
            generated_token_ids_sha256:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            non_finite_output_observed: false,
            serialized_representation_bytes: 24,
            logical_tensor_references: 2,
            logical_element_references: 32,
            unique_source_allocations: 1,
            max_abs_error: 0.25,
            mean_squared_error: 0.0625,
            materialization,
        }
    }

    #[test]
    fn full_model_evidence_builds_resident_accounting_from_materialized_graph() {
        let evidence = evidence();
        evidence.validate().unwrap();
        let record = evidence.qualification_record().unwrap();
        assert_eq!(
            record.execution_level,
            WeightExecutionQualificationLevelV1::FullModel
        );
        assert!(record.physical_execution_observed);
        assert!(record.generated_tokens_verified);
        assert_eq!(record.accounting.serialized_bytes, 24);
        assert_eq!(record.accounting.resident_bytes, 76);
        assert_eq!(
            record
                .accounting
                .resident_bits_per_unique_logical_value
                .to_bits(),
            38.0_f64.to_bits()
        );
    }

    #[test]
    fn evidence_rejects_unverified_execution_tokens_non_finite_and_identity_drift() {
        let mut value = evidence();
        value.physical_execution_observed = false;
        assert!(value.validate().is_err());

        let mut value = evidence();
        value.generated_token_count = 0;
        assert!(value.validate().is_err());

        let mut value = evidence();
        value.non_finite_output_observed = true;
        assert!(value.validate().is_err());

        let mut value = evidence();
        value.nnis_commit = "ABC".to_string();
        assert!(value.validate().is_err());

        let mut value = evidence();
        value.generated_token_ids_sha256 = "xyz".to_string();
        assert!(value.validate().is_err());
    }

    #[test]
    fn evidence_rejects_family_or_accounting_mismatch() {
        let mut value = evidence();
        value.family = WeightRepresentationFamilyV1::Int2Ternary;
        assert!(value.validate().is_err());

        let mut value = evidence();
        value.logical_element_references = 8;
        assert!(value.validate().is_err());
    }
}
