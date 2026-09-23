//! Evidence-bound qualified capability record for fixed full-model weight families.
//!
//! Unlike the conservative static capability manifest, this record can state
//! full-model readiness only when it is derived from a fully validated
//! WeightFullModelCampaignArtifactV1.

use crate::{
    WeightCampaignRecipeV1, WeightFullModelCampaignArtifactV1, WeightRepresentationFamilyV1,
};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Version of the evidence-bound qualified weight capability record.
pub const NNIS_QUALIFIED_WEIGHT_CAPABILITY_RECORD_VERSION: u32 = 1;

/// One physically qualified full-model weight family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedWeightFamilyCapabilityV1 {
    pub family: WeightRepresentationFamilyV1,
    pub representation_version: u32,
    pub runtime_entrypoint: String,
    pub unique_logical_values: u64,
    pub serialized_bits_per_unique_logical_value: f64,
    pub resident_bits_per_unique_logical_value: f64,
    pub materialization_duration_ns: u64,
    pub peak_scoped_owned_device_bytes: u64,
    pub peak_host_temporary_payload_bytes: u64,
    pub execution_storage: String,
    pub low_bit_compute: bool,
    pub generated_token_count: u64,
    pub generated_token_ids_sha256: String,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
}

impl QualifiedWeightFamilyCapabilityV1 {
    fn validate(&self) -> Result<()> {
        if self.representation_version == 0
            || self.unique_logical_values == 0
            || self.peak_scoped_owned_device_bytes == 0
            || self.generated_token_count == 0
        {
            return Err(NnisError::invalid_input(
                "qualified weight family contains a zero required count",
            ));
        }
        if self.runtime_entrypoint.is_empty()
            || self.runtime_entrypoint.trim() != self.runtime_entrypoint
        {
            return Err(NnisError::invalid_input(
                "qualified weight runtime entrypoint must be non-empty and trimmed",
            ));
        }
        if self.execution_storage != "dense_f32_materialized" {
            return Err(NnisError::unsupported(
                "qualified weight family has an unsupported execution-storage identity",
            ));
        }
        if self.low_bit_compute {
            return Err(NnisError::invalid_input(
                "dense-materialized qualified weight family cannot claim low-bit compute",
            ));
        }
        if self.generated_token_ids_sha256.len() != 64
            || !self
                .generated_token_ids_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(NnisError::invalid_input(
                "qualified weight token hash must be 64 lowercase hexadecimal characters",
            ));
        }
        for (label, value) in [
            (
                "serialized bits/value",
                self.serialized_bits_per_unique_logical_value,
            ),
            (
                "resident bits/value",
                self.resident_bits_per_unique_logical_value,
            ),
            ("mean squared error", self.mean_squared_error),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(NnisError::invalid_input(format!(
                    "qualified weight {label} is invalid"
                )));
            }
        }
        if !self.max_abs_error.is_finite() || self.max_abs_error < 0.0 {
            return Err(NnisError::invalid_input(
                "qualified weight maximum absolute error is invalid",
            ));
        }
        Ok(())
    }
}

/// Same-commit qualified capability record suitable for downstream
/// preregistration updates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedWeightCapabilityRecordV1 {
    pub schema_version: u32,
    pub nnis_commit: String,
    pub exact_checkpoint: String,
    pub tokenizer_file: String,
    pub tokenizer_sha256: String,
    pub recipe: WeightCampaignRecipeV1,
    pub families: Vec<QualifiedWeightFamilyCapabilityV1>,
    pub elastic_stage_b_backend_ready: bool,
}

impl QualifiedWeightCapabilityRecordV1 {
    /// Derive a qualified capability record from one already validated campaign artifact.
    pub fn from_artifact(artifact: &WeightFullModelCampaignArtifactV1) -> Result<Self> {
        artifact.validate()?;
        let bundle = artifact.campaign.qualification_bundle()?;
        bundle.validate_elastic_stage_b_readiness()?;

        let mut families = Vec::with_capacity(artifact.campaign.evidences.len());
        for evidence in &artifact.campaign.evidences {
            evidence.validate()?;
            let qualification = evidence.qualification_record()?;
            let device = &evidence.materialization.device_ownership;
            families.push(QualifiedWeightFamilyCapabilityV1 {
                family: evidence.family,
                representation_version: evidence.representation_version,
                runtime_entrypoint: evidence.runtime_entrypoint.clone(),
                unique_logical_values: qualification.accounting.denominator.unique_logical_values,
                serialized_bits_per_unique_logical_value: qualification
                    .accounting
                    .serialized_bits_per_unique_logical_value,
                resident_bits_per_unique_logical_value: qualification
                    .accounting
                    .resident_bits_per_unique_logical_value,
                materialization_duration_ns: device.materialization_duration_ns,
                peak_scoped_owned_device_bytes: device.peak_scoped_owned_device_bytes,
                peak_host_temporary_payload_bytes: evidence
                    .materialization
                    .peak_host_temporary_payload_bytes,
                execution_storage: device.execution_storage.clone(),
                low_bit_compute: device.low_bit_compute,
                generated_token_count: evidence.generated_token_count,
                generated_token_ids_sha256: evidence.generated_token_ids_sha256.clone(),
                max_abs_error: evidence.max_abs_error,
                mean_squared_error: evidence.mean_squared_error,
            });
        }

        let record = Self {
            schema_version: NNIS_QUALIFIED_WEIGHT_CAPABILITY_RECORD_VERSION,
            nnis_commit: artifact.campaign.nnis_commit.clone(),
            exact_checkpoint: artifact.campaign.exact_checkpoint.clone(),
            tokenizer_file: artifact.tokenizer_file.clone(),
            tokenizer_sha256: artifact.tokenizer_sha256.clone(),
            recipe: artifact.recipe.clone(),
            families,
            elastic_stage_b_backend_ready: true,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_QUALIFIED_WEIGHT_CAPABILITY_RECORD_VERSION {
            return Err(NnisError::unsupported(format!(
                "qualified weight capability record schema {}; supported version is {}",
                self.schema_version, NNIS_QUALIFIED_WEIGHT_CAPABILITY_RECORD_VERSION
            )));
        }
        if self.nnis_commit.len() != 40
            || !self
                .nnis_commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(NnisError::invalid_input(
                "qualified weight capability NNIS commit must be 40 lowercase hexadecimal characters",
            ));
        }
        if self.exact_checkpoint.is_empty()
            || self.exact_checkpoint.trim() != self.exact_checkpoint
            || self.tokenizer_file.is_empty()
            || self.tokenizer_file.trim() != self.tokenizer_file
        {
            return Err(NnisError::invalid_input(
                "qualified weight capability identity fields must be non-empty and trimmed",
            ));
        }
        if self.tokenizer_sha256.len() != 64
            || !self
                .tokenizer_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(NnisError::invalid_input(
                "qualified weight tokenizer SHA-256 must be 64 lowercase hexadecimal characters",
            ));
        }
        self.recipe.validate()?;
        if !self.elastic_stage_b_backend_ready {
            return Err(NnisError::invalid_input(
                "qualified weight capability record must represent a Stage-B-ready backend",
            ));
        }
        if self.families.len() != 3 {
            return Err(NnisError::invalid_input(format!(
                "qualified weight capability record requires exactly three families; found {}",
                self.families.len()
            )));
        }

        let mut seen = BTreeSet::new();
        for family in &self.families {
            family.validate()?;
            if family.generated_token_count != self.recipe.max_new_tokens {
                return Err(NnisError::invalid_input(
                    "qualified weight family token count disagrees with frozen campaign recipe",
                ));
            }
            let key = match family.family {
                WeightRepresentationFamilyV1::Int4Symmetric => 0_u8,
                WeightRepresentationFamilyV1::Int2Ternary => 1_u8,
                WeightRepresentationFamilyV1::MagnitudeSparse => 2_u8,
            };
            if !seen.insert(key) {
                return Err(NnisError::invalid_input(
                    "qualified weight capability record contains a duplicate family",
                ));
            }
        }
        if seen != BTreeSet::from([0_u8, 1_u8, 2_u8]) {
            return Err(NnisError::invalid_input(
                "qualified weight capability record must contain INT4, INT2 and magnitude-sparse families",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DenseWeightMaterializationEvidenceV1, DenseWeightMaterializationEvidenceV2,
        WeightCampaignRecipeV1, WeightFullModelCampaignV1, WeightFullModelExecutionEvidenceV1,
        NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_VERSION, NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION,
    };

    fn evidence(family: WeightRepresentationFamilyV1) -> WeightFullModelExecutionEvidenceV1 {
        let (resident, serialized) = match family {
            WeightRepresentationFamilyV1::Int4Symmetric => (12, 24),
            WeightRepresentationFamilyV1::Int2Ternary => (8, 20),
            WeightRepresentationFamilyV1::MagnitudeSparse => (18, 34),
        };
        let device =
            DenseWeightMaterializationEvidenceV1::new(family, 16, 64, resident, 64, 0, 7).unwrap();
        WeightFullModelExecutionEvidenceV1 {
            schema_version: NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION,
            family,
            representation_version: 1,
            exact_checkpoint: "smollm2@sha256:test".to_string(),
            nnis_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            runtime_entrypoint: format!("{family:?}::model"),
            physical_execution_observed: true,
            generated_token_count: 4,
            generated_token_ids_sha256:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
            non_finite_output_observed: false,
            serialized_representation_bytes: serialized,
            logical_tensor_references: 1,
            logical_element_references: 16,
            unique_source_allocations: 1,
            max_abs_error: 0.25,
            mean_squared_error: 0.0625,
            materialization: DenseWeightMaterializationEvidenceV2::new(device, 80).unwrap(),
        }
    }

    fn artifact() -> WeightFullModelCampaignArtifactV1 {
        WeightFullModelCampaignArtifactV1 {
            schema_version: NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_VERSION,
            campaign: WeightFullModelCampaignV1::new(vec![
                evidence(WeightRepresentationFamilyV1::Int4Symmetric),
                evidence(WeightRepresentationFamilyV1::Int2Ternary),
                evidence(WeightRepresentationFamilyV1::MagnitudeSparse),
            ])
            .unwrap(),
            recipe: WeightCampaignRecipeV1::new(vec![1, 2], 4, 0.05).unwrap(),
            tokenizer_file: "tokenizer.json".to_string(),
            tokenizer_sha256:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
        }
    }

    #[test]
    fn qualified_record_is_derived_from_complete_artifact() {
        let record = QualifiedWeightCapabilityRecordV1::from_artifact(&artifact()).unwrap();
        record.validate().unwrap();
        assert!(record.elastic_stage_b_backend_ready);
        assert_eq!(record.families.len(), 3);
        assert!(record
            .families
            .iter()
            .all(|family| !family.low_bit_compute));
    }

    #[test]
    fn qualified_record_rejects_stage_b_or_identity_drift() {
        let mut record = QualifiedWeightCapabilityRecordV1::from_artifact(&artifact()).unwrap();
        record.elastic_stage_b_backend_ready = false;
        assert!(record.validate().is_err());

        let mut record = QualifiedWeightCapabilityRecordV1::from_artifact(&artifact()).unwrap();
        record.families[1].generated_token_count = 3;
        assert!(record.validate().is_err());

        let mut record = QualifiedWeightCapabilityRecordV1::from_artifact(&artifact()).unwrap();
        record.families[2] = record.families[0].clone();
        assert!(record.validate().is_err());
    }
}
