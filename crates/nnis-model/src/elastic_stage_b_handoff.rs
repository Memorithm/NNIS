//! Fail-closed preregistration handoff for the Elastic Stage-B consumer.
//!
//! NNIS can attest backend capability, but this record deliberately does not
//! authorize Elastic development measurements or final-test access.

use crate::{QualifiedWeightHandoffV1, WeightFullModelCampaignArtifactV2};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of the NNIS -> Elastic Stage-B preregistration handoff.
pub const NNIS_ELASTIC_STAGE_B_HANDOFF_VERSION: u32 = 1;
/// Frozen downstream consumer owned by the cross-repository protocol.
pub const NNIS_ELASTIC_STAGE_B_CONSUMER: &str = "Memorithm/ElasticXxx#29";
/// Machine-readable locked-partition state.
pub const NNIS_ELASTIC_STAGE_B_FINAL_TEST_PARTITION_LOCKED: &str = "locked";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElasticStageBPreregistrationHandoffV1 {
    pub schema_version: u32,
    pub consumer: String,
    pub qualified_weight_handoff: QualifiedWeightHandoffV1,
    pub backend_ready_for_preregistration: bool,
    pub downstream_preregistration_update_required: bool,
    pub development_measurement_authorized: bool,
    pub final_test_access_authorized: bool,
    pub final_test_partition: String,
}

impl ElasticStageBPreregistrationHandoffV1 {
    /// Derive the NNIS-side handoff from one exact physical campaign artifact.
    ///
    /// This constructor intentionally keeps all measurement authorization false.
    pub fn from_artifact(artifact: &WeightFullModelCampaignArtifactV2) -> Result<Self> {
        let record = Self {
            schema_version: NNIS_ELASTIC_STAGE_B_HANDOFF_VERSION,
            consumer: NNIS_ELASTIC_STAGE_B_CONSUMER.to_string(),
            qualified_weight_handoff: QualifiedWeightHandoffV1::from_artifact(artifact)?,
            backend_ready_for_preregistration: true,
            downstream_preregistration_update_required: true,
            development_measurement_authorized: false,
            final_test_access_authorized: false,
            final_test_partition: NNIS_ELASTIC_STAGE_B_FINAL_TEST_PARTITION_LOCKED.to_string(),
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_ELASTIC_STAGE_B_HANDOFF_VERSION {
            return Err(NnisError::unsupported(format!(
                "Elastic Stage-B handoff schema {}; supported version is {}",
                self.schema_version, NNIS_ELASTIC_STAGE_B_HANDOFF_VERSION
            )));
        }
        self.qualified_weight_handoff.validate()?;
        if self.consumer != NNIS_ELASTIC_STAGE_B_CONSUMER {
            return Err(NnisError::invalid_input(
                "Elastic Stage-B handoff consumer identity drifted",
            ));
        }
        if !self.backend_ready_for_preregistration
            || !self.downstream_preregistration_update_required
        {
            return Err(NnisError::invalid_input(
                "Elastic Stage-B handoff must require downstream preregistration before measurement",
            ));
        }
        if self.development_measurement_authorized || self.final_test_access_authorized {
            return Err(NnisError::invalid_input(
                "NNIS Elastic Stage-B handoff cannot authorize measurements or final-test access",
            ));
        }
        if self.final_test_partition != NNIS_ELASTIC_STAGE_B_FINAL_TEST_PARTITION_LOCKED {
            return Err(NnisError::invalid_input(
                "Elastic Stage-B final-test partition must remain locked",
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
        WeightCampaignEnvironmentV1, WeightCampaignRecipeV1, WeightFullModelCampaignArtifactV1,
        WeightFullModelCampaignV1, WeightFullModelExecutionEvidenceV1,
        WeightRepresentationFamilyV1, NNIS_WEIGHT_CAMPAIGN_ENVIRONMENT_VERSION,
        NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION,
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

    fn artifact() -> WeightFullModelCampaignArtifactV2 {
        let campaign = WeightFullModelCampaignV1::new(vec![
            evidence(WeightRepresentationFamilyV1::Int4Symmetric),
            evidence(WeightRepresentationFamilyV1::Int2Ternary),
            evidence(WeightRepresentationFamilyV1::MagnitudeSparse),
        ])
        .unwrap();
        let v1 = WeightFullModelCampaignArtifactV1::new(
            campaign,
            WeightCampaignRecipeV1::new(vec![1, 2], 4, 0.05).unwrap(),
            "tokenizer.json",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .unwrap();
        let environment = WeightCampaignEnvironmentV1 {
            schema_version: NNIS_WEIGHT_CAMPAIGN_ENVIRONMENT_VERSION,
            device_ordinal: 0,
            device_name: "NVIDIA Test GPU".to_string(),
            device_uuid: "GPU-CUuuid([0, 1, 2, 3])".to_string(),
            compute_capability_major: 12,
            compute_capability_minor: 1,
            sm_arch: "sm_121".to_string(),
            multiprocessor_count: 16,
            clock_khz: 1_000_000,
            memory_clock_khz: 500_000,
            integrated: true,
            cuda_driver_major: 13,
            cuda_driver_minor: 0,
        };
        WeightFullModelCampaignArtifactV2::new(v1, environment).unwrap()
    }

    #[test]
    fn elastic_handoff_keeps_measurements_and_final_test_locked() {
        let record = ElasticStageBPreregistrationHandoffV1::from_artifact(&artifact()).unwrap();
        record.validate().unwrap();
        assert!(record.backend_ready_for_preregistration);
        assert!(record.downstream_preregistration_update_required);
        assert!(!record.development_measurement_authorized);
        assert!(!record.final_test_access_authorized);
        assert_eq!(record.final_test_partition, "locked");
        assert_eq!(record.consumer, "Memorithm/ElasticXxx#29");
    }

    #[test]
    fn elastic_handoff_rejects_authorization_or_consumer_drift() {
        let mut record = ElasticStageBPreregistrationHandoffV1::from_artifact(&artifact()).unwrap();
        record.development_measurement_authorized = true;
        assert!(record.validate().is_err());

        let mut record = ElasticStageBPreregistrationHandoffV1::from_artifact(&artifact()).unwrap();
        record.consumer = "other".to_string();
        assert!(record.validate().is_err());

        let mut record = ElasticStageBPreregistrationHandoffV1::from_artifact(&artifact()).unwrap();
        record.final_test_partition = "open".to_string();
        assert!(record.validate().is_err());
    }
}
