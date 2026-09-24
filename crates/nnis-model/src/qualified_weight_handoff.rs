//! Immutable downstream handoff for qualified full-model weight capabilities.
//!
//! This record joins the environment-bound Stage-B capability record to the
//! SHA-256 fingerprint of the exact campaign artifact that produced it.

use crate::{
    QualifiedWeightCapabilityRecordV2, WeightCampaignArtifactFingerprintV1,
    WeightFullModelCampaignArtifactV2,
};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of the immutable qualified-weight handoff record.
pub const NNIS_QUALIFIED_WEIGHT_HANDOFF_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedWeightHandoffV1 {
    pub schema_version: u32,
    pub artifact_fingerprint: WeightCampaignArtifactFingerprintV1,
    pub qualified_capability: QualifiedWeightCapabilityRecordV2,
}

impl QualifiedWeightHandoffV1 {
    /// Derive a downstream handoff from one validated physical campaign artifact.
    pub fn from_artifact(artifact: &WeightFullModelCampaignArtifactV2) -> Result<Self> {
        artifact.validate()?;
        let handoff = Self {
            schema_version: NNIS_QUALIFIED_WEIGHT_HANDOFF_VERSION,
            artifact_fingerprint: artifact.fingerprint()?,
            qualified_capability: QualifiedWeightCapabilityRecordV2::from_artifact(artifact)?,
        };
        handoff.validate()?;
        Ok(handoff)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_QUALIFIED_WEIGHT_HANDOFF_VERSION {
            return Err(NnisError::unsupported(format!(
                "qualified weight handoff schema {}; supported version is {}",
                self.schema_version, NNIS_QUALIFIED_WEIGHT_HANDOFF_VERSION
            )));
        }
        self.artifact_fingerprint.validate()?;
        self.qualified_capability.validate()?;

        let capability = &self.qualified_capability.capability;
        let environment = &self.qualified_capability.environment;
        let fingerprint = &self.artifact_fingerprint;

        if fingerprint.nnis_commit != capability.nnis_commit
            || fingerprint.exact_checkpoint != capability.exact_checkpoint
            || fingerprint.tokenizer_sha256 != capability.tokenizer_sha256
        {
            return Err(NnisError::invalid_input(
                "qualified weight handoff artifact identity disagrees with capability record",
            ));
        }
        if fingerprint.device_uuid != environment.device_uuid
            || fingerprint.sm_arch != environment.sm_arch
            || fingerprint.cuda_driver_major != environment.cuda_driver_major
            || fingerprint.cuda_driver_minor != environment.cuda_driver_minor
        {
            return Err(NnisError::invalid_input(
                "qualified weight handoff artifact environment disagrees with capability record",
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
    fn handoff_is_derived_from_one_immutable_artifact() {
        let handoff = QualifiedWeightHandoffV1::from_artifact(&artifact()).unwrap();
        handoff.validate().unwrap();
        assert!(
            handoff
                .qualified_capability
                .capability
                .elastic_stage_b_backend_ready
        );
        assert_eq!(
            handoff.artifact_fingerprint.nnis_commit,
            handoff.qualified_capability.capability.nnis_commit
        );

        let encoded = serde_json::to_string(&handoff).unwrap();
        let decoded: QualifiedWeightHandoffV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, handoff);
    }

    #[test]
    fn handoff_rejects_detached_capability_or_environment() {
        let mut handoff = QualifiedWeightHandoffV1::from_artifact(&artifact()).unwrap();
        handoff.qualified_capability.capability.tokenizer_sha256 =
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff".to_string();
        assert!(handoff.validate().is_err());

        let mut handoff = QualifiedWeightHandoffV1::from_artifact(&artifact()).unwrap();
        handoff.qualified_capability.environment.sm_arch = "sm_120".to_string();
        assert!(handoff.validate().is_err());
    }
}
