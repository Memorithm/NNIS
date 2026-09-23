//! Same-commit, same-checkpoint campaign evidence for fixed weight baselines.

use crate::{
    WeightFullModelExecutionEvidenceV1, WeightQualificationBundleV1, WeightRepresentationFamilyV1,
};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Version of the full-model weight qualification campaign contract.
pub const NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_VERSION: u32 = 1;

/// Three-family campaign bound to one exact NNIS commit and checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightFullModelCampaignV1 {
    pub schema_version: u32,
    pub nnis_commit: String,
    pub exact_checkpoint: String,
    pub evidences: Vec<WeightFullModelExecutionEvidenceV1>,
}

impl WeightFullModelCampaignV1 {
    pub fn new(evidences: Vec<WeightFullModelExecutionEvidenceV1>) -> Result<Self> {
        if evidences.is_empty() {
            return Err(NnisError::invalid_input(
                "full-model weight campaign requires evidence records",
            ));
        }
        let nnis_commit = evidences[0].nnis_commit.clone();
        let exact_checkpoint = evidences[0].exact_checkpoint.clone();
        let campaign = Self {
            schema_version: NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_VERSION,
            nnis_commit,
            exact_checkpoint,
            evidences,
        };
        campaign.validate()?;
        Ok(campaign)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_VERSION {
            return Err(NnisError::unsupported(format!(
                "full-model weight campaign schema {}; supported version is {}",
                self.schema_version, NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_VERSION
            )));
        }
        if self.evidences.len() != 3 {
            return Err(NnisError::invalid_input(format!(
                "full-model weight campaign requires exactly 3 family evidences; found {}",
                self.evidences.len()
            )));
        }
        let mut families = BTreeSet::new();
        for evidence in &self.evidences {
            evidence.validate()?;
            if evidence.nnis_commit != self.nnis_commit {
                return Err(NnisError::invalid_input(
                    "full-model weight campaign mixes NNIS commits",
                ));
            }
            if evidence.exact_checkpoint != self.exact_checkpoint {
                return Err(NnisError::invalid_input(
                    "full-model weight campaign mixes exact checkpoints",
                ));
            }
            let family_key = match evidence.family {
                WeightRepresentationFamilyV1::Int4Symmetric => 0_u8,
                WeightRepresentationFamilyV1::Int2Ternary => 1_u8,
                WeightRepresentationFamilyV1::MagnitudeSparse => 2_u8,
            };
            if !families.insert(family_key) {
                return Err(NnisError::invalid_input(
                    "full-model weight campaign contains a duplicate representation family",
                ));
            }
        }
        if families != BTreeSet::from([0_u8, 1_u8, 2_u8]) {
            return Err(NnisError::invalid_input(
                "full-model weight campaign must contain INT4, INT2 and magnitude-sparse evidence",
            ));
        }
        Ok(())
    }

    pub fn qualification_bundle(&self) -> Result<WeightQualificationBundleV1> {
        self.validate()?;
        let records = self
            .evidences
            .iter()
            .map(WeightFullModelExecutionEvidenceV1::qualification_record)
            .collect::<Result<Vec<_>>>()?;
        let bundle = WeightQualificationBundleV1::new(self.exact_checkpoint.clone(), records)?;
        bundle.validate_elastic_stage_b_readiness()?;
        Ok(bundle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DenseWeightMaterializationEvidenceV1, DenseWeightMaterializationEvidenceV2,
        WeightRepresentationFamilyV1, NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION,
    };

    fn evidence(family: WeightRepresentationFamilyV1) -> WeightFullModelExecutionEvidenceV1 {
        let (representation_version, resident, serialized, max_abs_error, mse) = match family {
            WeightRepresentationFamilyV1::Int4Symmetric => (1, 12, 24, 0.25, 0.0625),
            WeightRepresentationFamilyV1::Int2Ternary => (1, 8, 20, 0.5, 0.125),
            WeightRepresentationFamilyV1::MagnitudeSparse => (1, 18, 34, 0.2, 0.03),
        };
        let device = DenseWeightMaterializationEvidenceV1::new(
            family,
            16,
            64,
            resident,
            64,
            0,
            1,
        )
        .unwrap();
        WeightFullModelExecutionEvidenceV1 {
            schema_version: NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION,
            family,
            representation_version,
            exact_checkpoint: "smollm2@sha256:test".to_string(),
            nnis_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            runtime_entrypoint: "fixture::model".to_string(),
            physical_execution_observed: true,
            generated_token_count: 2,
            generated_token_ids_sha256:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                    .to_string(),
            non_finite_output_observed: false,
            serialized_representation_bytes: serialized,
            logical_tensor_references: 1,
            logical_element_references: 16,
            unique_source_allocations: 1,
            max_abs_error,
            mean_squared_error: mse,
            materialization: DenseWeightMaterializationEvidenceV2::new(device, 80).unwrap(),
        }
    }

    fn complete() -> Vec<WeightFullModelExecutionEvidenceV1> {
        vec![
            evidence(WeightRepresentationFamilyV1::Int4Symmetric),
            evidence(WeightRepresentationFamilyV1::Int2Ternary),
            evidence(WeightRepresentationFamilyV1::MagnitudeSparse),
        ]
    }

    #[test]
    fn complete_same_commit_campaign_builds_stage_b_bundle() {
        let campaign = WeightFullModelCampaignV1::new(complete()).unwrap();
        campaign.validate().unwrap();
        let bundle = campaign.qualification_bundle().unwrap();
        bundle.validate_elastic_stage_b_readiness().unwrap();
        assert_eq!(bundle.records.len(), 3);
    }

    #[test]
    fn mixed_commit_checkpoint_duplicate_or_missing_family_fail_closed() {
        let mut values = complete();
        values[1].nnis_commit = "fedcba9876543210fedcba9876543210fedcba98".to_string();
        assert!(WeightFullModelCampaignV1::new(values).is_err());

        let mut values = complete();
        values[1].exact_checkpoint = "other@sha256:test".to_string();
        assert!(WeightFullModelCampaignV1::new(values).is_err());

        let mut values = complete();
        values[2] = values[0].clone();
        assert!(WeightFullModelCampaignV1::new(values).is_err());

        let mut values = complete();
        values.pop();
        assert!(WeightFullModelCampaignV1::new(values).is_err());
    }
}
