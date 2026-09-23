//! Self-contained full-model weight campaign artifact.
//!
//! This envelope joins the same-commit campaign evidence, fixed token-level
//! execution recipe, and tokenizer artifact identity used by the preregistered
//! qualification workflow.

use crate::{WeightCampaignRecipeV1, WeightFullModelCampaignV1};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of the self-contained campaign artifact envelope.
pub const NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightFullModelCampaignArtifactV1 {
    pub schema_version: u32,
    pub campaign: WeightFullModelCampaignV1,
    pub recipe: WeightCampaignRecipeV1,
    pub tokenizer_file: String,
    pub tokenizer_sha256: String,
}

impl WeightFullModelCampaignArtifactV1 {
    pub fn new(
        campaign: WeightFullModelCampaignV1,
        recipe: WeightCampaignRecipeV1,
        tokenizer_file: impl Into<String>,
        tokenizer_sha256: impl Into<String>,
    ) -> Result<Self> {
        let artifact = Self {
            schema_version: NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_VERSION,
            campaign,
            recipe,
            tokenizer_file: tokenizer_file.into(),
            tokenizer_sha256: tokenizer_sha256.into(),
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight campaign artifact schema {}; supported version is {}",
                self.schema_version, NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_VERSION
            )));
        }
        self.campaign.validate()?;
        self.campaign.qualification_bundle()?;
        self.recipe.validate()?;

        if self.tokenizer_file.is_empty()
            || self.tokenizer_file.trim() != self.tokenizer_file
            || self.tokenizer_file.contains('/')
            || self.tokenizer_file.contains('\\')
        {
            return Err(NnisError::invalid_input(
                "weight campaign tokenizer file must be a trimmed basename",
            ));
        }
        validate_lower_sha256("weight campaign tokenizer SHA-256", &self.tokenizer_sha256)?;

        for evidence in &self.campaign.evidences {
            if evidence.generated_token_count != self.recipe.max_new_tokens {
                return Err(NnisError::invalid_input(format!(
                    "weight campaign {:?} generated {} tokens but recipe requires {}",
                    evidence.family,
                    evidence.generated_token_count,
                    self.recipe.max_new_tokens
                )));
            }
        }
        Ok(())
    }
}

fn validate_lower_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NnisError::invalid_input(format!(
            "{label} must contain exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DenseWeightMaterializationEvidenceV1, DenseWeightMaterializationEvidenceV2,
        WeightFullModelExecutionEvidenceV1, WeightRepresentationFamilyV1,
        NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION,
    };

    fn evidence(family: WeightRepresentationFamilyV1) -> WeightFullModelExecutionEvidenceV1 {
        let resident = match family {
            WeightRepresentationFamilyV1::Int4Symmetric => 12,
            WeightRepresentationFamilyV1::Int2Ternary => 8,
            WeightRepresentationFamilyV1::MagnitudeSparse => 18,
        };
        let device =
            DenseWeightMaterializationEvidenceV1::new(family, 16, 64, resident, 64, 0, 1).unwrap();
        WeightFullModelExecutionEvidenceV1 {
            schema_version: NNIS_WEIGHT_FULL_MODEL_EVIDENCE_VERSION,
            family,
            representation_version: 1,
            exact_checkpoint: "smollm2@sha256:test".to_string(),
            nnis_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            runtime_entrypoint: "fixture::model".to_string(),
            physical_execution_observed: true,
            generated_token_count: 4,
            generated_token_ids_sha256:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                    .to_string(),
            non_finite_output_observed: false,
            serialized_representation_bytes: 24,
            logical_tensor_references: 1,
            logical_element_references: 16,
            unique_source_allocations: 1,
            max_abs_error: 0.25,
            mean_squared_error: 0.0625,
            materialization: DenseWeightMaterializationEvidenceV2::new(device, 80).unwrap(),
        }
    }

    fn campaign() -> WeightFullModelCampaignV1 {
        WeightFullModelCampaignV1::new(vec![
            evidence(WeightRepresentationFamilyV1::Int4Symmetric),
            evidence(WeightRepresentationFamilyV1::Int2Ternary),
            evidence(WeightRepresentationFamilyV1::MagnitudeSparse),
        ])
        .unwrap()
    }

    #[test]
    fn artifact_joins_campaign_recipe_and_tokenizer_identity() {
        let artifact = WeightFullModelCampaignArtifactV1::new(
            campaign(),
            WeightCampaignRecipeV1::new(vec![1, 2], 4, 0.05).unwrap(),
            "tokenizer.json",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .unwrap();
        artifact.validate().unwrap();

        let encoded = serde_json::to_string(&artifact).unwrap();
        let decoded: WeightFullModelCampaignArtifactV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, artifact);
    }

    #[test]
    fn artifact_rejects_token_count_or_tokenizer_drift() {
        let mut wrong_count = campaign();
        wrong_count.evidences[0].generated_token_count = 3;
        assert!(WeightFullModelCampaignArtifactV1::new(
            wrong_count,
            WeightCampaignRecipeV1::new(vec![1], 4, 0.05).unwrap(),
            "tokenizer.json",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .is_err());

        assert!(WeightFullModelCampaignArtifactV1::new(
            campaign(),
            WeightCampaignRecipeV1::new(vec![1], 4, 0.05).unwrap(),
            "../tokenizer.json",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .is_err());
        assert!(WeightFullModelCampaignArtifactV1::new(
            campaign(),
            WeightCampaignRecipeV1::new(vec![1], 4, 0.05).unwrap(),
            "tokenizer.json",
            "BAD",
        )
        .is_err());
    }
}
