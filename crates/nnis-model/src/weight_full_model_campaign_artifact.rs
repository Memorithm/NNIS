//! Self-contained full-model weight campaign artifact.
//!
//! This envelope joins the same-commit campaign evidence, fixed token-level
//! execution recipe, and tokenizer artifact identity used by the preregistered
//! qualification workflow.

use crate::{WeightCampaignRecipeV1, WeightFullModelCampaignV1};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

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

    /// Verify the concrete tokenizer file bound by this campaign artifact.
    ///
    /// The basename must match the recorded tokenizer file and the bytes are
    /// hashed incrementally to avoid whole-file buffering.
    pub fn verify_tokenizer_file(&self, path: impl AsRef<Path>) -> Result<String> {
        self.validate()?;
        let path = path.as_ref();
        let basename = path
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                NnisError::invalid_input("weight campaign tokenizer path has no UTF-8 basename")
            })?;
        if basename != self.tokenizer_file {
            return Err(NnisError::invalid_input(format!(
                "weight campaign tokenizer basename mismatch: got {basename:?}, expected {:?}",
                self.tokenizer_file
            )));
        }

        let mut file = File::open(path)
            .map_err(|error| NnisError::io("open weight campaign tokenizer file", error))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|error| NnisError::io("hash weight campaign tokenizer file", error))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let actual = format!("{:x}", hasher.finalize());
        if actual != self.tokenizer_sha256 {
            return Err(NnisError::invalid_input(format!(
                "weight campaign tokenizer SHA-256 mismatch: got {actual}, expected {}",
                self.tokenizer_sha256
            )));
        }
        Ok(actual)
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
                    evidence.family, evidence.generated_token_count, self.recipe.max_new_tokens
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
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string(),
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
    fn tokenizer_file_verification_binds_basename_and_bytes() {
        let directory = std::env::temp_dir().join(format!(
            "nnis-weight-campaign-tokenizer-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("tokenizer.json");
        std::fs::write(&path, b"abc").unwrap();

        let artifact = WeightFullModelCampaignArtifactV1::new(
            campaign(),
            WeightCampaignRecipeV1::new(vec![1], 4, 0.05).unwrap(),
            "tokenizer.json",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        .unwrap();
        assert_eq!(
            artifact.verify_tokenizer_file(&path).unwrap(),
            artifact.tokenizer_sha256
        );

        let wrong_name = directory.join("other.json");
        std::fs::write(&wrong_name, b"abc").unwrap();
        assert!(artifact.verify_tokenizer_file(&wrong_name).is_err());

        std::fs::write(&path, b"abd").unwrap();
        assert!(artifact.verify_tokenizer_file(&path).is_err());

        std::fs::remove_dir_all(&directory).unwrap();
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
