//! Self-contained full-model weight campaign artifact.
//!
//! This envelope joins the same-commit campaign evidence, fixed token-level
//! execution recipe, and tokenizer artifact identity used by the preregistered
//! qualification workflow.

use crate::{WeightCampaignEnvironmentV1, WeightCampaignRecipeV1, WeightFullModelCampaignV1};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Version of the self-contained campaign artifact envelope.
pub const NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_VERSION: u32 = 1;
/// Version of the environment-bound campaign artifact envelope.
pub const NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_V2_VERSION: u32 = 2;
/// Version of the artifact fingerprint handoff identity.
pub const NNIS_WEIGHT_CAMPAIGN_ARTIFACT_FINGERPRINT_VERSION: u32 = 1;

/// Immutable identity of one validated environment-bound campaign artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightCampaignArtifactFingerprintV1 {
    pub schema_version: u32,
    pub artifact_schema_version: u32,
    pub artifact_json_sha256: String,
    pub nnis_commit: String,
    pub exact_checkpoint: String,
    pub tokenizer_sha256: String,
    pub device_uuid: String,
    pub sm_arch: String,
    pub cuda_driver_major: i32,
    pub cuda_driver_minor: i32,
}

impl WeightCampaignArtifactFingerprintV1 {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_WEIGHT_CAMPAIGN_ARTIFACT_FINGERPRINT_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight campaign artifact fingerprint schema {}; supported version is {}",
                self.schema_version, NNIS_WEIGHT_CAMPAIGN_ARTIFACT_FINGERPRINT_VERSION
            )));
        }
        if self.artifact_schema_version != NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_V2_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight campaign fingerprint artifact schema {}; expected {}",
                self.artifact_schema_version, NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_V2_VERSION
            )));
        }
        validate_lower_sha256(
            "weight campaign artifact JSON SHA-256",
            &self.artifact_json_sha256,
        )?;
        validate_lower_sha256(
            "weight campaign fingerprint tokenizer SHA-256",
            &self.tokenizer_sha256,
        )?;
        if self.nnis_commit.len() != 40
            || !self
                .nnis_commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || self.exact_checkpoint.is_empty()
            || self.exact_checkpoint.trim() != self.exact_checkpoint
            || self.device_uuid.is_empty()
            || self.device_uuid.trim() != self.device_uuid
            || self.sm_arch.is_empty()
            || self.sm_arch.trim() != self.sm_arch
            || self.cuda_driver_major <= 0
            || self.cuda_driver_minor < 0
        {
            return Err(NnisError::invalid_input(
                "weight campaign artifact fingerprint identity fields are invalid",
            ));
        }
        Ok(())
    }
}

/// Environment-bound wrapper around one validated V1 campaign artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightFullModelCampaignArtifactV2 {
    pub schema_version: u32,
    pub campaign_artifact: WeightFullModelCampaignArtifactV1,
    pub environment: WeightCampaignEnvironmentV1,
}

impl WeightFullModelCampaignArtifactV2 {
    /// Derive a SHA-256 identity over the complete compact JSON wire payload.
    ///
    /// The digest covers every V2 field, including nested family evidence,
    /// recipe, tokenizer identity and physical CUDA environment.
    pub fn fingerprint(&self) -> Result<WeightCampaignArtifactFingerprintV1> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|error| {
            NnisError::invalid_input(format!(
                "failed to serialize weight campaign artifact v2 for fingerprinting: {error}"
            ))
        })?;
        let artifact_json_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let campaign = &self.campaign_artifact.campaign;
        let fingerprint = WeightCampaignArtifactFingerprintV1 {
            schema_version: NNIS_WEIGHT_CAMPAIGN_ARTIFACT_FINGERPRINT_VERSION,
            artifact_schema_version: self.schema_version,
            artifact_json_sha256,
            nnis_commit: campaign.nnis_commit.clone(),
            exact_checkpoint: campaign.exact_checkpoint.clone(),
            tokenizer_sha256: self.campaign_artifact.tokenizer_sha256.clone(),
            device_uuid: self.environment.device_uuid.clone(),
            sm_arch: self.environment.sm_arch.clone(),
            cuda_driver_major: self.environment.cuda_driver_major,
            cuda_driver_minor: self.environment.cuda_driver_minor,
        };
        fingerprint.validate()?;
        Ok(fingerprint)
    }

    pub fn new(
        campaign_artifact: WeightFullModelCampaignArtifactV1,
        environment: WeightCampaignEnvironmentV1,
    ) -> Result<Self> {
        let artifact = Self {
            schema_version: NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_V2_VERSION,
            campaign_artifact,
            environment,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_V2_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight campaign artifact v2 schema {}; supported version is {}",
                self.schema_version, NNIS_WEIGHT_FULL_MODEL_CAMPAIGN_ARTIFACT_V2_VERSION
            )));
        }
        self.campaign_artifact.validate()?;
        self.environment.validate()
    }

    pub fn verify_tokenizer_file(&self, path: impl AsRef<Path>) -> Result<String> {
        self.validate()?;
        self.campaign_artifact.verify_tokenizer_file(path)
    }
}

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

    fn environment() -> WeightCampaignEnvironmentV1 {
        WeightCampaignEnvironmentV1 {
            schema_version: crate::NNIS_WEIGHT_CAMPAIGN_ENVIRONMENT_VERSION,
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
        }
    }

    #[test]
    fn v2_fingerprint_is_deterministic_and_environment_sensitive() {
        let v1 = WeightFullModelCampaignArtifactV1::new(
            campaign(),
            WeightCampaignRecipeV1::new(vec![1, 2], 4, 0.05).unwrap(),
            "tokenizer.json",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .unwrap();
        let artifact = WeightFullModelCampaignArtifactV2::new(v1, environment()).unwrap();
        let first = artifact.fingerprint().unwrap();
        let second = artifact.fingerprint().unwrap();
        assert_eq!(first, second);
        first.validate().unwrap();
        assert_eq!(first.nnis_commit, artifact.campaign_artifact.campaign.nnis_commit);
        assert_eq!(first.sm_arch, artifact.environment.sm_arch);

        let mut changed = artifact;
        changed.environment.clock_khz += 1;
        let changed_fingerprint = changed.fingerprint().unwrap();
        assert_ne!(
            first.artifact_json_sha256,
            changed_fingerprint.artifact_json_sha256
        );
    }

    #[test]
    fn v2_binds_validated_campaign_artifact_to_physical_environment() {
        let v1 = WeightFullModelCampaignArtifactV1::new(
            campaign(),
            WeightCampaignRecipeV1::new(vec![1, 2], 4, 0.05).unwrap(),
            "tokenizer.json",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .unwrap();
        let artifact = WeightFullModelCampaignArtifactV2::new(v1, environment()).unwrap();
        artifact.validate().unwrap();
        assert_eq!(artifact.schema_version, 2);

        let encoded = serde_json::to_string(&artifact).unwrap();
        let decoded: WeightFullModelCampaignArtifactV2 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, artifact);

        let mut drifted = artifact;
        drifted.environment.sm_arch = "sm_120".to_string();
        assert!(drifted.validate().is_err());
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
