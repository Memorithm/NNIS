//! Preregistered SmolLM2 protocol for full-model weight qualification.
//!
//! This protocol is intentionally narrow. It binds the physical INT4/INT2/
//! sparse campaign to the exact SmolLM2 checkpoint and the already-reproduced
//! trusted greedy reference used by the SmolLM2 CI fixture.

use crate::{GeneratedTokenEvidenceV1, WeightCampaignRecipeV1, SMOLLM2_135M_BF16};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of the preregistered SmolLM2 weight qualification protocol.
pub const NNIS_SMOLLM2_WEIGHT_QUALIFICATION_PROTOCOL_VERSION: u32 = 1;
pub const NNIS_SMOLLM2_WEIGHT_QUALIFICATION_PROMPT_TOKEN_IDS: [u32; 3] = [22_007, 6_463, 314];
pub const NNIS_SMOLLM2_WEIGHT_QUALIFICATION_EXPECTED_GREEDY_TOKEN_IDS: [u32; 2] = [260, 3_075];
pub const NNIS_SMOLLM2_WEIGHT_QUALIFICATION_MAX_NEW_TOKENS: usize = 2;
pub const NNIS_SMOLLM2_WEIGHT_QUALIFICATION_SPARSE_THRESHOLD: f32 = 0.05;

/// Machine-readable preregistration for one exact physical qualification run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmolLm2WeightQualificationProtocolV1 {
    pub schema_version: u32,
    pub checkpoint_evidence_key: String,
    pub prompt_token_ids: Vec<u32>,
    pub expected_generated_token_ids: Vec<u32>,
    pub expected_generated_token_ids_sha256: String,
    pub max_new_tokens: u64,
    pub sparse_threshold: f32,
}

impl SmolLm2WeightQualificationProtocolV1 {
    pub fn reference() -> Result<Self> {
        let expected = GeneratedTokenEvidenceV1::from_token_ids(
            &NNIS_SMOLLM2_WEIGHT_QUALIFICATION_EXPECTED_GREEDY_TOKEN_IDS,
        )?;
        let protocol = Self {
            schema_version: NNIS_SMOLLM2_WEIGHT_QUALIFICATION_PROTOCOL_VERSION,
            checkpoint_evidence_key: SMOLLM2_135M_BF16.evidence_key(),
            prompt_token_ids: NNIS_SMOLLM2_WEIGHT_QUALIFICATION_PROMPT_TOKEN_IDS.to_vec(),
            expected_generated_token_ids:
                NNIS_SMOLLM2_WEIGHT_QUALIFICATION_EXPECTED_GREEDY_TOKEN_IDS.to_vec(),
            expected_generated_token_ids_sha256: expected.token_ids_sha256,
            max_new_tokens: NNIS_SMOLLM2_WEIGHT_QUALIFICATION_MAX_NEW_TOKENS as u64,
            sparse_threshold: NNIS_SMOLLM2_WEIGHT_QUALIFICATION_SPARSE_THRESHOLD,
        };
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn recipe(&self) -> Result<WeightCampaignRecipeV1> {
        self.validate()?;
        WeightCampaignRecipeV1::new(
            self.prompt_token_ids.clone(),
            usize::try_from(self.max_new_tokens).map_err(|_| {
                NnisError::invalid_input("SmolLM2 qualification generation length exceeds usize")
            })?,
            self.sparse_threshold,
        )
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_SMOLLM2_WEIGHT_QUALIFICATION_PROTOCOL_VERSION {
            return Err(NnisError::unsupported(format!(
                "SmolLM2 weight qualification protocol schema {}; supported version is {}",
                self.schema_version, NNIS_SMOLLM2_WEIGHT_QUALIFICATION_PROTOCOL_VERSION
            )));
        }
        if self.checkpoint_evidence_key != SMOLLM2_135M_BF16.evidence_key()
            || self.prompt_token_ids != NNIS_SMOLLM2_WEIGHT_QUALIFICATION_PROMPT_TOKEN_IDS
            || self.expected_generated_token_ids
                != NNIS_SMOLLM2_WEIGHT_QUALIFICATION_EXPECTED_GREEDY_TOKEN_IDS
            || self.max_new_tokens
                != NNIS_SMOLLM2_WEIGHT_QUALIFICATION_MAX_NEW_TOKENS as u64
            || self.sparse_threshold.to_bits()
                != NNIS_SMOLLM2_WEIGHT_QUALIFICATION_SPARSE_THRESHOLD.to_bits()
        {
            return Err(NnisError::invalid_input(
                "SmolLM2 weight qualification protocol drifted from the preregistered reference",
            ));
        }
        let expected = GeneratedTokenEvidenceV1::from_token_ids(
            &NNIS_SMOLLM2_WEIGHT_QUALIFICATION_EXPECTED_GREEDY_TOKEN_IDS,
        )?;
        if self.expected_generated_token_ids_sha256 != expected.token_ids_sha256 {
            return Err(NnisError::invalid_input(
                "SmolLM2 weight qualification expected token hash drifted",
            ));
        }
        let recipe = WeightCampaignRecipeV1::new(
            self.prompt_token_ids.clone(),
            NNIS_SMOLLM2_WEIGHT_QUALIFICATION_MAX_NEW_TOKENS,
            self.sparse_threshold,
        )?;
        recipe.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_protocol_matches_reproduced_smollm2_oracle() {
        let protocol = SmolLm2WeightQualificationProtocolV1::reference().unwrap();
        protocol.validate().unwrap();
        assert_eq!(protocol.prompt_token_ids, vec![22_007, 6_463, 314]);
        assert_eq!(protocol.expected_generated_token_ids, vec![260, 3_075]);
        assert_eq!(protocol.max_new_tokens, 2);
        assert_eq!(protocol.sparse_threshold.to_bits(), 0.05_f32.to_bits());
        assert_eq!(protocol.recipe().unwrap().prompt_token_ids, protocol.prompt_token_ids);

        let encoded = serde_json::to_string(&protocol).unwrap();
        let decoded: SmolLm2WeightQualificationProtocolV1 =
            serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, protocol);
    }

    #[test]
    fn protocol_drift_fails_closed() {
        let mut protocol = SmolLm2WeightQualificationProtocolV1::reference().unwrap();
        protocol.prompt_token_ids[0] += 1;
        assert!(protocol.validate().is_err());

        let mut protocol = SmolLm2WeightQualificationProtocolV1::reference().unwrap();
        protocol.expected_generated_token_ids[0] += 1;
        assert!(protocol.validate().is_err());

        let mut protocol = SmolLm2WeightQualificationProtocolV1::reference().unwrap();
        protocol.sparse_threshold = 0.1;
        assert!(protocol.validate().is_err());

        let mut protocol = SmolLm2WeightQualificationProtocolV1::reference().unwrap();
        protocol.expected_generated_token_ids_sha256 =
            "0000000000000000000000000000000000000000000000000000000000000000".to_string();
        assert!(protocol.validate().is_err());
    }
}
