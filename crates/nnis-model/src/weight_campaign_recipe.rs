//! Frozen execution recipe for one full-model weight qualification campaign.
//!
//! The recipe is deliberately token-level. It prevents representation families
//! from silently changing prompt token IDs, decoding strategy, generation
//! length or the fixed sparse threshold while preserving tokenizer identity as
//! a separate preregistered evidence axis.

use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of the fixed full-model weight campaign execution recipe.
pub const NNIS_WEIGHT_CAMPAIGN_RECIPE_VERSION: u32 = 1;

/// Decoding policy admitted by the fixed campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightCampaignDecodingV1 {
    GreedyFixed,
}

/// Same-input execution controls shared by INT4, INT2 and sparse runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightCampaignRecipeV1 {
    pub schema_version: u32,
    pub prompt_token_ids: Vec<u32>,
    pub max_new_tokens: u64,
    pub decoding: WeightCampaignDecodingV1,
    pub sparse_threshold: f32,
}

impl WeightCampaignRecipeV1 {
    pub fn new(
        prompt_token_ids: Vec<u32>,
        max_new_tokens: usize,
        sparse_threshold: f32,
    ) -> Result<Self> {
        let recipe = Self {
            schema_version: NNIS_WEIGHT_CAMPAIGN_RECIPE_VERSION,
            prompt_token_ids,
            max_new_tokens: u64::try_from(max_new_tokens).map_err(|_| {
                NnisError::invalid_input("weight campaign generation length exceeds u64")
            })?,
            decoding: WeightCampaignDecodingV1::GreedyFixed,
            sparse_threshold,
        };
        recipe.validate()?;
        Ok(recipe)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_WEIGHT_CAMPAIGN_RECIPE_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight campaign recipe schema {}; supported version is {}",
                self.schema_version, NNIS_WEIGHT_CAMPAIGN_RECIPE_VERSION
            )));
        }
        if self.prompt_token_ids.is_empty() {
            return Err(NnisError::invalid_input(
                "weight campaign recipe requires at least one prompt token",
            ));
        }
        if self.max_new_tokens == 0 {
            return Err(NnisError::invalid_input(
                "weight campaign recipe requires a non-zero generation length",
            ));
        }
        if !self.sparse_threshold.is_finite() || self.sparse_threshold < 0.0 {
            return Err(NnisError::invalid_input(
                "weight campaign sparse threshold must be finite and non-negative",
            ));
        }
        if self.decoding != WeightCampaignDecodingV1::GreedyFixed {
            return Err(NnisError::unsupported(
                "weight campaign recipe supports only fixed-length greedy decoding",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_freezes_same_prompt_decoding_and_sparse_threshold() {
        let recipe = WeightCampaignRecipeV1::new(vec![1, 2, 3], 8, 0.05).unwrap();
        recipe.validate().unwrap();
        assert_eq!(recipe.prompt_token_ids, vec![1, 2, 3]);
        assert_eq!(recipe.max_new_tokens, 8);
        assert_eq!(recipe.decoding, WeightCampaignDecodingV1::GreedyFixed);
        assert_eq!(recipe.sparse_threshold.to_bits(), 0.05_f32.to_bits());

        let encoded = serde_json::to_string(&recipe).unwrap();
        let decoded: WeightCampaignRecipeV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, recipe);
    }

    #[test]
    fn malformed_recipe_fails_closed() {
        assert!(WeightCampaignRecipeV1::new(Vec::new(), 8, 0.05).is_err());
        assert!(WeightCampaignRecipeV1::new(vec![1], 0, 0.05).is_err());
        assert!(WeightCampaignRecipeV1::new(vec![1], 8, -0.1).is_err());
        assert!(WeightCampaignRecipeV1::new(vec![1], 8, f32::NAN).is_err());
    }
}
