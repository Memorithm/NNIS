use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Numeric storage used by a model's persisted/device weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightDType {
    F32,
    Bf16,
}

/// Activation implemented by the reusable decoder MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    Silu,
}

/// Model-neutral decoder-only transformer shape and numeric policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub activation: Activation,
    pub weight_dtype: WeightDType,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if value == 0 {
                return Err(NnisError::invalid_input(format!(
                    "model config {name} must be non-zero"
                )));
            }
        }
        if let Some(eos_token_id) = self.eos_token_id {
            if eos_token_id as usize >= self.vocab_size {
                return Err(NnisError::invalid_input(format!(
                    "eos_token_id {eos_token_id} is out of range for vocabulary {}",
                    self.vocab_size
                )));
            }
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(NnisError::invalid_input(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(NnisError::invalid_input(format!(
                "num_attention_heads {} is not divisible by num_key_value_heads {}",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.head_dim() % 2 != 0 {
            return Err(NnisError::invalid_input(format!(
                "rotary head dimension must be even; got {}",
                self.head_dim()
            )));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(NnisError::invalid_input(format!(
                "rms_norm_eps must be finite and positive; got {}",
                self.rms_norm_eps
            )));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(NnisError::invalid_input(format!(
                "rope_theta must be finite and positive; got {}",
                self.rope_theta
            )));
        }
        let _ = self
            .vocab_size
            .checked_mul(self.hidden_size)
            .ok_or_else(|| NnisError::invalid_input("embedding shape overflows usize"))?;
        let _ = self
            .max_position_embeddings
            .checked_mul(self.head_dim())
            .ok_or_else(|| NnisError::invalid_input("RoPE cache shape overflows usize"))?;
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn key_value_width(&self) -> Result<usize> {
        self.num_key_value_heads
            .checked_mul(self.head_dim())
            .ok_or_else(|| NnisError::invalid_input("key/value width overflows usize"))
    }

    /// Validate the numeric/activation combinations executable by the
    /// current decoder. Multi-head and grouped-query attention share the
    /// same head dimension; `validate` requires Q heads to be an integer
    /// multiple of KV heads.
    pub fn validate_execution_support(&self) -> Result<()> {
        self.validate()?;
        if self.activation != Activation::Silu {
            return Err(NnisError::unsupported(
                "decoder runtime currently supports only SiLU/SwiGLU MLPs",
            ));
        }
        Ok(())
    }
}

/// Deterministic generation policy. Sampling is intentionally not exposed yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub eos_token_id: Option<u32>,
}

impl GenerationConfig {
    /// Fixed-length greedy generation. This preserves the fully
    /// device-resident generation graph and does not stop on EOS.
    pub const fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            eos_token_id: None,
        }
    }

    /// Greedy generation that stops after producing `eos_token_id`.
    ///
    /// EOS-aware generation deliberately introduces one host-visible
    /// token observation per step so the session can stop submitting
    /// decoder work once the termination token has been produced.
    pub const fn greedy_until_eos(max_new_tokens: usize, eos_token_id: u32) -> Self {
        Self {
            max_new_tokens,
            eos_token_id: Some(eos_token_id),
        }
    }

    pub fn validate(&self, vocab_size: usize) -> Result<()> {
        if let Some(eos_token_id) = self.eos_token_id {
            if eos_token_id as usize >= vocab_size {
                return Err(NnisError::invalid_input(format!(
"generation eos_token_id {eos_token_id} is out of range for vocabulary {vocab_size}"
      )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 32_000,
            eos_token_id: Some(2),
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            max_position_embeddings: 128,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10_000.0,
            activation: Activation::Silu,
            weight_dtype: WeightDType::F32,
        }
    }

    #[test]
    fn tiny_llama_shape_is_accepted() {
        let config = valid_config();
        config.validate_execution_support().unwrap();
        assert_eq!(config.head_dim(), 32);
        assert_eq!(config.key_value_width().unwrap(), 64);
    }

    #[test]
    fn grouped_query_execution_is_supported_when_heads_divide_evenly() {
        let mut config = valid_config();
        config.num_attention_heads = 4;
        config.num_key_value_heads = 2;
        config.validate_execution_support().unwrap();
        assert_eq!(config.head_dim(), 16);
        assert_eq!(config.key_value_width().unwrap(), 32);
    }

    #[test]
    fn invalid_shapes_and_numeric_policy_are_rejected() {
        let mut config = valid_config();
        config.hidden_size = 63;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.hidden_size = 30;
        config.num_attention_heads = 2;
        config.num_key_value_heads = 2;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.rms_norm_eps = f32::NAN;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.rope_theta = 0.0;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.eos_token_id = Some(32_000);
        assert!(config.validate().is_err());

        assert!(GenerationConfig::greedy_until_eos(4, 32_000)
            .validate(32_000)
            .is_err());
    }
}
