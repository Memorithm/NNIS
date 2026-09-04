use crate::{ModelConfig, WeightDType};
use nnis_rt::Result;
use serde::{Deserialize, Serialize};

pub const NNIS_DECODER_CAPABILITY_VERSION: u32 = 1;

/// Attention head topology executed by the current decoder block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderAttentionTopology {
    MultiHead,
    GroupedQuery,
    MultiQuery,
}

impl DecoderAttentionTopology {
    const fn stable_name(self) -> &'static str {
        match self {
            Self::MultiHead => "multi_head",
            Self::GroupedQuery => "grouped_query",
            Self::MultiQuery => "multi_query",
        }
    }
}

/// Exact RoPE arithmetic/layout family currently implemented by NNIS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderRopeSemantics {
    /// Llama rotate-half pairing with a fixed theta and no scaling/interleaving.
    LlamaRotateHalfUnscaled,
}

impl DecoderRopeSemantics {
    const fn stable_name(self) -> &'static str {
        match self {
            Self::LlamaRotateHalfUnscaled => "llama_rotate_half_unscaled",
        }
    }
}

/// Exact MLP semantic family currently implemented by NNIS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecoderMlpSemantics {
    SwiGluSilu,
}

impl DecoderMlpSemantics {
    const fn stable_name(self) -> &'static str {
        match self {
            Self::SwiGluSilu => "swiglu_silu",
        }
    }
}

/// Versioned executable capability profile derived from a validated `ModelConfig`.
///
/// This describes runtime semantics and geometry only. It is not a claim that a
/// Hugging Face architecture/family is supported, and it does not establish
/// model-quality or performance parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecoderExecutionCapabilities {
    pub contract_version: u32,
    pub attention_topology: DecoderAttentionTopology,
    pub rope_semantics: DecoderRopeSemantics,
    pub mlp_semantics: DecoderMlpSemantics,
    pub weight_dtype: WeightDType,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

impl DecoderExecutionCapabilities {
    /// Deterministic, human-readable identity for evidence and test fixtures.
    ///
    /// The record is a provenance/cache key, not an authentication primitive.
    pub fn canonical_record(&self) -> String {
        format!(
            "NNIS-DECODER-CAPABILITY-V{}\nattention={}\nrope={}\nmlp={}\nweight_dtype={}\nq_heads={}\nkv_heads={}\nhead_dim={}\n",
            self.contract_version,
            self.attention_topology.stable_name(),
            self.rope_semantics.stable_name(),
            self.mlp_semantics.stable_name(),
            match self.weight_dtype {
                WeightDType::F32 => "f32",
                WeightDType::Bf16 => "bf16",
            },
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim
        )
    }
}

impl ModelConfig {
    /// Derive the exact decoder execution capability profile after validating
    /// that the current NNIS decoder can execute this configuration.
    pub fn decoder_capabilities(&self) -> Result<DecoderExecutionCapabilities> {
        self.validate_execution_support()?;
        let attention_topology = if self.num_key_value_heads == self.num_attention_heads {
            DecoderAttentionTopology::MultiHead
        } else if self.num_key_value_heads == 1 {
            DecoderAttentionTopology::MultiQuery
        } else {
            DecoderAttentionTopology::GroupedQuery
        };

        Ok(DecoderExecutionCapabilities {
            contract_version: NNIS_DECODER_CAPABILITY_VERSION,
            attention_topology,
            rope_semantics: DecoderRopeSemantics::LlamaRotateHalfUnscaled,
            mlp_semantics: DecoderMlpSemantics::SwiGluSilu,
            weight_dtype: self.weight_dtype,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Activation, WeightDType};

    fn config(q_heads: usize, kv_heads: usize) -> ModelConfig {
        ModelConfig {
            vocab_size: 32_000,
            eos_token_id: Some(2),
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: q_heads,
            num_key_value_heads: kv_heads,
            max_position_embeddings: 128,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10_000.0,
            activation: Activation::Silu,
            weight_dtype: WeightDType::F32,
        }
    }

    #[test]
    fn classifies_mha_gqa_and_mqa_without_changing_geometry() {
        assert_eq!(
            config(4, 4)
                .decoder_capabilities()
                .unwrap()
                .attention_topology,
            DecoderAttentionTopology::MultiHead
        );
        assert_eq!(
            config(4, 2)
                .decoder_capabilities()
                .unwrap()
                .attention_topology,
            DecoderAttentionTopology::GroupedQuery
        );
        assert_eq!(
            config(4, 1)
                .decoder_capabilities()
                .unwrap()
                .attention_topology,
            DecoderAttentionTopology::MultiQuery
        );
    }

    #[test]
    fn capability_record_is_versioned_and_semantically_explicit() {
        let capabilities = config(4, 1).decoder_capabilities().unwrap();
        let record = capabilities.canonical_record();
        assert!(record.starts_with("NNIS-DECODER-CAPABILITY-V1\n"));
        assert!(record.contains("attention=multi_query\n"));
        assert!(record.contains("rope=llama_rotate_half_unscaled\n"));
        assert!(record.contains("mlp=swiglu_silu\n"));
        assert!(record.contains("q_heads=4\nkv_heads=1\nhead_dim=16\n"));
    }
}
