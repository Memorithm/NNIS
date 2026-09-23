use crate::{Activation, DecoderExecutionCapabilities, ModelConfig, WeightDType};
use nnis_rt::{NnisError, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub const NNIS_EXACT_DECODER_CHECKPOINT_SPEC_VERSION: u32 = 1;

const SMOLLM2_135M_CAPABILITY_RECORD: &str = concat!(
    "NNIS-DECODER-CAPABILITY-V1\n",
    "attention=grouped_query\n",
    "rope=llama_rotate_half_unscaled\n",
    "mlp=swiglu_silu\n",
    "weight_dtype=bf16\n",
    "q_heads=9\n",
    "kv_heads=3\n",
    "head_dim=64\n",
);

const TINYLLAMA_1P1B_CHAT_CAPABILITY_RECORD: &str = concat!(
    "NNIS-DECODER-CAPABILITY-V1\n",
    "attention=grouped_query\n",
    "rope=llama_rotate_half_unscaled\n",
    "mlp=swiglu_silu\n",
    "weight_dtype=bf16\n",
    "q_heads=32\n",
    "kv_heads=4\n",
    "head_dim=64\n",
);

/// Exact external checkpoint identity plus the decoder configuration that NNIS
/// is allowed to associate with that checkpoint.
///
/// A spec is deliberately narrower than a model-family capability declaration.
/// Matching a spec does not establish tokenizer parity, generation quality,
/// performance, or support for sibling checkpoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExactDecoderCheckpointSpec {
    pub contract_version: u32,
    pub name: &'static str,
    pub source_repo: &'static str,
    pub source_revision: &'static str,
    pub source_model_sha256: &'static str,
    pub vocab_size: usize,
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
    pub expected_capability_record: &'static str,
}

impl ExactDecoderCheckpointSpec {
    pub fn expected_config(&self) -> ModelConfig {
        ModelConfig {
            vocab_size: self.vocab_size,
            eos_token_id: self.eos_token_id,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            activation: self.activation,
            weight_dtype: self.weight_dtype,
        }
    }

    /// Validate one already-loaded decoder configuration against this exact
    /// checkpoint identity and return the versioned execution-capability profile.
    pub fn validate_config(&self, config: &ModelConfig) -> Result<DecoderExecutionCapabilities> {
        if self.contract_version != NNIS_EXACT_DECODER_CHECKPOINT_SPEC_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported exact checkpoint spec version {} for {}",
                self.contract_version, self.name
            )));
        }
        config.validate_execution_support()?;
        let expected = self.expected_config();
        let geometry_matches = config.vocab_size == expected.vocab_size
            && config.eos_token_id == expected.eos_token_id
            && config.hidden_size == expected.hidden_size
            && config.intermediate_size == expected.intermediate_size
            && config.num_hidden_layers == expected.num_hidden_layers
            && config.num_attention_heads == expected.num_attention_heads
            && config.num_key_value_heads == expected.num_key_value_heads
            && config.max_position_embeddings == expected.max_position_embeddings
            && config.rms_norm_eps.to_bits() == expected.rms_norm_eps.to_bits()
            && config.rope_theta.to_bits() == expected.rope_theta.to_bits()
            && config.activation == expected.activation
            && config.weight_dtype == expected.weight_dtype;
        if !geometry_matches {
            return Err(NnisError::invalid_input(format!(
                "loaded model config does not match exact checkpoint spec {}: got {config:?}, expected {expected:?}",
                self.name
            )));
        }

        let capabilities = config.decoder_capabilities()?;
        let actual_record = capabilities.canonical_record();
        if actual_record != self.expected_capability_record {
            return Err(NnisError::invalid_input(format!(
                "decoder capability record for {} drifted: got {actual_record:?}, expected {:?}",
                self.name, self.expected_capability_record
            )));
        }
        Ok(capabilities)
    }

    /// Verify one local model payload against the exact checkpoint SHA-256.
    ///
    /// The file is hashed incrementally and is never read wholly into memory.
    /// This verifies bytes only; repository/revision provenance remains a
    /// separate part of the exact checkpoint identity contract.
    pub fn verify_model_file_sha256(&self, path: impl AsRef<Path>) -> Result<String> {
        if self.source_model_sha256.len() != 64
            || !self
                .source_model_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(NnisError::invalid_input(format!(
                "exact checkpoint spec {} contains an invalid source model SHA-256",
                self.name
            )));
        }

        let path = path.as_ref();
        let mut file = File::open(path)
            .map_err(|error| NnisError::io("open exact checkpoint model payload", error))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 1024 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|error| NnisError::io("hash exact checkpoint model payload", error))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let actual = format!("{:x}", hasher.finalize());
        if actual != self.source_model_sha256 {
            return Err(NnisError::invalid_input(format!(
                "model payload SHA-256 mismatch for {}: got {actual}, expected {}",
                self.name, self.source_model_sha256
            )));
        }
        Ok(actual)
    }

    /// Compact deterministic key for JSON artifacts and qualification joins.
    ///
    /// The digest covers the complete canonical identity record, including
    /// repository, revision, model SHA-256, geometry and numeric policy.
    pub fn evidence_key(&self) -> String {
        let canonical = self.canonical_identity();
        let digest = Sha256::digest(canonical.as_bytes());
        format!("{}@sha256:{digest:x}", self.name)
    }

    /// Deterministic source/config identity for provenance and evidence keys.
    /// This record is not an authentication primitive.
    pub fn canonical_identity(&self) -> String {
        format!(
            concat!(
                "NNIS-EXACT-DECODER-CHECKPOINT-SPEC-V{}\n",
                "name={}\n",
                "repo={}\n",
                "revision={}\n",
                "model_sha256={}\n",
                "vocab_size={}\n",
                "eos_token_id={}\n",
                "hidden_size={}\n",
                "intermediate_size={}\n",
                "layers={}\n",
                "q_heads={}\n",
                "kv_heads={}\n",
                "max_positions={}\n",
                "rms_norm_eps_bits={}\n",
                "rope_theta_bits={}\n",
                "activation=silu\n",
                "weight_dtype={}\n"
            ),
            self.contract_version,
            self.name,
            self.source_repo,
            self.source_revision,
            self.source_model_sha256,
            self.vocab_size,
            self.eos_token_id
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
            self.rms_norm_eps.to_bits(),
            self.rope_theta.to_bits(),
            match self.weight_dtype {
                WeightDType::F32 => "f32",
                WeightDType::Bf16 => "bf16",
            }
        )
    }
}

pub const SMOLLM2_135M_BF16: ExactDecoderCheckpointSpec = ExactDecoderCheckpointSpec {
    contract_version: NNIS_EXACT_DECODER_CHECKPOINT_SPEC_VERSION,
    name: "smollm2-135m-bf16",
    source_repo: "HuggingFaceTB/SmolLM2-135M",
    source_revision: "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
    source_model_sha256: "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1",
    vocab_size: 49_152,
    eos_token_id: Some(0),
    hidden_size: 576,
    intermediate_size: 1_536,
    num_hidden_layers: 30,
    num_attention_heads: 9,
    num_key_value_heads: 3,
    max_position_embeddings: 8_192,
    rms_norm_eps: 1.0e-5,
    rope_theta: 100_000.0,
    activation: Activation::Silu,
    weight_dtype: WeightDType::Bf16,
    expected_capability_record: SMOLLM2_135M_CAPABILITY_RECORD,
};

pub const TINYLLAMA_1P1B_CHAT_BF16: ExactDecoderCheckpointSpec = ExactDecoderCheckpointSpec {
    contract_version: NNIS_EXACT_DECODER_CHECKPOINT_SPEC_VERSION,
    name: "tinyllama-1.1b-chat-v1.0-bf16",
    source_repo: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    source_revision: "d9128824c0c80111be21424e68086f52413fb413",
    source_model_sha256: "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
    vocab_size: 32_000,
    eos_token_id: Some(2),
    hidden_size: 2_048,
    intermediate_size: 5_632,
    num_hidden_layers: 22,
    num_attention_heads: 32,
    num_key_value_heads: 4,
    max_position_embeddings: 2_048,
    rms_norm_eps: 1.0e-5,
    rope_theta: 10_000.0,
    activation: Activation::Silu,
    weight_dtype: WeightDType::Bf16,
    expected_capability_record: TINYLLAMA_1P1B_CHAT_CAPABILITY_RECORD,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DecoderAttentionTopology;

    #[test]
    fn local_model_payload_sha256_verification_is_streaming_and_fail_closed() {
        let path = std::env::temp_dir().join(format!(
            "nnis-exact-checkpoint-sha-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, b"abc").unwrap();

        let mut spec = SMOLLM2_135M_BF16;
        spec.source_model_sha256 =
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(
            spec.verify_model_file_sha256(&path).unwrap(),
            spec.source_model_sha256
        );

        spec.source_model_sha256 =
            "0000000000000000000000000000000000000000000000000000000000000000";
        assert!(spec.verify_model_file_sha256(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn smollm2_exact_spec_accepts_only_its_decoder_config() {
        let config = SMOLLM2_135M_BF16.expected_config();
        let capabilities = SMOLLM2_135M_BF16.validate_config(&config).unwrap();
        assert_eq!(
            capabilities.attention_topology,
            DecoderAttentionTopology::GroupedQuery
        );
        assert_eq!(
            capabilities.canonical_record(),
            SMOLLM2_135M_CAPABILITY_RECORD
        );

        let mut drifted = config;
        drifted.rope_theta = 10_000.0;
        assert!(SMOLLM2_135M_BF16.validate_config(&drifted).is_err());
    }

    #[test]
    fn tinyllama_exact_spec_freezes_the_existing_campaign_checkpoint() {
        let config = TINYLLAMA_1P1B_CHAT_BF16.expected_config();
        let capabilities = TINYLLAMA_1P1B_CHAT_BF16.validate_config(&config).unwrap();
        assert_eq!(
            capabilities.attention_topology,
            DecoderAttentionTopology::GroupedQuery
        );
        assert_eq!(
            capabilities.canonical_record(),
            TINYLLAMA_1P1B_CHAT_CAPABILITY_RECORD
        );
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.key_value_width().unwrap(), 256);
    }

    #[test]
    fn evidence_key_is_compact_deterministic_and_spec_sensitive() {
        let first = SMOLLM2_135M_BF16.evidence_key();
        let second = SMOLLM2_135M_BF16.evidence_key();
        assert_eq!(first, second);
        assert!(first.starts_with("smollm2-135m-bf16@sha256:"));
        let digest = first.split_once("@sha256:").unwrap().1;
        assert_eq!(digest.len(), 64);
        assert!(digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));

        assert_ne!(first, TINYLLAMA_1P1B_CHAT_BF16.evidence_key());
    }

    #[test]
    fn exact_checkpoint_identity_is_versioned_and_source_pinned() {
        let identity = TINYLLAMA_1P1B_CHAT_BF16.canonical_identity();
        assert!(identity.starts_with("NNIS-EXACT-DECODER-CHECKPOINT-SPEC-V1\n"));
        assert!(identity.contains("repo=TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"));
        assert!(identity.contains(
            "model_sha256=6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933\n"
        ));
    }
}
