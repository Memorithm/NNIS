//! Canonical generated-token evidence for full-model qualification.
//!
//! Token ids are hashed from a versioned binary envelope so Python/Rust
//! harnesses can compare one stable representation without formatting drift.

use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Version of generated-token verification evidence.
pub const NNIS_GENERATED_TOKEN_EVIDENCE_VERSION: u32 = 1;

const DOMAIN: &[u8] = b"NNIS-GENERATED-TOKENS-V1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratedTokenEvidenceV1 {
    pub schema_version: u32,
    pub token_count: u64,
    pub token_ids_sha256: String,
}

impl GeneratedTokenEvidenceV1 {
    /// Build a deterministic evidence record from emitted token ids.
    pub fn from_token_ids(token_ids: &[u32]) -> Result<Self> {
        if token_ids.is_empty() {
            return Err(NnisError::invalid_input(
                "generated-token evidence requires at least one token",
            ));
        }
        let token_count = u64::try_from(token_ids.len())
            .map_err(|_| NnisError::invalid_input("generated token count exceeds u64"))?;
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN);
        hasher.update(token_count.to_le_bytes());
        for token in token_ids {
            hasher.update(token.to_le_bytes());
        }
        let token_ids_sha256 = format!("{:x}", hasher.finalize());
        let evidence = Self {
            schema_version: NNIS_GENERATED_TOKEN_EVIDENCE_VERSION,
            token_count,
            token_ids_sha256,
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_GENERATED_TOKEN_EVIDENCE_VERSION {
            return Err(NnisError::unsupported(format!(
                "generated-token evidence schema {}; supported version is {}",
                self.schema_version, NNIS_GENERATED_TOKEN_EVIDENCE_VERSION
            )));
        }
        if self.token_count == 0 {
            return Err(NnisError::invalid_input(
                "generated-token evidence requires a non-zero token count",
            ));
        }
        if self.token_ids_sha256.len() != 64
            || !self
                .token_ids_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(NnisError::invalid_input(
                "generated-token SHA-256 must be 64 lowercase hexadecimal characters",
            ));
        }
        Ok(())
    }
}

/// Fail closed if a runtime output vector contains NaN or infinity.
pub fn validate_finite_runtime_output(label: &str, values: &[f32]) -> Result<()> {
    if label.is_empty() || label.trim() != label {
        return Err(NnisError::invalid_input(
            "runtime output label must be non-empty and trimmed",
        ));
    }
    if values.is_empty() {
        return Err(NnisError::invalid_input(format!(
            "runtime output {label:?} is empty"
        )));
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(NnisError::invalid_input(format!(
            "runtime output {label:?} contains a non-finite value at index {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_hash_is_deterministic_versioned_and_order_sensitive() {
        let a = GeneratedTokenEvidenceV1::from_token_ids(&[1, 2, 3, 4]).unwrap();
        let b = GeneratedTokenEvidenceV1::from_token_ids(&[1, 2, 3, 4]).unwrap();
        let c = GeneratedTokenEvidenceV1::from_token_ids(&[4, 3, 2, 1]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a.token_ids_sha256, c.token_ids_sha256);
        assert_eq!(a.token_count, 4);
        a.validate().unwrap();

        let encoded = serde_json::to_string(&a).unwrap();
        let decoded: GeneratedTokenEvidenceV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, a);
    }

    #[test]
    fn token_evidence_rejects_empty_and_malformed_hashes() {
        assert!(GeneratedTokenEvidenceV1::from_token_ids(&[]).is_err());

        let mut evidence = GeneratedTokenEvidenceV1::from_token_ids(&[7]).unwrap();
        evidence.token_ids_sha256 = "ABC".to_string();
        assert!(evidence.validate().is_err());
    }

    #[test]
    fn finite_runtime_output_validation_fails_closed() {
        validate_finite_runtime_output("logits", &[0.0, -1.0, 3.5]).unwrap();
        assert!(validate_finite_runtime_output("logits", &[]).is_err());
        assert!(validate_finite_runtime_output("logits", &[f32::NAN]).is_err());
        assert!(validate_finite_runtime_output("logits", &[f32::INFINITY]).is_err());
        assert!(validate_finite_runtime_output(" logits", &[0.0]).is_err());
    }
}
