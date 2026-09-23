//! Versioned capability manifest for experimental NNIS weight representations.
//!
//! The manifest reports software surfaces that actually exist in the repository.
//! It deliberately keeps full-model execution qualification separate from
//! storage and isolated projection support.

use crate::{
    WeightRepresentationFamilyV1, NNIS_INT2_REFERENCE_PROJECTION_PLAN_VERSION,
    NNIS_INT2_REFERENCE_STORAGE_VERSION, NNIS_INT4_REFERENCE_PROJECTION_PLAN_VERSION,
    NNIS_INT4_REFERENCE_STORAGE_VERSION, NNIS_SPARSE_CSC_REFERENCE_VERSION,
};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Version of the static reference-weight capability manifest.
pub const NNIS_WEIGHT_CAPABILITY_MANIFEST_VERSION: u32 = 1;

/// One representation capability entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightRepresentationCapabilityV1 {
    pub family: WeightRepresentationFamilyV1,
    pub storage_contract_version: u32,
    pub exact_serialization_available: bool,
    pub exact_accounting_available: bool,
    pub isolated_projection_kernel: Option<String>,
    pub projection_plan_contract_version: Option<u32>,
    pub full_model_execution_qualified: bool,
}

/// Static capability set for the fixed baselines currently implemented by NNIS.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightCapabilityManifestV1 {
    pub schema_version: u32,
    pub entries: Vec<WeightRepresentationCapabilityV1>,
}

impl WeightCapabilityManifestV1 {
    /// Validate uniqueness and claim-boundary coherence.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_WEIGHT_CAPABILITY_MANIFEST_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight capability manifest schema {}; supported version is {}",
                self.schema_version, NNIS_WEIGHT_CAPABILITY_MANIFEST_VERSION
            )));
        }
        if self.entries.is_empty() {
            return Err(NnisError::invalid_input(
                "weight capability manifest must contain at least one entry",
            ));
        }

        let mut families = BTreeSet::new();
        for entry in &self.entries {
            let family_key = match entry.family {
                WeightRepresentationFamilyV1::Int4Symmetric => 0_u8,
                WeightRepresentationFamilyV1::Int2Ternary => 1_u8,
                WeightRepresentationFamilyV1::MagnitudeSparse => 2_u8,
            };
            if !families.insert(family_key) {
                return Err(NnisError::invalid_input(
                    "weight capability manifest contains a duplicate representation family",
                ));
            }
            if entry.storage_contract_version == 0 {
                return Err(NnisError::invalid_input(
                    "weight capability storage contract version must be non-zero",
                ));
            }
            if !entry.exact_serialization_available || !entry.exact_accounting_available {
                return Err(NnisError::invalid_input(
                    "reference weight capability requires exact serialization and accounting",
                ));
            }
            if let Some(kernel) = &entry.isolated_projection_kernel {
                if kernel.is_empty() || kernel.trim() != kernel {
                    return Err(NnisError::invalid_input(
                        "weight capability kernel identity must be non-empty and trimmed",
                    ));
                }
            }
            if entry
                .projection_plan_contract_version
                .is_some_and(|version| version == 0)
            {
                return Err(NnisError::invalid_input(
                    "weight capability projection-plan version must be non-zero",
                ));
            }
            if entry.full_model_execution_qualified
                && (entry.isolated_projection_kernel.is_none()
                    || entry.projection_plan_contract_version.is_none())
            {
                return Err(NnisError::invalid_input(
                    "full-model qualification requires an isolated kernel and projection-plan contract",
                ));
            }
        }
        Ok(())
    }
}

/// Return the software capabilities proven by the current reference surfaces.
///
/// All three fixed baselines have exact serialization/accounting and an isolated
/// projection kernel. INT4 and INT2 also have explicit projection-plan
/// contracts. None is marked full-model execution qualified here.
pub fn reference_weight_capability_manifest_v1() -> WeightCapabilityManifestV1 {
    WeightCapabilityManifestV1 {
        schema_version: NNIS_WEIGHT_CAPABILITY_MANIFEST_VERSION,
        entries: vec![
            WeightRepresentationCapabilityV1 {
                family: WeightRepresentationFamilyV1::Int4Symmetric,
                storage_contract_version: NNIS_INT4_REFERENCE_STORAGE_VERSION,
                exact_serialization_available: true,
                exact_accounting_available: true,
                isolated_projection_kernel: Some("F32Int4Gemv".to_string()),
                projection_plan_contract_version: Some(NNIS_INT4_REFERENCE_PROJECTION_PLAN_VERSION),
                full_model_execution_qualified: false,
            },
            WeightRepresentationCapabilityV1 {
                family: WeightRepresentationFamilyV1::Int2Ternary,
                storage_contract_version: NNIS_INT2_REFERENCE_STORAGE_VERSION,
                exact_serialization_available: true,
                exact_accounting_available: true,
                isolated_projection_kernel: Some("F32Int2Gemv".to_string()),
                projection_plan_contract_version: Some(NNIS_INT2_REFERENCE_PROJECTION_PLAN_VERSION),
                full_model_execution_qualified: false,
            },
            WeightRepresentationCapabilityV1 {
                family: WeightRepresentationFamilyV1::MagnitudeSparse,
                storage_contract_version: NNIS_SPARSE_CSC_REFERENCE_VERSION,
                exact_serialization_available: true,
                exact_accounting_available: true,
                isolated_projection_kernel: Some("F32SparseCscGemv".to_string()),
                projection_plan_contract_version: None,
                full_model_execution_qualified: false,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_manifest_is_versioned_unique_and_claim_conservative() {
        let manifest = reference_weight_capability_manifest_v1();
        manifest.validate().unwrap();
        assert_eq!(manifest.entries.len(), 3);
        assert!(manifest
            .entries
            .iter()
            .all(|entry| !entry.full_model_execution_qualified));
        assert!(manifest
            .entries
            .iter()
            .all(|entry| entry.isolated_projection_kernel.is_some()));

        let json = serde_json::to_string(&manifest).unwrap();
        let decoded: WeightCapabilityManifestV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, manifest);
    }

    #[test]
    fn duplicate_and_overclaimed_capabilities_fail_closed() {
        let mut manifest = reference_weight_capability_manifest_v1();
        manifest.entries.push(manifest.entries[0].clone());
        assert!(manifest.validate().is_err());

        let mut manifest = reference_weight_capability_manifest_v1();
        let sparse = manifest
            .entries
            .iter_mut()
            .find(|entry| entry.family == WeightRepresentationFamilyV1::MagnitudeSparse)
            .unwrap();
        sparse.full_model_execution_qualified = true;
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn malformed_kernel_and_versions_fail_closed() {
        let mut manifest = reference_weight_capability_manifest_v1();
        manifest.entries[0].isolated_projection_kernel = Some(" F32Int4Gemv".to_string());
        assert!(manifest.validate().is_err());

        let mut manifest = reference_weight_capability_manifest_v1();
        manifest.entries[0].storage_contract_version = 0;
        assert!(manifest.validate().is_err());
    }
}
