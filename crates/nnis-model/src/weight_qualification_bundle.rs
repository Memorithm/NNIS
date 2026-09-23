//! Multi-representation qualification bundle for one exact checkpoint.
//!
//! This bundles independently validated representation records without
//! weakening their claim boundaries. The Elastic Stage-B gate is deliberately
//! strict: INT4, INT2 and sparse must all carry FullModel qualification for the
//! same checkpoint before the bundle is considered consumable.

use crate::{
    WeightExecutionQualificationLevelV1, WeightRepresentationFamilyV1,
    WeightRepresentationQualificationRecordV1,
};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Version of the multi-family qualification bundle.
pub const NNIS_WEIGHT_QUALIFICATION_BUNDLE_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightQualificationBundleV1 {
    pub schema_version: u32,
    pub exact_checkpoint: String,
    pub records: Vec<WeightRepresentationQualificationRecordV1>,
}

impl WeightQualificationBundleV1 {
    pub fn new(
        exact_checkpoint: impl Into<String>,
        records: Vec<WeightRepresentationQualificationRecordV1>,
    ) -> Result<Self> {
        let bundle = Self {
            schema_version: NNIS_WEIGHT_QUALIFICATION_BUNDLE_VERSION,
            exact_checkpoint: exact_checkpoint.into(),
            records,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_WEIGHT_QUALIFICATION_BUNDLE_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight qualification bundle schema {}; supported version is {}",
                self.schema_version, NNIS_WEIGHT_QUALIFICATION_BUNDLE_VERSION
            )));
        }
        if self.exact_checkpoint.is_empty() || self.exact_checkpoint.trim() != self.exact_checkpoint
        {
            return Err(NnisError::invalid_input(
                "weight qualification bundle checkpoint must be non-empty and trimmed",
            ));
        }
        if self.records.is_empty() {
            return Err(NnisError::invalid_input(
                "weight qualification bundle requires at least one record",
            ));
        }

        let mut families = BTreeSet::new();
        for record in &self.records {
            record.validate()?;
            if record.exact_checkpoint != self.exact_checkpoint {
                return Err(NnisError::invalid_input(format!(
                    "weight qualification record checkpoint {:?} does not match bundle checkpoint {:?}",
                    record.exact_checkpoint, self.exact_checkpoint
                )));
            }
            let family_key = match record.family {
                WeightRepresentationFamilyV1::Int4Symmetric => 0_u8,
                WeightRepresentationFamilyV1::Int2Ternary => 1_u8,
                WeightRepresentationFamilyV1::MagnitudeSparse => 2_u8,
            };
            if !families.insert(family_key) {
                return Err(NnisError::invalid_input(
                    "weight qualification bundle contains duplicate representation families",
                ));
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn record(
        &self,
        family: WeightRepresentationFamilyV1,
    ) -> Option<&WeightRepresentationQualificationRecordV1> {
        self.records.iter().find(|record| record.family == family)
    }

    /// Return success only when all fixed Stage-B families are full-model
    /// qualified for this exact checkpoint.
    pub fn validate_elastic_stage_b_readiness(&self) -> Result<()> {
        self.validate()?;
        for family in [
            WeightRepresentationFamilyV1::Int4Symmetric,
            WeightRepresentationFamilyV1::Int2Ternary,
            WeightRepresentationFamilyV1::MagnitudeSparse,
        ] {
            let record = self.record(family).ok_or_else(|| {
                NnisError::invalid_input(format!(
                    "Elastic Stage-B readiness is missing {family:?} qualification"
                ))
            })?;
            if record.execution_level != WeightExecutionQualificationLevelV1::FullModel {
                return Err(NnisError::unsupported(format!(
                    "Elastic Stage-B readiness requires FullModel {family:?} qualification; found {:?}",
                    record.execution_level
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CanonicalWeightDenominatorV1, WeightRepresentationAccountingV1,
        NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION,
    };

    fn accounting() -> WeightRepresentationAccountingV1 {
        WeightRepresentationAccountingV1::new(
            CanonicalWeightDenominatorV1 {
                schema_version: NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION,
                logical_tensor_references: 1,
                logical_element_references: 16,
                unique_source_allocations: 1,
                unique_logical_values: 16,
                source_owned_bytes: 64,
            },
            16,
            16,
        )
        .unwrap()
    }

    fn record(
        family: WeightRepresentationFamilyV1,
        level: WeightExecutionQualificationLevelV1,
    ) -> WeightRepresentationQualificationRecordV1 {
        let (physical, tokens) = match level {
            WeightExecutionQualificationLevelV1::StorageOnly => (false, false),
            WeightExecutionQualificationLevelV1::IsolatedProjection => (true, false),
            WeightExecutionQualificationLevelV1::FullModel => (true, true),
        };
        WeightRepresentationQualificationRecordV1::new(
            family,
            1,
            "smollm2@sha256:test",
            level,
            physical,
            tokens,
            0.0,
            0.0,
            accounting(),
        )
        .unwrap()
    }

    #[test]
    fn bundle_rejects_duplicate_families_and_mixed_checkpoints() {
        let first = record(
            WeightRepresentationFamilyV1::Int4Symmetric,
            WeightExecutionQualificationLevelV1::StorageOnly,
        );
        assert!(WeightQualificationBundleV1::new(
            "smollm2@sha256:test",
            vec![first.clone(), first]
        )
        .is_err());

        let mut mixed = record(
            WeightRepresentationFamilyV1::Int2Ternary,
            WeightExecutionQualificationLevelV1::StorageOnly,
        );
        mixed.exact_checkpoint = "other@sha256:test".to_string();
        assert!(WeightQualificationBundleV1::new("smollm2@sha256:test", vec![mixed]).is_err());
    }

    #[test]
    fn stage_b_gate_requires_all_three_full_model_records() {
        let storage_bundle = WeightQualificationBundleV1::new(
            "smollm2@sha256:test",
            vec![
                record(
                    WeightRepresentationFamilyV1::Int4Symmetric,
                    WeightExecutionQualificationLevelV1::StorageOnly,
                ),
                record(
                    WeightRepresentationFamilyV1::Int2Ternary,
                    WeightExecutionQualificationLevelV1::FullModel,
                ),
                record(
                    WeightRepresentationFamilyV1::MagnitudeSparse,
                    WeightExecutionQualificationLevelV1::FullModel,
                ),
            ],
        )
        .unwrap();
        assert!(storage_bundle.validate_elastic_stage_b_readiness().is_err());

        let ready = WeightQualificationBundleV1::new(
            "smollm2@sha256:test",
            vec![
                record(
                    WeightRepresentationFamilyV1::Int4Symmetric,
                    WeightExecutionQualificationLevelV1::FullModel,
                ),
                record(
                    WeightRepresentationFamilyV1::Int2Ternary,
                    WeightExecutionQualificationLevelV1::FullModel,
                ),
                record(
                    WeightRepresentationFamilyV1::MagnitudeSparse,
                    WeightExecutionQualificationLevelV1::FullModel,
                ),
            ],
        )
        .unwrap();
        ready.validate_elastic_stage_b_readiness().unwrap();
    }
}
