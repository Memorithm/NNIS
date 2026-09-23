//! Machine-readable qualification evidence for experimental weight representations.
//!
//! Storage, isolated projection and full-model execution are deliberately
//! distinct qualification levels. The validator prevents a representation from
//! being labelled full-model qualified without explicit physical execution and
//! generated-token verification evidence.

use crate::WeightRepresentationAccountingV1;
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of the representation qualification evidence contract.
pub const NNIS_WEIGHT_REPRESENTATION_QUALIFICATION_VERSION: u32 = 1;

/// Fixed research families currently tracked by NNIS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightRepresentationFamilyV1 {
    Int4Symmetric,
    Int2Ternary,
    MagnitudeSparse,
}

/// Highest execution boundary demonstrated by the attached evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightExecutionQualificationLevelV1 {
    StorageOnly,
    IsolatedProjection,
    FullModel,
}

/// Versioned evidence record suitable for JSON artifacts and CI gates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightRepresentationQualificationRecordV1 {
    pub schema_version: u32,
    pub family: WeightRepresentationFamilyV1,
    pub representation_version: u32,
    pub exact_checkpoint: String,
    pub execution_level: WeightExecutionQualificationLevelV1,
    pub physical_execution_observed: bool,
    pub generated_tokens_verified: bool,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
    pub accounting: WeightRepresentationAccountingV1,
}

impl WeightRepresentationQualificationRecordV1 {
    /// Build and validate one immutable qualification record.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        family: WeightRepresentationFamilyV1,
        representation_version: u32,
        exact_checkpoint: impl Into<String>,
        execution_level: WeightExecutionQualificationLevelV1,
        physical_execution_observed: bool,
        generated_tokens_verified: bool,
        max_abs_error: f32,
        mean_squared_error: f64,
        accounting: WeightRepresentationAccountingV1,
    ) -> Result<Self> {
        let record = Self {
            schema_version: NNIS_WEIGHT_REPRESENTATION_QUALIFICATION_VERSION,
            family,
            representation_version,
            exact_checkpoint: exact_checkpoint.into(),
            execution_level,
            physical_execution_observed,
            generated_tokens_verified,
            max_abs_error,
            mean_squared_error,
            accounting,
        };
        record.validate()?;
        Ok(record)
    }

    /// Validate claim boundaries and exact accounting before evidence is used.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_WEIGHT_REPRESENTATION_QUALIFICATION_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight qualification schema {}; supported version is {}",
                self.schema_version, NNIS_WEIGHT_REPRESENTATION_QUALIFICATION_VERSION
            )));
        }
        if self.representation_version == 0 {
            return Err(NnisError::invalid_input(
                "weight representation version must be non-zero",
            ));
        }
        if self.exact_checkpoint.is_empty()
            || self.exact_checkpoint.trim() != self.exact_checkpoint
        {
            return Err(NnisError::invalid_input(
                "weight qualification checkpoint identity must be non-empty and trimmed",
            ));
        }
        if !self.max_abs_error.is_finite()
            || !self.mean_squared_error.is_finite()
            || self.max_abs_error < 0.0
            || self.mean_squared_error < 0.0
        {
            return Err(NnisError::invalid_input(
                "weight qualification reconstruction metrics are invalid",
            ));
        }
        self.accounting.validate()?;

        match self.execution_level {
            WeightExecutionQualificationLevelV1::StorageOnly => {
                if self.physical_execution_observed || self.generated_tokens_verified {
                    return Err(NnisError::invalid_input(
                        "storage-only qualification cannot claim execution or token verification",
                    ));
                }
            }
            WeightExecutionQualificationLevelV1::IsolatedProjection => {
                if !self.physical_execution_observed {
                    return Err(NnisError::invalid_input(
                        "isolated-projection qualification requires physical execution evidence",
                    ));
                }
                if self.generated_tokens_verified {
                    return Err(NnisError::invalid_input(
                        "isolated-projection qualification cannot claim generated-token verification",
                    ));
                }
            }
            WeightExecutionQualificationLevelV1::FullModel => {
                if !self.physical_execution_observed || !self.generated_tokens_verified {
                    return Err(NnisError::invalid_input(
                        "full-model qualification requires physical execution and generated-token verification",
                    ));
                }
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
                logical_tensor_references: 2,
                logical_element_references: 32,
                unique_source_allocations: 1,
                unique_logical_values: 16,
                source_owned_bytes: 64,
            },
            8,
            12,
        )
        .unwrap()
    }

    #[test]
    fn storage_only_record_is_machine_readable_and_fail_closed() {
        let record = WeightRepresentationQualificationRecordV1::new(
            WeightRepresentationFamilyV1::Int2Ternary,
            1,
            "fixture/tiny-decoder@sha256:test",
            WeightExecutionQualificationLevelV1::StorageOnly,
            false,
            false,
            0.5,
            0.125,
            accounting(),
        )
        .unwrap();
        record.validate().unwrap();

        let json = serde_json::to_string(&record).unwrap();
        let decoded: WeightRepresentationQualificationRecordV1 =
            serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, record);
    }

    #[test]
    fn isolated_projection_requires_physical_execution_and_not_tokens() {
        assert!(WeightRepresentationQualificationRecordV1::new(
            WeightRepresentationFamilyV1::Int4Symmetric,
            1,
            "fixture",
            WeightExecutionQualificationLevelV1::IsolatedProjection,
            false,
            false,
            0.0,
            0.0,
            accounting(),
        )
        .is_err());
        assert!(WeightRepresentationQualificationRecordV1::new(
            WeightRepresentationFamilyV1::Int4Symmetric,
            1,
            "fixture",
            WeightExecutionQualificationLevelV1::IsolatedProjection,
            true,
            true,
            0.0,
            0.0,
            accounting(),
        )
        .is_err());
    }

    #[test]
    fn full_model_claim_requires_execution_and_generated_token_verification() {
        for (physical, tokens) in [(false, false), (true, false), (false, true)] {
            assert!(WeightRepresentationQualificationRecordV1::new(
                WeightRepresentationFamilyV1::MagnitudeSparse,
                1,
                "fixture",
                WeightExecutionQualificationLevelV1::FullModel,
                physical,
                tokens,
                0.0,
                0.0,
                accounting(),
            )
            .is_err());
        }
        let record = WeightRepresentationQualificationRecordV1::new(
            WeightRepresentationFamilyV1::MagnitudeSparse,
            1,
            "fixture",
            WeightExecutionQualificationLevelV1::FullModel,
            true,
            true,
            0.0,
            0.0,
            accounting(),
        )
        .unwrap();
        record.validate().unwrap();
    }

    #[test]
    fn malformed_identity_metrics_and_accounting_fail_closed() {
        assert!(WeightRepresentationQualificationRecordV1::new(
            WeightRepresentationFamilyV1::Int2Ternary,
            0,
            "fixture",
            WeightExecutionQualificationLevelV1::StorageOnly,
            false,
            false,
            0.0,
            0.0,
            accounting(),
        )
        .is_err());
        assert!(WeightRepresentationQualificationRecordV1::new(
            WeightRepresentationFamilyV1::Int2Ternary,
            1,
            " fixture",
            WeightExecutionQualificationLevelV1::StorageOnly,
            false,
            false,
            f32::NAN,
            0.0,
            accounting(),
        )
        .is_err());
    }
}
