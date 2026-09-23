//! Machine-readable qualification evidence for experimental weight representations.
//!
//! Storage, isolated projection and full-model execution are deliberately
//! distinct qualification levels. The validator prevents a representation from
//! being labelled full-model qualified without explicit physical execution and
//! generated-token verification evidence.

use crate::{
    CanonicalWeightDenominatorV1, Int2ReferenceStorageSummaryV1, Int4ReferenceStorageSummaryV1,
    WeightRepresentationAccountingV1, NNIS_INT2_REFERENCE_STORAGE_VERSION,
    NNIS_INT4_REFERENCE_STORAGE_VERSION, NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION,
};
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

fn storage_accounting_from_parts(
    logical_tensor_references: u64,
    logical_element_references: u64,
    unique_source_allocations: u64,
    unique_logical_values: u64,
    source_owned_bytes: u64,
    serialized_bytes: u64,
    resident_bytes: u64,
) -> Result<WeightRepresentationAccountingV1> {
    WeightRepresentationAccountingV1::new(
        CanonicalWeightDenominatorV1 {
            schema_version: NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION,
            logical_tensor_references,
            logical_element_references,
            unique_source_allocations,
            unique_logical_values,
            source_owned_bytes,
        },
        serialized_bytes,
        resident_bytes,
    )
}

/// Convert an exact INT2 storage summary into a storage-only qualification record.
///
/// The builder recomputes bits/value from exact integer byte totals and rejects
/// summaries whose cached ratios disagree with that arithmetic.
pub fn int2_storage_qualification_record_v1(
    exact_checkpoint: impl Into<String>,
    summary: &Int2ReferenceStorageSummaryV1,
) -> Result<WeightRepresentationQualificationRecordV1> {
    if summary.schema_version != NNIS_INT2_REFERENCE_STORAGE_VERSION {
        return Err(NnisError::unsupported(format!(
            "INT2 storage summary schema {}; supported version is {}",
            summary.schema_version, NNIS_INT2_REFERENCE_STORAGE_VERSION
        )));
    }
    if summary.execution_qualified {
        return Err(NnisError::invalid_input(
            "INT2 storage-only qualification refuses a summary marked execution-qualified",
        ));
    }
    let accounting = storage_accounting_from_parts(
        summary.logical_tensor_references,
        summary.logical_element_references,
        summary.unique_source_allocations,
        summary.unique_logical_values,
        summary.source_f32_owned_bytes,
        summary.serialized_total_bytes,
        summary.resident_device_bytes,
    )?;
    if accounting
        .serialized_bits_per_unique_logical_value
        .to_bits()
        != summary.serialized_bits_per_unique_logical_value.to_bits()
        || accounting.resident_bits_per_unique_logical_value.to_bits()
            != summary.resident_bits_per_unique_logical_value.to_bits()
    {
        return Err(NnisError::invalid_input(
            "INT2 storage summary bits/value disagree with exact byte accounting",
        ));
    }
    WeightRepresentationQualificationRecordV1::new(
        WeightRepresentationFamilyV1::Int2Ternary,
        NNIS_INT2_REFERENCE_STORAGE_VERSION,
        exact_checkpoint,
        WeightExecutionQualificationLevelV1::StorageOnly,
        false,
        false,
        summary.max_abs_error,
        summary.mean_squared_error,
        accounting,
    )
}

/// Convert an exact INT4 storage summary into a storage-only qualification record.
pub fn int4_storage_qualification_record_v1(
    exact_checkpoint: impl Into<String>,
    summary: &Int4ReferenceStorageSummaryV1,
) -> Result<WeightRepresentationQualificationRecordV1> {
    if summary.schema_version != NNIS_INT4_REFERENCE_STORAGE_VERSION {
        return Err(NnisError::unsupported(format!(
            "INT4 storage summary schema {}; supported version is {}",
            summary.schema_version, NNIS_INT4_REFERENCE_STORAGE_VERSION
        )));
    }
    if summary.execution_qualified {
        return Err(NnisError::invalid_input(
            "INT4 storage-only qualification refuses a summary marked execution-qualified",
        ));
    }
    let accounting = storage_accounting_from_parts(
        summary.logical_tensor_references,
        summary.logical_element_references,
        summary.unique_source_allocations,
        summary.unique_logical_values,
        summary.source_f32_owned_bytes,
        summary.serialized_total_bytes,
        summary.resident_device_bytes,
    )?;
    if accounting.serialized_bits_per_unique_logical_value.to_bits()
        != summary.serialized_bits_per_unique_logical_value.to_bits()
        || accounting.resident_bits_per_unique_logical_value.to_bits()
            != summary.resident_bits_per_unique_logical_value.to_bits()
    {
        return Err(NnisError::invalid_input(
            "INT4 storage summary bits/value disagree with exact byte accounting",
        ));
    }
    WeightRepresentationQualificationRecordV1::new(
        WeightRepresentationFamilyV1::Int4Symmetric,
        NNIS_INT4_REFERENCE_STORAGE_VERSION,
        exact_checkpoint,
        WeightExecutionQualificationLevelV1::StorageOnly,
        false,
        false,
        summary.max_abs_error,
        summary.mean_squared_error,
        accounting,
    )
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
        if self.exact_checkpoint.is_empty() || self.exact_checkpoint.trim() != self.exact_checkpoint
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

    fn int2_summary() -> Int2ReferenceStorageSummaryV1 {
        Int2ReferenceStorageSummaryV1 {
            schema_version: NNIS_INT2_REFERENCE_STORAGE_VERSION,
            representation: "ternary-int2-reference".to_string(),
            quantization: "test".to_string(),
            scale_scope: "test".to_string(),
            execution_qualified: false,
            logical_tensor_references: 2,
            logical_element_references: 32,
            unique_source_allocations: 1,
            unique_logical_values: 16,
            source_f32_owned_bytes: 64,
            serialized_header_bytes: 16,
            serialized_payload_bytes: 4,
            serialized_total_bytes: 20,
            resident_payload_bytes: 4,
            resident_scale_bytes: 4,
            resident_device_bytes: 8,
            serialized_bits_per_unique_logical_value: 10.0,
            resident_bits_per_unique_logical_value: 4.0,
            source_f32_to_resident_int2_byte_ratio: 8.0,
            max_abs_error: 0.5,
            mean_squared_error: 0.125,
            allocations: Vec::new(),
        }
    }

    fn int4_summary() -> Int4ReferenceStorageSummaryV1 {
        Int4ReferenceStorageSummaryV1 {
            schema_version: NNIS_INT4_REFERENCE_STORAGE_VERSION,
            representation: "symmetric-signed-int4-reference".to_string(),
            quantization: "test".to_string(),
            scale_scope: "test".to_string(),
            quant_min: -7,
            quant_max: 7,
            execution_qualified: false,
            logical_tensor_references: 2,
            logical_element_references: 32,
            unique_source_allocations: 1,
            unique_logical_values: 16,
            source_f32_owned_bytes: 64,
            serialized_header_bytes: 16,
            serialized_payload_bytes: 8,
            serialized_total_bytes: 24,
            resident_payload_bytes: 8,
            resident_scale_bytes: 4,
            resident_device_bytes: 12,
            serialized_bits_per_unique_logical_value: 12.0,
            resident_bits_per_unique_logical_value: 6.0,
            source_f32_to_resident_int4_byte_ratio: 64.0 / 12.0,
            max_abs_error: 0.25,
            mean_squared_error: 0.0625,
            allocations: Vec::new(),
        }
    }

    #[test]
    fn storage_summary_builders_recompute_exact_accounting() {
        let int2 =
            int2_storage_qualification_record_v1("checkpoint/int2", &int2_summary()).unwrap();
        assert_eq!(int2.family, WeightRepresentationFamilyV1::Int2Ternary);
        assert_eq!(
            int2.execution_level,
            WeightExecutionQualificationLevelV1::StorageOnly
        );
        assert_eq!(
            int2.accounting
                .resident_bits_per_unique_logical_value
                .to_bits(),
            4.0_f64.to_bits()
        );

        let int4 =
            int4_storage_qualification_record_v1("checkpoint/int4", &int4_summary()).unwrap();
        assert_eq!(int4.family, WeightRepresentationFamilyV1::Int4Symmetric);
        assert_eq!(
            int4.accounting
                .serialized_bits_per_unique_logical_value
                .to_bits(),
            12.0_f64.to_bits()
        );
    }

    #[test]
    fn storage_summary_builders_reject_cached_ratio_drift_and_overclaim() {
        let mut int2 = int2_summary();
        int2.resident_bits_per_unique_logical_value = 3.0;
        assert!(int2_storage_qualification_record_v1("checkpoint", &int2).is_err());

        let mut int4 = int4_summary();
        int4.execution_qualified = true;
        assert!(int4_storage_qualification_record_v1("checkpoint", &int4).is_err());
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
