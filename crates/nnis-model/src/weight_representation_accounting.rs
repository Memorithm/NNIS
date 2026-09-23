//! Exact representation accounting over the canonical NNIS weight denominator.
//!
//! This module freezes the denominator used by low-bit and structural weight
//! experiments: unique logical values are counted once per unique owned device
//! allocation, while logical aliases remain visible through the source summary.
//! Representation-specific code supplies exact serialized and resident bytes;
//! this module derives comparable bits/value without relying on CUDA free-memory
//! deltas.

use crate::{WeightAllocationSummaryV1, NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of the generic representation-accounting contract.
pub const NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION: u32 = 1;

/// Canonical source denominator derived from an NNIS model weight graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalWeightDenominatorV1 {
    pub schema_version: u32,
    pub logical_tensor_references: u64,
    pub logical_element_references: u64,
    pub unique_source_allocations: u64,
    pub unique_logical_values: u64,
    pub source_owned_bytes: u64,
}

/// Comparable exact accounting for one representation over one denominator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightRepresentationAccountingV1 {
    pub schema_version: u32,
    pub denominator: CanonicalWeightDenominatorV1,
    pub serialized_bytes: u64,
    pub resident_bytes: u64,
    pub serialized_bits_per_unique_logical_value: f64,
    pub resident_bits_per_unique_logical_value: f64,
}

impl CanonicalWeightDenominatorV1 {
    /// Validate and freeze the canonical denominator from the source allocation
    /// summary. Aliased logical tensors therefore never inflate the denominator.
    pub fn from_weight_allocation_summary(source: &WeightAllocationSummaryV1) -> Result<Self> {
        validate_source_summary(source)?;
        Ok(Self {
            schema_version: NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION,
            logical_tensor_references: source.logical_tensor_references,
            logical_element_references: source.logical_element_references,
            unique_source_allocations: source.unique_device_allocations,
            unique_logical_values: source.unique_device_elements,
            source_owned_bytes: source.owned_device_allocation_bytes,
        })
    }
}

impl WeightRepresentationAccountingV1 {
    /// Build comparable exact storage ratios from integer byte counts.
    pub fn new(
        denominator: CanonicalWeightDenominatorV1,
        serialized_bytes: u64,
        resident_bytes: u64,
    ) -> Result<Self> {
        validate_denominator(&denominator)?;
        if serialized_bytes == 0 {
            return Err(NnisError::invalid_input(
                "representation serialized byte count must be non-zero",
            ));
        }
        if resident_bytes == 0 {
            return Err(NnisError::invalid_input(
                "representation resident byte count must be non-zero",
            ));
        }

        let values = denominator.unique_logical_values as f64;
        let serialized_bits_per_unique_logical_value = serialized_bytes as f64 * 8.0 / values;
        let resident_bits_per_unique_logical_value = resident_bytes as f64 * 8.0 / values;
        if !serialized_bits_per_unique_logical_value.is_finite()
            || !resident_bits_per_unique_logical_value.is_finite()
        {
            return Err(NnisError::invalid_input(
                "representation bits/value accounting is not finite",
            ));
        }

        Ok(Self {
            schema_version: NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION,
            denominator,
            serialized_bytes,
            resident_bytes,
            serialized_bits_per_unique_logical_value,
            resident_bits_per_unique_logical_value,
        })
    }

    /// Revalidate the immutable arithmetic contract.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight representation accounting schema {}; supported version is {}",
                self.schema_version, NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION
            )));
        }
        let rebuilt = Self::new(
            self.denominator.clone(),
            self.serialized_bytes,
            self.resident_bytes,
        )?;
        if self.serialized_bits_per_unique_logical_value.to_bits()
            != rebuilt.serialized_bits_per_unique_logical_value.to_bits()
            || self.resident_bits_per_unique_logical_value.to_bits()
                != rebuilt.resident_bits_per_unique_logical_value.to_bits()
        {
            return Err(NnisError::invalid_input(
                "weight representation bits/value fields disagree with exact byte accounting",
            ));
        }
        Ok(())
    }
}

fn validate_denominator(denominator: &CanonicalWeightDenominatorV1) -> Result<()> {
    if denominator.schema_version != NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION {
        return Err(NnisError::unsupported(format!(
            "canonical weight denominator schema {}; supported version is {}",
            denominator.schema_version, NNIS_WEIGHT_REPRESENTATION_ACCOUNTING_VERSION
        )));
    }
    if denominator.logical_tensor_references == 0
        || denominator.logical_element_references == 0
        || denominator.unique_source_allocations == 0
        || denominator.unique_logical_values == 0
        || denominator.source_owned_bytes == 0
    {
        return Err(NnisError::invalid_input(
            "canonical weight denominator contains a zero required count",
        ));
    }
    if denominator.logical_element_references < denominator.unique_logical_values {
        return Err(NnisError::invalid_input(
            "logical element references cannot be smaller than unique logical values",
        ));
    }
    Ok(())
}

fn validate_source_summary(source: &WeightAllocationSummaryV1) -> Result<()> {
    if source.schema_version != NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION {
        return Err(NnisError::unsupported(format!(
            "weight allocation summary schema {}; supported version is {}",
            source.schema_version, NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION
        )));
    }
    if source.segments.len()
        != usize::try_from(source.unique_device_allocations)
            .map_err(|_| NnisError::invalid_input("weight allocation count does not fit usize"))?
    {
        return Err(NnisError::invalid_input(
            "weight allocation segment count disagrees with summary",
        ));
    }

    let mut unique_elements = 0_u64;
    let mut unique_bytes = 0_u64;
    let mut logical_tensors = 0_u64;
    let mut logical_elements = 0_u64;
    for (index, segment) in source.segments.iter().enumerate() {
        if segment.allocation_index
            != u32::try_from(index)
                .map_err(|_| NnisError::invalid_input("weight allocation index exceeds u32"))?
        {
            return Err(NnisError::invalid_input(
                "weight allocation indices are not canonical and contiguous",
            ));
        }
        if segment.elements == 0 || segment.bytes == 0 || segment.logical_names.is_empty() {
            return Err(NnisError::invalid_input(
                "weight allocation segment contains a zero count or no logical names",
            ));
        }
        unique_elements = unique_elements
            .checked_add(segment.elements)
            .ok_or_else(|| NnisError::invalid_input("unique device element count overflows u64"))?;
        unique_bytes = unique_bytes
            .checked_add(segment.bytes)
            .ok_or_else(|| NnisError::invalid_input("owned device byte count overflows u64"))?;
        let aliases = u64::try_from(segment.logical_names.len())
            .map_err(|_| NnisError::invalid_input("logical alias count exceeds u64"))?;
        logical_tensors = logical_tensors.checked_add(aliases).ok_or_else(|| {
            NnisError::invalid_input("logical tensor reference count overflows u64")
        })?;
        logical_elements = logical_elements
            .checked_add(segment.elements.checked_mul(aliases).ok_or_else(|| {
                NnisError::invalid_input("logical element alias accounting overflows u64")
            })?)
            .ok_or_else(|| {
                NnisError::invalid_input("logical element reference count overflows u64")
            })?;
    }

    if unique_elements != source.unique_device_elements
        || unique_bytes != source.owned_device_allocation_bytes
        || logical_tensors != source.logical_tensor_references
        || logical_elements != source.logical_element_references
    {
        return Err(NnisError::invalid_input(
            "weight allocation summary totals do not reconcile with segments",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{WeightAllocationDTypeV1, WeightAllocationSegmentV1};

    fn source_summary() -> WeightAllocationSummaryV1 {
        WeightAllocationSummaryV1 {
            schema_version: NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION,
            logical_tensor_references: 3,
            logical_element_references: 40,
            unique_device_allocations: 2,
            unique_device_elements: 24,
            owned_device_allocation_bytes: 96,
            segments: vec![
                WeightAllocationSegmentV1 {
                    allocation_index: 0,
                    dtype: WeightAllocationDTypeV1::F32,
                    elements: 16,
                    bytes: 64,
                    logical_names: vec!["embedding".to_string(), "lm_head".to_string()],
                },
                WeightAllocationSegmentV1 {
                    allocation_index: 1,
                    dtype: WeightAllocationDTypeV1::F32,
                    elements: 8,
                    bytes: 32,
                    logical_names: vec!["norm".to_string()],
                },
            ],
        }
    }

    #[test]
    fn denominator_counts_aliases_once_for_bits_per_value() {
        let denominator =
            CanonicalWeightDenominatorV1::from_weight_allocation_summary(&source_summary())
                .unwrap();
        assert_eq!(denominator.logical_tensor_references, 3);
        assert_eq!(denominator.logical_element_references, 40);
        assert_eq!(denominator.unique_source_allocations, 2);
        assert_eq!(denominator.unique_logical_values, 24);
        assert_eq!(denominator.source_owned_bytes, 96);

        let accounting = WeightRepresentationAccountingV1::new(denominator, 12, 16).unwrap();
        assert_eq!(
            accounting
                .serialized_bits_per_unique_logical_value
                .to_bits(),
            4.0_f64.to_bits()
        );
        assert_eq!(
            accounting.resident_bits_per_unique_logical_value.to_bits(),
            (16.0_f64 * 8.0 / 24.0).to_bits()
        );
        accounting.validate().unwrap();
    }

    #[test]
    fn inconsistent_source_summary_fails_closed() {
        let mut source = source_summary();
        source.unique_device_elements += 1;
        assert!(CanonicalWeightDenominatorV1::from_weight_allocation_summary(&source).is_err());

        let mut source = source_summary();
        source.logical_element_references -= 1;
        assert!(CanonicalWeightDenominatorV1::from_weight_allocation_summary(&source).is_err());
    }

    #[test]
    fn zero_representation_bytes_fail_closed() {
        let denominator =
            CanonicalWeightDenominatorV1::from_weight_allocation_summary(&source_summary())
                .unwrap();
        assert!(WeightRepresentationAccountingV1::new(denominator.clone(), 0, 1).is_err());
        assert!(WeightRepresentationAccountingV1::new(denominator, 1, 0).is_err());
    }
}
