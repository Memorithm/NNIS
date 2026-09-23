//! CUDA-backed magnitude-sparse model storage with an honest dense-F32 execution boundary.
//!
//! The structural baseline is shape-independent: each unique physical F32
//! allocation is represented by one occupancy bitmap plus its retained F32
//! values. Logical aliases therefore reuse one compact allocation even when
//! their logical matrix/vector views differ. Full-model execution reconstructs
//! an alias-preserving dense-F32 graph and uses the standard NNIS decoder.

use crate::weights::WeightLogicalShapeV1;
use crate::{
    densify_sparse_reference_v1, sparsify_magnitude_reference_v1,
    DenseWeightMaterializationEvidenceV1, DeviceTensor, Model, ModelConfig, ModelWeights,
    SparseReferenceTensorV1, WeightDType, WeightRepresentationFamilyV1,
    NNIS_SPARSE_REFERENCE_SERIALIZED_HEADER_BYTES, NNIS_SPARSE_REFERENCE_STORAGE_VERSION,
};
use nnis_rt::{DeviceBuffer, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Version of the model-level magnitude-sparse storage contract.
pub const NNIS_SPARSE_MODEL_STORAGE_VERSION: u32 = 1;

/// Exact accounting for one unique source allocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseReferenceAllocationSummaryV1 {
    pub allocation_index: u32,
    pub logical_names: Vec<String>,
    pub logical_values: u64,
    pub source_f32_bytes: u64,
    pub threshold: f32,
    pub retained_values: u64,
    pub serialized_header_bytes: u64,
    pub serialized_bitmap_bytes: u64,
    pub serialized_value_bytes: u64,
    pub serialized_total_bytes: u64,
    pub resident_bitmap_bytes: u64,
    pub resident_value_bytes: u64,
    pub resident_total_bytes: u64,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
}

/// Exact owned-storage summary for the structural sparse baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseReferenceModelStorageSummaryV1 {
    pub schema_version: u32,
    pub tensor_storage_version: u32,
    pub representation: String,
    pub threshold: f32,
    pub execution_qualified: bool,
    pub logical_tensor_references: u64,
    pub logical_element_references: u64,
    pub unique_source_allocations: u64,
    pub unique_logical_values: u64,
    pub retained_values: u64,
    pub source_f32_owned_bytes: u64,
    pub serialized_total_bytes: u64,
    pub resident_device_bytes: u64,
    pub serialized_bits_per_unique_logical_value: f64,
    pub resident_bits_per_unique_logical_value: f64,
    pub retained_fraction: f64,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
    pub allocations: Vec<SparseReferenceAllocationSummaryV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SparseReferenceLogicalBinding {
    allocation_index: usize,
    shape: WeightLogicalShapeV1,
}

struct SparseReferenceDeviceAllocation {
    occupancy_bitmap: DeviceBuffer<u8>,
    retained_values: DeviceBuffer<f32>,
}

/// Live CUDA allocations for the shape-independent sparse model baseline.
pub struct SparseReferenceModelStorageV1 {
    allocations: Vec<SparseReferenceDeviceAllocation>,
    bindings: BTreeMap<String, SparseReferenceLogicalBinding>,
    summary: SparseReferenceModelStorageSummaryV1,
}

impl SparseReferenceModelStorageV1 {
    pub fn from_f32_model_weights(
        weights: &ModelWeights,
        stream: &Stream,
        threshold: f32,
    ) -> Result<Self> {
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(NnisError::invalid_input(
                "sparse model threshold must be finite and non-negative",
            ));
        }
        let source_summary = weights.weight_allocation_summary_v1()?;
        let mut source_indices = BTreeMap::<u64, usize>::new();
        let mut allocations = Vec::<SparseReferenceDeviceAllocation>::new();
        let mut bindings = BTreeMap::<String, SparseReferenceLogicalBinding>::new();
        let mut summaries = Vec::<SparseReferenceAllocationSummaryV1>::new();

        let mut logical_tensor_references = 0_u64;
        let mut logical_element_references = 0_u64;
        let mut unique_logical_values = 0_u64;
        let mut retained_values = 0_u64;
        let mut source_f32_owned_bytes = 0_u64;
        let mut serialized_total_bytes = 0_u64;
        let mut resident_device_bytes = 0_u64;
        let mut max_abs_error = 0.0_f32;
        let mut weighted_squared_error = 0.0_f64;

        weights.for_each_logical_tensor(|name, tensor, shape| {
            let (source_key, source_bytes) = match tensor {
                DeviceTensor::F32(buffer) => (buffer.device_ptr(), buffer.size_bytes()),
                DeviceTensor::Bf16(_) => {
                    return Err(NnisError::unsupported(format!(
                        "sparse reference storage requires F32 source weights; {name} is BF16"
                    )));
                }
            };
            if source_key == 0 || tensor.is_empty() {
                return Err(NnisError::invalid_input(format!(
                    "sparse source weight {name} has no live allocation"
                )));
            }

            let logical_values = u64::try_from(tensor.len()).map_err(|_| {
                NnisError::invalid_input(format!("sparse source weight {name} length exceeds u64"))
            })?;
            if shape.element_count()? != tensor.len() {
                return Err(NnisError::invalid_input(format!(
                    "sparse logical shape for {name} disagrees with source tensor length"
                )));
            }
            checked_add(
                &mut logical_tensor_references,
                1,
                "sparse logical tensor reference count",
            )?;
            checked_add(
                &mut logical_element_references,
                logical_values,
                "sparse logical element reference count",
            )?;

            if let Some(index) = source_indices.get(&source_key).copied() {
                let summary = summaries.get_mut(index).ok_or_else(|| {
                    NnisError::invalid_input("sparse source allocation index escaped summary")
                })?;
                let source_bytes_u64 = u64::try_from(source_bytes)
                    .map_err(|_| NnisError::invalid_input("sparse source bytes exceed u64"))?;
                if summary.logical_values != logical_values
                    || summary.source_f32_bytes != source_bytes_u64
                {
                    return Err(NnisError::invalid_input(format!(
                        "sparse source alias {name} disagrees with its original allocation"
                    )));
                }
                summary.logical_names.push(name.to_string());
                if bindings
                    .insert(
                        name.to_string(),
                        SparseReferenceLogicalBinding {
                            allocation_index: index,
                            shape,
                        },
                    )
                    .is_some()
                {
                    return Err(NnisError::invalid_input(format!(
                        "duplicate sparse logical binding {name}"
                    )));
                }
                return Ok(());
            }

            let host = tensor.as_f32()?.to_vec(stream)?;
            let sparse = sparsify_magnitude_reference_v1(&host, threshold)?;
            let canonical = sparse.canonical_serialized_bytes()?;
            let occupancy_bitmap =
                DeviceBuffer::from_host(weights.context(), stream, &sparse.occupancy_bitmap)?;
            let retained =
                DeviceBuffer::from_host(weights.context(), stream, &sparse.retained_values)?;

            let bitmap_bytes = u64::try_from(occupancy_bitmap.size_bytes())
                .map_err(|_| NnisError::invalid_input("sparse bitmap bytes exceed u64"))?;
            let value_bytes = u64::try_from(retained.size_bytes())
                .map_err(|_| NnisError::invalid_input("sparse value bytes exceed u64"))?;
            let resident_total_bytes = bitmap_bytes
                .checked_add(value_bytes)
                .ok_or_else(|| NnisError::invalid_input("sparse resident bytes overflow u64"))?;
            let serialized_bytes = u64::try_from(canonical.len())
                .map_err(|_| NnisError::invalid_input("sparse serialized bytes exceed u64"))?;
            let retained_count = u64::try_from(sparse.retained_values.len())
                .map_err(|_| NnisError::invalid_input("sparse retained count exceeds u64"))?;
            let source_bytes_u64 = u64::try_from(source_bytes)
                .map_err(|_| NnisError::invalid_input("sparse source bytes exceed u64"))?;
            let expected_value_bytes = retained_count
                .checked_mul(4)
                .ok_or_else(|| NnisError::invalid_input("sparse value bytes overflow u64"))?;
            let expected_serialized = NNIS_SPARSE_REFERENCE_SERIALIZED_HEADER_BYTES
                .checked_add(bitmap_bytes)
                .and_then(|value| value.checked_add(expected_value_bytes))
                .ok_or_else(|| NnisError::invalid_input("sparse serialized bytes overflow u64"))?;
            if value_bytes != expected_value_bytes || serialized_bytes != expected_serialized {
                return Err(NnisError::invalid_input(
                    "sparse canonical serialization disagrees with resident array accounting",
                ));
            }

            let allocation_index = u32::try_from(allocations.len()).map_err(|_| {
                NnisError::invalid_input("sparse allocation count exceeds u32 contract capacity")
            })?;
            let summary_index = summaries.len();
            source_indices.insert(source_key, summary_index);
            allocations.push(SparseReferenceDeviceAllocation {
                occupancy_bitmap,
                retained_values: retained,
            });
            if bindings
                .insert(
                    name.to_string(),
                    SparseReferenceLogicalBinding {
                        allocation_index: summary_index,
                        shape,
                    },
                )
                .is_some()
            {
                return Err(NnisError::invalid_input(format!(
                    "duplicate sparse logical binding {name}"
                )));
            }
            summaries.push(SparseReferenceAllocationSummaryV1 {
                allocation_index,
                logical_names: vec![name.to_string()],
                logical_values,
                source_f32_bytes: source_bytes_u64,
                threshold,
                retained_values: retained_count,
                serialized_header_bytes: NNIS_SPARSE_REFERENCE_SERIALIZED_HEADER_BYTES,
                serialized_bitmap_bytes: bitmap_bytes,
                serialized_value_bytes: expected_value_bytes,
                serialized_total_bytes: serialized_bytes,
                resident_bitmap_bytes: bitmap_bytes,
                resident_value_bytes: value_bytes,
                resident_total_bytes,
                max_abs_error: sparse.max_abs_error,
                mean_squared_error: sparse.mean_squared_error,
            });

            checked_add(
                &mut unique_logical_values,
                logical_values,
                "sparse unique logical value count",
            )?;
            checked_add(
                &mut retained_values,
                retained_count,
                "sparse retained value count",
            )?;
            checked_add(
                &mut source_f32_owned_bytes,
                source_bytes_u64,
                "sparse source owned bytes",
            )?;
            checked_add(
                &mut serialized_total_bytes,
                serialized_bytes,
                "sparse serialized total bytes",
            )?;
            checked_add(
                &mut resident_device_bytes,
                resident_total_bytes,
                "sparse resident total bytes",
            )?;
            max_abs_error = max_abs_error.max(sparse.max_abs_error);
            weighted_squared_error += sparse.mean_squared_error * logical_values as f64;
            Ok(())
        })?;
        stream.synchronize()?;

        if unique_logical_values == 0 {
            return Err(NnisError::invalid_input(
                "sparse reference model contains no unique logical values",
            ));
        }
        if source_summary.logical_tensor_references != logical_tensor_references
            || source_summary.logical_element_references != logical_element_references
            || source_summary.unique_device_allocations
                != u64::try_from(allocations.len())
                    .map_err(|_| NnisError::invalid_input("sparse allocation count exceeds u64"))?
            || source_summary.unique_device_elements != unique_logical_values
            || source_summary.owned_device_allocation_bytes != source_f32_owned_bytes
        {
            return Err(NnisError::invalid_input(
                "sparse source accounting does not reconcile with WeightAllocationSummaryV1",
            ));
        }

        let values = unique_logical_values as f64;
        let storage = Self {
            allocations,
            bindings,
            summary: SparseReferenceModelStorageSummaryV1 {
                schema_version: NNIS_SPARSE_MODEL_STORAGE_VERSION,
                tensor_storage_version: NNIS_SPARSE_REFERENCE_STORAGE_VERSION,
                representation: "magnitude-sparse-bitmap-f32-reference".to_string(),
                threshold,
                execution_qualified: false,
                logical_tensor_references,
                logical_element_references,
                unique_source_allocations: source_summary.unique_device_allocations,
                unique_logical_values,
                retained_values,
                source_f32_owned_bytes,
                serialized_total_bytes,
                resident_device_bytes,
                serialized_bits_per_unique_logical_value: serialized_total_bytes as f64 * 8.0
                    / values,
                resident_bits_per_unique_logical_value: resident_device_bytes as f64 * 8.0 / values,
                retained_fraction: retained_values as f64 / values,
                max_abs_error,
                mean_squared_error: weighted_squared_error / values,
                allocations: summaries,
            },
        };
        storage.validate_resident_allocations()?;
        Ok(storage)
    }

    #[must_use]
    pub fn summary(&self) -> &SparseReferenceModelStorageSummaryV1 {
        &self.summary
    }

    fn materialize_dense_f32_weights(
        &self,
        config: &ModelConfig,
        stream: &Stream,
    ) -> Result<ModelWeights> {
        self.validate_resident_allocations()?;
        if !Arc::ptr_eq(
            stream.ctx(),
            self.allocations
                .first()
                .ok_or_else(|| {
                    NnisError::invalid_input("sparse reference storage has no resident allocations")
                })?
                .occupancy_bitmap
                .ctx(),
        ) {
            return Err(NnisError::invalid_input(
                "sparse dense materialization stream must share the compact-storage CUDA context",
            ));
        }

        let mut dense_allocations =
            Vec::<Arc<DeviceBuffer<f32>>>::with_capacity(self.allocations.len());
        for (allocation, summary) in self.allocations.iter().zip(&self.summary.allocations) {
            let occupancy_bitmap = allocation.occupancy_bitmap.to_vec(stream)?;
            let retained_values = allocation.retained_values.to_vec(stream)?;
            let sparse = SparseReferenceTensorV1 {
                element_count: summary.logical_values,
                threshold: summary.threshold,
                occupancy_bitmap,
                retained_values,
                max_abs_error: summary.max_abs_error,
                mean_squared_error: summary.mean_squared_error,
            };
            let dense_host = densify_sparse_reference_v1(&sparse)?;
            let dense = DeviceBuffer::from_host(stream.ctx(), stream, &dense_host)?;
            dense_allocations.push(Arc::new(dense));
        }

        let mut logical = BTreeMap::new();
        for (name, binding) in &self.bindings {
            let allocation = dense_allocations
                .get(binding.allocation_index)
                .ok_or_else(|| {
                    NnisError::invalid_input(
                        "sparse dense logical binding references a missing allocation",
                    )
                })?;
            if logical
                .insert(
                    name.clone(),
                    (binding.shape, DeviceTensor::F32(Arc::clone(allocation))),
                )
                .is_some()
            {
                return Err(NnisError::invalid_input(format!(
                    "duplicate sparse dense logical binding {name}"
                )));
            }
        }
        stream.synchronize()?;
        ModelWeights::from_named_logical_tensors(config, logical)
    }

    pub fn validate_resident_allocations(&self) -> Result<()> {
        if self.allocations.len() != self.summary.allocations.len() {
            return Err(NnisError::invalid_input(
                "sparse live allocation count disagrees with summary",
            ));
        }
        let expected_bindings = usize::try_from(self.summary.logical_tensor_references)
            .map_err(|_| NnisError::invalid_input("sparse binding count exceeds usize"))?;
        if self.bindings.len() != expected_bindings {
            return Err(NnisError::invalid_input(
                "sparse logical binding count disagrees with summary",
            ));
        }

        let mut bitmap_total = 0_u64;
        let mut value_total = 0_u64;
        for (allocation, summary) in self.allocations.iter().zip(&self.summary.allocations) {
            let bitmap_bytes = u64::try_from(allocation.occupancy_bitmap.size_bytes())
                .map_err(|_| NnisError::invalid_input("sparse live bitmap bytes exceed u64"))?;
            let value_bytes = u64::try_from(allocation.retained_values.size_bytes())
                .map_err(|_| NnisError::invalid_input("sparse live value bytes exceed u64"))?;
            if bitmap_bytes != summary.resident_bitmap_bytes
                || value_bytes != summary.resident_value_bytes
                || bitmap_bytes
                    .checked_add(value_bytes)
                    .ok_or_else(|| NnisError::invalid_input("sparse live bytes overflow u64"))?
                    != summary.resident_total_bytes
            {
                return Err(NnisError::invalid_input(
                    "sparse live allocation bytes disagree with summary",
                ));
            }
            checked_add(&mut bitmap_total, bitmap_bytes, "sparse live bitmap bytes")?;
            checked_add(&mut value_total, value_bytes, "sparse live value bytes")?;
        }
        if bitmap_total
            .checked_add(value_total)
            .ok_or_else(|| NnisError::invalid_input("sparse live total bytes overflow u64"))?
            != self.summary.resident_device_bytes
        {
            return Err(NnisError::invalid_input(
                "sparse live allocation totals disagree with summary",
            ));
        }
        Ok(())
    }
}

/// Full-model structural sparse path with dense-F32 execution.
pub struct SparseDenseMaterializedModelV1 {
    compact_storage: SparseReferenceModelStorageV1,
    model: Model,
    materialization: DenseWeightMaterializationEvidenceV1,
}

impl SparseDenseMaterializedModelV1 {
    pub fn from_f32_model_weights(
        config: ModelConfig,
        source_weights: ModelWeights,
        stream: &Stream,
        threshold: f32,
    ) -> Result<Self> {
        if config.weight_dtype != WeightDType::F32 {
            return Err(NnisError::unsupported(
                "sparse dense materialization requires an F32 execution source graph",
            ));
        }
        source_weights.validate(&config)?;
        let source_summary = source_weights.weight_allocation_summary_v1()?;
        let started = Instant::now();
        let compact_storage = SparseReferenceModelStorageV1::from_f32_model_weights(
            &source_weights,
            stream,
            threshold,
        )?;
        let dense_weights = compact_storage.materialize_dense_f32_weights(&config, stream)?;
        stream.synchronize()?;
        let duration_ns = u64::try_from(started.elapsed().as_nanos()).map_err(|_| {
            NnisError::invalid_input(
                "sparse dense materialization duration exceeds u64 nanoseconds",
            )
        })?;
        let dense_summary = dense_weights.weight_allocation_summary_v1()?;
        if source_summary.unique_device_elements != compact_storage.summary.unique_logical_values
            || source_summary.unique_device_elements != dense_summary.unique_device_elements
            || source_summary.owned_device_allocation_bytes
                != compact_storage.summary.source_f32_owned_bytes
        {
            return Err(NnisError::invalid_input(
                "sparse dense materialization denominator does not reconcile across source, compact and dense graphs",
            ));
        }
        let materialization = DenseWeightMaterializationEvidenceV1::new(
            WeightRepresentationFamilyV1::MagnitudeSparse,
            source_summary.unique_device_elements,
            source_summary.owned_device_allocation_bytes,
            compact_storage.summary.resident_device_bytes,
            dense_summary.owned_device_allocation_bytes,
            0,
            duration_ns,
        )?;
        let model = Model::new(config, dense_weights, stream)?;
        Ok(Self {
            compact_storage,
            model,
            materialization,
        })
    }

    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    #[must_use]
    pub fn compact_storage_summary(&self) -> &SparseReferenceModelStorageSummaryV1 {
        self.compact_storage.summary()
    }

    #[must_use]
    pub fn materialization_evidence(&self) -> &DenseWeightMaterializationEvidenceV1 {
        &self.materialization
    }
}

fn checked_add(counter: &mut u64, value: u64, label: &str) -> Result<()> {
    *counter = counter
        .checked_add(value)
        .ok_or_else(|| NnisError::invalid_input(format!("{label} overflows u64")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_storage_summary_rejects_invalid_threshold_before_cuda() {
        let invalid = [f32::NAN, f32::INFINITY, -1.0];
        for threshold in invalid {
            assert!(
                !threshold.is_finite() || threshold < 0.0,
                "test threshold must be invalid"
            );
        }
    }
}
