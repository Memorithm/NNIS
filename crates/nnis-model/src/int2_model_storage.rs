//! Live CUDA-backed ternary INT2 model storage and exact accounting.
//!
//! This module materializes every unique F32 source allocation into the fixed
//! ternary INT2 reference representation. Aliased logical weights reuse one
//! packed allocation. The only executable access is an explicitly versioned
//! projection plan; full-model INT2 execution remains unqualified.

use crate::dense_weight_materialization::checked_host_payload_bytes;
use crate::weights::WeightLogicalShapeV1;
use crate::{
    dequantize_int2_ternary_reference_v1, quantize_int2_ternary_reference_v1,
    DenseWeightMaterializationEvidenceV1, DenseWeightMaterializationEvidenceV2, DeviceTensor,
    Int2ReferenceProjectionPlanV1, Model, ModelConfig, ModelWeights, WeightDType,
    WeightRepresentationFamilyV1,
    NNIS_INT2_REFERENCE_SERIALIZED_HEADER_BYTES, NNIS_INT2_REFERENCE_STORAGE_VERSION,
};
use nnis_kernels::F32Int2Gemv;
use nnis_rt::{DeviceBuffer, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// Accounting and reconstruction evidence for one unique INT2 source allocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Int2ReferenceAllocationSummaryV1 {
    pub allocation_index: u32,
    pub logical_names: Vec<String>,
    pub logical_values: u64,
    pub source_f32_bytes: u64,
    pub serialized_header_bytes: u64,
    pub serialized_payload_bytes: u64,
    pub serialized_total_bytes: u64,
    pub resident_payload_bytes: u64,
    pub resident_scale_bytes: u64,
    pub resident_total_bytes: u64,
    pub scale: f32,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
}

/// Exact owned-storage summary for the non-promoted full-model INT2 reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Int2ReferenceStorageSummaryV1 {
    pub schema_version: u32,
    pub representation: String,
    pub quantization: String,
    pub scale_scope: String,
    pub execution_qualified: bool,
    pub logical_tensor_references: u64,
    pub logical_element_references: u64,
    pub unique_source_allocations: u64,
    pub unique_logical_values: u64,
    pub source_f32_owned_bytes: u64,
    pub serialized_header_bytes: u64,
    pub serialized_payload_bytes: u64,
    pub serialized_total_bytes: u64,
    pub resident_payload_bytes: u64,
    pub resident_scale_bytes: u64,
    pub resident_device_bytes: u64,
    pub serialized_bits_per_unique_logical_value: f64,
    pub resident_bits_per_unique_logical_value: f64,
    pub source_f32_to_resident_int2_byte_ratio: f64,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
    pub allocations: Vec<Int2ReferenceAllocationSummaryV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Int2ReferenceLogicalBinding {
    allocation_index: usize,
    shape: WeightLogicalShapeV1,
}

struct Int2ReferenceDeviceAllocation {
    packed_values: DeviceBuffer<u8>,
    scale: DeviceBuffer<f32>,
}

/// Live CUDA allocations backing the ternary INT2 reference storage.
pub struct Int2ReferenceModelStorageV1 {
    allocations: Vec<Int2ReferenceDeviceAllocation>,
    bindings: BTreeMap<String, Int2ReferenceLogicalBinding>,
    summary: Int2ReferenceStorageSummaryV1,
}

/// Full-model ternary-INT2 reference path with dense-F32 execution.
///
/// Compact INT2 payloads remain resident while a separate F32 graph feeds the
/// standard NNIS decoder. This is an honest materialization path, not low-bit
/// compute, and remains unqualified until exact-checkpoint generation evidence
/// is recorded.
pub struct Int2DenseMaterializedModelV1 {
    compact_storage: Int2ReferenceModelStorageV1,
    model: Model,
    materialization: DenseWeightMaterializationEvidenceV2,
}

impl Int2DenseMaterializedModelV1 {
    pub fn from_f32_model_weights(
        config: ModelConfig,
        source_weights: ModelWeights,
        stream: &Stream,
    ) -> Result<Self> {
        if config.weight_dtype != WeightDType::F32 {
            return Err(NnisError::unsupported(
                "INT2 dense materialization requires an F32 execution source graph",
            ));
        }
        source_weights.validate(&config)?;
        let source_summary = source_weights.weight_allocation_summary_v1()?;
        let started = Instant::now();
        let compact_storage =
            Int2ReferenceModelStorageV1::from_f32_model_weights(&source_weights, stream)?;
        let (dense_weights, peak_host_temporary_payload_bytes) =
            compact_storage.materialize_dense_f32_weights(&config, stream)?;
        stream.synchronize()?;
        let duration_ns = u64::try_from(started.elapsed().as_nanos()).map_err(|_| {
            NnisError::invalid_input("INT2 dense materialization duration exceeds u64 nanoseconds")
        })?;
        let dense_summary = dense_weights.weight_allocation_summary_v1()?;
        if source_summary.unique_device_elements != compact_storage.summary.unique_logical_values
            || source_summary.unique_device_elements != dense_summary.unique_device_elements
            || source_summary.owned_device_allocation_bytes
                != compact_storage.summary.source_f32_owned_bytes
        {
            return Err(NnisError::invalid_input(
                "INT2 dense materialization denominator does not reconcile across source, compact and dense graphs",
            ));
        }
        let device_ownership = DenseWeightMaterializationEvidenceV1::new(
            WeightRepresentationFamilyV1::Int2Ternary,
            source_summary.unique_device_elements,
            source_summary.owned_device_allocation_bytes,
            compact_storage.summary.resident_device_bytes,
            dense_summary.owned_device_allocation_bytes,
            0,
            duration_ns,
        )?;
        let materialization = DenseWeightMaterializationEvidenceV2::new(
            device_ownership,
            peak_host_temporary_payload_bytes,
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
    pub fn compact_storage_summary(&self) -> &Int2ReferenceStorageSummaryV1 {
        self.compact_storage.summary()
    }

    #[must_use]
    pub fn materialization_evidence(&self) -> &DenseWeightMaterializationEvidenceV1 {
        &self.materialization.device_ownership
    }

    #[must_use]
    pub fn materialization_evidence_v2(&self) -> &DenseWeightMaterializationEvidenceV2 {
        &self.materialization
    }
}

impl Int2ReferenceModelStorageV1 {
    /// Quantize every unique F32 model-weight allocation and materialize one
    /// packed payload plus one F32 scale as real CUDA allocations.
    pub fn from_f32_model_weights(weights: &ModelWeights, stream: &Stream) -> Result<Self> {
        let source_summary = weights.weight_allocation_summary_v1()?;
        let mut source_indices = BTreeMap::<u64, usize>::new();
        let mut allocations = Vec::<Int2ReferenceDeviceAllocation>::new();
        let mut bindings = BTreeMap::<String, Int2ReferenceLogicalBinding>::new();
        let mut summaries = Vec::<Int2ReferenceAllocationSummaryV1>::new();

        let mut logical_tensor_references = 0_u64;
        let mut logical_element_references = 0_u64;
        let mut unique_logical_values = 0_u64;
        let mut source_f32_owned_bytes = 0_u64;
        let mut serialized_header_bytes = 0_u64;
        let mut serialized_payload_bytes = 0_u64;
        let mut serialized_total_bytes = 0_u64;
        let mut resident_payload_bytes = 0_u64;
        let mut resident_scale_bytes = 0_u64;
        let mut resident_device_bytes = 0_u64;
        let mut max_abs_error = 0.0_f32;
        let mut weighted_squared_error = 0.0_f64;

        weights.for_each_logical_tensor(|name, tensor, shape| {
            let (source_key, source_bytes) = match tensor {
                DeviceTensor::F32(buffer) => (buffer.device_ptr(), buffer.size_bytes()),
                DeviceTensor::Bf16(_) => {
                    return Err(NnisError::unsupported(format!(
                        "INT2 reference storage requires F32 source weights; {name} is BF16"
                    )));
                }
            };
            if source_key == 0 || tensor.is_empty() {
                return Err(NnisError::invalid_input(format!(
                    "INT2 source weight {name} has no live allocation"
                )));
            }
            let logical_values = u64::try_from(tensor.len()).map_err(|_| {
                NnisError::invalid_input(format!("INT2 source weight {name} length exceeds u64"))
            })?;
            let shape_values = shape.element_count()?;
            if shape_values != tensor.len() {
                return Err(NnisError::invalid_input(format!(
                    "INT2 logical shape for {name} contains {shape_values} values but the source tensor contains {}",
                    tensor.len()
                )));
            }
            checked_add(
                &mut logical_tensor_references,
                1,
                "INT2 logical tensor reference count",
            )?;
            checked_add(
                &mut logical_element_references,
                logical_values,
                "INT2 logical element reference count",
            )?;

            if let Some(index) = source_indices.get(&source_key).copied() {
                let summary = summaries.get_mut(index).ok_or_else(|| {
                    NnisError::invalid_input("INT2 source allocation index escaped summary")
                })?;
                if summary.logical_values != logical_values
                    || summary.source_f32_bytes
                        != u64::try_from(source_bytes).map_err(|_| {
                            NnisError::invalid_input("INT2 source byte count exceeds u64")
                        })?
                {
                    return Err(NnisError::invalid_input(format!(
                        "INT2 source alias {name} disagrees with its original allocation"
                    )));
                }
                summary.logical_names.push(name.to_string());
                if bindings
                    .insert(
                        name.to_string(),
                        Int2ReferenceLogicalBinding {
                            allocation_index: index,
                            shape,
                        },
                    )
                    .is_some()
                {
                    return Err(NnisError::invalid_input(format!(
                        "duplicate INT2 logical binding {name}"
                    )));
                }
                return Ok(());
            }

            let host = tensor.as_f32()?.to_vec(stream)?;
            let quantized = quantize_int2_ternary_reference_v1(&host)?;
            let canonical = quantized.canonical_serialized_bytes()?;
            let packed_values =
                DeviceBuffer::from_host(weights.context(), stream, &quantized.packed_values)?;
            let scale = DeviceBuffer::from_host(weights.context(), stream, &[quantized.scale])?;

            let payload_bytes = u64::try_from(packed_values.size_bytes())
                .map_err(|_| NnisError::invalid_input("INT2 payload byte count exceeds u64"))?;
            let scale_bytes = u64::try_from(scale.size_bytes())
                .map_err(|_| NnisError::invalid_input("INT2 scale byte count exceeds u64"))?;
            let allocation_resident_bytes = payload_bytes
                .checked_add(scale_bytes)
                .ok_or_else(|| NnisError::invalid_input("INT2 resident bytes overflow u64"))?;
            let allocation_serialized_bytes = u64::try_from(canonical.len())
                .map_err(|_| NnisError::invalid_input("INT2 serialized bytes exceed u64"))?;
            let source_bytes = u64::try_from(source_bytes)
                .map_err(|_| NnisError::invalid_input("INT2 source byte count exceeds u64"))?;
            let canonical_payload_bytes = u64::try_from(quantized.packed_values.len())
                .map_err(|_| NnisError::invalid_input("INT2 payload length exceeds u64"))?;
            let expected_serialized = NNIS_INT2_REFERENCE_SERIALIZED_HEADER_BYTES
                .checked_add(canonical_payload_bytes)
                .ok_or_else(|| NnisError::invalid_input("INT2 serialized bytes overflow u64"))?;
            if allocation_serialized_bytes != expected_serialized {
                return Err(NnisError::invalid_input(
                    "INT2 canonical serialization length disagrees with accounting",
                ));
            }
            if payload_bytes != canonical_payload_bytes {
                return Err(NnisError::invalid_input(
                    "INT2 device payload bytes disagree with packed host payload bytes",
                ));
            }

            let allocation_index = u32::try_from(allocations.len()).map_err(|_| {
                NnisError::invalid_input("INT2 allocation count exceeds u32 contract capacity")
            })?;
            let summary_index = summaries.len();
            source_indices.insert(source_key, summary_index);
            allocations.push(Int2ReferenceDeviceAllocation {
                packed_values,
                scale,
            });
            if bindings
                .insert(
                    name.to_string(),
                    Int2ReferenceLogicalBinding {
                        allocation_index: summary_index,
                        shape,
                    },
                )
                .is_some()
            {
                return Err(NnisError::invalid_input(format!(
                    "duplicate INT2 logical binding {name}"
                )));
            }
            summaries.push(Int2ReferenceAllocationSummaryV1 {
                allocation_index,
                logical_names: vec![name.to_string()],
                logical_values,
                source_f32_bytes: source_bytes,
                serialized_header_bytes: NNIS_INT2_REFERENCE_SERIALIZED_HEADER_BYTES,
                serialized_payload_bytes: canonical_payload_bytes,
                serialized_total_bytes: allocation_serialized_bytes,
                resident_payload_bytes: payload_bytes,
                resident_scale_bytes: scale_bytes,
                resident_total_bytes: allocation_resident_bytes,
                scale: quantized.scale,
                max_abs_error: quantized.max_abs_error,
                mean_squared_error: quantized.mean_squared_error,
            });

            checked_add(
                &mut unique_logical_values,
                logical_values,
                "INT2 unique logical value count",
            )?;
            checked_add(
                &mut source_f32_owned_bytes,
                source_bytes,
                "INT2 source owned bytes",
            )?;
            checked_add(
                &mut serialized_header_bytes,
                NNIS_INT2_REFERENCE_SERIALIZED_HEADER_BYTES,
                "INT2 serialized header bytes",
            )?;
            checked_add(
                &mut serialized_payload_bytes,
                canonical_payload_bytes,
                "INT2 serialized payload bytes",
            )?;
            checked_add(
                &mut serialized_total_bytes,
                allocation_serialized_bytes,
                "INT2 serialized total bytes",
            )?;
            checked_add(
                &mut resident_payload_bytes,
                payload_bytes,
                "INT2 resident payload bytes",
            )?;
            checked_add(
                &mut resident_scale_bytes,
                scale_bytes,
                "INT2 resident scale bytes",
            )?;
            checked_add(
                &mut resident_device_bytes,
                allocation_resident_bytes,
                "INT2 resident device bytes",
            )?;
            max_abs_error = max_abs_error.max(quantized.max_abs_error);
            weighted_squared_error += quantized.mean_squared_error * logical_values as f64;
            Ok(())
        })?;
        stream.synchronize()?;

        if unique_logical_values == 0 {
            return Err(NnisError::invalid_input(
                "INT2 reference model contains no unique logical values",
            ));
        }
        if source_summary.logical_tensor_references != logical_tensor_references
            || source_summary.logical_element_references != logical_element_references
            || source_summary.unique_device_allocations
                != u64::try_from(allocations.len())
                    .map_err(|_| NnisError::invalid_input("INT2 allocation count exceeds u64"))?
            || source_summary.unique_device_elements != unique_logical_values
            || source_summary.owned_device_allocation_bytes != source_f32_owned_bytes
        {
            return Err(NnisError::invalid_input(
                "INT2 source accounting does not reconcile with WeightAllocationSummaryV1",
            ));
        }

        let serialized_bits_per_unique_logical_value =
            serialized_total_bytes as f64 * 8.0 / unique_logical_values as f64;
        let resident_bits_per_unique_logical_value =
            resident_device_bytes as f64 * 8.0 / unique_logical_values as f64;
        let source_f32_to_resident_int2_byte_ratio =
            source_f32_owned_bytes as f64 / resident_device_bytes as f64;
        let mean_squared_error = weighted_squared_error / unique_logical_values as f64;

        let storage = Self {
            allocations,
            bindings,
            summary: Int2ReferenceStorageSummaryV1 {
                schema_version: NNIS_INT2_REFERENCE_STORAGE_VERSION,
                representation: "ternary-int2-reference".to_string(),
                quantization: "maxabs-half-threshold-minus1-zero-plus1".to_string(),
                scale_scope: "one-f32-scale-per-unique-source-allocation".to_string(),
                execution_qualified: false,
                logical_tensor_references,
                logical_element_references,
                unique_source_allocations: source_summary.unique_device_allocations,
                unique_logical_values,
                source_f32_owned_bytes,
                serialized_header_bytes,
                serialized_payload_bytes,
                serialized_total_bytes,
                resident_payload_bytes,
                resident_scale_bytes,
                resident_device_bytes,
                serialized_bits_per_unique_logical_value,
                resident_bits_per_unique_logical_value,
                source_f32_to_resident_int2_byte_ratio,
                max_abs_error,
                mean_squared_error,
                allocations: summaries,
            },
        };
        storage.validate_resident_allocations()?;
        Ok(storage)
    }

    #[must_use]
    pub fn summary(&self) -> &Int2ReferenceStorageSummaryV1 {
        &self.summary
    }

    fn materialize_dense_f32_weights(
        &self,
        config: &ModelConfig,
        stream: &Stream,
    ) -> Result<(ModelWeights, u64)> {
        self.validate_resident_allocations()?;
        if !Arc::ptr_eq(
            stream.ctx(),
            self.allocations
                .first()
                .ok_or_else(|| {
                    NnisError::invalid_input("INT2 reference storage has no resident allocations")
                })?
                .packed_values
                .ctx(),
        ) {
            return Err(NnisError::invalid_input(
                "INT2 dense materialization stream must share the compact-storage CUDA context",
            ));
        }

        let mut dense_allocations =
            Vec::<Arc<DeviceBuffer<f32>>>::with_capacity(self.allocations.len());
        let mut peak_host_temporary_payload_bytes = 0_u64;
        for (allocation, summary) in self.allocations.iter().zip(&self.summary.allocations) {
            let packed_values = allocation.packed_values.to_vec(stream)?;
            let packed_host_bytes = u64::try_from(packed_values.len())
                .map_err(|_| NnisError::invalid_input("INT2 host packed bytes exceed u64"))?;
            let scale_values = allocation.scale.to_vec(stream)?;
            let scale_host_bytes = u64::try_from(scale_values.len())
                .map_err(|_| NnisError::invalid_input("INT2 host scale count exceeds u64"))?
                .checked_mul(4)
                .ok_or_else(|| NnisError::invalid_input("INT2 host scale bytes overflow u64"))?;
            if scale_values.len() != 1 || scale_values[0].to_bits() != summary.scale.to_bits() {
                return Err(NnisError::invalid_input(
                    "INT2 live scale disagrees with immutable storage summary",
                ));
            }
            let quantized = crate::Int2ReferenceQuantizedTensorV1 {
                element_count: summary.logical_values,
                scale: summary.scale,
                packed_values,
                max_abs_error: summary.max_abs_error,
                mean_squared_error: summary.mean_squared_error,
            };
            let dense_host = dequantize_int2_ternary_reference_v1(&quantized)?;
            let dense_host_bytes = u64::try_from(dense_host.len())
                .map_err(|_| NnisError::invalid_input("INT2 dense host value count exceeds u64"))?
                .checked_mul(4)
                .ok_or_else(|| NnisError::invalid_input("INT2 dense host bytes overflow u64"))?;
            let host_payload_bytes = checked_host_payload_bytes([
                packed_host_bytes,
                scale_host_bytes,
                dense_host_bytes,
            ])?;
            peak_host_temporary_payload_bytes =
                peak_host_temporary_payload_bytes.max(host_payload_bytes);
            let dense = DeviceBuffer::from_host(stream.ctx(), stream, &dense_host)?;
            dense_allocations.push(Arc::new(dense));
        }

        let mut logical = BTreeMap::new();
        for (name, binding) in &self.bindings {
            let allocation = dense_allocations
                .get(binding.allocation_index)
                .ok_or_else(|| {
                    NnisError::invalid_input(
                        "INT2 dense logical binding references a missing allocation",
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
                    "duplicate INT2 dense logical binding {name}"
                )));
            }
        }
        stream.synchronize()?;
        let weights = ModelWeights::from_named_logical_tensors(config, logical)?;
        Ok((weights, peak_host_temporary_payload_bytes))
    }

    /// Execute one explicitly bound matrix projection directly from packed INT2 storage.
    pub fn execute_projection(
        &self,
        plan: &Int2ReferenceProjectionPlanV1,
        kernel: &F32Int2Gemv,
        stream: &Stream,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        plan.validate()?;
        self.validate_resident_allocations()?;
        let binding = self.bindings.get(plan.logical_weight()).ok_or_else(|| {
            NnisError::invalid_input(format!(
                "INT2 projection logical weight {:?} is not present in storage",
                plan.logical_weight()
            ))
        })?;
        let (rows, cols) = match binding.shape {
            WeightLogicalShapeV1::Matrix { rows, cols } => (rows, cols),
            WeightLogicalShapeV1::Vector { .. } => {
                return Err(NnisError::invalid_input(format!(
                    "INT2 logical weight {:?} is a vector, not a projection matrix",
                    plan.logical_weight()
                )));
            }
        };
        let plan_rows = usize::try_from(plan.rows())
            .map_err(|_| NnisError::invalid_input("INT2 projection rows do not fit usize"))?;
        let plan_cols = usize::try_from(plan.cols())
            .map_err(|_| NnisError::invalid_input("INT2 projection cols do not fit usize"))?;
        if rows != plan_rows || cols != plan_cols {
            return Err(NnisError::invalid_input(format!(
                "INT2 projection plan shape ({plan_rows}, {plan_cols}) does not match logical weight {:?} shape ({rows}, {cols})",
                plan.logical_weight()
            )));
        }
        let allocation = self
            .allocations
            .get(binding.allocation_index)
            .ok_or_else(|| {
                NnisError::invalid_input("INT2 projection binding references a missing allocation")
            })?;
        kernel.project_kn(
            stream,
            input,
            &allocation.packed_values,
            &allocation.scale,
            output,
            rows,
            cols,
        )
    }

    /// Reconcile live CUDA allocations with the immutable accounting summary.
    pub fn validate_resident_allocations(&self) -> Result<()> {
        let expected_bindings = usize::try_from(self.summary.logical_tensor_references)
            .map_err(|_| NnisError::invalid_input("INT2 logical binding count exceeds usize"))?;
        if self.bindings.len() != expected_bindings {
            return Err(NnisError::invalid_input(
                "INT2 logical binding count disagrees with storage summary",
            ));
        }
        if self.allocations.len() != self.summary.allocations.len() {
            return Err(NnisError::invalid_input(
                "INT2 live allocation count disagrees with summary",
            ));
        }
        let mut payload_total = 0_u64;
        let mut scale_total = 0_u64;
        for (allocation, summary) in self.allocations.iter().zip(&self.summary.allocations) {
            let payload_bytes = u64::try_from(allocation.packed_values.size_bytes())
                .map_err(|_| NnisError::invalid_input("INT2 live payload bytes exceed u64"))?;
            let scale_bytes = u64::try_from(allocation.scale.size_bytes())
                .map_err(|_| NnisError::invalid_input("INT2 live scale bytes exceed u64"))?;
            if payload_bytes != summary.resident_payload_bytes
                || scale_bytes != summary.resident_scale_bytes
                || payload_bytes
                    .checked_add(scale_bytes)
                    .ok_or_else(|| NnisError::invalid_input("INT2 live bytes overflow u64"))?
                    != summary.resident_total_bytes
            {
                return Err(NnisError::invalid_input(
                    "INT2 live allocation bytes disagree with summary",
                ));
            }
            checked_add(&mut payload_total, payload_bytes, "INT2 live payload total")?;
            checked_add(&mut scale_total, scale_bytes, "INT2 live scale total")?;
        }
        if payload_total != self.summary.resident_payload_bytes
            || scale_total != self.summary.resident_scale_bytes
            || payload_total
                .checked_add(scale_total)
                .ok_or_else(|| NnisError::invalid_input("INT2 live total bytes overflow u64"))?
                != self.summary.resident_device_bytes
        {
            return Err(NnisError::invalid_input(
                "INT2 live allocation totals disagree with summary",
            ));
        }
        Ok(())
    }
}

fn checked_add(counter: &mut u64, value: u64, label: &str) -> Result<()> {
    *counter = counter
        .checked_add(value)
        .ok_or_else(|| NnisError::invalid_input(format!("{label} overflows u64")))?;
    Ok(())
}
