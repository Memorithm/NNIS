//! Deterministic INT4 reference weight storage for NNIS.
//!
//! This module owns a narrowly scoped weight-only representation experiment.
//! It quantizes each unique F32 device allocation independently with one
//! symmetric F32 scale and stores two signed INT4 values per byte. The packed
//! payload and scale are materialized as real CUDA allocations so resident
//! bytes are counted from the buffers that actually exist.
//!
//! This is not an execution format. No NNIS projection, embedding, attention,
//! MLP, or LM-head kernel consumes these buffers yet. The type deliberately
//! exposes accounting and reconstruction evidence but no device-buffer
//! accessor that could be mistaken for an executable model path.

use crate::{DeviceTensor, ModelWeights};
use nnis_rt::{DeviceBuffer, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Version of the deterministic INT4 reference storage contract.
pub const NNIS_INT4_REFERENCE_STORAGE_VERSION: u32 = 1;
/// Canonical per-allocation serialized header: magic + element count + F32 scale.
pub const NNIS_INT4_REFERENCE_SERIALIZED_HEADER_BYTES: u64 = 16;
/// Smallest quantized value emitted by the symmetric reference quantizer.
pub const NNIS_INT4_REFERENCE_QUANT_MIN: i8 = -7;
/// Largest quantized value emitted by the symmetric reference quantizer.
pub const NNIS_INT4_REFERENCE_QUANT_MAX: i8 = 7;

const SERIALIZED_MAGIC: [u8; 4] = *b"NI41";

/// Host-side deterministic quantization result for one unique logical tensor allocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Int4ReferenceQuantizedTensorV1 {
    pub element_count: u64,
    pub scale: f32,
    pub packed_values: Vec<u8>,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
}

impl Int4ReferenceQuantizedTensorV1 {
    /// Canonical bytes owned by this reference representation.
    ///
    /// Layout:
    /// - 4 bytes: ASCII magic `NI41`;
    /// - 8 bytes: little-endian logical element count;
    /// - 4 bytes: little-endian IEEE-754 F32 scale bits;
    /// - remaining bytes: two signed four-bit two's-complement values per byte,
    ///   low nibble first.
    pub fn canonical_serialized_bytes(&self) -> Result<Vec<u8>> {
        validate_quantized_tensor(self)?;
        let payload_len = self.packed_values.len();
        let header_len = usize::try_from(NNIS_INT4_REFERENCE_SERIALIZED_HEADER_BYTES)
            .map_err(|_| NnisError::invalid_input("INT4 serialized header does not fit usize"))?;
        let capacity = header_len
            .checked_add(payload_len)
            .ok_or_else(|| NnisError::invalid_input("INT4 serialized size overflows usize"))?;
        let mut encoded = Vec::with_capacity(capacity);
        encoded.extend_from_slice(&SERIALIZED_MAGIC);
        encoded.extend_from_slice(&self.element_count.to_le_bytes());
        encoded.extend_from_slice(&self.scale.to_bits().to_le_bytes());
        encoded.extend_from_slice(&self.packed_values);
        Ok(encoded)
    }
}

/// Quantize one F32 slice with deterministic symmetric signed INT4.
///
/// The reference deliberately uses one scale for one unique source allocation.
/// This is a simple fixed baseline, not a claim that per-tensor scaling is an
/// optimal production quantizer. `-8` is reserved and never emitted.
pub fn quantize_int4_symmetric_reference_v1(
    values: &[f32],
) -> Result<Int4ReferenceQuantizedTensorV1> {
    if values.is_empty() {
        return Err(NnisError::invalid_input(
            "INT4 reference quantization requires at least one value",
        ));
    }

    let mut max_abs = 0.0_f32;
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(NnisError::invalid_input(format!(
                "INT4 reference source value {index} is not finite"
            )));
        }
        max_abs = max_abs.max(value.abs());
    }

    let scale = if max_abs == 0.0 {
        1.0
    } else {
        max_abs / f32::from(NNIS_INT4_REFERENCE_QUANT_MAX)
    };
    if !scale.is_finite() || scale <= 0.0 {
        return Err(NnisError::invalid_input(
            "INT4 reference scale is not a finite positive F32 value",
        ));
    }

    let packed_len = values
        .len()
        .checked_add(1)
        .ok_or_else(|| NnisError::invalid_input("INT4 element count overflows usize"))?
        / 2;
    let mut packed_values = vec![0_u8; packed_len];
    let mut max_abs_error = 0.0_f32;
    let mut squared_error_sum = 0.0_f64;

    for (index, value) in values.iter().copied().enumerate() {
        let scaled = (value / scale).round();
        let quantized = scaled
            .max(f32::from(NNIS_INT4_REFERENCE_QUANT_MIN))
            .min(f32::from(NNIS_INT4_REFERENCE_QUANT_MAX)) as i8;
        let nibble = (quantized as u8) & 0x0f;
        let slot = &mut packed_values[index / 2];
        if index % 2 == 0 {
            *slot |= nibble;
        } else {
            *slot |= nibble << 4;
        }

        let reconstructed = f32::from(quantized) * scale;
        let error = (value - reconstructed).abs();
        max_abs_error = max_abs_error.max(error);
        let error_f64 = f64::from(value) - f64::from(reconstructed);
        squared_error_sum += error_f64 * error_f64;
    }

    let element_count = u64::try_from(values.len())
        .map_err(|_| NnisError::invalid_input("INT4 element count exceeds u64"))?;
    let quantized = Int4ReferenceQuantizedTensorV1 {
        element_count,
        scale,
        packed_values,
        max_abs_error,
        mean_squared_error: squared_error_sum / values.len() as f64,
    };
    validate_quantized_tensor(&quantized)?;
    Ok(quantized)
}

/// Reconstruct F32 values from the deterministic reference representation.
pub fn dequantize_int4_symmetric_reference_v1(
    quantized: &Int4ReferenceQuantizedTensorV1,
) -> Result<Vec<f32>> {
    validate_quantized_tensor(quantized)?;
    let element_count = usize::try_from(quantized.element_count)
        .map_err(|_| NnisError::invalid_input("INT4 element count does not fit usize"))?;
    let mut values = Vec::with_capacity(element_count);
    for index in 0..element_count {
        let byte = quantized.packed_values[index / 2];
        let nibble = if index % 2 == 0 {
            byte & 0x0f
        } else {
            (byte >> 4) & 0x0f
        };
        let signed = if nibble & 0x08 != 0 {
            (nibble as i8) - 16
        } else {
            nibble as i8
        };
        if signed == -8 {
            return Err(NnisError::invalid_input(
                "INT4 reference payload contains reserved -8 code",
            ));
        }
        values.push(f32::from(signed) * quantized.scale);
    }
    Ok(values)
}

fn validate_quantized_tensor(quantized: &Int4ReferenceQuantizedTensorV1) -> Result<()> {
    if quantized.element_count == 0 {
        return Err(NnisError::invalid_input(
            "INT4 reference tensor has zero logical elements",
        ));
    }
    if !quantized.scale.is_finite() || quantized.scale <= 0.0 {
        return Err(NnisError::invalid_input(
            "INT4 reference tensor scale must be finite and positive",
        ));
    }
    if !quantized.max_abs_error.is_finite()
        || !quantized.mean_squared_error.is_finite()
        || quantized.max_abs_error < 0.0
        || quantized.mean_squared_error < 0.0
    {
        return Err(NnisError::invalid_input(
            "INT4 reference reconstruction evidence is invalid",
        ));
    }
    let element_count = usize::try_from(quantized.element_count)
        .map_err(|_| NnisError::invalid_input("INT4 element count does not fit usize"))?;
    let expected_payload = element_count
        .checked_add(1)
        .ok_or_else(|| NnisError::invalid_input("INT4 payload size overflows usize"))?
        / 2;
    if quantized.packed_values.len() != expected_payload {
        return Err(NnisError::invalid_input(format!(
            "INT4 packed payload has {} bytes; expected {expected_payload}",
            quantized.packed_values.len()
        )));
    }
    for index in 0..element_count {
        let byte = quantized.packed_values[index / 2];
        let nibble = if index % 2 == 0 {
            byte & 0x0f
        } else {
            (byte >> 4) & 0x0f
        };
        if nibble == 0x08 {
            return Err(NnisError::invalid_input(
                "INT4 reference payload contains reserved -8 code",
            ));
        }
    }
    Ok(())
}

/// Accounting and reconstruction evidence for one unique source allocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Int4ReferenceAllocationSummaryV1 {
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

/// Exact owned-storage summary for the non-executable full-model INT4 reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Int4ReferenceStorageSummaryV1 {
    pub schema_version: u32,
    pub representation: String,
    pub quantization: String,
    pub scale_scope: String,
    pub quant_min: i8,
    pub quant_max: i8,
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
    pub source_f32_to_resident_int4_byte_ratio: f64,
    pub max_abs_error: f32,
    pub mean_squared_error: f64,
    pub allocations: Vec<Int4ReferenceAllocationSummaryV1>,
}

struct Int4ReferenceDeviceAllocation {
    packed_values: DeviceBuffer<u8>,
    scale: DeviceBuffer<f32>,
}

/// Live CUDA allocations backing the reference INT4 storage.
///
/// The packed buffers are intentionally private. Until a separately qualified
/// execution contract exists, callers can inspect evidence but cannot ask NNIS
/// to execute a model with this storage.
pub struct Int4ReferenceModelStorageV1 {
    allocations: Vec<Int4ReferenceDeviceAllocation>,
    summary: Int4ReferenceStorageSummaryV1,
}

impl Int4ReferenceModelStorageV1 {
    /// Quantize every unique F32 model-weight allocation and materialize the
    /// packed payload plus one F32 scale as real CUDA allocations.
    pub fn from_f32_model_weights(weights: &ModelWeights, stream: &Stream) -> Result<Self> {
        let source_summary = weights.weight_allocation_summary_v1()?;
        let mut source_indices = BTreeMap::<u64, usize>::new();
        let mut allocations = Vec::<Int4ReferenceDeviceAllocation>::new();
        let mut summaries = Vec::<Int4ReferenceAllocationSummaryV1>::new();

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

        visit_weight_tensors(weights, |name, tensor| {
            let (source_key, source_bytes) = match tensor {
                DeviceTensor::F32(buffer) => (buffer.device_ptr(), buffer.size_bytes()),
                DeviceTensor::Bf16(_) => {
                    return Err(NnisError::unsupported(format!(
                        "INT4 reference storage requires F32 source weights; {name} is BF16"
                    )));
                }
            };
            if source_key == 0 || tensor.is_empty() {
                return Err(NnisError::invalid_input(format!(
                    "INT4 source weight {name} has no live allocation"
                )));
            }
            let logical_values = u64::try_from(tensor.len()).map_err(|_| {
                NnisError::invalid_input(format!("INT4 source weight {name} length exceeds u64"))
            })?;
            checked_add(
                &mut logical_tensor_references,
                1,
                "INT4 logical tensor reference count",
            )?;
            checked_add(
                &mut logical_element_references,
                logical_values,
                "INT4 logical element reference count",
            )?;

            if let Some(index) = source_indices.get(&source_key).copied() {
                let summary = summaries.get_mut(index).ok_or_else(|| {
                    NnisError::invalid_input("INT4 source allocation index escaped summary")
                })?;
                if summary.logical_values != logical_values
                    || summary.source_f32_bytes
                        != u64::try_from(source_bytes).map_err(|_| {
                            NnisError::invalid_input("INT4 source byte count exceeds u64")
                        })?
                {
                    return Err(NnisError::invalid_input(format!(
                        "INT4 source alias {name} disagrees with its original allocation"
                    )));
                }
                summary.logical_names.push(name.to_string());
                return Ok(());
            }

            let host = tensor.as_f32()?.to_vec(stream)?;
            let quantized = quantize_int4_symmetric_reference_v1(&host)?;
            let canonical = quantized.canonical_serialized_bytes()?;
            let packed_values =
                DeviceBuffer::from_host(weights.context(), stream, &quantized.packed_values)?;
            let scale_values = [quantized.scale];
            let scale = DeviceBuffer::from_host(weights.context(), stream, &scale_values)?;

            let payload_bytes = u64::try_from(packed_values.size_bytes())
                .map_err(|_| NnisError::invalid_input("INT4 payload byte count exceeds u64"))?;
            let scale_bytes = u64::try_from(scale.size_bytes())
                .map_err(|_| NnisError::invalid_input("INT4 scale byte count exceeds u64"))?;
            let allocation_resident_bytes = payload_bytes
                .checked_add(scale_bytes)
                .ok_or_else(|| NnisError::invalid_input("INT4 resident bytes overflow u64"))?;
            let allocation_serialized_bytes = u64::try_from(canonical.len())
                .map_err(|_| NnisError::invalid_input("INT4 serialized bytes exceed u64"))?;
            let source_bytes = u64::try_from(source_bytes)
                .map_err(|_| NnisError::invalid_input("INT4 source byte count exceeds u64"))?;
            let canonical_payload_bytes = u64::try_from(quantized.packed_values.len())
                .map_err(|_| NnisError::invalid_input("INT4 payload length exceeds u64"))?;
            let expected_serialized = NNIS_INT4_REFERENCE_SERIALIZED_HEADER_BYTES
                .checked_add(canonical_payload_bytes)
                .ok_or_else(|| NnisError::invalid_input("INT4 serialized bytes overflow u64"))?;
            if allocation_serialized_bytes != expected_serialized {
                return Err(NnisError::invalid_input(
                    "INT4 canonical serialization length disagrees with accounting",
                ));
            }
            if payload_bytes != canonical_payload_bytes {
                return Err(NnisError::invalid_input(
                    "INT4 device payload bytes disagree with packed host payload bytes",
                ));
            }

            let allocation_index = u32::try_from(allocations.len()).map_err(|_| {
                NnisError::invalid_input("INT4 allocation count exceeds u32 contract capacity")
            })?;
            let summary_index = summaries.len();
            source_indices.insert(source_key, summary_index);
            allocations.push(Int4ReferenceDeviceAllocation {
                packed_values,
                scale,
            });
            summaries.push(Int4ReferenceAllocationSummaryV1 {
                allocation_index,
                logical_names: vec![name.to_string()],
                logical_values,
                source_f32_bytes: source_bytes,
                serialized_header_bytes: NNIS_INT4_REFERENCE_SERIALIZED_HEADER_BYTES,
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
                "INT4 unique logical value count",
            )?;
            checked_add(
                &mut source_f32_owned_bytes,
                source_bytes,
                "INT4 source owned bytes",
            )?;
            checked_add(
                &mut serialized_header_bytes,
                NNIS_INT4_REFERENCE_SERIALIZED_HEADER_BYTES,
                "INT4 serialized header bytes",
            )?;
            checked_add(
                &mut serialized_payload_bytes,
                canonical_payload_bytes,
                "INT4 serialized payload bytes",
            )?;
            checked_add(
                &mut serialized_total_bytes,
                allocation_serialized_bytes,
                "INT4 serialized total bytes",
            )?;
            checked_add(
                &mut resident_payload_bytes,
                payload_bytes,
                "INT4 resident payload bytes",
            )?;
            checked_add(
                &mut resident_scale_bytes,
                scale_bytes,
                "INT4 resident scale bytes",
            )?;
            checked_add(
                &mut resident_device_bytes,
                allocation_resident_bytes,
                "INT4 resident device bytes",
            )?;
            max_abs_error = max_abs_error.max(quantized.max_abs_error);
            weighted_squared_error += quantized.mean_squared_error * logical_values as f64;
            Ok(())
        })?;
        stream.synchronize()?;

        if unique_logical_values == 0 {
            return Err(NnisError::invalid_input(
                "INT4 reference model contains no unique logical values",
            ));
        }
        if source_summary.logical_tensor_references != logical_tensor_references
            || source_summary.logical_element_references != logical_element_references
            || source_summary.unique_device_allocations
                != u64::try_from(allocations.len())
                    .map_err(|_| NnisError::invalid_input("INT4 allocation count exceeds u64"))?
            || source_summary.unique_device_elements != unique_logical_values
            || source_summary.owned_device_allocation_bytes != source_f32_owned_bytes
        {
            return Err(NnisError::invalid_input(
                "INT4 source accounting does not reconcile with WeightAllocationSummaryV1",
            ));
        }

        let serialized_bits_per_unique_logical_value =
            serialized_total_bytes as f64 * 8.0 / unique_logical_values as f64;
        let resident_bits_per_unique_logical_value =
            resident_device_bytes as f64 * 8.0 / unique_logical_values as f64;
        let source_f32_to_resident_int4_byte_ratio =
            source_f32_owned_bytes as f64 / resident_device_bytes as f64;
        let mean_squared_error = weighted_squared_error / unique_logical_values as f64;

        let storage = Self {
            allocations,
            summary: Int4ReferenceStorageSummaryV1 {
                schema_version: NNIS_INT4_REFERENCE_STORAGE_VERSION,
                representation: "symmetric-signed-int4-reference".to_string(),
                quantization: "round-to-nearest-clamped-minus7-plus7".to_string(),
                scale_scope: "one-f32-scale-per-unique-source-allocation".to_string(),
                quant_min: NNIS_INT4_REFERENCE_QUANT_MIN,
                quant_max: NNIS_INT4_REFERENCE_QUANT_MAX,
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
                source_f32_to_resident_int4_byte_ratio,
                max_abs_error,
                mean_squared_error,
                allocations: summaries,
            },
        };
        storage.validate_resident_allocations()?;
        Ok(storage)
    }

    #[must_use]
    pub fn summary(&self) -> &Int4ReferenceStorageSummaryV1 {
        &self.summary
    }

    /// Reconcile the retained CUDA allocations with the immutable accounting summary.
    pub fn validate_resident_allocations(&self) -> Result<()> {
        if self.allocations.len() != self.summary.allocations.len() {
            return Err(NnisError::invalid_input(
                "INT4 live allocation count disagrees with summary",
            ));
        }
        let mut payload_total = 0_u64;
        let mut scale_total = 0_u64;
        for (allocation, summary) in self.allocations.iter().zip(&self.summary.allocations) {
            let payload_bytes = u64::try_from(allocation.packed_values.size_bytes())
                .map_err(|_| NnisError::invalid_input("INT4 live payload bytes exceed u64"))?;
            let scale_bytes = u64::try_from(allocation.scale.size_bytes())
                .map_err(|_| NnisError::invalid_input("INT4 live scale bytes exceed u64"))?;
            if payload_bytes != summary.resident_payload_bytes
                || scale_bytes != summary.resident_scale_bytes
                || payload_bytes
                    .checked_add(scale_bytes)
                    .ok_or_else(|| NnisError::invalid_input("INT4 live bytes overflow u64"))?
                    != summary.resident_total_bytes
            {
                return Err(NnisError::invalid_input(
                    "INT4 live allocation bytes disagree with summary",
                ));
            }
            checked_add(&mut payload_total, payload_bytes, "INT4 live payload total")?;
            checked_add(&mut scale_total, scale_bytes, "INT4 live scale total")?;
        }
        if payload_total != self.summary.resident_payload_bytes
            || scale_total != self.summary.resident_scale_bytes
            || payload_total
                .checked_add(scale_total)
                .ok_or_else(|| NnisError::invalid_input("INT4 live total bytes overflow u64"))?
                != self.summary.resident_device_bytes
        {
            return Err(NnisError::invalid_input(
                "INT4 live allocation totals disagree with summary",
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

fn visit_weight_tensors(
    weights: &ModelWeights,
    mut visit: impl FnMut(&str, &DeviceTensor) -> Result<()>,
) -> Result<()> {
    visit("token_embedding", weights.token_embedding.tensor())?;
    for (index, layer) in weights.layers.iter().enumerate() {
        visit(
            &format!("layers.{index}.input_norm"),
            layer.input_norm.tensor(),
        )?;
        visit(&format!("layers.{index}.q_proj"), layer.q_proj.tensor())?;
        visit(&format!("layers.{index}.k_proj"), layer.k_proj.tensor())?;
        visit(&format!("layers.{index}.v_proj"), layer.v_proj.tensor())?;
        visit(&format!("layers.{index}.o_proj"), layer.o_proj.tensor())?;
        visit(
            &format!("layers.{index}.post_attention_norm"),
            layer.post_attention_norm.tensor(),
        )?;
        visit(
            &format!("layers.{index}.gate_proj"),
            layer.gate_proj.tensor(),
        )?;
        visit(&format!("layers.{index}.up_proj"), layer.up_proj.tensor())?;
        visit(
            &format!("layers.{index}.down_proj"),
            layer.down_proj.tensor(),
        )?;
    }
    visit("final_norm", weights.final_norm.tensor())?;
    visit("lm_head", weights.lm_head.tensor())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_int4_known_values_pack_and_reconstruct_deterministically() {
        let values = [-1.0_f32, -0.6, 0.0, 0.6, 1.0];
        let quantized = quantize_int4_symmetric_reference_v1(&values).unwrap();
        assert_eq!(quantized.element_count, 5);
        assert_eq!(quantized.packed_values.len(), 3);
        assert_eq!(quantized.packed_values[0], 0xc9);
        assert_eq!(quantized.packed_values[1], 0x40);
        assert_eq!(quantized.packed_values[2], 0x07);
        let reconstructed = dequantize_int4_symmetric_reference_v1(&quantized).unwrap();
        assert_eq!(reconstructed.len(), values.len());
        assert_eq!(reconstructed[0], -1.0);
        assert_eq!(reconstructed[2], 0.0);
        assert_eq!(reconstructed[4], 1.0);
        assert!(quantized.max_abs_error > 0.0);
        assert!(quantized.mean_squared_error > 0.0);
    }

    #[test]
    fn zero_tensor_uses_finite_scale_and_roundtrips_exactly() {
        let quantized = quantize_int4_symmetric_reference_v1(&[0.0_f32; 5]).unwrap();
        assert_eq!(quantized.scale, 1.0);
        assert_eq!(quantized.packed_values, vec![0, 0, 0]);
        assert_eq!(quantized.max_abs_error, 0.0);
        assert_eq!(quantized.mean_squared_error, 0.0);
        assert_eq!(
            dequantize_int4_symmetric_reference_v1(&quantized).unwrap(),
            vec![0.0; 5]
        );
    }

    #[test]
    fn canonical_serialization_accounts_header_and_payload_exactly() {
        let quantized = quantize_int4_symmetric_reference_v1(&[1.0_f32, -1.0, 0.25]).unwrap();
        let encoded = quantized.canonical_serialized_bytes().unwrap();
        assert_eq!(&encoded[..4], b"NI41");
        assert_eq!(
            encoded.len() as u64,
            NNIS_INT4_REFERENCE_SERIALIZED_HEADER_BYTES + quantized.packed_values.len() as u64
        );
        assert_eq!(
            u64::from_le_bytes(encoded[4..12].try_into().unwrap()),
            quantized.element_count
        );
        assert_eq!(
            u32::from_le_bytes(encoded[12..16].try_into().unwrap()),
            quantized.scale.to_bits()
        );
    }

    #[test]
    fn invalid_sources_and_reserved_code_fail_closed() {
        assert!(quantize_int4_symmetric_reference_v1(&[]).is_err());
        assert!(quantize_int4_symmetric_reference_v1(&[f32::NAN]).is_err());
        assert!(quantize_int4_symmetric_reference_v1(&[f32::INFINITY]).is_err());

        let invalid = Int4ReferenceQuantizedTensorV1 {
            element_count: 1,
            scale: 1.0,
            packed_values: vec![0x08],
            max_abs_error: 0.0,
            mean_squared_error: 0.0,
        };
        assert!(dequantize_int4_symmetric_reference_v1(&invalid).is_err());
    }
}
