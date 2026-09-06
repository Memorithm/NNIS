use crate::config::{ModelConfig, WeightDType};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

pub const NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION: u32 = 1;

/// Numeric storage of one device allocation in the weight-accounting contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightAllocationDTypeV1 {
    F32,
    Bf16,
    F16,
}

/// One unique device allocation referenced by one or more logical model weights.
///
/// `allocation_index` is deterministic within the summary and intentionally does
/// not expose the process-local CUDA address used to deduplicate aliases.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightAllocationSegmentV1 {
    pub allocation_index: u32,
    pub dtype: WeightAllocationDTypeV1,
    pub elements: u64,
    pub bytes: u64,
    pub logical_names: Vec<String>,
}

/// Exact accounting of device allocations owned by a model weight graph.
///
/// This reports bytes passed to `cuMemAlloc` and still owned by the weight graph.
/// It is not a measurement of physical page residency, CUDA allocator overhead,
/// process-wide VRAM use, temporary conversion memory, KV state, workspaces, or
/// kernel/module storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightAllocationSummaryV1 {
    pub schema_version: u32,
    pub logical_tensor_references: u64,
    pub logical_element_references: u64,
    pub unique_device_allocations: u64,
    pub unique_device_elements: u64,
    pub owned_device_allocation_bytes: u64,
    pub segments: Vec<WeightAllocationSegmentV1>,
}

#[derive(Debug, Clone)]
pub(crate) struct WeightAllocationObservation {
    pub(crate) logical_name: String,
    pub(crate) allocation_key: u64,
    pub(crate) dtype: WeightAllocationDTypeV1,
    pub(crate) elements: u64,
    pub(crate) bytes: u64,
}

fn checked_add(counter: &mut u64, value: u64, label: &str) -> Result<()> {
    *counter = counter
        .checked_add(value)
        .ok_or_else(|| NnisError::invalid_input(format!("{label} overflows u64")))?;
    Ok(())
}

pub(crate) fn summarize_weight_allocations(
    observations: impl IntoIterator<Item = WeightAllocationObservation>,
) -> Result<WeightAllocationSummaryV1> {
    let mut logical_tensor_references = 0_u64;
    let mut logical_element_references = 0_u64;
    let mut unique_device_elements = 0_u64;
    let mut owned_device_allocation_bytes = 0_u64;
    let mut allocation_indices = BTreeMap::<u64, usize>::new();
    let mut segments = Vec::<WeightAllocationSegmentV1>::new();

    for observation in observations {
        checked_add(
            &mut logical_tensor_references,
            1,
            "logical tensor reference count",
        )?;
        checked_add(
            &mut logical_element_references,
            observation.elements,
            "logical element reference count",
        )?;

        if let Some(&segment_index) = allocation_indices.get(&observation.allocation_key) {
            let segment = segments.get_mut(segment_index).ok_or_else(|| {
                NnisError::invalid_input("weight allocation index points outside summary")
            })?;
            if segment.dtype != observation.dtype
                || segment.elements != observation.elements
                || segment.bytes != observation.bytes
            {
                return Err(NnisError::invalid_input(format!(
                    "weight allocation alias {:?} disagrees with the original allocation contract",
                    observation.logical_name
                )));
            }
            segment.logical_names.push(observation.logical_name);
            continue;
        }

        let allocation_index = u32::try_from(segments.len()).map_err(|_| {
            NnisError::invalid_input("weight allocation count exceeds u32 contract capacity")
        })?;
        checked_add(
            &mut unique_device_elements,
            observation.elements,
            "unique device element count",
        )?;
        checked_add(
            &mut owned_device_allocation_bytes,
            observation.bytes,
            "owned device allocation bytes",
        )?;
        allocation_indices.insert(observation.allocation_key, segments.len());
        segments.push(WeightAllocationSegmentV1 {
            allocation_index,
            dtype: observation.dtype,
            elements: observation.elements,
            bytes: observation.bytes,
            logical_names: vec![observation.logical_name],
        });
    }

    let unique_device_allocations = u64::try_from(segments.len()).map_err(|_| {
        NnisError::invalid_input("weight allocation count exceeds u64 contract capacity")
    })?;
    Ok(WeightAllocationSummaryV1 {
        schema_version: NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION,
        logical_tensor_references,
        logical_element_references,
        unique_device_allocations,
        unique_device_elements,
        owned_device_allocation_bytes,
        segments,
    })
}

/// Device-resident tensor storage in the numeric formats NNIS currently owns.
#[derive(Debug, Clone)]
pub enum DeviceTensor {
    F32(Arc<DeviceBuffer<f32>>),
    Bf16(Arc<DeviceBuffer<u16>>),
}

impl DeviceTensor {
    pub fn dtype(&self) -> WeightDType {
        match self {
            Self::F32(_) => WeightDType::F32,
            Self::Bf16(_) => WeightDType::Bf16,
        }
    }

    fn allocation_dtype(&self) -> WeightAllocationDTypeV1 {
        match self {
            Self::F32(_) => WeightAllocationDTypeV1::F32,
            Self::Bf16(_) => WeightAllocationDTypeV1::Bf16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(buffer) => buffer.len(),
            Self::Bf16(buffer) => buffer.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn allocation_size_bytes(&self) -> usize {
        match self {
            Self::F32(buffer) => buffer.size_bytes(),
            Self::Bf16(buffer) => buffer.size_bytes(),
        }
    }

    fn allocation_key(&self) -> u64 {
        match self {
            Self::F32(buffer) => buffer.device_ptr(),
            Self::Bf16(buffer) => buffer.device_ptr(),
        }
    }

    pub fn context(&self) -> &Arc<Context> {
        match self {
            Self::F32(buffer) => buffer.ctx(),
            Self::Bf16(buffer) => buffer.ctx(),
        }
    }

    pub fn as_f32(&self) -> Result<&DeviceBuffer<f32>> {
        match self {
            Self::F32(buffer) => Ok(buffer),
            Self::Bf16(_) => Err(NnisError::unsupported(
                "this decoder execution path currently requires f32 weights",
            )),
        }
    }
}

/// Row-major matrix with explicit logical dimensions.
#[derive(Debug)]
pub struct MatrixWeight {
    tensor: DeviceTensor,
    rows: usize,
    cols: usize,
}

impl MatrixWeight {
    pub fn new(tensor: DeviceTensor, rows: usize, cols: usize) -> Result<Self> {
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| NnisError::invalid_input("matrix weight shape overflows usize"))?;
        if tensor.len() != expected {
            return Err(NnisError::invalid_input(format!(
                "matrix weight shape ({rows}, {cols}) requires {expected} elements; got {}",
                tensor.len()
            )));
        }
        Ok(Self { tensor, rows, cols })
    }

    pub fn tensor(&self) -> &DeviceTensor {
        &self.tensor
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// Per-channel weight vector.
#[derive(Debug)]
pub struct VectorWeight {
    tensor: DeviceTensor,
    len: usize,
}

impl VectorWeight {
    pub fn new(tensor: DeviceTensor, len: usize) -> Result<Self> {
        if tensor.len() != len {
            return Err(NnisError::invalid_input(format!(
                "vector weight requires {len} elements; got {}",
                tensor.len()
            )));
        }
        Ok(Self { tensor, len })
    }

    pub fn tensor(&self) -> &DeviceTensor {
        &self.tensor
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// One reusable pre-norm decoder block's weights.
#[derive(Debug)]
pub struct DecoderLayerWeights {
    pub input_norm: VectorWeight,
    /// Internal GEMM orientation `[hidden, q_width]`.
    pub q_proj: MatrixWeight,
    /// Internal GEMM orientation `[hidden, kv_width]`.
    pub k_proj: MatrixWeight,
    pub v_proj: MatrixWeight,
    /// Internal GEMM orientation `[hidden, hidden]`.
    pub o_proj: MatrixWeight,
    pub post_attention_norm: VectorWeight,
    /// Internal GEMM orientation `[hidden, intermediate]`.
    pub gate_proj: MatrixWeight,
    pub up_proj: MatrixWeight,
    /// Internal GEMM orientation `[intermediate, hidden]`.
    pub down_proj: MatrixWeight,
}

/// Complete model-neutral decoder weight graph.
#[derive(Debug)]
pub struct ModelWeights {
    /// Row-major `[vocab, hidden]` lookup table.
    pub token_embedding: MatrixWeight,
    pub layers: Vec<DecoderLayerWeights>,
    pub final_norm: VectorWeight,
    /// Internal GEMM orientation `[hidden, vocab]`.
    pub lm_head: MatrixWeight,
}

impl ModelWeights {
    pub fn validate(&self, config: &ModelConfig) -> Result<()> {
        config.validate()?;
        if self.layers.len() != config.num_hidden_layers {
            return Err(NnisError::invalid_input(format!(
                "model has {} decoder layers; config requires {}",
                self.layers.len(),
                config.num_hidden_layers
            )));
        }
        Self::expect_matrix(
            "token_embedding",
            &self.token_embedding,
            config.vocab_size,
            config.hidden_size,
        )?;
        Self::expect_vector("final_norm", &self.final_norm, config.hidden_size)?;
        Self::expect_matrix(
            "lm_head",
            &self.lm_head,
            config.hidden_size,
            config.vocab_size,
        )?;

        let kv_width = config.key_value_width()?;
        for (index, layer) in self.layers.iter().enumerate() {
            Self::expect_vector(
                &format!("layers.{index}.input_norm"),
                &layer.input_norm,
                config.hidden_size,
            )?;
            Self::expect_matrix(
                &format!("layers.{index}.q_proj"),
                &layer.q_proj,
                config.hidden_size,
                config.hidden_size,
            )?;
            for (name, weight) in [("k_proj", &layer.k_proj), ("v_proj", &layer.v_proj)] {
                Self::expect_matrix(
                    &format!("layers.{index}.{name}"),
                    weight,
                    config.hidden_size,
                    kv_width,
                )?;
            }
            Self::expect_matrix(
                &format!("layers.{index}.o_proj"),
                &layer.o_proj,
                config.hidden_size,
                config.hidden_size,
            )?;
            Self::expect_vector(
                &format!("layers.{index}.post_attention_norm"),
                &layer.post_attention_norm,
                config.hidden_size,
            )?;
            for (name, weight) in [("gate_proj", &layer.gate_proj), ("up_proj", &layer.up_proj)] {
                Self::expect_matrix(
                    &format!("layers.{index}.{name}"),
                    weight,
                    config.hidden_size,
                    config.intermediate_size,
                )?;
            }
            Self::expect_matrix(
                &format!("layers.{index}.down_proj"),
                &layer.down_proj,
                config.intermediate_size,
                config.hidden_size,
            )?;
        }

        let expected_dtype = config.weight_dtype;
        let context = self.token_embedding.tensor().context();
        self.for_each_tensor(|name, tensor| {
            if tensor.dtype() != expected_dtype {
                return Err(NnisError::invalid_input(format!(
                    "weight {name} uses {:?}; config requires {:?}",
                    tensor.dtype(),
                    expected_dtype
                )));
            }
            if !Arc::ptr_eq(context, tensor.context()) {
                return Err(NnisError::invalid_input(format!(
                    "weight {name} belongs to a different CUDA context"
                )));
            }
            Ok(())
        })
    }

    pub fn context(&self) -> &Arc<Context> {
        self.token_embedding.tensor().context()
    }

    /// Build an exact summary of the device allocations currently owned by
    /// this weight graph. Aliases of the same live `DeviceBuffer` are counted
    /// once in `owned_device_allocation_bytes` and retained as logical names.
    pub fn weight_allocation_summary_v1(&self) -> Result<WeightAllocationSummaryV1> {
        let mut observations = Vec::new();
        self.for_each_tensor(|name, tensor| {
            if tensor.is_empty() || tensor.allocation_key() == 0 {
                return Err(NnisError::invalid_input(format!(
                    "weight {name} has no live device allocation to account"
                )));
            }
            observations.push(WeightAllocationObservation {
                logical_name: name.to_string(),
                allocation_key: tensor.allocation_key(),
                dtype: tensor.allocation_dtype(),
                elements: u64::try_from(tensor.len()).map_err(|_| {
                    NnisError::invalid_input(format!("weight {name} element count exceeds u64"))
                })?,
                bytes: u64::try_from(tensor.allocation_size_bytes()).map_err(|_| {
                    NnisError::invalid_input(format!("weight {name} byte size exceeds u64"))
                })?,
            });
            Ok(())
        })?;
        summarize_weight_allocations(observations)
    }

    fn expect_matrix(name: &str, weight: &MatrixWeight, rows: usize, cols: usize) -> Result<()> {
        if weight.rows() != rows || weight.cols() != cols {
            return Err(NnisError::invalid_input(format!(
                "weight {name} has shape ({}, {}); expected ({rows}, {cols})",
                weight.rows(),
                weight.cols()
            )));
        }
        Ok(())
    }

    fn expect_vector(name: &str, weight: &VectorWeight, len: usize) -> Result<()> {
        if weight.len() != len {
            return Err(NnisError::invalid_input(format!(
                "weight {name} has length {}; expected {len}",
                weight.len()
            )));
        }
        Ok(())
    }

    fn for_each_tensor(
        &self,
        mut visit: impl FnMut(&str, &DeviceTensor) -> Result<()>,
    ) -> Result<()> {
        visit("token_embedding", self.token_embedding.tensor())?;
        for (index, layer) in self.layers.iter().enumerate() {
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
        visit("final_norm", self.final_norm.tensor())?;
        visit("lm_head", self.lm_head.tensor())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        logical_name: &str,
        allocation_key: u64,
        dtype: WeightAllocationDTypeV1,
        elements: u64,
        bytes: u64,
    ) -> WeightAllocationObservation {
        WeightAllocationObservation {
            logical_name: logical_name.to_string(),
            allocation_key,
            dtype,
            elements,
            bytes,
        }
    }

    #[test]
    fn allocation_summary_counts_distinct_allocations_exactly() {
        let summary = summarize_weight_allocations([
            observation("embedding", 100, WeightAllocationDTypeV1::F32, 16, 64),
            observation("lm_head", 200, WeightAllocationDTypeV1::Bf16, 8, 16),
        ])
        .expect("summary");

        assert_eq!(
            summary.schema_version,
            NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION
        );
        assert_eq!(summary.logical_tensor_references, 2);
        assert_eq!(summary.logical_element_references, 24);
        assert_eq!(summary.unique_device_allocations, 2);
        assert_eq!(summary.unique_device_elements, 24);
        assert_eq!(summary.owned_device_allocation_bytes, 80);
        assert_eq!(summary.segments[0].allocation_index, 0);
        assert_eq!(summary.segments[1].allocation_index, 1);
    }

    #[test]
    fn allocation_summary_deduplicates_aliases_by_live_allocation() {
        let summary = summarize_weight_allocations([
            observation("token_embedding", 100, WeightAllocationDTypeV1::F32, 16, 64),
            observation("tied_alias", 100, WeightAllocationDTypeV1::F32, 16, 64),
        ])
        .expect("summary");

        assert_eq!(summary.logical_tensor_references, 2);
        assert_eq!(summary.logical_element_references, 32);
        assert_eq!(summary.unique_device_allocations, 1);
        assert_eq!(summary.unique_device_elements, 16);
        assert_eq!(summary.owned_device_allocation_bytes, 64);
        assert_eq!(
            summary.segments[0].logical_names,
            ["token_embedding".to_string(), "tied_alias".to_string()]
        );
    }

    #[test]
    fn allocation_summary_rejects_incoherent_alias_contract() {
        let error = summarize_weight_allocations([
            observation("first", 100, WeightAllocationDTypeV1::F32, 16, 64),
            observation("alias", 100, WeightAllocationDTypeV1::Bf16, 16, 32),
        ])
        .expect_err("incoherent alias must fail closed");

        assert!(error.to_string().contains("disagrees"));
    }

    #[test]
    fn allocation_summary_rejects_total_byte_overflow() {
        let error = summarize_weight_allocations([
            observation("first", 100, WeightAllocationDTypeV1::F32, 1, u64::MAX),
            observation("second", 200, WeightAllocationDTypeV1::F32, 1, 1),
        ])
        .expect_err("overflow must fail closed");

        assert!(error.to_string().contains("owned device allocation bytes"));
    }
}
