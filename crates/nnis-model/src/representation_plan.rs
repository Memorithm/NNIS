use crate::{
    load_model_directory, DeviceTensor, MatrixWeight, ModelConfig, ModelWeights, WeightDType,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// Current schema version for runtime-only physical weight representation plans.
pub const WEIGHT_REPRESENTATION_PLAN_VERSION: u32 = 1;

/// Physical storage selected for a logical weight tensor.
///
/// This is deliberately separate from projection/kernel selection. Logical
/// model values and model format v1 remain unchanged by this choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicalWeightRepresentation {
    F32,
    Bf16,
}

/// Versioned, explicit physical representation plan.
///
/// Version 1 intentionally exposes only the LM-head axis because W1 is the only
/// heterogeneous representation candidate with physical evidence. Every other
/// tensor remains f32. Expanding this structure requires executable contracts
/// and evidence for the new tensor families; there is no implicit global BF16
/// switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightRepresentationPlan {
    pub schema_version: u32,
    pub lm_head: PhysicalWeightRepresentation,
}

impl WeightRepresentationPlan {
    /// Historical runtime representation: every weight is resident as f32.
    #[must_use]
    pub const fn all_f32() -> Self {
        Self {
            schema_version: WEIGHT_REPRESENTATION_PLAN_VERSION,
            lm_head: PhysicalWeightRepresentation::F32,
        }
    }

    /// W1 candidate representation: only the tied SmolLM2 LM-head copy is
    /// resident as BF16. This is candidate-only and is not a runtime default.
    #[must_use]
    pub const fn w1_lm_head_bf16() -> Self {
        Self {
            schema_version: WEIGHT_REPRESENTATION_PLAN_VERSION,
            lm_head: PhysicalWeightRepresentation::Bf16,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.schema_version != WEIGHT_REPRESENTATION_PLAN_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight representation plan schema {}; supported version is {}",
                self.schema_version, WEIGHT_REPRESENTATION_PLAN_VERSION
            )));
        }
        Ok(())
    }

    /// Validate the complete device weight graph against this explicit plan.
    ///
    /// The base model contract remains f32 in model format v1. A BF16 LM-head
    /// is therefore accepted only as a runtime representation override while
    /// every other tensor must remain f32.
    pub fn validate_weights(self, config: &ModelConfig, weights: &ModelWeights) -> Result<()> {
        self.validate()?;
        config.validate()?;
        if config.weight_dtype != WeightDType::F32 {
            return Err(NnisError::unsupported(
                "weight representation plan v1 requires an f32 model-format-v1 base graph",
            ));
        }
        if weights.layers.len() != config.num_hidden_layers {
            return Err(NnisError::invalid_input(format!(
                "model has {} decoder layers; config requires {}",
                weights.layers.len(),
                config.num_hidden_layers
            )));
        }

        let context = weights.token_embedding.tensor().context();
        expect_matrix(
            "token_embedding",
            &weights.token_embedding,
            config.vocab_size,
            config.hidden_size,
            WeightDType::F32,
            context,
        )?;
        expect_vector(
            "final_norm",
            &weights.final_norm,
            config.hidden_size,
            WeightDType::F32,
            context,
        )?;
        expect_matrix(
            "lm_head",
            &weights.lm_head,
            config.hidden_size,
            config.vocab_size,
            match self.lm_head {
                PhysicalWeightRepresentation::F32 => WeightDType::F32,
                PhysicalWeightRepresentation::Bf16 => WeightDType::Bf16,
            },
            context,
        )?;

        let kv_width = config.key_value_width()?;
        for (index, layer) in weights.layers.iter().enumerate() {
            expect_vector(
                &format!("layers.{index}.input_norm"),
                &layer.input_norm,
                config.hidden_size,
                WeightDType::F32,
                context,
            )?;
            expect_matrix(
                &format!("layers.{index}.q_proj"),
                &layer.q_proj,
                config.hidden_size,
                config.hidden_size,
                WeightDType::F32,
                context,
            )?;
            for (name, weight) in [("k_proj", &layer.k_proj), ("v_proj", &layer.v_proj)] {
                expect_matrix(
                    &format!("layers.{index}.{name}"),
                    weight,
                    config.hidden_size,
                    kv_width,
                    WeightDType::F32,
                    context,
                )?;
            }
            expect_matrix(
                &format!("layers.{index}.o_proj"),
                &layer.o_proj,
                config.hidden_size,
                config.hidden_size,
                WeightDType::F32,
                context,
            )?;
            expect_vector(
                &format!("layers.{index}.post_attention_norm"),
                &layer.post_attention_norm,
                config.hidden_size,
                WeightDType::F32,
                context,
            )?;
            for (name, weight) in [("gate_proj", &layer.gate_proj), ("up_proj", &layer.up_proj)] {
                expect_matrix(
                    &format!("layers.{index}.{name}"),
                    weight,
                    config.hidden_size,
                    config.intermediate_size,
                    WeightDType::F32,
                    context,
                )?;
            }
            expect_matrix(
                &format!("layers.{index}.down_proj"),
                &layer.down_proj,
                config.intermediate_size,
                config.hidden_size,
                WeightDType::F32,
                context,
            )?;
        }
        Ok(())
    }
}

/// Load model format v1 unchanged, then apply a runtime-only representation
/// plan to the resident weight graph.
///
/// For W1, the loader first uses the ordinary strict f32 loader, proves every
/// LM-head value is exactly BF16-representable, and only then replaces that
/// resident device tensor with packed BF16. The model manifest and files are
/// never rewritten.
pub fn load_model_directory_with_representation_plan(
    context: &Arc<Context>,
    stream: &Stream,
    directory: impl AsRef<Path>,
    plan: WeightRepresentationPlan,
) -> Result<(ModelConfig, ModelWeights)> {
    plan.validate()?;
    let (config, mut weights) = load_model_directory(context, stream, directory)?;

    if plan.lm_head == PhysicalWeightRepresentation::Bf16 {
        if config.weight_dtype != WeightDType::F32 {
            return Err(NnisError::unsupported(
                "BF16 LM-head runtime representation requires an f32 model-format-v1 source tensor",
            ));
        }
        let rows = weights.lm_head.rows();
        let cols = weights.lm_head.cols();
        let host_f32 = weights.lm_head.tensor().as_f32()?.to_vec(stream)?;
        let mut host_bf16 = Vec::with_capacity(host_f32.len());
        for (index, value) in host_f32.iter().enumerate() {
            let bits = value.to_bits();
            if bits & 0xffff != 0 {
                return Err(NnisError::invalid_input(format!(
                    "lm_head value at element {index} is not exactly BF16-representable; refusing lossy runtime representation"
                )));
            }
            host_bf16.push((bits >> 16) as u16);
        }
        let packed = DeviceTensor::Bf16(Arc::new(DeviceBuffer::from_host(
            context, stream, &host_bf16,
        )?));
        weights.lm_head = MatrixWeight::new(packed, rows, cols)?;
    }

    plan.validate_weights(&config, &weights)?;
    Ok((config, weights))
}

fn expect_matrix(
    name: &str,
    weight: &MatrixWeight,
    rows: usize,
    cols: usize,
    dtype: WeightDType,
    context: &Arc<Context>,
) -> Result<()> {
    if weight.rows() != rows || weight.cols() != cols {
        return Err(NnisError::invalid_input(format!(
            "weight {name} has shape ({}, {}); expected ({rows}, {cols})",
            weight.rows(),
            weight.cols()
        )));
    }
    expect_tensor(name, weight.tensor(), dtype, context)
}

fn expect_vector(
    name: &str,
    weight: &crate::VectorWeight,
    len: usize,
    dtype: WeightDType,
    context: &Arc<Context>,
) -> Result<()> {
    if weight.len() != len {
        return Err(NnisError::invalid_input(format!(
            "weight {name} has length {}; expected {len}",
            weight.len()
        )));
    }
    expect_tensor(name, weight.tensor(), dtype, context)
}

fn expect_tensor(
    name: &str,
    tensor: &DeviceTensor,
    dtype: WeightDType,
    context: &Arc<Context>,
) -> Result<()> {
    if tensor.dtype() != dtype {
        return Err(NnisError::invalid_input(format!(
            "weight {name} uses {:?}; representation plan requires {:?}",
            tensor.dtype(),
            dtype
        )));
    }
    if !Arc::ptr_eq(context, tensor.context()) {
        return Err(NnisError::invalid_input(format!(
            "weight {name} belongs to a different CUDA context"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representation_plan_is_versioned_and_explicit() {
        let baseline = WeightRepresentationPlan::all_f32();
        assert_eq!(baseline.schema_version, WEIGHT_REPRESENTATION_PLAN_VERSION);
        assert_eq!(baseline.lm_head, PhysicalWeightRepresentation::F32);
        baseline.validate().unwrap();

        let w1 = WeightRepresentationPlan::w1_lm_head_bf16();
        assert_eq!(w1.lm_head, PhysicalWeightRepresentation::Bf16);
        w1.validate().unwrap();

        let mut future = baseline;
        future.schema_version += 1;
        assert!(future.validate().is_err());
    }
}
