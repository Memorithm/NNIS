use crate::config::{ModelConfig, WeightDType};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result};
use std::sync::Arc;

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

    pub fn len(&self) -> usize {
        match self {
            Self::F32(buffer) => buffer.len(),
            Self::Bf16(buffer) => buffer.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
