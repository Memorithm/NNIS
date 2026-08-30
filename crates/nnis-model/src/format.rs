use crate::{
    DecoderLayerWeights, DeviceTensor, MatrixWeight, ModelConfig, ModelWeights, VectorWeight,
    WeightDType,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

pub const NNIS_MODEL_FORMAT: &str = "nnis-model";
pub const NNIS_MODEL_VERSION: u32 = 1;
pub const NNIS_MODEL_MANIFEST: &str = "model.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub format: String,
    pub version: u32,
    pub config: ModelConfig,
    pub tensors: Vec<TensorManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorManifest {
    pub name: String,
    pub dtype: WeightDType,
    pub shape: Vec<usize>,
    pub file: String,
}

struct LoadedTensor {
    shape: Vec<usize>,
    tensor: DeviceTensor,
}

/// Load NNIS's deliberately simple model directory format.
///
/// The format is not a Hugging Face or Safetensors compatibility claim. A
/// directory contains `model.json` plus raw little-endian tensor files named
/// by the manifest. Matrix tensors are already transposed into NNIS's internal
/// GEMM orientation where required.
pub fn load_model_directory(
    context: &Arc<Context>,
    stream: &Stream,
    directory: impl AsRef<Path>,
) -> Result<(ModelConfig, ModelWeights)> {
    if !Arc::ptr_eq(context, stream.ctx()) {
        return Err(NnisError::invalid_input(
            "model loader context and upload stream must match",
        ));
    }
    let directory = directory.as_ref();
    let manifest_path = directory.join(NNIS_MODEL_MANIFEST);
    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|error| NnisError::io("read NNIS model manifest", error))?;
    let manifest: ModelManifest = serde_json::from_slice(&manifest_bytes).map_err(|error| {
        NnisError::invalid_input(format!("invalid NNIS model manifest JSON: {error}"))
    })?;
    validate_manifest_header(&manifest)?;
    manifest.config.validate()?;

    let mut tensors = HashMap::with_capacity(manifest.tensors.len());
    for entry in &manifest.tensors {
        validate_tensor_entry(entry)?;
        if tensors.contains_key(&entry.name) {
            return Err(NnisError::invalid_input(format!(
                "duplicate tensor name {} in model manifest",
                entry.name
            )));
        }
        let path = resolve_tensor_path(directory, &entry.file)?;
        let bytes = fs::read(&path).map_err(|error| NnisError::io("read NNIS tensor", error))?;
        let elements = checked_elements(&entry.shape)?;
        let tensor = match entry.dtype {
            WeightDType::F32 => {
                let expected = elements
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| NnisError::invalid_input("f32 tensor byte size overflows"))?;
                if bytes.len() != expected {
                    return Err(NnisError::invalid_input(format!(
                        "tensor {} file has {} bytes; shape {:?} f32 requires {expected}",
                        entry.name,
                        bytes.len(),
                        entry.shape
                    )));
                }
                let host: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                DeviceTensor::F32(Arc::new(DeviceBuffer::from_host(context, stream, &host)?))
            }
            WeightDType::Bf16 => {
                let expected = elements
                    .checked_mul(std::mem::size_of::<u16>())
                    .ok_or_else(|| NnisError::invalid_input("bf16 tensor byte size overflows"))?;
                if bytes.len() != expected {
                    return Err(NnisError::invalid_input(format!(
                        "tensor {} file has {} bytes; shape {:?} bf16 requires {expected}",
                        entry.name,
                        bytes.len(),
                        entry.shape
                    )));
                }
                let host: Vec<u16> = bytes
                    .chunks_exact(2)
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                DeviceTensor::Bf16(Arc::new(DeviceBuffer::from_host(context, stream, &host)?))
            }
        };
        tensors.insert(
            entry.name.clone(),
            LoadedTensor {
                shape: entry.shape.clone(),
                tensor,
            },
        );
    }

    let config = manifest.config;
    let weights = build_weight_graph(&config, &mut tensors)?;
    if !tensors.is_empty() {
        let mut names: Vec<_> = tensors.keys().cloned().collect();
        names.sort();
        return Err(NnisError::invalid_input(format!(
            "model manifest contains unrecognized tensors: {}",
            names.join(", ")
        )));
    }
    weights.validate(&config)?;
    Ok((config, weights))
}

fn validate_manifest_header(manifest: &ModelManifest) -> Result<()> {
    if manifest.format != NNIS_MODEL_FORMAT {
        return Err(NnisError::invalid_input(format!(
            "model format {:?} is not {:?}",
            manifest.format, NNIS_MODEL_FORMAT
        )));
    }
    if manifest.version != NNIS_MODEL_VERSION {
        return Err(NnisError::unsupported(format!(
            "NNIS model format version {}; supported version is {}",
            manifest.version, NNIS_MODEL_VERSION
        )));
    }
    Ok(())
}

fn validate_tensor_entry(entry: &TensorManifest) -> Result<()> {
    if entry.name.is_empty() {
        return Err(NnisError::invalid_input("tensor name must not be empty"));
    }
    if entry.shape.is_empty() || entry.shape.contains(&0) {
        return Err(NnisError::invalid_input(format!(
            "tensor {} shape must contain only non-zero dimensions; got {:?}",
            entry.name, entry.shape
        )));
    }
    let _ = checked_elements(&entry.shape)?;
    let _ = resolve_tensor_path(Path::new("."), &entry.file)?;
    Ok(())
}

fn checked_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or_else(|| NnisError::invalid_input("tensor shape overflows usize"))
    })
}

fn resolve_tensor_path(directory: &Path, file: &str) -> Result<PathBuf> {
    let relative = Path::new(file);
    if file.is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
    {
        return Err(NnisError::invalid_input(format!(
            "tensor file path {file:?} must be a relative path without parent traversal"
        )));
    }
    Ok(directory.join(relative))
}

fn build_weight_graph(
    config: &ModelConfig,
    tensors: &mut HashMap<String, LoadedTensor>,
) -> Result<ModelWeights> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let kv_width = config.key_value_width()?;

    let token_embedding = take_matrix(tensors, "token_embedding", config.vocab_size, hidden)?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("layers.{layer}");
        layers.push(DecoderLayerWeights {
            input_norm: take_vector(tensors, &format!("{prefix}.input_norm"), hidden)?,
            q_proj: take_matrix(tensors, &format!("{prefix}.q_proj"), hidden, hidden)?,
            k_proj: take_matrix(tensors, &format!("{prefix}.k_proj"), hidden, kv_width)?,
            v_proj: take_matrix(tensors, &format!("{prefix}.v_proj"), hidden, kv_width)?,
            o_proj: take_matrix(tensors, &format!("{prefix}.o_proj"), hidden, hidden)?,
            post_attention_norm: take_vector(
                tensors,
                &format!("{prefix}.post_attention_norm"),
                hidden,
            )?,
            gate_proj: take_matrix(
                tensors,
                &format!("{prefix}.gate_proj"),
                hidden,
                intermediate,
            )?,
            up_proj: take_matrix(tensors, &format!("{prefix}.up_proj"), hidden, intermediate)?,
            down_proj: take_matrix(
                tensors,
                &format!("{prefix}.down_proj"),
                intermediate,
                hidden,
            )?,
        });
    }
    Ok(ModelWeights {
        token_embedding,
        layers,
        final_norm: take_vector(tensors, "final_norm", hidden)?,
        lm_head: take_matrix(tensors, "lm_head", hidden, config.vocab_size)?,
    })
}

fn take_matrix(
    tensors: &mut HashMap<String, LoadedTensor>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<MatrixWeight> {
    let loaded = take_tensor(tensors, name, &[rows, cols])?;
    MatrixWeight::new(loaded.tensor, rows, cols)
}

fn take_vector(
    tensors: &mut HashMap<String, LoadedTensor>,
    name: &str,
    len: usize,
) -> Result<VectorWeight> {
    let loaded = take_tensor(tensors, name, &[len])?;
    VectorWeight::new(loaded.tensor, len)
}

fn take_tensor(
    tensors: &mut HashMap<String, LoadedTensor>,
    name: &str,
    expected_shape: &[usize],
) -> Result<LoadedTensor> {
    let loaded = tensors.remove(name).ok_or_else(|| {
        NnisError::invalid_input(format!("model manifest is missing required tensor {name}"))
    })?;
    if loaded.shape != expected_shape {
        return Err(NnisError::invalid_input(format!(
            "tensor {name} has manifest shape {:?}; expected {:?}",
            loaded.shape, expected_shape
        )));
    }
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_header_and_paths_are_strict() {
        let config = ModelConfig {
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            max_position_embeddings: 16,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10_000.0,
            activation: crate::Activation::Silu,
            weight_dtype: WeightDType::F32,
        };
        let manifest = ModelManifest {
            format: NNIS_MODEL_FORMAT.to_string(),
            version: NNIS_MODEL_VERSION,
            config,
            tensors: Vec::new(),
        };
        validate_manifest_header(&manifest).unwrap();
        assert!(resolve_tensor_path(Path::new("model"), "weights/q.bin").is_ok());
        assert!(resolve_tensor_path(Path::new("model"), "../escape.bin").is_err());
        assert!(resolve_tensor_path(Path::new("model"), "/tmp/escape.bin").is_err());
    }

    #[test]
    fn shape_product_detects_overflow_and_zero_shape_is_rejected() {
        assert!(checked_elements(&[2, 3, 4]).is_ok());
        assert!(checked_elements(&[usize::MAX, 2]).is_err());
        let entry = TensorManifest {
            name: "x".to_string(),
            dtype: WeightDType::F32,
            shape: vec![2, 0],
            file: "x.bin".to_string(),
        };
        assert!(validate_tensor_entry(&entry).is_err());
    }
}
