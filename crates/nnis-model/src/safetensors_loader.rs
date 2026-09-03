//! Strict local Hugging Face Safetensors loader for NNIS.
//!
//! This loader consumes an already-materialized local model directory. It
//! performs no network access, reads `config.json` as the source of model
//! geometry, supports single-file and indexed-sharded Safetensors, preserves
//! F32/BF16 source values exactly, and fails closed on unsupported model
//! features or dtypes. Hugging Face matrices stored as `[out, in]` are
//! transposed to NNIS's internal `[in, out]` GEMM orientation before upload.

use crate::{
    Activation, DecoderLayerWeights, DeviceTensor, ModelConfig, ModelWeights, WeightDType,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

const HF_CONFIG: &str = "config.json";
const SINGLE_SAFETENSORS: &str = "model.safetensors";
const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";

/// Configuration for loading an already-materialized Hugging Face model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsLoadConfig {
    /// Optional provenance identifier, e.g. `HuggingFaceTB/SmolLM2-135M`.
    #[serde(default)]
    pub repo_id: Option<String>,
    /// Optional provenance revision/commit. No network resolution is performed.
    #[serde(default)]
    pub revision: Option<String>,
    /// Local directory containing `config.json` and Safetensors weights.
    pub local_dir: String,
}

/// Metadata validated from Hugging Face `config.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SafetensorsMetadata {
    pub architecture: String,
    pub model_type: String,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    pub eos_token_id: Option<u32>,
    pub tie_word_embeddings: bool,
    pub weight_dtype: WeightDType,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceConfig {
    architectures: Vec<String>,
    model_type: String,
    vocab_size: usize,
    #[serde(default)]
    eos_token_id: Option<EosTokenId>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    hidden_act: String,
    #[serde(default)]
    tie_word_embeddings: bool,
    torch_dtype: String,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    rope_interleaved: bool,
    #[serde(default)]
    rope_scaling: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Debug)]
struct TensorSpec {
    internal_name: String,
    hf_shape: Vec<usize>,
    internal_shape: Vec<usize>,
    transpose: bool,
}

enum HostTensor {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

fn parse_metadata(directory: &Path) -> Result<SafetensorsMetadata> {
    let path = directory.join(HF_CONFIG);
    let bytes =
        fs::read(&path).map_err(|error| NnisError::io("read Hugging Face config.json", error))?;
    parse_metadata_bytes(&bytes)
}

fn parse_metadata_bytes(bytes: &[u8]) -> Result<SafetensorsMetadata> {
    let config: HuggingFaceConfig = serde_json::from_slice(bytes).map_err(|error| {
        NnisError::invalid_input(format!("invalid Hugging Face config.json: {error}"))
    })?;
    validate_hf_config(config)
}

fn validate_hf_config(config: HuggingFaceConfig) -> Result<SafetensorsMetadata> {
    if config.architectures.len() != 1
        || config.architectures[0] != "LlamaForCausalLM"
        || config.model_type != "llama"
    {
        return Err(NnisError::unsupported(format!(
            "unsupported Hugging Face architecture {:?} / model_type {:?}; current Safetensors loader supports only LlamaForCausalLM with model_type=llama",
            config.architectures, config.model_type
        )));
    }
    if config.hidden_act != "silu" {
        return Err(NnisError::unsupported(format!(
            "unsupported hidden_act {:?}; current decoder requires silu/SwiGLU",
            config.hidden_act
        )));
    }
    if config.attention_bias || config.mlp_bias {
        return Err(NnisError::unsupported(
            "attention/MLP bias tensors are not supported by the current decoder weight graph",
        ));
    }
    if config.rope_interleaved || config.rope_scaling.is_some() {
        return Err(NnisError::unsupported(
            "interleaved or scaled RoPE is not supported by this Safetensors loader",
        ));
    }

    let weight_dtype = match config.torch_dtype.as_str() {
        "float32" | "f32" => WeightDType::F32,
        "bfloat16" | "bf16" => WeightDType::Bf16,
        other => {
            return Err(NnisError::unsupported(format!(
                "unsupported Hugging Face torch_dtype {other:?}; supported source dtypes are float32 and bfloat16"
            )))
        }
    };
    if config.num_attention_heads == 0 || config.hidden_size % config.num_attention_heads != 0 {
        return Err(NnisError::invalid_input(
            "hidden_size must be divisible by num_attention_heads",
        ));
    }
    let num_key_value_heads = config
        .num_key_value_heads
        .unwrap_or(config.num_attention_heads);
    let head_dim = config.hidden_size / config.num_attention_heads;
    let eos_token_id = match config.eos_token_id {
        None => None,
        Some(EosTokenId::Single(value)) => Some(value),
        Some(EosTokenId::Multiple(values)) => {
            return Err(NnisError::unsupported(format!(
                "multiple eos_token_id values are not supported yet: {values:?}"
            )))
        }
    };

    let metadata = SafetensorsMetadata {
        architecture: "LlamaForCausalLM".to_string(),
        model_type: "llama".to_string(),
        num_hidden_layers: config.num_hidden_layers,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings: config.max_position_embeddings,
        rms_norm_eps: config.rms_norm_eps,
        rope_theta: config.rope_theta,
        vocab_size: config.vocab_size,
        eos_token_id,
        tie_word_embeddings: config.tie_word_embeddings,
        weight_dtype,
    };
    metadata_to_model_config(&metadata)?.validate_execution_support()?;
    Ok(metadata)
}

fn metadata_to_model_config(metadata: &SafetensorsMetadata) -> Result<ModelConfig> {
    let config = ModelConfig {
        vocab_size: metadata.vocab_size,
        eos_token_id: metadata.eos_token_id,
        hidden_size: metadata.hidden_size,
        intermediate_size: metadata.intermediate_size,
        num_hidden_layers: metadata.num_hidden_layers,
        num_attention_heads: metadata.num_attention_heads,
        num_key_value_heads: metadata.num_key_value_heads,
        max_position_embeddings: metadata.max_position_embeddings,
        rms_norm_eps: metadata.rms_norm_eps,
        rope_theta: metadata.rope_theta,
        activation: Activation::Silu,
        weight_dtype: metadata.weight_dtype,
    };
    config.validate_execution_support()?;
    Ok(config)
}

fn discover_weight_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let single = directory.join(SINGLE_SAFETENSORS);
    let index = directory.join(SAFETENSORS_INDEX);
    if single.exists() && index.exists() {
        return Err(NnisError::invalid_input(
            "both model.safetensors and model.safetensors.index.json are present; refusing ambiguous source",
        ));
    }
    if single.is_file() {
        return Ok(vec![single]);
    }
    if !index.is_file() {
        return Err(NnisError::io(
            "discover Hugging Face Safetensors weights",
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "neither model.safetensors nor model.safetensors.index.json exists",
            ),
        ));
    }

    let bytes =
        fs::read(&index).map_err(|error| NnisError::io("read Safetensors shard index", error))?;
    let index: SafetensorsIndex = serde_json::from_slice(&bytes).map_err(|error| {
        NnisError::invalid_input(format!("invalid Safetensors shard index: {error}"))
    })?;
    if index.weight_map.is_empty() {
        return Err(NnisError::invalid_input(
            "Safetensors shard index has an empty weight_map",
        ));
    }

    let mut files = BTreeSet::new();
    for file in index.weight_map.values() {
        let relative = checked_relative_path(file)?;
        let path = directory.join(relative);
        if !path.is_file() {
            return Err(NnisError::io(
                "read Safetensors shard",
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("referenced shard {} does not exist", path.display()),
                ),
            ));
        }
        files.insert(path);
    }
    Ok(files.into_iter().collect())
}

fn checked_relative_path(file: &str) -> Result<&Path> {
    let path = Path::new(file);
    if file.is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
    {
        return Err(NnisError::invalid_input(format!(
            "Safetensors shard path {file:?} must be relative and may not traverse parents"
        )));
    }
    Ok(path)
}

fn tensor_spec(hf_name: &str, metadata: &SafetensorsMetadata) -> Option<TensorSpec> {
    let hidden = metadata.hidden_size;
    let intermediate = metadata.intermediate_size;
    let kv_width = metadata
        .num_key_value_heads
        .checked_mul(metadata.head_dim)?;

    let direct = match hf_name {
        "model.embed_tokens.weight" => Some((
            "token_embedding".to_string(),
            vec![metadata.vocab_size, hidden],
            vec![metadata.vocab_size, hidden],
            false,
        )),
        "model.norm.weight" => Some(("final_norm".to_string(), vec![hidden], vec![hidden], false)),
        "lm_head.weight" => Some((
            "lm_head".to_string(),
            vec![metadata.vocab_size, hidden],
            vec![hidden, metadata.vocab_size],
            true,
        )),
        _ => None,
    };
    if let Some((internal_name, hf_shape, internal_shape, transpose)) = direct {
        return Some(TensorSpec {
            internal_name,
            hf_shape,
            internal_shape,
            transpose,
        });
    }

    for layer in 0..metadata.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        let internal = format!("layers.{layer}");
        let spec = if hf_name == format!("{prefix}.input_layernorm.weight") {
            Some((
                format!("{internal}.input_norm"),
                vec![hidden],
                vec![hidden],
                false,
            ))
        } else if hf_name == format!("{prefix}.self_attn.q_proj.weight") {
            Some((
                format!("{internal}.q_proj"),
                vec![hidden, hidden],
                vec![hidden, hidden],
                true,
            ))
        } else if hf_name == format!("{prefix}.self_attn.k_proj.weight") {
            Some((
                format!("{internal}.k_proj"),
                vec![kv_width, hidden],
                vec![hidden, kv_width],
                true,
            ))
        } else if hf_name == format!("{prefix}.self_attn.v_proj.weight") {
            Some((
                format!("{internal}.v_proj"),
                vec![kv_width, hidden],
                vec![hidden, kv_width],
                true,
            ))
        } else if hf_name == format!("{prefix}.self_attn.o_proj.weight") {
            Some((
                format!("{internal}.o_proj"),
                vec![hidden, hidden],
                vec![hidden, hidden],
                true,
            ))
        } else if hf_name == format!("{prefix}.mlp.gate_proj.weight") {
            Some((
                format!("{internal}.gate_proj"),
                vec![intermediate, hidden],
                vec![hidden, intermediate],
                true,
            ))
        } else if hf_name == format!("{prefix}.mlp.up_proj.weight") {
            Some((
                format!("{internal}.up_proj"),
                vec![intermediate, hidden],
                vec![hidden, intermediate],
                true,
            ))
        } else if hf_name == format!("{prefix}.mlp.down_proj.weight") {
            Some((
                format!("{internal}.down_proj"),
                vec![hidden, intermediate],
                vec![intermediate, hidden],
                true,
            ))
        } else if hf_name == format!("{prefix}.post_attention_layernorm.weight") {
            Some((
                format!("{internal}.post_attention_norm"),
                vec![hidden],
                vec![hidden],
                false,
            ))
        } else {
            None
        };
        if let Some((internal_name, hf_shape, internal_shape, transpose)) = spec {
            return Some(TensorSpec {
                internal_name,
                hf_shape,
                internal_shape,
                transpose,
            });
        }
    }
    None
}

fn dtype_to_weight_dtype(dtype: Dtype) -> Result<WeightDType> {
    match dtype {
        Dtype::F32 => Ok(WeightDType::F32),
        Dtype::BF16 => Ok(WeightDType::Bf16),
        other => Err(NnisError::unsupported(format!(
            "unsupported Safetensors dtype {other:?}; supported source dtypes are F32 and BF16"
        ))),
    }
}

fn decode_host_tensor(data: &[u8], dtype: Dtype, elements: usize) -> Result<HostTensor> {
    let expected = elements
        .checked_mul(dtype.size())
        .ok_or_else(|| NnisError::invalid_input("Safetensors byte length overflows usize"))?;
    if data.len() != expected {
        return Err(NnisError::invalid_input(format!(
            "Safetensors tensor has {} bytes; dtype {dtype:?} and {elements} elements require {expected}",
            data.len()
        )));
    }
    match dtype {
        Dtype::F32 => Ok(HostTensor::F32(
            data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        )),
        Dtype::BF16 => Ok(HostTensor::Bf16(
            data.chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect(),
        )),
        other => Err(NnisError::unsupported(format!(
            "unsupported Safetensors dtype {other:?}; supported source dtypes are F32 and BF16"
        ))),
    }
}

fn transpose_matrix<T: Copy + Default>(values: Vec<T>, rows: usize, cols: usize) -> Result<Vec<T>> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| NnisError::invalid_input("matrix transpose shape overflows usize"))?;
    if values.len() != elements {
        return Err(NnisError::invalid_input(format!(
            "matrix transpose expected {elements} values for ({rows}, {cols}); got {}",
            values.len()
        )));
    }
    let mut output = vec![T::default(); elements];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = values[row * cols + col];
        }
    }
    Ok(output)
}

fn upload_host_tensor(
    context: &Arc<Context>,
    stream: &Stream,
    host: HostTensor,
) -> Result<DeviceTensor> {
    match host {
        HostTensor::F32(values) => Ok(DeviceTensor::F32(Arc::new(DeviceBuffer::from_host(
            context, stream, &values,
        )?))),
        HostTensor::Bf16(values) => Ok(DeviceTensor::Bf16(Arc::new(DeviceBuffer::from_host(
            context, stream, &values,
        )?))),
    }
}

fn insert_tensor(
    tensors: &mut HashMap<String, (Vec<usize>, DeviceTensor)>,
    name: String,
    shape: Vec<usize>,
    tensor: DeviceTensor,
) -> Result<()> {
    if tensors.insert(name.clone(), (shape, tensor)).is_some() {
        return Err(NnisError::invalid_input(format!(
            "duplicate logical tensor {name} across Safetensors files"
        )));
    }
    Ok(())
}

fn load_shard(
    context: &Arc<Context>,
    stream: &Stream,
    path: &Path,
    metadata: &SafetensorsMetadata,
    tensors: &mut HashMap<String, (Vec<usize>, DeviceTensor)>,
) -> Result<()> {
    let data =
        fs::read(path).map_err(|error| NnisError::io(format!("read {}", path.display()), error))?;
    let safetensors = SafeTensors::deserialize(&data).map_err(|error| {
        NnisError::invalid_input(format!("invalid {}: {error}", path.display()))
    })?;

    for (hf_name, view) in safetensors.tensors() {
        let Some(spec) = tensor_spec(&hf_name, metadata) else {
            continue;
        };
        if view.shape() != spec.hf_shape.as_slice() {
            return Err(NnisError::invalid_input(format!(
                "tensor {hf_name} has Safetensors shape {:?}; expected {:?}",
                view.shape(),
                spec.hf_shape
            )));
        }
        let source_dtype = dtype_to_weight_dtype(view.dtype())?;
        if source_dtype != metadata.weight_dtype {
            return Err(NnisError::invalid_input(format!(
                "tensor {hf_name} uses {source_dtype:?}; config.json declares {:?}",
                metadata.weight_dtype
            )));
        }
        let elements = view
            .shape()
            .iter()
            .try_fold(1_usize, |product, &dimension| {
                product.checked_mul(dimension).ok_or_else(|| {
                    NnisError::invalid_input("Safetensors tensor shape overflows usize")
                })
            })?;
        let mut host = decode_host_tensor(view.data(), view.dtype(), elements)?;
        if spec.transpose {
            host = match host {
                HostTensor::F32(values) => HostTensor::F32(transpose_matrix(
                    values,
                    spec.hf_shape[0],
                    spec.hf_shape[1],
                )?),
                HostTensor::Bf16(values) => HostTensor::Bf16(transpose_matrix(
                    values,
                    spec.hf_shape[0],
                    spec.hf_shape[1],
                )?),
            };
        }
        let device = upload_host_tensor(context, stream, host)?;
        insert_tensor(tensors, spec.internal_name, spec.internal_shape, device)?;
    }
    Ok(())
}

fn add_tied_lm_head_if_needed(
    context: &Arc<Context>,
    stream: &Stream,
    metadata: &SafetensorsMetadata,
    tensors: &mut HashMap<String, (Vec<usize>, DeviceTensor)>,
) -> Result<()> {
    if tensors.contains_key("lm_head") || !metadata.tie_word_embeddings {
        return Ok(());
    }
    let (_, embedding) = tensors
        .get("token_embedding")
        .ok_or_else(|| NnisError::invalid_input("token_embedding missing for tied lm_head"))?;
    let tied = match embedding {
        DeviceTensor::F32(buffer) => HostTensor::F32(transpose_matrix(
            buffer.to_vec(stream)?,
            metadata.vocab_size,
            metadata.hidden_size,
        )?),
        DeviceTensor::Bf16(buffer) => HostTensor::Bf16(transpose_matrix(
            buffer.to_vec(stream)?,
            metadata.vocab_size,
            metadata.hidden_size,
        )?),
    };
    let device = upload_host_tensor(context, stream, tied)?;
    insert_tensor(
        tensors,
        "lm_head".to_string(),
        vec![metadata.hidden_size, metadata.vocab_size],
        device,
    )
}

fn build_model_weights(
    mut tensors: HashMap<String, (Vec<usize>, DeviceTensor)>,
    config: &ModelConfig,
) -> Result<ModelWeights> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let kv_width = config.key_value_width()?;
    let token_embedding = take_mat(&mut tensors, "token_embedding", config.vocab_size, hidden)?;
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let p = format!("layers.{i}");
        layers.push(DecoderLayerWeights {
            input_norm: take_vec(&mut tensors, &format!("{p}.input_norm"), hidden)?,
            q_proj: take_mat(&mut tensors, &format!("{p}.q_proj"), hidden, hidden)?,
            k_proj: take_mat(&mut tensors, &format!("{p}.k_proj"), hidden, kv_width)?,
            v_proj: take_mat(&mut tensors, &format!("{p}.v_proj"), hidden, kv_width)?,
            o_proj: take_mat(&mut tensors, &format!("{p}.o_proj"), hidden, hidden)?,
            gate_proj: take_mat(
                &mut tensors,
                &format!("{p}.gate_proj"),
                hidden,
                intermediate,
            )?,
            up_proj: take_mat(&mut tensors, &format!("{p}.up_proj"), hidden, intermediate)?,
            down_proj: take_mat(
                &mut tensors,
                &format!("{p}.down_proj"),
                intermediate,
                hidden,
            )?,
            post_attention_norm: take_vec(
                &mut tensors,
                &format!("{p}.post_attention_norm"),
                hidden,
            )?,
        });
    }
    let weights = ModelWeights {
        token_embedding,
        layers,
        final_norm: take_vec(&mut tensors, "final_norm", hidden)?,
        lm_head: take_mat(&mut tensors, "lm_head", hidden, config.vocab_size)?,
    };
    if !tensors.is_empty() {
        let mut names: Vec<_> = tensors.keys().cloned().collect();
        names.sort();
        return Err(NnisError::invalid_input(format!(
            "unconsumed logical tensors remain after loading: {}",
            names.join(", ")
        )));
    }
    weights.validate(config)?;
    Ok(weights)
}

fn take_mat(
    tensors: &mut HashMap<String, (Vec<usize>, DeviceTensor)>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<crate::MatrixWeight> {
    let (shape, tensor) = tensors
        .remove(name)
        .ok_or_else(|| NnisError::invalid_input(format!("tensor {name} not found")))?;
    if shape.as_slice() != [rows, cols] {
        return Err(NnisError::invalid_input(format!(
            "tensor {name} shape {shape:?} != ({rows}, {cols})"
        )));
    }
    crate::MatrixWeight::new(tensor, rows, cols)
}

fn take_vec(
    tensors: &mut HashMap<String, (Vec<usize>, DeviceTensor)>,
    name: &str,
    len: usize,
) -> Result<crate::VectorWeight> {
    let (shape, tensor) = tensors
        .remove(name)
        .ok_or_else(|| NnisError::invalid_input(format!("tensor {name} not found")))?;
    if shape.as_slice() != [len] {
        return Err(NnisError::invalid_input(format!(
            "tensor {name} shape {shape:?} != ({len},)"
        )));
    }
    crate::VectorWeight::new(tensor, len)
}

/// Load a local Hugging Face Llama-style model from Safetensors.
///
/// No network access is performed. `config.json` is authoritative for model
/// shape and source dtype; unsupported architecture/dtype/features fail closed.
pub fn load_model_from_safetensors(
    context: &Arc<Context>,
    stream: &Stream,
    config: &SafetensorsLoadConfig,
) -> Result<(ModelConfig, ModelWeights)> {
    if !Arc::ptr_eq(context, stream.ctx()) {
        return Err(NnisError::invalid_input(
            "Safetensors loader context and upload stream must match",
        ));
    }
    let directory = Path::new(&config.local_dir);
    let metadata = parse_metadata(directory)?;
    let model_config = metadata_to_model_config(&metadata)?;
    let files = discover_weight_files(directory)?;
    let mut tensors = HashMap::new();
    for path in &files {
        load_shard(context, stream, path, &metadata, &mut tensors)?;
    }
    add_tied_lm_head_if_needed(context, stream, &metadata, &mut tensors)?;
    let weights = build_model_weights(tensors, &model_config)?;
    Ok((model_config, weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMOLLM2_CONFIG: &[u8] = br#"{
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": false,
        "eos_token_id": 0,
        "hidden_act": "silu",
        "hidden_size": 576,
        "intermediate_size": 1536,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "num_attention_heads": 9,
        "num_hidden_layers": 30,
        "num_key_value_heads": 3,
        "rms_norm_eps": 1e-5,
        "rope_interleaved": false,
        "rope_scaling": null,
        "rope_theta": 100000.0,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "vocab_size": 49152
    }"#;

    #[test]
    fn smollm2_config_is_read_without_hardcoded_dimensions() {
        let metadata = parse_metadata_bytes(SMOLLM2_CONFIG).unwrap();
        assert_eq!(metadata.hidden_size, 576);
        assert_eq!(metadata.num_hidden_layers, 30);
        assert_eq!(metadata.num_attention_heads, 9);
        assert_eq!(metadata.num_key_value_heads, 3);
        assert_eq!(metadata.head_dim, 64);
        assert_eq!(metadata.vocab_size, 49_152);
        assert_eq!(metadata.eos_token_id, Some(0));
        assert_eq!(metadata.weight_dtype, WeightDType::Bf16);
    }

    #[test]
    fn unsupported_dtype_and_architecture_fail_closed() {
        let mut dtype: Value = serde_json::from_slice(SMOLLM2_CONFIG).unwrap();
        dtype["torch_dtype"] = Value::from("float16");
        let dtype_bytes = serde_json::to_vec(&dtype).unwrap();
        assert!(parse_metadata_bytes(&dtype_bytes).is_err());

        let mut architecture: Value = serde_json::from_slice(SMOLLM2_CONFIG).unwrap();
        architecture["architectures"] = serde_json::json!(["MistralForCausalLM"]);
        architecture["model_type"] = Value::from("mistral");
        let architecture_bytes = serde_json::to_vec(&architecture).unwrap();
        assert!(parse_metadata_bytes(&architecture_bytes).is_err());
    }

    #[test]
    fn llama_tensor_shapes_use_real_gqa_geometry() {
        let metadata = parse_metadata_bytes(SMOLLM2_CONFIG).unwrap();
        let k = tensor_spec("model.layers.0.self_attn.k_proj.weight", &metadata).unwrap();
        assert_eq!(k.hf_shape, vec![192, 576]);
        assert_eq!(k.internal_shape, vec![576, 192]);
        assert!(k.transpose);
        let v = tensor_spec("model.layers.0.self_attn.v_proj.weight", &metadata).unwrap();
        assert_eq!(v.hf_shape, vec![192, 576]);
        assert_eq!(v.internal_shape, vec![576, 192]);
    }

    #[test]
    fn matrix_transpose_is_exact_for_f32_and_bf16_payloads() {
        assert_eq!(
            transpose_matrix(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        assert_eq!(
            transpose_matrix(vec![1_u16, 2, 3, 4, 5, 6], 2, 3).unwrap(),
            vec![1, 4, 2, 5, 3, 6]
        );
    }

    #[test]
    fn shard_paths_fail_closed_on_parent_traversal() {
        assert!(checked_relative_path("model-00001-of-00002.safetensors").is_ok());
        assert!(checked_relative_path("../escape.safetensors").is_err());
        assert!(checked_relative_path("/tmp/escape.safetensors").is_err());
    }
}
