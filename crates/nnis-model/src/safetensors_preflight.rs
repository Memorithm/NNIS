//! CPU-only validation for local Hugging Face Safetensors sources.
//!
//! This module mirrors the strict source-admission rules used by the local
//! Safetensors loader, but stops before CUDA context creation, device allocation,
//! tensor transposition, or upload. It exists so tooling can reject an invalid
//! model directory before touching GPU state.

use crate::safetensors_loader::{SafetensorsLoadConfig, SafetensorsMetadata};
use crate::{Activation, ModelConfig, WeightDType};
use nnis_rt::{NnisError, Result};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Component, Path, PathBuf};

const HF_CONFIG: &str = "config.json";
const SINGLE_SAFETENSORS: &str = "model.safetensors";
const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";

/// Version of the CPU-only Hugging Face Safetensors preflight report.
pub const NNIS_HF_SAFETENSORS_PREFLIGHT_VERSION: u32 = 1;

/// Machine-readable result of validating a local Hugging Face Safetensors source.
///
/// A successful report proves only that the source directory satisfies the
/// currently declared structural loader contract. It is not model-quality,
/// numerical-equivalence, performance, or physical GPU evidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HfSafetensorsPreflightReportV1 {
    pub schema_version: u32,
    pub metadata: SafetensorsMetadata,
    pub weight_files: Vec<String>,
    pub recognized_tensor_count: usize,
    pub ignored_tensor_count: usize,
    /// Logical tensors available to the decoder after accounting for an allowed
    /// tied LM head synthesized from the token embedding.
    pub logical_tensor_count: usize,
    /// True when `config.json` declares tied embeddings and the source omits an
    /// explicit `lm_head.weight`, matching the loader's materialization rule.
    pub tied_lm_head_required: bool,
    /// Whether the validated source dtype satisfies the current direct
    /// [`crate::Model`] base-graph requirement. Tokenizer validity is outside this
    /// model-source report and is checked by the CLI preflight separately.
    pub direct_f32_execution_ready: bool,
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
        )),
        "model.norm.weight" => Some(("final_norm".to_string(), vec![hidden])),
        "lm_head.weight" => Some((
            "lm_head".to_string(),
            vec![metadata.vocab_size, hidden],
        )),
        _ => None,
    };
    if let Some((internal_name, hf_shape)) = direct {
        return Some(TensorSpec {
            internal_name,
            hf_shape,
        });
    }

    for layer in 0..metadata.num_hidden_layers {
        let prefix = format!("model.layers.{layer}");
        let internal = format!("layers.{layer}");
        let spec = if hf_name == format!("{prefix}.input_layernorm.weight") {
            Some((format!("{internal}.input_norm"), vec![hidden]))
        } else if hf_name == format!("{prefix}.self_attn.q_proj.weight") {
            Some((format!("{internal}.q_proj"), vec![hidden, hidden]))
        } else if hf_name == format!("{prefix}.self_attn.k_proj.weight") {
            Some((format!("{internal}.k_proj"), vec![kv_width, hidden]))
        } else if hf_name == format!("{prefix}.self_attn.v_proj.weight") {
            Some((format!("{internal}.v_proj"), vec![kv_width, hidden]))
        } else if hf_name == format!("{prefix}.self_attn.o_proj.weight") {
            Some((format!("{internal}.o_proj"), vec![hidden, hidden]))
        } else if hf_name == format!("{prefix}.mlp.gate_proj.weight") {
            Some((
                format!("{internal}.gate_proj"),
                vec![intermediate, hidden],
            ))
        } else if hf_name == format!("{prefix}.mlp.up_proj.weight") {
            Some((
                format!("{internal}.up_proj"),
                vec![intermediate, hidden],
            ))
        } else if hf_name == format!("{prefix}.mlp.down_proj.weight") {
            Some((
                format!("{internal}.down_proj"),
                vec![hidden, intermediate],
            ))
        } else if hf_name == format!("{prefix}.post_attention_layernorm.weight") {
            Some((format!("{internal}.post_attention_norm"), vec![hidden]))
        } else {
            None
        };
        if let Some((internal_name, hf_shape)) = spec {
            return Some(TensorSpec {
                internal_name,
                hf_shape,
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

fn expected_logical_tensor_names(metadata: &SafetensorsMetadata) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    names.insert("token_embedding".to_string());
    names.insert("final_norm".to_string());
    names.insert("lm_head".to_string());
    for layer in 0..metadata.num_hidden_layers {
        let prefix = format!("layers.{layer}");
        for suffix in [
            "input_norm",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "post_attention_norm",
        ] {
            names.insert(format!("{prefix}.{suffix}"));
        }
    }
    names
}

fn validate_logical_tensor_set(
    logical_tensors: &BTreeSet<String>,
    metadata: &SafetensorsMetadata,
) -> Result<bool> {
    let mut expected = expected_logical_tensor_names(metadata);
    let tied_lm_head_required = metadata.tie_word_embeddings && !logical_tensors.contains("lm_head");
    if tied_lm_head_required {
        expected.remove("lm_head");
    }

    let missing: Vec<_> = expected.difference(logical_tensors).cloned().collect();
    if !missing.is_empty() {
        return Err(NnisError::invalid_input(format!(
            "required logical tensors are missing from the Safetensors source: {}",
            missing.join(", ")
        )));
    }
    Ok(tied_lm_head_required)
}

fn validate_shard(
    path: &Path,
    metadata: &SafetensorsMetadata,
    logical_tensors: &mut BTreeSet<String>,
) -> Result<(usize, usize)> {
    let data =
        fs::read(path).map_err(|error| NnisError::io(format!("read {}", path.display()), error))?;
    let safetensors = SafeTensors::deserialize(&data).map_err(|error| {
        NnisError::invalid_input(format!("invalid {}: {error}", path.display()))
    })?;

    let mut recognized = 0_usize;
    let mut ignored = 0_usize;
    for (hf_name, view) in safetensors.tensors() {
        let Some(spec) = tensor_spec(&hf_name, metadata) else {
            ignored = ignored
                .checked_add(1)
                .ok_or_else(|| NnisError::invalid_input("ignored tensor count overflows usize"))?;
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
        let expected_bytes = elements
            .checked_mul(view.dtype().size())
            .ok_or_else(|| NnisError::invalid_input("Safetensors byte length overflows usize"))?;
        if view.data().len() != expected_bytes {
            return Err(NnisError::invalid_input(format!(
                "tensor {hf_name} has {} bytes; dtype {:?} and {elements} elements require {expected_bytes}",
                view.data().len(),
                view.dtype()
            )));
        }
        if !logical_tensors.insert(spec.internal_name.clone()) {
            return Err(NnisError::invalid_input(format!(
                "duplicate logical tensor {} across Safetensors files",
                spec.internal_name
            )));
        }
        recognized = recognized
            .checked_add(1)
            .ok_or_else(|| NnisError::invalid_input("recognized tensor count overflows usize"))?;
    }
    Ok((recognized, ignored))
}

/// Validate a local Hugging Face Safetensors directory without creating a CUDA
/// context or allocating device memory.
///
/// The function checks the declared model capability, local shard layout,
/// recognized tensor names, shapes, dtypes, byte lengths, duplicate logical
/// tensors, and completeness of the decoder weight graph. Unknown tensors are
/// ignored exactly as they are by the current loader. A tied LM head may be
/// absent only when `tie_word_embeddings=true`.
pub fn preflight_hf_safetensors_source(
    config: &SafetensorsLoadConfig,
) -> Result<HfSafetensorsPreflightReportV1> {
    let directory = Path::new(&config.local_dir);
    let metadata = parse_metadata(directory)?;
    let files = discover_weight_files(directory)?;
    let mut logical_tensors = BTreeSet::new();
    let mut recognized_tensor_count = 0_usize;
    let mut ignored_tensor_count = 0_usize;

    for path in &files {
        let (recognized, ignored) = validate_shard(path, &metadata, &mut logical_tensors)?;
        recognized_tensor_count = recognized_tensor_count
            .checked_add(recognized)
            .ok_or_else(|| NnisError::invalid_input("recognized tensor count overflows usize"))?;
        ignored_tensor_count = ignored_tensor_count
            .checked_add(ignored)
            .ok_or_else(|| NnisError::invalid_input("ignored tensor count overflows usize"))?;
    }

    let tied_lm_head_required = validate_logical_tensor_set(&logical_tensors, &metadata)?;
    let synthesized_logical_tensors = if tied_lm_head_required { 1_usize } else { 0_usize };
    let logical_tensor_count = logical_tensors
        .len()
        .checked_add(synthesized_logical_tensors)
        .ok_or_else(|| NnisError::invalid_input("logical tensor count overflows usize"))?;
    let expected_count = expected_logical_tensor_names(&metadata).len();
    if logical_tensor_count != expected_count {
        return Err(NnisError::invalid_input(format!(
            "validated logical tensor count {logical_tensor_count} does not match expected decoder graph count {expected_count}"
        )));
    }

    let weight_files = files
        .iter()
        .map(|path| {
            path.strip_prefix(directory)
                .unwrap_or(path)
                .to_string_lossy()
                .into_owned()
        })
        .collect();

    Ok(HfSafetensorsPreflightReportV1 {
        schema_version: NNIS_HF_SAFETENSORS_PREFLIGHT_VERSION,
        direct_f32_execution_ready: metadata.weight_dtype == WeightDType::F32,
        metadata,
        weight_files,
        recognized_tensor_count,
        ignored_tensor_count,
        logical_tensor_count,
        tied_lm_head_required,
    })
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

    fn complete_logical_set(metadata: &SafetensorsMetadata) -> BTreeSet<String> {
        expected_logical_tensor_names(metadata)
    }

    #[test]
    fn config_preflight_preserves_strict_llama_geometry_and_dtype() {
        let metadata = parse_metadata_bytes(SMOLLM2_CONFIG).unwrap();
        assert_eq!(metadata.architecture, "LlamaForCausalLM");
        assert_eq!(metadata.hidden_size, 576);
        assert_eq!(metadata.num_attention_heads, 9);
        assert_eq!(metadata.num_key_value_heads, 3);
        assert_eq!(metadata.head_dim, 64);
        assert_eq!(metadata.weight_dtype, WeightDType::Bf16);
    }

    #[test]
    fn float16_and_non_llama_sources_fail_closed() {
        let mut dtype: Value = serde_json::from_slice(SMOLLM2_CONFIG).unwrap();
        dtype["torch_dtype"] = Value::from("float16");
        assert!(parse_metadata_bytes(&serde_json::to_vec(&dtype).unwrap()).is_err());

        let mut architecture: Value = serde_json::from_slice(SMOLLM2_CONFIG).unwrap();
        architecture["architectures"] = serde_json::json!(["MistralForCausalLM"]);
        architecture["model_type"] = Value::from("mistral");
        assert!(parse_metadata_bytes(&serde_json::to_vec(&architecture).unwrap()).is_err());
    }

    #[test]
    fn gqa_tensor_shapes_match_the_loader_contract() {
        let metadata = parse_metadata_bytes(SMOLLM2_CONFIG).unwrap();
        let k = tensor_spec("model.layers.0.self_attn.k_proj.weight", &metadata).unwrap();
        assert_eq!(k.hf_shape, vec![192, 576]);
        let v = tensor_spec("model.layers.0.self_attn.v_proj.weight", &metadata).unwrap();
        assert_eq!(v.hf_shape, vec![192, 576]);
    }

    #[test]
    fn tied_lm_head_may_be_absent_but_other_missing_weights_fail() {
        let metadata = parse_metadata_bytes(SMOLLM2_CONFIG).unwrap();
        let mut logical = complete_logical_set(&metadata);
        logical.remove("lm_head");
        assert!(validate_logical_tensor_set(&logical, &metadata).unwrap());

        logical.remove("layers.0.q_proj");
        assert!(validate_logical_tensor_set(&logical, &metadata).is_err());
    }

    #[test]
    fn untied_lm_head_is_required() {
        let mut config: Value = serde_json::from_slice(SMOLLM2_CONFIG).unwrap();
        config["tie_word_embeddings"] = Value::from(false);
        let metadata = parse_metadata_bytes(&serde_json::to_vec(&config).unwrap()).unwrap();
        let mut logical = complete_logical_set(&metadata);
        logical.remove("lm_head");
        assert!(validate_logical_tensor_set(&logical, &metadata).is_err());
    }

    #[test]
    fn direct_execution_readiness_is_narrower_than_source_admission() {
        let bf16 = parse_metadata_bytes(SMOLLM2_CONFIG).unwrap();
        assert_ne!(bf16.weight_dtype, WeightDType::F32);

        let mut f32_config: Value = serde_json::from_slice(SMOLLM2_CONFIG).unwrap();
        f32_config["torch_dtype"] = Value::from("float32");
        let f32 = parse_metadata_bytes(&serde_json::to_vec(&f32_config).unwrap()).unwrap();
        assert_eq!(f32.weight_dtype, WeightDType::F32);
    }

    #[test]
    fn preflight_wire_report_is_versioned_and_strict() {
        let metadata = parse_metadata_bytes(SMOLLM2_CONFIG).unwrap();
        let report = HfSafetensorsPreflightReportV1 {
            schema_version: NNIS_HF_SAFETENSORS_PREFLIGHT_VERSION,
            metadata,
            weight_files: vec!["model.safetensors".to_string()],
            recognized_tensor_count: 272,
            ignored_tensor_count: 0,
            logical_tensor_count: 273,
            tied_lm_head_required: true,
            direct_f32_execution_ready: false,
        };
        let json = serde_json::to_string(&report).unwrap();
        let decoded: HfSafetensorsPreflightReportV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, report);

        let mut value: Value = serde_json::from_str(&json).unwrap();
        value["unknown"] = Value::from(true);
        assert!(serde_json::from_value::<HfSafetensorsPreflightReportV1>(value).is_err());
    }
}
