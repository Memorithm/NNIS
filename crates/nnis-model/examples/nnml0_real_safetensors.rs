use nnis_model::{
    load_model_from_safetensors, Activation, ModelConfig, SafetensorsLoadConfig, WeightDType,
};
use nnis_rt::{Context, Device, NnisError, Result, Stream};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const SOURCE_REPO: &str = "HuggingFaceTB/SmolLM2-135M";
const SOURCE_REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
const SOURCE_MODEL_SHA256: &str =
    "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1";

fn parse_arguments() -> std::result::Result<PathBuf, String> {
    let mut arguments = env::args().skip(1);
    let mut model_dir = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model" => {
                model_dir = Some(PathBuf::from(
                    arguments.next().ok_or("--model requires a directory")?,
                ));
            }
            "--help" | "-h" => {
                return Err("usage: nnml0_real_safetensors --model DIR".to_string());
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    model_dir.ok_or_else(|| "missing --model DIR".to_string())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).map_err(|error| NnisError::io(format!("open {}", path.display()), error))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| NnisError::io(format!("read {}", path.display()), error))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_pinned_source(model_dir: &Path) -> Result<()> {
    let model_path = model_dir.join("model.safetensors");
    let actual = sha256_file(&model_path)?;
    if actual != SOURCE_MODEL_SHA256 {
        return Err(NnisError::invalid_input(format!(
            "pinned model.safetensors SHA256 mismatch: {actual} != {SOURCE_MODEL_SHA256}"
        )));
    }
    if !model_dir.join("config.json").is_file() {
        return Err(NnisError::invalid_input(format!(
            "missing config.json in {}",
            model_dir.display()
        )));
    }
    Ok(())
}

fn validate_pinned_config(config: &ModelConfig) -> Result<()> {
    if config.vocab_size != 49_152
        || config.eos_token_id != Some(0)
        || config.hidden_size != 576
        || config.intermediate_size != 1_536
        || config.num_hidden_layers != 30
        || config.num_attention_heads != 9
        || config.num_key_value_heads != 3
        || config.max_position_embeddings != 8_192
        || config.rms_norm_eps.to_bits() != 1.0e-5_f32.to_bits()
        || config.rope_theta.to_bits() != 100_000.0_f32.to_bits()
        || config.activation != Activation::Silu
        || config.weight_dtype != WeightDType::Bf16
        || config.head_dim() != 64
    {
        return Err(NnisError::invalid_input(format!(
            "loaded model config does not match pinned SmolLM2-135M: {config:?}"
        )));
    }
    Ok(())
}

fn run(model_dir: PathBuf) -> Result<()> {
    validate_pinned_source(&model_dir)?;

    let local_dir = model_dir
        .to_str()
        .ok_or_else(|| NnisError::invalid_input("--model path must be valid UTF-8"))?
        .to_string();
    let load_config = SafetensorsLoadConfig {
        repo_id: Some(SOURCE_REPO.to_string()),
        revision: Some(SOURCE_REVISION.to_string()),
        local_dir,
    };

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let (model_config, weights) =
        load_model_from_safetensors(&context, &stream, &load_config)?;
    stream.synchronize()?;

    validate_pinned_config(&model_config)?;
    weights.validate(&model_config)?;

    println!("source={SOURCE_REPO}@{SOURCE_REVISION}");
    println!("source_model_sha256={SOURCE_MODEL_SHA256}");
    println!("weight_dtype={:?}", model_config.weight_dtype);
    println!("layers={}", model_config.num_hidden_layers);
    println!("hidden_size={}", model_config.hidden_size);
    println!("kv_width={}", model_config.key_value_width()?);
    println!("NNML0_REAL_SAFETENSORS_LOAD_OK");
    Ok(())
}

fn main() {
    let model_dir = match parse_arguments() {
        Ok(model_dir) => model_dir,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(model_dir) {
        eprintln!("NNML0 real Safetensors qualification failed: {error}");
        std::process::exit(1);
    }
}
