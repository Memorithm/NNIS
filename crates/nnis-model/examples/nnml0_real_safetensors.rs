use nnis_model::{
    load_model_from_safetensors, DecoderExecutionCapabilities, ModelConfig, SafetensorsLoadConfig,
    SMOLLM2_135M_BF16,
};
use nnis_rt::{Context, Device, NnisError, Result, Stream};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const SOURCE_REPO: &str = SMOLLM2_135M_BF16.source_repo;
const SOURCE_REVISION: &str = SMOLLM2_135M_BF16.source_revision;
const SOURCE_MODEL_SHA256: &str = SMOLLM2_135M_BF16.source_model_sha256;
const EVIDENCE_KIND: &str = "nnis-nnml0-real-safetensors";
const EVIDENCE_SCHEMA_VERSION: u32 = 2;

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
    evidence_path: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct SourceEvidence {
    repo: &'static str,
    revision: &'static str,
    model_sha256: &'static str,
}

#[derive(Debug, Serialize)]
struct DeviceEvidence {
    ordinal: i32,
    name: String,
    uuid: Option<String>,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    multiprocessor_count: u32,
}

#[derive(Debug, Serialize)]
struct ModelEvidence {
    vocab_size: usize,
    eos_token_id: Option<u32>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    activation: &'static str,
    weight_dtype: &'static str,
    head_dim: usize,
    key_value_width: usize,
}

#[derive(Debug, Serialize)]
struct DecoderCapabilityEvidence {
    profile: DecoderExecutionCapabilities,
    canonical_record: String,
}

#[derive(Debug, Serialize)]
struct QualificationEvidence {
    schema_version: u32,
    kind: &'static str,
    result: &'static str,
    unix_timestamp_seconds: u64,
    nnis_git_commit: String,
    nnis_git_dirty: bool,
    host_arch: &'static str,
    host_os: &'static str,
    source: SourceEvidence,
    device: DeviceEvidence,
    model: ModelEvidence,
    decoder_capabilities: DecoderCapabilityEvidence,
}

fn parse_arguments() -> std::result::Result<Arguments, String> {
    let mut arguments = env::args().skip(1);
    let mut model_dir = None;
    let mut evidence_path = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model" => {
                model_dir = Some(PathBuf::from(
                    arguments.next().ok_or("--model requires a directory")?,
                ));
            }
            "--evidence" => {
                evidence_path = Some(PathBuf::from(
                    arguments.next().ok_or("--evidence requires a file path")?,
                ));
            }
            "--help" | "-h" => {
                return Err(
                    "usage: nnml0_real_safetensors --model DIR [--evidence FILE]".to_string(),
                );
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Arguments {
        model_dir: model_dir.ok_or_else(|| "missing --model DIR".to_string())?,
        evidence_path,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .map_err(|error| NnisError::io(format!("open {}", path.display()), error))?;
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

fn validate_pinned_config(config: &ModelConfig) -> Result<DecoderExecutionCapabilities> {
    SMOLLM2_135M_BF16.validate_config(config)
}

fn command_output(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| NnisError::io(format!("run {program}"), error))?;
    if !output.status.success() {
        return Err(NnisError::unsupported(format!(
            "{program} {:?} exited with {}",
            args, output.status
        )));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|_| NnisError::invalid_input(format!("{program} output is not UTF-8")))
}

fn git_identity() -> Result<(String, bool)> {
    let commit = command_output("git", &["rev-parse", "HEAD"])?;
    if commit.len() != 40 || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(NnisError::invalid_input(format!(
            "unexpected git commit identity {commit:?}"
        )));
    }
    let status = command_output("git", &["status", "--porcelain", "--untracked-files=all"])?;
    Ok((commit, !status.is_empty()))
}

fn write_evidence(path: &Path, evidence: &QualificationEvidence) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|error| {
                NnisError::io(
                    format!("create evidence directory {}", parent.display()),
                    error,
                )
            })?;
        }
    }
    let encoded = serde_json::to_vec_pretty(evidence)
        .map_err(|error| NnisError::invalid_input(format!("serialize evidence: {error}")))?;
    fs::write(path, encoded)
        .map_err(|error| NnisError::io(format!("write evidence {}", path.display()), error))
}

fn run(arguments: Arguments) -> Result<()> {
    validate_pinned_source(&arguments.model_dir)?;
    let (git_commit, git_dirty) = git_identity()?;
    if git_dirty {
        return Err(NnisError::invalid_input(
            "refusing NNML0 qualification from a dirty git worktree",
        ));
    }

    let local_dir = arguments
        .model_dir
        .to_str()
        .ok_or_else(|| NnisError::invalid_input("--model path must be valid UTF-8"))?
        .to_string();
    let load_config = SafetensorsLoadConfig {
        repo_id: Some(SOURCE_REPO.to_string()),
        revision: Some(SOURCE_REVISION.to_string()),
        local_dir,
    };

    let device = Device::first()?;
    let properties = device.props()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let (model_config, weights) = load_model_from_safetensors(&context, &stream, &load_config)?;
    stream.synchronize()?;

    let decoder_capabilities = validate_pinned_config(&model_config)?;
    weights.validate(&model_config)?;
    let canonical_capability_record = decoder_capabilities.canonical_record();

    let evidence = QualificationEvidence {
        schema_version: EVIDENCE_SCHEMA_VERSION,
        kind: EVIDENCE_KIND,
        result: "pass",
        unix_timestamp_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        nnis_git_commit: git_commit,
        nnis_git_dirty: false,
        host_arch: env::consts::ARCH,
        host_os: env::consts::OS,
        source: SourceEvidence {
            repo: SOURCE_REPO,
            revision: SOURCE_REVISION,
            model_sha256: SOURCE_MODEL_SHA256,
        },
        device: DeviceEvidence {
            ordinal: properties.ordinal,
            name: properties.name.clone(),
            uuid: properties.uuid.as_ref().map(|uuid| format!("{uuid:?}")),
            compute_capability_major: properties.compute_capability.0,
            compute_capability_minor: properties.compute_capability.1,
            multiprocessor_count: properties.multiprocessor_count,
        },
        model: ModelEvidence {
            vocab_size: model_config.vocab_size,
            eos_token_id: model_config.eos_token_id,
            hidden_size: model_config.hidden_size,
            intermediate_size: model_config.intermediate_size,
            num_hidden_layers: model_config.num_hidden_layers,
            num_attention_heads: model_config.num_attention_heads,
            num_key_value_heads: model_config.num_key_value_heads,
            max_position_embeddings: model_config.max_position_embeddings,
            rms_norm_eps: model_config.rms_norm_eps,
            rope_theta: model_config.rope_theta,
            activation: "silu",
            weight_dtype: "bf16",
            head_dim: model_config.head_dim(),
            key_value_width: model_config.key_value_width()?,
        },
        decoder_capabilities: DecoderCapabilityEvidence {
            profile: decoder_capabilities,
            canonical_record: canonical_capability_record,
        },
    };

    if let Some(path) = arguments.evidence_path.as_deref() {
        write_evidence(path, &evidence)?;
        println!("evidence={}", path.display());
    }
    println!("source={SOURCE_REPO}@{SOURCE_REVISION}");
    println!("source_model_sha256={SOURCE_MODEL_SHA256}");
    println!("nnis_git_commit={}", evidence.nnis_git_commit);
    println!("gpu_name={}", evidence.device.name);
    println!("weight_dtype={:?}", model_config.weight_dtype);
    println!("layers={}", model_config.num_hidden_layers);
    println!("hidden_size={}", model_config.hidden_size);
    println!("kv_width={}", model_config.key_value_width()?);
    println!(
        "decoder_capability_contract_version={}",
        evidence.decoder_capabilities.profile.contract_version
    );
    println!("NNML0_REAL_SAFETENSORS_LOAD_OK");
    Ok(())
}

fn main() {
    let arguments = match parse_arguments() {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(arguments) {
        eprintln!("NNML0 real Safetensors qualification failed: {error}");
        std::process::exit(1);
    }
}
