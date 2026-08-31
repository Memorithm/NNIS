use nnis_model::{GenerationConfig, Model};
use nnis_rt::{Context, Device, NnisError, Result, Stream};
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const SOURCE_REPO: &str = "HuggingFaceTB/SmolLM2-135M";
const SOURCE_REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
const SOURCE_MODEL_SHA256: &str =
    "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogitPolicy {
    Strict,
    Report,
}

impl LogitPolicy {
    fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "strict" => Ok(Self::Strict),
            "report" => Ok(Self::Report),
            other => Err(format!(
                "invalid --logit-policy {other:?}; expected strict or report"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Report => "report",
        }
    }
}

#[derive(Debug, Deserialize)]
struct ReferenceManifest {
    format: String,
    version: u32,
    source_repo: String,
    source_revision: String,
    source_model_sha256: String,
    transformers_version: String,
    source_weight_dtype: String,
    execution_weight_dtype: String,
    prompt: String,
    input_ids: Vec<u32>,
    decode_steps: usize,
    dtype: String,
    logit_files: Vec<String>,
    greedy_ids: Vec<u32>,
}

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
    reference_dir: PathBuf,
    atol: f32,
    rtol: f32,
    logit_policy: LogitPolicy,
}

#[derive(Debug)]
struct ErrorMetrics {
    max_abs: f32,
    max_rel: f32,
    rms: f64,
    worst_index: usize,
    failures: usize,
    non_finite: usize,
}

fn parse_arguments() -> std::result::Result<Arguments, String> {
    let mut args = env::args().skip(1);
    let mut model_dir = None;
    let mut reference_dir = None;
    // These are harness defaults, not a validated SmolLM2 tolerance claim.
    // Physical-CUDA qualification must report measured errors before release.
    let mut atol = 1.0e-4_f32;
    let mut rtol = 1.0e-3_f32;
    let mut logit_policy = LogitPolicy::Strict;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => {
                model_dir = Some(PathBuf::from(
                    args.next().ok_or("--model requires a directory")?,
                ));
            }
            "--reference" => {
                reference_dir = Some(PathBuf::from(
                    args.next().ok_or("--reference requires a directory")?,
                ));
            }
            "--atol" => {
                atol = args
                    .next()
                    .ok_or("--atol requires a value")?
                    .parse()
                    .map_err(|error| format!("invalid --atol: {error}"))?;
            }
            "--rtol" => {
                rtol = args
                    .next()
                    .ok_or("--rtol requires a value")?
                    .parse()
                    .map_err(|error| format!("invalid --rtol: {error}"))?;
            }
            "--logit-policy" => {
                logit_policy = LogitPolicy::parse(
                    &args
                        .next()
                        .ok_or("--logit-policy requires strict or report")?,
                )?;
            }
            "--help" | "-h" => {
                return Err(
                    "usage: compare_smollm2_135m --model DIR --reference DIR [--atol F32] [--rtol F32] [--logit-policy strict|report]"
                        .to_string(),
                );
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if !atol.is_finite() || atol < 0.0 || !rtol.is_finite() || rtol < 0.0 {
        return Err("--atol and --rtol must be finite and non-negative".to_string());
    }
    Ok(Arguments {
        model_dir: model_dir.ok_or("missing --model DIR")?,
        reference_dir: reference_dir.ok_or("missing --reference DIR")?,
        atol,
        rtol,
        logit_policy,
    })
}

fn read_reference_manifest(directory: &Path) -> Result<ReferenceManifest> {
    let bytes = fs::read(directory.join("reference.json"))
        .map_err(|error| NnisError::io("read reference manifest", error))?;
    let manifest: ReferenceManifest = serde_json::from_slice(&bytes).map_err(|error| {
        NnisError::invalid_input(format!("invalid reference manifest JSON: {error}"))
    })?;
    if manifest.format != "nnis-reference-logits" || manifest.version != 1 {
        return Err(NnisError::unsupported(format!(
            "unsupported reference format {:?} version {}",
            manifest.format, manifest.version
        )));
    }
    if manifest.source_repo != SOURCE_REPO
        || manifest.source_revision != SOURCE_REVISION
        || manifest.source_model_sha256 != SOURCE_MODEL_SHA256
    {
        return Err(NnisError::invalid_input(format!(
            "reference provenance does not match pinned SmolLM2 fixture: {}@{} sha256={}",
            manifest.source_repo, manifest.source_revision, manifest.source_model_sha256
        )));
    }
    if manifest.transformers_version != "4.40.1"
        || manifest.source_weight_dtype != "bfloat16"
        || manifest.execution_weight_dtype != "f32"
        || manifest.dtype != "f32"
    {
        return Err(NnisError::unsupported(format!(
            "unexpected reference numeric environment: transformers={} source={} execution={} logits={}",
            manifest.transformers_version,
            manifest.source_weight_dtype,
            manifest.execution_weight_dtype,
            manifest.dtype
        )));
    }
    if manifest.logit_files.len() != manifest.decode_steps + 1 {
        return Err(NnisError::invalid_input(format!(
            "reference has {} logit files for {} decode steps; expected {}",
            manifest.logit_files.len(),
            manifest.decode_steps,
            manifest.decode_steps + 1
        )));
    }
    if manifest.greedy_ids.len() != manifest.decode_steps {
        return Err(NnisError::invalid_input(format!(
            "reference has {} greedy IDs for {} decode steps",
            manifest.greedy_ids.len(),
            manifest.decode_steps
        )));
    }
    if manifest.input_ids.is_empty() {
        return Err(NnisError::invalid_input("reference input_ids are empty"));
    }
    Ok(manifest)
}

fn read_f32_le(path: &Path, expected_len: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path).map_err(|error| NnisError::io("read reference logits", error))?;
    let expected_bytes = expected_len
        .checked_mul(4)
        .ok_or_else(|| NnisError::invalid_input("reference logit byte length overflows usize"))?;
    if bytes.len() != expected_bytes {
        return Err(NnisError::invalid_input(format!(
            "reference logits {} contain {} bytes; expected {expected_bytes}",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn compare(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) -> ErrorMetrics {
    assert_eq!(actual.len(), expected.len());
    let mut metrics = ErrorMetrics {
        max_abs: 0.0,
        max_rel: 0.0,
        rms: 0.0,
        worst_index: 0,
        failures: 0,
        non_finite: 0,
    };
    let mut squared_sum = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let absolute = (actual - expected).abs();
        let relative = if expected == 0.0 {
            if absolute == 0.0 {
                0.0
            } else {
                f32::INFINITY
            }
        } else {
            absolute / expected.abs()
        };
        if absolute > metrics.max_abs {
            metrics.max_abs = absolute;
            metrics.worst_index = index;
        }
        metrics.max_rel = metrics.max_rel.max(relative);
        squared_sum += f64::from(absolute) * f64::from(absolute);
        if !actual.is_finite() {
            metrics.non_finite += 1;
            metrics.failures += 1;
        } else if absolute > atol + rtol * expected.abs() {
            metrics.failures += 1;
        }
    }
    metrics.rms = (squared_sum / actual.len() as f64).sqrt();
    metrics
}

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    best_index
}

fn report_and_require(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
    logit_policy: LogitPolicy,
) -> Result<()> {
    let metrics = compare(actual, expected, atol, rtol);
    println!(
        "{label}: max_abs={:.8e} max_rel={:.8e} rms={:.8e} worst_index={} failures={}",
        metrics.max_abs, metrics.max_rel, metrics.rms, metrics.worst_index, metrics.failures
    );
    if metrics.non_finite != 0 {
        return Err(NnisError::invalid_input(format!(
            "{label} contains {} non-finite NNIS logits",
            metrics.non_finite
        )));
    }
    if logit_policy == LogitPolicy::Strict && metrics.failures != 0 {
        return Err(NnisError::invalid_input(format!(
            "{label} differs from trusted reference at {} logits (atol={atol}, rtol={rtol})",
            metrics.failures
        )));
    }
    if logit_policy == LogitPolicy::Report && metrics.failures != 0 {
        println!(
            "{label}: report policy observed {} values outside the supplied tolerance; numeric equivalence is not asserted",
            metrics.failures
        );
    }
    Ok(())
}

fn require_greedy(step: usize, logits: &[f32], expected: u32) -> Result<()> {
    let actual = argmax(logits) as u32;
    if actual != expected {
        return Err(NnisError::invalid_input(format!(
            "greedy token mismatch at step {step}: NNIS {actual}, reference {expected}"
        )));
    }
    Ok(())
}

fn validate_model_shape(model: &Model) -> Result<()> {
    let config = model.config();
    if config.vocab_size != 49_152
        || config.hidden_size != 576
        || config.intermediate_size != 1_536
        || config.num_hidden_layers != 30
        || config.num_attention_heads != 9
        || config.num_key_value_heads != 3
        || config.head_dim() != 64
        || config.max_position_embeddings != 8_192
        || config.eos_token_id != Some(0)
        || config.rope_theta != 100_000.0
    {
        return Err(NnisError::invalid_input(format!(
            "loaded model config does not match pinned SmolLM2-135M: {config:?}"
        )));
    }
    Ok(())
}

fn run(arguments: Arguments) -> Result<()> {
    let reference = read_reference_manifest(&arguments.reference_dir)?;
    let device = Device::first()?;
    let context = Context::new(&device)?;
    let construction_stream = Stream::new(&context)?;
    let model = Model::load_directory(&context, &construction_stream, &arguments.model_dir)?;
    validate_model_shape(&model)?;
    let vocab = model.config().vocab_size;
    let mut session = model.new_session()?;

    println!(
        "source={}@{}",
        reference.source_repo, reference.source_revision
    );
    println!("source_model_sha256={}", reference.source_model_sha256);
    println!("prompt={:?}", reference.prompt);
    println!("input_ids={:?}", reference.input_ids);
    println!("atol={} rtol={}", arguments.atol, arguments.rtol);
    println!("logit_policy={}", arguments.logit_policy.as_str());

    let mut actual_logits = session.prefill(&reference.input_ids)?;
    let expected_prefill = read_f32_le(
        &arguments.reference_dir.join(&reference.logit_files[0]),
        vocab,
    )?;
    report_and_require(
        "prefill",
        &actual_logits,
        &expected_prefill,
        arguments.atol,
        arguments.rtol,
        arguments.logit_policy,
    )?;

    for step in 0..reference.decode_steps {
        let greedy = reference.greedy_ids[step];
        require_greedy(step, &actual_logits, greedy)?;
        actual_logits = session.decode_one(greedy)?;
        let expected = read_f32_le(
            &arguments
                .reference_dir
                .join(&reference.logit_files[step + 1]),
            vocab,
        )?;
        report_and_require(
            &format!("decode[{step}]"),
            &actual_logits,
            &expected,
            arguments.atol,
            arguments.rtol,
            arguments.logit_policy,
        )?;
    }

    let generated = model.new_session()?.generate(
        &reference.input_ids,
        GenerationConfig::greedy(reference.decode_steps),
    )?;
    if generated != reference.greedy_ids {
        return Err(NnisError::invalid_input(format!(
            "greedy sequence mismatch: NNIS {generated:?}, reference {:?}",
            reference.greedy_ids
        )));
    }
    println!("greedy_ids={generated:?}");
    match arguments.logit_policy {
        LogitPolicy::Strict => println!("SmolLM2 strict reference comparison passed"),
        LogitPolicy::Report => {
            println!("SmolLM2 semantic trajectory passed; numeric equivalence is not asserted")
        }
    }
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
        eprintln!("SmolLM2 reference comparison failed: {error}");
        std::process::exit(1);
    }
}
