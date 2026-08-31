use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::{F32Bf16Gemv, F32Gemv};
use nnis_rt::{Context, Device, DeviceBuffer, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

const SOURCE_REPO: &str = "HuggingFaceTB/SmolLM2-135M";
const SOURCE_REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
const SOURCE_MODEL_SHA256: &str =
    "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1";
const HIDDEN: usize = 576;
const VOCAB: usize = 49_152;
const REFERENCE_BLOCK_SIZE: u32 = 64;

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
    device: i32,
    config: BenchConfig,
    candidate_block_size: u32,
}

#[derive(Debug, Deserialize)]
struct Provenance {
    source_repo: String,
    source_revision: String,
    source_model_sha256: String,
    source_weight_dtype: String,
    execution_weight_dtype: String,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    tensors: Vec<TensorEntry>,
}

#[derive(Debug, Deserialize)]
struct TensorEntry {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    file: String,
}

#[derive(Debug, Serialize)]
struct RepresentationReport {
    source_checkpoint_dtype: &'static str,
    baseline_device_dtype: &'static str,
    candidate_device_dtype: &'static str,
    lm_head_elements: usize,
    baseline_storage_bytes: usize,
    candidate_storage_bytes: usize,
    storage_bytes_saved: usize,
    candidate_over_baseline_storage_ratio: f64,
    exact_bf16_roundtrip_from_fixture_f32: bool,
    candidate_changes_representation: bool,
}

#[derive(Debug, Serialize)]
struct BenchmarkConfigReport {
    reference_block_size: u32,
    candidate_block_size: u32,
    warmups: usize,
    iterations: usize,
}

#[derive(Debug, Serialize)]
struct IsolatedReport {
    latency_speedup_reference_over_candidate: f64,
    latency_ms_saved_per_lm_head_projection: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    experiment: &'static str,
    source_repo: &'static str,
    source_revision: &'static str,
    source_model_sha256: &'static str,
    benchmark_config: BenchmarkConfigReport,
    representation: RepresentationReport,
    bitwise_equivalent_all_logits: bool,
    reference: BenchmarkReport,
    candidate: BenchmarkReport,
    isolated: IsolatedReport,
    limitations: Vec<&'static str>,
}

fn parse_env_usize(name: &str, default: usize) -> std::result::Result<usize, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|error| format!("invalid {name}={value:?}: {error}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("failed reading {name}: {error}")),
    }
}

fn parse_env_u32(name: &str, default: u32) -> std::result::Result<u32, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<u32>()
            .map_err(|error| format!("invalid {name}={value:?}: {error}")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("failed reading {name}: {error}")),
    }
}

fn parse_arguments() -> std::result::Result<Arguments, String> {
    let mut args = env::args().skip(1);
    let mut model_dir = None;
    let mut device = 0_i32;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => {
                model_dir = Some(PathBuf::from(
                    args.next().ok_or("--model requires a directory")?,
                ));
            }
            "--device" => {
                device = args
                    .next()
                    .ok_or("--device requires an ordinal")?
                    .parse::<i32>()
                    .map_err(|error| format!("invalid --device: {error}"))?;
            }
            "--help" | "-h" => {
                return Err(
                    "usage: smollm2_lm_head_weight_representation --model DIR [--device N]"
                        .to_string(),
                );
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if device < 0 {
        return Err("--device must be non-negative".to_string());
    }
    let warmups = parse_env_usize("NNIS_PROFILE_WARMUPS", 20)?;
    let iterations = parse_env_usize("NNIS_PROFILE_ITERATIONS", 100)?;
    let candidate_block_size = parse_env_u32("NNIS_BF16_WEIGHT_BLOCK_SIZE", 64)?;
    Ok(Arguments {
        model_dir: model_dir.ok_or("missing --model DIR")?,
        device,
        config: BenchConfig::new(warmups, iterations),
        candidate_block_size,
    })
}

fn validate_provenance(model_dir: &Path) -> Result<()> {
    let bytes = fs::read(model_dir.join("provenance.json"))
        .map_err(|error| NnisError::io("read SmolLM2 provenance", error))?;
    let provenance: Provenance = serde_json::from_slice(&bytes).map_err(|error| {
        NnisError::invalid_input(format!("invalid SmolLM2 provenance JSON: {error}"))
    })?;
    if provenance.source_repo != SOURCE_REPO
        || provenance.source_revision != SOURCE_REVISION
        || provenance.source_model_sha256 != SOURCE_MODEL_SHA256
        || provenance.source_weight_dtype != "bfloat16"
        || provenance.execution_weight_dtype != "f32"
    {
        return Err(NnisError::invalid_input(
            "model provenance is not the pinned widened-f32 SmolLM2 fixture",
        ));
    }
    Ok(())
}

fn resolve_relative(model_dir: &Path, file: &str) -> Result<PathBuf> {
    let relative = Path::new(file);
    if file.is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
    {
        return Err(NnisError::invalid_input(format!(
            "LM-head tensor path {file:?} is not a safe relative path"
        )));
    }
    Ok(model_dir.join(relative))
}

fn read_lm_head(model_dir: &Path) -> Result<(Vec<f32>, Vec<u16>)> {
    let manifest_bytes = fs::read(model_dir.join("model.json"))
        .map_err(|error| NnisError::io("read SmolLM2 model manifest", error))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).map_err(|error| {
        NnisError::invalid_input(format!("invalid model manifest JSON: {error}"))
    })?;
    let entry = manifest
        .tensors
        .iter()
        .find(|entry| entry.name == "lm_head")
        .ok_or_else(|| NnisError::invalid_input("model manifest is missing lm_head"))?;
    if entry.dtype != "f32" || entry.shape != [HIDDEN, VOCAB] {
        return Err(NnisError::invalid_input(format!(
            "unexpected lm_head manifest contract: dtype={} shape={:?}",
            entry.dtype, entry.shape
        )));
    }
    let bytes = fs::read(resolve_relative(model_dir, &entry.file)?)
        .map_err(|error| NnisError::io("read SmolLM2 lm_head", error))?;
    let elements = HIDDEN
        .checked_mul(VOCAB)
        .ok_or_else(|| NnisError::invalid_input("LM-head shape overflows usize"))?;
    let expected_bytes = elements
        .checked_mul(4)
        .ok_or_else(|| NnisError::invalid_input("LM-head byte size overflows usize"))?;
    if bytes.len() != expected_bytes {
        return Err(NnisError::invalid_input(format!(
            "LM-head file has {} bytes; expected {expected_bytes}",
            bytes.len()
        )));
    }
    let mut f32_values = Vec::with_capacity(elements);
    let mut bf16_bits = Vec::with_capacity(elements);
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if bits & 0xffff != 0 {
            return Err(NnisError::invalid_input(format!(
                "fixture lm_head value {index} is not an exact widened BF16 value"
            )));
        }
        f32_values.push(f32::from_bits(bits));
        bf16_bits.push((bits >> 16) as u16);
    }
    Ok((f32_values, bf16_bits))
}

fn deterministic_input() -> Vec<f32> {
    (0..HIDDEN)
        .map(|index| ((index * 37 % 127) as f32 - 63.0) * 0.015625)
        .collect()
}

fn run(arguments: Arguments) -> Result<Report> {
    arguments.config.validate()?;
    validate_provenance(&arguments.model_dir)?;
    let (weight_f32_host, weight_bf16_host) = read_lm_head(&arguments.model_dir)?;
    let input_host = deterministic_input();
    let device = Device::get(arguments.device)?;
    let context = Context::new(&device)?;
    let context = Arc::new(context);
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let reference_kernel =
        F32Gemv::load_with_block_size(&context, &compiler, REFERENCE_BLOCK_SIZE)?;
    let candidate_kernel =
        F32Bf16Gemv::load_with_block_size(&context, &compiler, arguments.candidate_block_size)?;
    let input = DeviceBuffer::from_host(&context, &stream, &input_host)?;
    let weight_f32 = DeviceBuffer::from_host(&context, &stream, &weight_f32_host)?;
    let weight_bf16 = DeviceBuffer::from_host(&context, &stream, &weight_bf16_host)?;
    let reference_output = DeviceBuffer::<f32>::new(&context, VOCAB)?;
    let candidate_output = DeviceBuffer::<f32>::new(&context, VOCAB)?;

    reference_kernel.project_kn(
        &stream,
        &input,
        &weight_f32,
        &reference_output,
        HIDDEN,
        VOCAB,
    )?;
    candidate_kernel.project_kn(
        &stream,
        &input,
        &weight_bf16,
        &candidate_output,
        HIDDEN,
        VOCAB,
    )?;
    let reference_host = reference_output.to_vec(&stream)?;
    let candidate_host = candidate_output.to_vec(&stream)?;
    for (index, (&candidate, &reference)) in candidate_host.iter().zip(&reference_host).enumerate()
    {
        if candidate.to_bits() != reference.to_bits() {
            return Err(NnisError::invalid_input(format!(
                "W1 bitwise gate failed at logit {index}: candidate={candidate} reference={reference}"
            )));
        }
    }

    let reference = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("smollm2_lm_head_f32_gemv64", "f32"),
        arguments.config,
        || {
            // SAFETY: benchmark_gpu synchronizes each measured launch and
            // all captured buffers outlive the benchmark.
            unsafe {
                reference_kernel.enqueue_project_kn(
                    &stream,
                    &input,
                    &weight_f32,
                    &reference_output,
                    HIDDEN,
                    VOCAB,
                )
            }
        },
    )?;
    let candidate = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new(
            "smollm2_lm_head_f32_activation_bf16_weight",
            "f32xbf16->f32",
        ),
        arguments.config,
        || {
            // SAFETY: benchmark_gpu synchronizes each measured launch and
            // all captured buffers outlive the benchmark.
            unsafe {
                candidate_kernel.enqueue_project_kn(
                    &stream,
                    &input,
                    &weight_bf16,
                    &candidate_output,
                    HIDDEN,
                    VOCAB,
                )
            }
        },
    )?;

    reference
        .metadata
        .require_compatible_environment(&candidate.metadata)?;

    let reference_after = reference_output.to_vec(&stream)?;
    let candidate_after = candidate_output.to_vec(&stream)?;
    for (index, (&candidate, &reference)) in
        candidate_after.iter().zip(&reference_after).enumerate()
    {
        if candidate.to_bits() != reference.to_bits() {
            return Err(NnisError::invalid_input(format!(
                "W1 post-timing bitwise gate failed at logit {index}"
            )));
        }
    }

    let reference_ms = reference.statistics.median_ms;
    let candidate_ms = candidate.statistics.median_ms;
    let elements = HIDDEN * VOCAB;
    let baseline_storage_bytes = elements * std::mem::size_of::<f32>();
    let candidate_storage_bytes = elements * std::mem::size_of::<u16>();
    Ok(Report {
        schema_version: 1,
        experiment: "W1-smollm2-lm-head-f32-vs-bf16-weight-representation",
        source_repo: SOURCE_REPO,
        source_revision: SOURCE_REVISION,
        source_model_sha256: SOURCE_MODEL_SHA256,
        benchmark_config: BenchmarkConfigReport {
            reference_block_size: REFERENCE_BLOCK_SIZE,
            candidate_block_size: arguments.candidate_block_size,
            warmups: arguments.config.warmup_iterations,
            iterations: arguments.config.iterations,
        },
        representation: RepresentationReport {
            source_checkpoint_dtype: "bfloat16",
            baseline_device_dtype: "f32",
            candidate_device_dtype: "bfloat16",
            lm_head_elements: elements,
            baseline_storage_bytes,
            candidate_storage_bytes,
            storage_bytes_saved: baseline_storage_bytes - candidate_storage_bytes,
            candidate_over_baseline_storage_ratio: candidate_storage_bytes as f64
                / baseline_storage_bytes as f64,
            exact_bf16_roundtrip_from_fixture_f32: true,
            candidate_changes_representation: true,
        },
        bitwise_equivalent_all_logits: true,
        isolated: IsolatedReport {
            latency_speedup_reference_over_candidate: reference_ms / candidate_ms,
            latency_ms_saved_per_lm_head_projection: reference_ms - candidate_ms,
        },
        reference,
        candidate,
        limitations: vec![
            "candidate-only: nnis-model runtime and model format remain unchanged",
            "CUDA-event timings exclude host launch/submission overhead",
            "only the tied LM-head copy is represented as BF16 in this experiment",
            "end-to-end promotion requires a separate fingerprint-compatible AB/ABBA gate",
        ],
    })
}

fn main() {
    let arguments = match parse_arguments() {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    match run(arguments) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("failed to serialize W1 report: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("W1 LM-head representation benchmark failed: {error}");
            std::process::exit(1);
        }
    }
}
