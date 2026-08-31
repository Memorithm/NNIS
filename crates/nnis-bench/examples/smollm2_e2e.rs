use nnis_bench::{summarize_samples_ms, BenchConfig, BenchmarkMetadata, TimingStatistics};
use nnis_model::{F32ProjectionPlan, GenerationConfig, Model};
use nnis_rt::{Context, Device, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const SOURCE_REPO: &str = "HuggingFaceTB/SmolLM2-135M";
const SOURCE_REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
const SOURCE_MODEL_SHA256: &str =
    "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1";
const DEFAULT_INPUT_IDS: [u32; 3] = [22_007, 6_463, 314];
const QUALIFIED_GREEDY_PREFIX: [u32; 2] = [260, 3_075];

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
    device: i32,
    input_ids: Vec<u32>,
    decode_steps: usize,
    config: BenchConfig,
    projection_plan: F32ProjectionPlan,
}

#[derive(Debug, Deserialize)]
struct Provenance {
    source_repo: String,
    source_revision: String,
    source_model_sha256: String,
    source_weight_dtype: String,
    execution_weight_dtype: String,
}

#[derive(Debug, Serialize)]
struct ModelShape {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
}

#[derive(Debug, Serialize)]
struct MemorySnapshot {
    free_bytes: u64,
    total_bytes: u64,
}

#[derive(Debug, Serialize)]
struct MemoryReport {
    before_model: MemorySnapshot,
    after_model: MemorySnapshot,
    after_session: MemorySnapshot,
    cuda_free_delta_after_model_bytes: Option<u64>,
    cuda_free_delta_after_session_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct TimingReport {
    statistics: TimingStatistics,
    samples_ms: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    benchmark: &'static str,
    backend: &'static str,
    measurement: &'static str,
    source_repo: &'static str,
    source_revision: &'static str,
    source_model_sha256: &'static str,
    source_weight_dtype: &'static str,
    execution_weight_dtype: &'static str,
    input_ids: Vec<u32>,
    decode_steps: usize,
    warmup_iterations: usize,
    iterations: usize,
    metadata: BenchmarkMetadata,
    model: ModelShape,
    memory: MemoryReport,
    session_setup: TimingReport,
    generation: TimingReport,
    request_total: TimingReport,
    generated_tokens_per_second_generation_median: f64,
    generated_tokens_per_second_request_median: f64,
    generated_ids: Vec<u32>,
    qualified_greedy_prefix_checked: bool,
    projection_plan: F32ProjectionPlan,
}

fn parse_usize(name: &str, value: String) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))
}

fn parse_input_ids(value: &str) -> std::result::Result<Vec<u32>, String> {
    if value.trim().is_empty() {
        return Err("--input-ids must not be empty".to_string());
    }
    value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<u32>()
                .map_err(|error| format!("invalid token id {part:?}: {error}"))
        })
        .collect()
}

fn parse_arguments() -> std::result::Result<Arguments, String> {
    let mut args = env::args().skip(1);
    let mut model_dir = None;
    let mut device = 0_i32;
    let mut input_ids = DEFAULT_INPUT_IDS.to_vec();
    let mut decode_steps = 32_usize;
    let mut warmups = 2_usize;
    let mut iterations = 5_usize;
    let mut projection_plan = F32ProjectionPlan::baseline_gemm();

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
            "--input-ids" => {
                input_ids = parse_input_ids(&args.next().ok_or("--input-ids requires CSV ids")?)?;
            }
            "--decode-steps" => {
                decode_steps = parse_usize(
                    "--decode-steps",
                    args.next().ok_or("--decode-steps requires a value")?,
                )?;
            }
            "--warmups" => {
                warmups = parse_usize(
                    "--warmups",
                    args.next().ok_or("--warmups requires a value")?,
                )?;
            }
            "--iterations" => {
                iterations = parse_usize(
                    "--iterations",
                    args.next().ok_or("--iterations requires a value")?,
                )?;
            }
            "--projection-plan" => {
                projection_plan = match args
                    .next()
                    .ok_or("--projection-plan requires baseline-gemm or thor-e1-1-lm-head")?
                    .as_str()
                {
                    "baseline-gemm" => F32ProjectionPlan::baseline_gemm(),
                    "thor-e1-1-lm-head" => F32ProjectionPlan::thor_e1_1_smollm2_lm_head(),
                    other => return Err(format!("unknown --projection-plan {other:?}")),
                };
            }
            "--help" | "-h" => {
                return Err(
                    "usage: smollm2_e2e --model DIR [--device N] [--input-ids CSV] [--decode-steps N] [--warmups N] [--iterations N] [--projection-plan baseline-gemm|thor-e1-1-lm-head]"
                        .to_string(),
                );
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }

    if device < 0 {
        return Err("--device must be non-negative".to_string());
    }
    if input_ids.is_empty() {
        return Err("benchmark prompt must contain at least one token".to_string());
    }
    if decode_steps == 0 {
        return Err("--decode-steps must be greater than zero".to_string());
    }
    if iterations == 0 {
        return Err("--iterations must be greater than zero".to_string());
    }

    Ok(Arguments {
        model_dir: model_dir.ok_or("missing --model DIR")?,
        device,
        input_ids,
        decode_steps,
        config: BenchConfig::new(warmups, iterations),
        projection_plan,
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
        return Err(NnisError::invalid_input(format!(
            "model provenance is not the pinned SmolLM2 benchmark fixture: {}@{} sha256={} source_dtype={} execution_dtype={}",
            provenance.source_repo,
            provenance.source_revision,
            provenance.source_model_sha256,
            provenance.source_weight_dtype,
            provenance.execution_weight_dtype
        )));
    }
    Ok(())
}

fn validate_shape(model: &Model) -> Result<()> {
    let config = model.config();
    if config.vocab_size != 49_152
        || config.hidden_size != 576
        || config.intermediate_size != 1_536
        || config.num_hidden_layers != 30
        || config.num_attention_heads != 9
        || config.num_key_value_heads != 3
        || config.head_dim() != 64
        || config.max_position_embeddings != 8_192
        || config.rope_theta != 100_000.0
    {
        return Err(NnisError::invalid_input(format!(
            "loaded model shape does not match pinned SmolLM2-135M: {config:?}"
        )));
    }
    Ok(())
}

fn memory_snapshot(context: &Context) -> Result<MemorySnapshot> {
    let (free_bytes, total_bytes) = context.mem_info()?;
    Ok(MemorySnapshot {
        free_bytes,
        total_bytes,
    })
}

fn consumed_bytes(before: &MemorySnapshot, after: &MemorySnapshot) -> Option<u64> {
    before.free_bytes.checked_sub(after.free_bytes)
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn validate_qualified_prefix(input_ids: &[u32], generated: &[u32]) -> Result<bool> {
    if input_ids != DEFAULT_INPUT_IDS || generated.len() < QUALIFIED_GREEDY_PREFIX.len() {
        return Ok(false);
    }
    if generated[..QUALIFIED_GREEDY_PREFIX.len()] != QUALIFIED_GREEDY_PREFIX {
        return Err(NnisError::invalid_input(format!(
            "qualified SmolLM2 greedy prefix changed: actual {:?}, expected {:?}",
            &generated[..QUALIFIED_GREEDY_PREFIX.len()],
            QUALIFIED_GREEDY_PREFIX
        )));
    }
    Ok(true)
}

fn run(arguments: Arguments) -> Result<Report> {
    arguments.config.validate()?;
    validate_provenance(&arguments.model_dir)?;

    let device = Device::get(arguments.device)?;
    let context = Context::new(&device)?;
    let construction_stream = Stream::new(&context)?;
    let before_model = memory_snapshot(&context)?;
    let model = Model::load_directory_with_projection_plan(
        &context,
        &construction_stream,
        &arguments.model_dir,
        arguments.projection_plan,
    )?;
    construction_stream.synchronize()?;
    validate_shape(&model)?;
    let after_model = memory_snapshot(&context)?;

    let session_for_memory = model.new_session()?;
    let after_session = memory_snapshot(&context)?;
    drop(session_for_memory);

    let generation = GenerationConfig::greedy(arguments.decode_steps);
    for _ in 0..arguments.config.warmup_iterations {
        let mut session = model.new_session()?;
        let generated = session.generate(&arguments.input_ids, generation)?;
        if generated.len() != arguments.decode_steps {
            return Err(NnisError::invalid_input(format!(
                "warmup generated {} tokens; expected {}",
                generated.len(),
                arguments.decode_steps
            )));
        }
        let _ = validate_qualified_prefix(&arguments.input_ids, &generated)?;
    }

    let mut session_setup_samples_ms = Vec::with_capacity(arguments.config.iterations);
    let mut generation_samples_ms = Vec::with_capacity(arguments.config.iterations);
    let mut request_total_samples_ms = Vec::with_capacity(arguments.config.iterations);
    let mut expected_generated: Option<Vec<u32>> = None;
    let mut qualified_greedy_prefix_checked = false;

    for _ in 0..arguments.config.iterations {
        let request_start = Instant::now();
        let setup_start = Instant::now();
        let mut session = model.new_session()?;
        let setup_ms = elapsed_ms(setup_start);

        let generation_start = Instant::now();
        let generated = session.generate(&arguments.input_ids, generation)?;
        let generation_ms = elapsed_ms(generation_start);
        let request_total_ms = elapsed_ms(request_start);
        if generated.len() != arguments.decode_steps {
            return Err(NnisError::invalid_input(format!(
                "measured generation produced {} tokens; expected {}",
                generated.len(),
                arguments.decode_steps
            )));
        }
        qualified_greedy_prefix_checked |=
            validate_qualified_prefix(&arguments.input_ids, &generated)?;
        if let Some(expected) = &expected_generated {
            if generated != *expected {
                return Err(NnisError::invalid_input(format!(
                    "non-deterministic greedy output across benchmark iterations: expected {expected:?}, actual {generated:?}"
                )));
            }
        } else {
            expected_generated = Some(generated);
        }
        session_setup_samples_ms.push(setup_ms);
        generation_samples_ms.push(generation_ms);
        request_total_samples_ms.push(request_total_ms);
    }

    let session_setup_statistics = summarize_samples_ms(&session_setup_samples_ms)?;
    let generation_statistics = summarize_samples_ms(&generation_samples_ms)?;
    let request_total_statistics = summarize_samples_ms(&request_total_samples_ms)?;
    if generation_statistics.median_ms <= 0.0 || request_total_statistics.median_ms <= 0.0 {
        return Err(NnisError::unsupported(
            "end-to-end timer returned a non-positive median duration",
        ));
    }
    let generated_tokens_per_second_generation_median =
        arguments.decode_steps as f64 / (generation_statistics.median_ms / 1_000.0);
    let generated_tokens_per_second_request_median =
        arguments.decode_steps as f64 / (request_total_statistics.median_ms / 1_000.0);
    let config = model.config();

    Ok(Report {
        schema_version: 2,
        benchmark: "smollm2-135m-greedy-e2e",
        backend: "nnis",
        measurement: "host-wall-clock; request_total includes fresh session setup plus synchronized generate(); generation excludes session setup; model load excluded",
        source_repo: SOURCE_REPO,
        source_revision: SOURCE_REVISION,
        source_model_sha256: SOURCE_MODEL_SHA256,
        source_weight_dtype: "bfloat16",
        execution_weight_dtype: "f32",
        input_ids: arguments.input_ids,
        decode_steps: arguments.decode_steps,
        warmup_iterations: arguments.config.warmup_iterations,
        iterations: arguments.config.iterations,
        metadata: BenchmarkMetadata::collect(&context),
        model: ModelShape {
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim(),
            max_position_embeddings: config.max_position_embeddings,
        },
        memory: MemoryReport {
            cuda_free_delta_after_model_bytes: consumed_bytes(&before_model, &after_model),
            cuda_free_delta_after_session_bytes: consumed_bytes(&after_model, &after_session),
            before_model,
            after_model,
            after_session,
        },
        session_setup: TimingReport {
            statistics: session_setup_statistics,
            samples_ms: session_setup_samples_ms,
        },
        generation: TimingReport {
            statistics: generation_statistics,
            samples_ms: generation_samples_ms,
        },
        request_total: TimingReport {
            statistics: request_total_statistics,
            samples_ms: request_total_samples_ms,
        },
        generated_tokens_per_second_generation_median,
        generated_tokens_per_second_request_median,
        generated_ids: expected_generated.expect("iterations are validated non-zero"),
        qualified_greedy_prefix_checked,
        projection_plan: model.projection_plan(),
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
                eprintln!("failed to serialize benchmark report: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("SmolLM2 NNIS benchmark failed: {error}");
            std::process::exit(1);
        }
    }
}
