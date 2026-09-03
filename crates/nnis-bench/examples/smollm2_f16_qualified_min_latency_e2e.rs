use nnis_bench::{summarize_samples_ms, BenchmarkMetadata, TimingStatistics};
use nnis_model::{
    F16AttentionPlan, F16ParallelScorePolicy, F16ReferenceExecutionPlan,
    F16ReferenceGenerationProfile, F16ReferenceModel, F16ReferenceProjectionLayout,
};
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
const INPUT_IDS: [u32; 3] = [22_007, 6_463, 314];
const DECODE_STEPS: usize = 32;
const EXPECTED_GREEDY_IDS: [u32; DECODE_STEPS] = [
    260, 3_075, 338, 6_650, 260, 2_591, 284, 260, 8_872, 1_592, 30, 198, 198, 504, 8_872, 314, 253,
    8_304, 282, 260, 2_591, 30, 657, 314, 253, 19_284, 1_248, 338, 21_837, 260, 2_591, 30,
];

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
    device: i32,
    warmups: usize,
    iterations: usize,
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
struct TimingReport {
    statistics: TimingStatistics,
    samples_ms: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    benchmark: &'static str,
    backend: &'static str,
    plan_name: &'static str,
    promotion_state: &'static str,
    source_repo: &'static str,
    source_revision: &'static str,
    source_model_sha256: &'static str,
    source_weight_dtype: &'static str,
    persisted_execution_weight_dtype: &'static str,
    resident_weight_dtype: &'static str,
    input_ids: Vec<u32>,
    decode_steps: usize,
    warmup_iterations: usize,
    iterations: usize,
    metadata: BenchmarkMetadata,
    execution_plan: F16ReferenceExecutionPlan,
    attention_plan: F16AttentionPlan,
    session_setup_wall: TimingReport,
    generation_wall: TimingReport,
    request_total_wall: TimingReport,
    prefill_gpu: TimingReport,
    generation_stage_gpu: TimingReport,
    generation_tokens_per_second_edge_definition_from_gpu_median: f64,
    generated_ids: Vec<u32>,
    exact_greedy_32_of_32: bool,
    generation_forward_runs_per_request: usize,
    sampling_included_in_generation_stage_gpu_time: bool,
    final_generated_token_consumed_by_model: bool,
    limitations: Vec<&'static str>,
}

fn parse_usize(name: &str, value: String) -> std::result::Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))
}

fn parse_arguments() -> std::result::Result<Arguments, String> {
    let mut args = env::args().skip(1);
    let mut model_dir = None;
    let mut device = 0_i32;
    let mut warmups = 2_usize;
    let mut iterations = 5_usize;

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
            "--help" | "-h" => {
                return Err("usage: smollm2_f16_qualified_min_latency_e2e --model DIR [--device N] [--warmups N] [--iterations N]".to_string());
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }

    if device < 0 {
        return Err("--device must be non-negative".to_string());
    }
    if warmups == 0 || iterations == 0 {
        return Err("--warmups and --iterations must be greater than zero".to_string());
    }

    Ok(Arguments {
        model_dir: model_dir.ok_or("missing --model DIR")?,
        device,
        warmups,
        iterations,
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
            "model provenance is not the pinned SmolLM2-135M qualification fixture",
        ));
    }
    Ok(())
}

fn validate_model_shape(model: &F16ReferenceModel) -> Result<()> {
    let config = model.config();
    if config.vocab_size != 49_152
        || config.eos_token_id != Some(0)
        || config.hidden_size != 576
        || config.intermediate_size != 1_536
        || config.num_hidden_layers != 30
        || config.num_attention_heads != 9
        || config.num_key_value_heads != 3
        || config.head_dim() != 64
        || config.max_position_embeddings != 8_192
        || config.rms_norm_eps != 1.0e-5
        || config.rope_theta != 100_000.0
    {
        return Err(NnisError::invalid_input(format!(
            "loaded model config does not match pinned SmolLM2-135M: {config:?}"
        )));
    }
    Ok(())
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn require_expected_profile(profile: &F16ReferenceGenerationProfile) -> Result<()> {
    if profile.generated_ids.as_slice() != EXPECTED_GREEDY_IDS {
        return Err(NnisError::invalid_input(format!(
            "qualified F16 stack greedy trajectory changed: actual={:?} expected={:?}",
            profile.generated_ids, EXPECTED_GREEDY_IDS
        )));
    }
    if profile.generated_tokens != DECODE_STEPS
        || profile.generation_forward_runs != DECODE_STEPS - 1
        || profile.generation_forward_gpu_ms.len() != DECODE_STEPS - 1
        || profile.sampling_included_in_generation_stage_gpu_time
        || profile.final_generated_token_consumed_by_model
    {
        return Err(NnisError::invalid_input(format!(
            "qualified F16 stack metric contract drifted: {profile:?}"
        )));
    }
    Ok(())
}

fn timing_report(samples_ms: Vec<f64>) -> Result<TimingReport> {
    Ok(TimingReport {
        statistics: summarize_samples_ms(&samples_ms)?,
        samples_ms,
    })
}

fn run(arguments: Arguments) -> Result<Report> {
    validate_provenance(&arguments.model_dir)?;
    let device = Device::get(arguments.device)?;
    let context = Context::new(&device)?;
    let construction_stream = Stream::new(&context)?;

    let (config, weights) =
        nnis_model::load_model_directory(&context, &construction_stream, &arguments.model_dir)?;
    let execution_plan =
        F16ReferenceExecutionPlan::smollm2_135m_thor_min_latency(&config, &context)?;
    let attention_plan = F16AttentionPlan::smollm2_135m_thor_min_latency(&config, &context)?;
    let model = F16ReferenceModel::new_with_execution_and_attention_plan(
        config,
        weights,
        &construction_stream,
        execution_plan,
        attention_plan,
    )?;
    construction_stream.synchronize()?;

    validate_model_shape(&model)?;
    if model.execution_plan() != execution_plan
        || model.execution_plan().projection_layout
            != F16ReferenceProjectionLayout::NkTransposedFusedMlpCandidate
        || model.attention_plan() != attention_plan
        || model.attention_plan().parallel_score_policy
            != Some(F16ParallelScorePolicy::Ka17SmolLm2ShortContextV1)
    {
        return Err(NnisError::unsupported(
            "qualified SmolLM2/Thor F16 stack did not preserve its selected plans",
        ));
    }

    for _ in 0..arguments.warmups {
        let mut session = model.new_session()?;
        let profile = session.profile_greedy_edge_generation_semantics(&INPUT_IDS, DECODE_STEPS)?;
        require_expected_profile(&profile)?;
    }

    let mut session_setup_samples = Vec::with_capacity(arguments.iterations);
    let mut generation_wall_samples = Vec::with_capacity(arguments.iterations);
    let mut request_total_samples = Vec::with_capacity(arguments.iterations);
    let mut prefill_gpu_samples = Vec::with_capacity(arguments.iterations);
    let mut generation_gpu_samples = Vec::with_capacity(arguments.iterations);
    let mut generated_ids = None;

    for _ in 0..arguments.iterations {
        let request_start = Instant::now();
        let setup_start = Instant::now();
        let mut session = model.new_session()?;
        session_setup_samples.push(elapsed_ms(setup_start));

        let generation_start = Instant::now();
        let profile = session.profile_greedy_edge_generation_semantics(&INPUT_IDS, DECODE_STEPS)?;
        generation_wall_samples.push(elapsed_ms(generation_start));
        request_total_samples.push(elapsed_ms(request_start));
        require_expected_profile(&profile)?;

        prefill_gpu_samples.push(profile.prefill_gpu_ms);
        generation_gpu_samples.push(profile.generation_stage_total_gpu_ms);
        if let Some(expected) = &generated_ids {
            if profile.generated_ids != *expected {
                return Err(NnisError::invalid_input(
                    "greedy output changed across qualified F16 stack iterations",
                ));
            }
        } else {
            generated_ids = Some(profile.generated_ids);
        }
    }

    let generation_stage_gpu = timing_report(generation_gpu_samples)?;
    let gpu_median = generation_stage_gpu.statistics.median_ms;
    if !gpu_median.is_finite() || gpu_median <= 0.0 {
        return Err(NnisError::unsupported(
            "qualified F16 stack generation GPU median is non-positive",
        ));
    }

    Ok(Report {
        schema_version: 1,
        benchmark: "smollm2-135m-f16-qualified-min-latency-e2e",
        backend: "nnis",
        plan_name: "smollm2_135m_thor_min_latency_v1",
        promotion_state: "qualified scoped selector; generic NNIS defaults remain reference",
        source_repo: SOURCE_REPO,
        source_revision: SOURCE_REVISION,
        source_model_sha256: SOURCE_MODEL_SHA256,
        source_weight_dtype: "bfloat16",
        persisted_execution_weight_dtype: "f32",
        resident_weight_dtype: "f16",
        input_ids: INPUT_IDS.to_vec(),
        decode_steps: DECODE_STEPS,
        warmup_iterations: arguments.warmups,
        iterations: arguments.iterations,
        metadata: BenchmarkMetadata::collect(&context),
        execution_plan,
        attention_plan,
        session_setup_wall: timing_report(session_setup_samples)?,
        generation_wall: timing_report(generation_wall_samples)?,
        request_total_wall: timing_report(request_total_samples)?,
        prefill_gpu: timing_report(prefill_gpu_samples)?,
        generation_tokens_per_second_edge_definition_from_gpu_median: DECODE_STEPS as f64
            / (gpu_median / 1_000.0),
        generation_stage_gpu,
        generated_ids: generated_ids.unwrap_or_default(),
        exact_greedy_32_of_32: true,
        generation_forward_runs_per_request: DECODE_STEPS - 1,
        sampling_included_in_generation_stage_gpu_time: false,
        final_generated_token_consumed_by_model: false,
        limitations: vec![
            "qualification is scoped to the pinned SmolLM2-135M model and NVIDIA Thor class",
            "short-context attention policy is qualified only through KV row 35 and falls back to reference outside its declared domain",
            "this harness establishes the combined qualified stack identity; it does not generalize performance to other models or devices",
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
                eprintln!("failed to serialize qualified F16 stack report: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("qualified SmolLM2 F16 stack benchmark failed: {error}");
            std::process::exit(1);
        }
    }
}
