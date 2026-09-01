use nnis_bench::BenchmarkMetadata;
use nnis_model::{F16ReferenceGenerationProfile, F16ReferenceModel, F16ReferencePlan};
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
    260, 3_075, 338, 6_650, 260, 2_591, 284, 260, 8_872, 1_592, 30, 198, 198, 504,
    8_872, 314, 253, 8_304, 282, 260, 2_591, 30, 657, 314, 253, 19_284, 1_248, 338,
    21_837, 260, 2_591, 30,
];

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
    device: i32,
    warmups: usize,
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
    cross_runtime_memory_comparison_allowed: bool,
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
    persisted_execution_weight_dtype: &'static str,
    resident_weight_dtype: &'static str,
    activation_dtype: &'static str,
    kv_dtype: &'static str,
    projection_accumulator_dtype: &'static str,
    attention_accumulator_dtype: &'static str,
    logits_dtype: &'static str,
    input_ids: Vec<u32>,
    decode_steps: usize,
    warmup_iterations: usize,
    measured_iterations: usize,
    metadata: BenchmarkMetadata,
    f16_plan: F16ReferencePlan,
    memory: MemoryReport,
    request_wall_ms: f64,
    profile: F16ReferenceGenerationProfile,
    expected_ids: Vec<u32>,
    exact_greedy_32_of_32: bool,
    metric_semantics_aligned_to_edge_reference: bool,
    cross_runtime_speed_comparison_allowed: bool,
    speed_comparison_blocker: &'static str,
}

fn parse_arguments() -> std::result::Result<Arguments, String> {
    let mut args = env::args().skip(1);
    let mut model_dir = None;
    let mut device = 0_i32;
    let mut warmups = 2_usize;
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
                warmups = args
                    .next()
                    .ok_or("--warmups requires a value")?
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --warmups: {error}"))?;
            }
            "--help" | "-h" => {
                return Err(
                    "usage: smollm2_f16_edge_profile --model DIR [--device N] [--warmups N]"
                        .to_string(),
                );
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if device < 0 {
        return Err("--device must be non-negative".to_string());
    }
    Ok(Arguments {
        model_dir: model_dir.ok_or("missing --model DIR")?,
        device,
        warmups,
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
            "model provenance is not the pinned SmolLM2 fixture: {}@{} sha256={} source_dtype={} execution_dtype={}",
            provenance.source_repo,
            provenance.source_revision,
            provenance.source_model_sha256,
            provenance.source_weight_dtype,
            provenance.execution_weight_dtype
        )));
    }
    Ok(())
}

fn validate_model_shape(model: &F16ReferenceModel) -> Result<()> {
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

fn require_expected_ids(profile: &F16ReferenceGenerationProfile) -> Result<()> {
    if profile.generated_ids.as_slice() != EXPECTED_GREEDY_IDS {
        let divergence = profile
            .generated_ids
            .iter()
            .zip(EXPECTED_GREEDY_IDS.iter())
            .position(|(actual, expected)| actual != expected);
        return Err(NnisError::invalid_input(format!(
            "F16 Edge-semantic profile changed the qualified greedy trajectory at {divergence:?}: actual={:?} expected={:?}",
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
            "F16 profile metric contract drifted: {profile:?}"
        )));
    }
    Ok(())
}

fn run(arguments: Arguments) -> Result<Report> {
    validate_provenance(&arguments.model_dir)?;
    let device = Device::get(arguments.device)?;
    let context = Context::new(&device)?;
    let construction_stream = Stream::new(&context)?;
    let before_model = memory_snapshot(&context)?;
    let plan = F16ReferencePlan::edge_llm_v0_10_0_alignment();
    let model = F16ReferenceModel::load_directory(
        &context,
        &construction_stream,
        &arguments.model_dir,
        plan,
    )?;
    construction_stream.synchronize()?;
    validate_model_shape(&model)?;
    let after_model = memory_snapshot(&context)?;

    for _ in 0..arguments.warmups {
        let mut session = model.new_session()?;
        let profile = session.profile_greedy_edge_generation_semantics(&INPUT_IDS, DECODE_STEPS)?;
        require_expected_ids(&profile)?;
    }

    let mut session = model.new_session()?;
    let after_session = memory_snapshot(&context)?;
    let request_start = Instant::now();
    let profile = session.profile_greedy_edge_generation_semantics(&INPUT_IDS, DECODE_STEPS)?;
    let request_wall_ms = request_start.elapsed().as_secs_f64() * 1_000.0;
    require_expected_ids(&profile)?;

    if profile.session_position_after_profile != INPUT_IDS.len() + DECODE_STEPS - 1 {
        return Err(NnisError::invalid_input(format!(
            "unexpected profiled session position {}; expected {}",
            profile.session_position_after_profile,
            INPUT_IDS.len() + DECODE_STEPS - 1
        )));
    }

    Ok(Report {
        schema_version: 1,
        benchmark: "smollm2-135m-f16-edge-generation-stage",
        backend: "nnis",
        measurement: "CUDA events: prefill is total NNIS prompt GPU work; generation stage is cumulative GPU time of exactly 31 decoder forwards for 32 generated tokens; top-1 sampling/recording excluded; final generated token not consumed; model load excluded",
        source_repo: SOURCE_REPO,
        source_revision: SOURCE_REVISION,
        source_model_sha256: SOURCE_MODEL_SHA256,
        source_weight_dtype: "bfloat16",
        persisted_execution_weight_dtype: "f32",
        resident_weight_dtype: "f16",
        activation_dtype: "f16",
        kv_dtype: "f16",
        projection_accumulator_dtype: "f32",
        attention_accumulator_dtype: "f32",
        logits_dtype: "f32",
        input_ids: INPUT_IDS.to_vec(),
        decode_steps: DECODE_STEPS,
        warmup_iterations: arguments.warmups,
        measured_iterations: 1,
        metadata: BenchmarkMetadata::collect(&context),
        f16_plan: plan,
        memory: MemoryReport {
            cuda_free_delta_after_model_bytes: consumed_bytes(&before_model, &after_model),
            cuda_free_delta_after_session_bytes: consumed_bytes(&after_model, &after_session),
            before_model,
            after_model,
            after_session,
            cross_runtime_memory_comparison_allowed: false,
        },
        request_wall_ms,
        expected_ids: EXPECTED_GREEDY_IDS.to_vec(),
        exact_greedy_32_of_32: true,
        metric_semantics_aligned_to_edge_reference: true,
        cross_runtime_speed_comparison_allowed: false,
        speed_comparison_blocker: "single-process profile only; require repeated physical Thor campaign with exact-head evidence and fixed fingerprint before computing any cross-runtime ratio",
        profile,
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
                eprintln!("failed to serialize F16 Edge-semantic profile: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("SmolLM2 F16 Edge-semantic profile failed: {error}");
            std::process::exit(1);
        }
    }
}
