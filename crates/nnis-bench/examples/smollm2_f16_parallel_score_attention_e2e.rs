use nnis_bench::{summarize_samples_ms, BenchmarkMetadata, TimingStatistics};
use nnis_jit::JitCompiler;
use nnis_model::{
    F16AttentionPlan, F16CachedAttentionKernel, F16CachedAttentionParallelScoreCandidate,
    F16ReferenceExecutionPlan, F16ReferenceGenerationProfile, F16ReferenceModel,
    F16ReferenceProjectionLayout,
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
const MAX_PROFILE_KV_ROWS: usize = INPUT_IDS.len() + DECODE_STEPS - 1;
const KA17_MAX_QUALIFIED_KV_ROWS: usize = 35;
const EXPECTED_GREEDY_IDS: [u32; DECODE_STEPS] = [
    260, 3_075, 338, 6_650, 260, 2_591, 284, 260, 8_872, 1_592, 30, 198, 198, 504, 8_872, 314, 253,
    8_304, 282, 260, 2_591, 30, 657, 314, 253, 19_284, 1_248, 338, 21_837, 260, 2_591, 30,
];

#[derive(Debug, Clone, Copy)]
enum AttentionPlanName {
    Reference,
    ParallelScoreKa17,
}

impl AttentionPlanName {
    fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "reference" => Ok(Self::Reference),
            "parallel-score-ka17" => Ok(Self::ParallelScoreKa17),
            other => Err(format!(
                "unknown --attention {other:?}; expected reference or parallel-score-ka17"
            )),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::ParallelScoreKa17 => "parallel-score-ka17",
        }
    }

    const fn plan(self) -> F16AttentionPlan {
        match self {
            Self::Reference => F16AttentionPlan::reference(),
            Self::ParallelScoreKa17 => {
                F16AttentionPlan::smollm2_135m_thor_parallel_score_ka17_candidate()
            }
        }
    }
}

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
    device: i32,
    attention_name: AttentionPlanName,
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
    attention_name: &'static str,
    promotion_state: &'static str,
    source_repo: &'static str,
    source_revision: &'static str,
    source_model_sha256: &'static str,
    source_weight_dtype: &'static str,
    persisted_execution_weight_dtype: &'static str,
    resident_weight_dtype: &'static str,
    input_ids: Vec<u32>,
    decode_steps: usize,
    max_profile_kv_rows: usize,
    ka17_max_qualified_kv_rows: usize,
    warmup_iterations: usize,
    iterations: usize,
    metadata: BenchmarkMetadata,
    execution_plan: F16ReferenceExecutionPlan,
    attention_plan: F16AttentionPlan,
    parallel_score_candidate_max_supported_kv_rows: Option<usize>,
    memory: MemoryReport,
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
    let mut attention_name = None;
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
            "--attention" => {
                attention_name = Some(AttentionPlanName::parse(
                    &args.next().ok_or(
                        "--attention requires reference or parallel-score-ka17",
                    )?,
                )?);
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
                return Err("usage: smollm2_f16_parallel_score_attention_e2e --model DIR --attention reference|parallel-score-ka17 [--device N] [--warmups N] [--iterations N]".to_string());
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
        attention_name: attention_name.ok_or("missing --attention reference|parallel-score-ka17")?,
        warmups,
        iterations,
    })
}

fn require_run_context() -> Result<()> {
    match env::var("NNIS_BENCH_RUN_CONTEXT_ID") {
        Ok(value) if !value.trim().is_empty() => Ok(()),
        _ => Err(NnisError::invalid_input(
            "NNIS_BENCH_RUN_CONTEXT_ID is required for KA18 evidence",
        )),
    }
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

fn require_qualified_thor(context: &Context) -> Result<()> {
    let properties = context.props();
    if properties.name != "NVIDIA Thor"
        || properties.compute_capability != (11, 0)
        || properties.multiprocessor_count != 20
    {
        return Err(NnisError::unsupported(format!(
            "KA18 requires the qualified NVIDIA Thor cc 11.0 / 20-SM domain; observed {} cc {}.{} with {} SMs",
            properties.name,
            properties.compute_capability.0,
            properties.compute_capability.1,
            properties.multiprocessor_count
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
        || config.rms_norm_eps != 1.0e-5
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

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn require_expected_profile(profile: &F16ReferenceGenerationProfile) -> Result<()> {
    if profile.generated_ids.as_slice() != EXPECTED_GREEDY_IDS {
        let divergence = profile
            .generated_ids
            .iter()
            .zip(EXPECTED_GREEDY_IDS.iter())
            .position(|(actual, expected)| actual != expected);
        return Err(NnisError::invalid_input(format!(
            "KA18 F16 parallel-score trajectory changed at {divergence:?}: actual={:?} expected={:?}",
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
            "KA18 F16 parallel-score metric contract drifted: {profile:?}"
        )));
    }
    Ok(())
}

fn timing_report(samples_ms: Vec<f64>) -> Result<TimingReport> {
    let statistics = summarize_samples_ms(&samples_ms)?;
    Ok(TimingReport {
        statistics,
        samples_ms,
    })
}

fn run(arguments: Arguments) -> Result<Report> {
    require_run_context()?;
    validate_provenance(&arguments.model_dir)?;

    let device = Device::get(arguments.device)?;
    let context = Context::new(&device)?;
    require_qualified_thor(&context)?;

    let execution_plan =
        F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_fused_mlp_candidate();
    let attention_plan = arguments.attention_name.plan();
    let parallel_score_candidate_max_supported_kv_rows = match arguments.attention_name {
        AttentionPlanName::Reference => None,
        AttentionPlanName::ParallelScoreKa17 => {
            let candidate =
                F16CachedAttentionParallelScoreCandidate::load(&context, &JitCompiler::new())?;
            let max_rows = candidate.max_supported_kv_rows();
            if max_rows < KA17_MAX_QUALIFIED_KV_ROWS {
                return Err(NnisError::unsupported(format!(
                    "KA18 parallel-score attention supports at most {max_rows} KV rows on this device/kernel; KA17 qualification requires {KA17_MAX_QUALIFIED_KV_ROWS}"
                )));
            }
            Some(max_rows)
        }
    };

    let construction_stream = Stream::new(&context)?;
    let before_model = memory_snapshot(&context)?;
    let model = F16ReferenceModel::load_directory_with_execution_and_attention_plan(
        &context,
        &construction_stream,
        &arguments.model_dir,
        execution_plan,
        attention_plan,
    )?;
    construction_stream.synchronize()?;
    validate_model_shape(&model)?;
    execution_plan.validate_smollm2_135m_min_latency_model_domain(model.config())?;
    if model.execution_plan() != execution_plan
        || model.execution_plan().projection_layout
            != F16ReferenceProjectionLayout::NkTransposedFusedMlpCandidate
    {
        return Err(NnisError::unsupported(
            "KA18 did not preserve the qualified SmolLM2 F16 min-latency projection parent",
        ));
    }
    if model.attention_plan() != attention_plan {
        return Err(NnisError::unsupported(
            "KA18 model did not preserve the requested attention plan",
        ));
    }
    match arguments.attention_name {
        AttentionPlanName::Reference => {
            if attention_plan.kernel != F16CachedAttentionKernel::ReferencePerPositionBarriers {
                return Err(NnisError::unsupported(
                    "KA18 reference process selected a non-reference attention kernel",
                ));
            }
        }
        AttentionPlanName::ParallelScoreKa17 => {
            if attention_plan.kernel != F16CachedAttentionKernel::ParallelScoreKa17Candidate
                || attention_plan.parallel_score_ka17_threads_per_block(3).is_some()
                || attention_plan.parallel_score_ka17_threads_per_block(4) != Some(128)
                || attention_plan.parallel_score_ka17_threads_per_block(5) != Some(256)
                || attention_plan.parallel_score_ka17_threads_per_block(16) != Some(256)
                || attention_plan.parallel_score_ka17_threads_per_block(17) != Some(512)
                || attention_plan.parallel_score_ka17_threads_per_block(34) != Some(512)
                || attention_plan.parallel_score_ka17_threads_per_block(36).is_some()
            {
                return Err(NnisError::unsupported(
                    "KA18 parallel-score process did not preserve the predeclared KA17 launch policy",
                ));
            }
        }
    }
    let after_model = memory_snapshot(&context)?;

    for _ in 0..arguments.warmups {
        let mut session = model.new_session()?;
        let profile = session.profile_greedy_edge_generation_semantics(&INPUT_IDS, DECODE_STEPS)?;
        require_expected_profile(&profile)?;
    }

    let memory_session = model.new_session()?;
    let after_session = memory_snapshot(&context)?;
    drop(memory_session);

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
                    "greedy output changed across measured KA18 iterations",
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
            "KA18 generation GPU median is non-positive",
        ));
    }

    Ok(Report {
        schema_version: 1,
        benchmark: "smollm2-135m-f16-parallel-score-attention-e2e-ka18",
        backend: "nnis",
        attention_name: arguments.attention_name.as_str(),
        promotion_state: "candidate comparison only; repeated fingerprint-compatible ABBA required before promotion",
        source_repo: SOURCE_REPO,
        source_revision: SOURCE_REVISION,
        source_model_sha256: SOURCE_MODEL_SHA256,
        source_weight_dtype: "bfloat16",
        persisted_execution_weight_dtype: "f32",
        resident_weight_dtype: "f16",
        input_ids: INPUT_IDS.to_vec(),
        decode_steps: DECODE_STEPS,
        max_profile_kv_rows: MAX_PROFILE_KV_ROWS,
        ka17_max_qualified_kv_rows: KA17_MAX_QUALIFIED_KV_ROWS,
        warmup_iterations: arguments.warmups,
        iterations: arguments.iterations,
        metadata: BenchmarkMetadata::collect(&context),
        execution_plan,
        attention_plan,
        parallel_score_candidate_max_supported_kv_rows,
        memory: MemoryReport {
            cuda_free_delta_after_model_bytes: consumed_bytes(&before_model, &after_model),
            cuda_free_delta_after_session_bytes: consumed_bytes(&after_model, &after_session),
            before_model,
            after_model,
            after_session,
        },
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
            "both A and B retain the qualified resident [N,K] grouped-QKV fused-MLP SmolLM2 F16 min-latency projection parent",
            "the KA17 candidate is used only for KV rows 4..=35 with the predeclared block128/block256/block512 schedule and falls back to the qualified reference outside that measured domain",
            "decode32 generation forwards exercise KV rows 4..=34; the candidate process fails before timing unless the kernel can support the complete KA17 1..=35 qualification domain",
            "KA17 established bitwise equality at the final F16 attention output boundary on its qualification corpus, not equality of the internal F32 Q-dot-K reduction order",
            "model construction is excluded from request timing",
            "one process is not promotion evidence; use repeated fingerprint-compatible ABBA runs",
            "generation GPU throughput preserves the Edge definition of 32 generated tokens divided by 31 decoder-forward GPU intervals",
            "CUDA free-memory deltas are diagnostic and are not a cross-runtime memory equivalence metric",
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
                eprintln!("failed to serialize KA18 F16 parallel-score report: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("KA18 SmolLM2 F16 parallel-score benchmark failed: {error}");
            std::process::exit(1);
        }
    }
}
