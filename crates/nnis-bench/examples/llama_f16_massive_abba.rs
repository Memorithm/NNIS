use nnis_bench::{summarize_samples_ms, BenchmarkMetadata, TimingStatistics};
use nnis_jit::JitCompiler;
use nnis_model::{
    F16AttentionPlan, F16CachedAttentionStagedWeightsCandidate, F16ReferenceExecutionPlan,
    F16ReferenceGenerationProfile, F16ReferenceModel, F16ReferencePlan, ModelConfig, WeightDType,
};
use nnis_rt::{Context, Device, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const BENCHMARK_KIND: &str = "trained-llama-f16-massive-abba-v1";
const SUITE_KIND: &str = "nnis-trained-llama-reference-suite-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PlanName {
    Reference,
    Transposed,
    Fused,
    FusedMlp,
    Staged,
    FusedMlpStaged,
}

impl PlanName {
    fn parse(value: &str) -> std::result::Result<Self, String> {
        match value {
            "reference" => Ok(Self::Reference),
            "transposed" => Ok(Self::Transposed),
            "fused" => Ok(Self::Fused),
            "fused_mlp" => Ok(Self::FusedMlp),
            "staged" => Ok(Self::Staged),
            "fused_mlp_staged" => Ok(Self::FusedMlpStaged),
            other => Err(format!(
                "unknown plan {other:?}; expected reference, transposed, fused, fused_mlp, staged, or fused_mlp_staged"
            )),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Transposed => "transposed",
            Self::Fused => "fused",
            Self::FusedMlp => "fused_mlp",
            Self::Staged => "staged",
            Self::FusedMlpStaged => "fused_mlp_staged",
        }
    }

    const fn execution_plan(self) -> F16ReferenceExecutionPlan {
        match self {
            Self::Reference | Self::Staged => {
                F16ReferenceExecutionPlan::reference(F16ReferencePlan::edge_llm_v0_10_0_alignment())
            }
            Self::Transposed => {
                F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_projection_candidate()
            }
            Self::Fused => {
                F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_fused_groups_candidate()
            }
            Self::FusedMlp | Self::FusedMlpStaged => {
                F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_fused_mlp_candidate()
            }
        }
    }

    const fn attention_plan(self) -> F16AttentionPlan {
        match self {
            Self::Staged | Self::FusedMlpStaged => {
                F16AttentionPlan::thor_staged_weights_candidate()
            }
            Self::Reference | Self::Transposed | Self::Fused | Self::FusedMlp => {
                F16AttentionPlan::reference()
            }
        }
    }

    const fn uses_staged_attention(self) -> bool {
        matches!(self, Self::Staged | Self::FusedMlpStaged)
    }
}

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
    suite_path: PathBuf,
    device: i32,
    rounds: usize,
    warmups: usize,
    iterations: usize,
    candidates: Vec<PlanName>,
}

#[derive(Debug, Deserialize)]
struct Provenance {
    source_repo: String,
    source_revision: String,
    source_model_sha256: String,
    source_weight_dtype: String,
    execution_weight_dtype: String,
    tokenizer_sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ReferenceCase {
    name: String,
    family: String,
    target_prompt_tokens: usize,
    decode_steps: usize,
    input_ids: Vec<u32>,
    greedy_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct ReferenceSuite {
    schema_version: u32,
    kind: String,
    source_repo: String,
    source_revision: String,
    source_model_sha256: String,
    source_weight_dtype: String,
    execution_weight_dtype: String,
    tokenizer_sha256: String,
    transformers_version: String,
    expected_config: ModelConfig,
    cases: Vec<ReferenceCase>,
}

impl ReferenceSuite {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 || self.kind != SUITE_KIND {
            return Err(NnisError::invalid_input(format!(
                "reference suite identity mismatch: schema={} kind={:?}",
                self.schema_version, self.kind
            )));
        }
        if self.source_repo.trim().is_empty()
            || self.source_revision.trim().is_empty()
            || self.source_model_sha256.len() != 64
            || self.tokenizer_sha256.len() != 64
            || self.transformers_version.trim().is_empty()
        {
            return Err(NnisError::invalid_input(
                "reference suite provenance is incomplete",
            ));
        }
        if self.source_weight_dtype != "bfloat16" || self.execution_weight_dtype != "f32" {
            return Err(NnisError::invalid_input(format!(
                "massive F16 campaign requires BF16 source and F32 persisted execution weights; got source={} execution={}",
                self.source_weight_dtype, self.execution_weight_dtype
            )));
        }
        self.expected_config.validate_execution_support()?;
        if self.expected_config.weight_dtype != WeightDType::F32 {
            return Err(NnisError::invalid_input(
                "reference suite expected_config.weight_dtype must be f32",
            ));
        }
        if self.cases.is_empty() {
            return Err(NnisError::invalid_input("reference suite has no cases"));
        }
        let mut names = BTreeSet::new();
        for case in &self.cases {
            if case.name.trim().is_empty() || !names.insert(case.name.clone()) {
                return Err(NnisError::invalid_input(format!(
                    "reference case name is empty or duplicated: {:?}",
                    case.name
                )));
            }
            if case.input_ids.is_empty() || case.decode_steps < 2 {
                return Err(NnisError::invalid_input(format!(
                    "case {:?} requires a non-empty prompt and at least two decode steps",
                    case.name
                )));
            }
            if case.target_prompt_tokens != case.input_ids.len() {
                return Err(NnisError::invalid_input(format!(
                    "case {:?} target_prompt_tokens={} but input_ids has {} tokens",
                    case.name,
                    case.target_prompt_tokens,
                    case.input_ids.len()
                )));
            }
            if case.greedy_ids.len() != case.decode_steps {
                return Err(NnisError::invalid_input(format!(
                    "case {:?} decode_steps={} but greedy_ids has {} tokens",
                    case.name,
                    case.decode_steps,
                    case.greedy_ids.len()
                )));
            }
            if case
                .input_ids
                .iter()
                .chain(case.greedy_ids.iter())
                .any(|token| *token as usize >= self.expected_config.vocab_size)
            {
                return Err(NnisError::invalid_input(format!(
                    "case {:?} contains a token outside vocabulary {}",
                    case.name, self.expected_config.vocab_size
                )));
            }
            let required_positions = case
                .input_ids
                .len()
                .checked_add(case.decode_steps - 1)
                .ok_or_else(|| NnisError::invalid_input("case position count overflow"))?;
            if required_positions > self.expected_config.max_position_embeddings {
                return Err(NnisError::invalid_input(format!(
                    "case {:?} requires {required_positions} positions but model supports {}",
                    case.name, self.expected_config.max_position_embeddings
                )));
            }
        }
        Ok(())
    }

    fn max_active_kv_rows(&self) -> usize {
        self.cases
            .iter()
            .map(|case| case.input_ids.len() + case.decode_steps - 1)
            .max()
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Serialize)]
struct MemorySnapshot {
    free_bytes: u64,
    total_bytes: u64,
}

#[derive(Debug, Serialize)]
struct TimingReport {
    statistics: TimingStatistics,
    samples_ms: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct CaseMeasurement {
    case_name: String,
    family: String,
    input_tokens: usize,
    decode_steps: usize,
    success: bool,
    error: Option<String>,
    generated_ids: Vec<u32>,
    exact_oracle_greedy: bool,
    session_setup_wall: Option<TimingReport>,
    generation_wall: Option<TimingReport>,
    request_total_wall: Option<TimingReport>,
    prefill_gpu: Option<TimingReport>,
    generation_stage_gpu: Option<TimingReport>,
    generation_tokens_per_second_edge_definition_from_gpu_median: Option<f64>,
}

#[derive(Debug, Serialize)]
struct BlockReport {
    slot: &'static str,
    plan_name: &'static str,
    success: bool,
    error: Option<String>,
    model_setup_wall_ms: Option<f64>,
    staged_candidate_max_supported_kv_rows: Option<usize>,
    memory_before_model: Option<MemorySnapshot>,
    memory_after_model: Option<MemorySnapshot>,
    cases: Vec<CaseMeasurement>,
}

struct BlockExecution {
    model_setup_wall_ms: f64,
    staged_max_rows: Option<usize>,
    memory_before: MemorySnapshot,
    memory_after: MemorySnapshot,
    cases: Vec<CaseMeasurement>,
}

#[derive(Debug, Serialize)]
struct CaseRoundEvidence {
    case_name: String,
    candidate: &'static str,
    complete: bool,
    generation_stage_gpu_reference_ms: Option<f64>,
    generation_stage_gpu_candidate_ms: Option<f64>,
    generation_stage_gpu_relative_improvement: Option<f64>,
    generation_wall_relative_improvement: Option<f64>,
    request_total_wall_relative_improvement: Option<f64>,
}

#[derive(Debug, Serialize)]
struct RoundReport {
    round: usize,
    case_order: &'static str,
    blocks: Vec<BlockReport>,
    case_evidence: Vec<CaseRoundEvidence>,
}

#[derive(Debug, Serialize)]
struct CandidateReport {
    candidate: &'static str,
    rounds: Vec<RoundReport>,
}

#[derive(Debug, Serialize)]
struct MassiveReport {
    schema_version: u32,
    benchmark: &'static str,
    backend: &'static str,
    promotion_state: &'static str,
    source_repo: String,
    source_revision: String,
    source_model_sha256: String,
    source_weight_dtype: String,
    persisted_execution_weight_dtype: String,
    tokenizer_sha256: String,
    transformers_version: String,
    expected_config: ModelConfig,
    case_count: usize,
    max_active_kv_rows: usize,
    rounds_per_candidate: usize,
    warmups_per_slot: usize,
    iterations_per_slot: usize,
    candidates: Vec<&'static str>,
    metadata: BenchmarkMetadata,
    candidate_reports: Vec<CandidateReport>,
    campaign_complete: bool,
    limitations: Vec<&'static str>,
}

fn positive_usize(name: &str, value: String) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_candidates(value: &str) -> std::result::Result<Vec<PlanName>, String> {
    let mut plans = BTreeSet::new();
    for item in value.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let plan = PlanName::parse(item)?;
        if plan == PlanName::Reference {
            return Err("reference is implicit and must not appear in --candidates".to_string());
        }
        plans.insert(plan);
    }
    if plans.is_empty() {
        return Err("--candidates resolved to an empty set".to_string());
    }
    Ok(plans.into_iter().collect())
}

fn parse_arguments() -> std::result::Result<Arguments, String> {
    let mut args = env::args().skip(1);
    let mut model_dir = None;
    let mut suite_path = None;
    let mut device = 0_i32;
    let mut rounds = 4_usize;
    let mut warmups = 1_usize;
    let mut iterations = 3_usize;
    let mut candidates = parse_candidates("transposed,fused,fused_mlp,staged,fused_mlp_staged")?;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => {
                model_dir = Some(PathBuf::from(
                    args.next().ok_or("--model requires a directory")?,
                ));
            }
            "--suite" => {
                suite_path = Some(PathBuf::from(
                    args.next().ok_or("--suite requires a JSON file")?,
                ));
            }
            "--device" => {
                device = args
                    .next()
                    .ok_or("--device requires an ordinal")?
                    .parse::<i32>()
                    .map_err(|error| format!("invalid --device: {error}"))?;
            }
            "--rounds" => {
                rounds =
                    positive_usize("--rounds", args.next().ok_or("--rounds requires a value")?)?;
            }
            "--warmups" => {
                warmups = positive_usize(
                    "--warmups",
                    args.next().ok_or("--warmups requires a value")?,
                )?;
            }
            "--iterations" => {
                iterations = positive_usize(
                    "--iterations",
                    args.next().ok_or("--iterations requires a value")?,
                )?;
            }
            "--candidates" => {
                candidates = parse_candidates(
                    &args
                        .next()
                        .ok_or("--candidates requires a comma-separated plan list")?,
                )?;
            }
            "--help" | "-h" => {
                return Err("usage: llama_f16_massive_abba --model DIR --suite FILE [--device N] [--rounds N] [--warmups N] [--iterations N] [--candidates transposed,fused,fused_mlp,staged,fused_mlp_staged]".to_string());
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }

    if device < 0 {
        return Err("--device must be non-negative".to_string());
    }
    Ok(Arguments {
        model_dir: model_dir.ok_or("missing --model DIR")?,
        suite_path: suite_path.ok_or("missing --suite FILE")?,
        device,
        rounds,
        warmups,
        iterations,
        candidates,
    })
}

fn read_suite(path: &Path) -> Result<ReferenceSuite> {
    let bytes = fs::read(path).map_err(|error| NnisError::io("read reference suite", error))?;
    let suite: ReferenceSuite = serde_json::from_slice(&bytes).map_err(|error| {
        NnisError::invalid_input(format!("invalid reference suite JSON: {error}"))
    })?;
    suite.validate()?;
    Ok(suite)
}

fn validate_provenance(model_dir: &Path, suite: &ReferenceSuite) -> Result<()> {
    let bytes = fs::read(model_dir.join("provenance.json"))
        .map_err(|error| NnisError::io("read model provenance", error))?;
    let provenance: Provenance = serde_json::from_slice(&bytes)
        .map_err(|error| NnisError::invalid_input(format!("invalid provenance JSON: {error}")))?;
    if provenance.source_repo != suite.source_repo
        || provenance.source_revision != suite.source_revision
        || provenance.source_model_sha256 != suite.source_model_sha256
        || provenance.source_weight_dtype != suite.source_weight_dtype
        || provenance.execution_weight_dtype != suite.execution_weight_dtype
        || provenance.tokenizer_sha256 != suite.tokenizer_sha256
    {
        return Err(NnisError::invalid_input(format!(
            "model provenance and reference suite differ: model={}@{} model_sha={} tokenizer_sha={} source_dtype={} execution_dtype={}",
            provenance.source_repo,
            provenance.source_revision,
            provenance.source_model_sha256,
            provenance.tokenizer_sha256,
            provenance.source_weight_dtype,
            provenance.execution_weight_dtype
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

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn require_expected_profile(
    profile: &F16ReferenceGenerationProfile,
    case: &ReferenceCase,
) -> Result<()> {
    if profile.generated_ids != case.greedy_ids {
        let divergence = profile
            .generated_ids
            .iter()
            .zip(case.greedy_ids.iter())
            .position(|(actual, expected)| actual != expected)
            .or_else(|| {
                if profile.generated_ids.len() != case.greedy_ids.len() {
                    Some(profile.generated_ids.len().min(case.greedy_ids.len()))
                } else {
                    None
                }
            });
        return Err(NnisError::invalid_input(format!(
            "case {:?} greedy trajectory differs from trusted oracle at {divergence:?}: actual={:?} expected={:?}",
            case.name, profile.generated_ids, case.greedy_ids
        )));
    }
    if profile.generated_tokens != case.decode_steps
        || profile.generation_forward_runs != case.decode_steps - 1
        || profile.generation_forward_gpu_ms.len() != case.decode_steps - 1
        || profile.sampling_included_in_generation_stage_gpu_time
        || profile.final_generated_token_consumed_by_model
    {
        return Err(NnisError::invalid_input(format!(
            "case {:?} generation metric contract drifted: {profile:?}",
            case.name
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

fn failed_case(case: &ReferenceCase, error: impl ToString) -> CaseMeasurement {
    CaseMeasurement {
        case_name: case.name.clone(),
        family: case.family.clone(),
        input_tokens: case.input_ids.len(),
        decode_steps: case.decode_steps,
        success: false,
        error: Some(error.to_string()),
        generated_ids: Vec::new(),
        exact_oracle_greedy: false,
        session_setup_wall: None,
        generation_wall: None,
        request_total_wall: None,
        prefill_gpu: None,
        generation_stage_gpu: None,
        generation_tokens_per_second_edge_definition_from_gpu_median: None,
    }
}

fn measure_case(
    model: &F16ReferenceModel,
    case: &ReferenceCase,
    warmups: usize,
    iterations: usize,
) -> Result<CaseMeasurement> {
    for _ in 0..warmups {
        let mut session = model.new_session()?;
        let profile =
            session.profile_greedy_edge_generation_semantics(&case.input_ids, case.decode_steps)?;
        require_expected_profile(&profile, case)?;
    }

    let mut session_setup_samples = Vec::with_capacity(iterations);
    let mut generation_wall_samples = Vec::with_capacity(iterations);
    let mut request_total_samples = Vec::with_capacity(iterations);
    let mut prefill_gpu_samples = Vec::with_capacity(iterations);
    let mut generation_gpu_samples = Vec::with_capacity(iterations);
    let mut generated_ids = Vec::new();

    for iteration in 0..iterations {
        let request_start = Instant::now();
        let setup_start = Instant::now();
        let mut session = model.new_session()?;
        session_setup_samples.push(elapsed_ms(setup_start));

        let generation_start = Instant::now();
        let profile =
            session.profile_greedy_edge_generation_semantics(&case.input_ids, case.decode_steps)?;
        generation_wall_samples.push(elapsed_ms(generation_start));
        request_total_samples.push(elapsed_ms(request_start));
        require_expected_profile(&profile, case)?;
        prefill_gpu_samples.push(profile.prefill_gpu_ms);
        generation_gpu_samples.push(profile.generation_stage_total_gpu_ms);
        if iteration == 0 {
            generated_ids = profile.generated_ids;
        } else if profile.generated_ids != generated_ids {
            return Err(NnisError::invalid_input(format!(
                "case {:?} greedy output changed across measured iterations",
                case.name
            )));
        }
    }

    let generation_stage_gpu = timing_report(generation_gpu_samples)?;
    let gpu_median = generation_stage_gpu.statistics.median_ms;
    if !gpu_median.is_finite() || gpu_median <= 0.0 {
        return Err(NnisError::unsupported(format!(
            "case {:?} generation GPU median is non-positive",
            case.name
        )));
    }

    Ok(CaseMeasurement {
        case_name: case.name.clone(),
        family: case.family.clone(),
        input_tokens: case.input_ids.len(),
        decode_steps: case.decode_steps,
        success: true,
        error: None,
        generated_ids,
        exact_oracle_greedy: true,
        session_setup_wall: Some(timing_report(session_setup_samples)?),
        generation_wall: Some(timing_report(generation_wall_samples)?),
        request_total_wall: Some(timing_report(request_total_samples)?),
        prefill_gpu: Some(timing_report(prefill_gpu_samples)?),
        generation_tokens_per_second_edge_definition_from_gpu_median: Some(
            case.decode_steps as f64 / (gpu_median / 1_000.0),
        ),
        generation_stage_gpu: Some(generation_stage_gpu),
    })
}

fn run_block(
    arguments: &Arguments,
    suite: &ReferenceSuite,
    plan_name: PlanName,
    slot: &'static str,
    reverse_cases: bool,
) -> BlockReport {
    let result = (|| -> Result<BlockExecution> {
        let device = Device::get(arguments.device)?;
        let context = Context::new(&device)?;
        let construction_stream = Stream::new(&context)?;
        let memory_before = memory_snapshot(&context)?;
        let staged_max_rows = if plan_name.uses_staged_attention() {
            let candidate =
                F16CachedAttentionStagedWeightsCandidate::load(&context, &JitCompiler::new())?;
            let max_rows = candidate.max_supported_kv_rows();
            if max_rows < suite.max_active_kv_rows() {
                return Err(NnisError::unsupported(format!(
                    "staged attention supports at most {max_rows} KV rows on this kernel/device; suite requires {} so fallback would contaminate the campaign",
                    suite.max_active_kv_rows()
                )));
            }
            Some(max_rows)
        } else {
            None
        };

        let setup_start = Instant::now();
        let model = F16ReferenceModel::load_directory_with_execution_and_attention_plan(
            &context,
            &construction_stream,
            &arguments.model_dir,
            plan_name.execution_plan(),
            plan_name.attention_plan(),
        )?;
        construction_stream.synchronize()?;
        let model_setup_wall_ms = elapsed_ms(setup_start);
        if model.config() != &suite.expected_config {
            return Err(NnisError::invalid_input(format!(
                "loaded model config differs from suite expected_config: actual={:?} expected={:?}",
                model.config(),
                suite.expected_config
            )));
        }
        if model.execution_plan() != plan_name.execution_plan()
            || model.attention_plan() != plan_name.attention_plan()
        {
            return Err(NnisError::unsupported(format!(
                "model did not preserve requested plan {}",
                plan_name.as_str()
            )));
        }
        let memory_after = memory_snapshot(&context)?;

        let mut cases = Vec::with_capacity(suite.cases.len());
        let indices: Vec<usize> = if reverse_cases {
            (0..suite.cases.len()).rev().collect()
        } else {
            (0..suite.cases.len()).collect()
        };
        for index in indices {
            let case = &suite.cases[index];
            match measure_case(&model, case, arguments.warmups, arguments.iterations) {
                Ok(measurement) => cases.push(measurement),
                Err(error) => cases.push(failed_case(case, error)),
            }
        }
        Ok(BlockExecution {
            model_setup_wall_ms,
            staged_max_rows,
            memory_before,
            memory_after,
            cases,
        })
    })();

    match result {
        Ok(execution) => {
            let success = execution.cases.iter().all(|case| case.success);
            BlockReport {
                slot,
                plan_name: plan_name.as_str(),
                success,
                error: if success {
                    None
                } else {
                    Some("one or more cases failed correctness or execution".to_string())
                },
                model_setup_wall_ms: Some(execution.model_setup_wall_ms),
                staged_candidate_max_supported_kv_rows: execution.staged_max_rows,
                memory_before_model: Some(execution.memory_before),
                memory_after_model: Some(execution.memory_after),
                cases: execution.cases,
            }
        }
        Err(error) => BlockReport {
            slot,
            plan_name: plan_name.as_str(),
            success: false,
            error: Some(error.to_string()),
            model_setup_wall_ms: None,
            staged_candidate_max_supported_kv_rows: None,
            memory_before_model: None,
            memory_after_model: None,
            cases: suite
                .cases
                .iter()
                .map(|case| failed_case(case, "plan block failed before case execution"))
                .collect(),
        },
    }
}

fn case_in_block<'a>(block: &'a BlockReport, case_name: &str) -> Option<&'a CaseMeasurement> {
    block.cases.iter().find(|case| case.case_name == case_name)
}

fn median_of_metric(case: &CaseMeasurement, metric: &str) -> Option<f64> {
    let report = match metric {
        "gpu" => case.generation_stage_gpu.as_ref(),
        "wall" => case.generation_wall.as_ref(),
        "request" => case.request_total_wall.as_ref(),
        _ => None,
    }?;
    Some(report.statistics.median_ms)
}

fn paired_metric(blocks: &[BlockReport], case_name: &str, metric: &str) -> Option<(f64, f64, f64)> {
    if blocks.len() != 4 {
        return None;
    }
    let values: Vec<f64> = blocks
        .iter()
        .map(|block| {
            case_in_block(block, case_name).and_then(|case| median_of_metric(case, metric))
        })
        .collect::<Option<Vec<_>>>()?;
    let reference = (values[0] + values[3]) / 2.0;
    let candidate = (values[1] + values[2]) / 2.0;
    if !reference.is_finite() || reference <= 0.0 || !candidate.is_finite() || candidate <= 0.0 {
        return None;
    }
    Some((reference, candidate, (reference - candidate) / reference))
}

fn summarize_round(
    candidate: PlanName,
    blocks: &[BlockReport],
    suite: &ReferenceSuite,
) -> Vec<CaseRoundEvidence> {
    suite
        .cases
        .iter()
        .map(|case| {
            let gpu = paired_metric(blocks, &case.name, "gpu");
            let wall = paired_metric(blocks, &case.name, "wall");
            let request = paired_metric(blocks, &case.name, "request");
            CaseRoundEvidence {
                case_name: case.name.clone(),
                candidate: candidate.as_str(),
                complete: gpu.is_some() && wall.is_some() && request.is_some(),
                generation_stage_gpu_reference_ms: gpu.map(|value| value.0),
                generation_stage_gpu_candidate_ms: gpu.map(|value| value.1),
                generation_stage_gpu_relative_improvement: gpu.map(|value| value.2),
                generation_wall_relative_improvement: wall.map(|value| value.2),
                request_total_wall_relative_improvement: request.map(|value| value.2),
            }
        })
        .collect()
}

fn run(arguments: Arguments) -> Result<MassiveReport> {
    let suite = read_suite(&arguments.suite_path)?;
    validate_provenance(&arguments.model_dir, &suite)?;

    let device = Device::get(arguments.device)?;
    let metadata_context = Context::new(&device)?;
    let metadata = BenchmarkMetadata::collect(&metadata_context);
    drop(metadata_context);

    let mut candidate_reports = Vec::with_capacity(arguments.candidates.len());
    let mut campaign_complete = true;
    for candidate in &arguments.candidates {
        let mut rounds = Vec::with_capacity(arguments.rounds);
        for round in 0..arguments.rounds {
            let reverse_cases = round % 2 == 1;
            let blocks = vec![
                run_block(&arguments, &suite, PlanName::Reference, "A1", reverse_cases),
                run_block(&arguments, &suite, *candidate, "B1", reverse_cases),
                run_block(&arguments, &suite, *candidate, "B2", reverse_cases),
                run_block(&arguments, &suite, PlanName::Reference, "A2", reverse_cases),
            ];
            let case_evidence = summarize_round(*candidate, &blocks, &suite);
            if blocks.iter().any(|block| !block.success)
                || case_evidence.iter().any(|case| !case.complete)
            {
                campaign_complete = false;
            }
            rounds.push(RoundReport {
                round,
                case_order: if reverse_cases { "reverse" } else { "forward" },
                blocks,
                case_evidence,
            });
        }
        candidate_reports.push(CandidateReport {
            candidate: candidate.as_str(),
            rounds,
        });
    }

    Ok(MassiveReport {
        schema_version: 1,
        benchmark: BENCHMARK_KIND,
        backend: "nnis",
        promotion_state: "cross-model exploratory qualification only; no runtime default, model-family support, or candidate promotion is authorized by this campaign alone",
        source_repo: suite.source_repo.clone(),
        source_revision: suite.source_revision.clone(),
        source_model_sha256: suite.source_model_sha256.clone(),
        source_weight_dtype: suite.source_weight_dtype.clone(),
        persisted_execution_weight_dtype: suite.execution_weight_dtype.clone(),
        tokenizer_sha256: suite.tokenizer_sha256.clone(),
        transformers_version: suite.transformers_version.clone(),
        expected_config: suite.expected_config.clone(),
        case_count: suite.cases.len(),
        max_active_kv_rows: suite.max_active_kv_rows(),
        rounds_per_candidate: arguments.rounds,
        warmups_per_slot: arguments.warmups,
        iterations_per_slot: arguments.iterations,
        candidates: arguments
            .candidates
            .iter()
            .map(|plan| plan.as_str())
            .collect(),
        metadata,
        candidate_reports,
        campaign_complete,
        limitations: vec![
            "the trusted oracle is the pinned Transformers greedy trajectory from the converted checkpoint fixture",
            "the F16 runtime narrows the persisted F32 graph to resident F16; exact greedy equality is a semantic gate, not full-logit numerical equivalence",
            "candidate plans remain explicit comparison surfaces and are not promoted by cross-model evidence alone",
            "KA17 parallel-score attention is intentionally excluded because its validator is fail-closed to SmolLM2-135M geometry",
            "staged attention blocks fail before measurement unless the candidate supports the suite's complete KV-row range, preventing silent resource fallback from contaminating comparisons",
            "model construction is excluded from request latency but reported separately for diagnostics",
            "CUDA free-memory snapshots are diagnostics and are not exact resident-storage or cross-runtime memory-equivalence metrics",
            "ABBA slots are separate model constructions inside one process; the outer launcher must repeat the complete campaign under distinct run_context_id values for independent-run evidence",
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
                eprintln!("failed to serialize massive Llama F16 campaign report: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("massive Llama F16 campaign failed before evidence collection: {error}");
            std::process::exit(1);
        }
    }
}
