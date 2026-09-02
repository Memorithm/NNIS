use nnis_bench::{summarize_samples_ms, BenchmarkMetadata};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

const EXPECTED_INPUT_IDS: [u64; 3] = [22_007, 6_463, 314];
const EXPECTED_GREEDY_IDS: [u64; 32] = [
    260, 3_075, 338, 6_650, 260, 2_591, 284, 260, 8_872, 1_592, 30, 198, 198, 504, 8_872, 314, 253,
    8_304, 282, 260, 2_591, 30, 657, 314, 253, 19_284, 1_248, 338, 21_837, 260, 2_591, 30,
];
const EXPECTED_REPORTS: usize = 24;
const EXPECTED_ROUNDS: usize = 6;
const EXPECTED_ITERATIONS: usize = 7;
const MIN_REVIEW_EFFECT: f64 = 0.03;
const CAMPAIGN_SUFFIX: &str = "-smollm2-f16-parallel-score-e2e-abba-v1";
const EXPECTED_SOURCE_REPO: &str = "HuggingFaceTB/SmolLM2-135M";
const EXPECTED_SOURCE_REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
const EXPECTED_SOURCE_MODEL_SHA256: &str =
    "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1";

#[derive(Debug, Deserialize)]
struct Bundle {
    schema_version: u32,
    campaign: String,
    git_commit: String,
    run_context_id: String,
    raw_reports: Vec<RawReport>,
}

#[derive(Debug, Deserialize)]
struct RawReport {
    round: usize,
    slot: String,
    mode: String,
    report: Value,
}

#[derive(Debug, Clone, Serialize)]
struct RoundEvidence {
    round: usize,
    a1_reference_ms: f64,
    b1_candidate_ms: f64,
    b2_candidate_ms: f64,
    a2_reference_ms: f64,
    paired_relative_improvement: f64,
    ab_relative_improvement: f64,
    ba_relative_improvement: f64,
}

#[derive(Debug, Clone, Serialize)]
struct MetricSummary {
    candidate_round_wins: usize,
    reference_round_wins: usize,
    ties: usize,
    rounds_at_or_above_3pct: usize,
    median_paired_relative_improvement: f64,
    mean_paired_relative_improvement: f64,
    min_paired_relative_improvement: f64,
    max_paired_relative_improvement: f64,
    sample_stdev_paired_relative_improvement: f64,
    median_ab_relative_improvement: f64,
    median_ba_relative_improvement: f64,
    median_order_bias_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CampaignEvidence {
    campaign: String,
    git_commit: String,
    run_context_id: String,
    validated_reports: usize,
    exact_greedy_reports: usize,
    generation_stage_gpu_rounds: Vec<RoundEvidence>,
    generation_stage_gpu: MetricSummary,
    generation_wall: MetricSummary,
    request_total_wall: MetricSummary,
}

#[derive(Debug, Serialize)]
struct ConsensusReport {
    schema_version: u32,
    kind: &'static str,
    environment_compatible_across_distinct_campaigns: bool,
    exact_git_commit_equal: bool,
    campaign_count: usize,
    minimum_review_effect: f64,
    first: CampaignEvidence,
    second: CampaignEvidence,
    promotion_review_eligible: bool,
    claim_boundary: &'static str,
}

#[derive(Debug, Clone)]
struct ValidatedReport {
    round: usize,
    slot: String,
    mode: String,
    metadata: BenchmarkMetadata,
    generation_stage_gpu_ms: f64,
    generation_wall_ms: f64,
    request_total_wall_ms: f64,
    execution_plan: Value,
}

fn read_bundle(path: &Path) -> Result<Bundle, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {}: {error}", path.display()))
}

fn require(condition: bool, message: impl Into<String>) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

fn value_at<'a>(value: &'a Value, pointer: &str) -> Result<&'a Value, String> {
    value
        .pointer(pointer)
        .ok_or_else(|| format!("missing JSON field {pointer}"))
}

fn string_at<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value_at(value, pointer)?
        .as_str()
        .ok_or_else(|| format!("JSON field {pointer} is not a string"))
}

fn usize_at(value: &Value, pointer: &str) -> Result<usize, String> {
    let integer = value_at(value, pointer)?
        .as_u64()
        .ok_or_else(|| format!("JSON field {pointer} is not an unsigned integer"))?;
    usize::try_from(integer).map_err(|_| format!("JSON field {pointer} exceeds usize"))
}

fn bool_at(value: &Value, pointer: &str) -> Result<bool, String> {
    value_at(value, pointer)?
        .as_bool()
        .ok_or_else(|| format!("JSON field {pointer} is not a bool"))
}

fn u64_array_at(value: &Value, pointer: &str) -> Result<Vec<u64>, String> {
    value_at(value, pointer)?
        .as_array()
        .ok_or_else(|| format!("JSON field {pointer} is not an array"))?
        .iter()
        .map(|item| {
            item.as_u64()
                .ok_or_else(|| format!("JSON field {pointer} contains a non-u64 value"))
        })
        .collect()
}

fn recompute_median(report: &Value, field: &str) -> Result<f64, String> {
    let samples_pointer = format!("/{field}/samples_ms");
    let samples = value_at(report, &samples_pointer)?
        .as_array()
        .ok_or_else(|| format!("JSON field {samples_pointer} is not an array"))?
        .iter()
        .map(|item| {
            item.as_f64()
                .ok_or_else(|| format!("JSON field {samples_pointer} contains a non-f64 value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    require(
        samples.len() == EXPECTED_ITERATIONS,
        format!(
            "{field} contains {} samples; expected {EXPECTED_ITERATIONS}",
            samples.len()
        ),
    )?;
    let recomputed = summarize_samples_ms(&samples)
        .map_err(|error| format!("recompute {field}: {error}"))?
        .median_ms;
    let declared_pointer = format!("/{field}/statistics/median_ms");
    let declared = value_at(report, &declared_pointer)?
        .as_f64()
        .ok_or_else(|| format!("JSON field {declared_pointer} is not an f64"))?;
    require(
        declared.to_bits() == recomputed.to_bits(),
        format!("{field} declared median {declared:?} does not match recomputed {recomputed:?}"),
    )?;
    Ok(recomputed)
}

fn validate_attention_plan(mode: &str, report: &Value) -> Result<(), String> {
    let kernel = string_at(report, "/attention_plan/kernel")?;
    let staged_min = usize_at(report, "/attention_plan/staged_min_kv_rows")?;
    match mode {
        "reference" => {
            require(
                kernel == "reference_per_position_barriers",
                format!("reference report selected unexpected kernel {kernel:?}"),
            )?;
            require(staged_min == 0, "reference staged_min_kv_rows must be zero")?;
            require(
                report
                    .pointer("/attention_plan/parallel_score_policy")
                    .is_none(),
                "reference report unexpectedly carries parallel_score_policy",
            )?;
            require(
                value_at(report, "/parallel_score_candidate_max_supported_kv_rows")?.is_null(),
                "reference report unexpectedly loaded the parallel-score candidate",
            )?;
        }
        "parallel-score-ka17" => {
            require(
                kernel == "parallel_score_candidate",
                format!("candidate report selected unexpected kernel {kernel:?}"),
            )?;
            require(staged_min == 0, "candidate staged_min_kv_rows must be zero")?;
            require(
                string_at(report, "/attention_plan/parallel_score_policy")?
                    == "ka17_smol_lm2_short_context_v1",
                "candidate report does not carry the fixed KA17 policy",
            )?;
            require(
                usize_at(report, "/parallel_score_candidate_max_supported_kv_rows")? >= 34,
                "parallel-score candidate does not support the complete decode32 KV range",
            )?;
        }
        other => return Err(format!("unexpected attention mode {other:?}")),
    }
    Ok(())
}

fn validate_raw_report(
    entry: &RawReport,
    bundle_commit: &str,
    bundle_run_context: &str,
) -> Result<ValidatedReport, String> {
    let report = &entry.report;
    require(
        usize_at(report, "/schema_version")? == 1,
        "report schema_version must be 1",
    )?;
    require(
        string_at(report, "/benchmark")? == "smollm2-135m-f16-attention-plan-e2e",
        "unexpected benchmark kind",
    )?;
    require(
        string_at(report, "/backend")? == "nnis",
        "unexpected backend",
    )?;
    require(
        string_at(report, "/attention_name")? == entry.mode,
        "entry mode and report attention_name differ",
    )?;
    require(
        string_at(report, "/source_repo")? == EXPECTED_SOURCE_REPO,
        "unexpected source_repo",
    )?;
    require(
        string_at(report, "/source_revision")? == EXPECTED_SOURCE_REVISION,
        "unexpected source_revision",
    )?;
    require(
        string_at(report, "/source_model_sha256")? == EXPECTED_SOURCE_MODEL_SHA256,
        "unexpected source_model_sha256",
    )?;
    require(
        string_at(report, "/source_weight_dtype")? == "bfloat16",
        "unexpected source_weight_dtype",
    )?;
    require(
        string_at(report, "/persisted_execution_weight_dtype")? == "f32",
        "unexpected persisted execution dtype",
    )?;
    require(
        string_at(report, "/resident_weight_dtype")? == "f16",
        "unexpected resident weight dtype",
    )?;
    require(
        u64_array_at(report, "/input_ids")? == EXPECTED_INPUT_IDS,
        "input ids differ from the qualified fixture",
    )?;
    require(
        usize_at(report, "/decode_steps")? == 32,
        "decode_steps must be 32",
    )?;
    require(
        usize_at(report, "/max_profile_kv_rows")? == 34,
        "max_profile_kv_rows must be 34",
    )?;
    require(
        usize_at(report, "/warmup_iterations")? == 2,
        "warmup_iterations must be 2",
    )?;
    require(
        usize_at(report, "/iterations")? == EXPECTED_ITERATIONS,
        "iterations must be 7",
    )?;
    require(
        bool_at(report, "/exact_greedy_32_of_32")?,
        "report did not preserve exact greedy 32-of-32",
    )?;
    require(
        u64_array_at(report, "/generated_ids")? == EXPECTED_GREEDY_IDS,
        "generated ids differ from the qualified greedy trajectory",
    )?;
    require(
        usize_at(report, "/generation_forward_runs_per_request")? == 31,
        "generation forward-run count must be 31",
    )?;
    require(
        !bool_at(report, "/sampling_included_in_generation_stage_gpu_time")?,
        "sampling must remain outside generation-stage GPU timing",
    )?;
    require(
        !bool_at(report, "/final_generated_token_consumed_by_model")?,
        "final generated token must remain outside decoder-forward timing",
    )?;
    validate_attention_plan(&entry.mode, report)?;

    let metadata: BenchmarkMetadata =
        serde_json::from_value(value_at(report, "/metadata")?.clone())
            .map_err(|error| format!("decode benchmark metadata: {error}"))?;
    require(
        metadata.git_commit == bundle_commit,
        format!(
            "report git commit {:?} differs from bundle commit {bundle_commit:?}",
            metadata.git_commit
        ),
    )?;
    require(
        metadata.git_dirty == Some(false),
        "benchmark report must come from a clean git checkout",
    )?;
    require(
        metadata.environment_fingerprint.run_context_id.as_deref() == Some(bundle_run_context),
        "report run_context_id differs from bundle run_context_id",
    )?;

    Ok(ValidatedReport {
        round: entry.round,
        slot: entry.slot.clone(),
        mode: entry.mode.clone(),
        metadata,
        generation_stage_gpu_ms: recompute_median(report, "generation_stage_gpu")?,
        generation_wall_ms: recompute_median(report, "generation_wall")?,
        request_total_wall_ms: recompute_median(report, "request_total_wall")?,
        execution_plan: value_at(report, "/execution_plan")?.clone(),
    })
}

fn validate_campaign(
    bundle: Bundle,
) -> Result<(CampaignEvidence, BenchmarkMetadata, Value), String> {
    require(
        bundle.schema_version == 1,
        "bundle schema_version must be 1",
    )?;
    require(
        bundle.campaign.ends_with(CAMPAIGN_SUFFIX),
        format!("unexpected campaign contract {:?}", bundle.campaign),
    )?;
    require(
        !bundle.git_commit.trim().is_empty(),
        "bundle git_commit is empty",
    )?;
    require(
        !bundle.run_context_id.trim().is_empty(),
        "bundle run_context_id is empty",
    )?;
    require(
        bundle.raw_reports.len() == EXPECTED_REPORTS,
        format!(
            "bundle contains {} raw reports; expected {EXPECTED_REPORTS}",
            bundle.raw_reports.len()
        ),
    )?;

    let mut reports = Vec::with_capacity(EXPECTED_REPORTS);
    for entry in &bundle.raw_reports {
        reports.push(validate_raw_report(
            entry,
            &bundle.git_commit,
            &bundle.run_context_id,
        )?);
    }

    let first_metadata = reports
        .first()
        .ok_or_else(|| "campaign contains no reports".to_string())?
        .metadata
        .clone();
    let first_execution_plan = reports[0].execution_plan.clone();
    for report in &reports {
        first_metadata
            .require_compatible_environment(&report.metadata)
            .map_err(|error| format!("within-campaign environment mismatch: {error}"))?;
        require(
            report.execution_plan == first_execution_plan,
            "execution plan changed within one campaign",
        )?;
    }

    let mut by_round = BTreeMap::<usize, BTreeMap<String, ValidatedReport>>::new();
    for report in reports {
        let previous = by_round
            .entry(report.round)
            .or_default()
            .insert(report.slot.clone(), report);
        require(previous.is_none(), "duplicate round/slot report")?;
    }
    require(
        by_round.len() == EXPECTED_ROUNDS,
        format!(
            "campaign has {} rounds; expected {EXPECTED_ROUNDS}",
            by_round.len()
        ),
    )?;
    require(
        by_round.keys().copied().eq(1..=EXPECTED_ROUNDS),
        "campaign rounds must be exactly 1 through 6",
    )?;

    let gpu_rounds = round_evidence(&by_round, |report| report.generation_stage_gpu_ms)?;
    let wall_rounds = round_evidence(&by_round, |report| report.generation_wall_ms)?;
    let request_rounds = round_evidence(&by_round, |report| report.request_total_wall_ms)?;

    let evidence = CampaignEvidence {
        campaign: bundle.campaign,
        git_commit: bundle.git_commit,
        run_context_id: bundle.run_context_id,
        validated_reports: EXPECTED_REPORTS,
        exact_greedy_reports: EXPECTED_REPORTS,
        generation_stage_gpu: summarize_rounds(&gpu_rounds),
        generation_wall: summarize_rounds(&wall_rounds),
        request_total_wall: summarize_rounds(&request_rounds),
        generation_stage_gpu_rounds: gpu_rounds,
    };
    Ok((evidence, first_metadata, first_execution_plan))
}

fn round_evidence<F>(
    by_round: &BTreeMap<usize, BTreeMap<String, ValidatedReport>>,
    metric: F,
) -> Result<Vec<RoundEvidence>, String>
where
    F: Fn(&ValidatedReport) -> f64,
{
    let mut rows = Vec::with_capacity(EXPECTED_ROUNDS);
    for (&round, slots) in by_round {
        require(
            slots.len() == 4
                && slots.contains_key("A1")
                && slots.contains_key("B1")
                && slots.contains_key("B2")
                && slots.contains_key("A2"),
            format!("round {round} does not contain exact ABBA slots"),
        )?;
        let a1 = &slots["A1"];
        let b1 = &slots["B1"];
        let b2 = &slots["B2"];
        let a2 = &slots["A2"];
        require(
            a1.mode == "reference"
                && a2.mode == "reference"
                && b1.mode == "parallel-score-ka17"
                && b2.mode == "parallel-score-ka17",
            format!("round {round} has invalid ABBA attention modes"),
        )?;
        let a1_ms = metric(a1);
        let b1_ms = metric(b1);
        let b2_ms = metric(b2);
        let a2_ms = metric(a2);
        require(
            [a1_ms, b1_ms, b2_ms, a2_ms]
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            format!("round {round} contains non-positive or non-finite timing"),
        )?;
        let reference_pair = (a1_ms + a2_ms) / 2.0;
        let candidate_pair = (b1_ms + b2_ms) / 2.0;
        rows.push(RoundEvidence {
            round,
            a1_reference_ms: a1_ms,
            b1_candidate_ms: b1_ms,
            b2_candidate_ms: b2_ms,
            a2_reference_ms: a2_ms,
            paired_relative_improvement: (reference_pair - candidate_pair) / reference_pair,
            ab_relative_improvement: (a1_ms - b1_ms) / a1_ms,
            ba_relative_improvement: (a2_ms - b2_ms) / a2_ms,
        });
    }
    Ok(rows)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn sample_stdev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let sum = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    (sum / (values.len() - 1) as f64).sqrt()
}

fn summarize_rounds(rows: &[RoundEvidence]) -> MetricSummary {
    let paired = rows
        .iter()
        .map(|row| row.paired_relative_improvement)
        .collect::<Vec<_>>();
    let ab = rows
        .iter()
        .map(|row| row.ab_relative_improvement)
        .collect::<Vec<_>>();
    let ba = rows
        .iter()
        .map(|row| row.ba_relative_improvement)
        .collect::<Vec<_>>();
    let mean = paired.iter().sum::<f64>() / paired.len() as f64;
    MetricSummary {
        candidate_round_wins: paired.iter().filter(|value| **value > 0.0).count(),
        reference_round_wins: paired.iter().filter(|value| **value < 0.0).count(),
        ties: paired.iter().filter(|value| **value == 0.0).count(),
        rounds_at_or_above_3pct: paired
            .iter()
            .filter(|value| **value >= MIN_REVIEW_EFFECT)
            .count(),
        median_paired_relative_improvement: median(&paired),
        mean_paired_relative_improvement: mean,
        min_paired_relative_improvement: paired.iter().copied().fold(f64::INFINITY, f64::min),
        max_paired_relative_improvement: paired.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        sample_stdev_paired_relative_improvement: sample_stdev(&paired, mean),
        median_ab_relative_improvement: median(&ab),
        median_ba_relative_improvement: median(&ba),
        median_order_bias_delta: median(&ab) - median(&ba),
    }
}

fn require_independent_compatible_environment(
    left: &BenchmarkMetadata,
    right: &BenchmarkMetadata,
) -> Result<(), String> {
    let left_context = left
        .environment_fingerprint
        .run_context_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "first campaign is missing run_context_id".to_string())?;
    let right_context = right
        .environment_fingerprint
        .run_context_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "second campaign is missing run_context_id".to_string())?;
    require(
        left_context != right_context,
        "independent campaign gate requires distinct run_context_id values",
    )?;
    require(
        left.git_commit == right.git_commit,
        "independent campaigns were not executed from the same git commit",
    )?;
    require(
        left.git_dirty == Some(false) && right.git_dirty == Some(false),
        "independent campaigns require clean git checkouts",
    )?;
    require(
        left.nnis_version == right.nnis_version,
        "independent campaigns use different NNIS versions",
    )?;

    let mut normalized_right = right.clone();
    normalized_right.environment_fingerprint.run_context_id =
        left.environment_fingerprint.run_context_id.clone();
    left.require_compatible_environment(&normalized_right)
        .map_err(|error| format!("independent campaign environment mismatch: {error}"))
}

fn review_eligible(summary: &MetricSummary) -> bool {
    summary.candidate_round_wins == EXPECTED_ROUNDS
        && summary.reference_round_wins == 0
        && summary.ties == 0
        && summary.rounds_at_or_above_3pct == EXPECTED_ROUNDS
        && summary.median_paired_relative_improvement >= MIN_REVIEW_EFFECT
        && summary.median_ab_relative_improvement > 0.0
        && summary.median_ba_relative_improvement > 0.0
}

fn run(first_path: &Path, second_path: &Path) -> Result<ConsensusReport, String> {
    let (first, first_metadata, first_execution_plan) =
        validate_campaign(read_bundle(first_path)?)?;
    let (second, second_metadata, second_execution_plan) =
        validate_campaign(read_bundle(second_path)?)?;

    require(
        first.git_commit == second.git_commit,
        "bundle git commits differ",
    )?;
    require(
        first_execution_plan == second_execution_plan,
        "execution plan differs across independent campaigns",
    )?;
    require_independent_compatible_environment(&first_metadata, &second_metadata)?;

    let promotion_review_eligible = review_eligible(&first.generation_stage_gpu)
        && review_eligible(&second.generation_stage_gpu);

    Ok(ConsensusReport {
        schema_version: 1,
        kind: "nnis_smollm2_f16_parallel_score_abba_consensus_v1",
        environment_compatible_across_distinct_campaigns: true,
        exact_git_commit_equal: true,
        campaign_count: 2,
        minimum_review_effect: MIN_REVIEW_EFFECT,
        first,
        second,
        promotion_review_eligible,
        claim_boundary: "promotion-review evidence for the exact SmolLM2 decode32 trajectory and compatible physical environment only; this verifier does not change any NNIS runtime default",
    })
}

fn main() {
    let mut arguments = env::args().skip(1);
    let first = arguments.next();
    let second = arguments.next();
    if first.is_none() || second.is_none() || arguments.next().is_some() {
        eprintln!(
            "usage: smollm2_f16_attention_abba_consensus FIRST_BUNDLE.json SECOND_BUNDLE.json"
        );
        std::process::exit(2);
    }

    let first = first.expect("checked above");
    let second = second.expect("checked above");
    match run(Path::new(&first), Path::new(&second)) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("serialize consensus report: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("F16 attention ABBA consensus rejected: {error}");
            eprintln!("{}", json!({"promotion_review_eligible": false}));
            std::process::exit(1);
        }
    }
}
