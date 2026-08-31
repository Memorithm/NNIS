use nnis_bench::{BenchmarkMetadata, TimingStatistics};
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, PartialEq)]
struct TimingReport {
    statistics: TimingStatistics,
}

#[derive(Debug, Deserialize)]
struct ComparableReport {
    schema_version: u32,
    benchmark: String,
    backend: String,
    measurement: String,
    source_repo: String,
    source_revision: String,
    source_model_sha256: String,
    execution_weight_dtype: String,
    input_ids: Vec<u32>,
    decode_steps: usize,
    warmup_iterations: usize,
    iterations: usize,
    metadata: BenchmarkMetadata,
    model: serde_json::Value,
    generation: TimingReport,
    request_total: TimingReport,
    generated_ids: Vec<u32>,
    #[serde(default)]
    representation_plan: Option<serde_json::Value>,
    #[serde(default)]
    fusion_plan: Option<serde_json::Value>,
    #[serde(default)]
    attention_plan: Option<serde_json::Value>,
}

fn read_report(path: &Path) -> Result<ComparableReport, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid report {}: {error}", path.display()))
}

fn require_same<T>(name: &str, left: &T, right: &T) -> Result<(), String>
where
    T: PartialEq + std::fmt::Debug,
{
    if left != right {
        return Err(format!(
            "reports are not workload-compatible at {name}: left={left:?}, right={right:?}"
        ));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let left_path = args
        .next()
        .ok_or("usage: compare_smollm2_reports LEFT.json RIGHT.json")?;
    let right_path = args
        .next()
        .ok_or("usage: compare_smollm2_reports LEFT.json RIGHT.json")?;
    if args.next().is_some() {
        return Err("usage: compare_smollm2_reports LEFT.json RIGHT.json".to_string());
    }
    let left = read_report(Path::new(&left_path))?;
    let right = read_report(Path::new(&right_path))?;

    require_same(
        "schema_version",
        &left.schema_version,
        &right.schema_version,
    )?;
    if left.schema_version != 2 {
        return Err(format!(
            "unsupported SmolLM2 report schema {}; expected 2",
            left.schema_version
        ));
    }
    require_same("benchmark", &left.benchmark, &right.benchmark)?;
    require_same("backend", &left.backend, &right.backend)?;
    require_same("measurement", &left.measurement, &right.measurement)?;
    require_same("source_repo", &left.source_repo, &right.source_repo)?;
    require_same(
        "source_revision",
        &left.source_revision,
        &right.source_revision,
    )?;
    require_same(
        "source_model_sha256",
        &left.source_model_sha256,
        &right.source_model_sha256,
    )?;
    require_same(
        "execution_weight_dtype",
        &left.execution_weight_dtype,
        &right.execution_weight_dtype,
    )?;
    require_same("input_ids", &left.input_ids, &right.input_ids)?;
    require_same("decode_steps", &left.decode_steps, &right.decode_steps)?;
    require_same(
        "warmup_iterations",
        &left.warmup_iterations,
        &right.warmup_iterations,
    )?;
    require_same("iterations", &left.iterations, &right.iterations)?;
    require_same("model", &left.model, &right.model)?;
    require_same("generated_ids", &left.generated_ids, &right.generated_ids)?;
    require_same(
        "representation_plan",
        &left.representation_plan,
        &right.representation_plan,
    )?;
    require_same("fusion_plan", &left.fusion_plan, &right.fusion_plan)?;
    require_same(
        "attention_plan",
        &left.attention_plan,
        &right.attention_plan,
    )?;
    left.metadata
        .require_compatible_environment(&right.metadata)
        .map_err(|error| error.to_string())?;

    let left_generation = left.generation.statistics.median_ms;
    let right_generation = right.generation.statistics.median_ms;
    let left_request = left.request_total.statistics.median_ms;
    let right_request = right.request_total.statistics.median_ms;
    if left_generation <= 0.0
        || right_generation <= 0.0
        || left_request <= 0.0
        || right_request <= 0.0
    {
        return Err("report medians must be positive".to_string());
    }

    let report = json!({
        "schema_version": 1,
        "comparable": true,
        "run_context_id": left.metadata.environment_fingerprint.run_context_id,
        "left_git_commit": left.metadata.git_commit,
        "right_git_commit": right.metadata.git_commit,
        "generation": {
            "left_median_ms": left_generation,
            "right_median_ms": right_generation,
            "right_over_left_latency_ratio": right_generation / left_generation,
            "right_over_left_throughput_ratio": left_generation / right_generation,
        },
        "request_total": {
            "left_median_ms": left_request,
            "right_median_ms": right_request,
            "right_over_left_latency_ratio": right_request / left_request,
            "right_over_left_throughput_ratio": left_request / right_request,
        },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize comparison: {error}"))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("SmolLM2 report comparison rejected: {error}");
        std::process::exit(1);
    }
}
