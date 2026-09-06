use nnis_bench::BenchmarkMetadata;
use nnis_model::{load_model_directory, ModelConfig, WeightAllocationSummaryV1};
use nnis_rt::{Context, Device, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SOURCE_REPO: &str = "HuggingFaceTB/SmolLM2-135M";
const SOURCE_REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
const SOURCE_MODEL_SHA256: &str =
    "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1";

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
    device: i32,
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
struct Report {
    schema_version: u32,
    evidence: &'static str,
    measurement: &'static str,
    source_repo: &'static str,
    source_revision: &'static str,
    source_model_sha256: &'static str,
    source_weight_dtype: String,
    execution_weight_dtype: String,
    metadata: BenchmarkMetadata,
    model_config: ModelConfig,
    weight_allocations: WeightAllocationSummaryV1,
    limitations: Vec<&'static str>,
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
                    "usage: smollm2_weight_allocation_accounting --model DIR [--device N]"
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
    })
}

fn read_provenance(model_dir: &Path) -> Result<Provenance> {
    let bytes = fs::read(model_dir.join("provenance.json"))
        .map_err(|error| NnisError::io("read SmolLM2 provenance", error))?;
    let provenance: Provenance = serde_json::from_slice(&bytes).map_err(|error| {
        NnisError::invalid_input(format!("invalid SmolLM2 provenance JSON: {error}"))
    })?;
    if provenance.source_repo != SOURCE_REPO
        || provenance.source_revision != SOURCE_REVISION
        || provenance.source_model_sha256 != SOURCE_MODEL_SHA256
    {
        return Err(NnisError::invalid_input(
            "model provenance is not the pinned SmolLM2-135M checkpoint",
        ));
    }
    Ok(provenance)
}

fn run(arguments: Arguments) -> Result<Report> {
    let provenance = read_provenance(&arguments.model_dir)?;
    let device = Device::get(arguments.device)?;
    let context = Arc::new(Context::new(&device)?);
    let stream = Stream::new(&context)?;
    let metadata = BenchmarkMetadata::collect(&context);
    let (model_config, weights) = load_model_directory(&context, &stream, &arguments.model_dir)?;
    let weight_allocations = weights.weight_allocation_summary_v1()?;

    Ok(Report {
        schema_version: 1,
        evidence: "nnis.weight-allocation-accounting",
        measurement: "exact_owned_cuMemAlloc_bytes_not_physical_page_residency",
        source_repo: SOURCE_REPO,
        source_revision: SOURCE_REVISION,
        source_model_sha256: SOURCE_MODEL_SHA256,
        source_weight_dtype: provenance.source_weight_dtype,
        execution_weight_dtype: provenance.execution_weight_dtype,
        metadata,
        model_config,
        weight_allocations,
        limitations: vec![
            "counts only live DeviceBuffer allocations owned by the loaded model weight graph",
            "does not measure CUDA allocator metadata or physical page residency",
            "does not include KV state, sessions, workspaces, RoPE caches, kernels, modules, or process-wide VRAM",
            "does not include peak temporary conversion or materialization memory",
            "does not provide a latency, throughput, quality, or compression claim",
        ],
    })
}

fn main() {
    let result = parse_arguments()
        .map_err(NnisError::invalid_input)
        .and_then(run)
        .and_then(|report| {
            serde_json::to_string_pretty(&report)
                .map_err(|error| NnisError::invalid_input(format!("serialize report: {error}")))
        });
    match result {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}
