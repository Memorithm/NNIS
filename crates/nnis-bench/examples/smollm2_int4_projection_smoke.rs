use nnis_bench::BenchmarkMetadata;
use nnis_jit::JitCompiler;
use nnis_kernels::F32Int4Gemv;
use nnis_model::{
    dequantize_int4_symmetric_reference_v1, load_model_directory,
    quantize_int4_symmetric_reference_v1, Int4ReferenceModelStorageV1,
    Int4ReferenceProjectionPlanV1, Int4ReferenceStorageSummaryV1, ModelConfig,
};
use nnis_rt::{Context, Device, DeviceBuffer, NnisError, Result, Stream};
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
struct ProjectionEvidence {
    logical_weight: String,
    rows: usize,
    cols: usize,
    input_values: usize,
    output_values: usize,
    bitwise_equal_to_quantized_cpu_oracle: bool,
    max_absolute_output_error: f32,
    dense_weight_materialization: bool,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    evidence: &'static str,
    source_repo: &'static str,
    source_revision: &'static str,
    source_model_sha256: &'static str,
    source_weight_dtype: String,
    source_execution_weight_dtype: String,
    full_model_int4_execution_qualified: bool,
    metadata: BenchmarkMetadata,
    model_config: ModelConfig,
    storage: Int4ReferenceStorageSummaryV1,
    projection_plan: Int4ReferenceProjectionPlanV1,
    projection: ProjectionEvidence,
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
                    "usage: smollm2_int4_projection_smoke --model DIR [--device N]".to_string(),
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
        || provenance.source_weight_dtype != "bfloat16"
        || provenance.execution_weight_dtype != "f32"
    {
        return Err(NnisError::invalid_input(
            "model provenance is not the pinned SmolLM2 Stage-B source",
        ));
    }
    Ok(provenance)
}

fn ordered_projection(input: &[f32], weight: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let expected_weight = rows
        .checked_mul(cols)
        .ok_or_else(|| NnisError::invalid_input("INT4 projection oracle shape overflows usize"))?;
    if input.len() != rows || weight.len() != expected_weight {
        return Err(NnisError::invalid_input(
            "INT4 projection oracle input/weight shape mismatch",
        ));
    }
    Ok((0..cols)
        .map(|col| {
            (0..rows).fold(0.0_f32, |value, row| {
                input[row].mul_add(weight[row * cols + col], value)
            })
        })
        .collect())
}

fn run(arguments: Arguments) -> Result<Report> {
    let provenance = read_provenance(&arguments.model_dir)?;
    let device = Device::get(arguments.device)?;
    let context = Arc::new(Context::new(&device)?);
    let stream = Stream::new(&context)?;
    let metadata = BenchmarkMetadata::collect(&context);
    let (model_config, weights) = load_model_directory(&context, &stream, &arguments.model_dir)?;

    let layer = weights
        .layers
        .first()
        .ok_or_else(|| NnisError::invalid_input("SmolLM2 model has no decoder layer 0"))?;
    let rows = layer.q_proj.rows();
    let cols = layer.q_proj.cols();
    if rows != model_config.hidden_size || cols != model_config.hidden_size {
        return Err(NnisError::invalid_input(
            "SmolLM2 layer-0 q_proj does not match hidden-size square projection contract",
        ));
    }

    let source_weight = layer.q_proj.tensor().as_f32()?.to_vec(&stream)?;
    let quantized = quantize_int4_symmetric_reference_v1(&source_weight)?;
    let reconstructed = dequantize_int4_symmetric_reference_v1(&quantized)?;

    let input_host = (0..rows)
        .map(|index| ((index * 29 % 97) as f32 - 48.0) * 0.015625)
        .collect::<Vec<_>>();
    let expected = ordered_projection(&input_host, &reconstructed, rows, cols)?;

    let storage = Int4ReferenceModelStorageV1::from_f32_model_weights(&weights, &stream)?;
    let storage_summary = storage.summary().clone();
    if storage_summary.execution_qualified {
        return Err(NnisError::invalid_input(
            "reference INT4 storage unexpectedly claims full-model execution qualification",
        ));
    }

    let compiler = JitCompiler::new();
    let kernel = F32Int4Gemv::load(&context, &compiler)?;
    let input = DeviceBuffer::from_host(&context, &stream, &input_host)?;
    let output = DeviceBuffer::<f32>::new(&context, cols)?;
    let projection_plan = Int4ReferenceProjectionPlanV1::for_matrix("layers.0.q_proj", rows, cols)?;
    storage.execute_projection(&projection_plan, &kernel, &stream, &input, &output)?;
    let actual = output.to_vec(&stream)?;

    if actual.len() != expected.len() {
        return Err(NnisError::invalid_input(
            "INT4 projection output length disagrees with CPU oracle",
        ));
    }
    let mut bitwise_equal = true;
    let mut max_absolute_output_error = 0.0_f32;
    for (&actual, &expected) in actual.iter().zip(&expected) {
        bitwise_equal &= actual.to_bits() == expected.to_bits();
        max_absolute_output_error = max_absolute_output_error.max((actual - expected).abs());
    }
    if !bitwise_equal {
        return Err(NnisError::invalid_input(format!(
            "INT4 projection does not bit-match the ordered quantized CPU oracle; max_abs_error={max_absolute_output_error}"
        )));
    }

    Ok(Report {
        schema_version: 1,
        evidence: "nnis.int4-reference-projection-smoke",
        source_repo: SOURCE_REPO,
        source_revision: SOURCE_REVISION,
        source_model_sha256: SOURCE_MODEL_SHA256,
        source_weight_dtype: provenance.source_weight_dtype,
        source_execution_weight_dtype: provenance.execution_weight_dtype,
        full_model_int4_execution_qualified: false,
        metadata,
        model_config,
        storage: storage_summary,
        projection_plan,
        projection: ProjectionEvidence {
            logical_weight: "layers.0.q_proj".to_string(),
            rows,
            cols,
            input_values: input_host.len(),
            output_values: actual.len(),
            bitwise_equal_to_quantized_cpu_oracle: bitwise_equal,
            max_absolute_output_error,
            dense_weight_materialization: false,
        },
        limitations: vec![
            "qualifies only one isolated layer-0 q_proj projection",
            "compares against the quantized-weight oracle rather than the original dense model",
            "does not establish next-token NLL, perplexity, generation parity or model quality",
            "does not establish full-model INT4 execution, latency, throughput or conversion peak",
            "does not authorize ElasticBitAllocation search or final-test access",
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
