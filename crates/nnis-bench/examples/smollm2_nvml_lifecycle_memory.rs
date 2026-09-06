use nnis_bench::BenchmarkMetadata;
use nnis_model::{
    load_model_directory, F16ReferenceExecutionPlan, F16ReferenceModel,
    F16WeightMaterializationMemoryEvidenceV1, ModelConfig, WeightAllocationSummaryV1,
};
use nnis_rt::{
    current_process_gpu_memory, Context, Device, NnisError, NvmlProcessMemorySnapshotV1, Result,
    Stream,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

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
struct ProcessMemoryObservationV1 {
    schema_version: u32,
    pid: u32,
    device_ordinal: i32,
    device_uuid: String,
    used_gpu_memory_bytes: u64,
}

impl From<NvmlProcessMemorySnapshotV1> for ProcessMemoryObservationV1 {
    fn from(snapshot: NvmlProcessMemorySnapshotV1) -> Self {
        Self {
            schema_version: snapshot.schema_version,
            pid: snapshot.pid,
            device_ordinal: snapshot.device_ordinal,
            device_uuid: snapshot.device_uuid,
            used_gpu_memory_bytes: snapshot.used_gpu_memory_bytes,
        }
    }
}

#[derive(Debug, Serialize)]
struct WeightLifecycleObservationV1 {
    stage: &'static str,
    nvml_process_memory: ProcessMemoryObservationV1,
    exact_owned_weight_allocations: WeightAllocationSummaryV1,
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
    source_graph_execution_weight_dtype: String,
    metadata: BenchmarkMetadata,
    model_config: ModelConfig,
    execution_plan: F16ReferenceExecutionPlan,
    post_context_nvml_process_memory: ProcessMemoryObservationV1,
    source_f32_weights: WeightLifecycleObservationV1,
    resident_f16_weights: WeightLifecycleObservationV1,
    materialization_memory: F16WeightMaterializationMemoryEvidenceV1,
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
                    "usage: smollm2_nvml_lifecycle_memory --model DIR [--device N]".to_string(),
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
            "model provenance is not the pinned widened-f32 SmolLM2-135M fixture",
        ));
    }
    Ok(provenance)
}

fn validate_model_shape(config: &ModelConfig) -> Result<()> {
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

fn observe_process_memory(device: &Device) -> Result<ProcessMemoryObservationV1> {
    current_process_gpu_memory(device)
        .map(ProcessMemoryObservationV1::from)
        .map_err(|error| {
            NnisError::unsupported(format!(
                "NVML process-scoped memory observation unavailable: {error}"
            ))
        })
}

fn run(arguments: Arguments) -> Result<Report> {
    let provenance = read_provenance(&arguments.model_dir)?;
    let device = Device::get(arguments.device)?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let metadata = BenchmarkMetadata::collect(&context);

    stream.synchronize()?;
    let post_context_nvml_process_memory = observe_process_memory(&device)?;

    let (model_config, source_weights) =
        load_model_directory(&context, &stream, &arguments.model_dir)?;
    validate_model_shape(&model_config)?;
    stream.synchronize()?;
    let source_weight_allocations = source_weights.weight_allocation_summary_v1()?;
    let source_f32_weights = WeightLifecycleObservationV1 {
        stage: "source_f32_weights_loaded",
        nvml_process_memory: observe_process_memory(&device)?,
        exact_owned_weight_allocations: source_weight_allocations,
    };

    let execution_plan =
        F16ReferenceExecutionPlan::smollm2_135m_thor_min_latency(&model_config, &context)?;
    let report_config = model_config.clone();
    let model = F16ReferenceModel::new_with_execution_plan(
        model_config,
        source_weights,
        &stream,
        execution_plan,
    )?;
    stream.synchronize()?;

    let resident_f16_weights = WeightLifecycleObservationV1 {
        stage: "resident_f16_weights_materialized_source_f32_dropped",
        nvml_process_memory: observe_process_memory(&device)?,
        exact_owned_weight_allocations: model.weight_allocation_summary_v1()?,
    };
    let materialization_memory = model.weight_materialization_memory_evidence_v1().clone();

    Ok(Report {
        schema_version: 1,
        evidence: "nnis.smollm2-nvml-lifecycle-memory",
        measurement: "nvml_process_used_gpu_memory_plus_exact_owned_weight_allocations_not_physical_page_residency",
        source_repo: SOURCE_REPO,
        source_revision: SOURCE_REVISION,
        source_model_sha256: SOURCE_MODEL_SHA256,
        source_weight_dtype: provenance.source_weight_dtype,
        source_graph_execution_weight_dtype: provenance.execution_weight_dtype,
        metadata,
        model_config: report_config,
        execution_plan,
        post_context_nvml_process_memory,
        source_f32_weights,
        resident_f16_weights,
        materialization_memory,
        limitations: vec![
            "NVML usedGpuMemory is a process-scoped observation and is not physical GPU page residency",
            "exact_owned_weight_allocations counts NNIS DeviceBuffer allocations owned by the corresponding weight graph and is a different measurement from NVML process memory",
            "no subtraction in this report is defined as CUDA allocator, page-table, context, module, JIT, workspace, KV, session, or RoPE overhead",
            "post-context observation includes whatever GPU memory NVML attributes to the process after NNIS context and stream creation",
            "source and F16 observations are captured only after stream synchronization at explicit lifecycle boundaries",
            "the physical campaign must run with the pinned SmolLM2 fixture and declared Thor qualification environment",
            "does not provide latency, throughput, quality, compression ratio, effective bits-per-parameter, or live-transition evidence",
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
