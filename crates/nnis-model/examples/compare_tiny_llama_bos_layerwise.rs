use nnis_jit::JitCompiler;
use nnis_kernels::{F32Elementwise, F32Gather, F32Gemm};
use nnis_model::{load_model_directory, F32DecoderKernels, Model};
use nnis_rt::{Context, Device, DeviceBuffer, Stream};
use serde::Deserialize;
use std::env;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};

const SOURCE_REPO: &str = "amakhov/tiny-random-llama";
const SOURCE_REVISION: &str = "99160cb087861a1e3c54ff5d3f45fd9488d9c04e";
const SOURCE_MODEL_SHA256: &str =
    "a4eb5dcdfc71d3a8f297bb1c2a672d3babe04f102480addde293210778805d30";
const TRACE_FORMAT: &str = "nnis-tiny-llama-bos-layerwise";
const TRACE_VERSION: u32 = 1;
const DEFAULT_ATOL: f32 = 1.0e-4;
const DEFAULT_RTOL: f32 = 1.0e-3;

type AnyResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
    reference_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct TraceStage {
    name: String,
    file: String,
    elements: usize,
}

#[derive(Debug, Deserialize)]
struct TraceManifest {
    format: String,
    version: u32,
    source_repo: String,
    source_revision: String,
    source_model_sha256: String,
    transformers_version: String,
    input_ids: Vec<u32>,
    stages: Vec<TraceStage>,
    full_forward_max_abs: f64,
    full_forward_rms: f64,
}

#[derive(Debug, Clone, Copy)]
struct ErrorMetrics {
    max_abs: f32,
    max_rel: f32,
    rms: f64,
    worst_index: usize,
    failures: usize,
}

fn invalid_data(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::InvalidData, message.into())
}

fn parse_arguments() -> AnyResult<Arguments> {
    let mut args = env::args().skip(1);
    let mut model_dir = None;
    let mut reference_dir = None;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--reference" => reference_dir = args.next().map(PathBuf::from),
            "--help" | "-h" => {
                return Err(invalid_data(
                    "usage: compare_tiny_llama_bos_layerwise --model DIR --reference DIR",
                )
                .into());
            }
            other => return Err(invalid_data(format!("unknown argument {other:?}")).into()),
        }
    }
    Ok(Arguments {
        model_dir: model_dir.ok_or_else(|| invalid_data("missing --model DIR"))?,
        reference_dir: reference_dir.ok_or_else(|| invalid_data("missing --reference DIR"))?,
    })
}

fn load_manifest(reference_dir: &Path) -> AnyResult<TraceManifest> {
    let bytes = fs::read(reference_dir.join("trace.json"))?;
    let manifest: TraceManifest = serde_json::from_slice(&bytes)?;
    if manifest.format != TRACE_FORMAT || manifest.version != TRACE_VERSION {
        return Err(invalid_data(format!(
            "unsupported trace {:?} version {}",
            manifest.format, manifest.version
        ))
        .into());
    }
    if manifest.source_repo != SOURCE_REPO
        || manifest.source_revision != SOURCE_REVISION
        || manifest.source_model_sha256 != SOURCE_MODEL_SHA256
    {
        return Err(invalid_data(format!(
            "trace provenance mismatch: {}@{} sha256={}",
            manifest.source_repo, manifest.source_revision, manifest.source_model_sha256
        ))
        .into());
    }
    if manifest.input_ids != [1] {
        return Err(invalid_data(format!(
            "layerwise diagnostic requires BOS-only input_ids [1], got {:?}",
            manifest.input_ids
        ))
        .into());
    }
    Ok(manifest)
}

fn read_f32_le(path: &Path, expected_elements: usize) -> AnyResult<Vec<f32>> {
    let bytes = fs::read(path)?;
    let expected_bytes = expected_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| invalid_data("reference tensor byte length overflow"))?;
    if bytes.len() != expected_bytes {
        return Err(invalid_data(format!(
            "{} has {} bytes; expected {expected_bytes}",
            path.display(),
            bytes.len()
        ))
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn expected_stage(
    manifest: &TraceManifest,
    reference_dir: &Path,
    name: &str,
) -> AnyResult<Vec<f32>> {
    let stage = manifest
        .stages
        .iter()
        .find(|stage| stage.name == name)
        .ok_or_else(|| invalid_data(format!("trace is missing stage {name:?}")))?;
    read_f32_le(&reference_dir.join(&stage.file), stage.elements)
}

fn compare(actual: &[f32], expected: &[f32]) -> AnyResult<ErrorMetrics> {
    if actual.len() != expected.len() {
        return Err(invalid_data(format!(
            "length mismatch: actual={} expected={}",
            actual.len(),
            expected.len()
        ))
        .into());
    }
    if actual.is_empty() {
        return Ok(ErrorMetrics {
            max_abs: 0.0,
            max_rel: 0.0,
            rms: 0.0,
            worst_index: 0,
            failures: 0,
        });
    }
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut worst_index = 0;
    let mut squared_sum = 0.0_f64;
    let mut failures = 0;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let absolute = (actual - expected).abs();
        let relative = if expected == 0.0 {
            if absolute == 0.0 {
                0.0
            } else {
                f32::INFINITY
            }
        } else {
            absolute / expected.abs()
        };
        if absolute > max_abs {
            max_abs = absolute;
            worst_index = index;
        }
        max_rel = max_rel.max(relative);
        squared_sum += f64::from(absolute) * f64::from(absolute);
        if !actual.is_finite() || absolute > DEFAULT_ATOL + DEFAULT_RTOL * expected.abs() {
            failures += 1;
        }
    }
    Ok(ErrorMetrics {
        max_abs,
        max_rel,
        rms: (squared_sum / actual.len() as f64).sqrt(),
        worst_index,
        failures,
    })
}

fn report_values(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    first_outside: &mut Option<String>,
) -> AnyResult<ErrorMetrics> {
    let metrics = compare(actual, expected)?;
    println!(
        "{label}: max_abs={:.8e} max_rel={:.8e} rms={:.8e} worst_index={} failures={}",
        metrics.max_abs,
        metrics.max_rel,
        metrics.rms,
        metrics.worst_index,
        metrics.failures
    );
    if metrics.failures != 0 && first_outside.is_none() {
        *first_outside = Some(label.to_string());
    }
    Ok(metrics)
}

fn report_buffer(
    label: &str,
    buffer: &DeviceBuffer<f32>,
    stream: &Stream,
    manifest: &TraceManifest,
    reference_dir: &Path,
    first_outside: &mut Option<String>,
) -> AnyResult<Vec<f32>> {
    let actual = buffer.to_vec(stream)?;
    let expected = expected_stage(manifest, reference_dir, label)?;
    report_values(label, &actual, &expected, first_outside)?;
    Ok(actual)
}

fn run(arguments: Arguments) -> AnyResult<()> {
    let manifest = load_manifest(&arguments.reference_dir)?;
    println!("source={}@{}", manifest.source_repo, manifest.source_revision);
    println!("source_model_sha256={}", manifest.source_model_sha256);
    println!("transformers={}", manifest.transformers_version);
    println!("input_ids={:?}", manifest.input_ids);
    println!(
        "transformers_manual_vs_full: max_abs={:.8e} rms={:.8e}",
        manifest.full_forward_max_abs, manifest.full_forward_rms
    );
    println!("diagnostic_default_atol={DEFAULT_ATOL} rtol={DEFAULT_RTOL}");

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let (config, weights) = load_model_directory(&context, &stream, &arguments.model_dir)?;
    if config.num_attention_heads != config.num_key_value_heads {
        return Err(invalid_data("BOS shortcut requires equal query and KV head counts").into());
    }

    let compiler = JitCompiler::new();
    let gather = F32Gather::load(&context, &compiler)?;
    let gemm = F32Gemm::load(&context, &compiler)?;
    let elementwise = F32Elementwise::load(&context, &compiler)?;
    let decoder = F32DecoderKernels::load(&context, &compiler)?;

    let token = DeviceBuffer::from_host(&context, &stream, &[1_u32])?;
    let mut hidden = DeviceBuffer::<f32>::new(&context, config.hidden_size)?;
    gather.gather(
        &stream,
        weights.token_embedding.tensor().as_f32()?,
        &token,
        &hidden,
        config.vocab_size,
        config.hidden_size,
    )?;

    let mut first_outside = None;
    report_buffer(
        "embedding",
        &hidden,
        &stream,
        &manifest,
        &arguments.reference_dir,
        &mut first_outside,
    )?;

    for (index, layer) in weights.layers.iter().enumerate() {
        let input_norm = DeviceBuffer::<f32>::new(&context, config.hidden_size)?;
        decoder.weighted_rms_norm(
            &stream,
            &hidden,
            layer.input_norm.tensor().as_f32()?,
            &input_norm,
            1,
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        report_buffer(
            &format!("layer{index}.input_norm"),
            &input_norm,
            &stream,
            &manifest,
            &arguments.reference_dir,
            &mut first_outside,
        )?;

        let value_width = layer.v_proj.cols();
        if value_width != config.hidden_size {
            return Err(invalid_data(format!(
                "BOS trace currently requires V width == hidden size; got {value_width} != {}",
                config.hidden_size
            ))
            .into());
        }
        let value = DeviceBuffer::<f32>::new(&context, value_width)?;
        gemm.gemm(
            &stream,
            &input_norm,
            layer.v_proj.tensor().as_f32()?,
            &value,
            1,
            value_width,
            config.hidden_size,
        )?;
        report_buffer(
            &format!("layer{index}.v_proj"),
            &value,
            &stream,
            &manifest,
            &arguments.reference_dir,
            &mut first_outside,
        )?;

        let projected = DeviceBuffer::<f32>::new(&context, config.hidden_size)?;
        gemm.gemm(
            &stream,
            &value,
            layer.o_proj.tensor().as_f32()?,
            &projected,
            1,
            config.hidden_size,
            value_width,
        )?;
        report_buffer(
            &format!("layer{index}.o_proj"),
            &projected,
            &stream,
            &manifest,
            &arguments.reference_dir,
            &mut first_outside,
        )?;

        let residual = DeviceBuffer::<f32>::new(&context, config.hidden_size)?;
        elementwise.vector_add(&stream, &hidden, &projected, &residual)?;
        report_buffer(
            &format!("layer{index}.residual"),
            &residual,
            &stream,
            &manifest,
            &arguments.reference_dir,
            &mut first_outside,
        )?;

        let post_norm = DeviceBuffer::<f32>::new(&context, config.hidden_size)?;
        decoder.weighted_rms_norm(
            &stream,
            &residual,
            layer.post_attention_norm.tensor().as_f32()?,
            &post_norm,
            1,
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        report_buffer(
            &format!("layer{index}.post_attention_norm"),
            &post_norm,
            &stream,
            &manifest,
            &arguments.reference_dir,
            &mut first_outside,
        )?;

        let gate = DeviceBuffer::<f32>::new(&context, config.intermediate_size)?;
        gemm.gemm(
            &stream,
            &post_norm,
            layer.gate_proj.tensor().as_f32()?,
            &gate,
            1,
            config.intermediate_size,
            config.hidden_size,
        )?;
        report_buffer(
            &format!("layer{index}.gate_proj"),
            &gate,
            &stream,
            &manifest,
            &arguments.reference_dir,
            &mut first_outside,
        )?;

        let up = DeviceBuffer::<f32>::new(&context, config.intermediate_size)?;
        gemm.gemm(
            &stream,
            &post_norm,
            layer.up_proj.tensor().as_f32()?,
            &up,
            1,
            config.intermediate_size,
            config.hidden_size,
        )?;
        report_buffer(
            &format!("layer{index}.up_proj"),
            &up,
            &stream,
            &manifest,
            &arguments.reference_dir,
            &mut first_outside,
        )?;

        let activated = DeviceBuffer::<f32>::new(&context, config.intermediate_size)?;
        elementwise.silu(&stream, &gate, &activated)?;
        report_buffer(
            &format!("layer{index}.silu"),
            &activated,
            &stream,
            &manifest,
            &arguments.reference_dir,
            &mut first_outside,
        )?;

        let gated = DeviceBuffer::<f32>::new(&context, config.intermediate_size)?;
        decoder.multiply(&stream, &activated, &up, &gated)?;
        report_buffer(
            &format!("layer{index}.gated"),
            &gated,
            &stream,
            &manifest,
            &arguments.reference_dir,
            &mut first_outside,
        )?;

        let mlp = DeviceBuffer::<f32>::new(&context, config.hidden_size)?;
        gemm.gemm(
            &stream,
            &gated,
            layer.down_proj.tensor().as_f32()?,
            &mlp,
            1,
            config.hidden_size,
            config.intermediate_size,
        )?;
        report_buffer(
            &format!("layer{index}.down_proj"),
            &mlp,
            &stream,
            &manifest,
            &arguments.reference_dir,
            &mut first_outside,
        )?;

        let next_hidden = DeviceBuffer::<f32>::new(&context, config.hidden_size)?;
        elementwise.vector_add(&stream, &residual, &mlp, &next_hidden)?;
        report_buffer(
            &format!("layer{index}.hidden"),
            &next_hidden,
            &stream,
            &manifest,
            &arguments.reference_dir,
            &mut first_outside,
        )?;
        hidden = next_hidden;
    }

    let final_norm = DeviceBuffer::<f32>::new(&context, config.hidden_size)?;
    decoder.weighted_rms_norm(
        &stream,
        &hidden,
        weights.final_norm.tensor().as_f32()?,
        &final_norm,
        1,
        config.hidden_size,
        config.rms_norm_eps,
    )?;
    report_buffer(
        "final_norm",
        &final_norm,
        &stream,
        &manifest,
        &arguments.reference_dir,
        &mut first_outside,
    )?;

    let logits = DeviceBuffer::<f32>::new(&context, config.vocab_size)?;
    gemm.gemm(
        &stream,
        &final_norm,
        weights.lm_head.tensor().as_f32()?,
        &logits,
        1,
        config.vocab_size,
        config.hidden_size,
    )?;
    let manual_logits = report_buffer(
        "logits",
        &logits,
        &stream,
        &manifest,
        &arguments.reference_dir,
        &mut first_outside,
    )?;

    let runtime_model = Model::load_directory(&context, &stream, &arguments.model_dir)?;
    let mut runtime_session = runtime_model.new_session()?;
    let runtime_logits = runtime_session.prefill(&[1])?;
    let reference_logits = expected_stage(&manifest, &arguments.reference_dir, "logits")?;
    let mut runtime_first = None;
    report_values(
        "runtime_vs_transformers.logits",
        &runtime_logits,
        &reference_logits,
        &mut runtime_first,
    )?;
    let mut manual_first = None;
    report_values(
        "runtime_vs_manual_nnis.logits",
        &runtime_logits,
        &manual_logits,
        &mut manual_first,
    )?;

    println!(
        "first_stage_outside_default_tolerance={}",
        first_outside.as_deref().unwrap_or("none")
    );
    println!(
        "runtime_vs_manual_outside_default_tolerance={}",
        manual_first.as_deref().unwrap_or("none")
    );
    println!("layerwise diagnostic completed");
    Ok(())
}

fn main() {
    let result = parse_arguments().and_then(run);
    if let Err(error) = result {
        eprintln!("layerwise diagnostic failed: {error}");
        std::process::exit(1);
    }
}
