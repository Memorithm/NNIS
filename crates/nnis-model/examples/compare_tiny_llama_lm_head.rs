use nnis_jit::JitCompiler;
use nnis_kernels::F32Gemm;
use nnis_model::load_model_directory;
use nnis_rt::{Context, Device, DeviceBuffer, Stream};
use serde::Deserialize;
use std::env;
use std::error::Error;
use std::fs;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};

const TRACE_FORMAT: &str = "nnis-tiny-llama-bos-layerwise";
const TRACE_VERSION: u32 = 1;

type AnyResult<T> = Result<T, Box<dyn Error>>;

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
    stages: Vec<TraceStage>,
}

fn invalid_data(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::InvalidData, message.into())
}

fn read_f32_le(path: &Path, expected: usize) -> AnyResult<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.len() != expected * 4 {
        return Err(invalid_data(format!(
            "{} has {} bytes; expected {}",
            path.display(),
            bytes.len(),
            expected * 4
        ))
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn stage(manifest: &TraceManifest, root: &Path, name: &str) -> AnyResult<Vec<f32>> {
    let entry = manifest
        .stages
        .iter()
        .find(|entry| entry.name == name)
        .ok_or_else(|| invalid_data(format!("missing stage {name}")))?;
    read_f32_le(&root.join(&entry.file), entry.elements)
}

fn metrics(label: &str, actual: &[f32], expected: &[f32]) -> AnyResult<()> {
    if actual.len() != expected.len() {
        return Err(invalid_data(format!(
            "{label}: length mismatch {} != {}",
            actual.len(),
            expected.len()
        ))
        .into());
    }
    let mut max_abs = 0.0_f32;
    let mut rms_acc = 0.0_f64;
    let mut worst = 0usize;
    let mut bit_mismatches = 0usize;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let d = (a - e).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        rms_acc += f64::from(d) * f64::from(d);
        if a.to_bits() != e.to_bits() {
            bit_mismatches += 1;
        }
    }
    let rms = (rms_acc / actual.len() as f64).sqrt();
    println!(
        "{label}: max_abs={max_abs:.8e} rms={rms:.8e} worst_index={worst} bit_mismatches={bit_mismatches}"
    );
    Ok(())
}

fn ordered_fma(input: &[f32], weights: &[f32], outputs: usize) -> Vec<f32> {
    let hidden = input.len();
    assert_eq!(weights.len(), hidden * outputs);
    let mut result = vec![0.0_f32; outputs];
    for out in 0..outputs {
        let mut acc = 0.0_f32;
        for k in 0..hidden {
            acc = input[k].mul_add(weights[k * outputs + out], acc);
        }
        result[out] = acc;
    }
    result
}

fn ordered_mul_add_separate(input: &[f32], weights: &[f32], outputs: usize) -> Vec<f32> {
    let hidden = input.len();
    assert_eq!(weights.len(), hidden * outputs);
    let mut result = vec![0.0_f32; outputs];
    for out in 0..outputs {
        let mut acc = 0.0_f32;
        for k in 0..hidden {
            acc += input[k] * weights[k * outputs + out];
        }
        result[out] = acc;
    }
    result
}

fn f64_reference(input: &[f32], weights: &[f32], outputs: usize) -> Vec<f64> {
    let hidden = input.len();
    let mut result = vec![0.0_f64; outputs];
    for out in 0..outputs {
        let mut acc = 0.0_f64;
        for k in 0..hidden {
            acc += f64::from(input[k]) * f64::from(weights[k * outputs + out]);
        }
        result[out] = acc;
    }
    result
}

fn rms_vs_f64(label: &str, values: &[f32], reference: &[f64]) {
    let mut max_abs = 0.0_f64;
    let mut rms_acc = 0.0_f64;
    let mut worst = 0usize;
    for (i, (&v, &r)) in values.iter().zip(reference).enumerate() {
        let d = (f64::from(v) - r).abs();
        if d > max_abs {
            max_abs = d;
            worst = i;
        }
        rms_acc += d * d;
    }
    println!(
        "{label}: max_abs_vs_f64={max_abs:.8e} rms_vs_f64={:.8e} worst_index={worst}",
        (rms_acc / values.len() as f64).sqrt()
    );
}

fn main() -> AnyResult<()> {
    let mut args = env::args().skip(1);
    let mut model_dir = None::<PathBuf>;
    let mut reference_dir = None::<PathBuf>;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_dir = args.next().map(PathBuf::from),
            "--reference" => reference_dir = args.next().map(PathBuf::from),
            other => return Err(invalid_data(format!("unknown argument {other}")) .into()),
        }
    }
    let model_dir = model_dir.ok_or_else(|| invalid_data("missing --model DIR"))?;
    let reference_dir = reference_dir.ok_or_else(|| invalid_data("missing --reference DIR"))?;

    let manifest: TraceManifest = serde_json::from_slice(&fs::read(reference_dir.join("trace.json"))?)?;
    if manifest.format != TRACE_FORMAT || manifest.version != TRACE_VERSION {
        return Err(invalid_data("unexpected trace format/version").into());
    }
    let tf_final_norm = stage(&manifest, &reference_dir, "final_norm")?;
    let tf_logits = stage(&manifest, &reference_dir, "logits")?;

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let (config, weights) = load_model_directory(&context, &stream, &model_dir)?;
    let lm_head = weights.lm_head.tensor().as_f32()?.to_vec(&stream)?;
    let outputs = config.vocab_size;

    let ordered = ordered_fma(&tf_final_norm, &lm_head, outputs);
    let separate = ordered_mul_add_separate(&tf_final_norm, &lm_head, outputs);
    let ref64 = f64_reference(&tf_final_norm, &lm_head, outputs);

    let compiler = JitCompiler::new();
    let gemm = F32Gemm::load(&context, &compiler)?;
    let input_gpu = DeviceBuffer::from_host(&context, &stream, &tf_final_norm)?;
    let logits_gpu = DeviceBuffer::<f32>::new(&context, outputs)?;
    gemm.gemm(
        &stream,
        &input_gpu,
        weights.lm_head.tensor().as_f32()?,
        &logits_gpu,
        1,
        outputs,
        config.hidden_size,
    )?;
    let gpu = logits_gpu.to_vec(&stream)?;

    println!("hidden={} vocab={outputs}", config.hidden_size);
    metrics("gpu_vs_ordered_fma", &gpu, &ordered)?;
    metrics("ordered_fma_vs_transformers", &ordered, &tf_logits)?;
    metrics("separate_mul_add_vs_transformers", &separate, &tf_logits)?;
    metrics("gpu_vs_transformers", &gpu, &tf_logits)?;
    rms_vs_f64("gpu", &gpu, &ref64);
    rms_vs_f64("ordered_fma", &ordered, &ref64);
    rms_vs_f64("separate_mul_add", &separate, &ref64);
    rms_vs_f64("transformers", &tf_logits, &ref64);
    Ok(())
}
