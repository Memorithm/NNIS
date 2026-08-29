use nnis_model::{GenerationConfig, Model};
use nnis_rt::{Context, Device, NnisError, Result, Stream};
use serde::Deserialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct ReferenceManifest {
    format: String,
    version: u32,
    source_repo: String,
    source_revision: String,
    source_model_sha256: String,
    prompt: String,
    input_ids: Vec<u32>,
    decode_steps: usize,
    dtype: String,
    logit_files: Vec<String>,
    greedy_ids: Vec<u32>,
}

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
    reference_dir: PathBuf,
    atol: f32,
    rtol: f32,
}

#[derive(Debug)]
struct ErrorMetrics {
    max_abs: f32,
    max_rel: f32,
    rms: f64,
    worst_index: usize,
    failures: usize,
}

fn parse_arguments() -> std::result::Result<Arguments, String> {
    let mut args = env::args().skip(1);
    let mut model_dir = None;
    let mut reference_dir = None;
    let mut atol = 1.0e-4_f32;
    let mut rtol = 1.0e-3_f32;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => {
                model_dir = Some(PathBuf::from(
                    args.next().ok_or("--model requires a directory")?,
                ));
            }
            "--reference" => {
                reference_dir = Some(PathBuf::from(
                    args.next().ok_or("--reference requires a directory")?,
                ));
            }
            "--atol" => {
                atol = args
                    .next()
                    .ok_or("--atol requires a value")?
                    .parse()
                    .map_err(|error| format!("invalid --atol: {error}"))?;
            }
            "--rtol" => {
                rtol = args
                    .next()
                    .ok_or("--rtol requires a value")?
                    .parse()
                    .map_err(|error| format!("invalid --rtol: {error}"))?;
            }
            "--help" | "-h" => {
                return Err(
                    "usage: compare_tiny_llama --model DIR --reference DIR [--atol F32] [--rtol F32]"
                        .to_string(),
                );
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if !atol.is_finite() || atol < 0.0 || !rtol.is_finite() || rtol < 0.0 {
        return Err("--atol and --rtol must be finite and non-negative".to_string());
    }
    Ok(Arguments {
        model_dir: model_dir.ok_or("missing --model DIR")?,
        reference_dir: reference_dir.ok_or("missing --reference DIR")?,
        atol,
        rtol,
    })
}

fn read_reference_manifest(directory: &Path) -> Result<ReferenceManifest> {
    let bytes = fs::read(directory.join("reference.json"))
        .map_err(|error| NnisError::io("read reference manifest", error))?;
    let manifest: ReferenceManifest = serde_json::from_slice(&bytes).map_err(|error| {
        NnisError::invalid_input(format!("invalid reference manifest JSON: {error}"))
    })?;
    if manifest.format != "nnis-reference-logits" || manifest.version != 1 {
        return Err(NnisError::unsupported(format!(
            "unsupported reference format {:?} version {}",
            manifest.format, manifest.version
        )));
    }
    if manifest.dtype != "f32" {
        return Err(NnisError::unsupported(format!(
            "reference dtype {:?} is not f32",
            manifest.dtype
        )));
    }
    if manifest.logit_files.len() != manifest.decode_steps + 1 {
        return Err(NnisError::invalid_input(format!(
            "reference has {} logit files for {} decode steps; expected {}",
            manifest.logit_files.len(),
            manifest.decode_steps,
            manifest.decode_steps + 1
        )));
    }
    if manifest.greedy_ids.len() != manifest.decode_steps {
        return Err(NnisError::invalid_input(format!(
            "reference has {} greedy IDs for {} decode steps",
            manifest.greedy_ids.len(),
            manifest.decode_steps
        )));
    }
    if manifest.input_ids.is_empty() {
        return Err(NnisError::invalid_input("reference input_ids are empty"));
    }
    Ok(manifest)
}

fn read_f32_le(path: &Path, expected_len: usize) -> Result<Vec<f32>> {
    let bytes = fs::read(path).map_err(|error| NnisError::io("read reference logits", error))?;
    let expected_bytes = expected_len
        .checked_mul(4)
        .ok_or_else(|| NnisError::invalid_input("reference logit byte length overflows usize"))?;
    if bytes.len() != expected_bytes {
        return Err(NnisError::invalid_input(format!(
            "reference logits {} contain {} bytes; expected {expected_bytes}",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn compare(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) -> ErrorMetrics {
    assert_eq!(actual.len(), expected.len());
    let mut metrics = ErrorMetrics {
        max_abs: 0.0,
        max_rel: 0.0,
        rms: 0.0,
        worst_index: 0,
        failures: 0,
    };
    let mut squared_sum = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let absolute = (actual - expected).abs();
        let relative = if expected == 0.0 {
            if absolute == 0.0 { 0.0 } else { f32::INFINITY }
        } else {
            absolute / expected.abs()
        };
        if absolute > metrics.max_abs {
            metrics.max_abs = absolute;
            metrics.worst_index = index;
        }
        metrics.max_rel = metrics.max_rel.max(relative);
        squared_sum += f64::from(absolute) * f64::from(absolute);
        let tolerance = atol + rtol * expected.abs();
        if !actual.is_finite() || absolute > tolerance {
            metrics.failures += 1;
        }
    }
    metrics.rms = (squared_sum / actual.len() as f64).sqrt();
    metrics
}

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    best_index
}

fn report_and_require(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    atol: f32,
    rtol: f32,
) -> Result<()> {
    let metrics = compare(actual, expected, atol, rtol);
    println!(
        "{label}: max_abs={:.8e} max_rel={:.8e} rms={:.8e} worst_index={} failures={}",
        metrics.max_abs, metrics.max_rel, metrics.rms, metrics.worst_index, metrics.failures
    );
    if metrics.failures != 0 {
        return Err(NnisError::invalid_input(format!(
            "{label} differs from trusted reference at {} logits (atol={atol}, rtol={rtol})",
            metrics.failures
        )));
    }
    Ok(())
}

fn run(arguments: Arguments) -> Result<()> {
    let reference = read_reference_manifest(&arguments.reference_dir)?;
    let device = Device::first()?;
    let context = Context::new(&device)?;
    let construction_stream = Stream::new(&context)?;
    let model = Model::load_directory(&context, &construction_stream, &arguments.model_dir)?;
    let vocab = model.config().vocab_size;
    let mut session = model.new_session()?;

    println!("source={}@{}", reference.source_repo, reference.source_revision);
    println!("source_model_sha256={}", reference.source_model_sha256);
    println!("prompt={:?}", reference.prompt);
    println!("input_ids={:?}", reference.input_ids);
    println!("atol={} rtol={}", arguments.atol, arguments.rtol);

    let prefill = session.prefill(&reference.input_ids)?;
    let expected_prefill = read_f32_le(
        &arguments.reference_dir.join(&reference.logit_files[0]),
        vocab,
    )?;
    report_and_require(
        "prefill",
        &prefill,
        &expected_prefill,
        arguments.atol,
        arguments.rtol,
    )?;

    for step in 0..reference.decode_steps {
        let expected_greedy = reference.greedy_ids[step];
        let actual_greedy = if step == 0 {
            argmax(&prefill) as u32
        } else {
            // The previous decode comparison has already verified the logits
            // from which this reference token was chosen. Use the trusted token
            // for the next step so one mismatch does not cascade into a new
            // sequence and obscure the first failing layer/position.
            expected_greedy
        };
        if actual_greedy != expected_greedy {
            return Err(NnisError::invalid_input(format!(
                "greedy token mismatch at step {step}: NNIS {actual_greedy}, reference {expected_greedy}"
            )));
        }
        let actual = session.decode_one(expected_greedy)?;
        let expected = read_f32_le(
            &arguments.reference_dir.join(&reference.logit_files[step + 1]),
            vocab,
        )?;
        report_and_require(
            &format!("decode[{step}]"),
            &actual,
            &expected,
            arguments.atol,
            arguments.rtol,
        )?;
    }

    let generated = model
        .new_session()?
        .generate(
            &reference.input_ids,
            GenerationConfig::greedy(reference.decode_steps),
        )?;
    if generated != reference.greedy_ids {
        return Err(NnisError::invalid_input(format!(
            "greedy sequence mismatch: NNIS {generated:?}, reference {:?}",
            reference.greedy_ids
        )));
    }
    println!("greedy_ids={generated:?}");
    println!("reference comparison passed");
    Ok(())
}

fn main() {
    let arguments = match parse_arguments() {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(arguments) {
        eprintln!("reference comparison failed: {error}");
        std::process::exit(1);
    }
}
