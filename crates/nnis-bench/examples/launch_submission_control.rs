use nnis_bench::{
    benchmark_gpu, summarize_samples_ms, BenchConfig, BenchmarkCase, BenchmarkMetadata,
    TimingStatistics, BENCH_RUN_CONTEXT_ENV,
};
use nnis_jit::JitCompiler;
use nnis_kernels::F32Elementwise;
use nnis_rt::{gpu_context, DeviceBuffer, NnisError, Result, Stream};
use serde::Serialize;
use std::env;
use std::time::Instant;

const DEFAULT_LAUNCH_COUNTS: &[usize] = &[1, 8, 32, 128, 211, 512];
const CONTROL_ELEMENTS: usize = 1;
const CONTROL_SCALE: f32 = 1.25;

#[derive(Debug)]
struct Arguments {
    config: BenchConfig,
    launch_counts: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct ControlResult {
    launch_count: usize,
    gpu_timeline: nnis_bench::BenchmarkReport,
    host_submission: TimingStatistics,
    host_submission_sequence_average_us_per_launch: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    experiment: &'static str,
    promotion_state: &'static str,
    run_context_id: String,
    metadata: BenchmarkMetadata,
    config: ControlConfig,
    results: Vec<ControlResult>,
    limitations: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct ControlConfig {
    kernel: &'static str,
    elements_per_launch: usize,
    scale: f32,
    launch_counts: Vec<usize>,
    warmups: usize,
    iterations: usize,
    smollm2_projection_launch_reference: usize,
}

fn parse_env_usize(name: &str, default: usize) -> Result<usize> {
    match env::var(name) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            NnisError::invalid_input(format!("invalid {name}={value:?}: {error}"))
        }),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(NnisError::invalid_input(format!(
            "failed reading {name}: {error}"
        ))),
    }
}

fn parse_launch_counts() -> Result<Vec<usize>> {
    let raw = match env::var("NNIS_LAUNCH_COUNTS") {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => {
            return Ok(DEFAULT_LAUNCH_COUNTS.to_vec());
        }
        Err(error) => {
            return Err(NnisError::invalid_input(format!(
                "failed reading NNIS_LAUNCH_COUNTS: {error}"
            )));
        }
    };
    let mut counts = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            return Err(NnisError::invalid_input(
                "NNIS_LAUNCH_COUNTS contains an empty entry",
            ));
        }
        let count = trimmed.parse::<usize>().map_err(|error| {
            NnisError::invalid_input(format!(
                "invalid NNIS_LAUNCH_COUNTS entry {trimmed:?}: {error}"
            ))
        })?;
        if count == 0 {
            return Err(NnisError::invalid_input(
                "NNIS_LAUNCH_COUNTS entries must be positive",
            ));
        }
        if counts.contains(&count) {
            return Err(NnisError::invalid_input(format!(
                "NNIS_LAUNCH_COUNTS contains duplicate value {count}"
            )));
        }
        counts.push(count);
    }
    if counts.is_empty() {
        return Err(NnisError::invalid_input(
            "NNIS_LAUNCH_COUNTS must contain at least one value",
        ));
    }
    Ok(counts)
}

fn parse_arguments() -> Result<Arguments> {
    let config = BenchConfig::new(
        parse_env_usize("NNIS_PROFILE_WARMUPS", 20)?,
        parse_env_usize("NNIS_PROFILE_ITERATIONS", 100)?,
    );
    config.validate()?;
    Ok(Arguments {
        config,
        launch_counts: parse_launch_counts()?,
    })
}

fn required_run_context() -> Result<String> {
    match env::var(BENCH_RUN_CONTEXT_ENV) {
        Ok(value) if !value.trim().is_empty() => Ok(value.trim().to_string()),
        Ok(_) | Err(env::VarError::NotPresent) => Err(NnisError::invalid_input(format!(
            "{BENCH_RUN_CONTEXT_ENV} is required for launch-submission evidence"
        ))),
        Err(error) => Err(NnisError::invalid_input(format!(
            "failed reading {BENCH_RUN_CONTEXT_ENV}: {error}"
        ))),
    }
}

fn enqueue_repeated_scale(
    kernels: &F32Elementwise,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    launches: usize,
) -> Result<()> {
    for _ in 0..launches {
        // SAFETY: input/output, kernels and stream remain alive until the
        // caller synchronizes the stream after each complete sequence.
        unsafe { kernels.enqueue_scale(stream, input, output, CONTROL_SCALE)? };
    }
    Ok(())
}

fn measure_host_submission<F>(
    stream: &Stream,
    config: BenchConfig,
    mut enqueue: F,
) -> Result<TimingStatistics>
where
    F: FnMut() -> Result<()>,
{
    stream.synchronize()?;
    for _ in 0..config.warmup_iterations {
        if let Err(error) = enqueue() {
            let _ = stream.synchronize();
            return Err(error);
        }
        stream.synchronize()?;
    }

    let mut samples_ms = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        stream.synchronize()?;
        let started = Instant::now();
        if let Err(error) = enqueue() {
            let _ = stream.synchronize();
            return Err(error);
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
            let _ = stream.synchronize();
            return Err(NnisError::unsupported(format!(
                "host clock produced invalid submission duration {elapsed_ms} ms"
            )));
        }
        stream.synchronize()?;
        samples_ms.push(elapsed_ms);
    }
    summarize_samples_ms(&samples_ms)
}

fn require_same_evidence_context(
    reference: &BenchmarkMetadata,
    candidate: &BenchmarkMetadata,
) -> Result<()> {
    reference.require_compatible_environment(candidate)?;
    if candidate.git_commit != reference.git_commit {
        return Err(NnisError::invalid_input(format!(
            "launch-control git commit drifted inside one process: {} != {}",
            candidate.git_commit, reference.git_commit
        )));
    }
    if candidate.git_dirty != Some(false) {
        return Err(NnisError::invalid_input(
            "launch-control evidence requires a clean tracked worktree",
        ));
    }
    Ok(())
}

fn run() -> Result<Report> {
    let arguments = parse_arguments()?;
    let run_context_id = required_run_context()?;
    let context = gpu_context().ok_or_else(|| NnisError::unsupported("no CUDA device"))?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let kernels = F32Elementwise::load(&context, &compiler)?;
    let input_host = [2.0_f32; CONTROL_ELEMENTS];
    let input = DeviceBuffer::from_host(&context, &stream, &input_host)?;
    let output = DeviceBuffer::<f32>::new(&context, CONTROL_ELEMENTS)?;
    stream.synchronize()?;

    let mut results = Vec::with_capacity(arguments.launch_counts.len());
    let mut reference_metadata: Option<BenchmarkMetadata> = None;

    for &launch_count in &arguments.launch_counts {
        let case = BenchmarkCase::new("cuda_launch_submission_control_scale_f32", "f32")
            .with_dimension("launches", launch_count as u64)
            .with_dimension("elements_per_launch", CONTROL_ELEMENTS as u64)
            .with_work_items(launch_count as u64);
        let gpu_timeline = benchmark_gpu(&context, &stream, case, arguments.config, || {
            enqueue_repeated_scale(&kernels, &stream, &input, &output, launch_count)
        })?;

        if gpu_timeline.metadata.git_dirty != Some(false) {
            return Err(NnisError::invalid_input(
                "launch-control evidence requires a clean tracked worktree",
            ));
        }
        match &reference_metadata {
            Some(reference) => require_same_evidence_context(reference, &gpu_timeline.metadata)?,
            None => reference_metadata = Some(gpu_timeline.metadata.clone()),
        }

        let host_submission = measure_host_submission(&stream, arguments.config, || {
            enqueue_repeated_scale(&kernels, &stream, &input, &output, launch_count)
        })?;
        let sequence_average_us_per_launch =
            host_submission.median_ms * 1_000.0 / launch_count as f64;
        results.push(ControlResult {
            launch_count,
            gpu_timeline,
            host_submission,
            host_submission_sequence_average_us_per_launch: sequence_average_us_per_launch,
        });
    }

    let actual = output.to_vec(&stream)?;
    let expected = input_host[0] * CONTROL_SCALE;
    if actual.len() != CONTROL_ELEMENTS || actual[0].to_bits() != expected.to_bits() {
        return Err(NnisError::unsupported(format!(
            "launch-control correctness gate failed: actual={actual:?} expected={expected}"
        )));
    }

    let metadata = reference_metadata
        .ok_or_else(|| NnisError::invalid_input("launch-control benchmark produced no metadata"))?;
    Ok(Report {
        schema_version: 1,
        experiment: "R2-cuda-host-launch-submission-control-v1",
        promotion_state: "diagnostic-only; no runtime or kernel-selection change",
        run_context_id,
        metadata,
        config: ControlConfig {
            kernel: "nnis_scale_f32",
            elements_per_launch: CONTROL_ELEMENTS,
            scale: CONTROL_SCALE,
            launch_counts: arguments.launch_counts,
            warmups: arguments.config.warmup_iterations,
            iterations: arguments.config.iterations,
            smollm2_projection_launch_reference: 211,
        },
        results,
        limitations: vec![
            "this is an isolated Driver-API launch/submission control, not full-model execution",
            "the one-element scale kernel intentionally minimizes useful GPU work and is not a decoder kernel proxy",
            "host_submission measures only time spent issuing the enqueue calls; synchronization happens after the timed interval",
            "host submission and CUDA-event GPU timeline are asynchronous measurements and must not be added as independent costs",
            "211 is retained only as the current SmolLM2 projection-launch count reference; this control does not reproduce those 211 projection kernels",
            "results may vary with driver, power, clocks, OS scheduling, and process state; fingerprint-compatible evidence is required",
            "no result from this control is sufficient to justify a fusion or runtime promotion",
        ],
    })
}

fn main() {
    match run() {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("failed serializing launch-submission report: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("R2 launch-submission control failed: {error}");
            std::process::exit(1);
        }
    }
}
