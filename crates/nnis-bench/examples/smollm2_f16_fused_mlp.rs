use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_model::{
    F16FusedMlpCandidate, F16FusedProjectionGroupsCandidate, F16ReferenceKernels,
    F16TransposedProjectionCandidate,
};
use nnis_rt::{gpu_context, Context, DeviceBuffer, NnisError, Result, Stream};
use serde::Serialize;
use std::sync::Arc;

const HIDDEN: usize = 576;
const INTERMEDIATE: usize = 1536;
const LAYERS: usize = 30;

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    experiment: &'static str,
    promotion_state: &'static str,
    arithmetic_contract: &'static str,
    hidden_size: usize,
    intermediate_size: usize,
    layers: usize,
    warmups: usize,
    iterations: usize,
    bitwise_equal: bool,
    baseline_launches_per_layer: usize,
    candidate_launches_per_layer: usize,
    launches_removed_per_decoder_token: usize,
    logical_gate_up_intermediate_bytes_avoided_per_layer: u64,
    logical_gate_up_intermediate_bytes_avoided_per_decoder_token: u64,
    baseline: BenchmarkReport,
    candidate: BenchmarkReport,
    baseline_median_ms: f64,
    candidate_median_ms: f64,
    latency_ratio_baseline_over_candidate: f64,
    limitations: Vec<&'static str>,
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            NnisError::invalid_input(format!("invalid {name}={value:?}: {error}"))
        }),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(NnisError::invalid_input(format!(
            "failed to read {name}: {error}"
        ))),
    }
}

fn require_run_context() -> Result<()> {
    match std::env::var("NNIS_BENCH_RUN_CONTEXT_ID") {
        Ok(value) if !value.trim().is_empty() => Ok(()),
        _ => Err(NnisError::invalid_input(
            "NNIS_BENCH_RUN_CONTEXT_ID is required for fused MLP evidence",
        )),
    }
}

fn deterministic_f32(elements: usize, salt: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let value = ((index.wrapping_mul(29 + salt) + 13 * salt) % 127) as i32 - 63;
            value as f32 * 0.0078125
        })
        .collect()
}

fn narrow(
    context: &Arc<Context>,
    stream: &Stream,
    reference: &F16ReferenceKernels,
    host: &[f32],
) -> Result<DeviceBuffer<u16>> {
    let source = DeviceBuffer::from_host(context, stream, host)?;
    let output = DeviceBuffer::<u16>::new(context, host.len())?;
    unsafe { reference.enqueue_narrow_from_f32(stream, &source, &output)? };
    stream.synchronize()?;
    Ok(output)
}

fn prepare_nk_weight(
    context: &Arc<Context>,
    stream: &Stream,
    reference: &F16ReferenceKernels,
    transposed: &F16TransposedProjectionCandidate,
    salt: usize,
) -> Result<DeviceBuffer<u16>> {
    let kn = narrow(
        context,
        stream,
        reference,
        &deterministic_f32(HIDDEN * INTERMEDIATE, salt),
    )?;
    let nk = DeviceBuffer::<u16>::new(context, HIDDEN * INTERMEDIATE)?;
    unsafe {
        transposed.enqueue_transpose_kn_to_nk(
            stream,
            &kn,
            &nk,
            HIDDEN,
            INTERMEDIATE,
        )?;
    }
    stream.synchronize()?;
    Ok(nk)
}

fn require_equal(left: &[u16], right: &[u16]) -> Result<()> {
    if left == right {
        return Ok(());
    }
    let index = left
        .iter()
        .zip(right)
        .position(|(a, b)| a != b)
        .unwrap_or(0);
    Err(NnisError::unsupported(format!(
        "fused MLP changed F16 bits at output {index}: 0x{:04x} != 0x{:04x}",
        left[index], right[index]
    )))
}

fn run() -> Result<()> {
    require_run_context()?;
    let context = gpu_context().ok_or_else(|| NnisError::unsupported("no CUDA device"))?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let reference = F16ReferenceKernels::load(&context, &compiler)?;
    let transposed = F16TransposedProjectionCandidate::load(&context, &compiler)?;
    let grouped = F16FusedProjectionGroupsCandidate::load(&context, &compiler)?;
    let candidate_kernel = F16FusedMlpCandidate::load(&context, &compiler)?;
    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 50)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 500)?;
    let config = BenchConfig::new(warmups, iterations);
    config.validate()?;

    let input = narrow(
        &context,
        &stream,
        &reference,
        &deterministic_f32(HIDDEN, 1),
    )?;
    let gate_weight = prepare_nk_weight(&context, &stream, &reference, &transposed, 2)?;
    let up_weight = prepare_nk_weight(&context, &stream, &reference, &transposed, 3)?;

    let gate = DeviceBuffer::<u16>::new(&context, INTERMEDIATE)?;
    let up = DeviceBuffer::<u16>::new(&context, INTERMEDIATE)?;
    let baseline_output = DeviceBuffer::<u16>::new(&context, INTERMEDIATE)?;
    let candidate_output = DeviceBuffer::<u16>::new(&context, INTERMEDIATE)?;

    unsafe {
        grouped.enqueue_gate_up_nk(
            &stream,
            &input,
            &gate_weight,
            &up_weight,
            &gate,
            &up,
            HIDDEN,
            INTERMEDIATE,
        )?;
        reference.enqueue_silu_multiply(&stream, &gate, &up, &baseline_output)?;
        candidate_kernel.enqueue_gate_up_silu_nk(
            &stream,
            &input,
            &gate_weight,
            &up_weight,
            &candidate_output,
            HIDDEN,
            INTERMEDIATE,
        )?;
    }
    stream.synchronize()?;
    require_equal(
        &baseline_output.to_vec(&stream)?,
        &candidate_output.to_vec(&stream)?,
    )?;

    let work_items = (2 * HIDDEN * INTERMEDIATE) as u64;
    let baseline = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("smollm2_f16_gate_up_grouped_then_silu", "f16")
            .with_dimension("hidden", HIDDEN as u64)
            .with_dimension("intermediate", INTERMEDIATE as u64)
            .with_dimension("launches", 2)
            .with_work_items(work_items),
        config,
        || unsafe {
            grouped.enqueue_gate_up_nk(
                &stream,
                &input,
                &gate_weight,
                &up_weight,
                &gate,
                &up,
                HIDDEN,
                INTERMEDIATE,
            )?;
            reference.enqueue_silu_multiply(&stream, &gate, &up, &baseline_output)
        },
    )?;
    let candidate = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("smollm2_f16_gate_up_silu_fused", "f16")
            .with_dimension("hidden", HIDDEN as u64)
            .with_dimension("intermediate", INTERMEDIATE as u64)
            .with_dimension("launches", 1)
            .with_work_items(work_items),
        config,
        || unsafe {
            candidate_kernel.enqueue_gate_up_silu_nk(
                &stream,
                &input,
                &gate_weight,
                &up_weight,
                &candidate_output,
                HIDDEN,
                INTERMEDIATE,
            )
        },
    )?;

    baseline
        .metadata
        .require_compatible_environment(&candidate.metadata)?;
    if baseline.metadata.git_commit != candidate.metadata.git_commit {
        return Err(NnisError::invalid_input(
            "baseline and fused MLP candidate were not measured from one git commit",
        ));
    }
    require_equal(
        &baseline_output.to_vec(&stream)?,
        &candidate_output.to_vec(&stream)?,
    )?;

    let baseline_median = baseline.statistics.median_ms;
    let candidate_median = candidate.statistics.median_ms;
    if baseline_median <= 0.0 || candidate_median <= 0.0 {
        return Err(NnisError::unsupported(
            "fused MLP benchmark produced a non-positive median",
        ));
    }

    let avoided_per_layer = (4usize)
        .checked_mul(INTERMEDIATE)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| NnisError::invalid_input("fused MLP byte accounting overflow"))?
        as u64;
    let avoided_per_token = avoided_per_layer
        .checked_mul(LAYERS as u64)
        .ok_or_else(|| NnisError::invalid_input("fused MLP token byte accounting overflow"))?;

    let report = Report {
        schema_version: 1,
        experiment: "R1-F16-gate-up-silu-projection-fusion",
        promotion_state: "candidate-only; current fused projection execution plan unchanged",
        arithmetic_contract: "same 128-lane F32 FMA partition and reduction tree for gate/up; projection outputs rounded to F16, SiLU rounded to F16, product rounded to F16",
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        layers: LAYERS,
        warmups,
        iterations,
        bitwise_equal: true,
        baseline_launches_per_layer: 2,
        candidate_launches_per_layer: 1,
        launches_removed_per_decoder_token: LAYERS,
        logical_gate_up_intermediate_bytes_avoided_per_layer: avoided_per_layer,
        logical_gate_up_intermediate_bytes_avoided_per_decoder_token: avoided_per_token,
        baseline_median_ms: baseline_median,
        candidate_median_ms: candidate_median,
        latency_ratio_baseline_over_candidate: baseline_median / candidate_median,
        baseline,
        candidate,
        limitations: vec![
            "isolated CUDA-event evidence does not establish end-to-end generation speedup",
            "logical intermediate-byte accounting is not measured DRAM traffic",
            "down projection and all attention kernels are unchanged",
            "runtime integration requires a separate explicit execution-plan gate",
            "physical Thor qualification is required before integration",
        ],
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| {
            NnisError::unsupported(format!("failed to serialize fused MLP report: {error}"))
        })?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("F16 fused MLP benchmark failed: {error}");
        std::process::exit(1);
    }
}
