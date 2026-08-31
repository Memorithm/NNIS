use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::{F32Gemm, F32Gemv};
use nnis_rt::{gpu_context, Context, DeviceBuffer, NnisError, Result, Stream};
use serde_json::json;
use std::sync::Arc;

const HIDDEN: usize = 576;
const KV_WIDTH: usize = 192;
const INTERMEDIATE: usize = 1536;
const VOCAB: usize = 49_152;
const LAYERS: usize = 30;
const PROMPT_TOKENS: usize = 3;
const DECODE_STEPS: usize = 32;
const ACCEPTED_REQUEST_MEDIAN_MS: f64 = 786.630_657;

#[derive(Clone, Copy)]
struct ProjectionCase {
    name: &'static str,
    k: usize,
    n: usize,
    uses_per_layer: usize,
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

fn deterministic_input(k: usize) -> Vec<f32> {
    (0..k)
        .map(|index| ((index * 29 % 61) as f32 - 30.0) * 0.015625)
        .collect()
}

fn deterministic_weight(k: usize, n: usize) -> Vec<f32> {
    (0..k * n)
        .map(|index| ((index * 17 % 67) as f32 - 33.0) * 0.001953125)
        .collect()
}

fn compare_bits(name: &str, baseline: &[f32], candidate: &[f32]) -> Result<()> {
    if baseline.len() != candidate.len() {
        return Err(NnisError::unsupported(format!(
            "{name}: output length mismatch {} != {}",
            baseline.len(),
            candidate.len()
        )));
    }
    if let Some((index, (&left, &right))) = baseline
        .iter()
        .zip(candidate)
        .enumerate()
        .find(|(_, (left, right))| left.to_bits() != right.to_bits())
    {
        return Err(NnisError::unsupported(format!(
            "{name}: GEMV is not bitwise-equivalent to GEMM at output {index}: {left:e} != {right:e}"
        )));
    }
    Ok(())
}

fn benchmark_case(
    context: &Arc<Context>,
    stream: &Stream,
    gemm: &F32Gemm,
    gemv: &F32Gemv,
    config: BenchConfig,
    case: ProjectionCase,
) -> Result<(BenchmarkReport, BenchmarkReport)> {
    let input = DeviceBuffer::from_host(context, stream, &deterministic_input(case.k))?;
    let weight_host = deterministic_weight(case.k, case.n);
    let weight = DeviceBuffer::from_host(context, stream, &weight_host)?;
    drop(weight_host);
    let gemm_output = DeviceBuffer::<f32>::new(context, case.n)?;
    let gemv_output = DeviceBuffer::<f32>::new(context, case.n)?;

    gemm.gemm(stream, &input, &weight, &gemm_output, 1, case.n, case.k)?;
    gemv.project_kn(stream, &input, &weight, &gemv_output, case.k, case.n)?;
    compare_bits(
        case.name,
        &gemm_output.to_vec(stream)?,
        &gemv_output.to_vec(stream)?,
    )?;

    let base_case = BenchmarkCase::new(format!("{}_gemm", case.name), "f32")
        .with_dimension("m", 1)
        .with_dimension("k", case.k as u64)
        .with_dimension("n", case.n as u64)
        .with_work_items((case.k * case.n) as u64);
    let candidate_case = BenchmarkCase::new(format!("{}_gemv_kn", case.name), "f32")
        .with_dimension("m", 1)
        .with_dimension("k", case.k as u64)
        .with_dimension("n", case.n as u64)
        .with_work_items((case.k * case.n) as u64);

    let gemm_report = benchmark_gpu(context, stream, base_case, config, || {
        // SAFETY: all buffers outlive the benchmark and the harness drains the stream.
        unsafe { gemm.enqueue_gemm(stream, &input, &weight, &gemm_output, 1, case.n, case.k) }
    })?;
    let gemv_report = benchmark_gpu(context, stream, candidate_case, config, || {
        // SAFETY: all buffers outlive the benchmark and the harness drains the stream.
        unsafe { gemv.enqueue_project_kn(stream, &input, &weight, &gemv_output, case.k, case.n) }
    })?;

    compare_bits(
        case.name,
        &gemm_output.to_vec(stream)?,
        &gemv_output.to_vec(stream)?,
    )?;
    Ok((gemm_report, gemv_report))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("SmolLM2 projection candidate benchmark failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let context = gpu_context().ok_or_else(|| NnisError::unsupported("no CUDA device"))?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let gemm = F32Gemm::load(&context, &compiler)?;
    let gemv_block_size = u32::try_from(env_usize("NNIS_GEMV_BLOCK_SIZE", 256)?)
        .map_err(|_| NnisError::invalid_input("NNIS_GEMV_BLOCK_SIZE exceeds u32"))?;
    let gemv = F32Gemv::load_with_block_size(&context, &compiler, gemv_block_size)?;
    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 100)?;
    let config = BenchConfig::new(warmups, iterations);
    config.validate()?;

    let layer_cases = [
        ProjectionCase {
            name: "q_o_m1_k576_n576",
            k: HIDDEN,
            n: HIDDEN,
            uses_per_layer: 2,
        },
        ProjectionCase {
            name: "k_v_m1_k576_n192",
            k: HIDDEN,
            n: KV_WIDTH,
            uses_per_layer: 2,
        },
        ProjectionCase {
            name: "gate_up_m1_k576_n1536",
            k: HIDDEN,
            n: INTERMEDIATE,
            uses_per_layer: 2,
        },
        ProjectionCase {
            name: "down_m1_k1536_n576",
            k: INTERMEDIATE,
            n: HIDDEN,
            uses_per_layer: 1,
        },
    ];

    let mut cases_json = Vec::with_capacity(layer_cases.len());
    let mut gemm_layer_ms = 0.0_f64;
    let mut gemv_layer_ms = 0.0_f64;
    for case in layer_cases {
        let (gemm_report, gemv_report) =
            benchmark_case(&context, &stream, &gemm, &gemv, config, case)?;
        let gemm_weighted = gemm_report.statistics.median_ms * case.uses_per_layer as f64;
        let gemv_weighted = gemv_report.statistics.median_ms * case.uses_per_layer as f64;
        gemm_layer_ms += gemm_weighted;
        gemv_layer_ms += gemv_weighted;
        cases_json.push(json!({
            "name": case.name,
            "k": case.k,
            "n": case.n,
            "uses_per_layer": case.uses_per_layer,
            "bitwise_equivalent": true,
            "gemm_median_ms": gemm_report.statistics.median_ms,
            "gemm_p95_ms": gemm_report.statistics.p95_ms,
            "gemv_median_ms": gemv_report.statistics.median_ms,
            "gemv_p95_ms": gemv_report.statistics.p95_ms,
            "latency_speedup_gemm_over_gemv": gemm_report.statistics.median_ms / gemv_report.statistics.median_ms,
            "gemm_weighted_ms_per_layer": gemm_weighted,
            "gemv_weighted_ms_per_layer": gemv_weighted,
        }));
    }

    let lm_case = ProjectionCase {
        name: "lm_head_m1_k576_n49152",
        k: HIDDEN,
        n: VOCAB,
        uses_per_layer: 0,
    };
    let (lm_gemm, lm_gemv) = benchmark_case(&context, &stream, &gemm, &gemv, config, lm_case)?;

    let gemm_per_token = gemm_layer_ms * LAYERS as f64 + lm_gemm.statistics.median_ms;
    let gemv_per_token = gemv_layer_ms * LAYERS as f64 + lm_gemv.statistics.median_ms;
    let executed_decoder_tokens = PROMPT_TOKENS + DECODE_STEPS;
    let gemm_request_projection_ms = gemm_per_token * executed_decoder_tokens as f64;
    let gemv_request_projection_ms = gemv_per_token * executed_decoder_tokens as f64;

    let report = json!({
        "schema_version": 1,
        "experiment": "E1-f32-decode-projection-gemm-vs-gemv-kn",
        "promotion_state": "candidate-only; model runtime unchanged",
        "correctness_gate": "bitwise equality with current F32Gemm on every exact SmolLM2 projection shape before and after timing",
        "hardware": {
            "gpu_name": context.props().name,
            "compute_capability_major": context.props().compute_capability.0,
            "compute_capability_minor": context.props().compute_capability.1,
            "multiprocessor_count": context.props().multiprocessor_count,
        },
        "benchmark_config": {
            "warmups": warmups,
            "iterations": iterations,
            "gemv_block_size": gemv_block_size,
        },
        "cases": cases_json,
        "lm_head": {
            "bitwise_equivalent": true,
            "gemm_median_ms": lm_gemm.statistics.median_ms,
            "gemm_p95_ms": lm_gemm.statistics.p95_ms,
            "gemv_median_ms": lm_gemv.statistics.median_ms,
            "gemv_p95_ms": lm_gemv.statistics.p95_ms,
            "latency_speedup_gemm_over_gemv": lm_gemm.statistics.median_ms / lm_gemv.statistics.median_ms,
        },
        "weighted_projection_estimate": {
            "gemm_ms_per_decoder_token": gemm_per_token,
            "gemv_ms_per_decoder_token": gemv_per_token,
            "gemm_ms_per_35_token_request": gemm_request_projection_ms,
            "gemv_ms_per_35_token_request": gemv_request_projection_ms,
            "isolated_projection_latency_speedup": gemm_request_projection_ms / gemv_request_projection_ms,
            "isolated_projection_ms_saved_per_request": gemm_request_projection_ms - gemv_request_projection_ms,
            "accepted_end_to_end_request_median_ms": ACCEPTED_REQUEST_MEDIAN_MS,
        },
        "limitations": [
            "isolated CUDA-event kernel medians are not an end-to-end throughput result",
            "host launch overhead is not included in these kernel durations",
            "the candidate is not used by nnis-model in this experiment",
            "end-to-end tok/s must be remeasured after a separate promotion change",
        ],
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| {
            NnisError::unsupported(format!("failed to serialize candidate report: {error}"))
        })?
    );
    Ok(())
}
