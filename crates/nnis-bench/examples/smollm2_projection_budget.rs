use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::F32Gemm;
use nnis_rt::{gpu_context, DeviceBuffer, NnisError, Result, Stream};
use serde_json::json;

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
        Ok(value) => {
            let parsed = value.parse::<usize>().map_err(|error| {
                NnisError::invalid_input(format!("invalid {name}={value:?}: {error}"))
            })?;
            Ok(parsed)
        }
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(NnisError::invalid_input(format!(
            "failed to read {name}: {error}"
        ))),
    }
}

fn benchmark_projection(
    context: &std::sync::Arc<nnis_rt::Context>,
    stream: &Stream,
    gemm: &F32Gemm,
    config: BenchConfig,
    case: ProjectionCase,
) -> Result<BenchmarkReport> {
    let input = DeviceBuffer::from_host(context, stream, &vec![0.0_f32; case.k])?;
    let weight = DeviceBuffer::from_host(context, stream, &vec![0.0_f32; case.k * case.n])?;
    let output = DeviceBuffer::<f32>::new(context, case.n)?;

    let descriptor = BenchmarkCase::new(case.name, "f32")
        .with_dimension("m", 1)
        .with_dimension("k", case.k as u64)
        .with_dimension("n", case.n as u64)
        .with_work_items((case.k * case.n) as u64)
        .with_bytes_per_iteration(
            ((case.k + case.k * case.n + case.n) * core::mem::size_of::<f32>()) as u64,
        );

    let report = benchmark_gpu(context, stream, descriptor, config, || {
        // SAFETY: all buffers outlive the benchmark, have the exact GEMM
        // shapes declared above, and the harness drains the shared stream.
        unsafe { gemm.enqueue_gemm(stream, &input, &weight, &output, 1, case.n, case.k) }
    })?;

    let actual = output.to_vec(stream)?;
    if actual.iter().any(|value| value.to_bits() != 0) {
        return Err(NnisError::unsupported(format!(
            "{} zero-input correctness guard failed",
            case.name
        )));
    }
    Ok(report)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("SmolLM2 projection budget failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let context = gpu_context().ok_or_else(|| NnisError::unsupported("no CUDA device"))?;
    let stream = Stream::new(&context)?;
    let gemm = F32Gemm::load(&context, &JitCompiler::new())?;
    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 10)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 50)?;
    let config = BenchConfig::new(warmups, iterations);
    config.validate()?;

    // Equal shapes are intentionally benchmarked once and weighted by their
    // actual decoder use count. This keeps the measurement compact while the
    // accounting remains faithful to the current SmolLM2 f32 runtime graph.
    let layer_cases = [
        ProjectionCase {
            name: "q_o_f32_gemm_m1_k576_n576",
            k: HIDDEN,
            n: HIDDEN,
            uses_per_layer: 2,
        },
        ProjectionCase {
            name: "k_v_f32_gemm_m1_k576_n192",
            k: HIDDEN,
            n: KV_WIDTH,
            uses_per_layer: 2,
        },
        ProjectionCase {
            name: "gate_up_f32_gemm_m1_k576_n1536",
            k: HIDDEN,
            n: INTERMEDIATE,
            uses_per_layer: 2,
        },
        ProjectionCase {
            name: "down_f32_gemm_m1_k1536_n576",
            k: INTERMEDIATE,
            n: HIDDEN,
            uses_per_layer: 1,
        },
    ];

    let mut layer_reports = Vec::with_capacity(layer_cases.len());
    let mut one_layer_projection_ms = 0.0_f64;
    for case in layer_cases {
        let report = benchmark_projection(&context, &stream, &gemm, config, case)?;
        let weighted_ms = report.statistics.median_ms * case.uses_per_layer as f64;
        one_layer_projection_ms += weighted_ms;
        layer_reports.push(json!({
            "name": case.name,
            "k": case.k,
            "n": case.n,
            "uses_per_layer": case.uses_per_layer,
            "median_ms": report.statistics.median_ms,
            "p95_ms": report.statistics.p95_ms,
            "weighted_median_ms_per_layer": weighted_ms,
            "samples_ms": report.samples_ms,
        }));
    }

    let lm_head_case = ProjectionCase {
        name: "lm_head_f32_gemm_m1_k576_n49152",
        k: HIDDEN,
        n: VOCAB,
        uses_per_layer: 0,
    };
    let lm_head = benchmark_projection(&context, &stream, &gemm, config, lm_head_case)?;

    let transformer_layer_projection_ms = one_layer_projection_ms * LAYERS as f64;
    let projection_ms_per_decoder_token =
        transformer_layer_projection_ms + lm_head.statistics.median_ms;
    let executed_decoder_tokens = PROMPT_TOKENS + DECODE_STEPS;
    let estimated_projection_gpu_ms_per_request =
        projection_ms_per_decoder_token * executed_decoder_tokens as f64;
    let fraction_of_accepted_request =
        estimated_projection_gpu_ms_per_request / ACCEPTED_REQUEST_MEDIAN_MS;

    // This is an attribution estimate, not an additive reconstruction of host
    // wall time. CUDA-event kernel durations omit host submission overhead and
    // event instrumentation is measured outside the production request path.
    let report = json!({
        "schema_version": 1,
        "profile": "smollm2-135m-current-f32-projection-budget",
        "measurement": "isolated CUDA-event timing of current F32Gemm at exact batch-one SmolLM2 projection shapes; weighted estimate, not an additive end-to-end reconstruction",
        "model": {
            "hidden_size": HIDDEN,
            "kv_width": KV_WIDTH,
            "intermediate_size": INTERMEDIATE,
            "vocab_size": VOCAB,
            "layers": LAYERS,
        },
        "request_shape": {
            "prompt_tokens": PROMPT_TOKENS,
            "decode_steps": DECODE_STEPS,
            "executed_decoder_tokens": executed_decoder_tokens,
            "accepted_request_median_ms": ACCEPTED_REQUEST_MEDIAN_MS,
            "accepted_request_tok_s": 40.67982822083172_f64,
        },
        "benchmark_config": {
            "warmups": warmups,
            "iterations": iterations,
        },
        "hardware": {
            "gpu_name": context.props().name,
            "compute_capability_major": context.props().compute_capability.0,
            "compute_capability_minor": context.props().compute_capability.1,
            "multiprocessor_count": context.props().multiprocessor_count,
        },
        "layer_projection_cases": layer_reports,
        "lm_head": {
            "median_ms": lm_head.statistics.median_ms,
            "p95_ms": lm_head.statistics.p95_ms,
            "samples_ms": lm_head.samples_ms,
        },
        "weighted_estimate": {
            "one_layer_projection_median_ms": one_layer_projection_ms,
            "all_layers_projection_median_ms_per_decoder_token": transformer_layer_projection_ms,
            "lm_head_median_ms_per_decoder_token": lm_head.statistics.median_ms,
            "projection_gpu_ms_per_decoder_token": projection_ms_per_decoder_token,
            "projection_gpu_ms_per_request": estimated_projection_gpu_ms_per_request,
            "projection_gpu_fraction_of_accepted_request_median": fraction_of_accepted_request,
        },
        "limitations": [
            "isolated kernel timings do not include host launch/submission overhead",
            "the weighted sum assumes isolated median durations compose linearly",
            "attention cost grows with sequence position and is intentionally outside this projection-only slice",
            "no optimization candidate is evaluated by this profile",
        ],
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| {
            NnisError::unsupported(format!("failed to serialize projection budget: {error}"))
        })?
    );
    Ok(())
}
