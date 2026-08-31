use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::{F32Gemm, F32Gemv};
use nnis_rt::{gpu_context, Context, DeviceBuffer, NnisError, Result, Stream};
use serde_json::json;
use std::sync::Arc;

const HIDDEN: usize = 576;
const KV_WIDTH: usize = 192;
const INTERMEDIATE: usize = 1_536;
const VOCAB: usize = 49_152;
const LAYERS: usize = 30;
const PROJECTION_LAUNCHES_PER_TOKEN: usize = LAYERS * 7 + 1;

struct LayerWeights {
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    o: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    down: DeviceBuffer<f32>,
}

struct ProjectionFixture {
    layers: Vec<LayerWeights>,
    lm_head: DeviceBuffer<f32>,
    hidden_input: DeviceBuffer<f32>,
    intermediate_input: DeviceBuffer<f32>,
    hidden_output: DeviceBuffer<f32>,
    kv_output: DeviceBuffer<f32>,
    intermediate_output: DeviceBuffer<f32>,
    vocab_output: DeviceBuffer<f32>,
}

#[derive(Clone, Copy)]
enum MeasurementOrder {
    BaselineFirst,
    HybridFirst,
}

impl MeasurementOrder {
    fn from_env() -> Result<Self> {
        match std::env::var("NNIS_STREAM_ORDER") {
            Ok(value) if value == "baseline-first" => Ok(Self::BaselineFirst),
            Ok(value) if value == "hybrid-first" => Ok(Self::HybridFirst),
            Ok(value) => Err(NnisError::invalid_input(format!(
                "NNIS_STREAM_ORDER must be baseline-first or hybrid-first, got {value:?}"
            ))),
            Err(std::env::VarError::NotPresent) => Ok(Self::BaselineFirst),
            Err(error) => Err(NnisError::invalid_input(format!(
                "failed to read NNIS_STREAM_ORDER: {error}"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::BaselineFirst => "baseline-first",
            Self::HybridFirst => "hybrid-first",
        }
    }
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

fn deterministic_input(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|index| (((index + salt) * 29 % 61) as f32 - 30.0) * 0.015625)
        .collect()
}

fn deterministic_weight(k: usize, n: usize, salt: usize) -> Vec<f32> {
    (0..k * n)
        .map(|index| (((index + salt) * 17 % 67) as f32 - 33.0) * 0.001953125)
        .collect()
}

fn upload_weight(
    context: &Arc<Context>,
    stream: &Stream,
    host: &[f32],
) -> Result<DeviceBuffer<f32>> {
    DeviceBuffer::from_host(context, stream, host)
}

impl ProjectionFixture {
    fn new(context: &Arc<Context>, stream: &Stream) -> Result<Self> {
        let qo_host = deterministic_weight(HIDDEN, HIDDEN, 1);
        let kv_host = deterministic_weight(HIDDEN, KV_WIDTH, 3);
        let gate_up_host = deterministic_weight(HIDDEN, INTERMEDIATE, 5);
        let down_host = deterministic_weight(INTERMEDIATE, HIDDEN, 7);

        let mut layers = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            layers.push(LayerWeights {
                q: upload_weight(context, stream, &qo_host)?,
                k: upload_weight(context, stream, &kv_host)?,
                v: upload_weight(context, stream, &kv_host)?,
                o: upload_weight(context, stream, &qo_host)?,
                gate: upload_weight(context, stream, &gate_up_host)?,
                up: upload_weight(context, stream, &gate_up_host)?,
                down: upload_weight(context, stream, &down_host)?,
            });
        }

        let lm_head_host = deterministic_weight(HIDDEN, VOCAB, 11);
        let lm_head = upload_weight(context, stream, &lm_head_host)?;
        let hidden_input = DeviceBuffer::from_host(context, stream, &deterministic_input(HIDDEN, 13))?;
        let intermediate_input =
            DeviceBuffer::from_host(context, stream, &deterministic_input(INTERMEDIATE, 17))?;
        let hidden_output = DeviceBuffer::<f32>::new(context, HIDDEN)?;
        let kv_output = DeviceBuffer::<f32>::new(context, KV_WIDTH)?;
        let intermediate_output = DeviceBuffer::<f32>::new(context, INTERMEDIATE)?;
        let vocab_output = DeviceBuffer::<f32>::new(context, VOCAB)?;
        stream.synchronize()?;

        Ok(Self {
            layers,
            lm_head,
            hidden_input,
            intermediate_input,
            hidden_output,
            kv_output,
            intermediate_output,
            vocab_output,
        })
    }
}

fn projection_weight_elements() -> usize {
    let per_layer = 2 * HIDDEN * HIDDEN
        + 2 * HIDDEN * KV_WIDTH
        + 2 * HIDDEN * INTERMEDIATE
        + INTERMEDIATE * HIDDEN;
    per_layer * LAYERS + HIDDEN * VOCAB
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
            "{name}: hybrid projection is not bitwise-equivalent to GEMM at output {index}: {left:e} != {right:e}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn check_shape(
    name: &str,
    stream: &Stream,
    gemm: &F32Gemm,
    gemv: &F32Gemv,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    baseline_output: &DeviceBuffer<f32>,
    candidate_output: &DeviceBuffer<f32>,
    k: usize,
    n: usize,
) -> Result<()> {
    gemm.gemm(stream, input, weight, baseline_output, 1, n, k)?;
    gemv.project_kn(stream, input, weight, candidate_output, k, n)?;
    compare_bits(
        name,
        &baseline_output.to_vec(stream)?,
        &candidate_output.to_vec(stream)?,
    )
}

fn validate_selected_shapes(
    context: &Arc<Context>,
    stream: &Stream,
    fixture: &ProjectionFixture,
    gemm: &F32Gemm,
    gemv_512: &F32Gemv,
    gemv_128: &F32Gemv,
    gemv_64: &F32Gemv,
) -> Result<()> {
    let hidden_candidate = DeviceBuffer::<f32>::new(context, HIDDEN)?;
    let intermediate_candidate = DeviceBuffer::<f32>::new(context, INTERMEDIATE)?;
    let vocab_candidate = DeviceBuffer::<f32>::new(context, VOCAB)?;
    let first = &fixture.layers[0];

    check_shape(
        "q_o_m1_k576_n576",
        stream,
        gemm,
        gemv_512,
        &fixture.hidden_input,
        &first.q,
        &fixture.hidden_output,
        &hidden_candidate,
        HIDDEN,
        HIDDEN,
    )?;
    check_shape(
        "gate_up_m1_k576_n1536",
        stream,
        gemm,
        gemv_128,
        &fixture.hidden_input,
        &first.gate,
        &fixture.intermediate_output,
        &intermediate_candidate,
        HIDDEN,
        INTERMEDIATE,
    )?;
    check_shape(
        "lm_head_m1_k576_n49152",
        stream,
        gemm,
        gemv_64,
        &fixture.hidden_input,
        &fixture.lm_head,
        &fixture.vocab_output,
        &vocab_candidate,
        HIDDEN,
        VOCAB,
    )?;
    Ok(())
}

fn enqueue_baseline(
    stream: &Stream,
    fixture: &ProjectionFixture,
    gemm: &F32Gemm,
) -> Result<()> {
    for layer in &fixture.layers {
        unsafe {
            gemm.enqueue_gemm(
                stream,
                &fixture.hidden_input,
                &layer.q,
                &fixture.hidden_output,
                1,
                HIDDEN,
                HIDDEN,
            )?;
            gemm.enqueue_gemm(
                stream,
                &fixture.hidden_input,
                &layer.k,
                &fixture.kv_output,
                1,
                KV_WIDTH,
                HIDDEN,
            )?;
            gemm.enqueue_gemm(
                stream,
                &fixture.hidden_input,
                &layer.v,
                &fixture.kv_output,
                1,
                KV_WIDTH,
                HIDDEN,
            )?;
            gemm.enqueue_gemm(
                stream,
                &fixture.hidden_input,
                &layer.o,
                &fixture.hidden_output,
                1,
                HIDDEN,
                HIDDEN,
            )?;
            gemm.enqueue_gemm(
                stream,
                &fixture.hidden_input,
                &layer.gate,
                &fixture.intermediate_output,
                1,
                INTERMEDIATE,
                HIDDEN,
            )?;
            gemm.enqueue_gemm(
                stream,
                &fixture.hidden_input,
                &layer.up,
                &fixture.intermediate_output,
                1,
                INTERMEDIATE,
                HIDDEN,
            )?;
            gemm.enqueue_gemm(
                stream,
                &fixture.intermediate_input,
                &layer.down,
                &fixture.hidden_output,
                1,
                HIDDEN,
                INTERMEDIATE,
            )?;
        }
    }
    unsafe {
        gemm.enqueue_gemm(
            stream,
            &fixture.hidden_input,
            &fixture.lm_head,
            &fixture.vocab_output,
            1,
            VOCAB,
            HIDDEN,
        )
    }
}

fn enqueue_hybrid(
    stream: &Stream,
    fixture: &ProjectionFixture,
    gemm: &F32Gemm,
    gemv_512: &F32Gemv,
    gemv_128: &F32Gemv,
    gemv_64: &F32Gemv,
) -> Result<()> {
    for layer in &fixture.layers {
        unsafe {
            gemv_512.enqueue_project_kn(
                stream,
                &fixture.hidden_input,
                &layer.q,
                &fixture.hidden_output,
                HIDDEN,
                HIDDEN,
            )?;
            gemm.enqueue_gemm(
                stream,
                &fixture.hidden_input,
                &layer.k,
                &fixture.kv_output,
                1,
                KV_WIDTH,
                HIDDEN,
            )?;
            gemm.enqueue_gemm(
                stream,
                &fixture.hidden_input,
                &layer.v,
                &fixture.kv_output,
                1,
                KV_WIDTH,
                HIDDEN,
            )?;
            gemv_512.enqueue_project_kn(
                stream,
                &fixture.hidden_input,
                &layer.o,
                &fixture.hidden_output,
                HIDDEN,
                HIDDEN,
            )?;
            gemv_128.enqueue_project_kn(
                stream,
                &fixture.hidden_input,
                &layer.gate,
                &fixture.intermediate_output,
                HIDDEN,
                INTERMEDIATE,
            )?;
            gemv_128.enqueue_project_kn(
                stream,
                &fixture.hidden_input,
                &layer.up,
                &fixture.intermediate_output,
                HIDDEN,
                INTERMEDIATE,
            )?;
            gemm.enqueue_gemm(
                stream,
                &fixture.intermediate_input,
                &layer.down,
                &fixture.hidden_output,
                1,
                HIDDEN,
                INTERMEDIATE,
            )?;
        }
    }
    unsafe {
        gemv_64.enqueue_project_kn(
            stream,
            &fixture.hidden_input,
            &fixture.lm_head,
            &fixture.vocab_output,
            HIDDEN,
            VOCAB,
        )
    }
}

fn benchmark_sequence(
    context: &Arc<Context>,
    stream: &Stream,
    fixture: &ProjectionFixture,
    gemm: &F32Gemm,
    gemv_512: &F32Gemv,
    gemv_128: &F32Gemv,
    gemv_64: &F32Gemv,
    config: BenchConfig,
    order: MeasurementOrder,
) -> Result<(BenchmarkReport, BenchmarkReport)> {
    let work_items = u64::try_from(projection_weight_elements())
        .map_err(|_| NnisError::invalid_input("projection work-item count exceeds u64"))?;
    let baseline_case = BenchmarkCase::new("smollm2_projection_stream_all_gemm", "f32")
        .with_dimension("layers", LAYERS as u64)
        .with_dimension("launches", PROJECTION_LAUNCHES_PER_TOKEN as u64)
        .with_work_items(work_items);
    let hybrid_case = BenchmarkCase::new("smollm2_projection_stream_e1_hybrid", "f32")
        .with_dimension("layers", LAYERS as u64)
        .with_dimension("launches", PROJECTION_LAUNCHES_PER_TOKEN as u64)
        .with_work_items(work_items);

    match order {
        MeasurementOrder::BaselineFirst => {
            let baseline = benchmark_gpu(context, stream, baseline_case, config, || {
                enqueue_baseline(stream, fixture, gemm)
            })?;
            let hybrid = benchmark_gpu(context, stream, hybrid_case, config, || {
                enqueue_hybrid(stream, fixture, gemm, gemv_512, gemv_128, gemv_64)
            })?;
            Ok((baseline, hybrid))
        }
        MeasurementOrder::HybridFirst => {
            let hybrid = benchmark_gpu(context, stream, hybrid_case, config, || {
                enqueue_hybrid(stream, fixture, gemm, gemv_512, gemv_128, gemv_64)
            })?;
            let baseline = benchmark_gpu(context, stream, baseline_case, config, || {
                enqueue_baseline(stream, fixture, gemm)
            })?;
            Ok((baseline, hybrid))
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("SmolLM2 projection-stream diagnostic failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let context = gpu_context().ok_or_else(|| NnisError::unsupported("no CUDA device"))?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let gemm = F32Gemm::load(&context, &compiler)?;
    let gemv_512 = F32Gemv::load_with_block_size(&context, &compiler, 512)?;
    let gemv_128 = F32Gemv::load_with_block_size(&context, &compiler, 128)?;
    let gemv_64 = F32Gemv::load_with_block_size(&context, &compiler, 64)?;
    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 5)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 20)?;
    let config = BenchConfig::new(warmups, iterations);
    config.validate()?;
    let order = MeasurementOrder::from_env()?;

    let fixture = ProjectionFixture::new(&context, &stream)?;
    validate_selected_shapes(
        &context,
        &stream,
        &fixture,
        &gemm,
        &gemv_512,
        &gemv_128,
        &gemv_64,
    )?;

    let (baseline, hybrid) = benchmark_sequence(
        &context,
        &stream,
        &fixture,
        &gemm,
        &gemv_512,
        &gemv_128,
        &gemv_64,
        config,
        order,
    )?;

    let weight_elements = projection_weight_elements();
    let weight_bytes = weight_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| NnisError::invalid_input("projection working-set bytes overflow usize"))?;
    let speedup = baseline.statistics.median_ms / hybrid.statistics.median_ms;

    let report = json!({
        "schema_version": 1,
        "experiment": "E1b-f32-smollm2-streaming-projection-sequence",
        "promotion_state": "diagnostic-only; model runtime unchanged",
        "hypothesis": "E1 hot-matrix microbench overestimated GEMV because real decode streams distinct weights across 30 layers",
        "hardware": {
            "gpu_name": context.props().name,
            "compute_capability_major": context.props().compute_capability.0,
            "compute_capability_minor": context.props().compute_capability.1,
            "multiprocessor_count": context.props().multiprocessor_count,
        },
        "benchmark_config": {
            "warmups": warmups,
            "iterations": iterations,
            "measurement_order": order.as_str(),
            "layers": LAYERS,
            "projection_launches_per_token": PROJECTION_LAUNCHES_PER_TOKEN,
            "projection_weight_elements": weight_elements,
            "projection_weight_bytes": weight_bytes,
            "distinct_layer_weight_allocations": true,
        },
        "correctness_gate": {
            "q_o_gemv_512_bitwise_equal_to_gemm": true,
            "gate_up_gemv_128_bitwise_equal_to_gemm": true,
            "lm_head_gemv_64_bitwise_equal_to_gemm": true,
        },
        "baseline_all_gemm": {
            "median_ms_per_projection_sequence": baseline.statistics.median_ms,
            "p95_ms_per_projection_sequence": baseline.statistics.p95_ms,
        },
        "e1_hybrid": {
            "q_o": "gemv-512",
            "k_v": "gemm",
            "gate_up": "gemv-128",
            "down": "gemm",
            "lm_head": "gemv-64",
            "median_ms_per_projection_sequence": hybrid.statistics.median_ms,
            "p95_ms_per_projection_sequence": hybrid.statistics.p95_ms,
        },
        "baseline_over_hybrid_speedup": speedup,
        "hybrid_latency_delta_percent": (hybrid.statistics.median_ms / baseline.statistics.median_ms - 1.0) * 100.0,
        "limitations": [
            "synthetic activations and synthetic weight values are used",
            "weights use distinct allocations and the exact SmolLM2 projection shapes/order, but this is not full-model execution",
            "attention, normalization, elementwise kernels, KV cache, host orchestration, and session setup are excluded",
            "the result explains projection-chain behavior only and is not an end-to-end throughput claim"
        ],
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| {
            NnisError::unsupported(format!("failed to serialize projection-stream report: {error}"))
        })?
    );
    Ok(())
}
