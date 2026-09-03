use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_model::{
    F16FusedProjectionGroupsCandidate, F16ReferenceKernels, F16TransposedProjectionCandidate,
};
use nnis_rt::{gpu_context, Context, DeviceBuffer, NnisError, Result, Stream};
use serde::Serialize;
use std::sync::Arc;

const HIDDEN: usize = 576;
const KV_WIDTH: usize = 192;
const INTERMEDIATE: usize = 1536;
const LAYERS: usize = 30;

#[derive(Debug, Serialize)]
struct GroupResult {
    name: &'static str,
    sequential_launches: usize,
    fused_launches: usize,
    bitwise_equal: bool,
    sequential: BenchmarkReport,
    fused: BenchmarkReport,
    latency_ratio_sequential_over_fused: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    experiment: &'static str,
    promotion_state: &'static str,
    arithmetic_contract: &'static str,
    resident_layout_contract: &'static str,
    warmups: usize,
    iterations: usize,
    qkv: GroupResult,
    gate_up: GroupResult,
    projection_launches_per_layer_sequential: usize,
    projection_launches_per_layer_fused: usize,
    projection_launches_removed_per_layer: usize,
    projection_launches_removed_per_decoder_token: usize,
    grouped_projection_gpu_ms_per_decoder_token_sequential: f64,
    grouped_projection_gpu_ms_per_decoder_token_fused: f64,
    grouped_projection_latency_ratio_sequential_over_fused: f64,
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

fn deterministic_f32(elements: usize, salt: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let value = ((index.wrapping_mul(31 + salt) + 11 * salt) % 127) as i32 - 63;
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
    k: usize,
    n: usize,
    salt: usize,
) -> Result<DeviceBuffer<u16>> {
    let kn = narrow(context, stream, reference, &deterministic_f32(k * n, salt))?;
    let nk = DeviceBuffer::<u16>::new(context, k * n)?;
    unsafe { transposed.enqueue_transpose_kn_to_nk(stream, &kn, &nk, k, n)? };
    stream.synchronize()?;
    Ok(nk)
}

fn require_equal(name: &str, left: &[u16], right: &[u16]) -> Result<()> {
    if left == right {
        return Ok(());
    }
    let index = left
        .iter()
        .zip(right)
        .position(|(a, b)| a != b)
        .unwrap_or(0);
    Err(NnisError::unsupported(format!(
        "{name} fused projection is not bitwise equal at output {index}: 0x{:04x} != 0x{:04x}",
        left[index], right[index]
    )))
}

fn benchmark_qkv(
    context: &Arc<Context>,
    stream: &Stream,
    reference: &F16ReferenceKernels,
    transposed: &F16TransposedProjectionCandidate,
    fused: &F16FusedProjectionGroupsCandidate,
    config: BenchConfig,
) -> Result<GroupResult> {
    let input = narrow(context, stream, reference, &deterministic_f32(HIDDEN, 1))?;
    let q_weight = prepare_nk_weight(context, stream, reference, transposed, HIDDEN, HIDDEN, 2)?;
    let k_weight = prepare_nk_weight(context, stream, reference, transposed, HIDDEN, KV_WIDTH, 3)?;
    let v_weight = prepare_nk_weight(context, stream, reference, transposed, HIDDEN, KV_WIDTH, 4)?;

    let q_sequential = DeviceBuffer::<u16>::new(context, HIDDEN)?;
    let k_sequential = DeviceBuffer::<u16>::new(context, KV_WIDTH)?;
    let v_sequential = DeviceBuffer::<u16>::new(context, KV_WIDTH)?;
    let q_fused = DeviceBuffer::<u16>::new(context, HIDDEN)?;
    let k_fused = DeviceBuffer::<u16>::new(context, KV_WIDTH)?;
    let v_fused = DeviceBuffer::<u16>::new(context, KV_WIDTH)?;

    unsafe {
        transposed.enqueue_project_nk(stream, &input, &q_weight, &q_sequential, HIDDEN, HIDDEN)?;
        transposed.enqueue_project_nk(
            stream,
            &input,
            &k_weight,
            &k_sequential,
            HIDDEN,
            KV_WIDTH,
        )?;
        transposed.enqueue_project_nk(
            stream,
            &input,
            &v_weight,
            &v_sequential,
            HIDDEN,
            KV_WIDTH,
        )?;
        fused.enqueue_qkv_nk(
            stream, &input, &q_weight, &k_weight, &v_weight, &q_fused, &k_fused, &v_fused, HIDDEN,
            HIDDEN, KV_WIDTH,
        )?;
    }
    stream.synchronize()?;
    require_equal("q", &q_sequential.to_vec(stream)?, &q_fused.to_vec(stream)?)?;
    require_equal("k", &k_sequential.to_vec(stream)?, &k_fused.to_vec(stream)?)?;
    require_equal("v", &v_sequential.to_vec(stream)?, &v_fused.to_vec(stream)?)?;

    let sequential_case = BenchmarkCase::new("smollm2_qkv_nk_sequential", "f16")
        .with_dimension("hidden", HIDDEN as u64)
        .with_dimension("kv_width", KV_WIDTH as u64)
        .with_work_items((HIDDEN * HIDDEN + 2 * HIDDEN * KV_WIDTH) as u64);
    let fused_case = BenchmarkCase::new("smollm2_qkv_nk_fused", "f16")
        .with_dimension("hidden", HIDDEN as u64)
        .with_dimension("kv_width", KV_WIDTH as u64)
        .with_work_items((HIDDEN * HIDDEN + 2 * HIDDEN * KV_WIDTH) as u64);

    let sequential_report = benchmark_gpu(context, stream, sequential_case, config, || unsafe {
        transposed.enqueue_project_nk(stream, &input, &q_weight, &q_sequential, HIDDEN, HIDDEN)?;
        transposed.enqueue_project_nk(
            stream,
            &input,
            &k_weight,
            &k_sequential,
            HIDDEN,
            KV_WIDTH,
        )?;
        transposed.enqueue_project_nk(stream, &input, &v_weight, &v_sequential, HIDDEN, KV_WIDTH)
    })?;
    let fused_report = benchmark_gpu(context, stream, fused_case, config, || unsafe {
        fused.enqueue_qkv_nk(
            stream, &input, &q_weight, &k_weight, &v_weight, &q_fused, &k_fused, &v_fused, HIDDEN,
            HIDDEN, KV_WIDTH,
        )
    })?;

    require_equal("q", &q_sequential.to_vec(stream)?, &q_fused.to_vec(stream)?)?;
    require_equal("k", &k_sequential.to_vec(stream)?, &k_fused.to_vec(stream)?)?;
    require_equal("v", &v_sequential.to_vec(stream)?, &v_fused.to_vec(stream)?)?;

    Ok(GroupResult {
        name: "qkv",
        sequential_launches: 3,
        fused_launches: 1,
        bitwise_equal: true,
        latency_ratio_sequential_over_fused: sequential_report.statistics.median_ms
            / fused_report.statistics.median_ms,
        sequential: sequential_report,
        fused: fused_report,
    })
}

fn benchmark_gate_up(
    context: &Arc<Context>,
    stream: &Stream,
    reference: &F16ReferenceKernels,
    transposed: &F16TransposedProjectionCandidate,
    fused: &F16FusedProjectionGroupsCandidate,
    config: BenchConfig,
) -> Result<GroupResult> {
    let input = narrow(context, stream, reference, &deterministic_f32(HIDDEN, 7))?;
    let gate_weight = prepare_nk_weight(
        context,
        stream,
        reference,
        transposed,
        HIDDEN,
        INTERMEDIATE,
        8,
    )?;
    let up_weight = prepare_nk_weight(
        context,
        stream,
        reference,
        transposed,
        HIDDEN,
        INTERMEDIATE,
        9,
    )?;

    let gate_sequential = DeviceBuffer::<u16>::new(context, INTERMEDIATE)?;
    let up_sequential = DeviceBuffer::<u16>::new(context, INTERMEDIATE)?;
    let gate_fused = DeviceBuffer::<u16>::new(context, INTERMEDIATE)?;
    let up_fused = DeviceBuffer::<u16>::new(context, INTERMEDIATE)?;

    unsafe {
        transposed.enqueue_project_nk(
            stream,
            &input,
            &gate_weight,
            &gate_sequential,
            HIDDEN,
            INTERMEDIATE,
        )?;
        transposed.enqueue_project_nk(
            stream,
            &input,
            &up_weight,
            &up_sequential,
            HIDDEN,
            INTERMEDIATE,
        )?;
        fused.enqueue_gate_up_nk(
            stream,
            &input,
            &gate_weight,
            &up_weight,
            &gate_fused,
            &up_fused,
            HIDDEN,
            INTERMEDIATE,
        )?;
    }
    stream.synchronize()?;
    require_equal(
        "gate",
        &gate_sequential.to_vec(stream)?,
        &gate_fused.to_vec(stream)?,
    )?;
    require_equal(
        "up",
        &up_sequential.to_vec(stream)?,
        &up_fused.to_vec(stream)?,
    )?;

    let sequential_case = BenchmarkCase::new("smollm2_gate_up_nk_sequential", "f16")
        .with_dimension("hidden", HIDDEN as u64)
        .with_dimension("intermediate", INTERMEDIATE as u64)
        .with_work_items((2 * HIDDEN * INTERMEDIATE) as u64);
    let fused_case = BenchmarkCase::new("smollm2_gate_up_nk_fused", "f16")
        .with_dimension("hidden", HIDDEN as u64)
        .with_dimension("intermediate", INTERMEDIATE as u64)
        .with_work_items((2 * HIDDEN * INTERMEDIATE) as u64);

    let sequential_report = benchmark_gpu(context, stream, sequential_case, config, || unsafe {
        transposed.enqueue_project_nk(
            stream,
            &input,
            &gate_weight,
            &gate_sequential,
            HIDDEN,
            INTERMEDIATE,
        )?;
        transposed.enqueue_project_nk(
            stream,
            &input,
            &up_weight,
            &up_sequential,
            HIDDEN,
            INTERMEDIATE,
        )
    })?;
    let fused_report = benchmark_gpu(context, stream, fused_case, config, || unsafe {
        fused.enqueue_gate_up_nk(
            stream,
            &input,
            &gate_weight,
            &up_weight,
            &gate_fused,
            &up_fused,
            HIDDEN,
            INTERMEDIATE,
        )
    })?;

    require_equal(
        "gate",
        &gate_sequential.to_vec(stream)?,
        &gate_fused.to_vec(stream)?,
    )?;
    require_equal(
        "up",
        &up_sequential.to_vec(stream)?,
        &up_fused.to_vec(stream)?,
    )?;

    Ok(GroupResult {
        name: "gate_up",
        sequential_launches: 2,
        fused_launches: 1,
        bitwise_equal: true,
        latency_ratio_sequential_over_fused: sequential_report.statistics.median_ms
            / fused_report.statistics.median_ms,
        sequential: sequential_report,
        fused: fused_report,
    })
}

fn run() -> Result<()> {
    let context = gpu_context().ok_or_else(|| NnisError::unsupported("no CUDA device"))?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let reference = F16ReferenceKernels::load(&context, &compiler)?;
    let transposed = F16TransposedProjectionCandidate::load(&context, &compiler)?;
    let fused = F16FusedProjectionGroupsCandidate::load(&context, &compiler)?;
    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 100)?;
    let config = BenchConfig::new(warmups, iterations);
    config.validate()?;

    let qkv = benchmark_qkv(&context, &stream, &reference, &transposed, &fused, config)?;
    let gate_up = benchmark_gate_up(&context, &stream, &reference, &transposed, &fused, config)?;

    let sequential_per_layer =
        qkv.sequential.statistics.median_ms + gate_up.sequential.statistics.median_ms;
    let fused_per_layer = qkv.fused.statistics.median_ms + gate_up.fused.statistics.median_ms;
    let sequential_per_token = sequential_per_layer * LAYERS as f64;
    let fused_per_token = fused_per_layer * LAYERS as f64;

    let report = Report {
        schema_version: 1,
        experiment: "R1-F16-fused-projection-launch-groups",
        promotion_state: "candidate-only; current transposed execution plan unchanged",
        arithmetic_contract: "same 128-lane per-output F32 FMA partition, same shared-memory reduction tree, same F16 rounding; only QKV and gate/up launch envelopes are grouped",
        resident_layout_contract: "all candidate weights remain resident [N,K] F16 with unchanged storage bytes",
        warmups,
        iterations,
        qkv,
        gate_up,
        projection_launches_per_layer_sequential: 5,
        projection_launches_per_layer_fused: 2,
        projection_launches_removed_per_layer: 3,
        projection_launches_removed_per_decoder_token: 3 * LAYERS,
        grouped_projection_gpu_ms_per_decoder_token_sequential: sequential_per_token,
        grouped_projection_gpu_ms_per_decoder_token_fused: fused_per_token,
        grouped_projection_latency_ratio_sequential_over_fused: sequential_per_token
            / fused_per_token,
        limitations: vec![
            "isolated CUDA-event medians do not establish end-to-end generation speedup",
            "Q/O/down and LM-head projections are unchanged",
            "attention and all non-projection kernels are unchanged",
            "runtime integration requires a separate explicit execution-plan gate",
            "physical Thor qualification is required before integration",
        ],
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| {
            NnisError::unsupported(format!(
                "failed to serialize fused projection group report: {error}"
            ))
        })?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("F16 fused projection group benchmark failed: {error}");
        std::process::exit(1);
    }
}
