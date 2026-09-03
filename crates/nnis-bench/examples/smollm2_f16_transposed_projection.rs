use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_model::{F16ReferenceKernels, F16TransposedProjectionCandidate};
use nnis_rt::{gpu_context, Context, DeviceBuffer, NnisError, Result, Stream};
use serde::Serialize;
use std::sync::Arc;

const HIDDEN: usize = 576;
const KV_WIDTH: usize = 192;
const INTERMEDIATE: usize = 1536;
const VOCAB: usize = 49_152;
const LAYERS: usize = 30;

#[derive(Debug, Clone, Copy)]
struct ProjectionShape {
    name: &'static str,
    k: usize,
    n: usize,
    uses_per_layer: usize,
}

#[derive(Debug, Serialize)]
struct ProjectionResult {
    name: String,
    k: usize,
    n: usize,
    uses_per_layer: usize,
    bitwise_equal: bool,
    reference: BenchmarkReport,
    transposed: BenchmarkReport,
    latency_ratio_reference_over_transposed: f64,
}

#[derive(Debug, Serialize)]
struct LmHeadResult {
    k: usize,
    n: usize,
    bitwise_equal: bool,
    reference: BenchmarkReport,
    transposed: BenchmarkReport,
    latency_ratio_reference_over_transposed: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    experiment: &'static str,
    promotion_state: &'static str,
    arithmetic_contract: &'static str,
    resident_layout_contract: &'static str,
    transpose_excluded_from_decode_timing: bool,
    resident_storage_bytes_equal: bool,
    warmups: usize,
    iterations: usize,
    layer_cases: Vec<ProjectionResult>,
    lm_head: LmHeadResult,
    weighted_projection_gpu_ms_per_decoder_token_reference: f64,
    weighted_projection_gpu_ms_per_decoder_token_transposed: f64,
    weighted_projection_latency_ratio_reference_over_transposed: f64,
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
            let value = ((index.wrapping_mul(29 + salt) + 7 * salt) % 127) as i32 - 63;
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

fn compare_projection_case(
    context: &Arc<Context>,
    stream: &Stream,
    reference: &F16ReferenceKernels,
    candidate: &F16TransposedProjectionCandidate,
    config: BenchConfig,
    shape: ProjectionShape,
) -> Result<ProjectionResult> {
    let input = narrow(
        context,
        stream,
        reference,
        &deterministic_f32(shape.k, shape.k + shape.n),
    )?;
    let weight_kn = narrow(
        context,
        stream,
        reference,
        &deterministic_f32(shape.k * shape.n, shape.n),
    )?;
    let weight_nk = DeviceBuffer::<u16>::new(context, shape.k * shape.n)?;
    let reference_output = DeviceBuffer::<u16>::new(context, shape.n)?;
    let transposed_output = DeviceBuffer::<u16>::new(context, shape.n)?;

    unsafe {
        candidate.enqueue_transpose_kn_to_nk(stream, &weight_kn, &weight_nk, shape.k, shape.n)?;
        reference.enqueue_project_kn(
            stream,
            &input,
            &weight_kn,
            &reference_output,
            shape.k,
            shape.n,
        )?;
        candidate.enqueue_project_nk(
            stream,
            &input,
            &weight_nk,
            &transposed_output,
            shape.k,
            shape.n,
        )?;
    }
    stream.synchronize()?;

    let expected = reference_output.to_vec(stream)?;
    let actual = transposed_output.to_vec(stream)?;
    if expected != actual {
        let index = expected
            .iter()
            .zip(&actual)
            .position(|(left, right)| left != right)
            .unwrap_or(0);
        return Err(NnisError::unsupported(format!(
            "{} transposed F16 projection is not bitwise equal at output {index}: 0x{:04x} != 0x{:04x}",
            shape.name, expected[index], actual[index]
        )));
    }

    let reference_case = BenchmarkCase::new(format!("{}_kn_reference", shape.name), "f16")
        .with_dimension("k", shape.k as u64)
        .with_dimension("n", shape.n as u64)
        .with_work_items((shape.k * shape.n) as u64);
    let transposed_case = BenchmarkCase::new(format!("{}_nk_transposed", shape.name), "f16")
        .with_dimension("k", shape.k as u64)
        .with_dimension("n", shape.n as u64)
        .with_work_items((shape.k * shape.n) as u64);

    let reference_report = benchmark_gpu(context, stream, reference_case, config, || unsafe {
        reference.enqueue_project_kn(
            stream,
            &input,
            &weight_kn,
            &reference_output,
            shape.k,
            shape.n,
        )
    })?;
    let transposed_report = benchmark_gpu(context, stream, transposed_case, config, || unsafe {
        candidate.enqueue_project_nk(
            stream,
            &input,
            &weight_nk,
            &transposed_output,
            shape.k,
            shape.n,
        )
    })?;

    let expected_after = reference_output.to_vec(stream)?;
    let actual_after = transposed_output.to_vec(stream)?;
    if expected_after != actual_after {
        return Err(NnisError::unsupported(format!(
            "{} transposed F16 projection drifted after timing",
            shape.name
        )));
    }

    Ok(ProjectionResult {
        name: shape.name.to_owned(),
        k: shape.k,
        n: shape.n,
        uses_per_layer: shape.uses_per_layer,
        bitwise_equal: true,
        latency_ratio_reference_over_transposed: reference_report.statistics.median_ms
            / transposed_report.statistics.median_ms,
        reference: reference_report,
        transposed: transposed_report,
    })
}

fn compare_lm_head(
    context: &Arc<Context>,
    stream: &Stream,
    reference: &F16ReferenceKernels,
    candidate: &F16TransposedProjectionCandidate,
    config: BenchConfig,
) -> Result<LmHeadResult> {
    let input = narrow(
        context,
        stream,
        reference,
        &deterministic_f32(HIDDEN, VOCAB),
    )?;
    let weight_kn = narrow(
        context,
        stream,
        reference,
        &deterministic_f32(HIDDEN * VOCAB, 17),
    )?;
    let weight_nk = DeviceBuffer::<u16>::new(context, HIDDEN * VOCAB)?;
    let reference_output = DeviceBuffer::<f32>::new(context, VOCAB)?;
    let transposed_output = DeviceBuffer::<f32>::new(context, VOCAB)?;

    unsafe {
        candidate.enqueue_transpose_kn_to_nk(stream, &weight_kn, &weight_nk, HIDDEN, VOCAB)?;
        reference.enqueue_lm_head_f32_logits(
            stream,
            &input,
            &weight_kn,
            &reference_output,
            HIDDEN,
            VOCAB,
        )?;
        candidate.enqueue_lm_head_nk_f32_logits(
            stream,
            &input,
            &weight_nk,
            &transposed_output,
            HIDDEN,
            VOCAB,
        )?;
    }
    stream.synchronize()?;

    let expected = reference_output.to_vec(stream)?;
    let actual = transposed_output.to_vec(stream)?;
    if expected.len() != actual.len()
        || expected
            .iter()
            .zip(&actual)
            .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err(NnisError::unsupported(
            "transposed F16 LM-head is not bitwise equal to reference",
        ));
    }

    let reference_case = BenchmarkCase::new("lm_head_kn_reference", "f16")
        .with_dimension("k", HIDDEN as u64)
        .with_dimension("n", VOCAB as u64)
        .with_work_items((HIDDEN * VOCAB) as u64);
    let transposed_case = BenchmarkCase::new("lm_head_nk_transposed", "f16")
        .with_dimension("k", HIDDEN as u64)
        .with_dimension("n", VOCAB as u64)
        .with_work_items((HIDDEN * VOCAB) as u64);

    let reference_report = benchmark_gpu(context, stream, reference_case, config, || unsafe {
        reference.enqueue_lm_head_f32_logits(
            stream,
            &input,
            &weight_kn,
            &reference_output,
            HIDDEN,
            VOCAB,
        )
    })?;
    let transposed_report = benchmark_gpu(context, stream, transposed_case, config, || unsafe {
        candidate.enqueue_lm_head_nk_f32_logits(
            stream,
            &input,
            &weight_nk,
            &transposed_output,
            HIDDEN,
            VOCAB,
        )
    })?;

    let expected_after = reference_output.to_vec(stream)?;
    let actual_after = transposed_output.to_vec(stream)?;
    for (index, (&left, &right)) in expected_after.iter().zip(&actual_after).enumerate() {
        if left.to_bits() != right.to_bits() {
            return Err(NnisError::unsupported(format!(
                "transposed F16 LM-head drifted after timing at output {index}"
            )));
        }
    }

    Ok(LmHeadResult {
        k: HIDDEN,
        n: VOCAB,
        bitwise_equal: true,
        latency_ratio_reference_over_transposed: reference_report.statistics.median_ms
            / transposed_report.statistics.median_ms,
        reference: reference_report,
        transposed: transposed_report,
    })
}

fn run() -> Result<()> {
    let context = gpu_context().ok_or_else(|| NnisError::unsupported("no CUDA device"))?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let reference = F16ReferenceKernels::load(&context, &compiler)?;
    let candidate = F16TransposedProjectionCandidate::load(&context, &compiler)?;
    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 100)?;
    let config = BenchConfig::new(warmups, iterations);
    config.validate()?;

    let shapes = [
        ProjectionShape {
            name: "q_o_k576_n576",
            k: HIDDEN,
            n: HIDDEN,
            uses_per_layer: 2,
        },
        ProjectionShape {
            name: "k_v_k576_n192",
            k: HIDDEN,
            n: KV_WIDTH,
            uses_per_layer: 2,
        },
        ProjectionShape {
            name: "gate_up_k576_n1536",
            k: HIDDEN,
            n: INTERMEDIATE,
            uses_per_layer: 2,
        },
        ProjectionShape {
            name: "down_k1536_n576",
            k: INTERMEDIATE,
            n: HIDDEN,
            uses_per_layer: 1,
        },
    ];

    let mut layer_cases = Vec::with_capacity(shapes.len());
    let mut reference_layer_ms = 0.0_f64;
    let mut transposed_layer_ms = 0.0_f64;
    for shape in shapes {
        let result =
            compare_projection_case(&context, &stream, &reference, &candidate, config, shape)?;
        reference_layer_ms += result.reference.statistics.median_ms * shape.uses_per_layer as f64;
        transposed_layer_ms += result.transposed.statistics.median_ms * shape.uses_per_layer as f64;
        layer_cases.push(result);
    }

    let lm_head = compare_lm_head(&context, &stream, &reference, &candidate, config)?;
    let reference_per_token =
        reference_layer_ms * LAYERS as f64 + lm_head.reference.statistics.median_ms;
    let transposed_per_token =
        transposed_layer_ms * LAYERS as f64 + lm_head.transposed.statistics.median_ms;

    let report = Report {
        schema_version: 1,
        experiment: "R1-F16-transposed-resident-projection-layout",
        promotion_state: "candidate-only; qualified F16 runtime unchanged",
        arithmetic_contract: "same 128-lane strided F32 FMA partials, same shared-memory reduction tree, same F16 output rounding",
        resident_layout_contract: "reference weights [K,N]; candidate weights one-time transposed to [N,K] with equal F16 storage bytes",
        transpose_excluded_from_decode_timing: true,
        resident_storage_bytes_equal: true,
        warmups,
        iterations,
        layer_cases,
        lm_head,
        weighted_projection_gpu_ms_per_decoder_token_reference: reference_per_token,
        weighted_projection_gpu_ms_per_decoder_token_transposed: transposed_per_token,
        weighted_projection_latency_ratio_reference_over_transposed: reference_per_token
            / transposed_per_token,
        limitations: vec![
            "isolated CUDA-event medians are not an end-to-end generation result",
            "one-time resident transposition cost is intentionally excluded from decode timing",
            "host launch overhead is not represented by kernel elapsed time",
            "runtime integration requires a separate explicit plan and exact 32-token SmolLM2 gate",
        ],
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| {
            NnisError::unsupported(format!(
                "failed to serialize F16 projection report: {error}"
            ))
        })?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("F16 transposed projection benchmark failed: {error}");
        std::process::exit(1);
    }
}
