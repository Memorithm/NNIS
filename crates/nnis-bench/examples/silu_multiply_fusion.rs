use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::F32Elementwise;
use nnis_model::{F32DecoderKernels, F32SiluMultiply};
use nnis_rt::{Context, Device, DeviceBuffer, NnisError, Result, Stream};
use serde::Serialize;

const SMOLLM2_INTERMEDIATE_SIZE: usize = 1_536;
const SMOLLM2_LAYERS: usize = 30;
const REFERENCE_LOGICAL_BYTES_PER_ELEMENT: u64 = 20;
const CANDIDATE_LOGICAL_BYTES_PER_ELEMENT: u64 = 12;

#[derive(Debug, Serialize)]
struct ConfigReport {
    elements: usize,
    warmups: usize,
    iterations: usize,
    smollm2_intermediate_size: usize,
    smollm2_layers: usize,
    reference_launches_per_layer: usize,
    candidate_launches_per_layer: usize,
    launches_removed_per_layer_if_integrated: usize,
    launches_removed_per_token_if_integrated: usize,
    logical_intermediate_bytes_avoided_per_layer_if_integrated: u64,
    logical_intermediate_bytes_avoided_per_token_if_integrated: u64,
}

#[derive(Debug, Serialize)]
struct CorrectnessReport {
    bitwise_equal: bool,
    elements_checked: usize,
}

#[derive(Debug, Serialize)]
struct ComparisonReport {
    reference_median_ms: f64,
    candidate_median_ms: f64,
    candidate_over_reference_latency_ratio: f64,
    reference_over_candidate_speed_ratio: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    experiment: &'static str,
    promotion_state: &'static str,
    hypothesis: &'static str,
    config: ConfigReport,
    correctness: CorrectnessReport,
    reference: BenchmarkReport,
    candidate: BenchmarkReport,
    comparison: ComparisonReport,
    limitations: [&'static str; 5],
}

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|error| NnisError::invalid_input(format!("invalid {name}: {error}"))),
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
            "NNIS_BENCH_RUN_CONTEXT_ID is required for fusion evidence",
        )),
    }
}

fn deterministic_inputs(elements: usize) -> (Vec<f32>, Vec<f32>) {
    let gate = (0..elements)
        .map(|index| ((index % 257) as f32 - 128.0) * 0.03125)
        .collect::<Vec<_>>();
    let up = (0..elements)
        .map(|index| ((index % 193) as f32 - 96.0) * 0.015625)
        .collect::<Vec<_>>();
    (gate, up)
}

fn main() -> Result<()> {
    require_run_context()?;
    let elements = env_usize("NNIS_FUSION_ELEMENTS", SMOLLM2_INTERMEDIATE_SIZE)?;
    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 100)?;
    if elements == 0 || iterations == 0 {
        return Err(NnisError::invalid_input(
            "fusion benchmark elements and iterations must be non-zero",
        ));
    }

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let elementwise = F32Elementwise::load(&context, &compiler)?;
    let decoder = F32DecoderKernels::load(&context, &compiler)?;
    let fused = F32SiluMultiply::load(&context, &compiler)?;

    let (gate_host, up_host) = deterministic_inputs(elements);
    let gate = DeviceBuffer::from_host(&context, &stream, &gate_host)?;
    let up = DeviceBuffer::from_host(&context, &stream, &up_host)?;
    let activated = DeviceBuffer::<f32>::new(&context, elements)?;
    let reference_output = DeviceBuffer::<f32>::new(&context, elements)?;
    let candidate_output = DeviceBuffer::<f32>::new(&context, elements)?;
    let config = BenchConfig::new(warmups, iterations);

    let reference_bytes = (elements as u64)
        .checked_mul(REFERENCE_LOGICAL_BYTES_PER_ELEMENT)
        .ok_or_else(|| NnisError::invalid_input("reference byte accounting overflow"))?;
    let candidate_bytes = (elements as u64)
        .checked_mul(CANDIDATE_LOGICAL_BYTES_PER_ELEMENT)
        .ok_or_else(|| NnisError::invalid_input("candidate byte accounting overflow"))?;

    let reference = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("smollm2_silu_then_multiply_reference", "f32")
            .with_dimension("elements", elements as u64)
            .with_dimension("launches", 2)
            .with_work_items(elements as u64)
            .with_bytes_per_iteration(reference_bytes),
        config,
        || {
            // SAFETY: captured buffers and kernels outlive each measured sequence;
            // the benchmark harness provides the synchronization boundary.
            unsafe {
                elementwise.enqueue_silu(&stream, &gate, &activated)?;
                decoder.enqueue_multiply(&stream, &activated, &up, &reference_output)
            }
        },
    )?;

    let candidate = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("smollm2_fused_silu_multiply_candidate", "f32")
            .with_dimension("elements", elements as u64)
            .with_dimension("launches", 1)
            .with_work_items(elements as u64)
            .with_bytes_per_iteration(candidate_bytes),
        config,
        || {
            // SAFETY: captured buffers and kernel outlive each measured launch;
            // the benchmark harness provides the synchronization boundary.
            unsafe { fused.enqueue_silu_multiply(&stream, &gate, &up, &candidate_output) }
        },
    )?;

    reference
        .metadata
        .require_compatible_environment(&candidate.metadata)?;
    if reference.metadata.git_commit != candidate.metadata.git_commit {
        return Err(NnisError::invalid_input(
            "reference and candidate were not measured from the same git commit",
        ));
    }
    if reference.metadata.git_dirty != Some(false) || candidate.metadata.git_dirty != Some(false) {
        return Err(NnisError::invalid_input(
            "fusion evidence requires a clean tracked worktree",
        ));
    }

    let reference_host = reference_output.to_vec(&stream)?;
    let candidate_host = candidate_output.to_vec(&stream)?;
    for (index, (&reference_value, &candidate_value)) in
        reference_host.iter().zip(&candidate_host).enumerate()
    {
        if reference_value.to_bits() != candidate_value.to_bits() {
            return Err(NnisError::invalid_input(format!(
                "fused SiLU-multiply changed f32 bits at element {index}: reference=0x{:08x}, candidate=0x{:08x}",
                reference_value.to_bits(),
                candidate_value.to_bits()
            )));
        }
    }

    let reference_median = reference.statistics.median_ms;
    let candidate_median = candidate.statistics.median_ms;
    if reference_median <= 0.0 || candidate_median <= 0.0 {
        return Err(NnisError::unsupported(
            "fusion benchmark produced a non-positive median duration",
        ));
    }

    let avoided_per_layer = (elements as u64)
        .checked_mul(
            REFERENCE_LOGICAL_BYTES_PER_ELEMENT - CANDIDATE_LOGICAL_BYTES_PER_ELEMENT,
        )
        .ok_or_else(|| NnisError::invalid_input("avoided-byte accounting overflow"))?;
    let avoided_per_token = avoided_per_layer
        .checked_mul(SMOLLM2_LAYERS as u64)
        .ok_or_else(|| NnisError::invalid_input("per-token avoided-byte accounting overflow"))?;

    let report = Report {
        schema_version: 1,
        experiment: "R2-smollm2-silu-multiply-fusion-isolated-v1",
        promotion_state: "candidate-only; decoder runtime remains unchanged",
        hypothesis: "fusing SiLU(gate) and multiplication by up removes one launch and the logical activated-buffer write/read per decoder layer",
        config: ConfigReport {
            elements,
            warmups,
            iterations,
            smollm2_intermediate_size: SMOLLM2_INTERMEDIATE_SIZE,
            smollm2_layers: SMOLLM2_LAYERS,
            reference_launches_per_layer: 2,
            candidate_launches_per_layer: 1,
            launches_removed_per_layer_if_integrated: 1,
            launches_removed_per_token_if_integrated: SMOLLM2_LAYERS,
            logical_intermediate_bytes_avoided_per_layer_if_integrated: avoided_per_layer,
            logical_intermediate_bytes_avoided_per_token_if_integrated: avoided_per_token,
        },
        correctness: CorrectnessReport {
            bitwise_equal: true,
            elements_checked: elements,
        },
        comparison: ComparisonReport {
            reference_median_ms: reference_median,
            candidate_median_ms: candidate_median,
            candidate_over_reference_latency_ratio: candidate_median / reference_median,
            reference_over_candidate_speed_ratio: reference_median / candidate_median,
        },
        reference,
        candidate,
        limitations: [
            "isolated elementwise evidence is not end-to-end decoder evidence",
            "logical byte accounting describes explicit buffer accesses, not measured DRAM traffic",
            "the current decoder workspace still allocates the activated buffer because runtime integration is intentionally deferred",
            "launch-count reduction is a structural count and must not be converted into an end-to-end speedup estimate",
            "runtime integration requires a separate opt-in candidate and fingerprint-compatible end-to-end verification",
        ],
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| NnisError::invalid_input(format!("serialize fusion report: {error}")))?
    );
    Ok(())
}
