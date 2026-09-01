use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_model::{F16CachedAttentionStagedWeightsCandidate, F16ReferenceKernels};
use nnis_rt::{gpu_context, Context, DeviceBuffer, KvCache, KvCacheConfig, NnisError, Result, Stream};
use serde::Serialize;
use std::sync::Arc;

const QUERY_HEADS: usize = 9;
const KV_HEADS: usize = 3;
const HEAD_DIM: usize = 64;
const DEFAULT_KV_ROWS: usize = 35;
const SMOLLM2_LAYERS: usize = 30;

#[derive(Debug, Serialize)]
struct ConfigReport {
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_rows: usize,
    warmups: usize,
    iterations: usize,
    reference_block_barriers: usize,
    candidate_block_barriers: usize,
    launches_per_layer_unchanged: usize,
    candidate_dynamic_shared_bytes: usize,
    smollm2_layers: usize,
}

#[derive(Debug, Serialize)]
struct CorrectnessReport {
    bitwise_equal: bool,
    f16_elements_checked: usize,
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
    arithmetic_contract: &'static str,
    config: ConfigReport,
    correctness: CorrectnessReport,
    reference: BenchmarkReport,
    candidate: BenchmarkReport,
    comparison: ComparisonReport,
    limitations: [&'static str; 6],
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
            "NNIS_BENCH_RUN_CONTEXT_ID is required for staged F16 attention evidence",
        )),
    }
}

fn deterministic_f32(
    elements: usize,
    multiplier: usize,
    modulus: usize,
    scale: f32,
) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let centered = ((index * multiplier + 11) % modulus) as i32 - (modulus as i32 / 2);
            centered as f32 * scale
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
    // SAFETY: both buffers stay alive through the immediately following sync.
    unsafe { reference.enqueue_narrow_from_f32(stream, &source, &output)? };
    stream.synchronize()?;
    Ok(output)
}

fn run() -> Result<()> {
    require_run_context()?;
    let kv_rows = env_usize("NNIS_ATTENTION_KV_ROWS", DEFAULT_KV_ROWS)?;
    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 100)?;
    if kv_rows == 0 || iterations == 0 {
        return Err(NnisError::invalid_input(
            "staged F16 attention kv_rows and iterations must be non-zero",
        ));
    }

    let context = gpu_context().ok_or_else(|| NnisError::unsupported("no CUDA device"))?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let reference = F16ReferenceKernels::load(&context, &compiler)?;
    let candidate = F16CachedAttentionStagedWeightsCandidate::load(&context, &compiler)?;

    let query_elements = QUERY_HEADS * HEAD_DIM;
    let kv_elements = KV_HEADS
        .checked_mul(kv_rows)
        .and_then(|value| value.checked_mul(HEAD_DIM))
        .ok_or_else(|| NnisError::invalid_input("staged F16 attention fixture overflow"))?;

    let query = narrow(
        &context,
        &stream,
        &reference,
        &deterministic_f32(query_elements, 17, 97, 0.015625),
    )?;
    let source_keys = Arc::new(narrow(
        &context,
        &stream,
        &reference,
        &deterministic_f32(kv_elements, 29, 113, 0.0078125),
    )?);
    let source_values = Arc::new(narrow(
        &context,
        &stream,
        &reference,
        &deterministic_f32(kv_elements, 31, 127, 0.015625),
    )?);

    let mut cache = KvCache::<u16>::new(
        &stream,
        KvCacheConfig::new(1, KV_HEADS, HEAD_DIM, kv_rows)?,
    )?;
    cache.append_layer(0, source_keys, source_values, kv_rows)?;

    let reference_output = DeviceBuffer::<u16>::new(&context, query_elements)?;
    let candidate_output = DeviceBuffer::<u16>::new(&context, query_elements)?;
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();

    // Correctness is checked before timing so a numerically different candidate
    // never produces performance evidence.
    unsafe {
        reference.enqueue_cached_attention_decode(
            &stream,
            &query,
            &cache,
            0,
            &reference_output,
            scale,
        )?;
        candidate.enqueue_cached_attention_decode(
            &stream,
            &query,
            &cache,
            0,
            &candidate_output,
            scale,
        )?;
    }
    stream.synchronize()?;
    let expected = reference_output.to_vec(&stream)?;
    let actual = candidate_output.to_vec(&stream)?;
    if expected != actual {
        let index = expected
            .iter()
            .zip(&actual)
            .position(|(left, right)| left != right)
            .unwrap_or(0);
        return Err(NnisError::unsupported(format!(
            "staged F16 attention changed output bits at element {index}: reference=0x{:04x}, candidate=0x{:04x}",
            expected[index], actual[index]
        )));
    }

    let bench_config = BenchConfig::new(warmups, iterations);
    bench_config.validate()?;
    let work_items = QUERY_HEADS
        .checked_mul(kv_rows)
        .and_then(|value| value.checked_mul(HEAD_DIM))
        .ok_or_else(|| NnisError::invalid_input("staged F16 attention work-item overflow"))?;

    let reference_report = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("smollm2_f16_attention_reference_per_position_barriers", "f16")
            .with_dimension("query_heads", QUERY_HEADS as u64)
            .with_dimension("kv_heads", KV_HEADS as u64)
            .with_dimension("head_dim", HEAD_DIM as u64)
            .with_dimension("kv_rows", kv_rows as u64)
            .with_dimension("block_barriers", (2 * kv_rows + 1) as u64)
            .with_work_items(work_items as u64),
        bench_config,
        || unsafe {
            reference.enqueue_cached_attention_decode(
                &stream,
                &query,
                &cache,
                0,
                &reference_output,
                scale,
            )
        },
    )?;

    let candidate_report = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("smollm2_f16_attention_staged_weights_candidate", "f16")
            .with_dimension("query_heads", QUERY_HEADS as u64)
            .with_dimension("kv_heads", KV_HEADS as u64)
            .with_dimension("head_dim", HEAD_DIM as u64)
            .with_dimension("kv_rows", kv_rows as u64)
            .with_dimension("block_barriers", 1)
            .with_work_items(work_items as u64),
        bench_config,
        || unsafe {
            candidate.enqueue_cached_attention_decode(
                &stream,
                &query,
                &cache,
                0,
                &candidate_output,
                scale,
            )
        },
    )?;

    reference_report
        .metadata
        .require_compatible_environment(&candidate_report.metadata)?;
    if reference_report.metadata.git_commit != candidate_report.metadata.git_commit {
        return Err(NnisError::invalid_input(
            "reference and staged F16 attention runs came from different commits",
        ));
    }
    if reference_report.metadata.git_dirty != Some(false)
        || candidate_report.metadata.git_dirty != Some(false)
    {
        return Err(NnisError::invalid_input(
            "staged F16 attention evidence requires a clean tracked worktree",
        ));
    }

    let expected_after = reference_output.to_vec(&stream)?;
    let actual_after = candidate_output.to_vec(&stream)?;
    if expected_after != actual_after {
        return Err(NnisError::unsupported(
            "staged F16 attention output drifted after timing",
        ));
    }

    let reference_median = reference_report.statistics.median_ms;
    let candidate_median = candidate_report.statistics.median_ms;
    if reference_median <= 0.0 || candidate_median <= 0.0 {
        return Err(NnisError::unsupported(
            "staged F16 attention benchmark produced a non-positive median duration",
        ));
    }

    let candidate_dynamic_shared_bytes = kv_rows
        .checked_mul(2)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| NnisError::invalid_input("staged F16 shared-memory report overflow"))?;

    let report = Report {
        schema_version: 1,
        experiment: "R1-F16-staged-attention-barrier-elimination-isolated-v1",
        promotion_state: "candidate-only; promoted F16 runtime attention unchanged",
        hypothesis: "precompute the exact serial score/online-softmax weights in lane zero, stage them in shared memory, then let all value lanes consume the full KV sequence after one block barrier instead of synchronizing twice per position",
        arithmetic_contract: "score FMA order unchanged; online-softmax order unchanged; per-output value accumulation order unchanged; F32 accumulators and final F16 rounding unchanged",
        config: ConfigReport {
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            kv_rows,
            warmups,
            iterations,
            reference_block_barriers: 2 * kv_rows + 1,
            candidate_block_barriers: 1,
            launches_per_layer_unchanged: 1,
            candidate_dynamic_shared_bytes,
            smollm2_layers: SMOLLM2_LAYERS,
        },
        correctness: CorrectnessReport {
            bitwise_equal: true,
            f16_elements_checked: query_elements,
        },
        comparison: ComparisonReport {
            reference_median_ms: reference_median,
            candidate_median_ms: candidate_median,
            candidate_over_reference_latency_ratio: candidate_median / reference_median,
            reference_over_candidate_speed_ratio: reference_median / candidate_median,
        },
        reference: reference_report,
        candidate: candidate_report,
        limitations: [
            "isolated CUDA-event attention evidence is not end-to-end generation evidence",
            "the candidate intentionally leaves the serial QK score loop unchanged",
            "the launch count is unchanged; only intra-block synchronization is reduced",
            "dynamic shared memory grows by eight bytes per valid KV row",
            "bitwise equality is required before any runtime integration is considered",
            "runtime integration requires a separate explicit versioned plan and fingerprint-compatible end-to-end verification",
        ],
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| {
            NnisError::unsupported(format!(
                "failed to serialize staged F16 attention report: {error}"
            ))
        })?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("staged F16 attention benchmark failed: {error}");
        std::process::exit(1);
    }
}
