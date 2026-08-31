use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_model::{F32CachedAttentionDecodeParallelValue, F32DecoderKernels};
use nnis_rt::{
    Context, Device, DeviceBuffer, KvCache, KvCacheConfig, NnisError, Result, Stream,
};
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
    reference_threads_per_query_head: usize,
    candidate_threads_per_query_head: usize,
    launches_per_layer_unchanged: usize,
    smollm2_layers: usize,
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
    limitations: [&'static str; 6],
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
            "NNIS_BENCH_RUN_CONTEXT_ID is required for cached-attention evidence",
        )),
    }
}

fn deterministic_values(elements: usize, multiplier: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let centered = ((index * multiplier + 11) % modulus) as i32 - (modulus as i32 / 2);
            centered as f32 * scale
        })
        .collect()
}

fn main() -> Result<()> {
    require_run_context()?;
    let kv_rows = env_usize("NNIS_ATTENTION_KV_ROWS", DEFAULT_KV_ROWS)?;
    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 100)?;
    if kv_rows == 0 || iterations == 0 {
        return Err(NnisError::invalid_input(
            "cached-attention kv_rows and iterations must be non-zero",
        ));
    }

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let reference_kernel = F32DecoderKernels::load(&context, &compiler)?;
    let candidate_kernel = F32CachedAttentionDecodeParallelValue::load(&context, &compiler)?;

    let query_elements = QUERY_HEADS * HEAD_DIM;
    let kv_elements = KV_HEADS
        .checked_mul(kv_rows)
        .and_then(|value| value.checked_mul(HEAD_DIM))
        .ok_or_else(|| NnisError::invalid_input("cached-attention fixture shape overflow"))?;
    let query_host = deterministic_values(query_elements, 17, 97, 0.015625);
    let keys_host = deterministic_values(kv_elements, 29, 113, 0.0078125);
    let values_host = deterministic_values(kv_elements, 31, 127, 0.015625);

    let query = DeviceBuffer::from_host(&context, &stream, &query_host)?;
    let source_keys = Arc::new(DeviceBuffer::from_host(&context, &stream, &keys_host)?);
    let source_values = Arc::new(DeviceBuffer::from_host(&context, &stream, &values_host)?);
    let mut cache = KvCache::new(
        &stream,
        KvCacheConfig::new(1, KV_HEADS, HEAD_DIM, kv_rows)?,
    )?;
    cache.append_layer(0, source_keys, source_values, kv_rows)?;

    let reference_output = DeviceBuffer::<f32>::new(&context, query_elements)?;
    let candidate_output = DeviceBuffer::<f32>::new(&context, query_elements)?;
    let bench_config = BenchConfig::new(warmups, iterations);
    bench_config.validate()?;
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let work_items = QUERY_HEADS
        .checked_mul(kv_rows)
        .and_then(|value| value.checked_mul(HEAD_DIM))
        .ok_or_else(|| NnisError::invalid_input("cached-attention work-item overflow"))?;

    let reference = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("smollm2_cached_attention_single_thread_reference", "f32")
            .with_dimension("query_heads", QUERY_HEADS as u64)
            .with_dimension("kv_heads", KV_HEADS as u64)
            .with_dimension("head_dim", HEAD_DIM as u64)
            .with_dimension("kv_rows", kv_rows as u64)
            .with_dimension("threads_per_query_head", 1)
            .with_work_items(work_items as u64),
        bench_config,
        || {
            // SAFETY: the cache, buffers, kernel set and stream outlive each
            // measured launch; the harness supplies the synchronization boundary.
            unsafe {
                reference_kernel.enqueue_cached_attention_decode(
                    &stream,
                    &query,
                    &cache,
                    0,
                    &reference_output,
                    scale,
                )
            }
        },
    )?;

    let candidate = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("smollm2_cached_attention_parallel_value_candidate", "f32")
            .with_dimension("query_heads", QUERY_HEADS as u64)
            .with_dimension("kv_heads", KV_HEADS as u64)
            .with_dimension("head_dim", HEAD_DIM as u64)
            .with_dimension("kv_rows", kv_rows as u64)
            .with_dimension("threads_per_query_head", HEAD_DIM as u64)
            .with_work_items(work_items as u64),
        bench_config,
        || {
            // SAFETY: the cache, buffers, candidate kernel and stream outlive
            // each measured launch; the harness synchronizes after the end event.
            unsafe {
                candidate_kernel.enqueue_cached_attention_decode(
                    &stream,
                    &query,
                    &cache,
                    0,
                    &candidate_output,
                    scale,
                )
            }
        },
    )?;

    reference
        .metadata
        .require_compatible_environment(&candidate.metadata)?;
    if reference.metadata.git_commit != candidate.metadata.git_commit {
        return Err(NnisError::invalid_input(
            "reference and candidate attention runs came from different commits",
        ));
    }
    if reference.metadata.git_dirty != Some(false) || candidate.metadata.git_dirty != Some(false) {
        return Err(NnisError::invalid_input(
            "cached-attention evidence requires a clean tracked worktree",
        ));
    }

    let reference_host = reference_output.to_vec(&stream)?;
    let candidate_host = candidate_output.to_vec(&stream)?;
    for (index, (&reference_value, &candidate_value)) in
        reference_host.iter().zip(&candidate_host).enumerate()
    {
        if reference_value.to_bits() != candidate_value.to_bits() {
            return Err(NnisError::invalid_input(format!(
                "parallel-value attention changed f32 bits at element {index}: reference=0x{:08x}, candidate=0x{:08x}",
                reference_value.to_bits(),
                candidate_value.to_bits()
            )));
        }
    }

    let reference_median = reference.statistics.median_ms;
    let candidate_median = candidate.statistics.median_ms;
    if reference_median <= 0.0 || candidate_median <= 0.0 {
        return Err(NnisError::unsupported(
            "cached-attention benchmark produced a non-positive median duration",
        ));
    }

    let report = Report {
        schema_version: 1,
        experiment: "R2-smollm2-cached-attention-parallel-value-isolated-v1",
        promotion_state: "candidate-only; production decoder attention remains unchanged",
        hypothesis: "keep the exact serial score/online-softmax chain on lane zero while parallelizing the 64 independent value/output dimensions across one block",
        config: ConfigReport {
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            kv_rows,
            warmups,
            iterations,
            reference_threads_per_query_head: 1,
            candidate_threads_per_query_head: HEAD_DIM,
            launches_per_layer_unchanged: 1,
            smollm2_layers: SMOLLM2_LAYERS,
        },
        correctness: CorrectnessReport {
            bitwise_equal: true,
            elements_checked: query_elements,
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
            "this is isolated cached-attention evidence, not end-to-end decoder evidence",
            "the default fixture uses kv_rows=35 because the qualifying SmolLM2 profile covered three prompt positions plus 32 decode steps",
            "the candidate intentionally leaves score FMA order and online-softmax sequencing serial rather than claiming a parallel reduction",
            "bitwise equality is required for this deterministic fixture before any runtime integration is considered",
            "the launch count is unchanged; the candidate only changes intra-block parallelism",
            "runtime integration requires an explicit versioned plan and fingerprint-compatible end-to-end verification in a later change",
        ],
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| NnisError::invalid_input(
            format!("serialize cached-attention report: {error}")
        ))?
    );
    Ok(())
}
