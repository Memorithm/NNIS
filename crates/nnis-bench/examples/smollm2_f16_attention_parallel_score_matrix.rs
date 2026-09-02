use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_model::{
    F16CachedAttentionParallelScoreCandidate, F16CachedAttentionStagedWeightsCandidate,
    F16ReferenceKernels,
};
use nnis_rt::{Context, Device, DeviceBuffer, KvCache, KvCacheConfig, NnisError, Result, Stream};
use serde::Serialize;
use std::sync::Arc;

const QUERY_HEADS: usize = 9;
const KV_HEADS: usize = 3;
const HEAD_DIM: usize = 64;
const KV_ROWS_MATRIX: [usize; 5] = [4, 8, 16, 24, 35];
const BLOCK_MATRIX: [u32; 4] = [64, 128, 256, 512];
const SMOLLM2_LAYERS: usize = 30;

#[derive(Debug, Serialize)]
struct CorrectnessReport {
    bitwise_equal: bool,
    differing_elements: usize,
    elements_checked: usize,
    max_absolute_error: f32,
    max_relative_error: f32,
}

#[derive(Debug, Serialize)]
struct ComparisonReport {
    reference_median_ms: f64,
    candidate_median_ms: f64,
    candidate_over_reference_latency_ratio: f64,
    reference_over_candidate_speed_ratio: f64,
    relative_improvement: f64,
}

#[derive(Debug, Serialize)]
struct CandidateReport {
    threads_per_block: u32,
    correctness: CorrectnessReport,
    benchmark: BenchmarkReport,
    comparison: ComparisonReport,
}

#[derive(Debug, Serialize)]
struct StagedReport {
    correctness: CorrectnessReport,
    benchmark: BenchmarkReport,
    comparison: ComparisonReport,
}

#[derive(Debug, Serialize)]
struct ShapeReport {
    kv_rows: usize,
    reference: BenchmarkReport,
    staged_weights: StagedReport,
    parallel_score_candidates: Vec<CandidateReport>,
    best_bitwise_parallel_score_block: Option<u32>,
    best_bitwise_parallel_score_speed_ratio: Option<f64>,
    best_bitwise_parallel_score_relative_improvement: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    experiment: &'static str,
    promotion_state: &'static str,
    hypothesis: &'static str,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    smollm2_layers: usize,
    warmups: usize,
    iterations: usize,
    kv_rows_matrix: [usize; 5],
    block_matrix: [u32; 4],
    shapes: Vec<ShapeReport>,
    limitations: [&'static str; 7],
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
            "NNIS_BENCH_RUN_CONTEXT_ID is required for F16 attention evidence",
        )),
    }
}

fn deterministic_values(
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

fn require_compatible(reference: &BenchmarkReport, candidate: &BenchmarkReport) -> Result<()> {
    reference
        .metadata
        .require_compatible_environment(&candidate.metadata)?;
    if reference.metadata.git_commit != candidate.metadata.git_commit {
        return Err(NnisError::invalid_input(
            "F16 attention benchmark reports came from different Git commits",
        ));
    }
    if reference.metadata.git_dirty != Some(false) || candidate.metadata.git_dirty != Some(false) {
        return Err(NnisError::invalid_input(
            "F16 attention evidence requires a clean tracked worktree",
        ));
    }
    Ok(())
}

fn compare_outputs(
    stream: &Stream,
    kernels: &F16ReferenceKernels,
    reference: &DeviceBuffer<u16>,
    candidate: &DeviceBuffer<u16>,
) -> Result<CorrectnessReport> {
    stream.synchronize()?;
    let reference_bits = reference.to_vec(stream)?;
    let candidate_bits = candidate.to_vec(stream)?;
    if reference_bits.len() != candidate_bits.len() {
        return Err(NnisError::invalid_input(
            "F16 attention correctness buffers differ in length",
        ));
    }

    let reference_f32 = DeviceBuffer::<f32>::new(stream.ctx(), reference.len())?;
    let candidate_f32 = DeviceBuffer::<f32>::new(stream.ctx(), candidate.len())?;
    unsafe {
        kernels.enqueue_widen_to_f32(stream, reference, &reference_f32)?;
        kernels.enqueue_widen_to_f32(stream, candidate, &candidate_f32)?;
    }
    stream.synchronize()?;
    let reference_values = reference_f32.to_vec(stream)?;
    let candidate_values = candidate_f32.to_vec(stream)?;

    let mut differing_elements = 0_usize;
    let mut max_absolute_error = 0.0_f32;
    let mut max_relative_error = 0.0_f32;
    for index in 0..reference_bits.len() {
        if reference_bits[index] != candidate_bits[index] {
            differing_elements += 1;
        }
        let reference_value = reference_values[index];
        let candidate_value = candidate_values[index];
        if !reference_value.is_finite() || !candidate_value.is_finite() {
            return Err(NnisError::invalid_input(format!(
                "F16 attention produced a non-finite value at element {index}: reference={reference_value}, candidate={candidate_value}"
            )));
        }
        let absolute = (reference_value - candidate_value).abs();
        let relative = absolute / reference_value.abs().max(1.0e-12);
        max_absolute_error = max_absolute_error.max(absolute);
        max_relative_error = max_relative_error.max(relative);
    }

    Ok(CorrectnessReport {
        bitwise_equal: differing_elements == 0,
        differing_elements,
        elements_checked: reference_bits.len(),
        max_absolute_error,
        max_relative_error,
    })
}

fn comparison(
    reference: &BenchmarkReport,
    candidate: &BenchmarkReport,
) -> Result<ComparisonReport> {
    let reference_median = reference.statistics.median_ms;
    let candidate_median = candidate.statistics.median_ms;
    if reference_median <= 0.0 || candidate_median <= 0.0 {
        return Err(NnisError::unsupported(
            "F16 attention benchmark produced a non-positive median duration",
        ));
    }
    Ok(ComparisonReport {
        reference_median_ms: reference_median,
        candidate_median_ms: candidate_median,
        candidate_over_reference_latency_ratio: candidate_median / reference_median,
        reference_over_candidate_speed_ratio: reference_median / candidate_median,
        relative_improvement: (reference_median - candidate_median) / reference_median,
    })
}

fn run_shape(
    context: &Arc<Context>,
    stream: &Stream,
    reference_kernel: &F16ReferenceKernels,
    staged_kernel: &F16CachedAttentionStagedWeightsCandidate,
    parallel_score_kernel: &F16CachedAttentionParallelScoreCandidate,
    kv_rows: usize,
    bench_config: BenchConfig,
) -> Result<ShapeReport> {
    let query_elements = QUERY_HEADS * HEAD_DIM;
    let kv_elements = KV_HEADS
        .checked_mul(kv_rows)
        .and_then(|value| value.checked_mul(HEAD_DIM))
        .ok_or_else(|| NnisError::invalid_input("F16 attention fixture shape overflow"))?;
    let query_host = deterministic_values(query_elements, 17, 97, 0.015625);
    let keys_host = deterministic_values(kv_elements, 29, 113, 0.0078125);
    let values_host = deterministic_values(kv_elements, 31, 127, 0.015625);

    let query_source = DeviceBuffer::from_host(context, stream, &query_host)?;
    let keys_source = DeviceBuffer::from_host(context, stream, &keys_host)?;
    let values_source = DeviceBuffer::from_host(context, stream, &values_host)?;
    let query = DeviceBuffer::<u16>::new(context, query_elements)?;
    let keys = Arc::new(DeviceBuffer::<u16>::new(context, kv_elements)?);
    let values = Arc::new(DeviceBuffer::<u16>::new(context, kv_elements)?);
    unsafe {
        reference_kernel.enqueue_narrow_from_f32(stream, &query_source, &query)?;
        reference_kernel.enqueue_narrow_from_f32(stream, &keys_source, &keys)?;
        reference_kernel.enqueue_narrow_from_f32(stream, &values_source, &values)?;
    }
    stream.synchronize()?;

    let mut cache = KvCache::new(stream, KvCacheConfig::new(1, KV_HEADS, HEAD_DIM, kv_rows)?)?;
    cache.append_layer(0, Arc::clone(&keys), Arc::clone(&values), kv_rows)?;

    let reference_output = DeviceBuffer::<u16>::new(context, query_elements)?;
    let staged_output = DeviceBuffer::<u16>::new(context, query_elements)?;
    let parallel_output = DeviceBuffer::<u16>::new(context, query_elements)?;
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let work_items = QUERY_HEADS
        .checked_mul(kv_rows)
        .and_then(|value| value.checked_mul(HEAD_DIM))
        .ok_or_else(|| NnisError::invalid_input("F16 attention work-item overflow"))?;

    let reference = benchmark_gpu(
        context,
        stream,
        BenchmarkCase::new("smollm2_f16_attention_reference", "f16")
            .with_dimension("query_heads", QUERY_HEADS as u64)
            .with_dimension("kv_heads", KV_HEADS as u64)
            .with_dimension("head_dim", HEAD_DIM as u64)
            .with_dimension("kv_rows", kv_rows as u64)
            .with_dimension("threads_per_block", HEAD_DIM as u64)
            .with_work_items(work_items as u64),
        bench_config,
        || unsafe {
            reference_kernel.enqueue_cached_attention_decode(
                stream,
                &query,
                &cache,
                0,
                &reference_output,
                scale,
            )
        },
    )?;

    let staged_benchmark = benchmark_gpu(
        context,
        stream,
        BenchmarkCase::new("smollm2_f16_attention_staged_weights", "f16")
            .with_dimension("query_heads", QUERY_HEADS as u64)
            .with_dimension("kv_heads", KV_HEADS as u64)
            .with_dimension("head_dim", HEAD_DIM as u64)
            .with_dimension("kv_rows", kv_rows as u64)
            .with_dimension("threads_per_block", HEAD_DIM as u64)
            .with_work_items(work_items as u64),
        bench_config,
        || unsafe {
            staged_kernel.enqueue_cached_attention_decode(
                stream,
                &query,
                &cache,
                0,
                &staged_output,
                scale,
            )
        },
    )?;
    require_compatible(&reference, &staged_benchmark)?;
    let staged_correctness =
        compare_outputs(stream, reference_kernel, &reference_output, &staged_output)?;
    let staged_comparison = comparison(&reference, &staged_benchmark)?;

    let mut candidates = Vec::with_capacity(BLOCK_MATRIX.len());
    for threads_per_block in BLOCK_MATRIX {
        let benchmark = benchmark_gpu(
            context,
            stream,
            BenchmarkCase::new("smollm2_f16_attention_parallel_scores", "f16")
                .with_dimension("query_heads", QUERY_HEADS as u64)
                .with_dimension("kv_heads", KV_HEADS as u64)
                .with_dimension("head_dim", HEAD_DIM as u64)
                .with_dimension("kv_rows", kv_rows as u64)
                .with_dimension("threads_per_block", threads_per_block as u64)
                .with_work_items(work_items as u64),
            bench_config,
            || unsafe {
                parallel_score_kernel.enqueue_cached_attention_decode(
                    stream,
                    &query,
                    &cache,
                    0,
                    &parallel_output,
                    scale,
                    threads_per_block,
                )
            },
        )?;
        require_compatible(&reference, &benchmark)?;
        let correctness = compare_outputs(
            stream,
            reference_kernel,
            &reference_output,
            &parallel_output,
        )?;
        let comparison = comparison(&reference, &benchmark)?;
        candidates.push(CandidateReport {
            threads_per_block,
            correctness,
            benchmark,
            comparison,
        });
    }

    let best = candidates
        .iter()
        .filter(|candidate| candidate.correctness.bitwise_equal)
        .min_by(|left, right| {
            left.benchmark
                .statistics
                .median_ms
                .total_cmp(&right.benchmark.statistics.median_ms)
        });
    let best_block = best.map(|candidate| candidate.threads_per_block);
    let best_speed =
        best.map(|candidate| candidate.comparison.reference_over_candidate_speed_ratio);
    let best_improvement = best.map(|candidate| candidate.comparison.relative_improvement);

    Ok(ShapeReport {
        kv_rows,
        reference,
        staged_weights: StagedReport {
            correctness: staged_correctness,
            benchmark: staged_benchmark,
            comparison: staged_comparison,
        },
        parallel_score_candidates: candidates,
        best_bitwise_parallel_score_block: best_block,
        best_bitwise_parallel_score_speed_ratio: best_speed,
        best_bitwise_parallel_score_relative_improvement: best_improvement,
    })
}

fn main() -> Result<()> {
    require_run_context()?;
    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 50)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 500)?;
    if warmups == 0 || iterations == 0 {
        return Err(NnisError::invalid_input(
            "F16 attention warmups and iterations must be non-zero",
        ));
    }

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let reference_kernel = F16ReferenceKernels::load(&context, &compiler)?;
    let staged_kernel = F16CachedAttentionStagedWeightsCandidate::load(&context, &compiler)?;
    let parallel_score_kernel =
        F16CachedAttentionParallelScoreCandidate::load(&context, &compiler)?;
    let bench_config = BenchConfig::new(warmups, iterations);
    bench_config.validate()?;

    let mut shapes = Vec::with_capacity(KV_ROWS_MATRIX.len());
    for kv_rows in KV_ROWS_MATRIX {
        shapes.push(run_shape(
            &context,
            &stream,
            &reference_kernel,
            &staged_kernel,
            &parallel_score_kernel,
            kv_rows,
            bench_config,
        )?);
    }

    let report = Report {
        schema_version: 1,
        experiment: "KA16-smollm2-f16-cached-attention-parallel-score-matrix-v1",
        promotion_state: "candidate-only; runtime attention plans and defaults remain unchanged",
        hypothesis: "parallelize Q·K score work across warps and KV positions while preserving serial online-softmax ordering and F32 value accumulation order",
        query_heads: QUERY_HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        smollm2_layers: SMOLLM2_LAYERS,
        warmups,
        iterations,
        kv_rows_matrix: KV_ROWS_MATRIX,
        block_matrix: BLOCK_MATRIX,
        shapes,
        limitations: [
            "the parallel-score reduction changes the Q·K F32 reduction tree relative to the qualified serial reference",
            "F16 output bitwise equality is measured rather than assumed and is required before a candidate can be considered for integration",
            "this matrix is isolated single-layer cached-attention evidence, not end-to-end decoder evidence",
            "the measured KV rows cover the short-context SmolLM2 generation regime profiled by KA15 but do not establish long-context performance",
            "the staged-weights candidate is included as a same-run comparator but remains unpromoted",
            "block-size selection must be based on measured rows and must not be extrapolated to unseen shapes or devices",
            "any later runtime integration still requires an explicit plan plus exact greedy trajectory and repeated end-to-end physical qualification",
        ],
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| NnisError::invalid_input(
            format!("serialize F16 attention matrix report: {error}")
        ))?
    );
    Ok(())
}
