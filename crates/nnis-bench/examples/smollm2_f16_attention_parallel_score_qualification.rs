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
const MIN_KV_ROWS: usize = 1;
const MAX_KV_ROWS: usize = 35;
const GENERATION_MIN_KV_ROWS: usize = 4;
const GENERATION_MAX_KV_ROWS: usize = 34;
const BLOCKS: [u32; 4] = [64, 128, 256, 512];
const SMALL_SCALE: f32 = 1.0 / 4096.0;
const SMALL_VALUE_SCALE: f32 = 1.0 / 2048.0;
const NEAR_TIE_STEP: f32 = 1.0 / 65536.0;
const FIXTURES: [FixtureKind; 6] = [
    FixtureKind::Baseline,
    FixtureKind::SmallMagnitude,
    FixtureKind::WideMagnitude,
    FixtureKind::Cancellation,
    FixtureKind::NearTie,
    FixtureKind::SignedRamp,
];

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum FixtureKind {
    Baseline,
    SmallMagnitude,
    WideMagnitude,
    Cancellation,
    NearTie,
    SignedRamp,
}

impl FixtureKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::SmallMagnitude => "small_magnitude",
            Self::WideMagnitude => "wide_magnitude",
            Self::Cancellation => "cancellation",
            Self::NearTie => "near_tie",
            Self::SignedRamp => "signed_ramp",
        }
    }
}

#[derive(Debug, Serialize)]
struct CorrectnessRecord {
    kv_rows: usize,
    fixture: FixtureKind,
    block: u32,
    bitwise_equal: bool,
    differing_elements: usize,
    elements_checked: usize,
    max_absolute_error: f32,
    max_relative_error: f32,
}

#[derive(Debug, Serialize)]
struct CorrectnessAggregate {
    block: u32,
    cases: usize,
    bitwise_equal_cases: usize,
    differing_elements_total: usize,
    worst_max_absolute_error: f32,
    worst_max_relative_error: f32,
    first_non_bitwise_case: Option<(usize, &'static str)>,
}

#[derive(Debug, Serialize)]
struct PerformanceRow {
    kv_rows: usize,
    reference_median_ms: f64,
    staged_median_ms: f64,
    staged_relative_improvement: f64,
    block_64_median_ms: f64,
    block_128_median_ms: f64,
    block_256_median_ms: f64,
    block_512_median_ms: f64,
    block_64_relative_improvement: f64,
    block_128_relative_improvement: f64,
    block_256_relative_improvement: f64,
    block_512_relative_improvement: f64,
    predeclared_policy_block: Option<u32>,
    predeclared_policy_median_ms: f64,
    predeclared_policy_relative_improvement: f64,
}

#[derive(Debug, Serialize)]
struct AggregatePerformance {
    kv_rows_start: usize,
    kv_rows_end: usize,
    row_count: usize,
    reference_sum_of_row_medians_ms: f64,
    staged_sum_of_row_medians_ms: f64,
    block_64_sum_of_row_medians_ms: f64,
    block_128_sum_of_row_medians_ms: f64,
    block_256_sum_of_row_medians_ms: f64,
    block_512_sum_of_row_medians_ms: f64,
    predeclared_policy_sum_of_row_medians_ms: f64,
    staged_relative_improvement: f64,
    block_64_relative_improvement: f64,
    block_128_relative_improvement: f64,
    block_256_relative_improvement: f64,
    block_512_relative_improvement: f64,
    predeclared_policy_relative_improvement: f64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    experiment: &'static str,
    promotion_state: &'static str,
    numerical_scope: &'static str,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    min_kv_rows: usize,
    max_kv_rows: usize,
    generation_kv_rows: (usize, usize),
    correctness_fixtures: Vec<&'static str>,
    candidate_blocks: [u32; 4],
    predeclared_policy: &'static str,
    warmups: usize,
    iterations: usize,
    correctness: Vec<CorrectnessRecord>,
    correctness_aggregates: Vec<CorrectnessAggregate>,
    performance_rows: Vec<PerformanceRow>,
    full_short_context_aggregate: AggregatePerformance,
    generation_path_aggregate: AggregatePerformance,
    limitations: [&'static str; 8],
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
            "NNIS_BENCH_RUN_CONTEXT_ID is required for KA17 evidence",
        )),
    }
}

fn baseline_sequence(elements: usize, multiplier: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let centered = ((index * multiplier + 11) % modulus) as i32 - (modulus as i32 / 2);
            centered as f32 * scale
        })
        .collect()
}

fn fixture_values(kind: FixtureKind, kv_rows: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let query_elements = QUERY_HEADS * HEAD_DIM;
    let kv_elements = KV_HEADS * kv_rows * HEAD_DIM;
    match kind {
        FixtureKind::Baseline => (
            baseline_sequence(query_elements, 17, 97, 0.015625),
            baseline_sequence(kv_elements, 29, 113, 0.0078125),
            baseline_sequence(kv_elements, 31, 127, 0.015625),
        ),
        FixtureKind::SmallMagnitude => (
            baseline_sequence(query_elements, 19, 101, SMALL_SCALE),
            baseline_sequence(kv_elements, 23, 103, SMALL_SCALE),
            baseline_sequence(kv_elements, 37, 109, SMALL_VALUE_SCALE),
        ),
        FixtureKind::WideMagnitude => (
            baseline_sequence(query_elements, 13, 61, 0.125),
            baseline_sequence(kv_elements, 41, 67, 0.125),
            baseline_sequence(kv_elements, 43, 71, 0.0625),
        ),
        FixtureKind::Cancellation => {
            let query = (0..query_elements)
                .map(|index| {
                    let sign = if index & 1 == 0 { 1.0 } else { -1.0 };
                    sign * (1.0 + (index % 7) as f32 * 0.015625)
                })
                .collect();
            let keys = (0..kv_elements)
                .map(|index| {
                    let dim = index % HEAD_DIM;
                    let pos = (index / HEAD_DIM) % kv_rows;
                    let sign = if (dim + pos) & 1 == 0 { -1.0 } else { 1.0 };
                    sign * (1.0 + ((dim * 3 + pos) % 11) as f32 * 0.0078125)
                })
                .collect();
            let values = baseline_sequence(kv_elements, 47, 79, 0.03125);
            (query, keys, values)
        }
        FixtureKind::NearTie => {
            let query = baseline_sequence(query_elements, 7, 59, 0.03125);
            let keys = (0..kv_elements)
                .map(|index| {
                    let dim = index % HEAD_DIM;
                    let pos = (index / HEAD_DIM) % kv_rows;
                    let base = ((dim * 5 + 3) % 31) as f32 - 15.0;
                    base * 0.015625 + pos as f32 * NEAR_TIE_STEP
                })
                .collect();
            let values = baseline_sequence(kv_elements, 53, 83, 0.015625);
            (query, keys, values)
        }
        FixtureKind::SignedRamp => {
            let query = (0..query_elements)
                .map(|index| {
                    let dim = index % HEAD_DIM;
                    let sign = if ((index / HEAD_DIM) & 1) == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    sign * (dim as f32 - 31.5) * 0.015625
                })
                .collect();
            let keys = (0..kv_elements)
                .map(|index| {
                    let dim = index % HEAD_DIM;
                    let pos = (index / HEAD_DIM) % kv_rows;
                    let sign = if pos & 1 == 0 { 1.0 } else { -1.0 };
                    sign * (dim as f32 - 31.5) * 0.0078125
                })
                .collect();
            let values = baseline_sequence(kv_elements, 61, 89, 0.015625);
            (query, keys, values)
        }
    }
}

struct FixtureBuffers {
    query: DeviceBuffer<u16>,
    cache: KvCache<u16>,
    reference_output: DeviceBuffer<u16>,
    candidate_output: DeviceBuffer<u16>,
}

fn materialize_fixture(
    context: &Arc<Context>,
    stream: &Stream,
    kernels: &F16ReferenceKernels,
    kind: FixtureKind,
    kv_rows: usize,
) -> Result<FixtureBuffers> {
    let query_elements = QUERY_HEADS * HEAD_DIM;
    let kv_elements = KV_HEADS * kv_rows * HEAD_DIM;
    let (query_host, keys_host, values_host) = fixture_values(kind, kv_rows);
    let query_f32 = DeviceBuffer::from_host(context, stream, &query_host)?;
    let keys_f32 = DeviceBuffer::from_host(context, stream, &keys_host)?;
    let values_f32 = DeviceBuffer::from_host(context, stream, &values_host)?;
    let query = DeviceBuffer::<u16>::new(context, query_elements)?;
    let keys = Arc::new(DeviceBuffer::<u16>::new(context, kv_elements)?);
    let values = Arc::new(DeviceBuffer::<u16>::new(context, kv_elements)?);
    unsafe {
        kernels.enqueue_narrow_from_f32(stream, &query_f32, &query)?;
        kernels.enqueue_narrow_from_f32(stream, &keys_f32, &keys)?;
        kernels.enqueue_narrow_from_f32(stream, &values_f32, &values)?;
    }
    stream.synchronize()?;

    let mut cache = KvCache::new(stream, KvCacheConfig::new(1, KV_HEADS, HEAD_DIM, kv_rows)?)?;
    cache.append_layer(0, keys, values, kv_rows)?;
    Ok(FixtureBuffers {
        query,
        cache,
        reference_output: DeviceBuffer::<u16>::new(context, query_elements)?,
        candidate_output: DeviceBuffer::<u16>::new(context, query_elements)?,
    })
}

fn compare_outputs(
    stream: &Stream,
    kernels: &F16ReferenceKernels,
    reference: &DeviceBuffer<u16>,
    candidate: &DeviceBuffer<u16>,
) -> Result<(bool, usize, f32, f32)> {
    stream.synchronize()?;
    let reference_bits = reference.to_vec(stream)?;
    let candidate_bits = candidate.to_vec(stream)?;
    let reference_f32 = DeviceBuffer::<f32>::new(stream.ctx(), reference.len())?;
    let candidate_f32 = DeviceBuffer::<f32>::new(stream.ctx(), candidate.len())?;
    unsafe {
        kernels.enqueue_widen_to_f32(stream, reference, &reference_f32)?;
        kernels.enqueue_widen_to_f32(stream, candidate, &candidate_f32)?;
    }
    stream.synchronize()?;
    let reference_values = reference_f32.to_vec(stream)?;
    let candidate_values = candidate_f32.to_vec(stream)?;

    let mut differing = 0_usize;
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for index in 0..reference_bits.len() {
        if reference_bits[index] != candidate_bits[index] {
            differing += 1;
        }
        let left = reference_values[index];
        let right = candidate_values[index];
        if !left.is_finite() || !right.is_finite() {
            return Err(NnisError::invalid_input(format!(
                "non-finite KA17 attention output at element {index}: reference={left}, candidate={right}"
            )));
        }
        let abs = (left - right).abs();
        let rel = abs / left.abs().max(1.0e-12);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    Ok((differing == 0, differing, max_abs, max_rel))
}

fn run_correctness(
    context: &Arc<Context>,
    stream: &Stream,
    kernels: &F16ReferenceKernels,
    candidate: &F16CachedAttentionParallelScoreCandidate,
) -> Result<Vec<CorrectnessRecord>> {
    let mut records = Vec::new();
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    for kv_rows in MIN_KV_ROWS..=MAX_KV_ROWS {
        for fixture in FIXTURES {
            let buffers = materialize_fixture(context, stream, kernels, fixture, kv_rows)?;
            unsafe {
                kernels.enqueue_cached_attention_decode(
                    stream,
                    &buffers.query,
                    &buffers.cache,
                    0,
                    &buffers.reference_output,
                    scale,
                )?;
            }
            for block in BLOCKS {
                unsafe {
                    candidate.enqueue_cached_attention_decode(
                        stream,
                        &buffers.query,
                        &buffers.cache,
                        0,
                        &buffers.candidate_output,
                        scale,
                        block,
                    )?;
                }
                let (bitwise, differing, max_abs, max_rel) = compare_outputs(
                    stream,
                    kernels,
                    &buffers.reference_output,
                    &buffers.candidate_output,
                )?;
                records.push(CorrectnessRecord {
                    kv_rows,
                    fixture,
                    block,
                    bitwise_equal: bitwise,
                    differing_elements: differing,
                    elements_checked: QUERY_HEADS * HEAD_DIM,
                    max_absolute_error: max_abs,
                    max_relative_error: max_rel,
                });
            }
        }
    }
    Ok(records)
}

fn correctness_aggregates(records: &[CorrectnessRecord]) -> Vec<CorrectnessAggregate> {
    BLOCKS
        .iter()
        .map(|&block| {
            let selected: Vec<_> = records
                .iter()
                .filter(|record| record.block == block)
                .collect();
            CorrectnessAggregate {
                block,
                cases: selected.len(),
                bitwise_equal_cases: selected
                    .iter()
                    .filter(|record| record.bitwise_equal)
                    .count(),
                differing_elements_total: selected
                    .iter()
                    .map(|record| record.differing_elements)
                    .sum(),
                worst_max_absolute_error: selected
                    .iter()
                    .map(|record| record.max_absolute_error)
                    .fold(0.0_f32, f32::max),
                worst_max_relative_error: selected
                    .iter()
                    .map(|record| record.max_relative_error)
                    .fold(0.0_f32, f32::max),
                first_non_bitwise_case: selected
                    .iter()
                    .find(|record| !record.bitwise_equal)
                    .map(|record| (record.kv_rows, record.fixture.as_str())),
            }
        })
        .collect()
}

fn require_compatible(reference: &BenchmarkReport, other: &BenchmarkReport) -> Result<()> {
    reference
        .metadata
        .require_compatible_environment(&other.metadata)?;
    if reference.metadata.git_commit != other.metadata.git_commit
        || reference.metadata.git_dirty != Some(false)
        || other.metadata.git_dirty != Some(false)
    {
        return Err(NnisError::invalid_input(
            "KA17 performance evidence requires one clean exact Git identity",
        ));
    }
    Ok(())
}

fn relative_improvement(reference: f64, candidate: f64) -> Result<f64> {
    if reference <= 0.0 || candidate <= 0.0 {
        return Err(NnisError::unsupported(
            "KA17 performance medians must be positive",
        ));
    }
    Ok((reference - candidate) / reference)
}

fn predeclared_policy_block(kv_rows: usize) -> Option<u32> {
    if kv_rows <= 3 {
        None
    } else if kv_rows <= 4 {
        Some(128)
    } else if kv_rows <= 16 {
        Some(256)
    } else {
        Some(512)
    }
}

fn benchmark_row(
    context: &Arc<Context>,
    stream: &Stream,
    kernels: &F16ReferenceKernels,
    staged: &F16CachedAttentionStagedWeightsCandidate,
    candidate: &F16CachedAttentionParallelScoreCandidate,
    kv_rows: usize,
    config: BenchConfig,
) -> Result<PerformanceRow> {
    let buffers = materialize_fixture(context, stream, kernels, FixtureKind::Baseline, kv_rows)?;
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let work_items = QUERY_HEADS * kv_rows * HEAD_DIM;
    let case = |name: &'static str, block: u32| {
        BenchmarkCase::new(name, "f16")
            .with_dimension("query_heads", QUERY_HEADS as u64)
            .with_dimension("kv_heads", KV_HEADS as u64)
            .with_dimension("head_dim", HEAD_DIM as u64)
            .with_dimension("kv_rows", kv_rows as u64)
            .with_dimension("threads_per_block", block as u64)
            .with_work_items(work_items as u64)
    };

    let reference = benchmark_gpu(
        context,
        stream,
        case("ka17_f16_attention_reference", HEAD_DIM as u32),
        config,
        || unsafe {
            kernels.enqueue_cached_attention_decode(
                stream,
                &buffers.query,
                &buffers.cache,
                0,
                &buffers.reference_output,
                scale,
            )
        },
    )?;
    let staged_report = benchmark_gpu(
        context,
        stream,
        case("ka17_f16_attention_staged", HEAD_DIM as u32),
        config,
        || unsafe {
            staged.enqueue_cached_attention_decode(
                stream,
                &buffers.query,
                &buffers.cache,
                0,
                &buffers.candidate_output,
                scale,
            )
        },
    )?;
    require_compatible(&reference, &staged_report)?;

    let mut block_reports = Vec::new();
    for block in BLOCKS {
        let report = benchmark_gpu(
            context,
            stream,
            case("ka17_f16_attention_parallel_score", block),
            config,
            || unsafe {
                candidate.enqueue_cached_attention_decode(
                    stream,
                    &buffers.query,
                    &buffers.cache,
                    0,
                    &buffers.candidate_output,
                    scale,
                    block,
                )
            },
        )?;
        require_compatible(&reference, &report)?;
        block_reports.push((block, report));
    }

    let median_for = |block: u32| -> f64 {
        block_reports
            .iter()
            .find(|(candidate_block, _)| *candidate_block == block)
            .expect("BLOCKS entry missing from KA17 row")
            .1
            .statistics
            .median_ms
    };
    let reference_median = reference.statistics.median_ms;
    let staged_median = staged_report.statistics.median_ms;
    let m64 = median_for(64);
    let m128 = median_for(128);
    let m256 = median_for(256);
    let m512 = median_for(512);
    let policy_block = predeclared_policy_block(kv_rows);
    let policy_median = match policy_block {
        None => reference_median,
        Some(64) => m64,
        Some(128) => m128,
        Some(256) => m256,
        Some(512) => m512,
        Some(other) => unreachable!("unexpected KA17 policy block {other}"),
    };

    Ok(PerformanceRow {
        kv_rows,
        reference_median_ms: reference_median,
        staged_median_ms: staged_median,
        staged_relative_improvement: relative_improvement(reference_median, staged_median)?,
        block_64_median_ms: m64,
        block_128_median_ms: m128,
        block_256_median_ms: m256,
        block_512_median_ms: m512,
        block_64_relative_improvement: relative_improvement(reference_median, m64)?,
        block_128_relative_improvement: relative_improvement(reference_median, m128)?,
        block_256_relative_improvement: relative_improvement(reference_median, m256)?,
        block_512_relative_improvement: relative_improvement(reference_median, m512)?,
        predeclared_policy_block: policy_block,
        predeclared_policy_median_ms: policy_median,
        predeclared_policy_relative_improvement: relative_improvement(
            reference_median,
            policy_median,
        )?,
    })
}

fn aggregate(rows: &[PerformanceRow], start: usize, end: usize) -> Result<AggregatePerformance> {
    let selected: Vec<_> = rows
        .iter()
        .filter(|row| row.kv_rows >= start && row.kv_rows <= end)
        .collect();
    if selected.len() != end - start + 1 {
        return Err(NnisError::invalid_input(format!(
            "KA17 aggregate {start}..={end} is incomplete: observed {} rows",
            selected.len()
        )));
    }
    let sum = |f: fn(&PerformanceRow) -> f64| selected.iter().map(|row| f(row)).sum::<f64>();
    let reference = sum(|row| row.reference_median_ms);
    let staged = sum(|row| row.staged_median_ms);
    let b64 = sum(|row| row.block_64_median_ms);
    let b128 = sum(|row| row.block_128_median_ms);
    let b256 = sum(|row| row.block_256_median_ms);
    let b512 = sum(|row| row.block_512_median_ms);
    let policy = sum(|row| row.predeclared_policy_median_ms);
    Ok(AggregatePerformance {
        kv_rows_start: start,
        kv_rows_end: end,
        row_count: selected.len(),
        reference_sum_of_row_medians_ms: reference,
        staged_sum_of_row_medians_ms: staged,
        block_64_sum_of_row_medians_ms: b64,
        block_128_sum_of_row_medians_ms: b128,
        block_256_sum_of_row_medians_ms: b256,
        block_512_sum_of_row_medians_ms: b512,
        predeclared_policy_sum_of_row_medians_ms: policy,
        staged_relative_improvement: relative_improvement(reference, staged)?,
        block_64_relative_improvement: relative_improvement(reference, b64)?,
        block_128_relative_improvement: relative_improvement(reference, b128)?,
        block_256_relative_improvement: relative_improvement(reference, b256)?,
        block_512_relative_improvement: relative_improvement(reference, b512)?,
        predeclared_policy_relative_improvement: relative_improvement(reference, policy)?,
    })
}

fn main() -> Result<()> {
    require_run_context()?;
    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 50)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 500)?;
    if warmups == 0 || iterations == 0 {
        return Err(NnisError::invalid_input(
            "KA17 warmups and iterations must be non-zero",
        ));
    }

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let kernels = F16ReferenceKernels::load(&context, &compiler)?;
    let staged = F16CachedAttentionStagedWeightsCandidate::load(&context, &compiler)?;
    let candidate = F16CachedAttentionParallelScoreCandidate::load(&context, &compiler)?;
    if !candidate.supports_kv_rows(MAX_KV_ROWS) {
        return Err(NnisError::unsupported(format!(
            "KA17 candidate supports fewer than {MAX_KV_ROWS} KV rows on this device/kernel"
        )));
    }

    let correctness = run_correctness(&context, &stream, &kernels, &candidate)?;
    let correctness_aggregates = correctness_aggregates(&correctness);
    let config = BenchConfig::new(warmups, iterations);
    config.validate()?;
    let mut performance_rows = Vec::with_capacity(MAX_KV_ROWS - MIN_KV_ROWS + 1);
    for kv_rows in MIN_KV_ROWS..=MAX_KV_ROWS {
        performance_rows.push(benchmark_row(
            &context, &stream, &kernels, &staged, &candidate, kv_rows, config,
        )?);
    }
    let full_short_context_aggregate = aggregate(&performance_rows, MIN_KV_ROWS, MAX_KV_ROWS)?;
    let generation_path_aggregate = aggregate(
        &performance_rows,
        GENERATION_MIN_KV_ROWS,
        GENERATION_MAX_KV_ROWS,
    )?;

    let report = Report {
        schema_version: 1,
        experiment: "KA17-smollm2-f16-attention-parallel-score-qualification-v1",
        promotion_state: "candidate-only; no runtime attention-plan or default change",
        numerical_scope: "F16 query/K/V storage, F32 score/softmax/value accumulation, final F16 output",
        query_heads: QUERY_HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        min_kv_rows: MIN_KV_ROWS,
        max_kv_rows: MAX_KV_ROWS,
        generation_kv_rows: (GENERATION_MIN_KV_ROWS, GENERATION_MAX_KV_ROWS),
        correctness_fixtures: FIXTURES.iter().map(|fixture| fixture.as_str()).collect(),
        candidate_blocks: BLOCKS,
        predeclared_policy: "reference for KV<=3; block128 for KV=4; block256 for KV=5..16; block512 for KV>=17",
        warmups,
        iterations,
        correctness,
        correctness_aggregates,
        performance_rows,
        full_short_context_aggregate,
        generation_path_aggregate,
        limitations: [
            "synthetic correctness fixtures do not substitute for the pinned SmolLM2 greedy trajectory",
            "bitwise equality is observed at the F16 output boundary; the parallel Q dot K reduction tree remains different internally",
            "the predeclared launch policy was fixed from KA16 before this dense-row run and must not be retuned post hoc from KA17 without a new confirmation campaign",
            "sum-of-row-medians is an isolated attention proxy, not a decoder latency or tokens-per-second claim",
            "performance rows are measured sequentially rather than with an end-to-end ABBA request protocol",
            "the qualified generation path for decode32 uses KV rows 4 through 34 after token-serial prefill",
            "results are device/software-environment specific and must not be generalized to unseen GPUs or long contexts",
            "runtime integration still requires an explicit attention plan, exact 32/32 greedy trajectory, repeated physical A/B evidence, and exact-head CI",
        ],
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| NnisError::invalid_input(format!("serialize KA17 report: {error}")))?
    );
    Ok(())
}
