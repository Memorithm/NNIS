use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::{AttentionMask, F32Attention, F32Elementwise, F32Gemm, F32Softmax2D};
use nnis_rt::{Context, Device, DeviceBuffer, Stream};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct AttentionBenchmark {
    schema_version: u32,
    query_rows: usize,
    head_dim: usize,
    kv_rows: usize,
    value_dim: usize,
    fused_report: BenchmarkReport,
    composed_report: BenchmarkReport,
    fused_max_absolute_error: f64,
    composed_max_absolute_error: f64,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query_rows = env_usize("NNIS_BENCH_ROWS", 2_048)?;
    let head_dim = env_usize("NNIS_BENCH_HEAD_DIM", 64)?;
    let kv_rows = env_usize("NNIS_BENCH_KV_ROWS", 2_048)?;
    let value_dim = env_usize("NNIS_BENCH_VALUE_DIM", 64)?;
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 100)?;

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let attention = F32Attention::load(&context, &compiler)?;
    let gemm = F32Gemm::load(&context, &compiler)?;
    let elementwise = F32Elementwise::load(&context, &compiler)?;
    let softmax_2d = F32Softmax2D::load(&context, &compiler)?;

    if !attention.fused_available(head_dim, value_dim) {
        return Err(format!(
            "fused attention unavailable for head ({head_dim}, {value_dim}) at block {}",
            attention.block_size()
        )
        .into());
    }

    let queries_host: Vec<f32> = (0..query_rows * head_dim)
        .map(|index| (((index * 13 % 97) as f32 - 48.0) * 0.0625) + ((index % 5) as f32 - 2.0))
        .collect();
    let keys_host: Vec<f32> = (0..kv_rows * head_dim)
        .map(|index| ((index * 29 % 61) as f32 - 30.0) * 0.125)
        .collect();
    let values_host: Vec<f32> = (0..kv_rows * value_dim)
        .map(|index| ((index * 7 % 43) as f32 - 21.0) * 0.03125)
        .collect();
    let queries = DeviceBuffer::from_host(&context, &stream, &queries_host)?;
    let keys = DeviceBuffer::from_host(&context, &stream, &keys_host)?;
    let values = DeviceBuffer::from_host(&context, &stream, &values_host)?;
    let fused_output = DeviceBuffer::<f32>::new(&context, query_rows * value_dim)?;
    let composed_output = DeviceBuffer::<f32>::new(&context, query_rows * value_dim)?;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Traffic model: the fused path streams Q once and every K/V chunk
    // ceil(kv/threads) times; the composed path materializes two full
    // score matrices on top of one QK^T read/write pair.
    let chunks = kv_rows.div_ceil(attention.block_size() as usize);
    let fused_bytes = (queries_host.len()
        + chunks * (keys_host.len() + values_host.len())
        + query_rows * value_dim)
        .checked_mul(4)
        .ok_or("attention traffic overflow")?;
    let score_bytes = query_rows
        .checked_mul(kv_rows)
        .ok_or("attention traffic overflow")?
        * 4;
    let composed_bytes = fused_bytes + 4 * score_bytes;

    let dims = |case: BenchmarkCase| {
        case.with_dimension("query_rows", query_rows as u64)
            .with_dimension("head_dim", head_dim as u64)
            .with_dimension("kv_rows", kv_rows as u64)
            .with_dimension("value_dim", value_dim as u64)
            .with_dimension("block_size", u64::from(attention.block_size()))
            .with_work_items((query_rows * kv_rows * head_dim) as u64)
    };
    let config = BenchConfig::new(warmups, iterations);

    let fused_report = benchmark_gpu(
        &context,
        &stream,
        dims(BenchmarkCase::new("nnis_attention_fused_f32", "f32"))
            .with_bytes_per_iteration(fused_bytes as u64),
        config,
        || {
            // SAFETY: all buffers outlive this harness, which synchronizes
            // the end event for each invocation.
            unsafe {
                attention.enqueue_attention_fused(
                    &stream,
                    &queries,
                    &keys,
                    &values,
                    &fused_output,
                    query_rows,
                    head_dim,
                    kv_rows,
                    value_dim,
                    scale,
                    AttentionMask::None,
                )
            }
        },
    )?;

    let composed_report = benchmark_gpu(
        &context,
        &stream,
        dims(BenchmarkCase::new("nnis_attention_composed_f32", "f32"))
            .with_bytes_per_iteration(composed_bytes as u64),
        config,
        || {
            attention.attention_composed(
                &gemm,
                &elementwise,
                &softmax_2d,
                &stream,
                &queries,
                &keys,
                &values,
                &composed_output,
                query_rows,
                head_dim,
                kv_rows,
                value_dim,
                scale,
                AttentionMask::None,
            )
        },
    )?;

    let validate =
        |output: &DeviceBuffer<f32>, name: &str| -> Result<f64, Box<dyn std::error::Error>> {
            let actual = output.to_vec(&stream)?;
            if actual.len() != query_rows * value_dim {
                return Err("attention output length mismatch".into());
            }
            let mut max_absolute_error = 0.0_f64;
            for row in 0..query_rows {
                let mut scores = vec![0.0_f64; kv_rows];
                for key in 0..kv_rows {
                    let score: f64 = (0..head_dim)
                        .map(|e| {
                            f64::from(queries_host[row * head_dim + e])
                                * f64::from(keys_host[key * head_dim + e])
                        })
                        .sum();
                    scores[key] = score * f64::from(scale);
                }
                let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let total: f64 = weights.iter().sum();
                for col in 0..value_dim {
                    let expected: f64 = (0..kv_rows)
                        .map(|key| weights[key] * f64::from(values_host[key * value_dim + col]))
                        .sum::<f64>()
                        / total;
                    let error = (f64::from(actual[row * value_dim + col]) - expected).abs();
                    max_absolute_error = max_absolute_error.max(error);
                    if error > 1.0e-3_f64.max(expected.abs() * 1.0e-4) {
                        return Err(format!(
                            "{name} mismatch at ({row}, {col}): {} != {expected}",
                            actual[row * value_dim + col]
                        )
                        .into());
                    }
                }
            }
            Ok(max_absolute_error)
        };

    let fused_max_absolute_error = validate(&fused_output, "fused")?;
    let composed_max_absolute_error = validate(&composed_output, "composed")?;

    let result = AttentionBenchmark {
        schema_version: 1,
        query_rows,
        head_dim,
        kv_rows,
        value_dim,
        fused_report,
        composed_report,
        fused_max_absolute_error,
        composed_max_absolute_error,
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
