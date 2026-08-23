//! Event-timed RoPE (rotate-half) at an attention-shaped size with
//! post-timing f64 validation.
use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase};
use nnis_jit::JitCompiler;
use nnis_kernels::F32Rope;
use nnis_rt::{gpu_context, DeviceBuffer, Stream};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct RopeBenchmark {
    schema_version: u32,
    rows: usize,
    cols: usize,
    median_ms: f64,
    gigabytes_per_second: f64,
    max_absolute_error: f64,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows = env_usize("NNIS_BENCH_ROWS", 8_192)?;
    let cols = env_usize("NNIS_BENCH_COLS", 128)?;
    let Some(context) = gpu_context() else {
        return Err("no CUDA device".into());
    };
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let rope = F32Rope::load(&context, &compiler)?;

    let half = cols / 2;
    let host: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.1875)
        .collect();
    let cos_host: Vec<f32> = (0..rows * half)
        .map(|i| ((i % 37) as f32 * 0.17).cos())
        .collect();
    let sin_host: Vec<f32> = (0..rows * half)
        .map(|i| ((i % 37) as f32 * 0.17).sin())
        .collect();
    let input = DeviceBuffer::from_host(&context, &stream, &host)?;
    let cos = DeviceBuffer::from_host(&context, &stream, &cos_host)?;
    let sin = DeviceBuffer::from_host(&context, &stream, &sin_host)?;
    let output = DeviceBuffer::<f32>::new(&context, rows * cols)?;

    // Read input + write output; cos/sin caches add cols/2 per row.
    let bytes = (rows as u64)
        .checked_mul(cols as u64 + 1)
        .and_then(|v| v.checked_mul(4))
        .ok_or("rope traffic overflow")?;

    let case = BenchmarkCase::new("nnis_rope_rotate_half_f32", "f32")
        .with_dimension("rows", rows as u64)
        .with_dimension("cols", cols as u64)
        .with_work_items((rows * half) as u64)
        .with_bytes_per_iteration(bytes);
    let report = benchmark_gpu(&context, &stream, case, BenchConfig::new(20, 100), || {
        rope.apply_rotate_half(&stream, &input, &cos, &sin, &output, rows, cols)
    })?;

    // Validate the timed configuration.
    let actual = output.to_vec(&stream)?;
    let mut max_error = 0.0_f64;
    for index in 0..rows * half {
        let row = index / half;
        let j = index % half;
        let expected_first = f64::from(host[row * cols + j]) * f64::from(cos_host[index])
            - f64::from(host[row * cols + half + j]) * f64::from(sin_host[index]);
        max_error = max_error.max((f64::from(actual[row * cols + j]) - expected_first).abs());
        if max_error > 1.0e-3 {
            return Err(format!("rope mismatch at pair {index}").into());
        }
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&RopeBenchmark {
            schema_version: 1,
            rows,
            cols,
            median_ms: report.statistics.median_ms,
            gigabytes_per_second: report
                .throughput
                .and_then(|t| t.gigabytes_per_second)
                .unwrap_or(0.0),
            max_absolute_error: max_error,
        })?
    );
    Ok(())
}
