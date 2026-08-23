use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::F32Gemv;
use nnis_rt::{Context, Device, DeviceBuffer, Stream};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct GemvBenchmark {
    schema_version: u32,
    report: BenchmarkReport,
    rows: usize,
    cols: usize,
    max_absolute_error: f64,
    correctness_validated: bool,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows = env_usize("NNIS_BENCH_ROWS", 4_096)?;
    let cols = env_usize("NNIS_BENCH_COLS", 4_096)?;
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 100)?;

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let gemv = F32Gemv::load(&context, &compiler)?;

    let matrix_host: Vec<f32> = (0..rows * cols)
        .map(|index| (((index * 13 % 97) as f32 - 48.0) * 0.0625) + ((index % 5) as f32 - 2.0))
        .collect();
    let vector_host: Vec<f32> = (0..cols)
        .map(|index| ((index * 29 % 61) as f32 - 30.0) * 0.125)
        .collect();
    let matrix = DeviceBuffer::from_host(&context, &stream, &matrix_host)?;
    let vector = DeviceBuffer::from_host(&context, &stream, &vector_host)?;
    let output = DeviceBuffer::<f32>::new(&context, rows)?;

    // Traffic model: every matrix element is read once, the vector is
    // re-read per row from cache or memory, and each output row is written.
    let bytes = (rows as u64)
        .checked_mul(cols as u64 + 1)
        .and_then(|value| value.checked_add(cols as u64))
        .and_then(|value| value.checked_mul(4))
        .ok_or("gemv traffic calculation overflow")?;

    let case = BenchmarkCase::new("nnis_gemv_f32", "f32")
        .with_dimension("rows", rows as u64)
        .with_dimension("cols", cols as u64)
        .with_dimension("block_size", u64::from(gemv.block_size()))
        .with_work_items((rows * cols) as u64)
        .with_bytes_per_iteration(bytes);
    let report = benchmark_gpu(
        &context,
        &stream,
        case,
        BenchConfig::new(warmups, iterations),
        || {
            // SAFETY: all buffers outlive this harness, which synchronizes
            // the end event for each invocation.
            unsafe { gemv.enqueue_gemv(&stream, &matrix, &vector, &output, rows, cols) }
        },
    )?;

    // Validate the timed configuration after measurement.
    let actual = output.to_vec(&stream)?;
    if actual.len() != rows {
        return Err("gemv output length mismatch".into());
    }
    let mut max_absolute_error = 0.0_f64;
    for row in 0..rows {
        let expected: f64 = (0..cols)
            .map(|col| f64::from(matrix_host[row * cols + col]) * f64::from(vector_host[col]))
            .sum();
        max_absolute_error = max_absolute_error.max((f64::from(actual[row]) - expected).abs());
        if (f64::from(actual[row]) - expected).abs() > 1.0e-3_f64.max(expected.abs() * 1.0e-5) {
            return Err(
                format!("gemv mismatch at row {row}: {} != {expected}", actual[row]).into(),
            );
        }
    }

    let result = GemvBenchmark {
        schema_version: 1,
        report,
        rows,
        cols,
        max_absolute_error,
        correctness_validated: true,
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
