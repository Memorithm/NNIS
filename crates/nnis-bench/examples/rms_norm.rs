use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::F32RmsNorm;
use nnis_rt::{Context, Device, DeviceBuffer, Stream};
use serde::Serialize;

const EPSILON: f32 = 1.0e-6;
const GAMMA: f32 = 1.625;

#[derive(Debug, Serialize)]
struct RmsNormBenchmark {
    schema_version: u32,
    report: BenchmarkReport,
    rows: usize,
    cols: usize,
    fused_path: bool,
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

fn expected_outputs(input: &[f32], rows: usize, cols: usize) -> impl Iterator<Item = f64> + '_ {
    input.chunks(cols).take(rows).flat_map(move |slice| {
        let count = slice.len() as f64;
        let mean_square: f64 = slice
            .iter()
            .map(|&value| {
                let widened = f64::from(value);
                widened * widened
            })
            .sum::<f64>()
            / count;
        let scale = (mean_square + f64::from(EPSILON)).sqrt().recip();
        slice
            .iter()
            .map(move |&value| f64::from(value) * scale * f64::from(GAMMA))
    })
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
    let rms_norm = F32RmsNorm::load(&context, &compiler)?;

    let matrix_host: Vec<f32> = (0..rows * cols)
        .map(|index| (((index * 13 % 97) as f32 - 48.0) * 0.0625) + ((index % 5) as f32 - 2.0))
        .collect();
    let matrix = DeviceBuffer::from_host(&context, &stream, &matrix_host)?;
    let output = DeviceBuffer::<f32>::new(&context, rows * cols)?;

    // Traffic model: every matrix element is read once and written once;
    // per-row statistic columns are negligible at these shapes.
    let bytes = (rows as u64)
        .checked_mul(2)
        .and_then(|value| value.checked_mul(cols as u64))
        .and_then(|value| value.checked_mul(4))
        .ok_or("rms norm traffic calculation overflow")?;

    let fused = rms_norm.fused_available(cols);
    let case = BenchmarkCase::new("nnis_rmsnorm_row_fused_f32", "f32")
        .with_dimension("rows", rows as u64)
        .with_dimension("cols", cols as u64)
        .with_dimension("block_size", u64::from(rms_norm.block_size()))
        .with_dimension("fused_path", u64::from(fused))
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
            unsafe {
                rms_norm.enqueue_fused_rows(&stream, &matrix, &output, rows, cols, EPSILON, GAMMA)
            }
        },
    )?;

    // Validate the timed configuration after measurement.
    let actual = output.to_vec(&stream)?;
    if actual.len() != rows * cols {
        return Err("rms norm output length mismatch".into());
    }
    let mut max_absolute_error = 0.0_f64;
    for (index, (&actual, expected)) in actual
        .iter()
        .zip(expected_outputs(&matrix_host, rows, cols))
        .enumerate()
    {
        max_absolute_error = max_absolute_error.max((f64::from(actual) - expected).abs());
        if (f64::from(actual) - expected).abs() > 1.0e-3_f64.max(expected.abs() * 1.0e-5) {
            return Err(format!("rms norm mismatch at {index}: {actual} != {expected}").into());
        }
    }

    let result = RmsNormBenchmark {
        schema_version: 1,
        report,
        rows,
        cols,
        fused_path: fused,
        max_absolute_error,
        correctness_validated: true,
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
