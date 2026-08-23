use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::F32Softmax2D;
use nnis_rt::{Context, Device, DeviceBuffer, Stream};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct RowSoftmaxBenchmark {
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
    let elements = env_usize("NNIS_BENCH_ELEMENTS", 1 << 24)?;
    let cols = env_usize("NNIS_BENCH_COLS", 2_048)?;
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 100)?;
    if elements % cols != 0 {
        return Err("NNIS_BENCH_ELEMENTS must be divisible by NNIS_BENCH_COLS".into());
    }
    let rows = elements / cols;

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let softmax = F32Softmax2D::load(&context, &compiler)?;
    let workspace = softmax.workspace(&context, rows)?;
    let host = (0..elements)
        .map(|index| {
            let spread = ((index % 23) as f32 - 11.0) * 29.5;
            let ripple = ((index * 11 % 127) as f32 - 63.0) * 0.125;
            spread + ripple
        })
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&context, &stream, &host)?;
    let output = DeviceBuffer::<f32>::new(&context, elements)?;

    // Traffic model: two full-matrix reads for the reductions, one read and
    // one write each for exp shift and in-place normalize.
    let bytes = (elements as u64)
        .checked_mul(6)
        .and_then(|value| value.checked_mul(4))
        .ok_or("traffic calculation overflow")?;

    let case = BenchmarkCase::new("nnis_softmax_row_f32", "f32")
        .with_dimension("rows", rows as u64)
        .with_dimension("cols", cols as u64)
        .with_dimension("block_size", u64::from(softmax.block_size()))
        .with_dimension("stages", 4)
        .with_work_items(elements as u64)
        .with_bytes_per_iteration(bytes);
    let report = benchmark_gpu(
        &context,
        &stream,
        case,
        BenchConfig::new(warmups, iterations),
        || {
            // SAFETY: all buffers and the exclusive workspace outlive this
            // harness, which synchronizes the end event per invocation.
            unsafe {
                softmax.enqueue_softmax_rows(&stream, &input, &output, rows, cols, &workspace)
            }
        },
    )?;

    // Validate the timed configuration after measurement.
    let actual = output.to_vec(&stream)?;
    if actual.len() != host.len() {
        return Err("row-softmax output length mismatch".into());
    }
    let mut max_absolute_error = 0.0_f64;
    for row in 0..rows {
        let slice = &host[row * cols..(row + 1) * cols];
        let maximum = slice
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |acc, value| acc.max(f64::from(value)));
        let exponentials: Vec<f64> = slice
            .iter()
            .map(|&value| f64::from(value) - maximum)
            .map(f64::exp)
            .collect();
        let total: f64 = exponentials.iter().sum();
        for (col, (&actual, &exponential)) in actual[row * cols..(row + 1) * cols]
            .iter()
            .zip(&exponentials)
            .enumerate()
        {
            let expected = exponential / total;
            max_absolute_error = max_absolute_error.max((f64::from(actual) - expected).abs());
            let tolerance = 1.0e-5_f64 * expected.abs().max(1.0e-30) + 1.0e-7;
            if (f64::from(actual) - expected).abs() > tolerance {
                return Err(format!(
                    "row-softmax mismatch at row {row} col {col}: {actual} != {expected}"
                )
                .into());
            }
        }
    }

    let result = RowSoftmaxBenchmark {
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
