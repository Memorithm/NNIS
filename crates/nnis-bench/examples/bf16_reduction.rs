//! Event-timed packed-bf16 sum reduction with post-timing validation.
//!
//! Traffic counts input bytes only (2 per element); the scalar output and
//! scratch passes are negligible at these sizes.
use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase};
use nnis_jit::JitCompiler;
use nnis_kernels::Bf16Reduction;
use nnis_rt::{gpu_context, DeviceBuffer, Stream};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Bf16ReductionBenchmark {
    schema_version: u32,
    elements: usize,
    median_ms: f64,
    gigabytes_per_second: f64,
    relative_error: f64,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let elements = env_usize("NNIS_BENCH_ELEMENTS", 1 << 20)?;
    let Some(context) = gpu_context() else {
        return Err("no CUDA device".into());
    };
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let reduction = Bf16Reduction::load(&context, &compiler)?;

    let host: Vec<f32> = (0..elements)
        .map(|index| ((index * 37 % 1_009) as f32 - 504.0) / 64.0)
        .collect();
    let bits: Vec<u16> = host
        .iter()
        .map(|&value| nnis_rt::f32_to_bf16_rne(value))
        .collect();
    let input = DeviceBuffer::from_host(&context, &stream, &bits)?;
    let workspace = reduction.workspace(&context, elements)?;
    let output = DeviceBuffer::<f32>::new(&context, 1)?;

    // Input bytes only; scratch traffic is orders of magnitude smaller.
    let bytes = (elements as u64)
        .checked_mul(2)
        .ok_or("bf16 reduction traffic overflow")?;

    let case = BenchmarkCase::new("nnis_reduce_sum_packed_bf16", "bf16-storage-f32-math")
        .with_dimension("elements", elements as u64)
        .with_work_items(elements as u64)
        .with_bytes_per_iteration(bytes);
    let report = benchmark_gpu(&context, &stream, case, BenchConfig::new(20, 200), || {
        // SAFETY: every borrow outlives the harness-synchronized iteration.
        unsafe { reduction.enqueue_sum(&stream, &input, &output, &workspace) }
    })?;

    // Validate the timed configuration against a widened host sum.
    let actual = reduction.sum(&stream, &input)?;
    let expected: f32 = bits
        .iter()
        .map(|&b| nnis_rt::bf16_bits_to_f32(b))
        .fold(0.0_f32, |accumulator, value| accumulator + value);
    let relative_error = ((actual - expected) / expected).abs() as f64;
    if !relative_error.is_finite() || relative_error > 1.0e-3 {
        return Err(format!("bf16 sum mismatch: {actual} vs {expected}").into());
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&Bf16ReductionBenchmark {
            schema_version: 1,
            elements,
            median_ms: report.statistics.median_ms,
            gigabytes_per_second: report
                .throughput
                .and_then(|t| t.gigabytes_per_second)
                .unwrap_or(0.0),
            relative_error,
        })?
    );
    Ok(())
}
