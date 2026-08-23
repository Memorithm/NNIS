//! Event-timed bf16 vector-add with post-timing bit-exact validation.
use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase};
use nnis_jit::JitCompiler;
use nnis_kernels::Bf16Elementwise;
use nnis_rt::{gpu_context, DeviceBuffer, Stream};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Bf16Benchmark {
    schema_version: u32,
    elements: usize,
    median_ms: f64,
    gigabytes_per_second: f64,
    bit_exact_validated: bool,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let elements = env_usize("NNIS_BENCH_ELEMENTS", 16_777_216)?;
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 100)?;
    let Some(context) = gpu_context() else {
        return Err("no CUDA device".into());
    };
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let bf16 = Bf16Elementwise::load(&context, &compiler)?;

    let left_host: Vec<u16> = (0..elements)
        .map(|i| nnis_rt::f32_to_bf16_rne(((i % 977) as f32 - 488.0) * 0.25))
        .collect();
    let right_host: Vec<u16> = (0..elements)
        .map(|i| nnis_rt::f32_to_bf16_rne(((i % 521) as f32 - 260.0) * 0.125))
        .collect();
    let expected: Vec<u16> = left_host
        .iter()
        .zip(&right_host)
        .map(|(&l, &r)| {
            nnis_rt::f32_to_bf16_rne(nnis_rt::bf16_bits_to_f32(l) + nnis_rt::bf16_bits_to_f32(r))
        })
        .collect();
    let left = DeviceBuffer::from_host(&context, &stream, &left_host)?;
    let right = DeviceBuffer::from_host(&context, &stream, &right_host)?;
    let output = DeviceBuffer::<u16>::new(&context, elements)?;

    let case = BenchmarkCase::new("nnis_bf16_vector_add_f32acc", "bf16")
        .with_dimension("elements", elements as u64)
        .with_dimension("block_size", u64::from(bf16.block_size()))
        .with_work_items(elements as u64)
        .with_bytes_per_iteration(elements as u64 * 6);
    let report = benchmark_gpu(
        &context,
        &stream,
        case,
        BenchConfig::new(warmups, iterations),
        || bf16.vector_add(&stream, &left, &right, &output),
    )?;

    assert_eq!(output.to_vec(&stream)?, expected);

    println!(
        "{}",
        serde_json::to_string_pretty(&Bf16Benchmark {
            schema_version: 1,
            elements,
            median_ms: report.statistics.median_ms,
            gigabytes_per_second: report
                .throughput
                .and_then(|t| t.gigabytes_per_second)
                .unwrap_or(0.0),
            bit_exact_validated: true,
        })?
    );
    Ok(())
}
