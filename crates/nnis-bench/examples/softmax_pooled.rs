//! Softmax pipeline scratch strategies versus the same GPU work.
//!
//! Per iteration all three paths run the identical four-stage stable
//! softmax; only scratch strategy differs:
//!
//! 1. `sync_alloc` - two scalars plus reduction workspace allocated and
//!    freed synchronously every call (the historical default)
//! 2. `pooled` - same allocations served stream-ordered from a memory pool
//! 3. `preallocated` - caller-owned reusable scratch (allocator-free floor)
//!
//! Every path is validated against an f64 oracle after timing.

use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase};
use nnis_jit::JitCompiler;
use nnis_kernels::F32Softmax;
use nnis_rt::{gpu_context, DeviceBuffer, Stream, StreamOrderedAllocator};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct SizeResult {
    elements: usize,
    sync_alloc_median_ms: f64,
    pooled_median_ms: f64,
    preallocated_median_ms: f64,
    pooled_speedup_over_sync: f64,
    max_absolute_error: f64,
    correctness_validated: bool,
}

#[derive(Debug, Serialize)]
struct PipelineBenchmark {
    schema_version: u32,
    warmups: usize,
    iterations: usize,
    results: Vec<SizeResult>,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn expected(input: &[f32]) -> Vec<f64> {
    let maximum = input
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, v| acc.max(f64::from(v)));
    let exponentials: Vec<f64> = input
        .iter()
        .map(|&v| f64::from(v) - maximum)
        .map(f64::exp)
        .collect();
    let total: f64 = exponentials.iter().sum();
    exponentials.into_iter().map(|v| v / total).collect()
}

fn median(report: &nnis_bench::BenchmarkReport) -> f64 {
    report.statistics.median_ms
}

fn validate(actual: &[f32], expected: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    let mut max_error = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        max_error = max_error.max((f64::from(actual) - expected).abs());
        if (f64::from(actual) - expected).abs() > 2.0e-6_f64.max(expected.abs() * 1.0e-5) {
            return Err(format!("softmax mismatch at {index}: {actual} != {expected}").into());
        }
    }
    Ok(max_error)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 5)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 50)?;
    let Some(context) = gpu_context() else {
        return Err("no CUDA device".into());
    };
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let softmax = F32Softmax::load(&context, &compiler)?;
    let allocator = StreamOrderedAllocator::new(&stream)?;

    let mut results = Vec::new();
    for &elements in &[65_536_usize, 1_048_576, 16_777_216] {
        let host: Vec<f32> = (0..elements)
            .map(|i| ((i * 13 % 977) as f32 - 488.0) * 0.5)
            .collect();
        let expected_values = expected(&host);
        let input = DeviceBuffer::from_host(&context, &stream, &host)?;
        let output = DeviceBuffer::<f32>::new(&context, elements)?;

        // 1. Historical default: synchronous scratch allocation per call.
        let case = BenchmarkCase::new("softmax_sync_alloc_scratch", "f32")
            .with_dimension("elements", elements as u64);
        let sync_report = benchmark_gpu(
            &context,
            &stream,
            case,
            BenchConfig::new(warmups, iterations),
            || softmax.softmax(&context, &stream, &input, &output),
        )?;

        // 2. Pooled stream-ordered scratch per call.
        let case = BenchmarkCase::new("softmax_pooled_scratch", "f32")
            .with_dimension("elements", elements as u64);
        let pooled_report = benchmark_gpu(
            &context,
            &stream,
            case,
            BenchConfig::new(warmups, iterations),
            || softmax.softmax_pooled(&stream, &input, &output, &allocator),
        )?;

        // 3. Pre-allocated reusable scratch: allocator-free floor.
        let workspace = softmax.reduction().workspace(&context, elements)?;
        let max_scratch = DeviceBuffer::<f32>::new(&context, 1)?;
        let sum_scratch = DeviceBuffer::<f32>::new(&context, 1)?;
        let case = BenchmarkCase::new("softmax_preallocated_scratch", "f32")
            .with_dimension("elements", elements as u64);
        let pre_report = benchmark_gpu(
            &context,
            &stream,
            case,
            BenchConfig::new(warmups, iterations),
            || softmax.softmax_into(&stream, &input, &output, &workspace),
        )?;
        // softmax_into still allocates its two scalars per call; measure the
        // true floor through the enqueue API with fully owned buffers.
        let floor_report = benchmark_gpu(
            &context,
            &stream,
            BenchmarkCase::new("softmax_enqueue_owned_scratch", "f32")
                .with_dimension("elements", elements as u64),
            BenchConfig::new(warmups, iterations),
            || {
                // SAFETY: all borrows outlive each synchronized invocation.
                unsafe {
                    softmax.enqueue_softmax(
                        &stream,
                        &input,
                        &output,
                        &max_scratch,
                        &sum_scratch,
                        &workspace,
                    )
                }
            },
        )?;
        let _ = pre_report;

        // Post-timing validation of each timed pattern.
        softmax.softmax(&context, &stream, &input, &output)?;
        let sync_error = validate(&output.to_vec(&stream)?, &expected_values)?;
        softmax
            .softmax_pooled(&stream, &input, &output, &allocator)
            .unwrap();
        let pooled_error = validate(&output.to_vec(&stream)?, &expected_values)?;
        // SAFETY: borrows retained through the synchronization below.
        unsafe {
            softmax.enqueue_softmax(
                &stream,
                &input,
                &output,
                &max_scratch,
                &sum_scratch,
                &workspace,
            )
        }?;
        stream.synchronize()?;
        let floor_error = validate(&output.to_vec(&stream)?, &expected_values)?;
        let max_absolute_error = sync_error.max(pooled_error).max(floor_error);

        results.push(SizeResult {
            elements,
            sync_alloc_median_ms: median(&sync_report),
            pooled_median_ms: median(&pooled_report),
            preallocated_median_ms: median(&floor_report),
            pooled_speedup_over_sync: median(&sync_report) / median(&pooled_report),
            max_absolute_error,
            correctness_validated: true,
        });
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&PipelineBenchmark {
            schema_version: 1,
            warmups,
            iterations,
            results,
        })?
    );
    Ok(())
}
