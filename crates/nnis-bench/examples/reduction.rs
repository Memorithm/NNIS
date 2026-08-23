use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::F32Reduction;
use nnis_rt::{Context, Device, DeviceBuffer, Stream};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct ReductionBenchmark {
    schema_version: u32,
    report: BenchmarkReport,
    passes: usize,
    gpu_sum: f32,
    reference_sum_f64: f64,
    absolute_error: f64,
    forward_error_bound: f64,
    correctness_validated: bool,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn traffic_and_passes(elements: usize, block_size: u32) -> Option<(u64, usize)> {
    let span = usize::try_from(block_size).ok()?.checked_mul(2)?;
    let mut current = elements;
    let mut values_moved = 0_u64;
    let mut passes = 0;
    while current > 0 {
        let output = current.div_ceil(span);
        values_moved = values_moved
            .checked_add(u64::try_from(current).ok()?)?
            .checked_add(u64::try_from(output).ok()?)?;
        passes += 1;
        if output == 1 {
            break;
        }
        current = output;
    }
    Some((values_moved.checked_mul(4)?, passes))
}

fn error_bound(input: &[f32]) -> f64 {
    let depth = (usize::BITS - (input.len() - 1).leading_zeros()) as f64 + 1.0;
    let epsilon = f32::EPSILON as f64;
    let gamma = depth * epsilon / (1.0 - depth * epsilon);
    let magnitude = input
        .iter()
        .map(|value| f64::from(value.abs()))
        .sum::<f64>();
    (gamma * magnitude).max(epsilon)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let elements = env_usize("NNIS_BENCH_ELEMENTS", 1 << 24)?;
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 100)?;
    if elements == 0 {
        return Err("NNIS_BENCH_ELEMENTS must be non-zero".into());
    }

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let reduction = F32Reduction::load(&context, &compiler)?;
    let workspace = reduction.workspace(&context, elements)?;
    let output = DeviceBuffer::<f32>::new(&context, 1)?;
    let host = (0..elements)
        .map(|index| {
            let numerator = (index * 37 % 1_009) as f32 - 504.0;
            numerator / 127.0
        })
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&context, &stream, &host)?;
    let (bytes, passes) = traffic_and_passes(elements, reduction.block_size())
        .ok_or("reduction traffic calculation overflow")?;
    let case = BenchmarkCase::new("nnis_reduce_sum_f32", "f32")
        .with_dimension("elements", elements as u64)
        .with_dimension("block_size", u64::from(reduction.block_size()))
        .with_dimension("passes", passes as u64)
        .with_work_items(elements as u64)
        .with_bytes_per_iteration(bytes);
    let report = benchmark_gpu(
        &context,
        &stream,
        case,
        BenchConfig::new(warmups, iterations),
        || {
            // SAFETY: all buffers and the exclusive workspace outlive this
            // harness, which synchronizes the end event for each invocation.
            unsafe { reduction.enqueue_sum(&stream, &input, &output, &workspace) }
        },
    )?;

    let gpu_sum = output.to_vec(&stream)?[0];
    let reference_sum_f64 = host.iter().map(|&value| f64::from(value)).sum::<f64>();
    let absolute_error = (f64::from(gpu_sum) - reference_sum_f64).abs();
    let forward_error_bound = error_bound(&host);
    if absolute_error > forward_error_bound {
        return Err(format!(
            "reduction error {absolute_error} exceeds bound {forward_error_bound}: {gpu_sum} != {reference_sum_f64}"
        )
        .into());
    }

    let result = ReductionBenchmark {
        schema_version: 1,
        report,
        passes,
        gpu_sum,
        reference_sum_f64,
        absolute_error,
        forward_error_bound,
        correctness_validated: true,
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
