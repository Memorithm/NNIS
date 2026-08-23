use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::F32Softmax;
use nnis_rt::{Context, Device, DeviceBuffer, Stream};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct SoftmaxBenchmark {
    schema_version: u32,
    report: BenchmarkReport,
    gpu_probability_sum_f32: f32,
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

/// Bytes moved per softmax invocation: three elementwise passes move six
/// f32 streams (exp shift r+w, normalize r+w, plus the sum-reduction read),
/// and two scalar reductions contribute their multi-pass traffic.
fn traffic_bytes(elements: usize, block_size: u32) -> Option<u64> {
    let span = usize::try_from(block_size).ok()?.checked_mul(2)?;
    fn reduction_bytes(elements: usize, span: usize) -> Option<u64> {
        let mut current = elements;
        let mut values_moved = 0_u64;
        while current > 0 {
            let output = current.div_ceil(span);
            values_moved = values_moved
                .checked_add(u64::try_from(current).ok()?)?
                .checked_add(u64::try_from(output).ok()?)?;
            if output == 1 {
                break;
            }
            current = output;
        }
        values_moved.checked_mul(4)
    }
    let elementwise = u64::try_from(elements)
        .ok()?
        .checked_mul(6)?
        .checked_mul(4)?;
    let reductions = reduction_bytes(elements, span)?.checked_mul(2)?;
    elementwise.checked_add(reductions)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let elements = env_usize("NNIS_BENCH_ELEMENTS", 1 << 20)?;
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 100)?;

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let softmax = F32Softmax::load(&context, &compiler)?;
    let workspace = softmax.reduction().workspace(&context, elements)?;
    let host = (0..elements)
        .map(|index| {
            let spread = ((index % 17) as f32 - 8.0) * 37.5;
            let ripple = ((index * 7 % 101) as f32 - 50.0) * 0.25;
            spread + ripple
        })
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&context, &stream, &host)?;
    let output = DeviceBuffer::<f32>::new(&context, elements)?;
    let max_scratch = DeviceBuffer::<f32>::new(&context, 1)?;
    let sum_scratch = DeviceBuffer::<f32>::new(&context, 1)?;
    let bytes = traffic_bytes(elements, softmax.block_size())
        .ok_or("softmax traffic calculation overflow")?;

    let case = BenchmarkCase::new("nnis_softmax_f32", "f32")
        .with_dimension("elements", elements as u64)
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
            // SAFETY: all buffers and the exclusive workspace/scalars outlive
            // this harness, which synchronizes the end event per invocation.
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

    // Validate the timed configuration after measurement.
    let actual = output.to_vec(&stream)?;
    if actual.len() != host.len() {
        return Err("softmax output length mismatch".into());
    }
    let maximum = host
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(f64::from(value)));
    let exponentials: Vec<f64> = host
        .iter()
        .map(|&value| f64::from(value) - maximum)
        .map(f64::exp)
        .collect();
    let total: f64 = exponentials.iter().sum();
    let mut max_absolute_error = 0.0_f64;
    for (index, (&actual, &exponential)) in actual.iter().zip(&exponentials).enumerate() {
        let expected = exponential / total;
        max_absolute_error = max_absolute_error.max((f64::from(actual) - expected).abs());
        let tolerance = 1.0e-5_f64 * expected.abs().max(1.0e-30) + 1.0e-7;
        if (f64::from(actual) - expected).abs() > tolerance {
            return Err(format!("softmax mismatch at {index}: {actual} != {expected}").into());
        }
    }
    let gpu_probability_sum_f32: f32 = actual.iter().sum();
    if (gpu_probability_sum_f32 - 1.0).abs() > 2.0e-3 {
        return Err(format!(
            "probabilities sum to {gpu_probability_sum_f32}; expected approximately 1"
        )
        .into());
    }

    let result = SoftmaxBenchmark {
        schema_version: 1,
        report,
        gpu_probability_sum_f32,
        max_absolute_error,
        correctness_validated: true,
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
