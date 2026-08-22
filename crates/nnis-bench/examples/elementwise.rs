use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase};
use nnis_jit::JitCompiler;
use nnis_kernels::F32Elementwise;
use nnis_rt::{Context, Device, DeviceBuffer, Stream};

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
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
    let kernels = F32Elementwise::load(&context, &compiler)?;
    let host = (0..elements)
        .map(|index| (index % 4_093) as f32 * 0.000_25 - 0.5)
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&context, &stream, &host)?;
    let output = DeviceBuffer::<f32>::new(&context, elements)?;
    let scale = -0.75_f32;
    let bytes = (elements as u64)
        .checked_mul(2 * std::mem::size_of::<f32>() as u64)
        .ok_or("benchmark byte count overflow")?;
    let case = BenchmarkCase::new("nnis_scale_f32", "f32")
        .with_dimension("elements", elements as u64)
        .with_work_items(elements as u64)
        .with_bytes_per_iteration(bytes);

    let report = benchmark_gpu(
        &context,
        &stream,
        case,
        BenchConfig::new(warmups, iterations),
        || {
            // SAFETY: all captured objects outlive the harness, which waits for
            // the end event after every measured launch.
            unsafe { kernels.enqueue_scale(&stream, &input, &output, scale) }
        },
    )?;

    let actual = output.to_vec(&stream)?;
    for (index, (&actual, &input)) in actual.iter().zip(&host).enumerate() {
        if actual != input * scale {
            return Err(format!("result mismatch at element {index}").into());
        }
    }

    println!("{}", report.to_json_pretty()?);
    Ok(())
}
