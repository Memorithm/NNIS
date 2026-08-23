use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::Bf16Gemm;
use nnis_rt::{bf16_bits_to_f32, f32_to_bf16_rne, Context, Device, DeviceBuffer, Stream};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Bf16GemmBenchmark {
    schema_version: u32,
    report: BenchmarkReport,
    m: usize,
    n: usize,
    k: usize,
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
    let m = env_usize("NNIS_BENCH_ROWS", 2_048)?;
    let n = env_usize("NNIS_BENCH_COLS", 2_048)?;
    let k = env_usize("NNIS_BENCH_K", 2_048)?;
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 100)?;

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let gemm = Bf16Gemm::load(&context, &compiler)?;

    // Coarse magnitudes with exact binary fractions keep every stored
    // operand close to its intended value after RNE narrowing.
    let a_host: Vec<f32> = (0..m * k)
        .map(|index| (((index * 13 % 97) as f32 - 48.0) * 0.0625) + ((index % 5) as f32 - 2.0))
        .collect();
    let b_host: Vec<f32> = (0..k * n)
        .map(|index| ((index * 29 % 61) as f32 - 30.0) * 0.125)
        .collect();
    let a_bits: Vec<u16> = a_host.iter().copied().map(f32_to_bf16_rne).collect();
    let b_bits: Vec<u16> = b_host.iter().copied().map(f32_to_bf16_rne).collect();
    let matrix_a = DeviceBuffer::from_host(&context, &stream, &a_bits)?;
    let matrix_b = DeviceBuffer::from_host(&context, &stream, &b_bits)?;
    let output = DeviceBuffer::<u16>::new(&context, m * n)?;

    // Ideal traffic model with packed-bf16 operands (2 bytes each) and a
    // packed-bf16 output.
    let bytes = (m as u64)
        .checked_mul(k as u64)
        .and_then(|value| value.checked_add((k as u64).checked_mul(n as u64)?))
        .and_then(|value| value.checked_add((m as u64).checked_mul(n as u64)?))
        .and_then(|value| value.checked_mul(2))
        .ok_or("bf16 gemm traffic calculation overflow")?;
    let macs = (m as u64)
        .checked_mul(n as u64)
        .and_then(|value| value.checked_mul(k as u64))
        .ok_or("bf16 gemm work calculation overflow")?;

    let case = BenchmarkCase::new("nnis_bf16_gemm_f32acc", "packed_bf16")
        .with_dimension("m", m as u64)
        .with_dimension("n", n as u64)
        .with_dimension("k", k as u64)
        .with_dimension("tile_side", u64::from(gemm.tile_side()))
        .with_work_items(macs)
        .with_bytes_per_iteration(bytes);
    let report = benchmark_gpu(
        &context,
        &stream,
        case,
        BenchConfig::new(warmups, iterations),
        || {
            // SAFETY: all buffers outlive this harness, which synchronizes
            // the end event for each invocation.
            unsafe { gemm.enqueue_gemm(&stream, &matrix_a, &matrix_b, &output, m, n, k) }
        },
    )?;

    // Validate the timed configuration after measurement against an f64
    // oracle over widened inputs inside a tolerance that reflects bf16
    // output quantization.
    let actual = output.to_vec(&stream)?;
    if actual.len() != m * n {
        return Err("bf16 gemm output length mismatch".into());
    }
    let mut max_absolute_error = 0.0_f64;
    for row in 0..m {
        for col in 0..n {
            let expected: f64 = (0..k)
                .map(|depth| {
                    f64::from(bf16_bits_to_f32(a_bits[row * k + depth]))
                        * f64::from(bf16_bits_to_f32(b_bits[depth * n + col]))
                })
                .sum();
            let error = (f64::from(bf16_bits_to_f32(actual[row * n + col])) - expected).abs();
            max_absolute_error = max_absolute_error.max(error);
            if error > 1.0e-1_f64.max(expected.abs() * 4.0e-3) {
                return Err(format!(
                    "bf16 gemm mismatch at ({row}, {col}): {} != {expected}",
                    bf16_bits_to_f32(actual[row * n + col])
                )
                .into());
            }
        }
    }

    let result = Bf16GemmBenchmark {
        schema_version: 1,
        report,
        m,
        n,
        k,
        max_absolute_error,
        correctness_validated: true,
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
