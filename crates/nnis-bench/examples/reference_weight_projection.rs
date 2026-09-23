use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::JitCompiler;
use nnis_kernels::{F32Gemv, F32Int2Gemv, F32Int4Gemv, F32SparseCscGemv};
use nnis_model::{
    dequantize_int2_ternary_reference_v1, dequantize_int4_symmetric_reference_v1,
    quantize_int2_ternary_reference_v1, quantize_int4_symmetric_reference_v1,
    sparsify_matrix_csc_reference_v1,
};
use nnis_rt::{Context, Device, DeviceBuffer, Stream};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct ProjectionVariantReport {
    representation: String,
    report: BenchmarkReport,
    logical_k: usize,
    logical_n: usize,
    resident_weight_bytes: u64,
    max_output_absolute_error: f64,
    correctness_validated: bool,
}

#[derive(Debug, Serialize)]
struct ReferenceWeightProjectionBenchmark {
    schema_version: u32,
    sparse_threshold: f32,
    dense_f32: ProjectionVariantReport,
    int4: ProjectionVariantReport,
    int2: ProjectionVariantReport,
    sparse_csc: ProjectionVariantReport,
    claim_boundary: String,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn env_f32(name: &str, default: f32) -> Result<f32, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn ordered_projection(input: &[f32], weight: &[f32], k: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|col| {
            (0..k).fold(0.0_f32, |value, row| {
                input[row].mul_add(weight[row * n + col], value)
            })
        })
        .collect()
}

fn validate_output(
    actual: &[f32],
    expected: &[f32],
    label: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label} output length {} does not match expected {}",
            actual.len(),
            expected.len()
        )
        .into());
    }
    let mut max_error = 0.0_f64;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if actual.to_bits() != expected.to_bits() {
            return Err(format!(
                "{label} ordered projection mismatch at output {index}: {actual} != {expected}"
            )
            .into());
        }
        max_error = max_error.max((f64::from(actual) - f64::from(expected)).abs());
    }
    Ok(max_error)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let k = env_usize("NNIS_BENCH_K", 1_024)?;
    let n = env_usize("NNIS_BENCH_N", 1_024)?;
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 10)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 50)?;
    let sparse_threshold = env_f32("NNIS_BENCH_SPARSE_THRESHOLD", 1.0)?;
    if k == 0 || n == 0 {
        return Err("NNIS_BENCH_K and NNIS_BENCH_N must be non-zero".into());
    }
    if !sparse_threshold.is_finite() || sparse_threshold < 0.0 {
        return Err("NNIS_BENCH_SPARSE_THRESHOLD must be finite and non-negative".into());
    }
    let elements = k.checked_mul(n).ok_or("projection shape overflows usize")?;

    let input_host = (0..k)
        .map(|index| ((index * 29 % 61) as f32 - 30.0) * 0.03125)
        .collect::<Vec<_>>();
    let weight_host = (0..elements)
        .map(|index| {
            let coarse = ((index * 17 % 101) as f32 - 50.0) * 0.0625;
            let structured_zero_band = if index % 7 < 3 { 0.125 } else { 1.0 };
            coarse * structured_zero_band
        })
        .collect::<Vec<_>>();

    let int4_host = quantize_int4_symmetric_reference_v1(&weight_host)?;
    let int4_reconstructed = dequantize_int4_symmetric_reference_v1(&int4_host)?;
    let int2_host = quantize_int2_ternary_reference_v1(&weight_host)?;
    let int2_reconstructed = dequantize_int2_ternary_reference_v1(&int2_host)?;
    let sparse_host = sparsify_matrix_csc_reference_v1(&weight_host, k, n, sparse_threshold)?;
    let sparse_reconstructed = nnis_model::densify_matrix_csc_reference_v1(&sparse_host)?;

    let dense_expected = ordered_projection(&input_host, &weight_host, k, n);
    let int4_expected = ordered_projection(&input_host, &int4_reconstructed, k, n);
    let int2_expected = ordered_projection(&input_host, &int2_reconstructed, k, n);
    let sparse_expected = ordered_projection(&input_host, &sparse_reconstructed, k, n);

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let dense_kernel = F32Gemv::load(&context, &compiler)?;
    let int4_kernel = F32Int4Gemv::load(&context, &compiler)?;
    let int2_kernel = F32Int2Gemv::load(&context, &compiler)?;
    let sparse_kernel = F32SparseCscGemv::load(&context, &compiler)?;

    let input = DeviceBuffer::from_host(&context, &stream, &input_host)?;
    let dense_weight = DeviceBuffer::from_host(&context, &stream, &weight_host)?;
    let int4_weight = DeviceBuffer::from_host(&context, &stream, &int4_host.packed_values)?;
    let int4_scale = DeviceBuffer::from_host(&context, &stream, &[int4_host.scale])?;
    let int2_weight = DeviceBuffer::from_host(&context, &stream, &int2_host.packed_values)?;
    let int2_scale = DeviceBuffer::from_host(&context, &stream, &[int2_host.scale])?;
    let sparse_offsets = DeviceBuffer::from_host(&context, &stream, &sparse_host.column_offsets)?;
    let sparse_rows = DeviceBuffer::from_host(&context, &stream, &sparse_host.row_indices)?;
    let sparse_values = DeviceBuffer::from_host(&context, &stream, &sparse_host.values)?;

    let dense_output = DeviceBuffer::<f32>::new(&context, n)?;
    let int4_output = DeviceBuffer::<f32>::new(&context, n)?;
    let int2_output = DeviceBuffer::<f32>::new(&context, n)?;
    let sparse_output = DeviceBuffer::<f32>::new(&context, n)?;

    let config = BenchConfig::new(warmups, iterations);
    let work_items = u64::try_from(elements)?;

    let dense_report = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("nnis_project_kn_f32", "f32")
            .with_dimension("k", u64::try_from(k)?)
            .with_dimension("n", u64::try_from(n)?)
            .with_work_items(work_items),
        config,
        || {
            // SAFETY: all buffers and kernels outlive the event-synchronized
            // benchmark invocation.
            unsafe {
                dense_kernel.enqueue_project_kn(&stream, &input, &dense_weight, &dense_output, k, n)
            }
        },
    )?;
    let dense_actual = dense_output.to_vec(&stream)?;
    let dense_error = validate_output(&dense_actual, &dense_expected, "dense-f32")?;

    let int4_report = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("nnis_project_kn_f32_int4_weight", "f32xint4")
            .with_dimension("k", u64::try_from(k)?)
            .with_dimension("n", u64::try_from(n)?)
            .with_work_items(work_items),
        config,
        || {
            // SAFETY: all buffers and kernels outlive the event-synchronized
            // benchmark invocation.
            unsafe {
                int4_kernel.enqueue_project_kn(
                    &stream,
                    &input,
                    &int4_weight,
                    &int4_scale,
                    &int4_output,
                    k,
                    n,
                )
            }
        },
    )?;
    let int4_actual = int4_output.to_vec(&stream)?;
    let int4_error = validate_output(&int4_actual, &int4_expected, "int4")?;

    let int2_report = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("nnis_project_kn_f32_int2_weight", "f32xint2")
            .with_dimension("k", u64::try_from(k)?)
            .with_dimension("n", u64::try_from(n)?)
            .with_work_items(work_items),
        config,
        || {
            // SAFETY: all buffers and kernels outlive the event-synchronized
            // benchmark invocation.
            unsafe {
                int2_kernel.enqueue_project_kn(
                    &stream,
                    &input,
                    &int2_weight,
                    &int2_scale,
                    &int2_output,
                    k,
                    n,
                )
            }
        },
    )?;
    let int2_actual = int2_output.to_vec(&stream)?;
    let int2_error = validate_output(&int2_actual, &int2_expected, "int2")?;

    let sparse_report = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("nnis_project_kn_f32_sparse_csc", "f32_sparse_csc")
            .with_dimension("k", u64::try_from(k)?)
            .with_dimension("n", u64::try_from(n)?)
            .with_dimension("nnz", u64::try_from(sparse_host.nnz())?)
            .with_work_items(u64::try_from(sparse_host.nnz())?),
        config,
        || {
            // SAFETY: all buffers and kernels outlive the event-synchronized
            // benchmark invocation.
            unsafe {
                sparse_kernel.enqueue_project_kn(
                    &stream,
                    &input,
                    &sparse_offsets,
                    &sparse_rows,
                    &sparse_values,
                    &sparse_output,
                    k,
                    n,
                )
            }
        },
    )?;
    let sparse_actual = sparse_output.to_vec(&stream)?;
    let sparse_error = validate_output(&sparse_actual, &sparse_expected, "sparse-csc")?;

    let result = ReferenceWeightProjectionBenchmark {
        schema_version: 1,
        sparse_threshold,
        dense_f32: ProjectionVariantReport {
            representation: "dense-f32-source".to_string(),
            report: dense_report,
            logical_k: k,
            logical_n: n,
            resident_weight_bytes: u64::try_from(dense_weight.size_bytes())?,
            max_output_absolute_error: dense_error,
            correctness_validated: true,
        },
        int4: ProjectionVariantReport {
            representation: "symmetric-signed-int4-reference".to_string(),
            report: int4_report,
            logical_k: k,
            logical_n: n,
            resident_weight_bytes: u64::try_from(int4_weight.size_bytes() + int4_scale.size_bytes())?,
            max_output_absolute_error: int4_error,
            correctness_validated: true,
        },
        int2: ProjectionVariantReport {
            representation: "ternary-int2-reference".to_string(),
            report: int2_report,
            logical_k: k,
            logical_n: n,
            resident_weight_bytes: u64::try_from(int2_weight.size_bytes() + int2_scale.size_bytes())?,
            max_output_absolute_error: int2_error,
            correctness_validated: true,
        },
        sparse_csc: ProjectionVariantReport {
            representation: "magnitude-sparse-csc-reference".to_string(),
            report: sparse_report,
            logical_k: k,
            logical_n: n,
            resident_weight_bytes: u64::try_from(
                sparse_offsets.size_bytes()
                    + sparse_rows.size_bytes()
                    + sparse_values.size_bytes(),
            )?,
            max_output_absolute_error: sparse_error,
            correctness_validated: true,
        },
        claim_boundary: "isolated projection benchmark only; not full-model qualification, model-quality evidence, or a retained performance result until run on a clean physical target"
            .to_string(),
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
