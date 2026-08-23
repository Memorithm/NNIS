//! Event-timed deterministic top-k selection with post-timing validation.
//!
//! Every selection round rescans the working copy, so derived traffic counts
//! `4*(k+1)*n` read bytes plus the `8*k` written pair bytes.
use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase};
use nnis_jit::JitCompiler;
use nnis_kernels::F32TopK;
use nnis_rt::{gpu_context, DeviceBuffer, Stream};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct TopKBenchmark {
    schema_version: u32,
    elements: usize,
    k: usize,
    median_ms: f64,
    gigabytes_per_second: f64,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let elements = env_usize("NNIS_BENCH_ELEMENTS", 262_144)?;
    let k = env_usize("NNIS_BENCH_K", 32)?;
    let Some(context) = gpu_context() else {
        return Err("no CUDA device".into());
    };
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let top_k = F32TopK::load(&context, &compiler)?;

    let host: Vec<f32> = (0..elements)
        .map(|index| ((index * 37 % 1_009) as f32 - 504.0) / 64.0)
        .collect();
    let input = DeviceBuffer::from_host(&context, &stream, &host)?;
    let values = DeviceBuffer::<f32>::new(&context, k)?;
    let indices = DeviceBuffer::<u32>::new(&context, k)?;
    let workspace = top_k.workspace(&context, elements)?;

    let read_bytes = 4u64 * (k as u64 + 1) * elements as u64;
    let write_bytes = 8 * k as u64;
    let bytes = read_bytes
        .checked_add(write_bytes)
        .ok_or("top-k traffic overflow")?;

    let case = BenchmarkCase::new("nnis_topk_f32", "iterative-tree-argmax")
        .with_dimension("elements", elements as u64)
        .with_dimension("k", k as u64)
        .with_work_items(elements as u64)
        .with_bytes_per_iteration(bytes);
    let report = benchmark_gpu(&context, &stream, case, BenchConfig::new(20, 200), || {
        // SAFETY: every borrow outlives the harness-synchronized iteration.
        unsafe { top_k.enqueue_top_k(&stream, &input, &values, &indices, k, &workspace) }
    })?;

    // Validate the timed configuration once against the CPU oracle.
    unsafe { top_k.enqueue_top_k(&stream, &input, &values, &indices, k, &workspace)? };
    stream.synchronize()?;
    let mut scratch = host.clone();
    for round in 0..k {
        let mut best = 0usize;
        for candidate in 1..scratch.len() {
            if scratch[candidate] > scratch[best] {
                best = candidate;
            }
        }
        if indices.to_vec(&stream)?[round] != best as u32 {
            return Err(format!(
                "top-k mismatch at round {round}: gpu index {}, cpu index {best}",
                indices.to_vec(&stream)?[round]
            )
            .into());
        }
        scratch[best] = f32::NEG_INFINITY;
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&TopKBenchmark {
            schema_version: 1,
            elements,
            k,
            median_ms: report.statistics.median_ms,
            gigabytes_per_second: report
                .throughput
                .and_then(|t| t.gigabytes_per_second)
                .unwrap_or(0.0),
        })?
    );
    Ok(())
}
