//! Pooled versus pre-allocated scratch across a multi-stage pipeline.
//!
//! Per iteration each path runs `alloc -> kernel -> free`; the pooled path
//! uses stream-ordered pool allocations (`cuMemAllocFromPoolAsync`) whose
//! drops enqueue `cuMemFreeAsync`, while the baseline reuses one
//! pre-allocated `DeviceBuffer` triple. Outputs are validated after timing.

use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase};
use nnis_jit::{CompileOptions, JitCompiler, KernelArgs, KernelLaunch, LaunchConfig, Module};
use nnis_rt::{gpu_context, Context, DeviceBuffer, PooledBuffer, Stream, StreamOrderedAllocator};
use serde::Serialize;

const SOURCE: &str = r#"
extern "C" __global__ void nnis_bench_vector_add_f32(
    const float* left,
    const float* right,
    float* output,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = left[index] + right[index];
    }
}
"#;

const BLOCK_SIZE: u32 = 256;

#[derive(Debug, Serialize)]
struct SizeResult {
    elements: usize,
    pooled_median_ms: f64,
    preallocated_median_ms: f64,
    speedup_preallocated_over_pooled: f64,
    max_absolute_error: f64,
    correctness_validated: bool,
}

#[derive(Debug, Serialize)]
struct PoolBenchmark {
    schema_version: u32,
    iterations: usize,
    warmups: usize,
    results: Vec<SizeResult>,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn median_ms(report: &nnis_bench::BenchmarkReport) -> f64 {
    report.statistics.median_ms
}

struct Harness<'a> {
    context: &'a Arc<Context>,
    stream: &'a Stream,
    kernel: &'a nnis_jit::Kernel,
    left_host: &'a [f32],
    right_host: &'a [f32],
    warmups: usize,
    iterations: usize,
}

use std::sync::Arc;

impl<'a> Harness<'a> {
    fn launch(&self, left: u64, right: u64, output: u64, elements: usize) -> nnis_rt::Result<()> {
        let mut args = KernelArgs::with_capacity(4, 3);
        args.push(left)
            .push(right)
            .push(output)
            .push(elements as u64);
        let launch = KernelLaunch::new(
            self.kernel,
            self.stream,
            LaunchConfig::for_num_elements(elements, BLOCK_SIZE)?,
        );
        // SAFETY: argument order/widths match the kernel above; every buffer
        // pointer stays alive through the harness's synchronization.
        // SAFETY: argument order/widths verified against the kernel above.
        unsafe { launch.launch(&mut args) }
    }

    fn run_pooled(
        &self,
        allocator: &StreamOrderedAllocator,
        elements: usize,
    ) -> Result<(PooledBuffer<f32>, nnis_bench::BenchmarkReport), Box<dyn std::error::Error>> {
        let left_host = &self.left_host[..elements];
        let right_host = &self.right_host[..elements];
        let report = benchmark_gpu(
            self.context,
            self.stream,
            BenchmarkCase::new("pooled_alloc_kernel_free", "f32")
                .with_dimension("elements", elements as u64),
            BenchConfig::new(self.warmups, self.iterations),
            || {
                // Inputs stay uninitialized on purpose: both paths must run
                // identical GPU work so only allocator cost differs.
                let left: PooledBuffer<f32> = allocator.alloc(elements)?;
                let right: PooledBuffer<f32> = allocator.alloc(elements)?;
                let output: PooledBuffer<f32> = allocator.alloc(elements)?;
                // SAFETY: pointers stay alive until the enqueued free that
                // Drop issues on this same stream, which orders reuse.
                self.launch(
                    left.device_ptr(),
                    right.device_ptr(),
                    output.device_ptr(),
                    elements,
                )?;
                drop(output);
                drop(right);
                drop(left);
                Ok(())
            },
        )?;
        // Fresh allocation for post-timing validation of the timed pattern.
        let left: PooledBuffer<f32> = allocator.alloc(elements)?;
        left.copy_from_host(self.stream, left_host)?;
        let right: PooledBuffer<f32> = allocator.alloc(elements)?;
        right.copy_from_host(self.stream, right_host)?;
        let output: PooledBuffer<f32> = allocator.alloc(elements)?;
        self.launch(
            left.device_ptr(),
            right.device_ptr(),
            output.device_ptr(),
            elements,
        )?;
        self.stream.synchronize()?;
        Ok((output, report))
    }

    fn run_preallocated(
        &self,
        left: &DeviceBuffer<f32>,
        right: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        elements: usize,
    ) -> Result<nnis_bench::BenchmarkReport, Box<dyn std::error::Error>> {
        let (l, r, o) = (left.device_ptr(), right.device_ptr(), output.device_ptr());
        let report = benchmark_gpu(
            self.context,
            self.stream,
            BenchmarkCase::new("preallocated_reused_buffers", "f32")
                .with_dimension("elements", elements as u64),
            BenchConfig::new(self.warmups, self.iterations),
            || {
                // SAFETY: buffers outlive the harness's synchronization.
                self.launch(l, r, o, elements)
            },
        )?;
        self.launch(l, r, o, elements)?;
        self.stream.synchronize()?;
        Ok(report)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 5)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 50)?;
    let Some(context) = gpu_context() else {
        return Err("no CUDA device".into());
    };
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let options = CompileOptions::for_device(&context);
    let code = compiler.compile_cubin(SOURCE, &options)?;
    let module = Module::load(&context, &code)?;
    let kernel = module.get_function("nnis_bench_vector_add_f32")?;
    let allocator = StreamOrderedAllocator::new(&stream)?;

    let max_elements = 16_777_216;
    let left_host: Vec<f32> = (0..max_elements).map(|i| (i % 977) as f32 * 0.25).collect();
    let right_host: Vec<f32> = (0..max_elements)
        .map(|i| (i % 521) as f32 * -0.125)
        .collect();
    let pre_left = DeviceBuffer::from_host(&context, &stream, &left_host[..max_elements])?;
    let pre_right = DeviceBuffer::from_host(&context, &stream, &right_host[..max_elements])?;
    let pre_output = DeviceBuffer::<f32>::new(&context, max_elements)?;

    let harness = Harness {
        context: &context,
        stream: &stream,
        kernel: &kernel,
        left_host: &left_host,
        right_host: &right_host,
        warmups,
        iterations,
    };

    let mut results = Vec::new();
    for &elements in &[256_usize, 4_096, 65_536, 1_048_576, 16_777_216] {
        let (pooled_out, pooled_report) = harness.run_pooled(&allocator, elements)?;
        let pooled_median = median_ms(&pooled_report);

        let expected: Vec<f32> = left_host[..elements]
            .iter()
            .zip(&right_host[..elements])
            .map(|(l, r)| l + r)
            .collect();
        let actual = pooled_out.to_vec(&stream)?;
        let mut max_error = 0.0_f64;
        for (a, e) in actual.iter().zip(&expected) {
            max_error = max_error.max((f64::from(*a) - f64::from(*e)).abs());
        }
        if actual != expected {
            return Err(format!("pooled path mismatch at {elements} elements").into());
        }

        let pre_report = harness.run_preallocated(&pre_left, &pre_right, &pre_output, elements)?;
        let actual = pre_output.to_vec(&stream)?[..elements].to_vec();
        if actual != expected {
            return Err(format!("preallocated path mismatch at {elements} elements").into());
        }

        results.push(SizeResult {
            elements,
            pooled_median_ms: pooled_median,
            preallocated_median_ms: median_ms(&pre_report),
            speedup_preallocated_over_pooled: pooled_median / median_ms(&pre_report),
            max_absolute_error: max_error,
            correctness_validated: true,
        });
    }

    let report = PoolBenchmark {
        schema_version: 1,
        iterations,
        warmups,
        results,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
