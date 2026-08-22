use nnis_bench::{
    benchmark_gpu, summarize_samples_ms, BenchConfig, BenchmarkCase, BenchmarkMetadata,
    BenchmarkReport, Throughput, TimingStatistics,
};
use nnis_jit::{CompileOptions, JitCompiler, KernelArgs, KernelLaunch, LaunchConfig, Module};
use nnis_rt::{Context, Device, DeviceBuffer, NnisError, Result, Stream};
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;

const KERNEL_SOURCE: &str = r#"
extern "C" __global__ void perf_scale(
    const float* input,
    float* output,
    float scale,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = input[index] * scale;
    }
}
"#;

const COLD_COMPILES: usize = 5;
const CACHE_LOOKUPS: usize = 1_000;
const MODULE_LOADS: usize = 50;
const ALLOCATIONS: usize = 100;
const ARGUMENT_PACKS: usize = 10_000;
const HOST_LAUNCHES: usize = 1_000;
const LARGE_ELEMENTS: usize = 1 << 24;

#[derive(Debug, Serialize)]
struct HostMetric {
    samples: usize,
    statistics: TimingStatistics,
}

impl HostMetric {
    fn from_samples(samples_ms: Vec<f64>) -> Result<Self> {
        Ok(Self {
            samples: samples_ms.len(),
            statistics: summarize_samples_ms(&samples_ms)?,
        })
    }
}

#[derive(Debug, Serialize)]
struct GpuMetric {
    case: BenchmarkCase,
    config: BenchConfig,
    statistics: TimingStatistics,
    throughput: Option<Throughput>,
    samples_ms: Vec<f64>,
}

impl From<BenchmarkReport> for GpuMetric {
    fn from(report: BenchmarkReport) -> Self {
        Self {
            case: report.case,
            config: report.config,
            statistics: report.statistics,
            throughput: report.throughput,
            samples_ms: report.samples_ms,
        }
    }
}

#[derive(Debug, Serialize)]
struct BreakdownConfig {
    cold_compiles: usize,
    cache_lookups: usize,
    module_loads: usize,
    allocations: usize,
    argument_packs: usize,
    host_launches: usize,
    allocation_bytes: usize,
    large_elements: usize,
}

#[derive(Debug, Serialize)]
struct PerformanceBreakdown {
    schema_version: u32,
    metadata: BenchmarkMetadata,
    config: BreakdownConfig,
    nvrtc_library_load: HostMetric,
    jit_cold_compile: HostMetric,
    jit_cache_lookup: HostMetric,
    module_load: HostMetric,
    module_unload: HostMetric,
    allocation: HostMetric,
    deallocation: HostMetric,
    argument_pack_host: HostMetric,
    host_kernel_submission: HostMetric,
    empty_event_pair_gpu: GpuMetric,
    tiny_kernel_gpu: GpuMetric,
    large_kernel_gpu: GpuMetric,
    h2d_gpu: GpuMetric,
    d2h_gpu: GpuMetric,
    d2d_gpu: GpuMetric,
    correctness_validated: bool,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn measure_host<F>(iterations: usize, mut operation: F) -> Result<HostMetric>
where
    F: FnMut() -> Result<()>,
{
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        operation()?;
        samples.push(elapsed_ms(started));
    }
    HostMetric::from_samples(samples)
}

fn bytes_for<T>(elements: usize) -> Result<u64> {
    elements
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| NnisError::invalid_input("benchmark byte count overflow"))
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;

    let nvrtc_started = Instant::now();
    let nvrtc_version = nnis_sys::nvrtc::version()
        .ok_or_else(|| NnisError::unsupported("NVRTC version query failed"))?;
    std::hint::black_box(nvrtc_version);
    let nvrtc_library_load = HostMetric::from_samples(vec![elapsed_ms(nvrtc_started)])?;

    let compiler = JitCompiler::new();
    let mut cold_samples = Vec::with_capacity(COLD_COMPILES);
    let mut first_code = None;
    let mut first_options = None;
    for variant in 0..COLD_COMPILES {
        let options =
            CompileOptions::for_device(&context).with_option(format!("-DNNIS_VARIANT={variant}"));
        let started = Instant::now();
        let code = compiler.compile_cubin(KERNEL_SOURCE, &options)?;
        cold_samples.push(elapsed_ms(started));
        if variant == 0 {
            first_code = Some(code);
            first_options = Some(options);
        }
    }
    let jit_cold_compile = HostMetric::from_samples(cold_samples)?;
    let code = first_code.expect("first compile must produce code");
    let options = first_options.expect("first compile must retain options");
    let jit_cache_lookup = measure_host(CACHE_LOOKUPS, || {
        let cached = compiler.compile_cubin(KERNEL_SOURCE, &options)?;
        if !Arc::ptr_eq(&code, &cached) {
            return Err(NnisError::unsupported("JIT cache did not reuse its entry"));
        }
        std::hint::black_box(cached);
        Ok(())
    })?;

    let mut loaded_modules = Vec::with_capacity(MODULE_LOADS);
    let mut module_load_samples = Vec::with_capacity(MODULE_LOADS);
    for _ in 0..MODULE_LOADS {
        let started = Instant::now();
        let module = Module::load(&context, &code)?;
        module_load_samples.push(elapsed_ms(started));
        loaded_modules.push(module);
    }
    let module_load = HostMetric::from_samples(module_load_samples)?;
    let mut module_unload_samples = Vec::with_capacity(MODULE_LOADS);
    while let Some(module) = loaded_modules.pop() {
        let started = Instant::now();
        drop(module);
        module_unload_samples.push(elapsed_ms(started));
    }
    let module_unload = HostMetric::from_samples(module_unload_samples)?;

    let allocation_bytes = 4 * 1024 * 1024;
    let mut allocation_samples = Vec::with_capacity(ALLOCATIONS);
    let mut deallocation_samples = Vec::with_capacity(ALLOCATIONS);
    for _ in 0..ALLOCATIONS {
        let started = Instant::now();
        let buffer = DeviceBuffer::<u8>::new(&context, allocation_bytes)?;
        allocation_samples.push(elapsed_ms(started));
        let started = Instant::now();
        drop(buffer);
        deallocation_samples.push(elapsed_ms(started));
    }
    let allocation = HostMetric::from_samples(allocation_samples)?;
    let deallocation = HostMetric::from_samples(deallocation_samples)?;

    let module = Module::load(&context, &code)?;
    let kernel = module.get_function("perf_scale")?;
    let tiny_input = DeviceBuffer::from_host(&context, &stream, &[2.0_f32])?;
    let tiny_output = DeviceBuffer::<f32>::new(&context, 1)?;
    let tiny_config = LaunchConfig::for_num_elements(1, 1)?;
    stream.synchronize()?;
    let argument_pack_host = measure_host(ARGUMENT_PACKS, || {
        let mut arguments = KernelArgs::with_capacity(4, 2);
        arguments
            .push_buffer(&tiny_input)
            .push_buffer(&tiny_output)
            .push(0.5_f32)
            .push(1_u64);
        std::hint::black_box(arguments);
        Ok(())
    })?;
    let host_kernel_submission = measure_host(HOST_LAUNCHES, || {
        let mut arguments = KernelArgs::with_capacity(4, 2);
        arguments
            .push_buffer(&tiny_input)
            .push_buffer(&tiny_output)
            .push(0.5_f32)
            .push(1_u64);
        let launch = KernelLaunch::new(&kernel, &stream, tiny_config);
        // SAFETY: signature and widths match `perf_scale`; resources live
        // through the synchronization immediately after this batch.
        unsafe { launch.launch(&mut arguments) }
    })?;
    stream.synchronize()?;

    let empty_event_pair_gpu = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("empty_event_pair", "none"),
        BenchConfig::new(20, 100),
        || Ok(()),
    )?
    .into();
    let tiny_kernel_gpu = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("perf_scale", "f32")
            .with_dimension("elements", 1)
            .with_work_items(1)
            .with_bytes_per_iteration(8),
        BenchConfig::new(20, 100),
        || {
            let mut arguments = KernelArgs::with_capacity(4, 2);
            arguments
                .push_buffer(&tiny_input)
                .push_buffer(&tiny_output)
                .push(0.5_f32)
                .push(1_u64);
            let launch = KernelLaunch::new(&kernel, &stream, tiny_config);
            // SAFETY: signature and lifetimes match, and the harness waits on
            // the end event after every measured launch.
            unsafe { launch.launch(&mut arguments) }
        },
    )?
    .into();

    let large_host = (0..LARGE_ELEMENTS)
        .map(|index| (index % 8_191) as f32 * 0.000_25 - 0.5)
        .collect::<Vec<_>>();
    let mut host_output = vec![0.0_f32; LARGE_ELEMENTS];
    let large_input = DeviceBuffer::<f32>::new(&context, LARGE_ELEMENTS)?;
    let large_output = DeviceBuffer::<f32>::new(&context, LARGE_ELEMENTS)?;
    let transfer_bytes = bytes_for::<f32>(LARGE_ELEMENTS)?;
    let transfer_case = |name: &str| {
        BenchmarkCase::new(name, "f32")
            .with_dimension("elements", LARGE_ELEMENTS as u64)
            .with_bytes_per_iteration(transfer_bytes)
    };

    let h2d_gpu = benchmark_gpu(
        &context,
        &stream,
        transfer_case("h2d"),
        BenchConfig::new(5, 30),
        || {
            // SAFETY: the harness waits on an end event after every copy, and
            // all captured storage outlives the complete benchmark.
            unsafe { large_input.copy_from_host_async(&stream, &large_host) }
        },
    )?
    .into();
    let d2h_gpu = benchmark_gpu(
        &context,
        &stream,
        transfer_case("d2h"),
        BenchConfig::new(5, 30),
        || {
            // SAFETY: the harness preserves exclusive destination access and
            // waits on an end event after every copy.
            unsafe { large_input.copy_to_host_async(&stream, &mut host_output) }
        },
    )?
    .into();
    if host_output != large_host {
        return Err("D2H validation failed".into());
    }

    let d2d_gpu = benchmark_gpu(
        &context,
        &stream,
        transfer_case("d2d"),
        BenchConfig::new(5, 30),
        || {
            // SAFETY: both buffers outlive the harness, which waits on an end
            // event after every copy.
            unsafe { large_output.copy_from_buffer_async(&stream, &large_input) }
        },
    )?
    .into();
    if large_output.to_vec(&stream)? != large_host {
        return Err("D2D validation failed".into());
    }

    let large_config = LaunchConfig::for_num_elements(LARGE_ELEMENTS, 256)?;
    let scale = -0.75_f32;
    let large_kernel_gpu = benchmark_gpu(
        &context,
        &stream,
        BenchmarkCase::new("perf_scale", "f32")
            .with_dimension("elements", LARGE_ELEMENTS as u64)
            .with_work_items(LARGE_ELEMENTS as u64)
            .with_bytes_per_iteration(transfer_bytes * 2),
        BenchConfig::new(20, 100),
        || {
            let mut arguments = KernelArgs::with_capacity(4, 2);
            arguments
                .push_buffer(&large_input)
                .push_buffer(&large_output)
                .push(scale)
                .push(LARGE_ELEMENTS as u64);
            let launch = KernelLaunch::new(&kernel, &stream, large_config);
            // SAFETY: signature and lifetimes match, and the harness waits on
            // the end event after every measured launch.
            unsafe { launch.launch(&mut arguments) }
        },
    )?
    .into();
    let actual = large_output.to_vec(&stream)?;
    for (index, (&actual, &input)) in actual.iter().zip(&large_host).enumerate() {
        if actual != input * scale {
            return Err(format!("kernel validation failed at {index}").into());
        }
    }

    let breakdown = PerformanceBreakdown {
        schema_version: 1,
        metadata: BenchmarkMetadata::collect(&context),
        config: BreakdownConfig {
            cold_compiles: COLD_COMPILES,
            cache_lookups: CACHE_LOOKUPS,
            module_loads: MODULE_LOADS,
            allocations: ALLOCATIONS,
            argument_packs: ARGUMENT_PACKS,
            host_launches: HOST_LAUNCHES,
            allocation_bytes,
            large_elements: LARGE_ELEMENTS,
        },
        nvrtc_library_load,
        jit_cold_compile,
        jit_cache_lookup,
        module_load,
        module_unload,
        allocation,
        deallocation,
        argument_pack_host,
        host_kernel_submission,
        empty_event_pair_gpu,
        tiny_kernel_gpu,
        large_kernel_gpu,
        h2d_gpu,
        d2h_gpu,
        d2d_gpu,
        correctness_validated: true,
    };
    println!("{}", serde_json::to_string_pretty(&breakdown)?);
    Ok(())
}
