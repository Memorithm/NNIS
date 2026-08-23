use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase};
use nnis_rt::{Context, Device, DeviceBuffer, PinnedBuffer, Stream};
use serde::Serialize;
use std::sync::Arc;

#[derive(Debug, Serialize)]
struct TransferComparison {
    schema_version: u32,
    bytes: usize,
    pinned_h2d_ms: f64,
    pageable_h2d_ms: f64,
    pinned_d2h_ms: f64,
    pageable_d2h_ms: f64,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

struct CaseParameters<'a> {
    context: &'a Arc<Context>,
    stream: &'a Stream,
    name: &'static str,
    direction: u64,
    elements: u64,
    warmups: usize,
    iterations: usize,
}

fn time_case<F>(
    parameters: &CaseParameters<'_>,
    enqueue: F,
) -> Result<f64, Box<dyn std::error::Error>>
where
    F: FnMut() -> nnis_rt::Result<()>,
{
    let case = BenchmarkCase::new(parameters.name, "f32")
        .with_dimension("elements", parameters.elements)
        .with_dimension("direction", parameters.direction);
    let report = benchmark_gpu(
        parameters.context,
        parameters.stream,
        case,
        BenchConfig::new(parameters.warmups, parameters.iterations),
        enqueue,
    )?;
    Ok(report.statistics.median_ms)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let elements = env_usize("NNIS_BENCH_ELEMENTS", 1 << 24)?;
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 100)?;

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let device_buffer = DeviceBuffer::<f32>::new(&context, elements)?;
    let host_pageable: Vec<f32> = (0..elements).map(|index| index as f32).collect();
    let mut host_pinned_source = PinnedBuffer::<f32>::new(&context, elements)?;
    host_pinned_source
        .as_mut_slice()
        .copy_from_slice(&host_pageable);
    let mut host_pageable_sink = vec![0.0_f32; elements];
    let mut host_pinned_sink = PinnedBuffer::<f32>::new(&context, elements)?;

    // Warm both paths once so pool/driver setup does not skew first samples.
    device_buffer.copy_from_host(&stream, &host_pageable)?;
    device_buffer.copy_from_host(&stream, host_pinned_source.as_slice())?;
    device_buffer.copy_to_host(&stream, &mut host_pageable_sink)?;
    device_buffer.copy_to_host(&stream, host_pinned_sink.as_mut_slice())?;

    let bytes = elements * std::mem::size_of::<f32>();
    let parameters = CaseParameters {
        context: &context,
        stream: &stream,
        name: "transfers",
        direction: 0,
        elements: elements as u64,
        warmups,
        iterations,
    };
    let h2d_pinned = time_case(&parameters, || {
        // SAFETY: harness synchronizes the end event while all borrows
        // of the pinned staging memory remain live.
        unsafe { device_buffer.copy_from_host_async(&stream, host_pinned_source.as_slice()) }
    })?;
    let h2d_pageable = time_case(&parameters, || {
        // SAFETY: same harness synchronization; Vec remains live.
        unsafe { device_buffer.copy_from_host_async(&stream, &host_pageable) }
    })?;
    let d2h_pinned = time_case(&parameters, || {
        // SAFETY: harness synchronizes before the mutable borrow unwinds.
        unsafe { device_buffer.copy_to_host_async(&stream, host_pinned_sink.as_mut_slice()) }
    })?;
    let d2h_pageable = time_case(&parameters, || {
        // SAFETY: harness synchronization covers the mutable borrow.
        unsafe { device_buffer.copy_to_host_async(&stream, &mut host_pageable_sink) }
    })?;

    // Validate that every path actually moved the expected data.
    device_buffer.copy_from_host(&stream, &host_pageable)?;
    device_buffer.copy_to_host(&stream, &mut host_pageable_sink)?;
    assert!(
        host_pageable_sink == host_pageable,
        "pageable validation failed"
    );
    device_buffer.copy_to_host(&stream, host_pinned_sink.as_mut_slice())?;
    assert!(
        host_pinned_sink.as_slice() == host_pageable.as_slice(),
        "pinned validation failed"
    );

    let result = TransferComparison {
        schema_version: 1,
        bytes,
        pinned_h2d_ms: h2d_pinned,
        pageable_h2d_ms: h2d_pageable,
        pinned_d2h_ms: d2h_pinned,
        pageable_d2h_ms: d2h_pageable,
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
