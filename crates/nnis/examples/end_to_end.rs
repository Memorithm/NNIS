use nnis::prelude::*;
use std::sync::Arc;
use std::time::Instant;

const AFFINE_SOURCE: &str = r#"
extern "C" __global__ void example_affine(
    const float* input,
    float* output,
    float scale,
    float bias,
    unsigned long long elements
) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = fmaf(input[index], scale, bias);
    }
}
"#;

fn main() -> Result<()> {
    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;

    let elements = 1_000_003usize;
    let host = (0..elements)
        .map(|index| (index % 8_191) as f32 * 0.000_5 - 2.0)
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&context, &stream, &host)?;
    let output = DeviceBuffer::<f32>::new(&context, elements)?;

    let compiler = JitCompiler::new();
    let compile_started = Instant::now();
    let code = compiler.compile_cubin(AFFINE_SOURCE, &CompileOptions::for_device(&context))?;
    let compile_ms = compile_started.elapsed().as_secs_f64() * 1_000.0;
    let cached = compiler.compile_cubin(AFFINE_SOURCE, &CompileOptions::for_device(&context))?;
    assert!(Arc::ptr_eq(&code, &cached));

    let module = Module::load(&context, &code)?;
    let kernel = module.get_function("example_affine")?;
    let config = LaunchConfig::for_num_elements(elements, 256)?;
    let scale = -0.75_f32;
    let bias = 1.125_f32;
    let enqueue = || -> Result<()> {
        let mut arguments = KernelArgs::with_capacity(5, 2);
        arguments
            .push_buffer(&input)
            .push_buffer(&output)
            .push(scale)
            .push(bias)
            .push(elements as u64);
        let launch = KernelLaunch::new(&kernel, &stream, config);
        // SAFETY: the argument order and widths match `example_affine`, and
        // every borrowed object remains alive through the synchronization.
        unsafe { launch.launch(&mut arguments) }
    };

    for _ in 0..10 {
        enqueue()?;
    }
    stream.synchronize()?;
    let start = Event::new(&context)?;
    let end = Event::new(&context)?;
    start.record(&stream)?;
    enqueue()?;
    end.record(&stream)?;
    end.synchronize()?;
    let kernel_ms = end.elapsed_ms(&start)?;

    let actual = output.to_vec(&stream)?;
    for (index, (&actual, &input)) in actual.iter().zip(&host).enumerate() {
        let expected = input.mul_add(scale, bias);
        let tolerance = 2.0e-6_f32.max(expected.abs() * 2.0e-6);
        if (actual - expected).abs() > tolerance {
            return Err(NnisError::unsupported(format!(
                "validation mismatch at {index}: {actual} != {expected}"
            )));
        }
    }

    println!(
        "{} | {} elements | JIT {:.3} ms | kernel {:.6} ms | validated",
        context.props().fingerprint(),
        elements,
        compile_ms,
        kernel_ms
    );
    Ok(())
}
