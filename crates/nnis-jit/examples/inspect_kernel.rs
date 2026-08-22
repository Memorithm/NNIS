use nnis_jit::{CompileOptions, JitCompiler, Module};
use nnis_rt::{Context, Device, Result};

const SOURCE: &str = r#"
extern "C" __global__ void inspect_scale(float* values, float scale) {
    const unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    values[index] *= scale;
}
"#;

fn main() -> Result<()> {
    let device = Device::first()?;
    let context = Context::new(&device)?;
    let compiler = JitCompiler::new();
    let code = compiler.compile_cubin(SOURCE, &CompileOptions::for_device(&context))?;
    let module = Module::load(&context, &code)?;
    let kernel = module.get_function("inspect_scale")?;
    let attributes = kernel.attributes()?;
    let occupancy = kernel.recommend_occupancy(0, None)?;

    println!("device: {}", context.props().fingerprint());
    println!("kernel: {}", kernel.name());
    println!("attributes: {attributes:#?}");
    println!("occupancy: {occupancy:#?}");
    Ok(())
}
