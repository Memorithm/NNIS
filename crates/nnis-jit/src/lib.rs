//! Runtime CUDA compilation, module ownership, and validated kernel launch.

mod cache;
mod launch;
mod module;
mod program;

pub use cache::{CompiledCode, JitCompiler};
pub use launch::{Dim3, KernelArgs, KernelLaunch, KernelParameter, LaunchConfig};
pub use module::{Kernel, Module};
pub use program::{CodeKind, CompileOptions, JitProgram, ProgramCacheKey};

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::{gpu_context, DeviceBuffer, ErrorKind, Stream};
    use std::sync::Arc;

    const VECTOR_ADD: &str = r#"
        extern "C" __global__ void vector_add(
            const float* left,
            const float* right,
            float* output,
            int elements
        ) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < elements) {
                output[index] = left[index] + right[index];
            }
        }
    "#;

    #[test]
    fn jit_vector_add_roundtrip_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let options = CompileOptions::for_device(&context);
        let compiler = JitCompiler::new();

        let ptx = compiler.compile_ptx(VECTOR_ADD, &options).unwrap();
        assert!(!ptx.bytes().is_empty());
        let cached = compiler.compile_ptx(VECTOR_ADD, &options).unwrap();
        assert!(Arc::ptr_eq(&ptx, &cached), "second compile must hit cache");

        let cubin = compiler.compile_cubin(VECTOR_ADD, &options).unwrap();
        assert!(cubin.bytes().starts_with(b"\x7fELF"));
        assert_eq!(compiler.len(), 2);

        let module = Module::load(&context, &ptx).unwrap();
        let kernel = module.get_function("vector_add").unwrap();
        assert!(module.get_function("missing_kernel").is_err());

        let elements = 1_025usize;
        let left_host = (0..elements)
            .map(|index| index as f32 * 0.25 - 10.0)
            .collect::<Vec<_>>();
        let right_host = (0..elements)
            .map(|index| index as f32 * -0.5 + 3.0)
            .collect::<Vec<_>>();
        let left = DeviceBuffer::from_host(&context, &stream, &left_host).unwrap();
        let right = DeviceBuffer::from_host(&context, &stream, &right_host).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, elements).unwrap();

        let mut arguments = KernelArgs::with_capacity(4, 3);
        arguments
            .push_buffer(&left)
            .push_buffer(&right)
            .push_buffer(&output)
            .push(elements as i32);
        let config = LaunchConfig::for_num_elements(elements, 256).unwrap();
        let launch = KernelLaunch::new(&kernel, &stream, config);
        // SAFETY: argument order/types match VECTOR_ADD; all referenced
        // objects remain alive through the synchronization below.
        unsafe { launch.launch(&mut arguments) }.unwrap();
        stream.synchronize().unwrap();

        let actual = output.to_vec(&stream).unwrap();
        for (index, ((left, right), actual)) in
            left_host.iter().zip(&right_host).zip(&actual).enumerate()
        {
            let expected = left + right;
            assert_eq!(*actual, expected, "mismatch at element {index}");
        }
    }

    #[test]
    fn compilation_failure_preserves_nvrtc_log() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let error = match JitProgram::compile(
            "extern \"C\" __global__ void broken( {",
            CompileOptions::for_device(&context),
        ) {
            Ok(_) => panic!("invalid CUDA source unexpectedly compiled"),
            Err(error) => error,
        };
        assert!(matches!(error.kind(), ErrorKind::Compile { .. }));
        let rendered = error.to_string();
        assert!(rendered.contains("compiler log"), "{rendered}");
        assert!(rendered.contains("error"), "{rendered}");
    }
}
