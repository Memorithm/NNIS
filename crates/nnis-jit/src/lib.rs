//! Runtime CUDA compilation, module ownership, and validated kernel launch.

mod cache;
mod frontend;
mod launch;
mod module;
mod program;

pub use cache::{CompiledCode, JitCompiler};
pub use frontend::{
    FrontendArtifactBoundary, FrontendQualification, FrontendQualificationEvidence,
    FrontendQualificationEvidenceError, KernelFrontend, KernelFrontendContract,
};
pub use launch::{Dim3, KernelArgs, KernelLaunch, KernelParameter, LaunchConfig};
pub use module::{Kernel, KernelAttributes, Module, OccupancyRecommendation};
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

        let attributes = kernel.attributes().unwrap();
        assert!(attributes.max_threads_per_block > 0);
        assert!(attributes.max_threads_per_block <= context.props().max_threads_per_block);
        assert!(attributes.registers_per_thread > 0);
        assert!(attributes.static_shared_memory_bytes <= context.props().shared_memory_per_block);
        assert_eq!(
            attributes.binary_version,
            Some((
                context.props().compute_capability.0 as u32,
                context.props().compute_capability.1 as u32,
            ))
        );
        assert_eq!(kernel.attributes().unwrap(), attributes);

        let occupancy = kernel.recommend_occupancy(0, None).unwrap();
        assert!(occupancy.block_size > 0);
        assert!(occupancy.block_size <= attributes.max_threads_per_block);
        assert!(occupancy.minimum_grid_size > 0);
        assert!(occupancy.active_blocks_per_multiprocessor > 0);
        assert_eq!(
            kernel.recommend_occupancy(0, None).unwrap(),
            occupancy
        );

        const N: usize = 1024;
        let left: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let right: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
        let mut output = vec![0.0_f32; N];
        let mut d_left = DeviceBuffer::<f32>::new(&context, N).unwrap();
        let mut d_right = DeviceBuffer::<f32>::new(&context, N).unwrap();
        let mut d_output = DeviceBuffer::<f32>::new(&context, N).unwrap();
        d_left.copy_from_host_async(&left, &stream).unwrap();
        d_right.copy_from_host_async(&right, &stream).unwrap();

        let launch = KernelLaunch::new(
            &kernel,
            LaunchConfig::linear(N as u32, occupancy.block_size),
        );
        launch
            .launch(
                &stream,
                KernelArgs::new()
                    .push_device_ptr(&d_left)
                    .push_device_ptr(&d_right)
                    .push_device_ptr_mut(&mut d_output)
                    .push_scalar(N as i32),
            )
            .unwrap();
        d_output.copy_to_host_async(&mut output, &stream).unwrap();
        stream.synchronize().unwrap();

        for i in 0..N {
            assert_eq!(output[i], left[i] + right[i]);
        }
    }

    #[test]
    fn launch_rejects_invalid_configuration() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let options = CompileOptions::for_device(&context);
        let compiler = JitCompiler::new();
        let ptx = compiler.compile_ptx(VECTOR_ADD, &options).unwrap();
        let module = Module::load(&context, &ptx).unwrap();
        let kernel = module.get_function("vector_add").unwrap();

        let invalid = KernelLaunch::new(
            &kernel,
            LaunchConfig::new(Dim3::new(1, 1, 1), Dim3::new(0, 1, 1), 0),
        );
        let stream = Stream::new(&context).unwrap();
        let err = invalid
            .launch(&stream, KernelArgs::new())
            .expect_err("zero block dimension must fail");
        assert_eq!(err.kind(), ErrorKind::InvalidValue);
    }
}
