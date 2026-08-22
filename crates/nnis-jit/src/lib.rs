//! Just-in-time CUDA compilation via NVRTC and driver module load.

pub mod error;
pub mod program;
pub mod module;
pub mod launch;

pub use program::{CompileOptions, JitProgram, ProgramCacheKey};
pub use module::Module;
pub use launch::{Dim3, KernelLaunch};

#[cfg(test)]
mod tests {
    use crate::program::{CompileOptions, JitProgram};
    use nnis_rt::{gpu_context, DeviceBuffer, Stream, Context};
    use std::sync::Arc;

    #[test]
    fn jit_vector_add_roundtrip() {
        let ctx = match nnis_rt::gpu_context() {
            Some(c) => c,
            None => {
                eprintln!("skipped: no CUDA device");
                return;
            }
        };
        let stream = Stream::new(&ctx).unwrap();
        let n = 1024usize;
        let kernel_src = r#"
            extern "C" __global__ void vec_add(const float* a, const float* b, float* c, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }
        "#;
        let opts = CompileOptions::for_device(&ctx);
        let prog = JitProgram::compile(kernel_src, opts).expect("compile");
        let ptx = prog.get_ptx().expect("ptx");
        let module = crate::Module::load_from_ptx(&ctx, &ptx).expect("load module");
        let func = module.get_function("vec_add").expect("get func");

        // allocate buffers
        let a_host: Vec<f32> = (0..n as u32).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..n as u32).map(|i| (i as f32) * 2.0).collect();
        let a_buf = DeviceBuffer::<f32>::from_host(&ctx, &stream, &a_host).unwrap();
        let b_buf = DeviceBuffer::<f32>::from_host(&ctx, &stream, &b_host).unwrap();
        let mut c_buf = DeviceBuffer::<f32>::new(&ctx, n).unwrap();

        // launch
        let n_i32 = n as i32;
        // args pointers must live for duration of launch
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            a_buf.device_ptr() as *mut _,
            b_buf.device_ptr() as *mut _,
            c_buf.device_ptr() as *mut _,
            &n_i32 as *const _ as *mut _,
        ];
        let grid = crate::launch::Dim3 { x: ((n as u32 + 255)/256), y:1, z:1 };
        let block = crate::launch::Dim3 { x:256, y:1, z:1 };
        // Use raw launch
        use nnis_sys::driver as drv;
        ctx.set_current().unwrap();
        let api = drv::api().unwrap();
        unsafe {
            let rc = (api.cuLaunchKernel)(
                func.handle,
                grid.x, grid.y, grid.z,
                block.x, block.y, block.z,
                0,
                stream.raw(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            assert_eq!(rc, 0, "launch failed");
        }
        stream.synchronize().unwrap();
        let c_host = c_buf.to_vec(&stream).unwrap();
        for i in 0..n {
            let expected = a_host[i] + b_host[i];
            assert!((c_host[i] - expected).abs() < 1e-5, "mismatch at {}: {} vs {}", i, c_host[i], expected);
        }
    }
}
