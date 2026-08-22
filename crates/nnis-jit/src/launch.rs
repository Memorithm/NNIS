use nnis_sys::driver as drv;
use nnis_rt::error::{NnisError, Result};
use nnis_rt::stream_event::Stream;
use std::ffi::c_void;

#[derive(Debug, Clone, Copy)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn new(x: u32, y: u32, z: u32) -> Self { Dim3 { x, y, z } }
    pub const fn x(x: u32) -> Self { Dim3 { x, y:1, z:1 } }
}

pub struct KernelLaunch<'a> {
    pub func: nnis_sys::CUfunction,
    pub grid: Dim3,
    pub block: Dim3,
    pub shared: u32,
    pub stream: &'a Stream,
    pub args: Vec<*mut c_void>,
}

impl<'a> KernelLaunch<'a> {
    pub fn launch(&self) -> Result<()> {
        let api = drv::api()?;
        let ctx = self.stream.ctx();
        ctx.set_current()?;
        // Ensure args are pinned for duration of call
        let arg_ptrs: Vec<*mut c_void> = self.args.iter().map(|p| *p as *mut c_void).collect();
        let mut arg_vec: Vec<*mut c_void> = arg_ptrs;
        unsafe {
            let rc = (api.cuLaunchKernel)(
                self.func,
                self.grid.x, self.grid.y, self.grid.z,
                self.block.x, self.block.y, self.block.z,
                self.shared,
                self.stream.raw(),
                arg_vec.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if rc != 0 {
                return Err(NnisError::driver("cuLaunchKernel", rc));
            }
        }
        Ok(())
    }
}
