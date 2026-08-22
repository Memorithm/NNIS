use nnis_sys::driver as drv;
use nnis_rt::error::{NnisError, Result};
use nnis_rt::context::Context;

pub struct Module {
    pub handle: nnis_sys::CUmodule,
    pub ctx: std::sync::Arc<Context>,
}

impl Module {
    pub fn load_from_ptx(ctx: &std::sync::Arc<Context>, ptx: &[u8]) -> Result<Self> {
        ctx.set_current()?;
        let api = drv::api()?;
        let mut module = std::ptr::null_mut();
        let rc = unsafe {
            (api.cuModuleLoadDataEx)(
                &mut module,
                ptx.as_ptr() as *const _,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(NnisError::driver("cuModuleLoadDataEx", rc));
        }
        Ok(Module { handle: module, ctx: ctx.clone() })
    }

    pub fn get_function(&self, name: &str) -> Result<KernelFunction> {
        let cname = std::ffi::CString::new(name).unwrap();
        let mut func = std::ptr::null_mut();
        let api = drv::api()?;
        self.ctx.set_current()?;
        let rc = unsafe { (api.cuModuleGetFunction)(&mut func, self.handle, cname.as_ptr()) };
        if rc != 0 {
            return Err(NnisError::driver("cuModuleGetFunction", rc).with("name", name));
        }
        Ok(KernelFunction { handle: func })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(api) = drv::api() {
            let _ = self.ctx.set_current();
            unsafe { (api.cuModuleUnload)(self.handle); }
        }
    }
}

pub struct KernelFunction {
    pub handle: nnis_sys::CUfunction,
}
