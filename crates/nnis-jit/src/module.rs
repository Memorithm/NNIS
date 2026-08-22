//! CUDA module and function ownership.

use crate::{CodeKind, CompiledCode};
use nnis_rt::context::Context;
use nnis_rt::error::{NnisError, Result};
use nnis_sys::driver;
use std::ffi::CString;
use std::sync::Arc;

/// A loaded CUDA module. Clones share one driver module handle.
#[derive(Clone)]
pub struct Module {
    inner: Arc<ModuleInner>,
}

struct ModuleInner {
    handle: nnis_sys::CUmodule,
    context: Arc<Context>,
}

// SAFETY: CUDA module handles are context-scoped and every operation first
// establishes `context` as current on the calling thread.
unsafe impl Send for ModuleInner {}
unsafe impl Sync for ModuleInner {}

impl Module {
    pub fn load(context: &Arc<Context>, code: &CompiledCode) -> Result<Self> {
        match code.kind() {
            CodeKind::Ptx => Self::load_from_ptx(context, code.bytes()),
            CodeKind::Cubin => Self::load_from_cubin(context, code.bytes()),
        }
    }

    /// Load PTX text. A missing terminal NUL is added for the duration of the
    /// synchronous driver call.
    pub fn load_from_ptx(context: &Arc<Context>, ptx: &[u8]) -> Result<Self> {
        if ptx.is_empty() {
            return Err(NnisError::invalid_input("PTX image is empty"));
        }
        let terminated;
        let image = if ptx.last() == Some(&0) {
            ptx
        } else {
            terminated = ptx
                .iter()
                .copied()
                .chain(std::iter::once(0))
                .collect::<Vec<_>>();
            &terminated
        };
        Self::load_image(context, image, "PTX")
    }

    pub fn load_from_cubin(context: &Arc<Context>, cubin: &[u8]) -> Result<Self> {
        if cubin.is_empty() {
            return Err(NnisError::invalid_input("CUBIN image is empty"));
        }
        Self::load_image(context, cubin, "CUBIN")
    }

    fn load_image(context: &Arc<Context>, image: &[u8], kind: &str) -> Result<Self> {
        context.set_current()?;
        let api = driver::api()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: image remains alive for the complete synchronous load call;
        // the context is current and all optional JIT arrays are omitted.
        let result = unsafe {
            (api.cuModuleLoadDataEx)(
                &mut handle,
                image.as_ptr().cast(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if result != 0 {
            return Err(NnisError::driver("cuModuleLoadDataEx", result)
                .with("image_kind", kind)
                .with("image_bytes", image.len()));
        }
        Ok(Self {
            inner: Arc::new(ModuleInner {
                handle,
                context: Arc::clone(context),
            }),
        })
    }

    /// Resolve a kernel and retain this module for the kernel's lifetime.
    pub fn get_function(&self, name: &str) -> Result<Kernel> {
        let name_c = CString::new(name)
            .map_err(|_| NnisError::invalid_input("kernel name contains an interior NUL"))?;
        self.inner.context.set_current()?;
        let api = driver::api()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: the module is live, name is NUL-terminated, and the
        // out-pointer is valid.
        let result =
            unsafe { (api.cuModuleGetFunction)(&mut handle, self.inner.handle, name_c.as_ptr()) };
        if result != 0 {
            return Err(NnisError::driver("cuModuleGetFunction", result).with("kernel", name));
        }
        Ok(Kernel {
            handle,
            name: Arc::from(name),
            module: Arc::clone(&self.inner),
        })
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.inner.context
    }
}

impl std::fmt::Debug for Module {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Module")
            .field("device_ordinal", &self.inner.context.device_ordinal())
            .finish_non_exhaustive()
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if let Ok(api) = driver::api() {
            let _ = self.context.set_current();
            // SAFETY: this is the final owner of the live module handle.
            unsafe {
                let _ = (api.cuModuleUnload)(self.handle);
            }
        }
    }
}

/// A CUDA kernel function that keeps its defining module loaded.
#[derive(Clone)]
pub struct Kernel {
    handle: nnis_sys::CUfunction,
    name: Arc<str>,
    module: Arc<ModuleInner>,
}

impl Kernel {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.module.context
    }

    pub(crate) fn raw(&self) -> nnis_sys::CUfunction {
        self.handle
    }
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Kernel")
            .field("name", &self.name)
            .field("device_ordinal", &self.module.context.device_ordinal())
            .finish_non_exhaustive()
    }
}
