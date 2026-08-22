//! Primary-context ownership.
//!
//! NNIS uses the device *primary context* (retained via
//! `cuDevicePrimaryCtxRetain`) so it can coexist with other CUDA users in the
//! process. Every raw call made through [`Context`] first makes the context
//! current on the calling thread; this keeps multi-threaded use correct and
//! costs a cheap idempotent `cuCtxSetCurrent` per call.

use crate::device::{Device, DeviceProps};
use crate::error::{NnisError, Result};
use nnis_sys::driver::{self};
use nnis_sys::CUcontext;
use std::sync::Arc;

/// An owned reference to a device primary context plus its cached properties.
#[derive(Debug)]
pub struct Context {
    raw: CUcontext,
    ordinal: i32,
    props: DeviceProps,
}

// SAFETY: CUcontext is process-global state; all entry points route through
// `set_current`, making concurrent use from multiple threads well-defined.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Retain the primary context of `device` and make it current.
    pub fn new(device: &Device) -> Result<Arc<Self>> {
        // Query fallible metadata before taking a primary-context retain so an
        // attribute failure cannot leak the retain count.
        let props = device.props()?;
        let api = driver::api()?;
        let mut ctx: CUcontext = std::ptr::null_mut();
        // SAFETY: out-pointer valid; retain is refcounted by the driver.
        let rc = unsafe { (api.cuDevicePrimaryCtxRetain)(&mut ctx, device.ordinal()) };
        if rc != 0 {
            return Err(
                NnisError::driver("cuDevicePrimaryCtxRetain", rc).with("ordinal", device.ordinal())
            );
        }
        let ctx = Arc::new(Context {
            raw: ctx,
            ordinal: device.ordinal(),
            props,
        });
        ctx.set_current()?;
        Ok(ctx)
    }

    /// Device ordinal backing this context.
    pub fn device_ordinal(&self) -> i32 {
        self.ordinal
    }

    /// Cached static device properties.
    pub fn props(&self) -> &DeviceProps {
        &self.props
    }

    /// Make this context current on the calling thread.
    pub fn set_current(&self) -> Result<()> {
        let api = driver::api()?;
        // SAFETY: handle is a retained, valid context.
        let rc = unsafe { (api.cuCtxSetCurrent)(self.raw) };
        if rc != 0 {
            return Err(NnisError::driver("cuCtxSetCurrent", rc));
        }
        Ok(())
    }

    /// Block until all work in this context completes.
    pub fn synchronize(&self) -> Result<()> {
        self.set_current()?;
        let api = driver::api()?;
        // SAFETY: context is current.
        let rc = unsafe { (api.cuCtxSynchronize)() };
        if rc != 0 {
            return Err(NnisError::driver("cuCtxSynchronize", rc));
        }
        Ok(())
    }

    /// `(free, total)` device memory in bytes. Requires a current context,
    /// which this call establishes itself.
    pub fn mem_info(&self) -> Result<(u64, u64)> {
        self.set_current()?;
        let api = driver::api()?;
        let (mut free, mut total): (usize, usize) = (0, 0);
        // SAFETY: out-pointers valid; context is current.
        let rc = unsafe { (api.cuMemGetInfo)(&mut free, &mut total) };
        if rc != 0 {
            return Err(NnisError::driver("cuMemGetInfo", rc));
        }
        Ok((free as u64, total as u64))
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Ok(api) = driver::api() {
            // SAFETY: we own one retain-count taken in `new`.
            unsafe {
                (api.cuDevicePrimaryCtxRelease)(self.ordinal);
            }
        }
    }
}

/// Test/CI helper: returns a context on the first visible device, or `None`
/// when no GPU is present (tests then report "skipped"). When
/// `NNIS_REQUIRE_GPU=1` is set, absence of a GPU becomes a hard failure so CI
/// machines that promise a GPU cannot silently pass nothing.
pub fn gpu_context() -> Option<Arc<Context>> {
    match Device::first().ok().and_then(|d| Context::new(&d).ok()) {
        Some(c) => Some(c),
        None => {
            if std::env::var("NNIS_REQUIRE_GPU").as_deref() == Ok("1") {
                panic!("NNIS_REQUIRE_GPU=1 but no usable CUDA device was found");
            }
            None
        }
    }
}
