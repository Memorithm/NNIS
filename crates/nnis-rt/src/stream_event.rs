//! Asynchronous execution primitives: streams and events.

use crate::context::Context;
use crate::error::{NnisError, Result};
use nnis_sys::driver;
use nnis_sys::{constants as cu, CUevent, CUstream};
use std::sync::Arc;

/// An asynchronous work queue bound to a context.
///
/// Streams are created `CU_STREAM_NON_BLOCKING` by default so they do not
/// implicitly synchronize with the legacy default stream.
#[derive(Debug, Clone)]
pub struct Stream {
    inner: Arc<StreamInner>,
}

#[derive(Debug)]
struct StreamInner {
    raw: CUstream,
    ctx: Arc<Context>,
}

unsafe impl Send for StreamInner {}
unsafe impl Sync for StreamInner {}

impl Stream {
    pub fn new(ctx: &Arc<Context>) -> Result<Self> {
        Self::with_flags(ctx, cu::CU_STREAM_NON_BLOCKING)
    }

    /// Create a stream with explicit flags (`CU_STREAM_*`).
    pub fn with_flags(ctx: &Arc<Context>, flags: u32) -> Result<Self> {
        ctx.set_current()?;
        let api = driver::api()?;
        let mut raw: CUstream = std::ptr::null_mut();
        // SAFETY: out-pointer valid; context is current.
        let rc = unsafe { (api.cuStreamCreate)(&mut raw, flags) };
        if rc != 0 {
            return Err(NnisError::driver("cuStreamCreate", rc));
        }
        Ok(Stream {
            inner: Arc::new(StreamInner {
                raw,
                ctx: Arc::clone(ctx),
            }),
        })
    }

    /// Block until all previously submitted work on this stream completes.
    pub fn synchronize(&self) -> Result<()> {
        self.inner.ctx.set_current()?;
        let api = driver::api()?;
        // SAFETY: valid stream handle; context is current.
        let rc = unsafe { (api.cuStreamSynchronize)(self.inner.raw) };
        if rc != 0 {
            return Err(NnisError::driver("cuStreamSynchronize", rc));
        }
        Ok(())
    }

    /// Make this stream wait until `event` has completed (non-blocking on the
    /// host). The event must have been recorded.
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        self.inner.ctx.set_current()?;
        let api = driver::api()?;
        // SAFETY: valid handles; context is current.
        let rc = unsafe { (api.cuStreamWaitEvent)(self.raw(), event.raw(), 0) };
        if rc != 0 {
            return Err(NnisError::driver("cuStreamWaitEvent", rc));
        }
        Ok(())
    }

    /// `true` if all submitted work has completed (no host blocking).
    pub fn query(&self) -> Result<bool> {
        self.inner.ctx.set_current()?;
        let api = driver::api()?;
        // SAFETY: valid stream handle.
        let rc = unsafe { (api.cuStreamQuery)(self.inner.raw) };
        match rc {
            0 => Ok(true),
            cu::error_codes::CUDA_ERROR_NOT_READY => Ok(false),
            _ => Err(NnisError::driver("cuStreamQuery", rc)),
        }
    }

    pub fn raw(&self) -> CUstream {
        self.inner.raw
    }

    pub fn ctx(&self) -> &Arc<Context> {
        &self.inner.ctx
    }
}

#[derive(Debug)]
struct StreamDropGuard;

impl Drop for StreamInner {
    fn drop(&mut self) {
        if let Ok(api) = driver::api() {
            let _ = self.ctx.set_current();
            // SAFETY: we own one creation reference.
            unsafe {
                (api.cuStreamDestroy)(self.raw);
            }
        }
    }
}

/// A GPU timeline marker usable for ordering (`wait_event`) and timing
/// (`elapsed_ms`).
///
/// Created *without* `CU_EVENT_DISABLE_TIMING` because NNIS's benchmarking
/// infrastructure depends on GPU-side timestamps.
#[derive(Debug, Clone)]
pub struct Event {
    inner: Arc<EventInner>,
}

#[derive(Debug)]
struct EventInner {
    raw: CUevent,
    ctx: Arc<Context>,
}

unsafe impl Send for EventInner {}
unsafe impl Sync for EventInner {}

impl Event {
    pub fn new(ctx: &Arc<Context>) -> Result<Self> {
        ctx.set_current()?;
        let api = driver::api()?;
        let mut raw: CUevent = std::ptr::null_mut();
        // SAFETY: out-pointer valid; context is current.
        let rc = unsafe { (api.cuEventCreate)(&mut raw, cu::CU_EVENT_DEFAULT) };
        if rc != 0 {
            return Err(NnisError::driver("cuEventCreate", rc));
        }
        Ok(Event {
            inner: Arc::new(EventInner {
                raw,
                ctx: Arc::clone(ctx),
            }),
        })
    }

    /// Record the event on `stream` at its current tail position.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        debug_assert!(Arc::ptr_eq(&self.inner.ctx, stream.ctx()));
        self.inner.ctx.set_current()?;
        let api = driver::api()?;
        // SAFETY: valid handles; same context asserted above.
        let rc = unsafe { (api.cuEventRecord)(self.inner.raw, stream.raw()) };
        if rc != 0 {
            return Err(NnisError::driver("cuEventRecord", rc));
        }
        Ok(())
    }

    /// Host-blocks until the event has been recorded (all preceding work done).
    pub fn synchronize(&self) -> Result<()> {
        self.inner.ctx.set_current()?;
        let api = driver::api()?;
        // SAFETY: valid event handle; context is current.
        let rc = unsafe { (api.cuEventSynchronize)(self.inner.raw) };
        if rc != 0 {
            return Err(NnisError::driver("cuEventSynchronize", rc));
        }
        Ok(())
    }

    /// `true` once the event has been recorded (no host blocking).
    pub fn query(&self) -> Result<bool> {
        self.inner.ctx.set_current()?;
        let api = driver::api()?;
        // SAFETY: valid event handle.
        let rc = unsafe { (api.cuEventQuery)(self.inner.raw) };
        match rc {
            0 => Ok(true),
            cu::error_codes::CUDA_ERROR_NOT_READY => Ok(false),
            _ => Err(NnisError::driver("cuEventQuery", rc)),
        }
    }

    /// Milliseconds of GPU time between `start` completing and `self`
    /// completing. Both events must have been recorded and synchronized.
    pub fn elapsed_ms(&self, start: &Event) -> Result<f64> {
        let api = driver::api()?;
        let mut ms: f32 = 0.0;
        // SAFETY: both events are valid handles owned by this process.
        let rc = unsafe { (api.cuEventElapsedTime)(&mut ms, start.inner.raw, self.inner.raw) };
        if rc != 0 {
            return Err(NnisError::driver("cuEventElapsedTime", rc));
        }
        Ok(ms as f64)
    }

    pub(crate) fn raw(&self) -> CUevent {
        self.inner.raw
    }
}

impl Drop for EventInner {
    fn drop(&mut self) {
        if let Ok(api) = driver::api() {
            // SAFETY: we own one creation reference.
            unsafe {
                (api.cuEventDestroy)(self.raw);
            }
        }
    }
}
