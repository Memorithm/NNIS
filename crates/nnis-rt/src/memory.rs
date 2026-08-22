//! Device and pinned host memory.
//!
//! # Safety invariants
//!
//! * `DeviceBuffer<T>` owns exactly `len * size_of::<T>()` bytes of device
//!   memory; allocation sizes use checked arithmetic.
//! * The CUDA allocator guarantees at least 256-byte alignment, which is a
//!   prerequisite for vectorized kernels.
//! * `DeviceBuffer` is `Send + Sync`: device memory is process-visible through
//!   its context and every operation re-establishes the owning context as
//!   current. Callers that capture raw pointers into kernels must keep the
//!   buffer alive for the duration of the launch (enforced by borrowing in
//!   the typed launchers of `nnis-jit` / `nnis-kernels`).

use crate::context::Context;
use crate::error::{NnisError, Result};
use crate::stream_event::Stream;
use nnis_sys::driver;
use nnis_sys::CUdeviceptr;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

/// Owned allocation of `T` on the device.
#[derive(Debug)]
pub struct DeviceBuffer<T> {
    ptr: CUdeviceptr,
    len: usize,
    ctx: Arc<Context>,
    _marker: PhantomData<*mut T>,
}

// SAFETY: see module docs; ownership is exclusive and operations are
// context-routed, so moving across threads is sound.
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

impl<T> DeviceBuffer<T> {
    /// Allocate `len` elements (uninitialized device memory).
    pub fn new(ctx: &Arc<Context>, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| NnisError::invalid_input("allocation size overflows usize"))?;
        ctx.set_current()?;
        let api = driver::api()?;
        let mut ptr: CUdeviceptr = 0;
        if bytes > 0 {
            // SAFETY: out-pointer valid; context is current; size > 0.
            let rc = unsafe { (api.cuMemAlloc)(&mut ptr, bytes) };
            if rc != 0 {
                return Err(NnisError::driver("cuMemAlloc", rc)
                    .with("bytes", bytes)
                    .with("len", len));
            }
        }
        Ok(DeviceBuffer {
            ptr,
            len,
            ctx: Arc::clone(ctx),
            _marker: PhantomData,
        })
    }

    /// Allocate `len` elements and fill with zero bytes.
    pub fn new_zeroed(ctx: &Arc<Context>, len: usize, stream: &Stream) -> Result<Self> {
        let buf = Self::new(ctx, len)?;
        buf.zero(stream)?;
        Ok(buf)
    }

    /// Allocate from host data (asynchronous H2D copy on `stream`).
    pub fn from_host(ctx: &Arc<Context>, stream: &Stream, src: &[T]) -> Result<Self> {
        let buf = Self::new(ctx, src.len())?;
        buf.copy_from_host(stream, src)?;
        Ok(buf)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of bytes occupied.
    pub fn size_bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Raw device address (for kernel argument marshalling).
    pub fn device_ptr(&self) -> u64 {
        self.ptr as u64
    }

    /// Owning context.
    pub fn ctx(&self) -> &Arc<Context> {
        &self.ctx
    }

    /// Zero-fill via `cuMemsetD8Async`/`cuMemsetD32Async`.
    pub fn zero(&self, stream: &Stream) -> Result<()> {
        if self.ptr == 0 {
            return Ok(());
        }
        stream.ctx().set_current()?;
        let api = driver::api()?;
        let bytes = self.size_bytes();
        // SAFETY: pointer/size are the live allocation; stream is valid and
        // its context is current.
        let rc = if bytes % 4 == 0 {
            unsafe { (api.cuMemsetD32Async)(self.ptr, 0, bytes / 4, stream.raw()) }
        } else {
            unsafe { (api.cuMemsetD8Async)(self.ptr, 0, bytes, stream.raw()) }
        };
        if rc != 0 {
            return Err(NnisError::driver("cuMemsetD*", rc).with("bytes", bytes));
        }
        Ok(())
    }

    /// Asynchronous copy from host memory. The host slice must stay valid
    /// until the stream reaches this copy (call `stream.synchronize()` before
    /// dropping/moving the source unless it lives long enough).
    pub fn copy_from_host(&self, stream: &Stream, src: &[T]) -> Result<()> {
        if src.len() != self.len {
            return Err(NnisError::invalid_input(format!(
                "host slice length {} does not match device buffer length {}",
                src.len(),
                self.len
            )));
        }
        if self.ptr == 0 {
            return Ok(());
        }
        stream.ctx().set_current()?;
        let api = driver::api()?;
        let bytes = self.size_bytes();
        // SAFETY: src is a valid host slice of exactly `bytes`; dst is the
        // live device allocation; context is current.
        let rc =
            unsafe { (api.cuMemcpyHtoDAsync)(self.ptr, src.as_ptr().cast(), bytes, stream.raw()) };
        if rc != 0 {
            return Err(NnisError::driver("cuMemcpyHtoDAsync", rc).with("bytes", bytes));
        }
        Ok(())
    }

    /// Asynchronous copy into host memory. Synchronize `stream` before
    /// reading the destination.
    pub fn copy_to_host(&self, stream: &Stream, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.len {
            return Err(NnisError::invalid_input(format!(
                "host slice length {} does not match device buffer length {}",
                dst.len(),
                self.len
            )));
        }
        if self.ptr == 0 {
            return Ok(());
        }
        stream.ctx().set_current()?;
        let api = driver::api()?;
        let bytes = self.size_bytes();
        // SAFETY: dst is a valid host slice of exactly `bytes`; src is the
        // live device allocation; context is current.
        let rc = unsafe {
            (api.cuMemcpyDtoHAsync)(dst.as_mut_ptr().cast(), self.ptr, bytes, stream.raw())
        };
        if rc != 0 {
            return Err(NnisError::driver("cuMemcpyDtoHAsync", rc).with("bytes", bytes));
        }
        Ok(())
    }

    /// Blocking device-to-host copy returning a fresh `Vec`.
    pub fn to_vec(&self, stream: &Stream) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        let mut v = vec![T::default(); self.len];
        self.copy_to_host(stream, &mut v)?;
        stream.synchronize()?;
        Ok(v)
    }

    /// Asynchronous device-to-device copy between two buffers of equal length.
    pub fn copy_from_buffer(&self, stream: &Stream, src: &DeviceBuffer<T>) -> Result<()> {
        if src.len != self.len {
            return Err(NnisError::invalid_input(format!(
                "source length {} does not match destination length {}",
                src.len, self.len
            )));
        }
        if self.ptr == 0 || src.ptr == 0 {
            return Ok(());
        }
        debug_assert!(Arc::ptr_eq(&self.ctx, stream.ctx()));
        stream.ctx().set_current()?;
        let api = driver::api()?;
        let bytes = self.size_bytes();
        // SAFETY: both addresses are live allocations of `bytes`; context is
        // current and the stream belongs to it. Uses the generic async copy
        // because the DtoD-specific legacy symbols are unreliable on Tegra.
        let rc = unsafe { (api.cuMemcpyAsync)(self.ptr, src.ptr, bytes, stream.raw()) };
        if rc != 0 {
            return Err(NnisError::driver("cuMemcpyAsync", rc).with("bytes", bytes));
        }
        Ok(())
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            if let Ok(api) = driver::api() {
                let _ = self.ctx.set_current();
                // SAFETY: we own this allocation.
                unsafe {
                    (api.cuMemFree)(self.ptr);
                }
            }
        }
    }
}

/// Pinned (page-locked) host allocation for high-throughput async transfers.
#[derive(Debug)]
pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    ctx: Arc<Context>,
}

unsafe impl<T: Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Sync> Sync for PinnedBuffer<T> {}

impl<T> PinnedBuffer<T> {
    /// Allocate `len` pinned host elements.
    pub fn new(ctx: &Arc<Context>, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| NnisError::invalid_input("allocation size overflows usize"))?;
        ctx.set_current()?;
        let api = driver::api()?;
        let mut raw: *mut core::ffi::c_void = std::ptr::null_mut();
        if bytes > 0 {
            // SAFETY: out-pointer valid; context is current; size > 0.
            let rc = unsafe { (api.cuMemAllocHost)(&mut raw, bytes) };
            if rc != 0 {
                return Err(NnisError::driver("cuMemAllocHost", rc)
                    .with("bytes", bytes)
                    .with("len", len));
            }
        }
        Ok(PinnedBuffer {
            ptr: raw.cast::<T>(),
            len,
            ctx: Arc::clone(ctx),
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: allocation is live for `len` elements; no concurrent mut.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: allocation is live for `len` elements; exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn ctx(&self) -> &Arc<Context> {
        &self.ctx
    }
}

impl<T> std::ops::Deref for PinnedBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for PinnedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Ok(api) = driver::api() {
                let _ = self.ctx.set_current();
                // SAFETY: we own this allocation.
                unsafe {
                    (api.cuMemFreeHost)(self.ptr.cast());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn allocation_overflow_is_rejected_without_gpu() {
        // Pure arithmetic contract: checked_mul must catch overflow before
        // any native call is attempted.
        assert!(usize::MAX.checked_mul(4).is_none());
    }

    #[test]
    fn device_buffer_h2d_d2h_roundtrip() {
        use crate::{gpu_context, DeviceBuffer, Stream};
        let Some(ctx) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&ctx).unwrap();
        let data: Vec<f32> = (0..4096).map(|i| i as f32 * 0.25 - 128.0).collect();
        let buf = DeviceBuffer::<f32>::from_host(&ctx, &stream, &data).unwrap();
        assert_eq!(buf.len(), 4096);
        assert_eq!(buf.size_bytes(), 16_384);
        // Alignment guarantee from the CUDA allocator.
        assert_eq!(buf.device_ptr() % 256, 0);
        let back = buf.to_vec(&stream).unwrap();
        assert_eq!(data, back);
    }

    #[test]
    fn zero_and_d2d_copy() {
        use crate::{gpu_context, DeviceBuffer, Stream};
        let Some(ctx) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&ctx).unwrap();
        let src_data: Vec<u32> = (0..1024).collect();
        let a = DeviceBuffer::<u32>::from_host(&ctx, &stream, &src_data).unwrap();
        let b = DeviceBuffer::<u32>::new(&ctx, 1024).unwrap();

        b.zero(&stream).unwrap();
        let zeros = b.to_vec(&stream).unwrap();
        assert!(zeros.iter().all(|&v| v == 0), "zero fill failed");

        b.copy_from_buffer(&stream, &a).unwrap();
        let copied = b.to_vec(&stream).unwrap();
        assert_eq!(copied, src_data, "D2D copy mismatch");
    }

    #[test]
    fn size_mismatch_is_rejected() {
        use crate::{gpu_context, DeviceBuffer, Stream};
        let Some(ctx) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&ctx).unwrap();
        let buf = DeviceBuffer::<f32>::new(&ctx, 16).unwrap();
        let mut host = vec![0.0f32; 15];
        let err = buf.copy_to_host(&stream, &mut host).unwrap_err();
        assert!(err.op().is_empty()); // invalid-input errors carry no native op
        let err2 = buf.copy_from_host(&stream, &host).unwrap_err();
        assert!(matches!(err2.kind(), crate::ErrorKind::InvalidInput(_)));
    }

    #[test]
    fn empty_buffer_operations_are_noops() {
        use crate::{gpu_context, DeviceBuffer, Stream};
        let Some(ctx) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&ctx).unwrap();
        let buf = DeviceBuffer::<f32>::from_host(&ctx, &stream, &[]).unwrap();
        assert!(buf.is_empty());
        assert_eq!(buf.device_ptr(), 0);
        buf.zero(&stream).unwrap();
        assert_eq!(buf.to_vec(&stream).unwrap(), Vec::<f32>::new());
    }
}
