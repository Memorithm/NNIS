//! Portable buffer, queue and fence contracts over WGPU.
//!
//! Semantics follow the CPU reference (`nnis-cpu`): descriptors are
//! revalidated at allocation, buffers are zero-initialized, usage bits are
//! enforced at the portable API (writes need `COPY_DST`, reads and copy
//! sources need `COPY_SRC`), and every byte range is checked against the
//! logical buffer size before any mutation. Arbitrary byte offsets and sizes
//! are accepted: WGPU's 4-byte copy alignment is handled internally by
//! padding allocations and read-modify-writing unaligned edges. The padding
//! is never observable through this API.
//!
//! `MemoryClass::Host` is rejected because WGPU buffers are not process host
//! memory. `Shared` and `DeviceLocal` are accepted as declared intent only:
//! WGPU does not report physical placement, so no residency is claimed.
//! Buffers are bound to the device that created them; using one with another
//! device's queue fails closed.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

use nnis_core::{
    BufferDesc, BufferUsages, FenceStatus, MemoryClass, PortableBuffer, PortableDevice,
    PortableError, PortableFence, PortableQueue, Result,
};

use crate::{block_on, WgpuDevice};

const ALIGN: u64 = wgpu::COPY_BUFFER_ALIGNMENT;

static NEXT_DEVICE_TOKEN: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_device_token() -> u64 {
    NEXT_DEVICE_TOKEN.fetch_add(1, Ordering::Relaxed)
}

/// WGPU buffer owned by one [`WgpuDevice`].
#[derive(Debug)]
pub struct WgpuBuffer {
    descriptor: BufferDesc,
    buffer: wgpu::Buffer,
    device_token: u64,
}

impl WgpuBuffer {
    /// Logical size in bytes (the allocation may be padded to 4 bytes).
    #[must_use]
    pub fn len(&self) -> u64 {
        self.descriptor.size_bytes
    }

    /// Always false: zero-sized buffers are rejected at allocation.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.descriptor.size_bytes == 0
    }

    pub(crate) fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl PortableBuffer for WgpuBuffer {
    fn descriptor(&self) -> BufferDesc {
        self.descriptor
    }
}

/// Completion of one WGPU queue submission.
#[derive(Debug, Clone)]
pub struct WgpuFence {
    device: Arc<wgpu::Device>,
    submission: wgpu::SubmissionIndex,
    done: Arc<AtomicBool>,
}

impl PortableFence for WgpuFence {
    /// Non-blocking. `Complete` is never reported early: the callback is
    /// registered after this submission, so it fires only once the
    /// submission finished. It may report `Pending` briefly after completion.
    fn status(&self) -> Result<FenceStatus> {
        if !self.done.load(Ordering::Acquire) {
            self.device.poll(wgpu::Maintain::Poll);
        }
        Ok(if self.done.load(Ordering::Acquire) {
            FenceStatus::Complete
        } else {
            FenceStatus::Pending
        })
    }

    fn wait(&self) -> Result<()> {
        if !self.done.load(Ordering::Acquire) {
            // A blocking poll for this submission index returns only after
            // that submission completed. The work-done callback may instead
            // run on another thread's concurrent poll, so completion is
            // recorded here rather than waiting for the callback.
            self.device
                .poll(wgpu::Maintain::wait_for(self.submission.clone()));
            self.done.store(true, Ordering::Release);
        }
        Ok(())
    }
}

/// Submission queue of one [`WgpuDevice`].
#[derive(Debug, Clone)]
pub struct WgpuQueue {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    device_token: u64,
}

impl WgpuQueue {
    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn check_owner(&self, buffer: &WgpuBuffer) -> Result<()> {
        if buffer.device_token == self.device_token {
            Ok(())
        } else {
            Err(PortableError::Unsupported(
                "WGPU buffer belongs to a different device".to_string(),
            ))
        }
    }

    pub(crate) fn fence_for(&self, submission: wgpu::SubmissionIndex) -> WgpuFence {
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        self.queue
            .on_submitted_work_done(move || flag.store(true, Ordering::Release));
        WgpuFence {
            device: Arc::clone(&self.device),
            submission,
            done,
        }
    }

    /// Copy between distinct buffers of this device.
    ///
    /// Both usages and both ranges are checked before any destination
    /// mutation. An empty copy is allowed at the end of each buffer, but not
    /// beyond it. Aligned copies run on the GPU; unaligned copies are staged
    /// through the host with the same result.
    pub fn copy_buffer(
        &self,
        source: &WgpuBuffer,
        source_offset_bytes: u64,
        destination: &mut WgpuBuffer,
        destination_offset_bytes: u64,
        size_bytes: u64,
    ) -> Result<WgpuFence> {
        self.check_owner(source)?;
        self.check_owner(destination)?;
        require_usage(source, BufferUsages::COPY_SRC, "WGPU copy source")?;
        require_usage(destination, BufferUsages::COPY_DST, "WGPU copy destination")?;
        checked_range(source.len(), source_offset_bytes, size_bytes)?;
        checked_range(destination.len(), destination_offset_bytes, size_bytes)?;
        if size_bytes == 0 {
            return Ok(self.fence_for(self.queue.submit(None)));
        }
        if source_offset_bytes % ALIGN == 0
            && destination_offset_bytes % ALIGN == 0
            && size_bytes % ALIGN == 0
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("nnis.wgpu.copy"),
                });
            encoder.copy_buffer_to_buffer(
                &source.buffer,
                source_offset_bytes,
                &destination.buffer,
                destination_offset_bytes,
                size_bytes,
            );
            return self.submit_checked(encoder);
        }
        let bytes = self.read_raw(source, source_offset_bytes, size_bytes)?;
        self.write_raw(destination, destination_offset_bytes, &bytes)
    }

    /// Device copy between buffers of this device without portable usage
    /// checks (internal plumbing; caller checks ownership and sizes).
    pub(crate) fn copy_unchecked(
        &self,
        source: &WgpuBuffer,
        destination: &WgpuBuffer,
        size_bytes: u64,
    ) -> Result<WgpuFence> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nnis.wgpu.copy_unchecked"),
            });
        encoder.copy_buffer_to_buffer(&source.buffer, 0, &destination.buffer, 0, size_bytes);
        self.submit_checked(encoder)
    }

    pub(crate) fn submit_checked(&self, encoder: wgpu::CommandEncoder) -> Result<WgpuFence> {
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let submission = self.queue.submit(Some(encoder.finish()));
        if let Some(error) = block_on(self.device.pop_error_scope()) {
            return Err(PortableError::Backend(format!(
                "WGPU submission failed: {error}"
            )));
        }
        Ok(self.fence_for(submission))
    }

    /// Read without the usage check (caller has checked ownership and range).
    pub(crate) fn read_raw(
        &self,
        buffer: &WgpuBuffer,
        offset_bytes: u64,
        size_bytes: u64,
    ) -> Result<Vec<u8>> {
        self.read_wgpu(&buffer.buffer, offset_bytes, size_bytes)
    }

    /// Read a range of any `COPY_SRC` WGPU buffer after all prior submissions.
    pub(crate) fn read_wgpu(
        &self,
        buffer: &wgpu::Buffer,
        offset_bytes: u64,
        size_bytes: u64,
    ) -> Result<Vec<u8>> {
        let mut result = reserve_bytes(size_bytes)?;
        if size_bytes == 0 {
            return Ok(result);
        }
        let start = offset_bytes - offset_bytes % ALIGN;
        let end = align_up(offset_bytes + size_bytes);
        let span = end - start;
        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nnis.wgpu.readback"),
            size: span,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nnis.wgpu.readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, start, &staging, 0, span);
        self.queue.submit(Some(encoder.finish()));
        pop_scopes(&self.device, "WGPU readback")?;
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |outcome| {
            let _ = sender.send(outcome);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| PortableError::Backend("WGPU readback callback dropped".to_string()))?
            .map_err(|error| PortableError::Backend(format!("WGPU readback failed: {error}")))?;
        {
            let mapped = slice.get_mapped_range();
            let first = (offset_bytes - start) as usize;
            result.extend_from_slice(&mapped[first..first + size_bytes as usize]);
        }
        staging.unmap();
        Ok(result)
    }

    /// Write without the usage check (caller has checked ownership and range).
    pub(crate) fn write_raw(
        &self,
        buffer: &WgpuBuffer,
        offset_bytes: u64,
        bytes: &[u8],
    ) -> Result<WgpuFence> {
        let size_bytes = bytes.len() as u64;
        if size_bytes == 0 {
            return Ok(self.fence_for(self.queue.submit(None)));
        }
        let start = offset_bytes - offset_bytes % ALIGN;
        let end = align_up(offset_bytes + size_bytes);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        if start == offset_bytes && end == offset_bytes + size_bytes {
            self.queue.write_buffer(&buffer.buffer, offset_bytes, bytes);
        } else {
            // Read-modify-write the aligned span; bytes outside the requested
            // range (including allocation padding) are written back unchanged.
            let mut span = self.read_raw(buffer, start, end - start)?;
            let first = (offset_bytes - start) as usize;
            span[first..first + bytes.len()].copy_from_slice(bytes);
            self.queue.write_buffer(&buffer.buffer, start, &span);
        }
        let submission = self.queue.submit(None);
        if let Some(error) = block_on(self.device.pop_error_scope()) {
            return Err(PortableError::Backend(format!(
                "WGPU write failed: {error}"
            )));
        }
        Ok(self.fence_for(submission))
    }
}

impl PortableQueue for WgpuQueue {
    type Buffer = WgpuBuffer;
    type Fence = WgpuFence;

    /// Stage `bytes` before returning (WGPU copies the slice), then submit.
    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset_bytes: u64,
        bytes: &[u8],
    ) -> Result<Self::Fence> {
        self.check_owner(buffer)?;
        require_usage(buffer, BufferUsages::COPY_DST, "WGPU buffer write")?;
        let size_bytes = u64::try_from(bytes.len())
            .map_err(|_| PortableError::Backend("write size does not fit u64".to_string()))?;
        checked_range(buffer.len(), offset_bytes, size_bytes)?;
        self.write_raw(buffer, offset_bytes, bytes)
    }

    /// Read after all previously submitted work on this queue completes.
    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset_bytes: u64,
        size_bytes: u64,
    ) -> Result<Vec<u8>> {
        self.check_owner(buffer)?;
        require_usage(buffer, BufferUsages::COPY_SRC, "WGPU buffer read")?;
        checked_range(buffer.len(), offset_bytes, size_bytes)?;
        self.read_raw(buffer, offset_bytes, size_bytes)
    }
}

impl PortableDevice for WgpuDevice {
    type Buffer = WgpuBuffer;
    type Queue = WgpuQueue;

    fn backend_id(&self) -> &nnis_core::BackendId {
        &self.backend_id
    }

    fn capabilities(&self) -> &nnis_core::CapabilitySet {
        &self.capabilities
    }

    fn create_buffer(&self, descriptor: BufferDesc) -> Result<Self::Buffer> {
        // Public descriptors may bypass their constructor or be mutated later.
        let descriptor =
            BufferDesc::new(descriptor.size_bytes, descriptor.usages, descriptor.memory)?;
        if descriptor.size_bytes > self.capabilities.max_buffer_bytes {
            return Err(PortableError::Unsupported(
                "buffer exceeds WGPU backend capability".to_string(),
            ));
        }
        if descriptor.memory == MemoryClass::Host {
            return Err(PortableError::Unsupported(
                "WGPU backend does not expose Host memory buffers".to_string(),
            ));
        }
        let size = align_up(descriptor.size_bytes);
        if size > self.limits.max_buffer_size {
            return Err(PortableError::Unsupported(
                "padded buffer exceeds WGPU max_buffer_size".to_string(),
            ));
        }
        let mut usage = wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        if descriptor.usages.contains(BufferUsages::STORAGE) {
            usage |= wgpu::BufferUsages::STORAGE;
        }
        if descriptor.usages.contains(BufferUsages::UNIFORM) {
            usage |= wgpu::BufferUsages::UNIFORM;
        }
        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nnis.wgpu.buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });
        pop_scopes(&self.device, "WGPU buffer allocation")?;
        Ok(WgpuBuffer {
            descriptor,
            buffer,
            device_token: self.token,
        })
    }

    fn create_queue(&self) -> Result<Self::Queue> {
        Ok(WgpuQueue {
            device: Arc::clone(&self.device),
            queue: Arc::clone(&self.queue),
            device_token: self.token,
        })
    }
}

/// Pop a Validation scope then an OutOfMemory scope pushed in that order.
fn pop_scopes(device: &wgpu::Device, operation: &str) -> Result<()> {
    let validation = block_on(device.pop_error_scope());
    let out_of_memory = block_on(device.pop_error_scope());
    match validation.or(out_of_memory) {
        None => Ok(()),
        Some(error) => Err(PortableError::Backend(format!(
            "{operation} failed: {error}"
        ))),
    }
}

fn align_up(bytes: u64) -> u64 {
    bytes.div_ceil(ALIGN) * ALIGN
}

fn reserve_bytes(size: u64) -> Result<Vec<u8>> {
    let size = usize::try_from(size).map_err(|_| {
        PortableError::Unsupported("read size does not fit host address space".to_string())
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(|error| {
        PortableError::Backend(format!("host byte reservation failed: {error}"))
    })?;
    Ok(bytes)
}

pub(crate) fn require_usage(
    buffer: &WgpuBuffer,
    required: BufferUsages,
    operation: &'static str,
) -> Result<()> {
    if !buffer.descriptor.usages.contains(required) {
        return Err(PortableError::Unsupported(format!(
            "{operation} requires buffer usage bits {}",
            required.bits()
        )));
    }
    Ok(())
}

pub(crate) fn checked_range(total: u64, offset_bytes: u64, size_bytes: u64) -> Result<()> {
    match offset_bytes.checked_add(size_bytes) {
        Some(end) if end <= total => Ok(()),
        _ => Err(PortableError::OutOfBounds {
            offset_bytes,
            size_bytes,
            buffer_bytes: total,
        }),
    }
}
