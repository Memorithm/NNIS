//! Deterministic pure-Rust CPU backend for the NNIS portable runtime contract.
//!
//! P2a implements buffer ownership, data movement, and immediate completion
//! semantics only. Numerical kernels and model scheduling are intentionally
//! separate later slices.

#![forbid(unsafe_code)]

use core::ops::Range;

use nnis_core::{
    BackendFamily, BackendId, BufferDesc, BufferUsages, CapabilitySet, FenceStatus, MemoryClass,
    PortableBuffer, PortableDevice, PortableError, PortableFence, PortableQueue, Result,
};

/// Portable CPU reference device.
#[derive(Debug, Clone)]
pub struct CpuDevice {
    backend_id: BackendId,
    capabilities: CapabilitySet,
}

impl CpuDevice {
    /// Construct the deterministic CPU reference backend.
    pub fn new() -> Result<Self> {
        Ok(Self {
            backend_id: BackendId::new(BackendFamily::Cpu, "nnis-cpu-reference")?,
            capabilities: CapabilitySet {
                max_buffer_bytes: u64::MAX,
                max_workgroup_invocations: 1,
                max_workgroup_size: [1, 1, 1],
                max_bindings: 1,
                supports_f16: false,
                supports_timestamps: false,
            }
            .validate()?,
        })
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new().expect("static CPU capability declaration is valid")
    }
}

/// Owned CPU byte buffer.
///
/// The portable descriptor remains authoritative for usage checks. Storage is
/// always process-host memory even when callers request the portable Shared
/// class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuBuffer {
    descriptor: BufferDesc,
    bytes: Vec<u8>,
}

impl CpuBuffer {
    /// Return the number of owned bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Return true when the owned byte storage is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

impl PortableBuffer for CpuBuffer {
    fn descriptor(&self) -> BufferDesc {
        self.descriptor
    }
}

/// CPU submission queue.
///
/// Operations complete before returning, so every returned fence is already
/// complete.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuQueue;

/// Immediate CPU completion fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFence;

impl PortableFence for CpuFence {
    fn status(&self) -> Result<FenceStatus> {
        Ok(FenceStatus::Complete)
    }

    fn wait(&self) -> Result<()> {
        Ok(())
    }
}

impl PortableQueue for CpuQueue {
    type Buffer = CpuBuffer;
    type Fence = CpuFence;

    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset_bytes: u64,
        bytes: &[u8],
    ) -> Result<Self::Fence> {
        require_usage(buffer, BufferUsages::COPY_DST, "CPU buffer write")?;
        let size_bytes = u64::try_from(bytes.len())
            .map_err(|_| PortableError::Backend("write size does not fit u64".to_string()))?;
        let range = checked_range(buffer.bytes.len(), offset_bytes, size_bytes)?;
        buffer.bytes[range].copy_from_slice(bytes);
        Ok(CpuFence)
    }

    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset_bytes: u64,
        size_bytes: u64,
    ) -> Result<Vec<u8>> {
        require_usage(buffer, BufferUsages::COPY_SRC, "CPU buffer read")?;
        let range = checked_range(buffer.bytes.len(), offset_bytes, size_bytes)?;
        Ok(buffer.bytes[range].to_vec())
    }
}

impl PortableDevice for CpuDevice {
    type Buffer = CpuBuffer;
    type Queue = CpuQueue;

    fn backend_id(&self) -> &BackendId {
        &self.backend_id
    }

    fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    fn create_buffer(&self, descriptor: BufferDesc) -> Result<Self::Buffer> {
        if descriptor.size_bytes > self.capabilities.max_buffer_bytes {
            return Err(PortableError::Unsupported(
                "buffer exceeds CPU backend capability".to_string(),
            ));
        }
        if descriptor.memory == MemoryClass::DeviceLocal {
            return Err(PortableError::Unsupported(
                "CPU reference backend does not expose DeviceLocal memory".to_string(),
            ));
        }
        let size = usize::try_from(descriptor.size_bytes).map_err(|_| {
            PortableError::Unsupported("buffer size does not fit host address space".to_string())
        })?;
        Ok(CpuBuffer {
            descriptor,
            bytes: vec![0; size],
        })
    }

    fn create_queue(&self) -> Result<Self::Queue> {
        Ok(CpuQueue)
    }
}

fn require_usage(
    buffer: &CpuBuffer,
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

fn checked_range(total: usize, offset_bytes: u64, size_bytes: u64) -> Result<Range<usize>> {
    let start = usize::try_from(offset_bytes).map_err(|_| PortableError::OutOfBounds {
        offset_bytes,
        size_bytes,
        buffer_bytes: total as u64,
    })?;
    let size = usize::try_from(size_bytes).map_err(|_| PortableError::OutOfBounds {
        offset_bytes,
        size_bytes,
        buffer_bytes: total as u64,
    })?;
    let end = start.checked_add(size).ok_or(PortableError::OutOfBounds {
        offset_bytes,
        size_bytes,
        buffer_bytes: total as u64,
    })?;
    if end > total {
        return Err(PortableError::OutOfBounds {
            offset_bytes,
            size_bytes,
            buffer_bytes: total as u64,
        });
    }
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(usages: BufferUsages) -> BufferDesc {
        BufferDesc::new(16, usages, MemoryClass::Host).unwrap()
    }

    #[test]
    fn cpu_backend_identity_is_portable_and_non_vendor_specific() {
        let device = CpuDevice::new().unwrap();
        assert_eq!(device.backend_id().family(), BackendFamily::Cpu);
        assert_eq!(device.backend_id().name(), "nnis-cpu-reference");
        assert!(!device.capabilities().supports_f16);
        assert!(!device.capabilities().supports_timestamps);
    }

    #[test]
    fn cpu_round_trip_uses_portable_queue_contract() {
        let device = CpuDevice::new().unwrap();
        let usages =
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let mut buffer = device.create_buffer(descriptor(usages)).unwrap();
        let queue = device.create_queue().unwrap();

        let fence = queue
            .write_buffer(&mut buffer, 5, &[10, 20, 30, 40])
            .unwrap();
        assert_eq!(fence.status().unwrap(), FenceStatus::Complete);
        fence.wait().unwrap();

        assert_eq!(
            queue.read_buffer(&buffer, 5, 4).unwrap(),
            vec![10, 20, 30, 40]
        );
        assert_eq!(buffer.len(), 16);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn usage_contract_fails_closed() {
        let device = CpuDevice::new().unwrap();
        let mut write_forbidden = device
            .create_buffer(descriptor(BufferUsages::COPY_SRC))
            .unwrap();
        let queue = device.create_queue().unwrap();

        assert!(matches!(
            queue.write_buffer(&mut write_forbidden, 0, &[1]),
            Err(PortableError::Unsupported(_))
        ));

        let read_forbidden = device
            .create_buffer(descriptor(BufferUsages::COPY_DST))
            .unwrap();
        assert!(matches!(
            queue.read_buffer(&read_forbidden, 0, 1),
            Err(PortableError::Unsupported(_))
        ));
    }

    #[test]
    fn out_of_bounds_copy_is_rejected() {
        let device = CpuDevice::new().unwrap();
        let usages = BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let mut buffer = device.create_buffer(descriptor(usages)).unwrap();
        let queue = device.create_queue().unwrap();

        assert!(matches!(
            queue.write_buffer(&mut buffer, 15, &[1, 2]),
            Err(PortableError::OutOfBounds { .. })
        ));
        assert!(matches!(
            queue.read_buffer(&buffer, 17, 1),
            Err(PortableError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn device_local_memory_is_not_faked_on_cpu() {
        let device = CpuDevice::new().unwrap();
        let descriptor =
            BufferDesc::new(8, BufferUsages::STORAGE, MemoryClass::DeviceLocal).unwrap();
        assert!(matches!(
            device.create_buffer(descriptor),
            Err(PortableError::Unsupported(_))
        ));
    }

    #[test]
    fn shared_memory_class_is_backed_by_host_storage() {
        let device = CpuDevice::new().unwrap();
        let descriptor =
            BufferDesc::new(8, BufferUsages::STORAGE, MemoryClass::Shared).unwrap();
        let buffer = device.create_buffer(descriptor).unwrap();
        assert_eq!(buffer.descriptor().memory, MemoryClass::Shared);
        assert_eq!(buffer.len(), 8);
    }
}
