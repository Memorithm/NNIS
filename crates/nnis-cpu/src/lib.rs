//! Deterministic pure-Rust CPU backend for the NNIS portable runtime contract.
//!
//! Buffer ownership, checked data movement and immediate completion are joined
//! by explicit finite-F32 reference operations in [`numerical`]. Per-buffer
//! limits are admission ceilings, not a promise of available RAM or a
//! process-wide memory budget. General model scheduling remains separate.

#![forbid(unsafe_code)]

pub mod numerical;

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
    /// Construct the reference backend with the host's representable byte limit.
    ///
    /// Actual allocation remains fallible; this does not measure available RAM.
    pub fn new() -> Result<Self> {
        Self::with_max_buffer_bytes(isize::MAX as u64)
    }

    /// Set an explicit per-buffer admission ceiling, without reserving memory.
    ///
    /// Total live allocations and readback copies are not governed by this
    /// per-buffer ceiling. A caller requiring a total budget must account them.
    pub fn with_max_buffer_bytes(max_buffer_bytes: u64) -> Result<Self> {
        if max_buffer_bytes == 0 || max_buffer_bytes > isize::MAX as u64 {
            return Err(PortableError::InvalidDescriptor(
                "CPU buffer limit must be positive and fit isize::MAX bytes",
            ));
        }
        Ok(Self {
            backend_id: BackendId::new(BackendFamily::Cpu, "nnis-cpu-reference")?,
            capabilities: CapabilitySet {
                max_buffer_bytes,
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
/// Storage is process-host memory even for Shared. Cloning makes a separate
/// host allocation; it does not share storage or reserve a global memory quota.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuBuffer {
    descriptor: BufferDesc,
    bytes: Vec<u8>,
}

impl CpuBuffer {
    /// Return the initialized payload length, not physical RAM residency.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Return true when the owned byte storage is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Return the retained Vec capacity in bytes, excluding allocator overhead.
    ///
    /// Capacity can exceed payload length; neither is process RSS or physical
    /// page residency. This accessor does not allocate.
    #[must_use]
    pub fn capacity_bytes(&self) -> usize {
        self.bytes.capacity()
    }
}

impl PortableBuffer for CpuBuffer {
    fn descriptor(&self) -> BufferDesc {
        self.descriptor
    }
}

/// Synchronous CPU submission queue; successful operations are already complete.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuQueue;

impl CpuQueue {
    /// Copy between distinct buffers without allocating a temporary payload.
    ///
    /// Both usages and both ranges are checked before any destination mutation.
    /// An empty copy is allowed at the end of each buffer, but not beyond it.
    /// Rust's borrows prohibit passing the same buffer as source and destination.
    pub fn copy_buffer(
        &self,
        source: &CpuBuffer,
        source_offset_bytes: u64,
        destination: &mut CpuBuffer,
        destination_offset_bytes: u64,
        size_bytes: u64,
    ) -> Result<CpuFence> {
        require_usage(source, BufferUsages::COPY_SRC, "CPU copy source")?;
        require_usage(destination, BufferUsages::COPY_DST, "CPU copy destination")?;
        let source_range = checked_range(source.len(), source_offset_bytes, size_bytes)?;
        let destination_range =
            checked_range(destination.len(), destination_offset_bytes, size_bytes)?;
        destination.bytes[destination_range].copy_from_slice(&source.bytes[source_range]);
        Ok(CpuFence)
    }
}

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
        let mut result = reserve_bytes(range.len())?;
        result.extend_from_slice(&buffer.bytes[range]);
        Ok(result)
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
        // Public descriptors may bypass their constructor or be mutated later.
        let descriptor =
            BufferDesc::new(descriptor.size_bytes, descriptor.usages, descriptor.memory)?;
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
        let mut bytes = reserve_bytes(size)?;
        bytes.resize(size, 0);
        Ok(CpuBuffer { descriptor, bytes })
    }

    fn create_queue(&self) -> Result<Self::Queue> {
        Ok(CpuQueue)
    }
}

// Reservation errors are returned before initialization/copy. Host overcommit,
// OS process termination and infallible Clone are outside this Result contract.
fn reserve_bytes(size: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|error| PortableError::Backend(format!("CPU byte reservation failed: {error}")))?;
    Ok(bytes)
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
    let error = || PortableError::OutOfBounds {
        offset_bytes,
        size_bytes,
        buffer_bytes: total as u64,
    };
    let start = usize::try_from(offset_bytes).map_err(|_| error())?;
    let size = usize::try_from(size_bytes).map_err(|_| error())?;
    let end = start.checked_add(size).ok_or_else(error)?;
    if end > total {
        return Err(error());
    }
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(usages: BufferUsages) -> BufferDesc {
        BufferDesc::new(16, usages, MemoryClass::Host).unwrap()
    }

    fn readable_writable() -> BufferDesc {
        descriptor(BufferUsages::COPY_SRC | BufferUsages::COPY_DST)
    }

    #[test]
    fn cpu_backend_identity_is_portable_and_non_vendor_specific() {
        let device = CpuDevice::new().unwrap();
        assert_eq!(device.backend_id().family(), BackendFamily::Cpu);
        assert_eq!(device.backend_id().name(), "nnis-cpu-reference");
        assert_eq!(device.capabilities().max_buffer_bytes, isize::MAX as u64);
        assert!(!device.capabilities().supports_f16);
        assert!(!device.capabilities().supports_timestamps);
    }

    #[test]
    fn cpu_round_trip_uses_portable_queue_contract() {
        let device = CpuDevice::new().unwrap();
        let usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let mut buffer = device.create_buffer(descriptor(usages)).unwrap();
        let queue = device.create_queue().unwrap();
        assert_eq!(queue.read_buffer(&buffer, 0, 16).unwrap(), vec![0; 16]);
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
        assert!(buffer.capacity_bytes() >= buffer.len());
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
        assert_eq!(
            queue.read_buffer(&write_forbidden, 0, 16).unwrap(),
            vec![0; 16]
        );
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
        let mut buffer = device.create_buffer(readable_writable()).unwrap();
        let queue = device.create_queue().unwrap();
        assert!(matches!(
            queue.write_buffer(&mut buffer, 15, &[1, 2]),
            Err(PortableError::OutOfBounds { .. })
        ));
        assert!(matches!(
            queue.read_buffer(&buffer, 17, 1),
            Err(PortableError::OutOfBounds { .. })
        ));
        assert_eq!(queue.read_buffer(&buffer, 0, 16).unwrap(), vec![0; 16]);
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
        let descriptor = BufferDesc::new(8, BufferUsages::STORAGE, MemoryClass::Shared).unwrap();
        let buffer = device.create_buffer(descriptor).unwrap();
        assert_eq!(buffer.descriptor().memory, MemoryClass::Shared);
        assert_eq!(buffer.len(), 8);
    }

    #[test]
    fn invalid_literal_descriptors_are_rejected_before_allocation() {
        let device = CpuDevice::new().unwrap();
        for descriptor in [
            BufferDesc {
                size_bytes: 0,
                ..readable_writable()
            },
            BufferDesc {
                usages: BufferUsages::EMPTY,
                ..readable_writable()
            },
        ] {
            assert!(matches!(
                device.create_buffer(descriptor),
                Err(PortableError::InvalidDescriptor(_))
            ));
        }
    }

    #[test]
    fn explicit_limits_are_checked_without_large_allocations() {
        assert!(CpuDevice::with_max_buffer_bytes(0).is_err());
        assert!(CpuDevice::with_max_buffer_bytes(u64::MAX).is_err());
        let device = CpuDevice::with_max_buffer_bytes(15).unwrap();
        assert!(matches!(
            device.create_buffer(readable_writable()),
            Err(PortableError::Unsupported(_))
        ));
        let descriptor = BufferDesc {
            size_bytes: 15,
            ..readable_writable()
        };
        assert_eq!(device.create_buffer(descriptor).unwrap().len(), 15);
        let impossible = BufferDesc {
            size_bytes: isize::MAX as u64 + 1,
            ..readable_writable()
        };
        assert!(matches!(
            CpuDevice::new().unwrap().create_buffer(impossible),
            Err(PortableError::Unsupported(_))
        ));
    }

    #[test]
    fn reservation_capacity_overflow_is_an_error_not_a_panic() {
        assert!(matches!(
            reserve_bytes(usize::MAX),
            Err(PortableError::Backend(_))
        ));
    }

    #[test]
    fn checked_buffer_copy_preserves_untouched_bytes_and_capacity() {
        let device = CpuDevice::new().unwrap();
        let queue = device.create_queue().unwrap();
        let mut source = device.create_buffer(readable_writable()).unwrap();
        let mut destination = device.create_buffer(readable_writable()).unwrap();
        queue
            .write_buffer(&mut source, 0, &[1, 2, 3, 4, 5, 6])
            .unwrap();
        queue.write_buffer(&mut destination, 0, &[9; 16]).unwrap();
        let capacity = destination.capacity_bytes();
        let fence = queue
            .copy_buffer(&source, 1, &mut destination, 4, 3)
            .unwrap();
        assert_eq!(fence.status().unwrap(), FenceStatus::Complete);
        let mut expected = vec![9; 16];
        expected[4..7].copy_from_slice(&[2, 3, 4]);
        assert_eq!(queue.read_buffer(&destination, 0, 16).unwrap(), expected);
        assert_eq!(destination.capacity_bytes(), capacity);
        assert_eq!(
            queue.read_buffer(&source, 0, 6).unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn copy_failure_never_mutates_destination() {
        let device = CpuDevice::new().unwrap();
        let queue = device.create_queue().unwrap();
        let source = device.create_buffer(readable_writable()).unwrap();
        let mut destination = device.create_buffer(readable_writable()).unwrap();
        queue.write_buffer(&mut destination, 0, &[7; 16]).unwrap();
        for (source_offset, destination_offset, size) in [
            (15, 0, 2),
            (0, 15, 2),
            (u64::MAX, 0, 2),
            (0, u64::MAX, 2),
            (0, 0, u64::MAX),
            (17, 0, 0),
            (0, 17, 0),
        ] {
            assert!(matches!(
                queue.copy_buffer(
                    &source,
                    source_offset,
                    &mut destination,
                    destination_offset,
                    size
                ),
                Err(PortableError::OutOfBounds { .. })
            ));
            assert_eq!(queue.read_buffer(&destination, 0, 16).unwrap(), vec![7; 16]);
        }
        queue
            .copy_buffer(&source, 16, &mut destination, 16, 0)
            .unwrap();
        assert_eq!(queue.read_buffer(&destination, 0, 16).unwrap(), vec![7; 16]);
    }

    #[test]
    fn buffer_copy_requires_both_usage_permissions() {
        let device = CpuDevice::new().unwrap();
        let queue = device.create_queue().unwrap();
        let source_forbidden = device
            .create_buffer(descriptor(BufferUsages::COPY_DST))
            .unwrap();
        let source = device.create_buffer(readable_writable()).unwrap();
        let mut destination = device.create_buffer(readable_writable()).unwrap();
        let mut destination_forbidden = device
            .create_buffer(descriptor(BufferUsages::COPY_SRC))
            .unwrap();
        assert!(matches!(
            queue.copy_buffer(&source_forbidden, 0, &mut destination, 0, 1),
            Err(PortableError::Unsupported(_))
        ));
        assert!(matches!(
            queue.copy_buffer(&source, 0, &mut destination_forbidden, 0, 1),
            Err(PortableError::Unsupported(_))
        ));
        assert_eq!(queue.read_buffer(&destination, 0, 16).unwrap(), vec![0; 16]);
        assert_eq!(
            queue.read_buffer(&destination_forbidden, 0, 16).unwrap(),
            vec![0; 16]
        );
    }
}
