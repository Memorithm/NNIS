//! Backend-neutral runtime contracts for the Native Neural Inference Stack.
//!
//! This crate deliberately has no dependency on CUDA, NVRTC, NVML, WGPU, or
//! any other hardware API. Backend implementations live outside this crate.

#![forbid(unsafe_code)]

pub mod graph;
pub mod replay_state;

use core::fmt;
use core::ops::{BitOr, BitOrAssign};

/// Version of the portable runtime contract.
pub const PORTABLE_RUNTIME_CONTRACT_VERSION: u32 = 1;

/// Portable backend family.
///
/// The enum names execution APIs, not hardware vendors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendFamily {
    Cpu,
    Wgpu,
}

/// Stable identity for one runtime backend instance.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendId {
    family: BackendFamily,
    name: String,
}

impl BackendId {
    /// Construct a backend identity with a non-empty implementation name.
    pub fn new(family: BackendFamily, name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(PortableError::InvalidDescriptor(
                "backend name must not be empty",
            ));
        }
        Ok(Self { family, name })
    }

    /// Return the portable backend family.
    pub const fn family(&self) -> BackendFamily {
        self.family
    }

    /// Return the implementation/device label supplied by the backend.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Backend capability limits used for fail-closed plan validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilitySet {
    pub max_buffer_bytes: u64,
    pub max_workgroup_invocations: u32,
    pub max_workgroup_size: [u32; 3],
    pub max_bindings: u32,
    pub supports_f16: bool,
    pub supports_timestamps: bool,
}

impl CapabilitySet {
    /// Validate that mandatory limits are non-zero and internally representable.
    pub fn validate(self) -> Result<Self> {
        if self.max_buffer_bytes == 0 {
            return Err(PortableError::InvalidDescriptor(
                "max_buffer_bytes must be non-zero",
            ));
        }
        if self.max_workgroup_invocations == 0 {
            return Err(PortableError::InvalidDescriptor(
                "max_workgroup_invocations must be non-zero",
            ));
        }
        if self.max_workgroup_size.contains(&0) {
            return Err(PortableError::InvalidDescriptor(
                "max_workgroup_size dimensions must be non-zero",
            ));
        }
        if self.max_bindings == 0 {
            return Err(PortableError::InvalidDescriptor(
                "max_bindings must be non-zero",
            ));
        }
        Ok(self)
    }
}

/// Logical memory class independent of a vendor allocation API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryClass {
    Host,
    Shared,
    DeviceLocal,
}

/// Combinable buffer usage flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferUsages(u32);

impl BufferUsages {
    pub const EMPTY: Self = Self(0);
    pub const STORAGE: Self = Self(1 << 0);
    pub const UNIFORM: Self = Self(1 << 1);
    pub const COPY_SRC: Self = Self(1 << 2);
    pub const COPY_DST: Self = Self(1 << 3);

    /// Return the raw stable usage bitset.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Test whether every requested usage bit is present.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Test whether no usage bit is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for BufferUsages {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for BufferUsages {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Backend-neutral buffer allocation descriptor.
///
/// Public fields allow literal construction and mutation. Backends must call
/// [`Self::validate`] immediately before allocating, not trust constructor use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    pub size_bytes: u64,
    pub usages: BufferUsages,
    pub memory: MemoryClass,
}

impl BufferDesc {
    /// Construct a non-empty buffer descriptor with at least one declared usage.
    pub fn new(size_bytes: u64, usages: BufferUsages, memory: MemoryClass) -> Result<Self> {
        Self {
            size_bytes,
            usages,
            memory,
        }
        .validate()
    }

    /// Revalidate the descriptor at a backend boundary.
    ///
    /// This does not prove device support, available memory, or allocation
    /// success. Each backend must independently enforce those constraints.
    pub fn validate(self) -> Result<Self> {
        if self.size_bytes == 0 {
            return Err(PortableError::InvalidDescriptor(
                "buffer size must be non-zero",
            ));
        }
        if self.usages.is_empty() {
            return Err(PortableError::InvalidDescriptor(
                "buffer usage set must not be empty",
            ));
        }
        Ok(self)
    }
}

/// Completion state of submitted portable work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceStatus {
    Pending,
    Complete,
}

/// Buffer object owned by one portable backend.
pub trait PortableBuffer {
    /// Return the descriptor used to create this buffer.
    fn descriptor(&self) -> BufferDesc;
}

/// Completion primitive for submitted queue operations.
pub trait PortableFence {
    /// Observe completion without changing the work item.
    fn status(&self) -> Result<FenceStatus>;

    /// Block until the associated work is complete.
    fn wait(&self) -> Result<()>;
}

/// Data-movement queue shared by CPU and future WGPU backends.
///
/// Kernel dispatch intentionally remains outside P1 and is added only after the
/// portable kernel-package contract is frozen.
pub trait PortableQueue {
    type Buffer: PortableBuffer;
    type Fence: PortableFence;

    /// Copy caller-owned bytes into a buffer.
    ///
    /// The backend must consume or stage the input bytes before returning so
    /// the returned fence never borrows the input slice.
    fn write_buffer(
        &self,
        buffer: &mut Self::Buffer,
        offset_bytes: u64,
        bytes: &[u8],
    ) -> Result<Self::Fence>;

    /// Read a byte range after synchronizing any work needed for that range.
    fn read_buffer(
        &self,
        buffer: &Self::Buffer,
        offset_bytes: u64,
        size_bytes: u64,
    ) -> Result<Vec<u8>>;
}

/// Device capable of creating portable buffers and a submission queue.
pub trait PortableDevice {
    type Buffer: PortableBuffer;
    type Queue: PortableQueue<Buffer = Self::Buffer>;

    /// Return stable backend identity without exposing vendor handles.
    fn backend_id(&self) -> &BackendId;

    /// Return validated device capabilities.
    fn capabilities(&self) -> &CapabilitySet;

    /// Allocate one buffer under the portable descriptor contract.
    ///
    /// Implementations must revalidate public descriptor fields and backend
    /// limits before allocation, including descriptors constructed as literals.
    fn create_buffer(&self, descriptor: BufferDesc) -> Result<Self::Buffer>;

    /// Create a queue suitable for data movement on this device.
    fn create_queue(&self) -> Result<Self::Queue>;
}

/// Portable-runtime error independent of vendor error codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortableError {
    InvalidDescriptor(&'static str),
    OutOfBounds {
        offset_bytes: u64,
        size_bytes: u64,
        buffer_bytes: u64,
    },
    Unsupported(String),
    Backend(String),
}

impl fmt::Display for PortableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDescriptor(message) => formatter.write_str(message),
            Self::OutOfBounds {
                offset_bytes,
                size_bytes,
                buffer_bytes,
            } => write!(
                formatter,
                "range offset={offset_bytes} size={size_bytes} exceeds buffer size {buffer_bytes}"
            ),
            Self::Unsupported(message) => write!(formatter, "unsupported operation: {message}"),
            Self::Backend(message) => write!(formatter, "backend error: {message}"),
        }
    }
}

impl std::error::Error for PortableError {}

/// Result type for backend-neutral runtime operations.
pub type Result<T> = std::result::Result<T, PortableError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestBuffer {
        descriptor: BufferDesc,
        bytes: Vec<u8>,
    }

    impl PortableBuffer for TestBuffer {
        fn descriptor(&self) -> BufferDesc {
            self.descriptor
        }
    }

    #[derive(Debug)]
    struct TestFence;

    impl PortableFence for TestFence {
        fn status(&self) -> Result<FenceStatus> {
            Ok(FenceStatus::Complete)
        }

        fn wait(&self) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Debug)]
    struct TestQueue;

    fn checked_range(total: usize, offset: u64, size: u64) -> Result<core::ops::Range<usize>> {
        let start = usize::try_from(offset)
            .map_err(|_| PortableError::Backend("offset does not fit usize".to_string()))?;
        let length = usize::try_from(size)
            .map_err(|_| PortableError::Backend("size does not fit usize".to_string()))?;
        let end = start
            .checked_add(length)
            .ok_or(PortableError::OutOfBounds {
                offset_bytes: offset,
                size_bytes: size,
                buffer_bytes: total as u64,
            })?;
        if end > total {
            return Err(PortableError::OutOfBounds {
                offset_bytes: offset,
                size_bytes: size,
                buffer_bytes: total as u64,
            });
        }
        Ok(start..end)
    }

    impl PortableQueue for TestQueue {
        type Buffer = TestBuffer;
        type Fence = TestFence;

        fn write_buffer(
            &self,
            buffer: &mut Self::Buffer,
            offset_bytes: u64,
            bytes: &[u8],
        ) -> Result<Self::Fence> {
            let range = checked_range(buffer.bytes.len(), offset_bytes, bytes.len() as u64)?;
            buffer.bytes[range].copy_from_slice(bytes);
            Ok(TestFence)
        }

        fn read_buffer(
            &self,
            buffer: &Self::Buffer,
            offset_bytes: u64,
            size_bytes: u64,
        ) -> Result<Vec<u8>> {
            let range = checked_range(buffer.bytes.len(), offset_bytes, size_bytes)?;
            Ok(buffer.bytes[range].to_vec())
        }
    }

    #[derive(Debug)]
    struct TestDevice {
        id: BackendId,
        capabilities: CapabilitySet,
    }

    impl PortableDevice for TestDevice {
        type Buffer = TestBuffer;
        type Queue = TestQueue;

        fn backend_id(&self) -> &BackendId {
            &self.id
        }

        fn capabilities(&self) -> &CapabilitySet {
            &self.capabilities
        }

        fn create_buffer(&self, descriptor: BufferDesc) -> Result<Self::Buffer> {
            let descriptor = descriptor.validate()?;
            let size = usize::try_from(descriptor.size_bytes).map_err(|_| {
                PortableError::Backend("buffer size does not fit usize".to_string())
            })?;
            if descriptor.size_bytes > self.capabilities.max_buffer_bytes {
                return Err(PortableError::Unsupported(
                    "buffer exceeds device capability".to_string(),
                ));
            }
            Ok(TestBuffer {
                descriptor,
                bytes: vec![0; size],
            })
        }

        fn create_queue(&self) -> Result<Self::Queue> {
            Ok(TestQueue)
        }
    }

    fn capabilities() -> CapabilitySet {
        CapabilitySet {
            max_buffer_bytes: 1 << 20,
            max_workgroup_invocations: 256,
            max_workgroup_size: [256, 256, 64],
            max_bindings: 8,
            supports_f16: false,
            supports_timestamps: true,
        }
        .validate()
        .unwrap()
    }

    #[test]
    fn backend_identity_rejects_empty_name() {
        assert!(BackendId::new(BackendFamily::Cpu, " ").is_err());
        let id = BackendId::new(BackendFamily::Wgpu, "portable-test").unwrap();
        assert_eq!(id.family(), BackendFamily::Wgpu);
        assert_eq!(id.name(), "portable-test");
    }

    #[test]
    fn buffer_descriptor_requires_size_and_usage() {
        assert!(BufferDesc::new(0, BufferUsages::STORAGE, MemoryClass::Host).is_err());
        assert!(BufferDesc::new(8, BufferUsages::EMPTY, MemoryClass::Host).is_err());
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let descriptor = BufferDesc::new(8, usage, MemoryClass::DeviceLocal).unwrap();
        assert!(descriptor.usages.contains(BufferUsages::STORAGE));
        assert!(descriptor.usages.contains(BufferUsages::COPY_DST));
        assert!(!descriptor.usages.contains(BufferUsages::COPY_SRC));
    }

    #[test]
    fn public_descriptor_mutation_is_revalidated() {
        let device = TestDevice {
            id: BackendId::new(BackendFamily::Cpu, "test-cpu").unwrap(),
            capabilities: capabilities(),
        };
        let valid = BufferDesc::new(8, BufferUsages::STORAGE, MemoryClass::Host).unwrap();
        assert_eq!(valid.validate().unwrap(), valid);
        let invalid_size = BufferDesc {
            size_bytes: 0,
            ..valid
        };
        let invalid_usage = BufferDesc {
            usages: BufferUsages::EMPTY,
            ..valid
        };
        for invalid in [invalid_size, invalid_usage] {
            assert!(matches!(
                invalid.validate(),
                Err(PortableError::InvalidDescriptor(_))
            ));
            assert!(matches!(
                device.create_buffer(invalid),
                Err(PortableError::InvalidDescriptor(_))
            ));
        }
    }

    #[test]
    fn capability_validation_fails_closed() {
        let mut invalid = capabilities();
        invalid.max_bindings = 0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn portable_traits_support_round_trip_without_vendor_types() {
        let device = TestDevice {
            id: BackendId::new(BackendFamily::Cpu, "test-cpu").unwrap(),
            capabilities: capabilities(),
        };
        let descriptor = BufferDesc::new(
            16,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            MemoryClass::Host,
        )
        .unwrap();
        let mut buffer = device.create_buffer(descriptor).unwrap();
        let queue = device.create_queue().unwrap();

        let fence = queue.write_buffer(&mut buffer, 4, &[1, 2, 3, 4]).unwrap();
        assert_eq!(fence.status().unwrap(), FenceStatus::Complete);
        fence.wait().unwrap();
        assert_eq!(queue.read_buffer(&buffer, 4, 4).unwrap(), vec![1, 2, 3, 4]);
        assert!(matches!(
            queue.read_buffer(&buffer, 14, 4),
            Err(PortableError::OutOfBounds { .. })
        ));
    }
}
