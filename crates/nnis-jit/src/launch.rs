//! Validated CUDA kernel launch configuration and argument packing.

use crate::module::Kernel;
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use nnis_sys::driver;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    pub const fn x(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    fn volume(self) -> Option<u64> {
        u64::from(self.x)
            .checked_mul(u64::from(self.y))?
            .checked_mul(u64::from(self.z))
    }

    fn has_zero_axis(self) -> bool {
        self.x == 0 || self.y == 0 || self.z == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid: Dim3,
    pub block: Dim3,
    pub dynamic_shared_memory_bytes: u32,
}

impl LaunchConfig {
    pub const fn new(grid: Dim3, block: Dim3) -> Self {
        Self {
            grid,
            block,
            dynamic_shared_memory_bytes: 0,
        }
    }

    pub const fn with_dynamic_shared_memory(mut self, bytes: u32) -> Self {
        self.dynamic_shared_memory_bytes = bytes;
        self
    }

    pub fn for_num_elements(elements: usize, block_size: u32) -> Result<Self> {
        if elements == 0 {
            return Err(NnisError::invalid_input(
                "cannot construct a kernel grid for zero elements",
            ));
        }
        if block_size == 0 {
            return Err(NnisError::invalid_input("kernel block size is zero"));
        }
        let blocks = elements.div_ceil(block_size as usize);
        let blocks = u32::try_from(blocks)
            .map_err(|_| NnisError::invalid_input("kernel grid exceeds u32::MAX blocks"))?;
        Ok(Self::new(Dim3::x(blocks), Dim3::x(block_size)))
    }
}

mod private {
    pub trait Sealed {}
}

/// Host values that can be copied into CUDA's kernel parameter buffer.
pub trait KernelParameter: private::Sealed + Copy + 'static {}

macro_rules! kernel_parameters {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl private::Sealed for $ty {}
            impl KernelParameter for $ty {}
        )+
    };
}

kernel_parameters!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize, f32, f64);

/// Inline storage sized and aligned for every sealed `KernelParameter`.
/// Values are copied byte-for-byte so CUDA observes their native host
/// representation without one heap allocation per argument.
#[repr(C, align(8))]
struct ArgumentStorage([u8; 8]);

impl ArgumentStorage {
    fn new<T: KernelParameter>(value: T) -> Self {
        assert!(std::mem::size_of::<T>() <= std::mem::size_of::<Self>());
        assert!(std::mem::align_of::<T>() <= std::mem::align_of::<Self>());
        let mut storage = Self([0; 8]);
        // SAFETY: the sealed parameter types are plain numeric values of at
        // most eight bytes. Both ranges are valid and cannot overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                std::ptr::from_ref(&value).cast::<u8>(),
                storage.0.as_mut_ptr(),
                std::mem::size_of::<T>(),
            );
        }
        storage
    }

    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.0.as_mut_ptr().cast()
    }
}

/// Owned, correctly aligned host storage for a kernel's arguments.
///
/// CUDA copies these values during `cuLaunchKernel`; each raw entry passed to
/// the driver points to the storage containing the value, never to the value
/// interpreted as a host address. Buffer borrows prevent allocation teardown
/// while this argument pack remains alive.
#[derive(Default)]
pub struct KernelArgs<'buffers> {
    values: Vec<ArgumentStorage>,
    buffer_contexts: Vec<Arc<Context>>,
    _buffers: PhantomData<&'buffers ()>,
}

impl<'buffers> KernelArgs<'buffers> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Preallocate storage for a known signature. `buffer_arguments` is the
    /// subset of arguments supplied through [`Self::push_buffer`].
    pub fn with_capacity(arguments: usize, buffer_arguments: usize) -> Self {
        Self {
            values: Vec::with_capacity(arguments),
            buffer_contexts: Vec::with_capacity(buffer_arguments),
            _buffers: PhantomData,
        }
    }

    pub fn push<T: KernelParameter>(&mut self, value: T) -> &mut Self {
        self.values.push(ArgumentStorage::new(value));
        self
    }

    pub fn push_buffer<T>(&mut self, buffer: &'buffers DeviceBuffer<T>) -> &mut Self {
        self.values.push(ArgumentStorage::new(buffer.device_ptr()));
        self.buffer_contexts.push(Arc::clone(buffer.ctx()));
        self
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn raw_pointers(&mut self) -> Vec<*mut c_void> {
        self.values
            .iter_mut()
            .map(|argument| argument.as_mut_ptr())
            .collect()
    }
}

/// A kernel, execution stream, and validated launch shape.
pub struct KernelLaunch<'a> {
    kernel: &'a Kernel,
    stream: &'a Stream,
    config: LaunchConfig,
}

impl<'a> KernelLaunch<'a> {
    pub fn new(kernel: &'a Kernel, stream: &'a Stream, config: LaunchConfig) -> Self {
        Self {
            kernel,
            stream,
            config,
        }
    }

    pub fn config(&self) -> LaunchConfig {
        self.config
    }

    fn validate(&self, args: &KernelArgs<'_>) -> Result<()> {
        if self.config.grid.has_zero_axis() {
            return Err(NnisError::invalid_input("kernel grid contains a zero axis"));
        }
        if self.config.block.has_zero_axis() {
            return Err(NnisError::invalid_input(
                "kernel block contains a zero axis",
            ));
        }
        let block_threads = self
            .config
            .block
            .volume()
            .ok_or_else(|| NnisError::invalid_input("kernel block volume overflows u64"))?;
        let properties = self.kernel.context().props();
        if block_threads > u64::from(properties.max_threads_per_block) {
            return Err(NnisError::invalid_input(format!(
                "kernel block has {block_threads} threads; device limit is {}",
                properties.max_threads_per_block
            )));
        }
        if self.config.dynamic_shared_memory_bytes > properties.shared_memory_per_block {
            return Err(NnisError::invalid_input(format!(
                "dynamic shared memory is {} bytes; default device limit is {}",
                self.config.dynamic_shared_memory_bytes, properties.shared_memory_per_block
            )));
        }
        if !Arc::ptr_eq(self.kernel.context(), self.stream.ctx()) {
            return Err(NnisError::invalid_input(
                "kernel module and stream belong to different contexts",
            ));
        }
        if args
            .buffer_contexts
            .iter()
            .any(|context| !Arc::ptr_eq(context, self.kernel.context()))
        {
            return Err(NnisError::invalid_input(
                "a kernel buffer belongs to a different context",
            ));
        }
        Ok(())
    }

    /// Submit this kernel to the stream.
    ///
    /// # Safety
    ///
    /// NNIS validates the launch dimensions, contexts, argument storage, and
    /// native handles. The caller must still ensure that the argument order
    /// and Rust value widths match the CUDA kernel signature, and that the
    /// kernel, arguments, and referenced buffers remain alive until the stream
    /// has completed this launch.
    pub unsafe fn launch(&self, args: &mut KernelArgs<'_>) -> Result<()> {
        self.validate(args)?;
        self.stream.ctx().set_current()?;
        let api = driver::api()?;
        let mut raw_arguments = args.raw_pointers();
        let argument_pointer = if raw_arguments.is_empty() {
            std::ptr::null_mut()
        } else {
            raw_arguments.as_mut_ptr()
        };
        // SAFETY: validated handles/configuration and caller-provided kernel
        // signature/lifetime invariants are documented above. CUDA consumes
        // the parameter values before this function returns.
        let result = unsafe {
            (api.cuLaunchKernel)(
                self.kernel.raw(),
                self.config.grid.x,
                self.config.grid.y,
                self.config.grid.z,
                self.config.block.x,
                self.config.block.y,
                self.config.block.z,
                self.config.dynamic_shared_memory_bytes,
                self.stream.raw(),
                argument_pointer,
                std::ptr::null_mut(),
            )
        };
        if result != 0 {
            return Err(NnisError::driver("cuLaunchKernel", result)
                .with("kernel", self.kernel.name())
                .with(
                    "grid",
                    format_args!(
                        "{}x{}x{}",
                        self.config.grid.x, self.config.grid.y, self.config.grid.z
                    ),
                )
                .with(
                    "block",
                    format_args!(
                        "{}x{}x{}",
                        self.config.block.x, self.config.block.y, self.config.block.z
                    ),
                ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_grid_rounds_up() {
        let config = LaunchConfig::for_num_elements(257, 256).unwrap();
        assert_eq!(config.grid, Dim3::x(2));
        assert_eq!(config.block, Dim3::x(256));
    }

    #[test]
    fn element_grid_rejects_zeroes() {
        assert!(LaunchConfig::for_num_elements(0, 256).is_err());
        assert!(LaunchConfig::for_num_elements(1, 0).is_err());
    }

    #[test]
    fn inline_argument_storage_preserves_values_and_alignment() {
        let mut arguments = KernelArgs::with_capacity(5, 0);
        arguments
            .push(0x5a_u8)
            .push(-12_345_i32)
            .push(0x1122_3344_5566_7788_u64)
            .push(-3.25_f32)
            .push(7.5_f64);
        let pointers = arguments.raw_pointers();
        for pointer in &pointers {
            assert_eq!((*pointer as usize) % std::mem::align_of::<u64>(), 0);
        }
        // SAFETY: each pointer targets live aligned storage populated with the
        // exact type read here, and `arguments` has not moved or mutated.
        unsafe {
            assert_eq!(*pointers[0].cast::<u8>(), 0x5a);
            assert_eq!(*pointers[1].cast::<i32>(), -12_345);
            assert_eq!(*pointers[2].cast::<u64>(), 0x1122_3344_5566_7788);
            assert_eq!(*pointers[3].cast::<f32>(), -3.25);
            assert_eq!(*pointers[4].cast::<f64>(), 7.5);
        }
    }
}
