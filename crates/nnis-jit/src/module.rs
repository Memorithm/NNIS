//! CUDA module and function ownership.

use crate::{CodeKind, CompiledCode};
use nnis_rt::context::Context;
use nnis_rt::error::{NnisError, Result};
use nnis_sys::constants as cu;
use nnis_sys::driver;
use std::ffi::CString;
use std::sync::{Arc, OnceLock};

/// Immutable resource usage and code-generation properties of a CUDA kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelAttributes {
    pub max_threads_per_block: u32,
    pub static_shared_memory_bytes: u32,
    pub constant_memory_bytes: u32,
    pub local_memory_bytes_per_thread: u32,
    pub registers_per_thread: u32,
    /// PTX ISA version as `(major, minor)`, when reported by the driver.
    pub ptx_version: Option<(u32, u32)>,
    /// Target binary compute capability as `(major, minor)`, when reported.
    pub binary_version: Option<(u32, u32)>,
    pub cache_mode_ca: bool,
    pub max_dynamic_shared_memory_bytes: u32,
}

/// CUDA's occupancy-based suggestion for a kernel launch shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OccupancyRecommendation {
    /// Thread-block width suggested by the CUDA occupancy calculator.
    pub block_size: u32,
    /// Grid width needed to reach the calculated maximum occupancy.
    pub minimum_grid_size: u32,
    /// Maximum simultaneously active blocks on one multiprocessor.
    pub active_blocks_per_multiprocessor: u32,
}

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
            attributes: Arc::new(OnceLock::new()),
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
    attributes: Arc<OnceLock<KernelAttributes>>,
}

impl Kernel {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.module.context
    }

    /// Query and cache this function's immutable CUDA attributes.
    pub fn attributes(&self) -> Result<KernelAttributes> {
        if let Some(attributes) = self.attributes.get() {
            return Ok(*attributes);
        }

        let attributes = KernelAttributes {
            max_threads_per_block: self
                .nonnegative_attribute(cu::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?,
            static_shared_memory_bytes: self
                .nonnegative_attribute(cu::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?,
            constant_memory_bytes: self
                .nonnegative_attribute(cu::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES)?,
            local_memory_bytes_per_thread: self
                .nonnegative_attribute(cu::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
            registers_per_thread: self.nonnegative_attribute(cu::CU_FUNC_ATTRIBUTE_NUM_REGS)?,
            ptx_version: encoded_version(
                self.nonnegative_attribute(cu::CU_FUNC_ATTRIBUTE_PTX_VERSION)?,
            ),
            binary_version: encoded_version(
                self.nonnegative_attribute(cu::CU_FUNC_ATTRIBUTE_BINARY_VERSION)?,
            ),
            cache_mode_ca: self.attribute(cu::CU_FUNC_ATTRIBUTE_CACHE_MODE_CA)? != 0,
            max_dynamic_shared_memory_bytes: self
                .nonnegative_attribute(cu::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)?,
        };

        // Concurrent first queries may race, but the immutable values are
        // identical. Keep whichever complete result reached the cell first.
        let _ = self.attributes.set(attributes);
        Ok(*self
            .attributes
            .get()
            .expect("kernel attributes were initialized"))
    }

    /// Maximum active blocks per multiprocessor for one proposed block shape.
    pub fn max_active_blocks_per_multiprocessor(
        &self,
        block_size: u32,
        dynamic_shared_memory_bytes: usize,
    ) -> Result<u32> {
        let attributes = self.attributes()?;
        let block_size = validate_block_size(block_size, attributes.max_threads_per_block)?;
        validate_dynamic_shared_memory(dynamic_shared_memory_bytes, attributes)?;
        self.module.context.set_current()?;
        let api = driver::api()?;
        let mut active_blocks = 0;
        // SAFETY: the function and context are live; the output pointer is
        // valid, and the proposed launch resources were validated above.
        let result = unsafe {
            (api.cuOccupancyMaxActiveBlocksPerMultiprocessor)(
                &mut active_blocks,
                self.handle,
                block_size,
                dynamic_shared_memory_bytes,
            )
        };
        if result != 0 {
            return Err(
                NnisError::driver("cuOccupancyMaxActiveBlocksPerMultiprocessor", result)
                    .with("kernel", self.name())
                    .with("block_size", block_size)
                    .with("dynamic_shared_memory_bytes", dynamic_shared_memory_bytes),
            );
        }
        nonnegative_driver_output("cuOccupancyMaxActiveBlocksPerMultiprocessor", active_blocks)
    }

    /// Ask CUDA for a block size that maximizes active-warps occupancy.
    ///
    /// `block_size_limit` is an optional application constraint. `None` lets
    /// CUDA use the function/device limit. Dynamic shared memory is treated as
    /// a constant per block; block-size-dependent callbacks are intentionally
    /// not exposed through this safe API.
    pub fn recommend_occupancy(
        &self,
        dynamic_shared_memory_bytes: usize,
        block_size_limit: Option<u32>,
    ) -> Result<OccupancyRecommendation> {
        let attributes = self.attributes()?;
        validate_dynamic_shared_memory(dynamic_shared_memory_bytes, attributes)?;
        let block_size_limit = match block_size_limit {
            None => 0,
            Some(limit) => validate_block_size(limit, attributes.max_threads_per_block)?,
        };

        self.module.context.set_current()?;
        let api = driver::api()?;
        let mut minimum_grid_size = 0;
        let mut block_size = 0;
        // SAFETY: the function/context and output pointers are valid. A null
        // callback means `dynamic_shared_memory_bytes` is constant per block.
        let result = unsafe {
            (api.cuOccupancyMaxPotentialBlockSize)(
                &mut minimum_grid_size,
                &mut block_size,
                self.handle,
                None,
                dynamic_shared_memory_bytes,
                block_size_limit,
            )
        };
        if result != 0 {
            return Err(
                NnisError::driver("cuOccupancyMaxPotentialBlockSize", result)
                    .with("kernel", self.name())
                    .with("block_size_limit", block_size_limit)
                    .with("dynamic_shared_memory_bytes", dynamic_shared_memory_bytes),
            );
        }

        let minimum_grid_size = positive_driver_output(
            "cuOccupancyMaxPotentialBlockSize(minGridSize)",
            minimum_grid_size,
        )?;
        let block_size =
            positive_driver_output("cuOccupancyMaxPotentialBlockSize(blockSize)", block_size)?;
        let active_blocks_per_multiprocessor =
            self.max_active_blocks_per_multiprocessor(block_size, dynamic_shared_memory_bytes)?;
        Ok(OccupancyRecommendation {
            block_size,
            minimum_grid_size,
            active_blocks_per_multiprocessor,
        })
    }

    fn attribute(&self, attribute: i32) -> Result<i32> {
        self.module.context.set_current()?;
        let api = driver::api()?;
        let mut value = 0;
        // SAFETY: this kernel keeps its module live, the context is current,
        // and `value` is a valid output pointer.
        let result = unsafe { (api.cuFuncGetAttribute)(&mut value, attribute, self.handle) };
        if result != 0 {
            return Err(NnisError::driver("cuFuncGetAttribute", result)
                .with("kernel", self.name())
                .with("attribute", attribute));
        }
        Ok(value)
    }

    fn nonnegative_attribute(&self, attribute: i32) -> Result<u32> {
        nonnegative_driver_output("cuFuncGetAttribute", self.attribute(attribute)?)
    }

    pub(crate) fn raw(&self) -> nnis_sys::CUfunction {
        self.handle
    }
}

fn encoded_version(value: u32) -> Option<(u32, u32)> {
    (value != 0).then_some((value / 10, value % 10))
}

fn validate_block_size(block_size: u32, maximum: u32) -> Result<i32> {
    if block_size == 0 {
        return Err(NnisError::invalid_input("occupancy block size is zero"));
    }
    if block_size > maximum {
        return Err(NnisError::invalid_input(format!(
            "occupancy block size {block_size} exceeds kernel limit {maximum}"
        )));
    }
    i32::try_from(block_size)
        .map_err(|_| NnisError::invalid_input("occupancy block size exceeds i32::MAX"))
}

fn validate_dynamic_shared_memory(
    dynamic_shared_memory_bytes: usize,
    attributes: KernelAttributes,
) -> Result<()> {
    if dynamic_shared_memory_bytes > attributes.max_dynamic_shared_memory_bytes as usize {
        return Err(NnisError::invalid_input(format!(
            "dynamic shared memory is {dynamic_shared_memory_bytes} bytes; kernel limit is {}",
            attributes.max_dynamic_shared_memory_bytes
        )));
    }
    Ok(())
}

fn nonnegative_driver_output(operation: &str, value: i32) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        NnisError::unsupported(format!(
            "{operation} returned an invalid negative value {value}"
        ))
    })
}

fn positive_driver_output(operation: &str, value: i32) -> Result<u32> {
    let value = nonnegative_driver_output(operation, value)?;
    if value == 0 {
        return Err(NnisError::unsupported(format!("{operation} returned zero")));
    }
    Ok(value)
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
