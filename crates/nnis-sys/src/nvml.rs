//! Runtime-resolved NVIDIA Management Library (NVML) surface.
//!
//! NNIS deliberately loads NVML at runtime. The inference runtime therefore
//! keeps no link-time dependency on `libnvidia-ml`, and platforms that do not
//! ship or support NVML remain an explicit capability-negative state.
//!
//! Signatures and return codes in this module mirror the public NVML API. Only
//! the small subset required for process-scoped compute-memory observation is
//! exposed.

use crate::LibraryError;
use libloading::Library;
use std::sync::OnceLock;

pub type nvmlReturn_t = core::ffi::c_int;
pub type nvmlDevice_t = *mut core::ffi::c_void;

pub const NVML_SUCCESS: nvmlReturn_t = 0;
pub const NVML_ERROR_UNINITIALIZED: nvmlReturn_t = 1;
pub const NVML_ERROR_INVALID_ARGUMENT: nvmlReturn_t = 2;
pub const NVML_ERROR_NOT_SUPPORTED: nvmlReturn_t = 3;
pub const NVML_ERROR_NO_PERMISSION: nvmlReturn_t = 4;
/// Legacy status returned by older NVML versions when already initialized.
pub const NVML_ERROR_ALREADY_INITIALIZED: nvmlReturn_t = 5;
pub const NVML_ERROR_NOT_FOUND: nvmlReturn_t = 6;
pub const NVML_ERROR_INSUFFICIENT_SIZE: nvmlReturn_t = 7;

/// NVML sentinel used when a per-process memory value cannot be reported.
pub const NVML_VALUE_NOT_AVAILABLE: u64 = u64::MAX;

/// `nvmlProcessInfo_t` as defined by the public NVML API.
///
/// The v3 compute-process query populates this structure. `usedGpuMemory` is
/// reported by NVML as the amount of GPU memory used by the application.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct nvmlProcessInfo_t {
    pub pid: core::ffi::c_uint,
    pub usedGpuMemory: u64,
    pub gpuInstanceId: core::ffi::c_uint,
    pub computeInstanceId: core::ffi::c_uint,
}

impl Default for nvmlProcessInfo_t {
    fn default() -> Self {
        Self {
            pid: 0,
            usedGpuMemory: NVML_VALUE_NOT_AVAILABLE,
            gpuInstanceId: 0,
            computeInstanceId: 0,
        }
    }
}

/// Minimal NVML API used by NNIS.
///
/// Function pointers remain valid for the process lifetime because the loaded
/// library handle is retained in this process-global structure.
pub struct NvmlApi {
    pub nvmlInit_v2: unsafe extern "C" fn() -> nvmlReturn_t,
    pub nvmlErrorString: unsafe extern "C" fn(nvmlReturn_t) -> *const core::ffi::c_char,
    pub nvmlDeviceGetHandleByUUID:
        unsafe extern "C" fn(*const core::ffi::c_char, *mut nvmlDevice_t) -> nvmlReturn_t,
    pub nvmlDeviceGetComputeRunningProcesses_v3: unsafe extern "C" fn(
        nvmlDevice_t,
        *mut core::ffi::c_uint,
        *mut nvmlProcessInfo_t,
    ) -> nvmlReturn_t,
    _library: Library,
}

unsafe fn resolve<T: Copy>(
    lib: &Library,
    library: &'static str,
    candidates: &[&str],
) -> Result<T, LibraryError> {
    let mut detail = String::from("no symbol candidate was provided");
    for symbol in candidates {
        match lib.get::<T>(symbol.as_bytes()) {
            Ok(function) => return Ok(*function),
            Err(error) => detail = error.to_string(),
        }
    }
    Err(LibraryError {
        library,
        candidates: candidates
            .iter()
            .map(|candidate| candidate.to_string())
            .collect(),
        detail,
    })
}

pub const LIB: &str = "libnvidia-ml.so.1";

fn open_library() -> Result<Library, LibraryError> {
    let mut candidates = Vec::new();
    let mut detail = String::from("no library could be opened");

    if let Ok(path) = std::env::var("NNIS_NVML_PATH") {
        if !path.is_empty() {
            candidates.push(path.clone());
            unsafe {
                match Library::new(&path) {
                    Ok(library) => return Ok(library),
                    Err(error) => detail = error.to_string(),
                }
            }
        }
    }

    for candidate in ["libnvidia-ml.so.1", "libnvidia-ml.so"] {
        candidates.push(candidate.to_string());
        unsafe {
            match Library::new(candidate) {
                Ok(library) => return Ok(library),
                Err(error) => detail = error.to_string(),
            }
        }
    }

    Err(LibraryError {
        library: LIB,
        candidates,
        detail,
    })
}

static API: OnceLock<Result<NvmlApi, LibraryError>> = OnceLock::new();
static INIT_STATUS: OnceLock<nvmlReturn_t> = OnceLock::new();

/// Acquire the process-global dynamically loaded NVML API.
pub fn api() -> Result<&'static NvmlApi, LibraryError> {
    API.get_or_init(|| {
        let library = open_library()?;
        unsafe {
            Ok(NvmlApi {
                nvmlInit_v2: resolve(&library, LIB, &["nvmlInit_v2"])?,
                nvmlErrorString: resolve(&library, LIB, &["nvmlErrorString"])?,
                nvmlDeviceGetHandleByUUID: resolve(&library, LIB, &["nvmlDeviceGetHandleByUUID"])?,
                nvmlDeviceGetComputeRunningProcesses_v3: resolve(
                    &library,
                    LIB,
                    &["nvmlDeviceGetComputeRunningProcesses_v3"],
                )?,
                _library: library,
            })
        }
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// Initialize NVML once and return the native status.
///
/// The status is retained exactly so callers can distinguish an unavailable or
/// unsupported management capability from a successful initialization.
pub fn init_status() -> Result<nvmlReturn_t, LibraryError> {
    let api = api()?;
    Ok(*INIT_STATUS.get_or_init(|| unsafe { (api.nvmlInit_v2)() }))
}

/// Best-effort human-readable NVML status string.
pub fn error_string(status: nvmlReturn_t) -> String {
    let Ok(api) = api() else {
        return format!("NVML status {status}");
    };
    let pointer = unsafe { (api.nvmlErrorString)(status) };
    if pointer.is_null() {
        return format!("NVML status {status}");
    }
    unsafe { std::ffi::CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_info_layout_matches_v3_abi_on_supported_targets() {
        assert_eq!(core::mem::size_of::<nvmlProcessInfo_t>(), 24);
        assert_eq!(core::mem::align_of::<nvmlProcessInfo_t>(), 8);
    }

    #[test]
    fn public_status_values_match_nvml_contract() {
        assert_eq!(NVML_SUCCESS, 0);
        assert_eq!(NVML_ERROR_NOT_SUPPORTED, 3);
        assert_eq!(NVML_ERROR_NO_PERMISSION, 4);
        assert_eq!(NVML_ERROR_ALREADY_INITIALIZED, 5);
        assert_eq!(NVML_ERROR_INSUFFICIENT_SIZE, 7);
        assert_eq!(NVML_VALUE_NOT_AVAILABLE, u64::MAX);
    }
}
