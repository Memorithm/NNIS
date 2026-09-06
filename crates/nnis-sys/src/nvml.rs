//! Runtime-resolved NVIDIA Management Library API (`libnvidia-ml.so.1`).
//!
//! Only the process-memory query surface required by NNIS is loaded. There is
//! no link-time NVML dependency: unsupported hosts fail closed at runtime.

use crate::LibraryError;
use libloading::Library;
use std::sync::OnceLock;

type CChar = core::ffi::c_char;
type CUInt = core::ffi::c_uint;
type CVoid = core::ffi::c_void;

/// Raw NVML return value (`nvmlReturn_t`).
pub type NvmlReturn = i32;
/// Opaque NVML device handle (`nvmlDevice_t`).
pub type NvmlDevice = *mut CVoid;

/// NVML success.
pub const NVML_SUCCESS: NvmlReturn = 0;
/// Caller-provided process array was too small.
pub const NVML_ERROR_INSUFFICIENT_SIZE: NvmlReturn = 7;
/// Sentinel used by unsigned NVML fields when the value is unavailable.
pub const NVML_VALUE_NOT_AVAILABLE: u64 = u64::MAX;

/// `nvmlProcessInfo_t` as defined by current NVML headers.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NvmlProcessInfo {
    pub pid: CUInt,
    pub used_gpu_memory: u64,
    pub gpu_instance_id: CUInt,
    pub compute_instance_id: CUInt,
}

/// The minimal NVML API used by NNIS process-scoped memory telemetry.
pub struct NvmlApi {
    pub nvmlInit_v2: unsafe extern "C" fn() -> NvmlReturn,
    pub nvmlShutdown: unsafe extern "C" fn() -> NvmlReturn,
    pub nvmlErrorString: unsafe extern "C" fn(NvmlReturn) -> *const CChar,
    pub nvmlDeviceGetHandleByUUID:
        unsafe extern "C" fn(*const CChar, *mut NvmlDevice) -> NvmlReturn,
    pub nvmlDeviceGetComputeRunningProcesses_v3:
        unsafe extern "C" fn(NvmlDevice, *mut CUInt, *mut NvmlProcessInfo) -> NvmlReturn,
    _library: Library,
}

unsafe fn resolve<T: Copy>(
    lib: &Library,
    library: &'static str,
    candidates: &[&str],
) -> Result<T, LibraryError> {
    let mut detail = String::from("no symbol candidate was provided");
    for sym in candidates {
        match lib.get::<T>(sym.as_bytes()) {
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

/// Canonical NVML soname.
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

/// Load the process-wide NVML symbol table without initializing NVML.
pub fn api() -> Result<&'static NvmlApi, LibraryError> {
    API.get_or_init(|| {
        let library = open_library()?;
        unsafe {
            Ok(NvmlApi {
                nvmlInit_v2: resolve(&library, LIB, &["nvmlInit_v2"])?,
                nvmlShutdown: resolve(&library, LIB, &["nvmlShutdown"])?,
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

/// Human-readable NVML error string when the library is available.
pub fn error_string(code: NvmlReturn) -> String {
    let Ok(api) = api() else {
        return format!("NVML error {code}");
    };
    let pointer = unsafe { (api.nvmlErrorString)(code) };
    if pointer.is_null() {
        return format!("NVML error {code}");
    }
    unsafe { std::ffi::CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}
