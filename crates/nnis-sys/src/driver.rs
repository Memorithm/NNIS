//! Runtime-resolved CUDA driver API (`libcuda.so.1`).
//!
//! All signatures are transcribed from `cuda.h` (CUDA 13.0). Symbol lookup
//! order: `$NNIS_CUDA_DRIVER_PATH` (if set), then the candidate sonames.

use crate::{
    CUcontext, CUdevice, CUdeviceptr, CUevent, CUfunction, CUmodule, CUresult, CUstream, CUuuid,
    LibraryError,
};
use libloading::Library;
use std::sync::OnceLock;

type CInt = core::ffi::c_int;
type CUInt = core::ffi::c_uint;
type CChar = core::ffi::c_char;
type CVoid = core::ffi::c_void;
type CSizeT = usize;

/// Occupancy block-size-to-dynamic-shared-memory callback (`CUoccupancyB2DSize`).
pub type OccupancyB2DSize = Option<unsafe extern "C" fn(CInt) -> CSizeT>;

/// The subset of the CUDA driver API used by NNIS.
///
/// Every field is a raw function pointer; calling them is `unsafe` and
/// requires a valid current context where the CUDA docs demand one.
pub struct DriverApi {
    pub cuInit: unsafe extern "C" fn(CUInt) -> CUresult,
    pub cuDriverGetVersion: unsafe extern "C" fn(*mut CInt) -> CUresult,
    pub cuGetErrorName: unsafe extern "C" fn(CUresult, *mut *const CChar) -> CUresult,
    pub cuGetErrorString: unsafe extern "C" fn(CUresult, *mut *const CChar) -> CUresult,

    pub cuDeviceGetCount: unsafe extern "C" fn(*mut CInt) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, CInt) -> CUresult,
    pub cuDeviceGetName: unsafe extern "C" fn(*mut CChar, CInt, CUdevice) -> CUresult,
    pub cuDeviceGetAttribute: unsafe extern "C" fn(*mut CInt, CInt, CUdevice) -> CUresult,
    pub cuDeviceGetUuid: unsafe extern "C" fn(*mut CUuuid, CUdevice) -> CUresult,
    pub cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
    pub cuDevicePrimaryCtxRelease: unsafe extern "C" fn(CUdevice) -> CUresult,

    pub cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxGetCurrent: unsafe extern "C" fn(*mut CUcontext) -> CUresult,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,

    pub cuMemGetInfo: unsafe extern "C" fn(*mut CSizeT, *mut CSizeT) -> CUresult,
    pub cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, CSizeT) -> CUresult,
    pub cuMemFree: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    pub cuMemAllocHost: unsafe extern "C" fn(*mut *mut CVoid, CSizeT) -> CUresult,
    pub cuMemFreeHost: unsafe extern "C" fn(*mut CVoid) -> CUresult,
    /// NOTE: the trailing count is in **elements** (u32 words / bytes).
    pub cuMemsetD32Async: unsafe extern "C" fn(CUdeviceptr, CUInt, CSizeT, CUstream) -> CUresult,
    pub cuMemsetD8Async: unsafe extern "C" fn(CUdeviceptr, CUInt, CSizeT, CUstream) -> CUresult,

    pub cuMemcpyHtoDAsync:
        unsafe extern "C" fn(CUdeviceptr, *const CVoid, CSizeT, CUstream) -> CUresult,
    pub cuMemcpyDtoHAsync:
        unsafe extern "C" fn(*mut CVoid, CUdeviceptr, CSizeT, CUstream) -> CUresult,
    /// Generic asynchronous copy; used for device-to-device transfers.
    /// (`cuMemcpyDtoD`/`cuMemcpyDtoDAsync` resolve to non-functional legacy
    /// stubs returning CUDA_ERROR_INVALID_CONTEXT on some Tegra drivers.)
    pub cuMemcpyAsync: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, CSizeT, CUstream) -> CUresult,

    pub cuStreamCreate: unsafe extern "C" fn(*mut CUstream, CUInt) -> CUresult,
    pub cuStreamDestroy: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuStreamWaitEvent: unsafe extern "C" fn(CUstream, CUevent, CUInt) -> CUresult,
    pub cuStreamQuery: unsafe extern "C" fn(CUstream) -> CUresult,

    pub cuEventCreate: unsafe extern "C" fn(*mut CUevent, CUInt) -> CUresult,
    pub cuEventDestroy: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    pub cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventQuery: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventElapsedTime: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult,

    pub cuModuleLoadDataEx: unsafe extern "C" fn(
        *mut CUmodule,
        *const CVoid,
        CUInt,
        *mut CInt,
        *mut *mut CVoid,
    ) -> CUresult,
    pub cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUresult,
    pub cuModuleGetFunction:
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const CChar) -> CUresult,

    pub cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        CUInt,
        CUInt,
        CUInt,
        CUInt,
        CUInt,
        CUInt,
        CUInt,
        CUstream,
        *mut *mut CVoid,
        *mut *mut CVoid,
    ) -> CUresult,
    pub cuFuncGetAttribute: unsafe extern "C" fn(*mut CInt, CInt, CUfunction) -> CUresult,
    pub cuFuncSetAttribute: unsafe extern "C" fn(CUfunction, CInt, CInt) -> CUresult,

    pub cuOccupancyMaxActiveBlocksPerMultiprocessor:
        unsafe extern "C" fn(*mut CInt, CUfunction, CInt, CSizeT) -> CUresult,
    pub cuOccupancyMaxPotentialBlockSize: unsafe extern "C" fn(
        *mut CInt,
        *mut CInt,
        CUfunction,
        crate::driver::OccupancyB2DSize,
        CSizeT,
        CInt,
    ) -> CUresult,
}

unsafe fn resolve<T: Copy>(
    lib: &Library,
    library: &'static str,
    candidates: &[&str],
) -> Result<T, LibraryError> {
    for sym in candidates {
        if let Ok(f) = lib.get::<T>(sym.as_bytes()) {
            return Ok(*f);
        }
    }
    Err(LibraryError {
        library,
        candidates: unsafe { std::mem::transmute::<&[&str], &'static [&'static str]>(candidates) },
        detail: format!("symbol {} not found", candidates[0]),
    })
}

macro_rules! load {
    ($lib:expr, $($field:ident : [$($sym:literal),+ $(,)?]),+ $(,)?) => {{
        Ok(DriverApi { $($field: resolve(&$lib, LIB, &[$($sym),+])?,)+ })
    }};
}

/// Soname candidates for the driver library.
pub const LIB: &str = "libcuda.so.1";

fn open_library() -> Result<Library, LibraryError> {
    let mut candidates: Vec<&str> = Vec::new();
    let mut detail = String::from("no library could be opened");
    if let Ok(p) = std::env::var("NNIS_CUDA_DRIVER_PATH") {
        if !p.is_empty() {
            unsafe {
                if let Ok(l) = Library::new(&p) {
                    return Ok(l);
                }
            }
            detail = format!("NNIS_CUDA_DRIVER_PATH={p} could not be opened");
        }
    }
    candidates.push("libcuda.so.1");
    candidates.push("libcuda.so");
    for c in &candidates {
        unsafe {
            if let Ok(l) = Library::new(c) {
                return Ok(l);
            }
        }
    }
    Err(LibraryError {
        library: "libcuda.so.1",
        candidates: Box::leak(candidates.into_boxed_slice()),
        detail,
    })
}

static API: OnceLock<Result<DriverApi, LibraryError>> = OnceLock::new();

/// Acquire the process-wide driver API. Idempotent; initializes CUDA exactly
/// once (`cuInit(0)`).
pub fn api() -> Result<&'static DriverApi, LibraryError> {
    API
        .get_or_init(|| {
            let lib = open_library()?;
            unsafe {
                let api: DriverApi = load!(
                    lib,
                    cuInit: ["cuInit"],
                    cuDriverGetVersion: ["cuDriverGetVersion"],
                    cuGetErrorName: ["cuGetErrorName"],
                    cuGetErrorString: ["cuGetErrorString"],
                    cuDeviceGetCount: ["cuDeviceGetCount"],
                    cuDeviceGet: ["cuDeviceGet"],
                    cuDeviceGetName: ["cuDeviceGetName"],
                    cuDeviceGetAttribute: ["cuDeviceGetAttribute"],
                    cuDeviceGetUuid: ["cuDeviceGetUuid_v2", "cuDeviceGetUuid"],
                    cuDevicePrimaryCtxRetain: ["cuDevicePrimaryCtxRetain"],
                    cuDevicePrimaryCtxRelease: ["cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease"],
                    cuCtxSetCurrent: ["cuCtxSetCurrent"],
                    cuCtxGetCurrent: ["cuCtxGetCurrent"],
                    cuCtxSynchronize: ["cuCtxSynchronize"],
                    cuMemGetInfo: ["cuMemGetInfo_v2", "cuMemGetInfo"],
                    cuMemAlloc: ["cuMemAlloc_v2", "cuMemAlloc"],
                    cuMemFree: ["cuMemFree_v2", "cuMemFree"],
                    cuMemAllocHost: ["cuMemAllocHost_v2", "cuMemAllocHost"],
                    cuMemFreeHost: ["cuMemFreeHost"],
                    cuMemsetD32Async: ["cuMemsetD32Async"],
                    cuMemsetD8Async: ["cuMemsetD8Async"],
                    cuMemcpyHtoDAsync: ["cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync"],
                    cuMemcpyDtoHAsync: ["cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync"],
                    // Generic async copy (works where DtoD legacy stubs do not).
                    cuMemcpyAsync: ["cuMemcpyAsync"],
                    cuStreamCreate: ["cuStreamCreate"],
                    cuStreamDestroy: ["cuStreamDestroy_v2", "cuStreamDestroy"],
                    cuStreamSynchronize: ["cuStreamSynchronize"],
                    cuStreamWaitEvent: ["cuStreamWaitEvent_v2", "cuStreamWaitEvent"],
                    cuStreamQuery: ["cuStreamQuery"],
                    cuEventCreate: ["cuEventCreate"],
                    cuEventDestroy: ["cuEventDestroy_v2", "cuEventDestroy"],
                    cuEventRecord: ["cuEventRecord", "cuEventRecord_v2"],
                    cuEventSynchronize: ["cuEventSynchronize"],
                    cuEventQuery: ["cuEventQuery"],
                    cuEventElapsedTime: ["cuEventElapsedTime"],
                    cuModuleLoadDataEx: ["cuModuleLoadDataEx"],
                    cuModuleUnload: ["cuModuleUnload"],
                    cuModuleGetFunction: ["cuModuleGetFunction"],
                    cuLaunchKernel: ["cuLaunchKernel"],
                    cuFuncGetAttribute: ["cuFuncGetAttribute"],
                    cuFuncSetAttribute: ["cuFuncSetAttribute"],
                    cuOccupancyMaxActiveBlocksPerMultiprocessor: ["cuOccupancyMaxActiveBlocksPerMultiprocessor"],
                    cuOccupancyMaxPotentialBlockSize: ["cuOccupancyMaxPotentialBlockSize"],
                )?;
                // cuInit is deferred-failure-safe: a non-zero result here is
                // surfaced by device enumeration rather than at load time.
                let _ = (api.cuInit)(0);
                Ok(api)
            }
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// Human-readable error name for a `CUresult` (`cuGetErrorName`), with a
/// fallback when the driver is unavailable.
pub fn error_name(code: CUresult) -> String {
    match api() {
        Ok(a) => {
            let mut p: *const CChar = std::ptr::null();
            unsafe {
                if (a.cuGetErrorName)(code, &mut p) == 0 && !p.is_null() {
                    return std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
                }
            }
            format!("CUresult({code})")
        }
        Err(_) => format!("CUresult({code})"),
    }
}

/// Human-readable description for a `CUresult` (`cuGetErrorString`).
pub fn error_string(code: CUresult) -> String {
    match api() {
        Ok(a) => {
            let mut p: *const CChar = std::ptr::null();
            unsafe {
                if (a.cuGetErrorString)(code, &mut p) == 0 && !p.is_null() {
                    return std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned();
                }
            }
            format!("CUDA error {code}")
        }
        Err(_) => format!("CUDA error {code}"),
    }
}

/// Driver version as `(major, minor)` (e.g. `(13, 0)`), if loadable.
pub fn driver_version() -> Option<(i32, i32)> {
    let a = api().ok()?;
    let mut v: CInt = 0;
    unsafe {
        if (a.cuDriverGetVersion)(&mut v) == 0 {
            Some((v / 1000, (v % 1000) / 10))
        } else {
            None
        }
    }
}
