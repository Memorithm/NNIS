//! Runtime-resolved NVRTC (`libnvrtc`).
//!
//! All signatures transcribed from `nvrtc.h` (CUDA 13.0). Lookup order:
//! `$NNIS_NVRTC_PATH`, then versioned sonames newest-first.

use crate::LibraryError;
use libloading::Library;
use std::sync::OnceLock;

type CInt = core::ffi::c_int;
type CChar = core::ffi::c_char;
type CVoid = core::ffi::c_void;
type CSizeT = usize;

/// Raw `nvrtcResult`.
pub type NvrtcResult = CInt;

/// The subset of the NVRTC API used by NNIS.
pub struct NvrtcApi {
    pub nvrtcVersion: unsafe extern "C" fn(*mut CInt, *mut CInt) -> NvrtcResult,
    pub nvrtcGetErrorString: unsafe extern "C" fn(NvrtcResult) -> *const CChar,
    pub nvrtcCreateProgram: unsafe extern "C" fn(
        *mut crate::nvrtcProgram,
        *const CChar,
        *const CChar,
        CInt,
        *const *const CChar,
        *const *const CChar,
    ) -> NvrtcResult,
    pub nvrtcDestroyProgram: unsafe extern "C" fn(crate::nvrtcProgram) -> NvrtcResult,
    pub nvrtcCompileProgram:
        unsafe extern "C" fn(crate::nvrtcProgram, CInt, *const *const CChar) -> NvrtcResult,
    pub nvrtcGetPTXSize: unsafe extern "C" fn(crate::nvrtcProgram, *mut CSizeT) -> NvrtcResult,
    pub nvrtcGetPTX: unsafe extern "C" fn(crate::nvrtcProgram, *mut CChar) -> NvrtcResult,
    pub nvrtcGetCUBINSize: unsafe extern "C" fn(crate::nvrtcProgram, *mut CSizeT) -> NvrtcResult,
    pub nvrtcGetCUBIN: unsafe extern "C" fn(crate::nvrtcProgram, *mut CChar) -> NvrtcResult,
    pub nvrtcGetProgramLogSize:
        unsafe extern "C" fn(crate::nvrtcProgram, *mut CSizeT) -> NvrtcResult,
    pub nvrtcGetProgramLog: unsafe extern "C" fn(crate::nvrtcProgram, *mut CChar) -> NvrtcResult,
    pub nvrtcAddNameExpression:
        unsafe extern "C" fn(crate::nvrtcProgram, *const CChar) -> NvrtcResult,
    pub nvrtcGetLoweredName:
        unsafe extern "C" fn(crate::nvrtcProgram, *const CChar, *mut *const CChar) -> NvrtcResult,
}

/// Soname candidates for NVRTC, newest first.
pub const NVRTC_CANDIDATES: &[&str] = &[
    "libnvrtc.so.13",
    "libnvrtc.so.12",
    "libnvrtc.so.11.2",
    "libnvrtc.so",
];

unsafe fn resolve<T: Copy>(
    lib: &Library,
    candidates: &'static [&'static str],
) -> Result<T, LibraryError> {
    for sym in candidates {
        if let Ok(f) = lib.get::<T>(sym.as_bytes()) {
            return Ok(*f);
        }
    }
    Err(LibraryError {
        library: "libnvrtc",
        candidates,
        detail: format!("symbol {} not found", candidates[0]),
    })
}

fn open_library() -> Result<Library, LibraryError> {
    let mut detail = String::from("no library could be opened");
    if let Ok(p) = std::env::var("NNIS_NVRTC_PATH") {
        if !p.is_empty() {
            unsafe {
                if let Ok(l) = Library::new(&p) {
                    return Ok(l);
                }
            }
            detail = format!("NNIS_NVRTC_PATH={p} could not be opened");
        }
    }
    for c in NVRTC_CANDIDATES {
        unsafe {
            if let Ok(l) = Library::new(c) {
                return Ok(l);
            }
        }
    }
    Err(LibraryError {
        library: "libnvrtc",
        candidates: NVRTC_CANDIDATES,
        detail,
    })
}

static API: OnceLock<Result<NvrtcApi, LibraryError>> = OnceLock::new();

/// Acquire the process-wide NVRTC API. Idempotent.
pub fn api() -> Result<&'static NvrtcApi, LibraryError> {
    API.get_or_init(|| {
        let lib = open_library()?;
        unsafe {
            Ok(NvrtcApi {
                nvrtcVersion: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcGetErrorString: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcCreateProgram: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcDestroyProgram: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcCompileProgram: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcGetPTXSize: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcGetPTX: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcGetCUBINSize: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcGetCUBIN: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcGetProgramLogSize: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcGetProgramLog: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcAddNameExpression: resolve(&lib, NVRTC_CANDIDATES)?,
                nvrtcGetLoweredName: resolve(&lib, NVRTC_CANDIDATES)?,
            })
        }
    })
    .as_ref()
    .map_err(|e| e.clone())
}

/// Human-readable message for an `nvrtcResult`.
pub fn error_string(code: NvrtcResult) -> String {
    match api() {
        Ok(a) => unsafe {
            let p = (a.nvrtcGetErrorString)(code);
            if p.is_null() {
                format!("NVRTC error {code}")
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        },
        Err(_) => format!("NVRTC error {code}"),
    }
}

/// NVRTC library version as `(major, minor)`, if loadable.
pub fn version() -> Option<(i32, i32)> {
    let a = api().ok()?;
    let (mut maj, mut min): (CInt, CInt) = (0, 0);
    unsafe {
        if (a.nvrtcVersion)(&mut maj, &mut min) == 0 {
            Some((maj, min))
        } else {
            None
        }
    }
}
