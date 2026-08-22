//! Runtime-resolved NVRTC (`libnvrtc`).
//!
//! All signatures transcribed from `nvrtc.h` (CUDA 13.0). Lookup order:
//! `$NNIS_NVRTC_PATH`, then versioned sonames newest-first.

use crate::LibraryError;
use libloading::Library;
use std::sync::OnceLock;

type CInt = core::ffi::c_int;
type CChar = core::ffi::c_char;
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
    pub nvrtcDestroyProgram: unsafe extern "C" fn(*mut crate::nvrtcProgram) -> NvrtcResult,
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

    // NVRTC function pointers become invalid after `dlclose`. The API is
    // cached process-wide, so it must own the loaded library for that lifetime.
    _library: Library,
}

/// Soname candidates for NVRTC, newest first.
pub const NVRTC_CANDIDATES: &[&str] = &[
    "libnvrtc.so.13",
    "libnvrtc.so.12",
    "libnvrtc.so.11.2",
    "libnvrtc.so",
];

unsafe fn resolve<T: Copy>(lib: &Library, candidates: &[&str]) -> Result<T, LibraryError> {
    let mut detail = String::from("no symbol candidate was provided");
    for sym in candidates {
        match lib.get::<T>(sym.as_bytes()) {
            Ok(f) => return Ok(*f),
            Err(error) => detail = error.to_string(),
        }
    }
    Err(LibraryError {
        library: "libnvrtc",
        candidates: candidates
            .iter()
            .map(|candidate| candidate.to_string())
            .collect(),
        detail,
    })
}

fn open_library() -> Result<Library, LibraryError> {
    let mut candidates = Vec::new();
    let mut detail = String::from("no library could be opened");
    if let Ok(p) = std::env::var("NNIS_NVRTC_PATH") {
        if !p.is_empty() {
            candidates.push(p.clone());
            unsafe {
                match Library::new(&p) {
                    Ok(library) => return Ok(library),
                    Err(error) => detail = error.to_string(),
                }
            }
        }
    }
    for candidate in NVRTC_CANDIDATES {
        candidates.push((*candidate).to_string());
        unsafe {
            match Library::new(candidate) {
                Ok(library) => return Ok(library),
                Err(error) => detail = error.to_string(),
            }
        }
    }
    Err(LibraryError {
        library: "libnvrtc",
        candidates,
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
                nvrtcVersion: resolve(&lib, &["nvrtcVersion"])?,
                nvrtcGetErrorString: resolve(&lib, &["nvrtcGetErrorString"])?,
                nvrtcCreateProgram: resolve(&lib, &["nvrtcCreateProgram"])?,
                nvrtcDestroyProgram: resolve(&lib, &["nvrtcDestroyProgram"])?,
                nvrtcCompileProgram: resolve(&lib, &["nvrtcCompileProgram"])?,
                nvrtcGetPTXSize: resolve(&lib, &["nvrtcGetPTXSize"])?,
                nvrtcGetPTX: resolve(&lib, &["nvrtcGetPTX"])?,
                nvrtcGetCUBINSize: resolve(&lib, &["nvrtcGetCUBINSize"])?,
                nvrtcGetCUBIN: resolve(&lib, &["nvrtcGetCUBIN"])?,
                nvrtcGetProgramLogSize: resolve(&lib, &["nvrtcGetProgramLogSize"])?,
                nvrtcGetProgramLog: resolve(&lib, &["nvrtcGetProgramLog"])?,
                nvrtcAddNameExpression: resolve(&lib, &["nvrtcAddNameExpression"])?,
                nvrtcGetLoweredName: resolve(&lib, &["nvrtcGetLoweredName"])?,
                _library: lib,
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

#[cfg(test)]
mod tests {
    #[test]
    fn nvrtc_is_loadable_when_gpu_is_required() {
        if std::env::var("NNIS_REQUIRE_GPU").as_deref() != Ok("1") {
            return;
        }
        let version = super::version().expect("NNIS_REQUIRE_GPU=1 requires a usable NVRTC");
        assert!(version.0 >= 11, "unexpected NVRTC version {version:?}");
    }
}
