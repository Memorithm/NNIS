//! Raw, dynamically-loaded FFI surface for the CUDA driver API and NVRTC.
//!
//! Design invariants (see ARCHITECTURE.md):
//! * No link-time dependency on `libcuda` / `libnvrtc`: both libraries are
//!   resolved at runtime via `dlopen`, so NNIS builds on machines without a
//!   CUDA toolkit and degrades to a typed "unsupported" state instead of a
//!   link error.
//! * The unsafe surface is confined to this crate. Every foreign function is
//!   declared with the exact signature from `/usr/include/cuda.h`
//!   (CUDA 13.0) or `nvrtc.h`; enum constants are transcribed from the
//!   installed headers, never from memory.
//! * Versioned symbol aliases (`cuMemAlloc_v2`, ...) are resolved through
//!   candidate lists so the crate works across driver generations.

// Native API names are mirrored 1:1 (cuLaunchKernel, nvrtcCreateProgram, ...).
#![allow(non_camel_case_types, non_snake_case)]

pub mod constants;
pub mod driver;
pub mod nvrtc;

/// Raw `CUresult` value.
pub type CUresult = i32;
/// Raw `CUdevice` handle.
pub type CUdevice = i32;
/// Opaque driver handles.
pub type CUcontext = *mut core::ffi::c_void;
pub type CUstream = *mut core::ffi::c_void;
pub type CUevent = *mut core::ffi::c_void;
pub type CUmodule = *mut core::ffi::c_void;
pub type CUfunction = *mut core::ffi::c_void;
pub type CUdeviceptr = usize;
#[allow(non_camel_case_types)]
pub type nvrtcProgram = *mut core::ffi::c_void;

/// 16-byte device UUID (`CUuuid`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct CUuuid(pub [core::ffi::c_uchar; 16]);

impl core::fmt::Debug for CUuuid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (i, b) in self.0.iter().enumerate() {
            if i == 4 || i == 6 || i == 8 || i == 10 {
                write!(f, "-")?;
            }
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Error returned when the native libraries cannot be acquired at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryError {
    /// Which library failed (`"libcuda.so.1"` / `"libnvrtc"`).
    pub library: &'static str,
    /// Candidate sonames that were attempted.
    pub candidates: Vec<String>,
    /// Underlying `dlerror`-style message of the last attempt.
    pub detail: String,
}

impl core::fmt::Display for LibraryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "failed to load {} (tried {}): {}",
            self.library,
            self.candidates.join(", "),
            self.detail
        )
    }
}

impl std::error::Error for LibraryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_debug_format() {
        let u = CUuuid([
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ]);
        assert_eq!(format!("{u:?}"), "12345678-9abc-def0-1122-334455667788");
    }
}
