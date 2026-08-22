//! NVRTC program ownership and compiled-code extraction.

use nnis_rt::context::Context;
use nnis_rt::error::{NnisError, Result};
use nnis_sys::nvrtc;
use sha2::{Digest, Sha256};
use std::ffi::{CStr, CString};

/// Output representation requested from NVRTC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeKind {
    /// Null-terminated PTX text suitable for `cuModuleLoadDataEx`.
    Ptx,
    /// Architecture-specific ELF/CUBIN bytes.
    Cubin,
}

/// Deterministic NVRTC configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompileOptions {
    architecture: String,
    extra_options: Vec<String>,
}

impl CompileOptions {
    /// Create options for an explicit NVRTC architecture such as `sm_110` or
    /// `compute_90`.
    pub fn new(architecture: impl Into<String>) -> Self {
        Self {
            architecture: architecture.into(),
            extra_options: vec!["--std=c++17".to_string()],
        }
    }

    /// Target the exact compute capability of `ctx`.
    pub fn for_device(ctx: &Context) -> Self {
        Self::new(ctx.props().sm_arch())
    }

    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    pub fn extra_options(&self) -> &[String] {
        &self.extra_options
    }

    /// Append one native NVRTC option. Options are kept in insertion order;
    /// that order is included in the deterministic cache key.
    pub fn with_option(mut self, option: impl Into<String>) -> Self {
        self.extra_options.push(option.into());
        self
    }

    pub(crate) fn nvrtc_args(&self) -> Vec<String> {
        let mut args = Vec::with_capacity(self.extra_options.len() + 1);
        args.push(format!("--gpu-architecture={}", self.architecture));
        args.extend(self.extra_options.iter().cloned());
        args
    }
}

/// Stable content key for one source/options/output tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgramCacheKey([u8; 32]);

impl ProgramCacheKey {
    pub fn from_source(source: &str, options: &CompileOptions, kind: CodeKind) -> Self {
        let mut hasher = Sha256::new();
        hash_field(&mut hasher, b"nnis-jit-cache-v1");
        hash_field(&mut hasher, source.as_bytes());
        hash_field(&mut hasher, options.architecture.as_bytes());
        for option in &options.extra_options {
            hash_field(&mut hasher, option.as_bytes());
        }
        hash_field(
            &mut hasher,
            match kind {
                CodeKind::Ptx => b"ptx",
                CodeKind::Cubin => b"cubin",
            },
        );
        Self(hasher.finalize().into())
    }

    pub fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

fn hash_field(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

/// An owned NVRTC program. The native handle is destroyed exactly once.
pub struct JitProgram {
    program: nnis_sys::nvrtcProgram,
    options: CompileOptions,
    log: String,
}

impl JitProgram {
    pub fn compile(source: &str, options: CompileOptions) -> Result<Self> {
        Self::compile_named(source, "nnis_runtime.cu", options)
    }

    pub fn compile_named(source: &str, name: &str, options: CompileOptions) -> Result<Self> {
        let source = CString::new(source)
            .map_err(|_| NnisError::invalid_input("CUDA source contains an interior NUL"))?;
        let name = CString::new(name)
            .map_err(|_| NnisError::invalid_input("CUDA source name contains an interior NUL"))?;
        let args = options.nvrtc_args();
        let c_args = args
            .iter()
            .map(|arg| {
                CString::new(arg.as_str()).map_err(|_| {
                    NnisError::invalid_input(format!(
                        "NVRTC option contains an interior NUL: {arg:?}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let arg_ptrs = c_args.iter().map(|arg| arg.as_ptr()).collect::<Vec<_>>();

        let api = nvrtc::api()?;
        let mut program = std::ptr::null_mut();
        // SAFETY: all C strings and out-pointers are valid for this call; no
        // headers are supplied, so both header arrays are null.
        let create_result = unsafe {
            (api.nvrtcCreateProgram)(
                &mut program,
                source.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if create_result != 0 {
            return Err(NnisError::nvrtc("nvrtcCreateProgram", create_result));
        }

        // From this point, every error path is covered by `ProgramGuard`.
        let mut guard = ProgramGuard { program };
        // SAFETY: the program is live and the pointer array references the
        // `CString`s above for the complete duration of the call.
        let compile_result = unsafe {
            (api.nvrtcCompileProgram)(
                guard.program,
                arg_ptrs.len() as i32,
                if arg_ptrs.is_empty() {
                    std::ptr::null()
                } else {
                    arg_ptrs.as_ptr()
                },
            )
        };
        let log = program_log(api, guard.program)?;
        if compile_result != 0 {
            return Err(NnisError::compile("nvrtcCompileProgram", log)
                .with("nvrtc_code", compile_result)
                .with("nvrtc_status", nvrtc::error_string(compile_result))
                .with("architecture", options.architecture()));
        }

        let program = guard.program;
        guard.program = std::ptr::null_mut();
        Ok(Self {
            program,
            options,
            log,
        })
    }

    pub fn options(&self) -> &CompileOptions {
        &self.options
    }

    /// Compiler diagnostics, including warnings from successful compilation.
    pub fn log(&self) -> &str {
        &self.log
    }

    pub fn ptx(&self) -> Result<Vec<u8>> {
        let api = nvrtc::api()?;
        get_output(
            self.program,
            "nvrtcGetPTXSize",
            "nvrtcGetPTX",
            api.nvrtcGetPTXSize,
            api.nvrtcGetPTX,
        )
    }

    pub fn cubin(&self) -> Result<Vec<u8>> {
        let api = nvrtc::api()?;
        get_output(
            self.program,
            "nvrtcGetCUBINSize",
            "nvrtcGetCUBIN",
            api.nvrtcGetCUBINSize,
            api.nvrtcGetCUBIN,
        )
    }

    /// Compatibility alias for the initial public API.
    pub fn get_ptx(&self) -> Result<Vec<u8>> {
        self.ptx()
    }

    /// Compatibility alias for the initial public API.
    pub fn get_log(&self) -> &str {
        self.log()
    }
}

impl Drop for JitProgram {
    fn drop(&mut self) {
        if self.program.is_null() {
            return;
        }
        if let Ok(api) = nvrtc::api() {
            // SAFETY: NNIS exclusively owns this live program handle. NVRTC
            // nulls the handle on successful destruction.
            unsafe {
                let _ = (api.nvrtcDestroyProgram)(&mut self.program);
            }
        }
    }
}

struct ProgramGuard {
    program: nnis_sys::nvrtcProgram,
}

impl Drop for ProgramGuard {
    fn drop(&mut self) {
        if self.program.is_null() {
            return;
        }
        if let Ok(api) = nvrtc::api() {
            // SAFETY: the guard owns the program on all early-return paths.
            unsafe {
                let _ = (api.nvrtcDestroyProgram)(&mut self.program);
            }
        }
    }
}

fn program_log(api: &nvrtc::NvrtcApi, program: nnis_sys::nvrtcProgram) -> Result<String> {
    let mut size = 0usize;
    // SAFETY: `program` is live and `size` is a valid out-pointer.
    let result = unsafe { (api.nvrtcGetProgramLogSize)(program, &mut size) };
    if result != 0 {
        return Err(NnisError::nvrtc("nvrtcGetProgramLogSize", result));
    }
    if size == 0 {
        return Ok(String::new());
    }
    let mut bytes = vec![0u8; size];
    // SAFETY: NVRTC reported the required size and the buffer is writable.
    let result = unsafe { (api.nvrtcGetProgramLog)(program, bytes.as_mut_ptr().cast()) };
    if result != 0 {
        return Err(NnisError::nvrtc("nvrtcGetProgramLog", result));
    }
    Ok(CStr::from_bytes_until_nul(&bytes)
        .map(|log| log.to_string_lossy().into_owned())
        .unwrap_or_else(|_| String::from_utf8_lossy(&bytes).into_owned()))
}

fn get_output(
    program: nnis_sys::nvrtcProgram,
    size_operation: &'static str,
    get_operation: &'static str,
    get_size: unsafe extern "C" fn(nnis_sys::nvrtcProgram, *mut usize) -> i32,
    get_bytes: unsafe extern "C" fn(nnis_sys::nvrtcProgram, *mut core::ffi::c_char) -> i32,
) -> Result<Vec<u8>> {
    let mut size = 0usize;
    // SAFETY: program is owned by the caller and `size` is a valid out-pointer.
    let result = unsafe { get_size(program, &mut size) };
    if result != 0 {
        return Err(NnisError::nvrtc(size_operation, result));
    }
    if size == 0 {
        return Err(NnisError::unsupported(format!(
            "{get_operation} produced an empty image"
        )));
    }
    let mut bytes = vec![0u8; size];
    // SAFETY: the buffer has exactly the size reported by NVRTC.
    let result = unsafe { get_bytes(program, bytes.as_mut_ptr().cast()) };
    if result != 0 {
        return Err(NnisError::nvrtc(get_operation, result));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_is_deterministic_and_complete() {
        let options = CompileOptions::new("sm_110").with_option("--use_fast_math");
        let first = ProgramCacheKey::from_source("kernel", &options, CodeKind::Ptx);
        let second = ProgramCacheKey::from_source("kernel", &options, CodeKind::Ptx);
        assert_eq!(first, second);
        assert_ne!(
            first,
            ProgramCacheKey::from_source("kernel ", &options, CodeKind::Ptx)
        );
        assert_ne!(
            first,
            ProgramCacheKey::from_source("kernel", &options, CodeKind::Cubin)
        );
        assert_eq!(first.hex().len(), 64);
    }
}
