use nnis_sys::nvrtc;
use nnis_rt::error::{NnisError, Result};
use nnis_rt::context::Context;
use std::ffi::CString;
use sha2::{Sha256, Digest};

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub arch: String,
    pub extra_flags: Vec<String>,
}

impl CompileOptions {
    pub fn for_device(ctx: &Context) -> Self {
        let props = ctx.props();
        let arch = format!("sm_{}{}", props.compute_capability.0, props.compute_capability.1);
        CompileOptions {
            arch,
            extra_flags: vec!["--fmad=false".to_string(), "-O3".to_string()],
        }
    }

    fn to_nvrtc_args(&self) -> Vec<String> {
        let mut args = vec![
            format!("-arch={}", self.arch),
            "-I/usr/local/cuda/include".to_string(),
        ];
        args.extend(self.extra_flags.clone());
        args
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProgramCacheKey {
    pub src_hash: [u8; 32],
    pub arch: String,
    pub flags: Vec<String>,
}

impl ProgramCacheKey {
    pub fn from_source(src: &str, opts: &CompileOptions) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(src.as_bytes());
        let src_hash: [u8; 32] = hasher.finalize().into();
        ProgramCacheKey {
            src_hash,
            arch: opts.arch.clone(),
            flags: opts.extra_flags.clone(),
        }
    }
}

pub struct JitProgram {
    pub program: nnis_sys::nvrtcProgram,
    pub source: String,
    pub options: CompileOptions,
    pub log: String,
}

impl JitProgram {
    pub fn compile(src: &str, opts: CompileOptions) -> Result<Self> {
        let api = nvrtc::api().map_err(|e| NnisError::from(e))?;
        let src_c = CString::new(src).map_err(|_| NnisError::invalid_input("source contains interior null"))?;
        let name = CString::new("nn_source").unwrap();
        let mut prog = std::ptr::null_mut();
        let rc = unsafe {
            (api.nvrtcCreateProgram)(
                &mut prog,
                src_c.as_ptr(),
                name.as_ptr(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        if rc != 0 {
            return Err(NnisError::nvrtc("nvrtcCreateProgram", rc));
        }
        let program = prog;

        // compile options
        let args = opts.to_nvrtc_args();
        let c_strings: Vec<CString> = args.iter().map(|s| CString::new(s.as_str()).unwrap()).collect();
        let ptrs: Vec<*const std::os::raw::c_char> = c_strings.iter().map(|s| s.as_ptr()).collect();
        let rc = unsafe {
            (api.nvrtcCompileProgram)(program, ptrs.len() as i32, ptrs.as_ptr())
        };
        // get log
        let mut log_size: usize = 0;
        let _rc_log = unsafe { (api.nvrtcGetProgramLogSize)(program, &mut log_size) };
        let mut log_bytes = vec![0u8; log_size.max(1)];
        let _rc_log_get = unsafe { (api.nvrtcGetProgramLog)(program, log_bytes.as_mut_ptr() as *mut _) };
        let log_str = if log_size > 0 {
            String::from_utf8_lossy(&log_bytes).into_owned()
        } else {
            String::new()
        };
        if rc != 0 {
            // destroy program before returning
            unsafe { (api.nvrtcDestroyProgram)(program) };
            return Err(NnisError::compile("nvrtcCompileProgram", log_str));
        }
        Ok(JitProgram { program, source: src.to_string(), options: opts, log: log_str })
    }

    pub fn get_ptx(&self) -> Result<Vec<u8>> {
        let api = nvrtc::api().map_err(|e| NnisError::from(e))?;
        let mut size: usize = 0;
        let rc = unsafe { (api.nvrtcGetPTXSize)(self.program, &mut size) };
        if rc != 0 { return Err(NnisError::nvrtc("nvrtcGetPTXSize", rc)); }
        let mut buf = vec![0u8; size];
        let rc = unsafe { (api.nvrtcGetPTX)(self.program, buf.as_mut_ptr() as *mut _) };
        if rc != 0 { return Err(NnisError::nvrtc("nvrtcGetPTX", rc)); }
        Ok(buf)
    }

    pub fn get_log(&self) -> &str { &self.log }
}

impl Drop for JitProgram {
    fn drop(&mut self) {
        if let Ok(api) = nvrtc::api() {
            unsafe { (api.nvrtcDestroyProgram)(self.program) };
        }
    }
}
