use nnis_sys::nvrtc as sys;

pub type Result<T> = nnis_rt::error::Result<T>;

#[derive(Debug)]
pub enum JitError {
    Nvrtc { op: String, code: i32 },
    Driver { op: String, code: i32 },
    CompileLog(String),
    InvalidSource(String),
}

impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JitError::Nvrtc { op, code } => write!(f, "nvrtc {} failed: {} (code {})", op, sys::error_string(*code), code),
            JitError::Driver { op, code } => write!(f, "driver {} failed (code {})", op, code),
            JitError::CompileLog(log) => write!(f, "compilation failed:\n{}", log),
            JitError::InvalidSource(e) => write!(f, "invalid source: {}", e),
        }
    }
}

impl std::error::Error for JitError {}
