//! NNIS error model.
//!
//! Native failures are never collapsed into booleans: every error carries the
//! failing operation, the native status (name + description when available)
//! and free-form context entries (device ordinal, shapes, kernel name, ...).

use nnis_sys::LibraryError;
use std::fmt;

/// Root error type for all NNIS runtime / compiler operations.
#[derive(Debug)]
pub struct NnisError {
    kind: ErrorKind,
    op: String,
    context: Vec<(String, String)>,
}

#[derive(Debug)]
pub enum ErrorKind {
    /// A CUDA driver call returned a non-success `CUresult`.
    Driver { code: i32 },
    /// An NVRTC call returned a non-success result.
    Nvrtc { code: i32 },
    /// A JIT compilation failed; carries the compiler log.
    Compile { log: String },
    /// A native library could not be loaded at all.
    Library(LibraryError),
    /// The environment cannot support the request (no device, missing
    /// feature, architecture mismatch, ...).
    Unsupported(String),
    /// Caller violated an API contract (size mismatch, zero grid, ...).
    InvalidInput(String),
    /// Host-side I/O failure (JIT cache read/write).
    Io(std::io::Error),
}

impl NnisError {
    pub(crate) fn driver(op: impl Into<String>, code: i32) -> Self {
        NnisError {
            kind: ErrorKind::Driver { code },
            op: op.into(),
            context: Vec::new(),
        }
    }

    pub(crate) fn nvrtc(op: impl Into<String>, code: i32) -> Self {
        NnisError {
            kind: ErrorKind::Nvrtc { code },
            op: op.into(),
            context: Vec::new(),
        }
    }

    pub fn compile(op: impl Into<String>, log: impl Into<String>) -> Self {
        NnisError {
            kind: ErrorKind::Compile { log: log.into() },
            op: op.into(),
            context: Vec::new(),
        }
    }

    pub fn unsupported(what: impl Into<String>) -> Self {
        NnisError {
            kind: ErrorKind::Unsupported(what.into()),
            op: String::new(),
            context: Vec::new(),
        }
    }

    pub fn invalid_input(what: impl Into<String>) -> Self {
        NnisError {
            kind: ErrorKind::InvalidInput(what.into()),
            op: String::new(),
            context: Vec::new(),
        }
    }

    pub(crate) fn io(op: impl Into<String>, err: std::io::Error) -> Self {
        NnisError {
            kind: ErrorKind::Io(err),
            op: op.into(),
            context: Vec::new(),
        }
    }

    /// Attach a context entry, e.g. `.with("shape", "128x768")`.
    pub fn with(mut self, key: &str, value: impl fmt::Display) -> Self {
        self.context.push((key.to_string(), value.to_string()));
        self
    }

    /// The operation that failed (`"cuMemAlloc"`, `"launch softmax_rows"`, ...).
    pub fn op(&self) -> &str {
        &self.op
    }

    /// The raw `CUresult` if this is a driver failure.
    pub fn driver_code(&self) -> Option<i32> {
        match &self.kind {
            ErrorKind::Driver { code } => Some(*code),
            _ => None,
        }
    }

    /// Error classification, for callers that must branch on failure mode.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }
}

impl fmt::Display for NnisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::Driver { code } => write!(
                f,
                "{} failed: {} ({}, code {})",
                self.op,
                nnis_sys::driver::error_string(*code),
                nnis_sys::driver::error_name(*code),
                code
            )?,
            ErrorKind::Nvrtc { code } => write!(
                f,
                "{} failed: {} (code {})",
                self.op,
                nnis_sys::nvrtc::error_string(*code),
                code
            )?,
            ErrorKind::Compile { log } => {
                write!(f, "{} failed to compile CUDA source", self.op)?;
                let trimmed = log.trim();
                if !trimmed.is_empty() {
                    write!(f, "\ncompiler log:\n{trimmed}")?;
                }
            }
            ErrorKind::Library(e) => write!(f, "native library unavailable: {e}")?,
            ErrorKind::Unsupported(w) => write!(f, "unsupported: {w}")?,
            ErrorKind::InvalidInput(w) => write!(f, "invalid input: {w}")?,
            ErrorKind::Io(e) => write!(f, "io error in {}: {e}", self.op)?,
        }
        for (k, v) in &self.context {
            write!(f, "\n  {k}: {v}")?;
        }
        Ok(())
    }
}

impl std::error::Error for NnisError {}

impl From<LibraryError> for NnisError {
    fn from(e: LibraryError) -> Self {
        NnisError {
            kind: ErrorKind::Library(e),
            op: String::new(),
            context: Vec::new(),
        }
    }
}

pub type Result<T> = std::result::Result<T, NnisError>;
