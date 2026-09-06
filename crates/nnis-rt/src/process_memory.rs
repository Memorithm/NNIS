//! Process-scoped GPU-memory telemetry backed by NVML.
//!
//! This module intentionally does not call NVML's process memory field
//! "physical residency". The contract reports the process-scoped
//! `usedGpuMemory` value returned by NVML for the current PID and CUDA device.

use crate::device::Device;
use nnis_sys::nvml::{self, NvmlApi, NvmlDevice, NvmlProcessInfo, NvmlReturn};
use nnis_sys::LibraryError;
use std::ffi::CString;
use std::fmt;

/// Version of [`NvmlProcessMemorySnapshotV1`].
pub const NNIS_NVML_PROCESS_MEMORY_SNAPSHOT_VERSION: u32 = 1;

const INITIAL_PROCESS_CAPACITY: u32 = 16;
const MAX_PROCESS_CAPACITY: u32 = 65_536;
const MAX_QUERY_ATTEMPTS: usize = 8;

/// NVML process-scoped GPU-memory observation for one PID/device pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvmlProcessMemorySnapshotV1 {
    pub schema_version: u32,
    pub pid: u32,
    pub device_ordinal: i32,
    pub device_uuid: String,
    pub used_gpu_memory_bytes: u64,
}

/// Fail-closed errors from the NVML process-memory probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessGpuMemoryError {
    Library(LibraryError),
    MissingCudaDeviceUuid {
        device_ordinal: i32,
    },
    InvalidCudaDeviceUuid {
        device_uuid: String,
    },
    Nvml {
        operation: &'static str,
        code: NvmlReturn,
        message: String,
    },
    ProcessListTooLarge {
        requested_records: u32,
    },
    ProcessListUnstable,
    ProcessNotFound {
        pid: u32,
        device_uuid: String,
    },
    DuplicateProcessRecords {
        pid: u32,
        count: usize,
    },
    UsedGpuMemoryUnavailable {
        pid: u32,
        device_uuid: String,
    },
}

impl fmt::Display for ProcessGpuMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Library(error) => write!(f, "NVML library unavailable: {error}"),
            Self::MissingCudaDeviceUuid { device_ordinal } => write!(
                f,
                "CUDA device {device_ordinal} did not expose a UUID required for NVML correlation"
            ),
            Self::InvalidCudaDeviceUuid { device_uuid } => {
                write!(f, "CUDA device UUID cannot be represented for NVML: {device_uuid}")
            }
            Self::Nvml {
                operation,
                code,
                message,
            } => write!(f, "{operation} failed: {message} (NVML code {code})"),
            Self::ProcessListTooLarge { requested_records } => write!(
                f,
                "NVML requested {requested_records} process records, above the NNIS safety bound"
            ),
            Self::ProcessListUnstable => write!(
                f,
                "NVML compute-process list did not stabilize within the bounded retry budget"
            ),
            Self::ProcessNotFound { pid, device_uuid } => write!(
                f,
                "NVML did not report PID {pid} as a compute process on {device_uuid}"
            ),
            Self::DuplicateProcessRecords { pid, count } => write!(
                f,
                "NVML returned {count} compute-process records for PID {pid}; attribution is ambiguous"
            ),
            Self::UsedGpuMemoryUnavailable { pid, device_uuid } => write!(
                f,
                "NVML reported usedGpuMemory unavailable for PID {pid} on {device_uuid}"
            ),
        }
    }
}

impl std::error::Error for ProcessGpuMemoryError {}

impl From<LibraryError> for ProcessGpuMemoryError {
    fn from(error: LibraryError) -> Self {
        Self::Library(error)
    }
}

struct NvmlSession {
    api: &'static NvmlApi,
}

impl NvmlSession {
    fn new() -> Result<Self, ProcessGpuMemoryError> {
        let api = nvml::api()?;
        let code = unsafe { (api.nvmlInit_v2)() };
        if code != nvml::NVML_SUCCESS {
            return Err(nvml_error("nvmlInit_v2", code));
        }
        Ok(Self { api })
    }
}

impl Drop for NvmlSession {
    fn drop(&mut self) {
        // NVML initialization is reference counted. Shutdown failure cannot
        // invalidate an already captured observation and is deliberately not
        // promoted over the query result from Drop.
        let _ = unsafe { (self.api.nvmlShutdown)() };
    }
}

fn nvml_error(operation: &'static str, code: NvmlReturn) -> ProcessGpuMemoryError {
    ProcessGpuMemoryError::Nvml {
        operation,
        code,
        message: nvml::error_string(code),
    }
}

fn cuda_uuid_for_nvml(device: &Device) -> Result<String, ProcessGpuMemoryError> {
    let uuid = device
        .uuid()
        .ok_or(ProcessGpuMemoryError::MissingCudaDeviceUuid {
            device_ordinal: device.ordinal(),
        })?;
    Ok(format!("GPU-{uuid:?}"))
}

fn nvml_device_by_uuid(
    api: &NvmlApi,
    device_uuid: &str,
) -> Result<NvmlDevice, ProcessGpuMemoryError> {
    let uuid =
        CString::new(device_uuid).map_err(|_| ProcessGpuMemoryError::InvalidCudaDeviceUuid {
            device_uuid: device_uuid.to_string(),
        })?;
    let mut device = std::ptr::null_mut();
    let code = unsafe { (api.nvmlDeviceGetHandleByUUID)(uuid.as_ptr(), &mut device) };
    if code != nvml::NVML_SUCCESS {
        return Err(nvml_error("nvmlDeviceGetHandleByUUID", code));
    }
    if device.is_null() {
        return Err(ProcessGpuMemoryError::Nvml {
            operation: "nvmlDeviceGetHandleByUUID",
            code: nvml::NVML_SUCCESS,
            message: "NVML returned a null device handle on success".to_string(),
        });
    }
    Ok(device)
}

fn query_compute_processes(
    api: &NvmlApi,
    device: NvmlDevice,
) -> Result<Vec<NvmlProcessInfo>, ProcessGpuMemoryError> {
    let mut capacity = INITIAL_PROCESS_CAPACITY;
    for _ in 0..MAX_QUERY_ATTEMPTS {
        let mut records = vec![NvmlProcessInfo::default(); capacity as usize];
        let mut count = capacity;
        let code = unsafe {
            (api.nvmlDeviceGetComputeRunningProcesses_v3)(device, &mut count, records.as_mut_ptr())
        };
        if code == nvml::NVML_SUCCESS {
            if count > capacity {
                return Err(ProcessGpuMemoryError::ProcessListUnstable);
            }
            records.truncate(count as usize);
            return Ok(records);
        }
        if code != nvml::NVML_ERROR_INSUFFICIENT_SIZE {
            return Err(nvml_error("nvmlDeviceGetComputeRunningProcesses_v3", code));
        }

        let requested = if count > capacity {
            count
        } else {
            capacity
                .checked_mul(2)
                .ok_or(ProcessGpuMemoryError::ProcessListTooLarge {
                    requested_records: u32::MAX,
                })?
        };
        if requested > MAX_PROCESS_CAPACITY {
            return Err(ProcessGpuMemoryError::ProcessListTooLarge {
                requested_records: requested,
            });
        }
        capacity = requested;
    }
    Err(ProcessGpuMemoryError::ProcessListUnstable)
}

fn select_process_memory(
    records: &[NvmlProcessInfo],
    pid: u32,
    device_uuid: &str,
) -> Result<u64, ProcessGpuMemoryError> {
    let matching: Vec<&NvmlProcessInfo> =
        records.iter().filter(|record| record.pid == pid).collect();
    if matching.is_empty() {
        return Err(ProcessGpuMemoryError::ProcessNotFound {
            pid,
            device_uuid: device_uuid.to_string(),
        });
    }
    if matching.len() != 1 {
        return Err(ProcessGpuMemoryError::DuplicateProcessRecords {
            pid,
            count: matching.len(),
        });
    }
    let used = matching[0].used_gpu_memory;
    if used == nvml::NVML_VALUE_NOT_AVAILABLE {
        return Err(ProcessGpuMemoryError::UsedGpuMemoryUnavailable {
            pid,
            device_uuid: device_uuid.to_string(),
        });
    }
    Ok(used)
}

/// Query NVML's process-scoped `usedGpuMemory` for the current process on
/// `device`.
///
/// This is not a physical-page-residency measurement and does not identify
/// which NNIS allocation or CUDA subsystem owns the reported bytes.
pub fn current_process_gpu_memory(
    device: &Device,
) -> Result<NvmlProcessMemorySnapshotV1, ProcessGpuMemoryError> {
    let device_uuid = cuda_uuid_for_nvml(device)?;
    let session = NvmlSession::new()?;
    let nvml_device = nvml_device_by_uuid(session.api, &device_uuid)?;
    let records = query_compute_processes(session.api, nvml_device)?;
    let pid = std::process::id();
    let used_gpu_memory_bytes = select_process_memory(&records, pid, &device_uuid)?;
    Ok(NvmlProcessMemorySnapshotV1 {
        schema_version: NNIS_NVML_PROCESS_MEMORY_SNAPSHOT_VERSION,
        pid,
        device_ordinal: device.ordinal(),
        device_uuid,
        used_gpu_memory_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn process(pid: u32, used_gpu_memory: u64) -> NvmlProcessInfo {
        NvmlProcessInfo {
            pid,
            used_gpu_memory,
            gpu_instance_id: u32::MAX,
            compute_instance_id: u32::MAX,
        }
    }

    #[test]
    fn process_selection_returns_exact_nvml_value() {
        let records = [process(11, 100), process(22, 987_654_321)];
        assert_eq!(
            select_process_memory(&records, 22, "GPU-test").unwrap(),
            987_654_321
        );
    }

    #[test]
    fn missing_process_fails_closed() {
        let error = select_process_memory(&[process(11, 100)], 22, "GPU-test").unwrap_err();
        assert!(matches!(
            error,
            ProcessGpuMemoryError::ProcessNotFound { pid: 22, .. }
        ));
    }

    #[test]
    fn unavailable_memory_fails_closed() {
        let error = select_process_memory(
            &[process(22, nvml::NVML_VALUE_NOT_AVAILABLE)],
            22,
            "GPU-test",
        )
        .unwrap_err();
        assert!(matches!(
            error,
            ProcessGpuMemoryError::UsedGpuMemoryUnavailable { pid: 22, .. }
        ));
    }

    #[test]
    fn duplicate_pid_records_fail_closed() {
        let error = select_process_memory(&[process(22, 100), process(22, 200)], 22, "GPU-test")
            .unwrap_err();
        assert_eq!(
            error,
            ProcessGpuMemoryError::DuplicateProcessRecords { pid: 22, count: 2 }
        );
    }
}
