//! Process-scoped GPU memory capability probe.
//!
//! This module deliberately distinguishes an NVML process-memory observation
//! from CUDA device-wide free/total telemetry. No `cuMemGetInfo` fallback is
//! permitted: if NVML is absent, unsupported, cannot identify the CUDA device,
//! or cannot report this process, the result is explicitly unavailable.

use crate::Context;
use nnis_sys::nvml;
use nnis_sys::nvml::nvmlProcessInfo_t;
use std::ffi::CString;

pub const NNIS_PROCESS_GPU_MEMORY_PROBE_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessGpuMemorySourceV1 {
    NvmlComputeRunningProcessesV3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessGpuMemoryUnavailableReasonV1 {
    NvmlLibraryUnavailable,
    NvmlInitializationFailed,
    CudaDeviceUuidUnavailable,
    NvmlDeviceLookupFailed,
    QueryNotSupported,
    PermissionDenied,
    CurrentProcessNotReported,
    UsedGpuMemoryNotAvailable,
    ProcessListUnstable,
    NativeQueryFailed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessGpuMemoryUnavailableV1 {
    pub schema_version: u32,
    pub reason: ProcessGpuMemoryUnavailableReasonV1,
    pub operation: Option<&'static str>,
    pub native_status: Option<i32>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessGpuMemorySnapshotV1 {
    pub schema_version: u32,
    pub source: ProcessGpuMemorySourceV1,
    pub pid: u32,
    pub cuda_device_ordinal: i32,
    pub device_uuid: String,
    pub used_gpu_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessGpuMemoryProbeV1 {
    Available(ProcessGpuMemorySnapshotV1),
    Unavailable(ProcessGpuMemoryUnavailableV1),
}

impl ProcessGpuMemoryProbeV1 {
    pub fn snapshot(&self) -> Option<&ProcessGpuMemorySnapshotV1> {
        match self {
            Self::Available(snapshot) => Some(snapshot),
            Self::Unavailable(_) => None,
        }
    }

    pub fn unavailable(&self) -> Option<&ProcessGpuMemoryUnavailableV1> {
        match self {
            Self::Available(_) => None,
            Self::Unavailable(unavailable) => Some(unavailable),
        }
    }
}

fn unavailable(
    reason: ProcessGpuMemoryUnavailableReasonV1,
    operation: Option<&'static str>,
    native_status: Option<i32>,
    detail: impl Into<String>,
) -> ProcessGpuMemoryProbeV1 {
    ProcessGpuMemoryProbeV1::Unavailable(ProcessGpuMemoryUnavailableV1 {
        schema_version: NNIS_PROCESS_GPU_MEMORY_PROBE_VERSION,
        reason,
        operation,
        native_status,
        detail: detail.into(),
    })
}

fn native_unavailable(
    reason: ProcessGpuMemoryUnavailableReasonV1,
    operation: &'static str,
    status: i32,
) -> ProcessGpuMemoryProbeV1 {
    unavailable(
        reason,
        Some(operation),
        Some(status),
        nvml::error_string(status),
    )
}

fn classify_query_status(operation: &'static str, status: i32) -> ProcessGpuMemoryProbeV1 {
    match status {
        nvml::NVML_ERROR_NOT_SUPPORTED => native_unavailable(
            ProcessGpuMemoryUnavailableReasonV1::QueryNotSupported,
            operation,
            status,
        ),
        nvml::NVML_ERROR_NO_PERMISSION => native_unavailable(
            ProcessGpuMemoryUnavailableReasonV1::PermissionDenied,
            operation,
            status,
        ),
        _ => native_unavailable(
            ProcessGpuMemoryUnavailableReasonV1::NativeQueryFailed,
            operation,
            status,
        ),
    }
}

fn nvml_uuid_string(uuid: nnis_sys::CUuuid) -> String {
    format!("GPU-{uuid:?}")
}

fn find_process(entries: &[nvmlProcessInfo_t], pid: u32) -> Option<nvmlProcessInfo_t> {
    entries.iter().copied().find(|entry| entry.pid == pid)
}

fn query_processes(
    api: &nvml::NvmlApi,
    device: nvml::nvmlDevice_t,
) -> std::result::Result<Vec<nvmlProcessInfo_t>, ProcessGpuMemoryProbeV1> {
    const MAX_ATTEMPTS: usize = 4;

    let mut required = 0_u32;
    let first_status = unsafe {
        (api.nvmlDeviceGetComputeRunningProcesses_v3)(
            device,
            &mut required,
            std::ptr::null_mut(),
        )
    };

    if first_status == nvml::NVML_SUCCESS {
        return Ok(Vec::new());
    }
    if first_status != nvml::NVML_ERROR_INSUFFICIENT_SIZE {
        return Err(classify_query_status(
            "nvmlDeviceGetComputeRunningProcesses_v3",
            first_status,
        ));
    }

    for _ in 0..MAX_ATTEMPTS {
        let capacity = required.saturating_add(4).max(1);
        let Ok(capacity_usize) = usize::try_from(capacity) else {
            return Err(unavailable(
                ProcessGpuMemoryUnavailableReasonV1::NativeQueryFailed,
                Some("nvmlDeviceGetComputeRunningProcesses_v3"),
                None,
                "NVML process count exceeds usize capacity",
            ));
        };
        let mut entries = vec![nvmlProcessInfo_t::default(); capacity_usize];
        let mut count = capacity;
        let status = unsafe {
            (api.nvmlDeviceGetComputeRunningProcesses_v3)(
                device,
                &mut count,
                entries.as_mut_ptr(),
            )
        };

        if status == nvml::NVML_SUCCESS {
            let Ok(count_usize) = usize::try_from(count) else {
                return Err(unavailable(
                    ProcessGpuMemoryUnavailableReasonV1::NativeQueryFailed,
                    Some("nvmlDeviceGetComputeRunningProcesses_v3"),
                    None,
                    "NVML returned process count exceeds usize capacity",
                ));
            };
            if count_usize > entries.len() {
                return Err(unavailable(
                    ProcessGpuMemoryUnavailableReasonV1::NativeQueryFailed,
                    Some("nvmlDeviceGetComputeRunningProcesses_v3"),
                    Some(status),
                    "NVML returned more process entries than the supplied buffer",
                ));
            }
            entries.truncate(count_usize);
            return Ok(entries);
        }

        if status == nvml::NVML_ERROR_INSUFFICIENT_SIZE {
            required = count.max(required.saturating_add(1));
            continue;
        }

        return Err(classify_query_status(
            "nvmlDeviceGetComputeRunningProcesses_v3",
            status,
        ));
    }

    Err(unavailable(
        ProcessGpuMemoryUnavailableReasonV1::ProcessListUnstable,
        Some("nvmlDeviceGetComputeRunningProcesses_v3"),
        Some(nvml::NVML_ERROR_INSUFFICIENT_SIZE),
        "NVML process list changed across all bounded query retries",
    ))
}

pub(crate) fn probe(context: &Context) -> ProcessGpuMemoryProbeV1 {
    let Some(cuda_uuid) = context.props().uuid else {
        return unavailable(
            ProcessGpuMemoryUnavailableReasonV1::CudaDeviceUuidUnavailable,
            None,
            None,
            "CUDA did not expose a device UUID for NVML identity binding",
        );
    };
    let device_uuid = nvml_uuid_string(cuda_uuid);

    let api = match nvml::api() {
        Ok(api) => api,
        Err(error) => {
            return unavailable(
                ProcessGpuMemoryUnavailableReasonV1::NvmlLibraryUnavailable,
                None,
                None,
                error.to_string(),
            )
        }
    };

    let init_status = match nvml::init_status() {
        Ok(status) => status,
        Err(error) => {
            return unavailable(
                ProcessGpuMemoryUnavailableReasonV1::NvmlLibraryUnavailable,
                None,
                None,
                error.to_string(),
            )
        }
    };
    if init_status != nvml::NVML_SUCCESS {
        return native_unavailable(
            ProcessGpuMemoryUnavailableReasonV1::NvmlInitializationFailed,
            "nvmlInit_v2",
            init_status,
        );
    }

    let uuid_c_string = match CString::new(device_uuid.as_bytes()) {
        Ok(value) => value,
        Err(_) => {
            return unavailable(
                ProcessGpuMemoryUnavailableReasonV1::NvmlDeviceLookupFailed,
                Some("nvmlDeviceGetHandleByUUID"),
                None,
                "CUDA UUID contains an interior NUL byte",
            )
        }
    };

    let mut device: nvml::nvmlDevice_t = std::ptr::null_mut();
    let lookup_status = unsafe {
        (api.nvmlDeviceGetHandleByUUID)(uuid_c_string.as_ptr(), &mut device)
    };
    if lookup_status != nvml::NVML_SUCCESS {
        let reason = match lookup_status {
            nvml::NVML_ERROR_NOT_SUPPORTED => {
                ProcessGpuMemoryUnavailableReasonV1::QueryNotSupported
            }
            nvml::NVML_ERROR_NO_PERMISSION => {
                ProcessGpuMemoryUnavailableReasonV1::PermissionDenied
            }
            _ => ProcessGpuMemoryUnavailableReasonV1::NvmlDeviceLookupFailed,
        };
        return native_unavailable(reason, "nvmlDeviceGetHandleByUUID", lookup_status);
    }
    if device.is_null() {
        return unavailable(
            ProcessGpuMemoryUnavailableReasonV1::NvmlDeviceLookupFailed,
            Some("nvmlDeviceGetHandleByUUID"),
            Some(lookup_status),
            "NVML returned a null device handle for the CUDA UUID",
        );
    }

    let entries = match query_processes(api, device) {
        Ok(entries) => entries,
        Err(unavailable) => return unavailable,
    };
    let pid = std::process::id();
    let Some(process) = find_process(&entries, pid) else {
        return unavailable(
            ProcessGpuMemoryUnavailableReasonV1::CurrentProcessNotReported,
            Some("nvmlDeviceGetComputeRunningProcesses_v3"),
            Some(nvml::NVML_SUCCESS),
            format!(
                "NVML did not report current PID {pid} among compute processes on {device_uuid}"
            ),
        );
    };
    if process.usedGpuMemory == nvml::NVML_VALUE_NOT_AVAILABLE {
        return unavailable(
            ProcessGpuMemoryUnavailableReasonV1::UsedGpuMemoryNotAvailable,
            Some("nvmlDeviceGetComputeRunningProcesses_v3"),
            Some(nvml::NVML_SUCCESS),
            format!("NVML did not provide usedGpuMemory for current PID {pid}"),
        );
    }

    ProcessGpuMemoryProbeV1::Available(ProcessGpuMemorySnapshotV1 {
        schema_version: NNIS_PROCESS_GPU_MEMORY_PROBE_VERSION,
        source: ProcessGpuMemorySourceV1::NvmlComputeRunningProcessesV3,
        pid,
        cuda_device_ordinal: context.device_ordinal(),
        device_uuid,
        used_gpu_memory_bytes: process.usedGpuMemory,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_uuid_is_bound_to_nvml_gpu_uuid_form() {
        let uuid = nnis_sys::CUuuid([
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55,
            0x66, 0x77, 0x88,
        ]);
        assert_eq!(
            nvml_uuid_string(uuid),
            "GPU-12345678-9abc-def0-1122-334455667788"
        );
    }

    #[test]
    fn current_process_selection_is_pid_exact() {
        let entries = [
            nvmlProcessInfo_t {
                pid: 7,
                usedGpuMemory: 11,
                gpuInstanceId: 0,
                computeInstanceId: 0,
            },
            nvmlProcessInfo_t {
                pid: 42,
                usedGpuMemory: 99,
                gpuInstanceId: 0,
                computeInstanceId: 0,
            },
        ];
        let selected = find_process(&entries, 42).expect("PID should be present");
        assert_eq!(selected.pid, 42);
        assert_eq!(selected.usedGpuMemory, 99);
        assert!(find_process(&entries, 1).is_none());
    }

    #[test]
    fn unsupported_and_permission_statuses_remain_capability_negative() {
        let unsupported =
            classify_query_status("query", nvml::NVML_ERROR_NOT_SUPPORTED);
        assert_eq!(
            unsupported.unavailable().unwrap().reason,
            ProcessGpuMemoryUnavailableReasonV1::QueryNotSupported
        );

        let denied = classify_query_status("query", nvml::NVML_ERROR_NO_PERMISSION);
        assert_eq!(
            denied.unavailable().unwrap().reason,
            ProcessGpuMemoryUnavailableReasonV1::PermissionDenied
        );
    }

    #[test]
    fn no_snapshot_is_synthesized_for_unavailable_probe() {
        let probe = unavailable(
            ProcessGpuMemoryUnavailableReasonV1::NvmlLibraryUnavailable,
            None,
            None,
            "missing",
        );
        assert!(probe.snapshot().is_none());
        assert_eq!(probe.unavailable().unwrap().detail, "missing");
    }
}
