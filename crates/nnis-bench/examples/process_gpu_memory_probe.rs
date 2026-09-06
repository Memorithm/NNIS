use nnis_rt::{
    Context, Device, ProcessGpuMemoryProbeV1, ProcessGpuMemorySourceV1,
    ProcessGpuMemoryUnavailableReasonV1, Result,
};
use serde::Serialize;
use std::env;

#[derive(Debug)]
struct Arguments {
    device: i32,
}

#[derive(Debug, Serialize)]
struct DeviceIdentity {
    ordinal: i32,
    name: String,
    uuid: Option<String>,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    multiprocessor_count: u32,
    integrated: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum ProbeState {
    Available {
        source: &'static str,
        pid: u32,
        device_uuid: String,
        used_gpu_memory_bytes: u64,
    },
    Unavailable {
        reason: &'static str,
        operation: Option<&'static str>,
        native_status: Option<i32>,
        detail: String,
    },
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    evidence: &'static str,
    measurement: &'static str,
    device: DeviceIdentity,
    probe: ProbeState,
    physical_residency_claimed: bool,
    exact_nnis_allocation_attribution_claimed: bool,
    cu_mem_get_info_fallback_used: bool,
    performance_claimed: bool,
}

fn parse_arguments() -> std::result::Result<Arguments, String> {
    let mut args = env::args().skip(1);
    let mut device = 0_i32;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--device" => {
                device = args
                    .next()
                    .ok_or("--device requires an ordinal")?
                    .parse::<i32>()
                    .map_err(|error| format!("invalid --device: {error}"))?;
            }
            "--help" | "-h" => {
                return Err("usage: process_gpu_memory_probe [--device N]".to_string());
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if device < 0 {
        return Err("--device must be non-negative".to_string());
    }
    Ok(Arguments { device })
}

fn source_name(source: ProcessGpuMemorySourceV1) -> &'static str {
    match source {
        ProcessGpuMemorySourceV1::NvmlComputeRunningProcessesV3 => {
            "nvml_compute_running_processes_v3"
        }
    }
}

fn reason_name(reason: ProcessGpuMemoryUnavailableReasonV1) -> &'static str {
    match reason {
        ProcessGpuMemoryUnavailableReasonV1::NvmlLibraryUnavailable => "nvml_library_unavailable",
        ProcessGpuMemoryUnavailableReasonV1::NvmlInitializationFailed => {
            "nvml_initialization_failed"
        }
        ProcessGpuMemoryUnavailableReasonV1::CudaDeviceUuidUnavailable => {
            "cuda_device_uuid_unavailable"
        }
        ProcessGpuMemoryUnavailableReasonV1::NvmlDeviceLookupFailed => "nvml_device_lookup_failed",
        ProcessGpuMemoryUnavailableReasonV1::QueryNotSupported => "query_not_supported",
        ProcessGpuMemoryUnavailableReasonV1::PermissionDenied => "permission_denied",
        ProcessGpuMemoryUnavailableReasonV1::CurrentProcessNotReported => {
            "current_process_not_reported"
        }
        ProcessGpuMemoryUnavailableReasonV1::UsedGpuMemoryNotAvailable => {
            "used_gpu_memory_not_available"
        }
        ProcessGpuMemoryUnavailableReasonV1::ProcessListUnstable => "process_list_unstable",
        ProcessGpuMemoryUnavailableReasonV1::NativeQueryFailed => "native_query_failed",
    }
}

fn run(arguments: Arguments) -> Result<(Report, bool)> {
    let device = Device::get(arguments.device)?;
    let context = Context::new(&device)?;
    let props = context.props();
    let device_identity = DeviceIdentity {
        ordinal: props.ordinal,
        name: props.name.clone(),
        uuid: props.uuid.map(|uuid| format!("GPU-{uuid:?}")),
        compute_capability_major: props.compute_capability.0,
        compute_capability_minor: props.compute_capability.1,
        multiprocessor_count: props.multiprocessor_count,
        integrated: props.integrated,
    };

    let (probe, available) = match context.process_gpu_memory_probe_v1() {
        ProcessGpuMemoryProbeV1::Available(snapshot) => (
            ProbeState::Available {
                source: source_name(snapshot.source),
                pid: snapshot.pid,
                device_uuid: snapshot.device_uuid,
                used_gpu_memory_bytes: snapshot.used_gpu_memory_bytes,
            },
            true,
        ),
        ProcessGpuMemoryProbeV1::Unavailable(unavailable) => (
            ProbeState::Unavailable {
                reason: reason_name(unavailable.reason),
                operation: unavailable.operation,
                native_status: unavailable.native_status,
                detail: unavailable.detail,
            },
            false,
        ),
    };

    Ok((
        Report {
            schema_version: 1,
            evidence: "nnis.process-gpu-memory-probe",
            measurement: "nvml_compute_process_used_gpu_memory_capability_probe",
            device: device_identity,
            probe,
            physical_residency_claimed: false,
            exact_nnis_allocation_attribution_claimed: false,
            cu_mem_get_info_fallback_used: false,
            performance_claimed: false,
        },
        available,
    ))
}

fn main() {
    let require_available = env::var("NNIS_REQUIRE_NVML_PROCESS_MEMORY").as_deref() == Ok("1");
    let result = parse_arguments()
        .map_err(nnis_rt::NnisError::invalid_input)
        .and_then(run)
        .and_then(|(report, available)| {
            let json = serde_json::to_string_pretty(&report).map_err(|error| {
                nnis_rt::NnisError::invalid_input(format!(
                    "serialize process-memory report: {error}"
                ))
            })?;
            Ok((json, available))
        });

    match result {
        Ok((json, available)) => {
            println!("{json}");
            if require_available && !available {
                eprintln!("NNIS_REQUIRE_NVML_PROCESS_MEMORY=1 but the NVML process-memory probe is unavailable");
                std::process::exit(3);
            }
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}
