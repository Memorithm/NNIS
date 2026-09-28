//! Mapping from WGPU adapter reports to portable adapter evidence records.
//!
//! The record format and validator live in
//! [`nnis_core::adapter_evidence`]. This module fixes the suite ids that the
//! `wgpu_adapter_evidence` example runs and that a hardware record must pass,
//! and converts a [`WgpuAdapterReportV1`] into the record's adapter identity.
//! Nothing here runs a suite or claims any result.

use nnis_core::adapter_evidence::{AdapterClassV1, AdapterIdentityV1};

use crate::{WgpuAdapterClassV1, WgpuAdapterReportV1};

/// Suite ids of the WGPU qualification harness, version 1, in run order.
///
/// Every id must pass for
/// [`HardwareParityObserved`](nnis_core::adapter_evidence::AdapterEvidenceVerdictV1::HardwareParityObserved).
pub const WGPU_QUALIFICATION_SUITES_V1: [&str; 7] = [
    "wgpu.add_f32",
    "wgpu.portable_memory",
    "wgpu.f32_kernels",
    "wgpu.f32_graph",
    "wgpu.dsv41_replay_kv",
    "wgpu.dsv41_fp4_decode",
    "wgpu.dsv41_speculative",
];

/// Adapter identity for an evidence record, copied from the driver report.
///
/// Text fields are trimmed and control characters replaced by spaces so the
/// record stays canonical. The class is the harness classification.
pub fn adapter_identity(report: &WgpuAdapterReportV1) -> AdapterIdentityV1 {
    AdapterIdentityV1 {
        backend_family: "wgpu".to_owned(),
        api_backend: canonical(&report.backend),
        name: canonical(&report.name),
        device_type: canonical(&report.device_type),
        vendor_id: report.vendor_id,
        device_id: report.device_id,
        driver: canonical(&report.driver),
        driver_info: canonical(&report.driver_info),
        class: match report.class {
            WgpuAdapterClassV1::Hardware => AdapterClassV1::Hardware,
            WgpuAdapterClassV1::Software => AdapterClassV1::Software,
            WgpuAdapterClassV1::Unknown => AdapterClassV1::Unknown,
        },
    }
}

fn canonical(text: &str) -> String {
    let replaced: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let trimmed = replaced.trim();
    let mut end = trimmed
        .len()
        .min(nnis_core::adapter_evidence::MAX_EVIDENCE_TEXT_BYTES);
    while !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    trimmed[..end].trim_end().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_canonical_and_keeps_class() {
        let report = WgpuAdapterReportV1 {
            name: " llvmpipe (LLVM 19)\n".to_owned(),
            backend: "Vulkan".to_owned(),
            device_type: "Cpu".to_owned(),
            vendor_id: 0x10005,
            device_id: 0,
            driver: "llvmpipe".to_owned(),
            driver_info: "Mesa\t24".to_owned(),
            class: WgpuAdapterClassV1::Software,
        };
        let identity = adapter_identity(&report);
        assert_eq!(identity.name, "llvmpipe (LLVM 19)");
        assert_eq!(identity.driver_info, "Mesa 24");
        assert_eq!(identity.class, AdapterClassV1::Software);
        assert_eq!(identity.backend_family, "wgpu");
        let long = WgpuAdapterReportV1 {
            driver_info: "é".repeat(400),
            ..report
        };
        assert!(
            adapter_identity(&long).driver_info.len()
                <= nnis_core::adapter_evidence::MAX_EVIDENCE_TEXT_BYTES
        );
    }
}
