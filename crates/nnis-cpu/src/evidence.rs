//! Suite ids and host identity for CPU host evidence records.
//!
//! The record format and validator live in [`nnis_core::host_evidence`]. This
//! module fixes the suite ids that the `cpu_host_evidence` example runs and
//! that a record must pass, and reports the compiled host identity. Nothing
//! here runs a suite or claims any result.

use nnis_core::adapter_evidence::MAX_EVIDENCE_TEXT_BYTES;
use nnis_core::host_evidence::CpuHostIdentityV1;

/// Suite ids of the CPU host qualification harness, version 1, in run order.
///
/// Every id must pass for
/// [`ReferenceOracleAgreementObserved`](nnis_core::host_evidence::CpuHostEvidenceVerdictV1::ReferenceOracleAgreementObserved).
pub const CPU_HOST_QUALIFICATION_SUITES_V1: [&str; 6] = [
    "cpu.f32_binary",
    "cpu.f32_relu_gather",
    "cpu.f32_serial_accumulation",
    "cpu.f32_fused_projection",
    "cpu.f32_graph",
    "cpu.dsv41_fp4_decode",
];

/// Identity of the host this binary was compiled for.
///
/// Architecture, operating system, endianness and pointer width come from
/// the compilation target. `cpu_model` is operator-supplied text; it is
/// trimmed, control characters are replaced by spaces and it is truncated to
/// the record text limit.
pub fn host_identity(cpu_model: &str) -> CpuHostIdentityV1 {
    CpuHostIdentityV1 {
        arch: std::env::consts::ARCH.to_owned(),
        os: std::env::consts::OS.to_owned(),
        endian: if cfg!(target_endian = "big") {
            "big"
        } else {
            "little"
        }
        .to_owned(),
        pointer_width: (core::mem::size_of::<usize>() * 8) as u32,
        cpu_model: canonical(cpu_model),
    }
}

fn canonical(text: &str) -> String {
    let replaced: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let trimmed = replaced.trim();
    let mut end = trimmed.len().min(MAX_EVIDENCE_TEXT_BYTES);
    while !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    trimmed[..end].trim_end().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matches_compiled_target_and_is_canonical() {
        let identity = host_identity(" Example\tCPU \n");
        assert_eq!(identity.arch, std::env::consts::ARCH);
        assert_eq!(identity.os, std::env::consts::OS);
        assert_eq!(
            identity.endian,
            if cfg!(target_endian = "big") {
                "big"
            } else {
                "little"
            }
        );
        assert_eq!(
            u64::from(identity.pointer_width),
            (usize::MAX as u64).count_ones() as u64
        );
        assert_eq!(identity.cpu_model, "Example CPU");
        assert_eq!(
            host_identity(&"é".repeat(MAX_EVIDENCE_TEXT_BYTES))
                .cpu_model
                .len(),
            MAX_EVIDENCE_TEXT_BYTES
        );
    }
}
