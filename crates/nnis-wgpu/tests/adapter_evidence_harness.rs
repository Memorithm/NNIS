//! Runs the hardware-evidence harness suites on the available adapter and
//! checks the resulting record through the portable validator.
//!
//! Without an adapter the test writes an explicit SKIP to stderr and passes;
//! that pass is not evidence. On a software adapter the verdict must be
//! code-path-only. No timing is measured.

#[path = "../examples/wgpu_adapter_evidence/suites.rs"]
mod suites;

use std::io::Write;

use nnis_core::adapter_evidence::{
    adapter_evidence_from_json, adapter_evidence_to_json, validate_adapter_evidence,
    AdapterClassV1, AdapterEvidenceVerdictV1, EvidenceSourceV1, PortableAdapterEvidenceV1,
    SuiteOutcomeV1,
};
use nnis_wgpu::evidence::{adapter_identity, WGPU_QUALIFICATION_SUITES_V1};
use nnis_wgpu::WgpuDevice;

#[test]
fn harness_suites_pass_and_record_validates() {
    let Some(device) = WgpuDevice::discover().unwrap() else {
        let _ = writeln!(
            std::io::stderr(),
            "SKIP nnis-wgpu harness_suites_pass_and_record_validates: no WGPU adapter available; \
             no WGPU execution was performed and this pass is not evidence"
        );
        return;
    };
    let _ = writeln!(
        std::io::stderr(),
        "nnis-wgpu harness_suites_pass_and_record_validates: adapter {:?} class={:?}",
        device.adapter().name,
        device.adapter().class
    );
    let suites = suites::run_all(&device);
    let ids: Vec<&str> = suites.iter().map(|suite| suite.suite_id.as_str()).collect();
    assert_eq!(ids, WGPU_QUALIFICATION_SUITES_V1);
    for suite in &suites {
        assert_eq!(suite.outcome, SuiteOutcomeV1::Pass, "{suite:?}");
        assert!(suite.checks > 0);
    }
    let record = PortableAdapterEvidenceV1 {
        source: EvidenceSourceV1 {
            git_commit: "0000000000000000000000000000000000000000".to_owned(),
            worktree_clean: true,
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            toolchain: "test-harness".to_owned(),
            target: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        },
        adapter: adapter_identity(device.adapter()),
        suites,
    };
    let parsed = adapter_evidence_from_json(&adapter_evidence_to_json(&record)).unwrap();
    assert_eq!(parsed, record);
    let verdict = validate_adapter_evidence(&parsed, &WGPU_QUALIFICATION_SUITES_V1).unwrap();
    let expected = match record.adapter.class {
        AdapterClassV1::Hardware => AdapterEvidenceVerdictV1::HardwareParityObserved,
        AdapterClassV1::Software => AdapterEvidenceVerdictV1::CodePathOnlySoftwareAdapter,
        AdapterClassV1::Unknown => AdapterEvidenceVerdictV1::UnclassifiedAdapter,
    };
    assert_eq!(verdict, expected);
}
