//! WGPU hardware-evidence harness: run the parity suites on one adapter and
//! write a versioned `nnis.portable.adapter-evidence` JSON record.
//!
//! Intended to be run by a person on the machine whose adapter is being
//! qualified, from a clean checkout of the exact commit:
//!
//! ```text
//! cargo run --release --locked -p nnis-wgpu --example wgpu_adapter_evidence -- \
//!     --commit "$(git rev-parse HEAD)" \
//!     --worktree-clean "$(test -z "$(git status --porcelain)" && echo true || echo false)" \
//!     --toolchain "$(rustc -V)" \
//!     --out adapter-evidence.json
//! ```
//!
//! A reviewer can re-check a record without running anything:
//!
//! ```text
//! cargo run --locked -p nnis-wgpu --example wgpu_adapter_evidence -- --validate adapter-evidence.json
//! ```
//!
//! Without an adapter it prints an explicit SKIP, writes nothing and exits
//! with status 2. It prints the validator verdict. A software adapter always
//! yields `CodePathOnlySoftwareAdapter`; only a hardware adapter with every
//! suite passing on a clean worktree yields `HardwareParityObserved`, which is
//! scoped to that exact adapter, driver, commit and toolchain and is not a
//! performance result. The harness measures no time.

use std::process::ExitCode;

use nnis_core::adapter_evidence::{
    adapter_evidence_from_json, adapter_evidence_to_json, validate_adapter_evidence,
    AdapterEvidenceVerdictV1, EvidenceSourceV1,
};
use nnis_wgpu::evidence::{wgpu_adapter_evidence_record, WGPU_QUALIFICATION_SUITES_V1};
use nnis_wgpu::WgpuDevice;

struct Arguments {
    commit: String,
    worktree_clean: bool,
    toolchain: String,
    out: String,
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut commit = None;
    let mut worktree_clean = None;
    let mut toolchain = None;
    let mut out = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--commit" => commit = Some(value),
            "--worktree-clean" => {
                worktree_clean = Some(match value.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => return Err("--worktree-clean must be true or false".to_owned()),
                })
            }
            "--toolchain" => toolchain = Some(value),
            "--out" => out = Some(value),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    Ok(Arguments {
        commit: commit.ok_or("--commit is required")?,
        worktree_clean: worktree_clean.ok_or("--worktree-clean is required")?,
        toolchain: toolchain.ok_or("--toolchain is required")?,
        out: out.ok_or("--out is required")?,
    })
}

/// Review mode: parse and validate an existing record without running suites.
fn validate_file(path: &str) -> ExitCode {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("wgpu_adapter_evidence: cannot read {path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let record = match adapter_evidence_from_json(text.trim_end()) {
        Ok(record) => record,
        Err(error) => {
            eprintln!("wgpu_adapter_evidence: {path} is not a v1 record: {error}");
            return ExitCode::FAILURE;
        }
    };
    match validate_adapter_evidence(&record, &WGPU_QUALIFICATION_SUITES_V1) {
        Ok(verdict) => {
            println!("verdict: {verdict:?}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("wgpu_adapter_evidence: {path} is invalid: {error}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.len() == 2 && raw[0] == "--validate" {
        return validate_file(&raw[1]);
    }
    let arguments = match parse_arguments() {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("wgpu_adapter_evidence: {error}");
            return ExitCode::from(64);
        }
    };
    let device = match WgpuDevice::discover() {
        Ok(Some(device)) => device,
        Ok(None) => {
            eprintln!(
                "SKIP nnis-wgpu adapter evidence: no WGPU adapter available; \
                 nothing was run and no record was written"
            );
            return ExitCode::from(2);
        }
        Err(error) => {
            eprintln!("wgpu_adapter_evidence: adapter discovery failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "nnis-wgpu adapter evidence: adapter {:?} class={:?}",
        device.adapter().name,
        device.adapter().class
    );
    let record = wgpu_adapter_evidence_record(
        &device,
        EvidenceSourceV1 {
            git_commit: arguments.commit,
            worktree_clean: arguments.worktree_clean,
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            toolchain: arguments.toolchain.trim().to_owned(),
            target: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        },
    );
    for suite in &record.suites {
        eprintln!(
            "  {} {:?} checks={} mismatches={} {}",
            suite.suite_id, suite.outcome, suite.checks, suite.mismatches, suite.detail
        );
    }
    let verdict = match validate_adapter_evidence(&record, &WGPU_QUALIFICATION_SUITES_V1) {
        Ok(verdict) => verdict,
        Err(error) => {
            eprintln!("wgpu_adapter_evidence: record is invalid, not written: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = std::fs::write(&arguments.out, adapter_evidence_to_json(&record) + "\n") {
        eprintln!(
            "wgpu_adapter_evidence: cannot write {}: {error}",
            arguments.out
        );
        return ExitCode::FAILURE;
    }
    eprintln!("verdict: {verdict:?} (record written to {})", arguments.out);
    match verdict {
        AdapterEvidenceVerdictV1::Failed { .. } => ExitCode::FAILURE,
        _ => ExitCode::SUCCESS,
    }
}
