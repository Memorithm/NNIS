//! CPU host-evidence harness: run the CPU reference qualification suites on
//! this host and write a versioned `nnis.portable.cpu-host-evidence` JSON
//! record.
//!
//! Intended to be run by a person on the host being qualified (for example an
//! ARM64 machine), from a clean checkout of the exact commit:
//!
//! ```text
//! cargo run --release --locked -p nnis-cpu --example cpu_host_evidence -- \
//!     --commit "$(git rev-parse HEAD)" \
//!     --worktree-clean "$(test -z "$(git status --porcelain)" && echo true || echo false)" \
//!     --toolchain "$(rustc -V)" \
//!     --target "$(rustc -vV | sed -n 's/^host: //p')" \
//!     --cpu-model "<CPU model, or empty>" \
//!     --out cpu-host-evidence.json
//! ```
//!
//! A reviewer can re-check a record without running anything:
//!
//! ```text
//! cargo run --locked -p nnis-cpu --example cpu_host_evidence -- --validate cpu-host-evidence.json
//! ```
//!
//! Every suite compares the reference bit for bit with an integer-only
//! oracle. The harness prints the validator verdict; only a clean worktree
//! with every suite passing yields `ReferenceOracleAgreementObserved`, which
//! is scoped to that exact host, toolchain and commit and is not a
//! performance result. The harness measures no time and touches no GPU.

use std::process::ExitCode;

use nnis_core::adapter_evidence::EvidenceSourceV1;
use nnis_core::host_evidence::{
    cpu_host_evidence_from_json, cpu_host_evidence_to_json, validate_cpu_host_evidence,
    CpuHostEvidenceVerdictV1,
};
use nnis_cpu::evidence::{cpu_host_evidence_record, CPU_HOST_QUALIFICATION_SUITES_V1};

struct Arguments {
    commit: String,
    worktree_clean: bool,
    toolchain: String,
    target: String,
    cpu_model: String,
    out: String,
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut commit = None;
    let mut worktree_clean = None;
    let mut toolchain = None;
    let mut target = None;
    let mut cpu_model = None;
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
            "--target" => target = Some(value),
            "--cpu-model" => cpu_model = Some(value),
            "--out" => out = Some(value),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    Ok(Arguments {
        commit: commit.ok_or("--commit is required")?,
        worktree_clean: worktree_clean.ok_or("--worktree-clean is required")?,
        toolchain: toolchain.ok_or("--toolchain is required")?,
        target: target
            .unwrap_or_else(|| format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)),
        cpu_model: cpu_model.unwrap_or_default(),
        out: out.ok_or("--out is required")?,
    })
}

/// Review mode: parse and validate an existing record without running suites.
fn validate_file(path: &str) -> ExitCode {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("cpu_host_evidence: cannot read {path}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let record = match cpu_host_evidence_from_json(text.trim_end()) {
        Ok(record) => record,
        Err(error) => {
            eprintln!("cpu_host_evidence: {path} is not a v1 record: {error}");
            return ExitCode::FAILURE;
        }
    };
    match validate_cpu_host_evidence(&record, &CPU_HOST_QUALIFICATION_SUITES_V1) {
        Ok(verdict) => {
            println!("verdict: {verdict:?}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("cpu_host_evidence: {path} is invalid: {error}");
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
            eprintln!("cpu_host_evidence: {error}");
            return ExitCode::from(64);
        }
    };
    let record = cpu_host_evidence_record(
        EvidenceSourceV1 {
            git_commit: arguments.commit,
            worktree_clean: arguments.worktree_clean,
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            toolchain: arguments.toolchain.trim().to_owned(),
            target: arguments.target.trim().to_owned(),
        },
        &arguments.cpu_model,
    );
    let host = &record.host;
    eprintln!(
        "nnis-cpu host evidence: arch={} os={} endian={} pointer_width={}",
        host.arch, host.os, host.endian, host.pointer_width
    );
    for suite in &record.suites {
        eprintln!(
            "  {} {:?} checks={} mismatches={} {}",
            suite.suite_id, suite.outcome, suite.checks, suite.mismatches, suite.detail
        );
    }
    let verdict = match validate_cpu_host_evidence(&record, &CPU_HOST_QUALIFICATION_SUITES_V1) {
        Ok(verdict) => verdict,
        Err(error) => {
            eprintln!("cpu_host_evidence: record is invalid, not written: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = std::fs::write(&arguments.out, cpu_host_evidence_to_json(&record) + "\n") {
        eprintln!("cpu_host_evidence: cannot write {}: {error}", arguments.out);
        return ExitCode::FAILURE;
    }
    eprintln!("verdict: {verdict:?} (record written to {})", arguments.out);
    match verdict {
        CpuHostEvidenceVerdictV1::Failed { .. } => ExitCode::FAILURE,
        _ => ExitCode::SUCCESS,
    }
}
