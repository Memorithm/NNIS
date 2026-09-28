//! `nnis evidence`: run or review portable evidence records.
//!
//! - `cpu-host` runs the CPU reference qualification suites against the
//!   integer oracle on this host and emits a `nnis.portable.cpu-host-evidence`
//!   record.
//! - `wgpu-adapter` runs the WGPU parity suites on the discovered adapter and
//!   emits a `nnis.portable.adapter-evidence` record. Without an adapter it
//!   prints an explicit SKIP, writes nothing and exits with status 3.
//! - `validate` parses and validates an existing record of either kind
//!   without running anything.
//!
//! Records are written or printed only when structurally valid. None of these
//! commands touch CUDA or measure time. A software adapter can only yield a
//! code-path-only verdict, and every verdict is scoped to the exact host or
//! adapter, toolchain and commit recorded; a person must still review it.

use std::fs;
use std::process::ExitCode;

use nnis_core::adapter_evidence::{
    adapter_evidence_from_json, adapter_evidence_to_json, validate_adapter_evidence,
    AdapterEvidenceJsonError, AdapterEvidenceVerdictV1, EvidenceSourceV1, SuiteResultV1,
    PORTABLE_ADAPTER_EVIDENCE_KIND,
};
use nnis_core::host_evidence::{
    cpu_host_evidence_from_json, cpu_host_evidence_to_json, validate_cpu_host_evidence,
    CpuHostEvidenceVerdictV1, PORTABLE_CPU_HOST_EVIDENCE_KIND,
};
use nnis_cpu::evidence::{cpu_host_evidence_record, CPU_HOST_QUALIFICATION_SUITES_V1};
use nnis_wgpu::evidence::{wgpu_adapter_evidence_record, WGPU_QUALIFICATION_SUITES_V1};
use nnis_wgpu::WgpuDevice;
use serde::Serialize;

pub(crate) const EVIDENCE_USAGE: &str = "  nnis evidence cpu-host --commit SHA --worktree-clean true|false --toolchain TEXT [--target TEXT] [--cpu-model TEXT] [--out FILE] [--json]\n  nnis evidence wgpu-adapter --commit SHA --worktree-clean true|false --toolchain TEXT [--target TEXT] [--out FILE] [--json]\n  nnis evidence validate --input FILE [--json]";

/// Exit status when no WGPU adapter exists and nothing was run.
pub(crate) const EXIT_NO_ADAPTER: u8 = 3;

/// Version of the `nnis evidence validate --json` envelope.
const EVIDENCE_VALIDATE_JSON_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EvidenceCommand {
    CpuHost(RunArgs),
    WgpuAdapter(RunArgs),
    Validate { input: String, json: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RunArgs {
    pub(crate) commit: String,
    pub(crate) worktree_clean: bool,
    pub(crate) toolchain: String,
    pub(crate) target: String,
    pub(crate) cpu_model: String,
    pub(crate) out: Option<String>,
    pub(crate) json: bool,
}

/// Parse the arguments after `nnis evidence`. `Ok(None)` requests help.
pub(crate) fn parse_evidence_args<I>(arguments: I) -> Result<Option<EvidenceCommand>, String>
where
    I: IntoIterator<Item = String>,
{
    let mut arguments = arguments.into_iter();
    let Some(sub) = arguments.next() else {
        return Err(format!(
            "nnis evidence requires a subcommand\n\n{EVIDENCE_USAGE}"
        ));
    };
    match sub.as_str() {
        "--help" | "-h" => Ok(None),
        "cpu-host" => Ok(parse_run(arguments, true)?.map(EvidenceCommand::CpuHost)),
        "wgpu-adapter" => Ok(parse_run(arguments, false)?.map(EvidenceCommand::WgpuAdapter)),
        "validate" => parse_validate(arguments),
        other => Err(format!(
            "unknown evidence subcommand {other:?}\n\n{EVIDENCE_USAGE}"
        )),
    }
}

fn parse_run<I>(arguments: I, cpu: bool) -> Result<Option<RunArgs>, String>
where
    I: Iterator<Item = String>,
{
    let mut commit = None;
    let mut worktree_clean = None;
    let mut toolchain = None;
    let mut target = None;
    let mut cpu_model = None;
    let mut out = None;
    let mut json = false;
    let mut arguments = arguments;
    while let Some(flag) = arguments.next() {
        let mut value = |name: &str| {
            arguments
                .next()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match flag.as_str() {
            "--help" | "-h" => return Ok(None),
            "--json" => json = true,
            "--commit" => commit = Some(value("--commit")?),
            "--worktree-clean" => {
                worktree_clean = Some(match value("--worktree-clean")?.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => return Err("--worktree-clean must be true or false".to_owned()),
                })
            }
            "--toolchain" => toolchain = Some(value("--toolchain")?),
            "--target" => target = Some(value("--target")?),
            "--cpu-model" if cpu => cpu_model = Some(value("--cpu-model")?),
            "--out" => out = Some(value("--out")?),
            other => {
                return Err(format!(
                    "unknown evidence argument {other:?}\n\n{EVIDENCE_USAGE}"
                ))
            }
        }
    }
    Ok(Some(RunArgs {
        commit: commit.ok_or("--commit is required")?,
        worktree_clean: worktree_clean.ok_or("--worktree-clean is required")?,
        toolchain: toolchain
            .ok_or("--toolchain is required")?
            .trim()
            .to_owned(),
        target: target
            .map(|target| target.trim().to_owned())
            .unwrap_or_else(|| format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)),
        cpu_model: cpu_model.unwrap_or_default(),
        out,
        json,
    }))
}

fn parse_validate<I>(arguments: I) -> Result<Option<EvidenceCommand>, String>
where
    I: Iterator<Item = String>,
{
    let mut input = None;
    let mut json = false;
    let mut arguments = arguments;
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--help" | "-h" => return Ok(None),
            "--json" => json = true,
            "--input" => input = Some(arguments.next().ok_or("--input requires a value")?),
            other => {
                return Err(format!(
                    "unknown evidence validate argument {other:?}\n\n{EVIDENCE_USAGE}"
                ))
            }
        }
    }
    Ok(Some(EvidenceCommand::Validate {
        input: input.ok_or("--input is required")?,
        json,
    }))
}

fn source(arguments: &RunArgs) -> EvidenceSourceV1 {
    EvidenceSourceV1 {
        git_commit: arguments.commit.clone(),
        worktree_clean: arguments.worktree_clean,
        crate_version: env!("CARGO_PKG_VERSION").to_owned(),
        toolchain: arguments.toolchain.clone(),
        target: arguments.target.clone(),
    }
}

/// Human-readable suite lines followed by the verdict.
pub(crate) fn format_run_text(header: &str, suites: &[SuiteResultV1], verdict: &str) -> String {
    let mut text = format!("{header}\n");
    for suite in suites {
        text.push_str(&format!(
            "  {} {} checks={} mismatches={}",
            suite.suite_id,
            suite.outcome.name(),
            suite.checks,
            suite.mismatches
        ));
        if !suite.detail.is_empty() {
            text.push_str(&format!(" ({})", suite.detail));
        }
        text.push('\n');
    }
    text.push_str(&format!("verdict: {verdict}"));
    text
}

fn emit(json_record: String, text: String, arguments: &RunArgs, name: &str) -> Result<(), String> {
    if let Some(path) = &arguments.out {
        fs::write(path, format!("{json_record}\n"))
            .map_err(|error| format!("nnis evidence {name}: cannot write {path}: {error}"))?;
    }
    if arguments.json {
        println!("{json_record}");
    } else {
        println!("{text}");
    }
    Ok(())
}

fn run_cpu_host(arguments: &RunArgs) -> ExitCode {
    let record = cpu_host_evidence_record(source(arguments), &arguments.cpu_model);
    let verdict = match validate_cpu_host_evidence(&record, &CPU_HOST_QUALIFICATION_SUITES_V1) {
        Ok(verdict) => verdict,
        Err(error) => {
            eprintln!("nnis evidence cpu-host: record is invalid, not written: {error}");
            return ExitCode::FAILURE;
        }
    };
    let host = &record.host;
    let header = format!(
        "nnis CPU host evidence: arch={} os={} endian={} pointer_width={}",
        host.arch, host.os, host.endian, host.pointer_width
    );
    let text = format_run_text(&header, &record.suites, &format!("{verdict:?}"));
    if let Err(error) = emit(
        cpu_host_evidence_to_json(&record),
        text,
        arguments,
        "cpu-host",
    ) {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }
    match verdict {
        CpuHostEvidenceVerdictV1::Failed { .. } => ExitCode::FAILURE,
        _ => ExitCode::SUCCESS,
    }
}

fn run_wgpu_adapter(arguments: &RunArgs) -> ExitCode {
    let device = match WgpuDevice::discover() {
        Ok(Some(device)) => device,
        Ok(None) => {
            eprintln!(
                "SKIP nnis-wgpu adapter evidence: no WGPU adapter available; \
                 nothing was run and no record was written"
            );
            return ExitCode::from(EXIT_NO_ADAPTER);
        }
        Err(error) => {
            eprintln!("nnis evidence wgpu-adapter: adapter discovery failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let record = wgpu_adapter_evidence_record(&device, source(arguments));
    let verdict = match validate_adapter_evidence(&record, &WGPU_QUALIFICATION_SUITES_V1) {
        Ok(verdict) => verdict,
        Err(error) => {
            eprintln!("nnis evidence wgpu-adapter: record is invalid, not written: {error}");
            return ExitCode::FAILURE;
        }
    };
    let adapter = &record.adapter;
    let header = format!(
        "nnis WGPU adapter evidence: adapter={:?} api={} device_type={} class={}",
        adapter.name,
        adapter.api_backend,
        adapter.device_type,
        adapter.class.name()
    );
    let text = format_run_text(&header, &record.suites, &format!("{verdict:?}"));
    if let Err(error) = emit(
        adapter_evidence_to_json(&record),
        text,
        arguments,
        "wgpu-adapter",
    ) {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }
    match verdict {
        AdapterEvidenceVerdictV1::Failed { .. } => ExitCode::FAILURE,
        _ => ExitCode::SUCCESS,
    }
}

#[derive(Debug, Serialize)]
struct EvidenceValidateJsonV1 {
    schema_version: u32,
    record_kind: &'static str,
    verdict: String,
}

/// Parse and validate a record of either kind; returns `(kind, verdict)`.
pub(crate) fn validate_record_text(raw: &str) -> Result<(&'static str, String), String> {
    let raw = raw.trim_end();
    match cpu_host_evidence_from_json(raw) {
        Ok(record) => {
            return validate_cpu_host_evidence(&record, &CPU_HOST_QUALIFICATION_SUITES_V1)
                .map(|verdict| (PORTABLE_CPU_HOST_EVIDENCE_KIND, format!("{verdict:?}")))
                .map_err(|error| format!("invalid CPU host evidence record: {error}"))
        }
        Err(AdapterEvidenceJsonError::WrongKind) => {}
        Err(error) => return Err(format!("not a v1 evidence record: {error}")),
    }
    let record = adapter_evidence_from_json(raw).map_err(|error| match error {
        AdapterEvidenceJsonError::WrongKind => format!(
            "unknown record kind; expected {PORTABLE_CPU_HOST_EVIDENCE_KIND} or \
             {PORTABLE_ADAPTER_EVIDENCE_KIND}"
        ),
        error => format!("not a v1 evidence record: {error}"),
    })?;
    validate_adapter_evidence(&record, &WGPU_QUALIFICATION_SUITES_V1)
        .map(|verdict| (PORTABLE_ADAPTER_EVIDENCE_KIND, format!("{verdict:?}")))
        .map_err(|error| format!("invalid adapter evidence record: {error}"))
}

fn run_validate(input: &str, json: bool) -> ExitCode {
    let raw = match fs::read_to_string(input) {
        Ok(raw) => raw,
        Err(error) => {
            eprintln!("nnis evidence validate: cannot read {input}: {error}");
            return ExitCode::FAILURE;
        }
    };
    match validate_record_text(&raw) {
        Ok((record_kind, verdict)) => {
            if json {
                let envelope = EvidenceValidateJsonV1 {
                    schema_version: EVIDENCE_VALIDATE_JSON_SCHEMA_VERSION,
                    record_kind,
                    verdict,
                };
                match serde_json::to_string(&envelope) {
                    Ok(text) => println!("{text}"),
                    Err(error) => {
                        eprintln!("nnis evidence validate: {error}");
                        return ExitCode::FAILURE;
                    }
                }
            } else {
                println!("kind: {record_kind}\nverdict: {verdict}");
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("nnis evidence validate: {input}: {error}");
            ExitCode::FAILURE
        }
    }
}

pub(crate) fn run_evidence(command: &EvidenceCommand) -> ExitCode {
    match command {
        EvidenceCommand::CpuHost(arguments) => run_cpu_host(arguments),
        EvidenceCommand::WgpuAdapter(arguments) => run_wgpu_adapter(arguments),
        EvidenceCommand::Validate { input, json } => run_validate(input, *json),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_core::host_evidence::cpu_host_evidence_to_json;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|item| (*item).to_owned()).collect()
    }

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn parses_run_and_validate_commands_fail_closed() {
        let parsed = parse_evidence_args(args(&[
            "cpu-host",
            "--commit",
            COMMIT,
            "--worktree-clean",
            "true",
            "--toolchain",
            " rustc 1.77.0 ",
            "--cpu-model",
            "Example",
            "--json",
        ]))
        .unwrap()
        .unwrap();
        let EvidenceCommand::CpuHost(run) = parsed else {
            panic!("expected cpu-host");
        };
        assert_eq!(run.commit, COMMIT);
        assert!(run.worktree_clean && run.json);
        assert_eq!(run.toolchain, "rustc 1.77.0");
        assert_eq!(run.cpu_model, "Example");
        assert_eq!(
            run.target,
            format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
        );
        assert_eq!(run.out, None);

        let base = [
            "--commit",
            COMMIT,
            "--worktree-clean",
            "false",
            "--toolchain",
            "t",
        ];
        let mut wgpu = vec!["wgpu-adapter"];
        wgpu.extend(base);
        wgpu.extend(["--target", "x", "--out", "r.json"]);
        let Some(EvidenceCommand::WgpuAdapter(run)) = parse_evidence_args(args(&wgpu)).unwrap()
        else {
            panic!("expected wgpu-adapter");
        };
        assert_eq!(
            (run.target.as_str(), run.out.as_deref()),
            ("x", Some("r.json"))
        );
        assert!(!run.worktree_clean);

        assert_eq!(
            parse_evidence_args(args(&["validate", "--input", "r.json", "--json"])).unwrap(),
            Some(EvidenceCommand::Validate {
                input: "r.json".to_owned(),
                json: true
            })
        );
        assert_eq!(parse_evidence_args(args(&["--help"])).unwrap(), None);
        assert_eq!(
            parse_evidence_args(args(&["cpu-host", "-h"])).unwrap(),
            None
        );

        for bad in [
            &["cpu-host"][..],
            &["cpu-host", "--commit", COMMIT, "--toolchain", "t"],
            &[
                "cpu-host",
                "--commit",
                COMMIT,
                "--worktree-clean",
                "yes",
                "--toolchain",
                "t",
            ],
            &[
                "wgpu-adapter",
                "--commit",
                COMMIT,
                "--worktree-clean",
                "true",
                "--toolchain",
                "t",
                "--cpu-model",
                "x",
            ],
            &["cpu-host", "--commit"],
            &["validate"],
            &["validate", "--input", "a", "--bogus"],
            &["frobnicate"],
            &[],
        ] {
            assert!(parse_evidence_args(args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn validate_detects_kind_and_rejects_foreign_or_malformed_records() {
        let record = cpu_host_evidence_record(
            EvidenceSourceV1 {
                git_commit: COMMIT.to_owned(),
                worktree_clean: true,
                crate_version: "0.1.0".to_owned(),
                toolchain: "test".to_owned(),
                target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            },
            "",
        );
        let text = cpu_host_evidence_to_json(&record);
        assert_eq!(
            validate_record_text(&format!("{text}\n")),
            Ok((
                PORTABLE_CPU_HOST_EVIDENCE_KIND,
                "ReferenceOracleAgreementObserved".to_owned()
            ))
        );
        let dirty = text.replace("\"worktree_clean\":true", "\"worktree_clean\":false");
        assert_eq!(
            validate_record_text(&dirty).unwrap().1,
            "DirtyWorktree".to_owned()
        );
        let foreign = text.replace(PORTABLE_CPU_HOST_EVIDENCE_KIND, "nnis.other");
        assert!(validate_record_text(&foreign)
            .unwrap_err()
            .contains("unknown record kind"));
        assert!(validate_record_text("{").is_err());
        let bad_commit = text.replace(COMMIT, "abc");
        assert!(validate_record_text(&bad_commit)
            .unwrap_err()
            .contains("git_commit"));

        let adapter = r#"{"kind":"nnis.portable.adapter-evidence","schema_version":1,"source":{"git_commit":"0123456789abcdef0123456789abcdef01234567","worktree_clean":true,"crate_version":"0.1.0","toolchain":"t","target":"linux-x86_64"},"adapter":{"backend_family":"wgpu","api_backend":"Vulkan","name":"llvmpipe (LLVM 19)","device_type":"Cpu","vendor_id":65541,"device_id":0,"driver":"llvmpipe","driver_info":"Mesa","class":"software"},"suites":[{"suite_id":"wgpu.add_f32","outcome":"pass","checks":1,"mismatches":0,"tolerance":"bit-exact","detail":""}]}"#;
        let (kind, verdict) = validate_record_text(adapter).unwrap();
        assert_eq!(kind, PORTABLE_ADAPTER_EVIDENCE_KIND);
        assert!(verdict.starts_with("Incomplete"), "{verdict}");
        let hardware_llvmpipe = adapter.replace("\"class\":\"software\"", "\"class\":\"hardware\"");
        assert!(validate_record_text(&hardware_llvmpipe)
            .unwrap_err()
            .contains("hardware class"));
    }

    #[test]
    fn text_output_lists_suites_and_verdict() {
        let suites = vec![SuiteResultV1 {
            suite_id: "cpu.f32_binary".to_owned(),
            outcome: nnis_core::adapter_evidence::SuiteOutcomeV1::Fail,
            checks: 10,
            mismatches: 2,
            tolerance: "bit-exact".to_owned(),
            detail: "2 of 10 checks differ".to_owned(),
        }];
        assert_eq!(
            format_run_text("header", &suites, "Failed"),
            "header\n  cpu.f32_binary fail checks=10 mismatches=2 (2 of 10 checks differ)\nverdict: Failed"
        );
    }
}
