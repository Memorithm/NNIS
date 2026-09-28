//! Versioned record format and fail-closed validator for CPU host evidence.
//!
//! A [`PortableCpuHostEvidenceV1`] records one run of the CPU reference
//! qualification suites on one identified host (architecture, operating
//! system, endianness, pointer width) at one exact source commit. Each suite
//! compares the `nnis-cpu` reference with an independent integer oracle, so a
//! passing record says the reference reproduced the oracle bit for bit on that
//! host. It is produced by the `nnis-cpu` `cpu_host_evidence` example and
//! reviewed before anything is cited.
//!
//! Source identity, suite results and the JSON error types are shared with
//! [`crate::adapter_evidence`]. Like adapter records, host records have no
//! timing, throughput, memory or energy fields.
//!
//! [`validate_cpu_host_evidence`] rejects malformed records with an error and
//! otherwise returns a verdict with the same precedence as adapter records:
//! failed suites, then missing or skipped required suites, then a dirty
//! worktree, and only then
//! [`CpuHostEvidenceVerdictV1::ReferenceOracleAgreementObserved`]. That
//! verdict is scoped to the exact host, toolchain and commit recorded. It is
//! not a performance result, does not extend to other hosts and does not
//! qualify any GPU backend.

use crate::adapter_evidence::{
    optional_text, parse_root, required_text, source_from, source_json, suites_from, suites_json,
    summarize_suites, text, validate_source, AdapterEvidenceError, AdapterEvidenceJsonError,
    EvidenceSourceV1, Object, SuiteResultV1, SuiteSummary,
};
use crate::json::{self, Json};

/// `kind` tag of a CPU host evidence record.
pub const PORTABLE_CPU_HOST_EVIDENCE_KIND: &str = "nnis.portable.cpu-host-evidence";

/// Schema version written and accepted.
pub const PORTABLE_CPU_HOST_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Identity of the host that ran the CPU reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuHostIdentityV1 {
    /// Target architecture as compiled (for example `x86_64` or `aarch64`).
    pub arch: String,
    /// Target operating system as compiled (for example `linux` or `macos`).
    pub os: String,
    /// `little` or `big`.
    pub endian: String,
    /// Pointer width in bits: 16, 32 or 64.
    pub pointer_width: u32,
    /// Operator-supplied CPU model; may be empty when unknown.
    pub cpu_model: String,
}

/// One CPU host evidence record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableCpuHostEvidenceV1 {
    pub source: EvidenceSourceV1,
    pub host: CpuHostIdentityV1,
    pub suites: Vec<SuiteResultV1>,
}

/// Verdict of a structurally valid CPU host record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuHostEvidenceVerdictV1 {
    /// Clean worktree and every required suite passed with zero mismatches.
    /// Scoped to the exact recorded host, toolchain and commit.
    ReferenceOracleAgreementObserved,
    /// Suites passed on a dirty worktree; the commit does not identify the code.
    DirtyWorktree,
    /// These required suites were missing or skipped.
    Incomplete { suite_ids: Vec<String> },
    /// These suites failed.
    Failed { suite_ids: Vec<String> },
}

/// Validate a record against the suites that must pass, returning its verdict.
pub fn validate_cpu_host_evidence(
    record: &PortableCpuHostEvidenceV1,
    required_suites: &[&str],
) -> Result<CpuHostEvidenceVerdictV1, AdapterEvidenceError> {
    validate_source(&record.source)?;
    let host = &record.host;
    required_text("arch", &host.arch)?;
    required_text("os", &host.os)?;
    optional_text("cpu_model", &host.cpu_model)?;
    if host.endian != "little" && host.endian != "big" {
        return Err(AdapterEvidenceError::InvalidHostField { field: "endian" });
    }
    if !matches!(host.pointer_width, 16 | 32 | 64) {
        return Err(AdapterEvidenceError::InvalidHostField {
            field: "pointer_width",
        });
    }
    if !record.source.target.contains(host.arch.as_str()) {
        return Err(AdapterEvidenceError::HostTargetMismatch);
    }
    Ok(match summarize_suites(&record.suites, required_suites)? {
        SuiteSummary::Failed(suite_ids) => CpuHostEvidenceVerdictV1::Failed { suite_ids },
        SuiteSummary::Incomplete(suite_ids) => CpuHostEvidenceVerdictV1::Incomplete { suite_ids },
        SuiteSummary::AllRequiredPassed if !record.source.worktree_clean => {
            CpuHostEvidenceVerdictV1::DirtyWorktree
        }
        SuiteSummary::AllRequiredPassed => {
            CpuHostEvidenceVerdictV1::ReferenceOracleAgreementObserved
        }
    })
}

/// Encode a record as compact JSON. Encoding does not validate.
pub fn cpu_host_evidence_to_json(record: &PortableCpuHostEvidenceV1) -> String {
    let host = &record.host;
    let member = |name: &str, value: Json| (name.to_owned(), value);
    let value = Json::Object(vec![
        member("kind", text(PORTABLE_CPU_HOST_EVIDENCE_KIND)),
        member(
            "schema_version",
            json::unsigned(u64::from(PORTABLE_CPU_HOST_EVIDENCE_SCHEMA_VERSION)),
        ),
        member("source", source_json(&record.source)),
        member(
            "host",
            Json::Object(vec![
                member("arch", text(&host.arch)),
                member("os", text(&host.os)),
                member("endian", text(&host.endian)),
                member(
                    "pointer_width",
                    json::unsigned(u64::from(host.pointer_width)),
                ),
                member("cpu_model", text(&host.cpu_model)),
            ]),
        ),
        member("suites", suites_json(&record.suites)),
    ]);
    let mut output = String::new();
    json::write(&value, &mut output);
    output
}

/// Parse a record strictly. Parsing does not validate; call
/// [`validate_cpu_host_evidence`] on the result.
pub fn cpu_host_evidence_from_json(
    input: &str,
) -> Result<PortableCpuHostEvidenceV1, AdapterEvidenceJsonError> {
    let mut root = parse_root(
        input,
        PORTABLE_CPU_HOST_EVIDENCE_KIND,
        PORTABLE_CPU_HOST_EVIDENCE_SCHEMA_VERSION,
    )?;
    let source = source_from(root.object("source")?)?;
    let host = host_from(root.object("host")?)?;
    let suites = suites_from(root.array("suites")?)?;
    root.finish()?;
    Ok(PortableCpuHostEvidenceV1 {
        source,
        host,
        suites,
    })
}

fn host_from(mut object: Object) -> Result<CpuHostIdentityV1, AdapterEvidenceJsonError> {
    let host = CpuHostIdentityV1 {
        arch: object.string("arch")?,
        os: object.string("os")?,
        endian: object.string("endian")?,
        pointer_width: object.u32("pointer_width")?,
        cpu_model: object.string("cpu_model")?,
    };
    object.finish()?;
    Ok(host)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter_evidence::SuiteOutcomeV1;

    const REQUIRED: [&str; 2] = ["cpu.f32_binary", "cpu.f32_graph"];

    fn suite(id: &str, outcome: SuiteOutcomeV1, checks: u64, mismatches: u64) -> SuiteResultV1 {
        SuiteResultV1 {
            suite_id: id.to_owned(),
            outcome,
            checks,
            mismatches,
            tolerance: "bit-exact".to_owned(),
            detail: if outcome == SuiteOutcomeV1::Pass {
                String::new()
            } else {
                "see log".to_owned()
            },
        }
    }

    fn record() -> PortableCpuHostEvidenceV1 {
        PortableCpuHostEvidenceV1 {
            source: EvidenceSourceV1 {
                git_commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
                worktree_clean: true,
                crate_version: "0.1.0".to_owned(),
                toolchain: "rustc 1.77.0 (aedd173a2 2024-03-17)".to_owned(),
                target: "aarch64-linux".to_owned(),
            },
            host: CpuHostIdentityV1 {
                arch: "aarch64".to_owned(),
                os: "linux".to_owned(),
                endian: "little".to_owned(),
                pointer_width: 64,
                cpu_model: String::new(),
            },
            suites: vec![
                suite("cpu.f32_binary", SuiteOutcomeV1::Pass, 2000, 0),
                suite("cpu.f32_graph", SuiteOutcomeV1::Pass, 300, 0),
            ],
        }
    }

    #[test]
    fn verdicts_follow_outcomes_and_worktree() {
        use CpuHostEvidenceVerdictV1 as V;
        assert_eq!(
            validate_cpu_host_evidence(&record(), &REQUIRED),
            Ok(V::ReferenceOracleAgreementObserved)
        );
        let mut dirty = record();
        dirty.source.worktree_clean = false;
        assert_eq!(
            validate_cpu_host_evidence(&dirty, &REQUIRED),
            Ok(V::DirtyWorktree)
        );
        let mut skipped = record();
        skipped.suites[1] = suite("cpu.f32_graph", SuiteOutcomeV1::Skip, 0, 0);
        assert_eq!(
            validate_cpu_host_evidence(&skipped, &REQUIRED),
            Ok(V::Incomplete {
                suite_ids: vec!["cpu.f32_graph".to_owned()]
            })
        );
        let mut failed = record();
        failed.suites[0] = suite("cpu.f32_binary", SuiteOutcomeV1::Fail, 2000, 1);
        failed.source.worktree_clean = false;
        assert_eq!(
            validate_cpu_host_evidence(&failed, &REQUIRED),
            Ok(V::Failed {
                suite_ids: vec!["cpu.f32_binary".to_owned()]
            })
        );
    }

    #[test]
    fn malformed_records_are_rejected() {
        use AdapterEvidenceError as E;
        let check = |edit: &dyn Fn(&mut PortableCpuHostEvidenceV1), expected: E| {
            let mut value = record();
            edit(&mut value);
            assert_eq!(validate_cpu_host_evidence(&value, &REQUIRED), Err(expected));
        };
        check(&|r| r.source.git_commit = "abc".into(), E::InvalidGitCommit);
        check(&|r| r.host.arch.clear(), E::EmptyField { field: "arch" });
        check(
            &|r| r.host.cpu_model = "x\ty".into(),
            E::NonCanonicalField { field: "cpu_model" },
        );
        check(
            &|r| r.host.endian = "middle".into(),
            E::InvalidHostField { field: "endian" },
        );
        check(
            &|r| r.host.pointer_width = 48,
            E::InvalidHostField {
                field: "pointer_width",
            },
        );
        check(
            &|r| r.source.target = "x86_64-unknown-linux-gnu".into(),
            E::HostTargetMismatch,
        );
        check(&|r| r.suites.clear(), E::NoSuites);
        check(
            &|r| r.suites[0] = suite("cpu.f32_binary", SuiteOutcomeV1::Pass, 10, 1),
            E::PassWithMismatches {
                suite_id: "cpu.f32_binary".into(),
            },
        );
    }

    #[test]
    fn json_round_trips_and_rejects_drift() {
        use AdapterEvidenceJsonError as E;
        let value = record();
        let encoded = cpu_host_evidence_to_json(&value);
        assert_eq!(cpu_host_evidence_from_json(&encoded), Ok(value));
        assert_eq!(
            cpu_host_evidence_from_json(&encoded.replace("cpu-host-evidence", "adapter-evidence")),
            Err(E::WrongKind)
        );
        assert_eq!(
            cpu_host_evidence_from_json(
                &encoded.replace("\"schema_version\":1", "\"schema_version\":2")
            ),
            Err(E::UnsupportedSchemaVersion)
        );
        assert_eq!(
            cpu_host_evidence_from_json(
                &encoded.replace("\"cpu_model\":\"\"", "\"cpu_model\":\"\",\"mhz\":1")
            ),
            Err(E::UnknownField {
                field: "mhz".into()
            })
        );
        assert_eq!(
            cpu_host_evidence_from_json(
                &encoded.replace("\"pointer_width\":64", "\"pointer_width\":-1")
            ),
            Err(E::InvalidInteger {
                field: "pointer_width"
            })
        );
        assert_eq!(
            crate::adapter_evidence::adapter_evidence_from_json(&encoded),
            Err(E::WrongKind)
        );
    }
}
