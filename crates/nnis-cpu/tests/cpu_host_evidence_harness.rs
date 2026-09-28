//! Runs the CPU host-evidence harness suites on this host, checks the record
//! through the portable validator, and checks the integer oracle itself on
//! hand-derived binary32 cases. No timing is measured.

use nnis_core::adapter_evidence::{EvidenceSourceV1, SuiteOutcomeV1};
use nnis_core::host_evidence::{
    cpu_host_evidence_from_json, cpu_host_evidence_to_json, validate_cpu_host_evidence,
    CpuHostEvidenceVerdictV1, PortableCpuHostEvidenceV1,
};
use nnis_cpu::evidence::oracle::{exact, product, round, sum};
use nnis_cpu::evidence::{host_identity, run_cpu_host_suites, CPU_HOST_QUALIFICATION_SUITES_V1};

#[test]
fn harness_suites_pass_and_record_validates() {
    let suites = run_cpu_host_suites();
    let ids: Vec<&str> = suites.iter().map(|suite| suite.suite_id.as_str()).collect();
    assert_eq!(ids, CPU_HOST_QUALIFICATION_SUITES_V1);
    for suite in &suites {
        assert_eq!(suite.outcome, SuiteOutcomeV1::Pass, "{suite:?}");
        assert!(suite.checks > 0);
    }
    let record = PortableCpuHostEvidenceV1 {
        source: EvidenceSourceV1 {
            git_commit: "0000000000000000000000000000000000000000".to_owned(),
            worktree_clean: true,
            crate_version: env!("CARGO_PKG_VERSION").to_owned(),
            toolchain: "test-harness".to_owned(),
            target: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
        },
        host: host_identity(""),
        suites,
    };
    let parsed = cpu_host_evidence_from_json(&cpu_host_evidence_to_json(&record)).unwrap();
    assert_eq!(parsed, record);
    assert_eq!(
        validate_cpu_host_evidence(&parsed, &CPU_HOST_QUALIFICATION_SUITES_V1),
        Ok(CpuHostEvidenceVerdictV1::ReferenceOracleAgreementObserved)
    );
}

#[test]
fn integer_oracle_matches_hand_derived_binary32_cases() {
    let add = |a: u32, b: u32| round(sum(exact(a), exact(b)));
    let mul = |a: u32, b: u32| round(product(exact(a), exact(b)));
    let fma = |a: u32, b: u32, c: u32| round(sum(product(exact(a), exact(b)), exact(c)));
    // 1 + 2^-24 is a tie and rounds to even (1); 1 + 3*2^-24 rounds up.
    assert_eq!(add(0x3f80_0000, 0x3380_0000), 0x3f80_0000);
    assert_eq!(add(0x3f80_0001, 0x3380_0000), 0x3f80_0002);
    assert_eq!(add(0x3f80_0000, 0x3440_0000), 0x3f80_0002);
    // Exact cancellation is +0; -0 + -0 is -0; -0 + +0 is +0.
    assert_eq!(add(0x3f80_0000, 0xbf80_0000), 0x0000_0000);
    assert_eq!(add(0x8000_0000, 0x8000_0000), 0x8000_0000);
    assert_eq!(add(0x8000_0000, 0x0000_0000), 0x0000_0000);
    // Subnormal arithmetic is gradual: 2^-149 + 2^-149 = 2^-148.
    assert_eq!(add(0x0000_0001, 0x0000_0001), 0x0000_0002);
    assert_eq!(add(0x007f_ffff, 0x0000_0001), 0x0080_0000);
    // A term far below the other only acts as a sticky bit: 2^24 + 2^-100 and
    // 2^24 - 2^-100 both round to 2^24 (the spacing just below 2^24 is 1).
    // 2^-100 * (1 + 2^-23) below 1 + 2^-23 + 2^-24 (a tie) breaks it upward.
    assert_eq!(add(0x4b80_0000, 0x0d80_0000), 0x4b80_0000);
    assert_eq!(add(0x4b80_0000, 0x8d80_0000), 0x4b80_0000);
    let tie = round(sum(exact(0x3f80_0001), exact(0x3380_0000)));
    assert_eq!(tie, 0x3f80_0002);
    let tie_plus_tiny = round(sum(
        sum(exact(0x3f80_0000), exact(0x3380_0000)),
        exact(0x0d80_0001),
    ));
    assert_eq!(tie_plus_tiny, 0x3f80_0001);
    let tie_minus_tiny = round(sum(
        sum(exact(0x3f80_0000), exact(0x3380_0000)),
        exact(0x8d80_0001),
    ));
    assert_eq!(tie_minus_tiny, 0x3f80_0000);
    // 1.5 * 2^-149 halves to a tie at subnormal granularity: rounds to even.
    assert_eq!(mul(0x3fc0_0000, 0x0000_0001), 0x0000_0002);
    assert_eq!(mul(0x3f00_0000, 0x0000_0001), 0x0000_0000);
    assert_eq!(mul(0x3f00_0000, 0x0000_0003), 0x0000_0002);
    assert_eq!(mul(0xbf80_0000, 0x0000_0000), 0x8000_0000);
    // Overflow is infinity; max finite times one is itself.
    assert_eq!(mul(0x7f7f_ffff, 0x4000_0000), 0x7f80_0000);
    assert_eq!(mul(0x7f7f_ffff, 0x3f80_0000), 0x7f7f_ffff);
    // Single rounding of (1+2^-23)(1-2^-23) - 1 = -2^-46.
    assert_eq!(fma(0x3f80_0001, 0x3f7f_fffe, 0xbf80_0000), 0xa880_0000);
    // The oracle agrees with the host on ordinary normal arithmetic.
    for (a, b) in [(1.25f32, -3.5f32), (0.1, 0.2), (1e-3, 7e4)] {
        assert_eq!(add(a.to_bits(), b.to_bits()), (a + b).to_bits());
        assert_eq!(mul(a.to_bits(), b.to_bits()), (a * b).to_bits());
    }
}
