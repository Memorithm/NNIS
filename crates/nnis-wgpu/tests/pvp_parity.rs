//! NNIS-PVP2 cross-session parity: CPU reference versus independent WGPU session.
//!
//! Without an adapter this integration test skips explicitly and is not
//! execution evidence. The dedicated CI job installs lavapipe and makes WGPU
//! execution mandatory for this test target.

use std::io::Write;
use std::sync::OnceLock;

use nnis_core::pvp::{PvpCpuU64V1, PvpLayoutAdapterV1, PvpPhysicalWordV1};
use nnis_cpu::pvp::CpuPvpSessionV1;
use nnis_wgpu::pvp::WgpuPvpSessionV1;
use nnis_wgpu::WgpuDevice;

mod pvp_anf_bank_support;
use pvp_anf_bank_support::{AnfBank, BANKS, GEOMETRIES};

#[test]
fn direct_anf_oracle_matches_known_truth_values() {
    assert_eq!(
        AnfBank::frozen(8, 6, "boundary").truth_words(),
        vec![18, 38, 50, 6, 50, 6, 18, 46]
    );
    let constants = AnfBank::frozen(1, 6, "boundary");
    assert_eq!(constants.truth_words(), vec![14]);
    assert_eq!(constants.coefficient_words(), vec![14]);
}

#[test]
fn cpu_session_matches_direct_anf_and_recovers_coefficients() {
    let mut cases = 0;
    for (k, g) in GEOMETRIES {
        for kind in BANKS {
            let bank = AnfBank::frozen(k, g, kind);
            let layout = PvpLayoutAdapterV1::new(k, g).unwrap();
            let source = PvpCpuU64V1::new(layout, bank.coefficient_words()).unwrap();
            let mut cpu = CpuPvpSessionV1::new(&source);
            let stats = cpu.execute_subset_zeta().unwrap();
            assert_eq!(
                cpu.words(),
                bank.truth_words(),
                "CPU K={k} G={g} bank={kind}"
            );
            let pairs = (k / 2) as u128 * u128::from(layout.stages());
            assert_eq!(stats.stages, layout.stages());
            assert_eq!(stats.logical_gate_xor_ops, pairs * g as u128);
            assert_eq!(stats.packed_u64_updates, pairs * g.div_ceil(64) as u128);
            assert_eq!(stats.state_words, source.words().len());
            assert_eq!(stats.state_bytes, source.words().len() * 8);
            assert_eq!(stats.scratch_words, 0);
            assert_eq!(stats.execution_index, 1);
            assert_eq!(cpu.execution_count(), 1);
            let recovered = cpu.execute_subset_zeta().unwrap();
            assert_eq!(recovered.execution_index, 2);
            assert_eq!(cpu.snapshot().unwrap(), source);
            assert_eq!(cpu.execution_count(), 2);
            cases += 1;
        }
    }
    assert_eq!(cases, 24);
    println!("PVP_ANF_BANK_COMPLETE,backend=nnis-cpu,cases=24,truth=exact,round_trip=exact,performance_claim=none");
}

#[test]
fn wgpu_session_matches_direct_anf_and_recovers_coefficients() {
    let Some(device) = adapter_or_skip() else {
        return;
    };
    let mut cases = 0;
    for (k, g) in GEOMETRIES {
        for kind in BANKS {
            let bank = AnfBank::frozen(k, g, kind);
            let layout = PvpLayoutAdapterV1::new(k, g).unwrap();
            let source = PvpCpuU64V1::new(layout, bank.coefficient_words()).unwrap();
            let initial = source.to_wgpu_u32().unwrap();
            assert_eq!(initial.to_cpu_u64().unwrap(), source);
            let mut wgpu = WgpuPvpSessionV1::new(device, &initial).unwrap();
            let stats = wgpu.execute_subset_zeta().unwrap();
            assert_eq!(
                wgpu.snapshot().unwrap().to_cpu_u64().unwrap().words(),
                bank.truth_words(),
                "WGPU K={k} G={g} bank={kind}"
            );
            assert_eq!(stats.stages, layout.stages());
            assert_eq!(
                stats.logical_gate_xor_ops,
                (k / 2) as u128 * u128::from(layout.stages()) * g as u128
            );
            assert_eq!(stats.logical_dispatches, layout.stages());
            assert_eq!(stats.scratch_state_words, 0);
            assert_eq!(stats.execution_index, 1);
            assert_eq!(wgpu.execution_count(), 1);
            let recovered = wgpu.execute_subset_zeta().unwrap();
            assert_eq!(recovered.execution_index, 2);
            assert_eq!(wgpu.snapshot().unwrap().to_cpu_u64().unwrap(), source);
            assert_eq!(wgpu.execution_count(), 2);
            cases += 1;
        }
    }
    assert_eq!(cases, 24);
    println!("PVP_ANF_BANK_COMPLETE,backend=nnis-wgpu,cases=24,truth=exact,round_trip=exact,performance_claim=none");
}

fn device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| WgpuDevice::discover().unwrap())
        .as_ref()
}

fn adapter_or_skip() -> Option<&'static WgpuDevice> {
    let found = device();
    let _ = match found {
        Some(device) => writeln!(
            std::io::stderr(),
            "nnis-wgpu PVP: adapter {:?} class={:?}{}",
            device.adapter().name,
            device.adapter().class,
            if device.adapter().is_hardware() {
                ""
            } else {
                " (correctness only, not hardware performance evidence)"
            }
        ),
        None if std::env::var_os("NNIS_REQUIRE_WGPU_PVP").is_some() => {
            panic!("NNIS PVP WGPU execution is mandatory in this qualification gate")
        }
        None => writeln!(
            std::io::stderr(),
            "SKIP nnis-wgpu PVP: no WGPU adapter available; no PVP WGPU execution occurred"
        ),
    };
    found
}

fn fixture(layout: PvpLayoutAdapterV1) -> PvpCpuU64V1 {
    let row_words = layout.words_per_address(PvpPhysicalWordV1::CpuU64).unwrap();
    let mut words = vec![0_u64; layout.storage_words(PvpPhysicalWordV1::CpuU64).unwrap()];
    for address in 0..layout.addresses() {
        for gate in 0..layout.gates() {
            if ((address * 43 + gate * 19 + (address ^ gate)) % 37) < 18 {
                words[address * row_words + gate / 64] |= 1_u64 << (gate % 64);
            }
        }
    }
    PvpCpuU64V1::new(layout, words).unwrap()
}

#[test]
fn wgpu_session_matches_cpu_session_bit_exactly() {
    let Some(device) = adapter_or_skip() else {
        return;
    };

    for (addresses, gates) in [
        (2, 1),
        (4, 31),
        (8, 65),
        (16, 97),
        (64, 129),
        (128, 257),
        (256, 513),
    ] {
        let layout = PvpLayoutAdapterV1::new(addresses, gates).unwrap();
        let source = fixture(layout);

        let mut cpu = CpuPvpSessionV1::new(&source);
        let cpu_stats = cpu.execute_subset_zeta().unwrap();
        let expected = cpu.snapshot().unwrap();

        let initial_wgpu = source.to_wgpu_u32().unwrap();
        let mut wgpu = WgpuPvpSessionV1::new(device, &initial_wgpu).unwrap();
        let wgpu_stats = wgpu.execute_subset_zeta().unwrap();
        let observed = wgpu.snapshot().unwrap().to_cpu_u64().unwrap();

        assert_eq!(
            observed, expected,
            "CPU/WGPU PVP mismatch for addresses={addresses} gates={gates}"
        );
        assert_eq!(wgpu_stats.stages, cpu_stats.stages);
        assert_eq!(
            wgpu_stats.logical_gate_xor_ops,
            cpu_stats.logical_gate_xor_ops
        );
        assert_eq!(wgpu_stats.logical_dispatches, layout.stages());
        assert_eq!(wgpu_stats.scratch_state_words, 0);
        assert_eq!(wgpu_stats.execution_index, 1);
    }
}

#[test]
fn wgpu_session_is_self_inverse_and_preserves_adapter_projection() {
    let Some(device) = adapter_or_skip() else {
        return;
    };

    let layout = PvpLayoutAdapterV1::new(64, 129).unwrap();
    let source = fixture(layout);
    let initial_wgpu = source.to_wgpu_u32().unwrap();
    let mut wgpu = WgpuPvpSessionV1::new(device, &initial_wgpu).unwrap();

    wgpu.execute_subset_zeta().unwrap();
    wgpu.execute_subset_zeta().unwrap();

    assert_eq!(wgpu.snapshot().unwrap().to_cpu_u64().unwrap(), source);
    assert_eq!(wgpu.execution_count(), 2);
    let record = wgpu.canonical_record().unwrap();
    assert!(record.contains("nnis.pvp-wgpu-session.v1"));
    assert!(record.contains("source_logical=pvp-bitplanes/v1"));
    assert!(record.contains("physical=wgpu-u32"));
}
