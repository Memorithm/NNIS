//! WGPU portable session compared with the CPU portable session.
//!
//! Without an adapter each test writes an explicit SKIP to stderr and passes;
//! that pass is not evidence. Software adapters exercise the API path only and
//! are not hardware evidence. No timing is measured. Synthetic fixtures are
//! not model-quality evidence.

use std::io::Write;
use std::sync::OnceLock;

use nnis_core::session::PortableSessionV1;
use nnis_cpu::session::CpuPortableSession;
use nnis_wgpu::{WgpuDevice, WgpuPortableSession};

fn device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| WgpuDevice::discover().unwrap())
        .as_ref()
}

fn adapter_or_skip(test: &str) -> Option<&'static WgpuDevice> {
    let found = device();
    let _ = match found {
        Some(device) => writeln!(
            std::io::stderr(),
            "nnis-wgpu {test}: adapter {:?} class={:?}{}",
            device.adapter().name,
            device.adapter().class,
            if device.adapter().is_hardware() {
                ""
            } else {
                " (not hardware evidence)"
            }
        ),
        None => writeln!(
            std::io::stderr(),
            "SKIP nnis-wgpu {test}: no WGPU adapter available; \
             no WGPU execution was performed and this pass is not evidence"
        ),
    };
    found
}

fn open_pair(device: &'static WgpuDevice) -> (CpuPortableSession, WgpuPortableSession<'static>) {
    (
        CpuPortableSession::tiny().unwrap(),
        WgpuPortableSession::tiny(device).unwrap(),
    )
}

#[test]
fn encode_decode_logits_match_cpu() {
    let Some(device) = adapter_or_skip("encode_decode_logits_match_cpu") else {
        return;
    };
    let (mut cpu, mut wgpu) = open_pair(device);
    let cpu_logits = cpu.encode(&[0, 1, 0]).unwrap().to_vec();
    let wgpu_logits = wgpu.encode(&[0, 1, 0]).unwrap().to_vec();
    assert_eq!(cpu_logits, wgpu_logits);
    assert_eq!(cpu_logits, vec![2.0, 1.0, 0.0, 0.0]);
    let cpu_logits = cpu.decode_one(2).unwrap().to_vec();
    let wgpu_logits = wgpu.decode_one(2).unwrap().to_vec();
    assert_eq!(cpu_logits, wgpu_logits);
    assert_eq!(cpu.position(), wgpu.position());
    assert_eq!(cpu.kv_len(), wgpu.kv_len());
}

#[test]
fn stage_advance_and_truncate_match_cpu() {
    let Some(device) = adapter_or_skip("stage_advance_and_truncate_match_cpu") else {
        return;
    };
    let (mut cpu, mut wgpu) = open_pair(device);
    cpu.encode(&[0, 1]).unwrap();
    wgpu.encode(&[0, 1]).unwrap();
    let cpu_staged = cpu.stage_token(2).unwrap().to_vec();
    let wgpu_staged = wgpu.stage_token(2).unwrap().to_vec();
    assert_eq!(cpu_staged, wgpu_staged);
    assert_eq!(cpu.kv_len(), wgpu.kv_len());
    cpu.advance_kv().unwrap();
    wgpu.advance_kv().unwrap();
    cpu.truncate(1).unwrap();
    wgpu.truncate(1).unwrap();
    assert_eq!(cpu.logits(), wgpu.logits());
    assert_eq!(cpu.position(), 1);
    assert_eq!(wgpu.position(), 1);
    let cpu_logits = cpu.decode_one(3).unwrap().to_vec();
    let wgpu_logits = wgpu.decode_one(3).unwrap().to_vec();
    assert_eq!(cpu_logits, wgpu_logits);
}

#[test]
fn capacity_and_invalid_token_errors_match_cpu() {
    let Some(device) = adapter_or_skip("capacity_and_invalid_token_errors_match_cpu") else {
        return;
    };
    let (mut cpu, mut wgpu) = open_pair(device);
    assert!(cpu.stage_token(99).is_err());
    assert!(wgpu.stage_token(99).is_err());
    cpu.encode(&[0, 1, 2, 3, 0, 1, 2, 3]).unwrap();
    wgpu.encode(&[0, 1, 2, 3, 0, 1, 2, 3]).unwrap();
    assert!(cpu.decode_one(0).is_err());
    assert!(wgpu.decode_one(0).is_err());
}

#[test]
fn reset_clears_both_backends_identically() {
    let Some(device) = adapter_or_skip("reset_clears_both_backends_identically") else {
        return;
    };
    let (mut cpu, mut wgpu) = open_pair(device);
    cpu.encode(&[1, 2]).unwrap();
    wgpu.encode(&[1, 2]).unwrap();
    cpu.reset().unwrap();
    wgpu.reset().unwrap();
    assert_eq!(cpu.position(), 0);
    assert_eq!(wgpu.position(), 0);
    assert!(cpu.logits().is_empty());
    assert!(wgpu.logits().is_empty());
    let cpu_logits = cpu.encode(&[0]).unwrap().to_vec();
    let wgpu_logits = wgpu.encode(&[0]).unwrap().to_vec();
    assert_eq!(cpu_logits, wgpu_logits);
}
