//! WGSL elementwise add against the CPU reference, only when an adapter exists.
//!
//! Without an adapter the test logs an explicit SKIP and passes; CI has no GPU,
//! so a pass there is not WGPU execution evidence. A software adapter (CPU
//! device type or known software implementation) runs the API path only and
//! is not hardware evidence. No timing is measured.

use std::io::Write;

use nnis_core::{BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableQueue};
use nnis_cpu::numerical::{CpuF32BinaryOp, CpuF32KernelsV1};
use nnis_cpu::CpuDevice;
use nnis_wgpu::{add_f32_artifact, WgpuDevice};

fn cpu_add(left: &[f32], right: &[f32]) -> Vec<f32> {
    let device = CpuDevice::new().unwrap();
    let queue = device.create_queue().unwrap();
    let bytes = (left.len() * 4) as u64;
    let descriptor = BufferDesc::new(
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        MemoryClass::Host,
    )
    .unwrap();
    let encode =
        |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let mut left_buffer = device.create_buffer(descriptor).unwrap();
    let mut right_buffer = device.create_buffer(descriptor).unwrap();
    let mut output = device.create_buffer(descriptor).unwrap();
    queue
        .write_buffer(&mut left_buffer, 0, &encode(left))
        .unwrap();
    queue
        .write_buffer(&mut right_buffer, 0, &encode(right))
        .unwrap();
    CpuF32KernelsV1::new(1 << 20)
        .unwrap()
        .binary(
            CpuF32BinaryOp::Add,
            &left_buffer,
            &right_buffer,
            &mut output,
        )
        .unwrap();
    queue
        .read_buffer(&output, 0, bytes)
        .unwrap()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Normal-range finite inputs (and signed zeros) whose sums are normal or zero,
/// covering rounding ties, cancellation, and a partial final workgroup.
fn inputs() -> (Vec<f32>, Vec<f32>) {
    let mut left = vec![0.0, -0.0, 1.0, 1.0, 16_777_216.0, 3.0e38, -1.5, 0.1];
    let mut right = vec![-0.0, -0.0, f32::EPSILON / 2.0, 1.0, 1.0, -3.0e38, 1.5, 0.2];
    let mut state = 0x2545_f491_u32;
    while left.len() < 1000 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let a = f32::from_bits(0x3c00_0000 | (state & 0x03ff_ffff));
        let b = f32::from_bits(0xbc00_0000 | (state.rotate_left(11) & 0x03ff_ffff));
        left.push(a * 1.0e3);
        right.push(b * 7.0);
    }
    (left, right)
}

#[test]
fn wgsl_add_matches_cpu_reference_bit_for_bit_when_adapter_exists() {
    let device = match WgpuDevice::discover().unwrap() {
        Some(device) => device,
        None => {
            // Written to the stderr handle directly so libtest output capture
            // does not hide the skip in CI logs.
            let _ = writeln!(
                std::io::stderr(),
                "SKIP nnis-wgpu adapter_add_f32: no WGPU adapter available; \
                 no WGPU execution was performed and this pass is not evidence"
            );
            return;
        }
    };
    let adapter = device.adapter();
    let _ = writeln!(
        std::io::stderr(),
        "nnis-wgpu adapter: name={:?} backend={} device_type={} driver={:?} class={:?}{}",
        adapter.name,
        adapter.backend,
        adapter.device_type,
        adapter.driver,
        adapter.class,
        if adapter.is_hardware() {
            ""
        } else {
            " (not hardware evidence)"
        }
    );
    let _ = writeln!(
        std::io::stderr(),
        "nnis-wgpu capabilities: {:?}",
        device.capabilities()
    );

    let (left, right) = inputs();
    let artifact = add_f32_artifact(left.len() as u64).unwrap();
    device
        .bind(&artifact, artifact.artifact_fingerprint())
        .unwrap();
    let expected = cpu_add(&left, &right);
    assert!(expected
        .iter()
        .all(|value| *value == 0.0 || value.is_normal()));
    let actual = device.add_f32(&left, &right).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (index, (gpu, cpu)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            gpu.to_bits(),
            cpu.to_bits(),
            "index {index}: {} + {} gave {gpu} on WGPU, {cpu} on CPU",
            left[index],
            right[index]
        );
    }
    assert!(device.add_f32(&left, &right[1..]).is_err());
    assert!(device.add_f32(&[f32::NAN], &[1.0]).is_err());
    assert!(device.add_f32(&[], &[]).is_err());
}
