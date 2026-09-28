//! Portable buffer/queue/fence contract on WGPU compared with the CPU reference.
//!
//! Without an adapter each test writes an explicit SKIP to stderr and passes;
//! that pass is not evidence. Software adapters exercise the API path only
//! and are not hardware evidence. No timing is measured.

use std::io::Write;
use std::sync::OnceLock;

use nnis_core::{
    BufferDesc, BufferUsages, FenceStatus, MemoryClass, PortableDevice, PortableError,
    PortableFence, PortableQueue,
};
use nnis_cpu::CpuDevice;
use nnis_wgpu::WgpuDevice;

/// Devices shared by all tests in this binary and never dropped, so the
/// driver never sees concurrent device creation or teardown.
fn devices() -> Option<&'static (WgpuDevice, WgpuDevice)> {
    static DEVICES: OnceLock<Option<(WgpuDevice, WgpuDevice)>> = OnceLock::new();
    DEVICES
        .get_or_init(|| {
            let first = WgpuDevice::discover().unwrap()?;
            let second = WgpuDevice::discover().unwrap()?;
            Some((first, second))
        })
        .as_ref()
}

fn adapter_or_skip(test: &str) -> Option<&'static WgpuDevice> {
    match devices() {
        Some((device, _)) => {
            let adapter = device.adapter();
            let _ = writeln!(
                std::io::stderr(),
                "nnis-wgpu {test}: adapter {:?} class={:?}{}",
                adapter.name,
                adapter.class,
                if adapter.is_hardware() {
                    ""
                } else {
                    " (not hardware evidence)"
                }
            );
            Some(device)
        }
        None => {
            let _ = writeln!(
                std::io::stderr(),
                "SKIP nnis-wgpu {test}: no WGPU adapter available; \
                 no WGPU execution was performed and this pass is not evidence"
            );
            None
        }
    }
}

/// Normalize backend-specific error text; variants and range fields must match.
fn outcome<T: std::fmt::Debug>(result: Result<T, PortableError>) -> String {
    match result {
        Ok(value) => format!("Ok({value:?})"),
        Err(PortableError::Unsupported(_)) => "Err(Unsupported)".to_string(),
        Err(PortableError::Backend(message)) => format!("Err(Backend({message}))"),
        Err(error) => format!("Err({error:?})"),
    }
}

fn usages_all() -> BufferUsages {
    BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
}

/// One scripted sequence through the generic portable traits only.
fn script<D: PortableDevice>(device: &D) -> Vec<String> {
    let queue = device.create_queue().unwrap();
    let mut log = Vec::new();
    let odd = BufferDesc::new(13, usages_all(), MemoryClass::Shared).unwrap();
    let mut buffer = device.create_buffer(odd).unwrap();
    log.push(outcome(queue.read_buffer(&buffer, 0, 13)));

    let fence = queue
        .write_buffer(&mut buffer, 5, &[10, 20, 30, 40])
        .unwrap();
    fence.wait().unwrap();
    log.push(format!("{:?}", fence.status().unwrap()));
    log.push(outcome(queue.read_buffer(&buffer, 0, 13)));

    let pattern: Vec<u8> = (100..113).collect();
    queue
        .write_buffer(&mut buffer, 0, &pattern)
        .unwrap()
        .wait()
        .unwrap();
    log.push(outcome(queue.read_buffer(&buffer, 3, 7)));
    queue.write_buffer(&mut buffer, 11, &[1, 2]).unwrap();
    queue.write_buffer(&mut buffer, 1, &[7]).unwrap();
    queue
        .write_buffer(&mut buffer, 6, &[8, 9, 10, 11, 12])
        .unwrap();
    log.push(outcome(queue.read_buffer(&buffer, 0, 13)));

    log.push(outcome(
        queue.write_buffer(&mut buffer, 13, &[]).map(|_| ()),
    ));
    log.push(outcome(queue.read_buffer(&buffer, 13, 0)));
    log.push(outcome(
        queue.write_buffer(&mut buffer, 13, &[1]).map(|_| ()),
    ));
    log.push(outcome(
        queue.write_buffer(&mut buffer, 10, &[1; 4]).map(|_| ()),
    ));
    log.push(outcome(queue.read_buffer(&buffer, 12, 2)));
    log.push(outcome(queue.read_buffer(&buffer, u64::MAX, 2)));
    // Failed operations left the payload unchanged.
    log.push(outcome(queue.read_buffer(&buffer, 0, 13)));

    let write_only = BufferDesc::new(8, BufferUsages::COPY_DST, MemoryClass::Shared).unwrap();
    let mut write_only = device.create_buffer(write_only).unwrap();
    log.push(outcome(
        queue.write_buffer(&mut write_only, 0, &[1; 8]).map(|_| ()),
    ));
    log.push(outcome(queue.read_buffer(&write_only, 0, 8)));
    let read_only = BufferDesc::new(8, BufferUsages::COPY_SRC, MemoryClass::Shared).unwrap();
    let mut read_only = device.create_buffer(read_only).unwrap();
    log.push(outcome(
        queue.write_buffer(&mut read_only, 0, &[1; 8]).map(|_| ()),
    ));
    log.push(outcome(queue.read_buffer(&read_only, 0, 8)));

    let zero = BufferDesc {
        size_bytes: 0,
        usages: usages_all(),
        memory: MemoryClass::Shared,
    };
    log.push(outcome(device.create_buffer(zero).map(|_| ())));
    let no_usage = BufferDesc {
        size_bytes: 4,
        usages: BufferUsages::EMPTY,
        memory: MemoryClass::Shared,
    };
    log.push(outcome(device.create_buffer(no_usage).map(|_| ())));
    log
}

#[test]
fn portable_queue_contract_matches_cpu_reference() {
    let Some(wgpu) = adapter_or_skip("portable_queue_contract") else {
        return;
    };
    let cpu = CpuDevice::new().unwrap();
    let expected = script(&cpu);
    let actual = script(wgpu);
    assert_eq!(actual, expected);
    // Sanity-check that the script observed real data, not only errors.
    assert_eq!(
        expected[4],
        "Ok([100, 7, 102, 103, 104, 105, 8, 9, 10, 11, 12, 1, 2])"
    );
}

#[test]
fn wgpu_specific_admission_rules_fail_closed() {
    let Some(device) = adapter_or_skip("admission_rules") else {
        return;
    };
    let host = BufferDesc::new(16, usages_all(), MemoryClass::Host).unwrap();
    assert!(matches!(
        device.create_buffer(host),
        Err(PortableError::Unsupported(_))
    ));
    let too_large = BufferDesc {
        size_bytes: device.capabilities().max_buffer_bytes + 1,
        usages: usages_all(),
        memory: MemoryClass::DeviceLocal,
    };
    assert!(matches!(
        device.create_buffer(too_large),
        Err(PortableError::Unsupported(_))
    ));
    let local = BufferDesc::new(16, usages_all(), MemoryClass::DeviceLocal).unwrap();
    assert_eq!(device.create_buffer(local).unwrap().len(), 16);

    let other = &devices().unwrap().1;
    let mut foreign = other.create_buffer(local).unwrap();
    let queue = device.create_queue().unwrap();
    assert!(matches!(
        queue.write_buffer(&mut foreign, 0, &[0; 4]),
        Err(PortableError::Unsupported(_))
    ));
    assert!(matches!(
        queue.read_buffer(&foreign, 0, 4),
        Err(PortableError::Unsupported(_))
    ));
}

#[test]
fn fences_report_completion() {
    let Some(device) = adapter_or_skip("fences") else {
        return;
    };
    let queue = device.create_queue().unwrap();
    let mut buffer = device
        .create_buffer(BufferDesc::new(64, usages_all(), MemoryClass::DeviceLocal).unwrap())
        .unwrap();
    let fence = queue.write_buffer(&mut buffer, 0, &[3; 64]).unwrap();
    fence.wait().unwrap();
    assert_eq!(fence.status().unwrap(), FenceStatus::Complete);
    fence.wait().unwrap();
    let clone = fence.clone();
    assert_eq!(clone.status().unwrap(), FenceStatus::Complete);
}

#[test]
fn copy_buffer_matches_cpu_reference_aligned_and_unaligned() {
    let Some(device) = adapter_or_skip("copy_buffer") else {
        return;
    };
    let cpu = CpuDevice::new().unwrap();
    let cpu_queue = cpu.create_queue().unwrap();
    let queue = device.create_queue().unwrap();
    let desc = BufferDesc::new(19, usages_all(), MemoryClass::Shared).unwrap();
    let source_bytes: Vec<u8> = (1..=19).collect();

    let mut cpu_source = cpu.create_buffer(desc).unwrap();
    let mut cpu_destination = cpu.create_buffer(desc).unwrap();
    cpu_queue
        .write_buffer(&mut cpu_source, 0, &source_bytes)
        .unwrap();
    let mut source = device.create_buffer(desc).unwrap();
    let mut destination = device.create_buffer(desc).unwrap();
    queue.write_buffer(&mut source, 0, &source_bytes).unwrap();

    for (from, to, size) in [(0, 0, 8), (4, 12, 4), (1, 6, 7), (3, 0, 16), (19, 19, 0)] {
        cpu_queue
            .copy_buffer(&cpu_source, from, &mut cpu_destination, to, size)
            .unwrap();
        queue
            .copy_buffer(&source, from, &mut destination, to, size)
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(
            queue.read_buffer(&destination, 0, 19).unwrap(),
            cpu_queue.read_buffer(&cpu_destination, 0, 19).unwrap(),
            "copy {from}->{to} size {size}"
        );
    }
    let before = queue.read_buffer(&destination, 0, 19).unwrap();
    assert!(matches!(
        queue.copy_buffer(&source, 16, &mut destination, 0, 4),
        Err(PortableError::OutOfBounds { .. })
    ));
    assert!(matches!(
        queue.copy_buffer(&source, 0, &mut destination, 17, 4),
        Err(PortableError::OutOfBounds { .. })
    ));
    let no_src = BufferDesc::new(19, BufferUsages::COPY_DST, MemoryClass::Shared).unwrap();
    let no_src = device.create_buffer(no_src).unwrap();
    assert!(matches!(
        queue.copy_buffer(&no_src, 0, &mut destination, 0, 4),
        Err(PortableError::Unsupported(_))
    ));
    assert_eq!(queue.read_buffer(&destination, 0, 19).unwrap(), before);
}
