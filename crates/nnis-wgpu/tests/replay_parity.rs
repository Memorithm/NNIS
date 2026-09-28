//! DSV41 replay and cross-layer KV reuse on WGPU compared with the CPU references.
//!
//! Without an adapter each test writes an explicit SKIP to stderr and passes;
//! that pass is not evidence. Software adapters exercise the API path only and
//! are not hardware evidence. No timing, memory, or residency is measured.

use std::io::Write;
use std::sync::OnceLock;

use nnis_core::kv_reuse_plan::{
    CrossLayerKvReusePlanV1, KvComponent, KvLayerReuseV1, KvLayerSourceV1, KvReusePlanError,
};
use nnis_core::replay_state::{
    ReplayIdentityError, ReplayRepresentationIdentityV1, ReplaySourceIdentityV1,
    ReplayStateProviderV1, ReplayWindowRequestV1,
};
use nnis_core::{BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableQueue};
use nnis_cpu::kv_reuse::{CpuCrossLayerKvBindingV1, CpuKvReuseError};
use nnis_cpu::replay::{CpuReplayError, CpuReplaySourceV1};
use nnis_wgpu::replay::{
    WgpuCrossLayerKvBindingV1, WgpuKvReuseError, WgpuReplayError, WgpuReplaySourceV1,
};
use nnis_wgpu::WgpuDevice;

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

fn identity(tag: &str, generation: u64, start: u64, end: u64) -> ReplaySourceIdentityV1 {
    ReplaySourceIdentityV1::new(
        "nnis-reference",
        tag,
        generation,
        ReplayRepresentationIdentityV1::new("dense.f32.rows", 1, 0).unwrap(),
        start,
        end,
    )
    .unwrap()
}

/// Deterministic finite payload including signed zeros, subnormals and
/// extreme normals, so any value-touching path would be visible.
fn payload(values: usize, salt: u32) -> Vec<f32> {
    (0..values)
        .map(|index| match (index as u32 + salt) % 7 {
            0 => -0.0,
            1 => f32::from_bits(1 + index as u32),
            2 => f32::MAX,
            3 => -f32::MIN_POSITIVE,
            _ => (index as f32 + salt as f32) * 0.37 - 5.0,
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn cpu_replay_error(error: CpuReplayError) -> WgpuReplayError {
    match error {
        CpuReplayError::Identity(error) => WgpuReplayError::Identity(error),
        CpuReplayError::ZeroRowWidth => WgpuReplayError::ZeroRowWidth,
        CpuReplayError::PayloadLengthMismatch { expected, actual } => {
            WgpuReplayError::PayloadLengthMismatch { expected, actual }
        }
        CpuReplayError::NonFiniteValue { index } => WgpuReplayError::NonFiniteValue { index },
        CpuReplayError::HostIndexOverflow => WgpuReplayError::HostIndexOverflow,
    }
}

fn cpu_kv_error(error: CpuKvReuseError) -> WgpuKvReuseError {
    match error {
        CpuKvReuseError::Plan(error) => WgpuKvReuseError::Plan(error),
        CpuKvReuseError::Replay(error) => WgpuKvReuseError::Replay(cpu_replay_error(error)),
        CpuKvReuseError::MissingOwnerSource { layer, component } => {
            WgpuKvReuseError::MissingOwnerSource { layer, component }
        }
        CpuKvReuseError::DuplicateOwnerSource { layer, component } => {
            WgpuKvReuseError::DuplicateOwnerSource { layer, component }
        }
        CpuKvReuseError::SourceForReusingLayer { layer, component } => {
            WgpuKvReuseError::SourceForReusingLayer { layer, component }
        }
        CpuKvReuseError::SourceRangeMismatch {
            expected_start,
            expected_end,
            actual_start,
            actual_end,
        } => WgpuKvReuseError::SourceRangeMismatch {
            expected_start,
            expected_end,
            actual_start,
            actual_end,
        },
    }
}

#[test]
fn every_replay_window_matches_cpu_bits() {
    let Some(device) = adapter_or_skip("every_replay_window_matches_cpu_bits") else {
        return;
    };
    let (start, end, width) = (40u64, 51u64, 5usize);
    let items = (end - start + 1) as usize;
    let values = payload(items * width, 3);
    let cpu = CpuReplaySourceV1::new(identity("kv", 2, start, end), width, values.clone()).unwrap();
    let wgpu = WgpuReplaySourceV1::from_host(device, identity("kv", 2, start, end), width, &values)
        .unwrap();
    assert_eq!(wgpu.row_width(), width);
    assert_eq!(bits(&wgpu.read_f32(wgpu.rows()).unwrap()), bits(&values));
    for window_items in 1..=items as u64 {
        let (cpu_request, cpu_rows) = cpu.replay_recent_window(window_items).unwrap();
        let (request, rows) = wgpu.replay_recent_window(window_items).unwrap();
        assert_eq!(request, cpu_request);
        assert_eq!(bits(&wgpu.read_f32(&rows).unwrap()), bits(&cpu_rows));
    }
    for first in start..=end {
        for last in first..=end {
            let request =
                ReplayWindowRequestV1::new(identity("kv", 2, start, end), first, last).unwrap();
            let expected = cpu.replay_window(&request).unwrap();
            let rows = wgpu.replay_window(&request).unwrap();
            assert_eq!(rows.len(), (expected.len() * 4) as u64);
            assert_eq!(bits(&wgpu.read_f32(&rows).unwrap()), bits(&expected));
        }
    }
}

#[test]
fn replayed_windows_are_independent_of_the_source() {
    let Some(device) = adapter_or_skip("replayed_windows_are_independent_of_the_source") else {
        return;
    };
    let values = payload(4 * 3, 1);
    let source =
        WgpuReplaySourceV1::from_host(device, identity("kv", 1, 0, 3), 3, &values).unwrap();
    let (_, mut window) = source.replay_recent_window(2).unwrap();
    let queue = device.create_queue().unwrap();
    queue.write_buffer(&mut window, 0, &[0u8; 24]).unwrap();
    let (_, again) = source.replay_recent_window(2).unwrap();
    assert_eq!(bits(&source.read_f32(&again).unwrap()), bits(&values[6..]));
    assert_eq!(
        bits(&source.read_f32(source.rows()).unwrap()),
        bits(&values)
    );
}

#[test]
fn replay_validation_errors_match_cpu() {
    let Some(device) = adapter_or_skip("replay_validation_errors_match_cpu") else {
        return;
    };
    let id = || identity("kv", 1, 0, 4);
    let good = payload(5 * 2, 0);
    let mut non_finite = good.clone();
    non_finite[7] = f32::NAN;
    let mut infinite = good.clone();
    infinite[3] = f32::NEG_INFINITY;
    let cases: Vec<(usize, Vec<f32>)> = vec![
        (0, Vec::new()),
        (2, good[..9].to_vec()),
        (2, [good.clone(), vec![1.0]].concat()),
        (2, non_finite),
        (2, infinite),
    ];
    for (width, rows) in cases {
        let expected = CpuReplaySourceV1::new(id(), width, rows.clone()).unwrap_err();
        let actual = WgpuReplaySourceV1::from_host(device, id(), width, &rows).unwrap_err();
        assert_eq!(actual, cpu_replay_error(expected));
    }

    let cpu = CpuReplaySourceV1::new(id(), 2, good.clone()).unwrap();
    let wgpu = WgpuReplaySourceV1::from_host(device, id(), 2, &good).unwrap();
    for window_items in [0, 6, u64::MAX] {
        assert_eq!(
            wgpu.replay_recent_window(window_items).unwrap_err(),
            cpu_replay_error(cpu.replay_recent_window(window_items).unwrap_err())
        );
    }
    let drifted = [
        identity("kv", 2, 0, 4),
        identity("other", 1, 0, 4),
        identity("kv", 1, 0, 5),
        ReplaySourceIdentityV1::new(
            "nnis-reference",
            "kv",
            1,
            ReplayRepresentationIdentityV1::new("dense.f32.rows", 1, 1).unwrap(),
            0,
            4,
        )
        .unwrap(),
    ];
    for stale in drifted {
        let request = ReplayWindowRequestV1::recent(stale, 1).unwrap();
        let actual = wgpu.replay_window(&request).unwrap_err();
        assert_eq!(
            actual,
            WgpuReplayError::Identity(ReplayIdentityError::SourceIdentityMismatch)
        );
        assert_eq!(
            actual,
            cpu_replay_error(cpu.replay_window(&request).unwrap_err())
        );
    }
}

#[test]
fn device_buffer_sources_are_validated_and_replay_exactly() {
    let Some(device) = adapter_or_skip("device_buffer_sources_are_validated_and_replay_exactly")
    else {
        return;
    };
    let queue = device.create_queue().unwrap();
    let upload = |values: &[f32], usages: BufferUsages| {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut buffer = device
            .create_buffer(
                BufferDesc::new(bytes.len() as u64, usages, MemoryClass::DeviceLocal).unwrap(),
            )
            .unwrap();
        queue.write_buffer(&mut buffer, 0, &bytes).unwrap();
        buffer
    };
    let all = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let values = payload(6 * 4, 5);
    let cpu = CpuReplaySourceV1::new(identity("kv", 3, 10, 15), 4, values.clone()).unwrap();
    let wgpu =
        WgpuReplaySourceV1::from_buffer(device, identity("kv", 3, 10, 15), 4, upload(&values, all))
            .unwrap();
    for window_items in 1..=6 {
        let (_, expected) = cpu.replay_recent_window(window_items).unwrap();
        let (_, rows) = wgpu.replay_recent_window(window_items).unwrap();
        assert_eq!(bits(&wgpu.read_f32(&rows).unwrap()), bits(&expected));
    }

    let mut bad = values.clone();
    bad[17] = f32::INFINITY;
    assert_eq!(
        WgpuReplaySourceV1::from_buffer(device, identity("kv", 3, 10, 15), 4, upload(&bad, all))
            .unwrap_err(),
        WgpuReplayError::NonFiniteDeviceValue
    );
    assert_eq!(
        WgpuReplaySourceV1::from_buffer(
            device,
            identity("kv", 3, 10, 15),
            4,
            upload(&values[..20], all)
        )
        .unwrap_err(),
        WgpuReplayError::PayloadByteLengthMismatch {
            expected_bytes: 96,
            actual_bytes: 80,
        }
    );
    assert!(matches!(
        WgpuReplaySourceV1::from_buffer(
            device,
            identity("kv", 3, 10, 15),
            4,
            upload(&values, BufferUsages::COPY_SRC | BufferUsages::COPY_DST)
        ),
        Err(WgpuReplayError::Portable(_))
    ));
    assert_eq!(
        WgpuReplaySourceV1::from_buffer(device, identity("kv", 3, 10, 15), 0, upload(&values, all))
            .unwrap_err(),
        WgpuReplayError::ZeroRowWidth
    );
}

const WIDTH: usize = 2;

fn host_rows(start: u64, end: u64, fill: f32) -> Vec<f32> {
    let items = (end - start + 1) as usize;
    (0..items * WIDTH)
        .map(|index| fill + index as f32 * 0.25)
        .collect()
}

fn cpu_source(tag: &str, start: u64, end: u64, fill: f32) -> CpuReplaySourceV1 {
    CpuReplaySourceV1::new(
        identity(tag, 1, start, end),
        WIDTH,
        host_rows(start, end, fill),
    )
    .unwrap()
}

fn wgpu_source(
    device: &WgpuDevice,
    tag: &str,
    start: u64,
    end: u64,
    fill: f32,
) -> WgpuReplaySourceV1 {
    WgpuReplaySourceV1::from_host(
        device,
        identity(tag, 1, start, end),
        WIDTH,
        &host_rows(start, end, fill),
    )
    .unwrap()
}

/// Layer 1 reuses layer 0's key but owns its value; layer 2 reuses both;
/// layer 3 owns both.
fn plan() -> CrossLayerKvReusePlanV1 {
    use KvLayerSourceV1::{Own, ReuseFrom};
    CrossLayerKvReusePlanV1::new(
        "fixture.kv-share.v1",
        "fixture-model",
        1,
        vec![
            KvLayerReuseV1::OWN,
            KvLayerReuseV1 {
                key: ReuseFrom(0),
                value: Own,
            },
            KvLayerReuseV1 {
                key: ReuseFrom(0),
                value: ReuseFrom(1),
            },
            KvLayerReuseV1::OWN,
        ],
    )
    .unwrap()
}

type Spec = (u32, &'static str, u64, u64, f32);

fn bind_both(
    device: &WgpuDevice,
    keys: &[Spec],
    values: &[Spec],
) -> (
    Result<CpuCrossLayerKvBindingV1, CpuKvReuseError>,
    Result<WgpuCrossLayerKvBindingV1, WgpuKvReuseError>,
) {
    let cpu = |specs: &[Spec]| {
        specs
            .iter()
            .map(|&(layer, tag, start, end, fill)| (layer, cpu_source(tag, start, end, fill)))
            .collect::<Vec<_>>()
    };
    let gpu = |specs: &[Spec]| {
        specs
            .iter()
            .map(|&(layer, tag, start, end, fill)| {
                (layer, wgpu_source(device, tag, start, end, fill))
            })
            .collect::<Vec<_>>()
    };
    (
        CpuCrossLayerKvBindingV1::new(plan(), cpu(keys), cpu(values)),
        WgpuCrossLayerKvBindingV1::new(plan(), gpu(keys), gpu(values)),
    )
}

const KEYS: &[Spec] = &[(0, "k0", 0, 7, 0.0), (3, "k3", 0, 7, 300.0)];
const VALUES: &[Spec] = &[
    (0, "v0", 0, 7, 100.0),
    (1, "v1", 0, 7, 200.0),
    (3, "v3", 0, 7, 400.0),
];

#[test]
fn kv_reuse_reads_match_cpu_owner_resolution() {
    let Some(device) = adapter_or_skip("kv_reuse_reads_match_cpu_owner_resolution") else {
        return;
    };
    let (cpu, wgpu) = bind_both(device, KEYS, VALUES);
    let (cpu, wgpu) = (cpu.unwrap(), wgpu.unwrap());
    for component in [KvComponent::Key, KvComponent::Value] {
        assert_eq!(
            wgpu.bound_source_count(component),
            cpu.bound_source_count(component)
        );
        for layer in 0..4 {
            let owner = wgpu.source(layer, component).unwrap();
            assert_eq!(
                owner.replay_source_identity(),
                cpu.source(layer, component)
                    .unwrap()
                    .replay_source_identity()
            );
            for window_items in 1..=8 {
                let (cpu_request, expected) = cpu
                    .replay_recent_window(layer, component, window_items)
                    .unwrap();
                let (request, rows) = wgpu
                    .replay_recent_window(layer, component, window_items)
                    .unwrap();
                assert_eq!(request, cpu_request);
                assert_eq!(bits(&owner.read_f32(&rows).unwrap()), bits(&expected));
                let explicit = wgpu.replay_window(layer, component, &request).unwrap();
                assert_eq!(bits(&owner.read_f32(&explicit).unwrap()), bits(&expected));
            }
        }
        assert_eq!(
            wgpu.source(4, component).unwrap_err(),
            cpu_kv_error(cpu.source(4, component).unwrap_err())
        );
    }
    // A request bound to layer 1's own value source is not layer 2's key.
    let foreign = ReplayWindowRequestV1::recent(
        wgpu.source(1, KvComponent::Value)
            .unwrap()
            .replay_source_identity()
            .clone(),
        1,
    )
    .unwrap();
    assert_eq!(
        wgpu.replay_window(2, KvComponent::Key, &foreign)
            .unwrap_err(),
        cpu_kv_error(
            cpu.replay_window(2, KvComponent::Key, &foreign)
                .unwrap_err()
        )
    );
}

#[test]
fn kv_binding_errors_match_cpu() {
    let Some(device) = adapter_or_skip("kv_binding_errors_match_cpu") else {
        return;
    };
    let cases: Vec<(Vec<Spec>, Vec<Spec>)> = vec![
        (KEYS.to_vec(), VALUES[..2].to_vec()),
        (KEYS[..1].to_vec(), VALUES.to_vec()),
        ([KEYS, &[(0, "k0b", 0, 7, 9.0)]].concat(), VALUES.to_vec()),
        ([KEYS, &[(1, "k1", 0, 7, 9.0)]].concat(), VALUES.to_vec()),
        (KEYS.to_vec(), [VALUES, &[(2, "v2", 0, 7, 9.0)]].concat()),
        ([KEYS, &[(4, "k4", 0, 7, 9.0)]].concat(), VALUES.to_vec()),
        (
            KEYS.to_vec(),
            vec![
                (0, "v0", 0, 7, 100.0),
                (1, "v1", 1, 8, 200.0),
                (3, "v3", 0, 7, 400.0),
            ],
        ),
        (
            vec![(0, "k0", 0, 7, 0.0), (3, "k3", 0, 6, 0.0)],
            VALUES.to_vec(),
        ),
    ];
    for (keys, values) in cases {
        let (cpu, wgpu) = bind_both(device, &keys, &values);
        let expected = cpu_kv_error(cpu.unwrap_err());
        assert_eq!(wgpu.unwrap_err(), expected);
    }
    let (_, wgpu) = bind_both(device, &[(4, "k4", 0, 7, 0.0)], &[]);
    assert_eq!(
        wgpu.unwrap_err(),
        WgpuKvReuseError::Plan(KvReusePlanError::LayerOutOfRange {
            layer: 4,
            layers: 4,
        })
    );
}
