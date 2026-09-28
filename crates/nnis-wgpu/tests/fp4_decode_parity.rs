//! DSV41-3 FP4 E2M1 decode on WGPU compared with the CPU reference decode.
//!
//! Without an adapter each test writes an explicit SKIP to stderr and passes;
//! that pass is not evidence. Software adapters exercise the API path only and
//! are not hardware evidence. No timing, memory, or residency is measured.

use std::io::Write;
use std::sync::OnceLock;

use nnis_core::kv_fp4::{Fp4E2M1KvLayoutV1, Fp4ScaleEncodingV1, FP4_E2M1_MAGNITUDES};
use nnis_core::PortableBuffer;
use nnis_cpu::fp4_kv::{CpuFp4E2M1KvBlockV1, CpuFp4Error};
use nnis_wgpu::fp4::{WgpuFp4E2M1KvBlockV1, WgpuFp4Error, WGSL_FP4_DECODE_CONTRACT_VERSION};
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

fn layout(rows: u64, width: u32, group: u32, scale: Fp4ScaleEncodingV1) -> Fp4E2M1KvLayoutV1 {
    Fp4E2M1KvLayoutV1::new(rows, width, group, scale).unwrap()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn cpu_error(error: CpuFp4Error) -> WgpuFp4Error {
    match error {
        CpuFp4Error::Layout(error) => WgpuFp4Error::Layout(error),
        CpuFp4Error::CodeBytesMismatch { expected, actual } => {
            WgpuFp4Error::CodeBytesMismatch { expected, actual }
        }
        CpuFp4Error::ScaleBytesMismatch { expected, actual } => {
            WgpuFp4Error::ScaleBytesMismatch { expected, actual }
        }
        CpuFp4Error::InvalidScale { group } => WgpuFp4Error::InvalidScale { group },
        CpuFp4Error::NonZeroPadding { row, position } => {
            WgpuFp4Error::NonZeroPadding { row, position }
        }
        CpuFp4Error::DecodeOverflow { group } => WgpuFp4Error::DecodeOverflow { group },
        CpuFp4Error::HostIndexOverflow => WgpuFp4Error::HostIndexOverflow,
        CpuFp4Error::AllocationFailed => WgpuFp4Error::AllocationFailed,
        other => panic!("CPU error {other:?} has no from_parts counterpart"),
    }
}

/// Decode the same parts on CPU and WGPU and require identical bits.
fn assert_decode_matches(
    device: &WgpuDevice,
    layout: Fp4E2M1KvLayoutV1,
    codes: Vec<u8>,
    scales: Vec<u8>,
) -> usize {
    let cpu = CpuFp4E2M1KvBlockV1::from_parts(layout, codes.clone(), scales.clone()).unwrap();
    let expected = cpu.decode().unwrap();
    let block = WgpuFp4E2M1KvBlockV1::from_parts(device, layout, &codes, &scales).unwrap();
    assert_eq!(*block.storage(), cpu.storage());
    let output = block.decode(device).unwrap();
    assert_eq!(
        output.report.schema_version,
        WGSL_FP4_DECODE_CONTRACT_VERSION
    );
    assert_eq!(output.report.decoded_values, expected.len());
    assert_ne!(output.report.artifact_fingerprint, [0; 32]);
    assert_eq!(
        output.values.descriptor().size_bytes,
        expected.len() as u64 * 4
    );
    let actual = block.decode_to_host(device).unwrap();
    let (expected, actual) = (bits(&expected), bits(&actual));
    if let Some(index) = (0..expected.len()).find(|&i| expected[i] != actual[i]) {
        panic!(
            "decode mismatch at {index}: cpu {:#010x} wgpu {:#010x}",
            expected[index], actual[index]
        );
    }
    expected.len()
}

/// Keep code `c` only if `|c| * scale` decodes finite; otherwise its sign only.
fn finite_code(code: u8, scale: f64) -> u8 {
    let magnitude = f64::from(FP4_E2M1_MAGNITUDES[usize::from(code & 7)]) * scale;
    if (magnitude as f32).is_finite() {
        code
    } else {
        code & 8
    }
}

/// One row per scale; each row holds all 16 codes (clamped to finite decode).
fn all_codes_rows(scales: &[f64]) -> Vec<u8> {
    scales
        .iter()
        .flat_map(|&scale| {
            (0u8..8).map(move |pair| {
                finite_code(2 * pair, scale) | (finite_code(2 * pair + 1, scale) << 4)
            })
        })
        .collect()
}

#[test]
fn every_e8m0_exponent_and_code_decodes_like_cpu() {
    let Some(device) = adapter_or_skip("every_e8m0_exponent_and_code_decodes_like_cpu") else {
        return;
    };
    let exponents: Vec<u8> = (0..=254).collect();
    let scales: Vec<f64> = exponents
        .iter()
        .map(|&k| 2f64.powi(i32::from(k) - 127))
        .collect();
    let decoded = assert_decode_matches(
        device,
        layout(255, 16, 16, Fp4ScaleEncodingV1::E8M0),
        all_codes_rows(&scales),
        exponents,
    );
    assert_eq!(decoded, 255 * 16);
}

#[test]
fn f32_scales_including_subnormal_rounding_decode_like_cpu() {
    let Some(device) = adapter_or_skip("f32_scales_including_subnormal_rounding_decode_like_cpu")
    else {
        return;
    };
    let mut scale_bits: Vec<u32> = vec![
        0,
        1,
        2,
        3,
        5,
        0x0000_0007,
        0x0000_ffff,
        0x0055_5555,
        0x007f_ffff,
        0x0080_0000,
        0x0080_0001,
        0x00ff_ffff,
        0x3f00_0000,
        0x3f80_0000,
        0x3fc0_0000,
        0x7e7f_ffff,
        0x7f7f_ffff,
        f32::MAX.to_bits(),
        (f32::MAX / 6.0).to_bits(),
    ];
    // Every small odd subnormal significand exercises the tie and
    // sticky-bit rounding of m * S at exponent -150.
    scale_bits.extend(1..=600u32);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    while scale_bits.len() < 4096 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let candidate = (state >> 33) as u32 & 0x7fff_ffff;
        if f32::from_bits(candidate).is_finite() {
            scale_bits.push(candidate);
        }
    }
    let scales: Vec<f64> = scale_bits
        .iter()
        .map(|&bits| f64::from(f32::from_bits(bits)))
        .collect();
    let scale_bytes: Vec<u8> = scale_bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    let rows = scale_bits.len() as u64;
    assert_decode_matches(
        device,
        layout(rows, 16, 16, Fp4ScaleEncodingV1::F32),
        all_codes_rows(&scales),
        scale_bytes,
    );
}

#[test]
fn cpu_encoded_blocks_with_padding_and_groups_decode_like_cpu() {
    let Some(device) =
        adapter_or_skip("cpu_encoded_blocks_with_padding_and_groups_decode_like_cpu")
    else {
        return;
    };
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
    };
    for encoding in [Fp4ScaleEncodingV1::F32, Fp4ScaleEncodingV1::E8M0] {
        for &(rows, width, group) in &[(1, 1, 2), (3, 13, 4), (7, 33, 8), (5, 64, 32), (2, 9, 64)] {
            let layout = layout(rows, width, group, encoding);
            let values: Vec<f32> = (0..rows as usize * width as usize)
                .map(|index| match index % 11 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f32::from_bits(3),
                    _ => next() * [1e-3, 1.0, 40.0, 3e5][index % 4],
                })
                .collect();
            let cpu = CpuFp4E2M1KvBlockV1::encode(layout, &values).unwrap();
            assert_decode_matches(device, layout, cpu.codes().to_vec(), cpu.scales().to_vec());
        }
    }
}

#[test]
fn malformed_parts_fail_closed_like_cpu() {
    let Some(device) = adapter_or_skip("malformed_parts_fail_closed_like_cpu") else {
        return;
    };
    let e8 = layout(2, 5, 4, Fp4ScaleEncodingV1::E8M0);
    let f32l = layout(1, 4, 2, Fp4ScaleEncodingV1::F32);
    let good_codes = vec![0x21, 0x43, 0x05, 0x00, 0x65, 0x07, 0x01, 0x00];
    let good_scales = vec![127, 127, 127, 127];
    let one = 1.0f32.to_le_bytes();
    let f32_scales =
        |second: f32| -> Vec<u8> { [one.to_vec(), second.to_le_bytes().to_vec()].concat() };
    let cases: Vec<(Fp4E2M1KvLayoutV1, Vec<u8>, Vec<u8>)> = vec![
        (e8, good_codes[..7].to_vec(), good_scales.clone()),
        (e8, good_codes.clone(), good_scales[..3].to_vec()),
        (
            e8,
            [&good_codes[..2], &[0x15], &good_codes[3..]].concat(),
            good_scales.clone(),
        ),
        (
            e8,
            [&good_codes[..7], &[0x10]].concat(),
            good_scales.clone(),
        ),
        (e8, good_codes.clone(), vec![127, 255, 127, 127]),
        (e8, good_codes.clone(), vec![127, 127, 254, 127]),
        (f32l, vec![0x77, 0x77], f32_scales(f32::NAN)),
        (f32l, vec![0x77, 0x77], f32_scales(f32::INFINITY)),
        (f32l, vec![0x77, 0x77], f32_scales(-0.0)),
        (f32l, vec![0x77, 0x77], f32_scales(-1.0)),
        (f32l, vec![0x77, 0x77], f32_scales(f32::MAX)),
    ];
    for (layout, codes, scales) in cases {
        let expected =
            CpuFp4E2M1KvBlockV1::from_parts(layout, codes.clone(), scales.clone()).unwrap_err();
        let actual = WgpuFp4E2M1KvBlockV1::from_parts(device, layout, &codes, &scales).unwrap_err();
        assert_eq!(actual, cpu_error(expected));
    }
}
