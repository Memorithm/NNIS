//! Parity suites of the WGPU hardware-evidence harness.
//!
//! Each suite compares WGPU results with the `nnis-cpu` reference under the
//! declared tolerance of the slice it covers, and counts checks and
//! mismatches. A backend error or panic becomes a failed suite with a detail
//! string. Nothing is timed.
//!
//! Shared by the `wgpu_adapter_evidence` example and the
//! `adapter_evidence_harness` integration test.

use std::panic::{catch_unwind, AssertUnwindSafe};

use nnis_core::adapter_evidence::{SuiteOutcomeV1, SuiteResultV1};
use nnis_core::graph::{
    F32GraphLimitsV1, F32GraphV1, F32NodeV1, F32OpV1, F32ShapeV1, F32_GRAPH_POLICY,
    F32_GRAPH_VERSION,
};
use nnis_core::kv_fp4::{Fp4E2M1KvLayoutV1, Fp4ScaleEncodingV1, FP4_E2M1_MAGNITUDES};
use nnis_core::replay_state::{
    ReplayRepresentationIdentityV1, ReplaySourceIdentityV1, ReplayWindowRequestV1,
};
use nnis_core::speculative_verify::{ConfidenceScheduleV1, SpeculativeDraftV1};
use nnis_core::{BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableQueue};
use nnis_cpu::fp4_kv::CpuFp4E2M1KvBlockV1;
use nnis_cpu::graph::execute_f32_graph as cpu_graph;
use nnis_cpu::numerical::{CpuF32BinaryOp, CpuF32KernelsV1};
use nnis_cpu::replay::CpuReplaySourceV1;
use nnis_cpu::speculative::CpuGreedySpeculativeVerifierV1;
use nnis_cpu::{CpuBuffer, CpuDevice};
use nnis_wgpu::fp4::{WgpuFp4E2M1KvBlockV1, WGSL_FP4_DECODE_NUMERICAL_POLICY};
use nnis_wgpu::graph::execute_f32_graph as wgpu_graph;
use nnis_wgpu::numerical::{
    WgpuF32BinaryOp, WgpuF32KernelsV1, WGSL_F32_EXACT_NUMERICAL_POLICY,
    WGSL_F32_PROJECTION_NUMERICAL_POLICY,
};
use nnis_wgpu::replay::WgpuReplaySourceV1;
use nnis_wgpu::speculative::{
    WgpuGreedySpeculativeVerifierV1, WGSL_GREEDY_ARGMAX_NUMERICAL_POLICY,
};
use nnis_wgpu::{WgpuBuffer, WgpuDevice};

type SuiteResult = Result<(), String>;

/// Checks and mismatches counted by one suite.
#[derive(Default)]
struct Tally {
    checks: u64,
    mismatches: u64,
}

impl Tally {
    fn bits(&mut self, expected: &[f32], actual: &[f32]) {
        self.checks += expected.len().max(actual.len()) as u64;
        if expected.len() != actual.len() {
            self.mismatches += expected.len().max(actual.len()) as u64;
            return;
        }
        self.mismatches += expected
            .iter()
            .zip(actual)
            .filter(|(e, a)| e.to_bits() != a.to_bits())
            .count() as u64;
    }

    fn equal<T: PartialEq>(&mut self, expected: &T, actual: &T) {
        self.checks += 1;
        self.mismatches += u64::from(expected != actual);
    }
}

/// Run every suite of `nnis_wgpu::evidence::WGPU_QUALIFICATION_SUITES_V1`.
pub fn run_all(device: &WgpuDevice) -> Vec<SuiteResultV1> {
    vec![
        run("wgpu.add_f32", "bit-exact-normal-range", device, add_f32),
        run(
            "wgpu.portable_memory",
            "byte-exact",
            device,
            portable_memory,
        ),
        run(
            "wgpu.f32_kernels",
            &format!("{WGSL_F32_EXACT_NUMERICAL_POLICY}+{WGSL_F32_PROJECTION_NUMERICAL_POLICY}"),
            device,
            f32_kernels,
        ),
        run(
            "wgpu.f32_graph",
            "projection-free-graph-bit-exact",
            device,
            f32_graph,
        ),
        run("wgpu.dsv41_replay_kv", "bit-exact", device, replay),
        run(
            "wgpu.dsv41_fp4_decode",
            WGSL_FP4_DECODE_NUMERICAL_POLICY,
            device,
            fp4_decode,
        ),
        run(
            "wgpu.dsv41_speculative",
            WGSL_GREEDY_ARGMAX_NUMERICAL_POLICY,
            device,
            speculative,
        ),
    ]
}

fn run(
    suite_id: &str,
    tolerance: &str,
    device: &WgpuDevice,
    suite: fn(&WgpuDevice, &mut Tally) -> SuiteResult,
) -> SuiteResultV1 {
    let mut tally = Tally::default();
    let outcome = catch_unwind(AssertUnwindSafe(|| suite(device, &mut tally)));
    let error = match outcome {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(error),
        Err(_) => Some("suite panicked".to_owned()),
    };
    let (outcome, detail) = match error {
        Some(error) => (SuiteOutcomeV1::Fail, format!("execution failed: {error}")),
        None if tally.mismatches > 0 => (
            SuiteOutcomeV1::Fail,
            format!(
                "{} of {} checks missed the tolerance",
                tally.mismatches, tally.checks
            ),
        ),
        None if tally.checks == 0 => (SuiteOutcomeV1::Fail, "no checks ran".to_owned()),
        None => (SuiteOutcomeV1::Pass, String::new()),
    };
    SuiteResultV1 {
        suite_id: suite_id.to_owned(),
        outcome,
        checks: tally.checks,
        mismatches: tally.mismatches,
        tolerance: tolerance.to_owned(),
        detail: detail
            .chars()
            .filter(|c| !c.is_control())
            .take(400)
            .collect(),
    }
}

fn err(error: impl std::fmt::Debug) -> String {
    format!("{error:?}")
}

fn usages() -> BufferUsages {
    BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
}

fn encode(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Deterministic normal-range values of mixed sign (2^-9 .. 2^6).
fn values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let exponent = 118 + (state >> 28);
            f32::from_bits((state & 0x8000_0000) | (exponent << 23) | (state & 0x007f_ffff))
        })
        .collect()
}

fn wgpu_tensor(device: &WgpuDevice, values: &[f32]) -> Result<WgpuBuffer, String> {
    let desc = BufferDesc::new(values.len() as u64 * 4, usages(), MemoryClass::DeviceLocal)
        .map_err(err)?;
    let mut buffer = device.create_buffer(desc).map_err(err)?;
    device
        .create_queue()
        .map_err(err)?
        .write_buffer(&mut buffer, 0, &encode(values))
        .map_err(err)?;
    Ok(buffer)
}

fn wgpu_values(device: &WgpuDevice, buffer: &WgpuBuffer) -> Result<Vec<f32>, String> {
    let bytes = device
        .create_queue()
        .map_err(err)?
        .read_buffer(buffer, 0, buffer.len())
        .map_err(err)?;
    Ok(decode(&bytes))
}

fn cpu_tensor(values: &[f32]) -> Result<CpuBuffer, String> {
    let device = CpuDevice::new().map_err(err)?;
    let desc =
        BufferDesc::new(values.len() as u64 * 4, usages(), MemoryClass::Host).map_err(err)?;
    let mut buffer = device.create_buffer(desc).map_err(err)?;
    device
        .create_queue()
        .map_err(err)?
        .write_buffer(&mut buffer, 0, &encode(values))
        .map_err(err)?;
    Ok(buffer)
}

fn cpu_values(buffer: &CpuBuffer) -> Result<Vec<f32>, String> {
    let queue = CpuDevice::new().map_err(err)?.create_queue().map_err(err)?;
    Ok(decode(
        &queue
            .read_buffer(buffer, 0, buffer.len() as u64)
            .map_err(err)?,
    ))
}

fn add_f32(device: &WgpuDevice, tally: &mut Tally) -> SuiteResult {
    for (count, seed) in [(1, 3), (63, 5), (64, 7), (4097, 9)] {
        let left = values(count, seed);
        let right = values(count, seed + 100);
        let expected: Vec<f32> = left.iter().zip(&right).map(|(l, r)| l + r).collect();
        let actual = device.add_f32(&left, &right).map_err(err)?;
        tally.bits(&expected, &actual);
    }
    Ok(())
}

fn portable_memory(device: &WgpuDevice, tally: &mut Tally) -> SuiteResult {
    let queue = device.create_queue().map_err(err)?;
    let size = 103u64;
    let mut model = vec![0u8; size as usize];
    let mut buffer = device
        .create_buffer(BufferDesc::new(size, usages(), MemoryClass::DeviceLocal).map_err(err)?)
        .map_err(err)?;
    let zero = queue.read_buffer(&buffer, 0, size).map_err(err)?;
    tally.equal(&model, &zero);
    for (offset, length, fill) in [(0u64, 7usize, 1u8), (5, 13, 2), (50, 53, 3), (101, 2, 4)] {
        let bytes: Vec<u8> = (0..length)
            .map(|i| fill.wrapping_mul(31).wrapping_add(i as u8))
            .collect();
        queue
            .write_buffer(&mut buffer, offset, &bytes)
            .map_err(err)?;
        model[offset as usize..offset as usize + length].copy_from_slice(&bytes);
        let read = queue.read_buffer(&buffer, 0, size).map_err(err)?;
        tally.equal(&model, &read);
    }
    let mut target = device
        .create_buffer(BufferDesc::new(size, usages(), MemoryClass::DeviceLocal).map_err(err)?)
        .map_err(err)?;
    let mut target_model = vec![0u8; size as usize];
    for (source_offset, destination_offset, length) in
        [(0u64, 0u64, 100u64), (3, 9, 41), (8, 4, 16)]
    {
        nnis_core::PortableFence::wait(
            &queue
                .copy_buffer(
                    &buffer,
                    source_offset,
                    &mut target,
                    destination_offset,
                    length,
                )
                .map_err(err)?,
        )
        .map_err(err)?;
        let (s, d, l) = (
            source_offset as usize,
            destination_offset as usize,
            length as usize,
        );
        target_model[d..d + l].copy_from_slice(&model[s..s + l]);
        let read = queue.read_buffer(&target, 0, size).map_err(err)?;
        tally.equal(&target_model, &read);
    }
    Ok(())
}

fn f32_kernels(device: &WgpuDevice, tally: &mut Tally) -> SuiteResult {
    let kernels = WgpuF32KernelsV1::new(device).map_err(err)?;
    let cpu = CpuF32KernelsV1::new(1 << 24).map_err(err)?;
    let left = values(1000, 11);
    let right = values(1000, 13);
    for (cpu_op, wgpu_op) in [
        (CpuF32BinaryOp::Add, WgpuF32BinaryOp::Add),
        (CpuF32BinaryOp::Multiply, WgpuF32BinaryOp::Multiply),
    ] {
        let mut cpu_out = cpu_tensor(&vec![0.0; left.len()])?;
        cpu.binary(
            cpu_op,
            &cpu_tensor(&left)?,
            &cpu_tensor(&right)?,
            &mut cpu_out,
        )
        .map_err(err)?;
        let mut out = wgpu_tensor(device, &vec![0.0; left.len()])?;
        kernels
            .binary(
                wgpu_op,
                &wgpu_tensor(device, &left)?,
                &wgpu_tensor(device, &right)?,
                &mut out,
            )
            .map_err(err)?;
        tally.bits(&cpu_values(&cpu_out)?, &wgpu_values(device, &out)?);
    }
    let mut input = vec![0.0, -0.0, f32::from_bits(1), -f32::from_bits(5), f32::MAX];
    input.extend(values(300, 17));
    let mut cpu_out = cpu_tensor(&vec![0.0; input.len()])?;
    cpu.relu(&cpu_tensor(&input)?, &mut cpu_out).map_err(err)?;
    let mut out = wgpu_tensor(device, &vec![0.0; input.len()])?;
    kernels
        .relu(&wgpu_tensor(device, &input)?, &mut out)
        .map_err(err)?;
    tally.bits(&cpu_values(&cpu_out)?, &wgpu_values(device, &out)?);

    let indices: Vec<usize> = (0..200).map(|i| (i * 37 + 5) % input.len()).collect();
    let mut cpu_out = cpu_tensor(&vec![0.0; indices.len()])?;
    cpu.gather(&cpu_tensor(&input)?, &indices, &mut cpu_out)
        .map_err(err)?;
    let mut out = wgpu_tensor(device, &vec![0.0; indices.len()])?;
    kernels
        .gather(&wgpu_tensor(device, &input)?, &indices, &mut out)
        .map_err(err)?;
    tally.bits(&cpu_values(&cpu_out)?, &wgpu_values(device, &out)?);

    let sum_input = values(1000, 19);
    let mut cpu_out = cpu_tensor(&[0.0])?;
    cpu.sum(&cpu_tensor(&sum_input)?, &mut cpu_out)
        .map_err(err)?;
    let mut out = wgpu_tensor(device, &[0.0])?;
    kernels
        .sum(&wgpu_tensor(device, &sum_input)?, &mut out)
        .map_err(err)?;
    tally.bits(&cpu_values(&cpu_out)?, &wgpu_values(device, &out)?);

    let destination = values(40, 23);
    let source = values(300, 29);
    let scatter: Vec<usize> = (0..300).map(|i| (i * i + 3) % 31).collect();
    let mut cpu_out = cpu_tensor(&destination)?;
    cpu.scatter_add(&cpu_tensor(&source)?, &scatter, &mut cpu_out)
        .map_err(err)?;
    let mut out = wgpu_tensor(device, &destination)?;
    kernels
        .scatter_add(&wgpu_tensor(device, &source)?, &scatter, &mut out)
        .map_err(err)?;
    tally.bits(&cpu_values(&cpu_out)?, &wgpu_values(device, &out)?);

    let (k, n) = (37, 70);
    let x = values(k, 31);
    let w = values(k * n, 37);
    let mut cpu_out = cpu_tensor(&vec![0.0; n])?;
    cpu.project_kn(&cpu_tensor(&x)?, &cpu_tensor(&w)?, &mut cpu_out, k, n)
        .map_err(err)?;
    let mut out = wgpu_tensor(device, &vec![0.0; n])?;
    kernels
        .project_kn(
            &wgpu_tensor(device, &x)?,
            &wgpu_tensor(device, &w)?,
            &mut out,
            k,
            n,
        )
        .map_err(err)?;
    let expected = cpu_values(&cpu_out)?;
    let actual = wgpu_values(device, &out)?;
    for column in 0..n {
        let magnitude: f64 = (0..k)
            .map(|row| (f64::from(x[row]) * f64::from(w[row * n + column])).abs())
            .sum();
        let bound = (3 * k + 1) as f64 * 2f64.powi(-24) * magnitude;
        let difference = (f64::from(actual[column]) - f64::from(expected[column])).abs();
        tally.checks += 1;
        // A NaN difference is a mismatch.
        let within = difference <= bound;
        tally.mismatches += u64::from(!within);
    }
    Ok(())
}

fn f32_graph(device: &WgpuDevice, tally: &mut Tally) -> SuiteResult {
    let n = 300u64;
    let gather: Vec<usize> = (0..n as usize).map(|i| (i * 7 + 3) % n as usize).collect();
    let scatter: Vec<usize> = (0..n as usize).map(|i| (i * 11) % n as usize).collect();
    let inputs = [F32ShapeV1::Vector(n), F32ShapeV1::Vector(n)];
    let nodes = [
        F32NodeV1 {
            operation: F32OpV1::Add { left: 0, right: 1 },
            output: inputs[0],
        },
        F32NodeV1 {
            operation: F32OpV1::Relu { input: 2 },
            output: inputs[0],
        },
        F32NodeV1 {
            operation: F32OpV1::Multiply { left: 3, right: 1 },
            output: inputs[0],
        },
        F32NodeV1 {
            operation: F32OpV1::Gather {
                input: 4,
                indices: &gather,
            },
            output: inputs[0],
        },
        F32NodeV1 {
            operation: F32OpV1::ScatterAdd {
                base: 5,
                source: 0,
                indices: &scatter,
            },
            output: inputs[0],
        },
        F32NodeV1 {
            operation: F32OpV1::Sum { input: 6 },
            output: F32ShapeV1::Vector(1),
        },
    ];
    let graph = F32GraphV1 {
        schema_version: F32_GRAPH_VERSION,
        numerical_policy: F32_GRAPH_POLICY,
        inputs: &inputs,
        nodes: &nodes,
    }
    .validate(F32GraphLimitsV1 {
        max_inputs: 16,
        max_nodes: 32,
        max_tensor_bytes: 1 << 20,
        max_live_payload_bytes: 1 << 24,
        max_scratch_bytes: 1 << 20,
        max_work_items: 1 << 24,
    })
    .map_err(err)?;
    let a = values(n as usize, 41);
    let b = values(n as usize, 43);
    let cpu_inputs = [cpu_tensor(&a)?, cpu_tensor(&b)?];
    let cpu = cpu_graph(
        &CpuDevice::new().map_err(err)?,
        &graph,
        &[&cpu_inputs[0], &cpu_inputs[1]],
    )
    .map_err(err)?;
    let wgpu_inputs = [wgpu_tensor(device, &a)?, wgpu_tensor(device, &b)?];
    let wgpu = wgpu_graph(device, &graph, &[&wgpu_inputs[0], &wgpu_inputs[1]]).map_err(err)?;
    tally.bits(
        &cpu_values(&cpu.output)?,
        &wgpu_values(device, &wgpu.output)?,
    );
    tally.bits(&a, &wgpu_values(device, &wgpu_inputs[0])?);
    Ok(())
}

fn replay_identity(start: u64, end: u64) -> Result<ReplaySourceIdentityV1, String> {
    ReplaySourceIdentityV1::new(
        "nnis-evidence",
        "kv",
        1,
        ReplayRepresentationIdentityV1::new("dense.f32.rows", 1, 0).map_err(err)?,
        start,
        end,
    )
    .map_err(err)
}

fn replay(device: &WgpuDevice, tally: &mut Tally) -> SuiteResult {
    let (start, end, width) = (10u64, 25u64, 6usize);
    let mut rows = values((end - start + 1) as usize * width, 47);
    rows[3] = -0.0;
    rows[4] = f32::from_bits(7);
    let cpu =
        CpuReplaySourceV1::new(replay_identity(start, end)?, width, rows.clone()).map_err(err)?;
    let gpu = WgpuReplaySourceV1::from_host(device, replay_identity(start, end)?, width, &rows)
        .map_err(err)?;
    for first in start..=end {
        for last in first..=end {
            let request = ReplayWindowRequestV1::new(replay_identity(start, end)?, first, last)
                .map_err(err)?;
            let expected = cpu.replay_window(&request).map_err(err)?;
            let window = gpu.replay_window(&request).map_err(err)?;
            tally.bits(&expected, &gpu.read_f32(&window).map_err(err)?);
        }
    }
    Ok(())
}

fn fp4_decode(device: &WgpuDevice, tally: &mut Tally) -> SuiteResult {
    let finite_code = |code: u8, scale: f64| {
        let magnitude = f64::from(FP4_E2M1_MAGNITUDES[usize::from(code & 7)]) * scale;
        if (magnitude as f32).is_finite() {
            code
        } else {
            code & 8
        }
    };
    let rows_for = |scales: &[f64]| -> Vec<u8> {
        scales
            .iter()
            .flat_map(|&scale| {
                (0u8..8).map(move |pair| {
                    finite_code(2 * pair, scale) | (finite_code(2 * pair + 1, scale) << 4)
                })
            })
            .collect()
    };
    let exponents: Vec<u8> = (0..=254).collect();
    let e8_scales: Vec<f64> = exponents
        .iter()
        .map(|&k| 2f64.powi(i32::from(k) - 127))
        .collect();
    let mut f32_bits: Vec<u32> = (0..=600).collect();
    f32_bits.extend([0x0080_0000, 0x3f80_0000, 0x3fc0_0000, 0x7f7f_ffff]);
    let f32_scales: Vec<f64> = f32_bits
        .iter()
        .map(|&b| f64::from(f32::from_bits(b)))
        .collect();
    let cases = [
        (
            Fp4ScaleEncodingV1::E8M0,
            rows_for(&e8_scales),
            exponents.clone(),
            255u64,
        ),
        (
            Fp4ScaleEncodingV1::F32,
            rows_for(&f32_scales),
            f32_bits.iter().flat_map(|b| b.to_le_bytes()).collect(),
            f32_bits.len() as u64,
        ),
    ];
    for (encoding, codes, scales, rows) in cases {
        let layout = Fp4E2M1KvLayoutV1::new(rows, 16, 16, encoding).map_err(err)?;
        let cpu =
            CpuFp4E2M1KvBlockV1::from_parts(layout, codes.clone(), scales.clone()).map_err(err)?;
        let gpu = WgpuFp4E2M1KvBlockV1::from_parts(device, layout, &codes, &scales).map_err(err)?;
        tally.bits(
            &cpu.decode().map_err(err)?,
            &gpu.decode_to_host(device).map_err(err)?,
        );
    }
    Ok(())
}

fn speculative(device: &WgpuDevice, tally: &mut Tally) -> SuiteResult {
    let mut state = 0x0123_4567_89ab_cdefu64;
    let mut next = move |bound: u32| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % u64::from(bound)) as u32
    };
    const PALETTE: [f32; 8] = [0.0, -0.0, 1.0, -1.0, 2.5, f32::MAX, 1.0e-45, -1.0e-45];
    for case in 0..64 {
        let vocab = [1, 3, 7, 64, 1000][case % 5];
        let len = 1 + next(8) as usize;
        let tokens: Vec<u32> = (0..len).map(|_| next(vocab)).collect();
        let confidences: Vec<f32> = (0..len).map(|_| next(11) as f32 / 10.0).collect();
        let draft = SpeculativeDraftV1::new(tokens, confidences).map_err(err)?;
        let schedule =
            ConfidenceScheduleV1::new("harness.threshold", 1, next(11) as f32 / 10.0, 1 + next(8))
                .map_err(err)?;
        let rows = schedule.scheduled_len(&draft) as usize + 1;
        let mut logits: Vec<f32> = (0..rows * vocab as usize)
            .map(|_| PALETTE[next(PALETTE.len() as u32) as usize])
            .collect();
        for (row, &token) in draft.tokens().iter().enumerate().take(rows - 1) {
            if next(3) != 0 {
                logits[row * vocab as usize + token as usize] = f32::MAX;
            }
        }
        let expected = CpuGreedySpeculativeVerifierV1::new(vocab)
            .map_err(err)?
            .verify(&draft, &schedule, &logits)
            .map_err(err)?;
        let outcome = WgpuGreedySpeculativeVerifierV1::new(vocab)
            .map_err(err)?
            .verify_host(device, &draft, &schedule, &logits)
            .map_err(err)?;
        tally.equal(&expected, &outcome.verification);
    }
    Ok(())
}
