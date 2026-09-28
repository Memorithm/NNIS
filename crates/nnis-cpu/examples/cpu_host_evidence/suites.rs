//! Qualification suites of the CPU host-evidence harness.
//!
//! Each suite runs the `nnis-cpu` reference and compares every output bit for
//! bit with an independent oracle. The oracle is integer-only: it decomposes
//! each binary32 into an exact integer mantissa and power-of-two exponent,
//! computes sums and products exactly in 128-bit integers, and rounds once to
//! nearest-even binary32 with gradual underflow. It never uses the host's
//! floating-point unit, fused multiply-add or float conversion, so a host
//! whose arithmetic, `mul_add` or subnormal handling differs from IEEE-754
//! binary32 fails the suite. A backend error or panic becomes a failed suite
//! with a detail string. Nothing is timed.
//!
//! Shared by the `cpu_host_evidence` example and the
//! `cpu_host_evidence_harness` integration test.

use std::panic::{catch_unwind, AssertUnwindSafe};

use nnis_core::adapter_evidence::{SuiteOutcomeV1, SuiteResultV1};
use nnis_core::graph::{
    F32GraphLimitsV1, F32GraphV1, F32NodeV1, F32OpV1, F32ShapeV1, F32_GRAPH_POLICY,
    F32_GRAPH_VERSION,
};
use nnis_core::kv_fp4::{Fp4E2M1KvLayoutV1, Fp4ScaleEncodingV1};
use nnis_core::{BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableQueue};
use nnis_cpu::fp4_kv::CpuFp4E2M1KvBlockV1;
use nnis_cpu::graph::execute_f32_graph;
use nnis_cpu::numerical::{CpuF32BinaryOp, CpuF32KernelsV1, CPU_F32_NUMERICAL_POLICY};
use nnis_cpu::{CpuBuffer, CpuDevice};

type SuiteResult = Result<(), String>;
type BinaryOracle = fn(u32, u32) -> u32;

/// Checks and mismatches counted by one suite.
#[derive(Default)]
struct Tally {
    checks: u64,
    mismatches: u64,
}

impl Tally {
    fn bits(&mut self, expected: &[u32], actual: &[f32]) {
        self.checks += expected.len().max(actual.len()) as u64;
        if expected.len() != actual.len() {
            self.mismatches += expected.len().max(actual.len()) as u64;
            return;
        }
        self.mismatches += expected
            .iter()
            .zip(actual)
            .filter(|(e, a)| **e != a.to_bits())
            .count() as u64;
    }
}

/// Run every suite of `nnis_cpu::evidence::CPU_HOST_QUALIFICATION_SUITES_V1`.
pub fn run_all() -> Vec<SuiteResultV1> {
    let exact = "integer-oracle-bit-exact";
    let fused = format!("{CPU_F32_NUMERICAL_POLICY}+integer-oracle-bit-exact");
    vec![
        run("cpu.f32_binary", exact, binary),
        run("cpu.f32_relu_gather", exact, relu_gather),
        run("cpu.f32_serial_accumulation", exact, serial_accumulation),
        run("cpu.f32_fused_projection", &fused, fused_projection),
        run("cpu.f32_graph", exact, graph),
        run("cpu.dsv41_fp4_decode", exact, fp4_decode),
    ]
}

fn run(suite_id: &str, tolerance: &str, suite: fn(&mut Tally) -> SuiteResult) -> SuiteResultV1 {
    let mut tally = Tally::default();
    let outcome = catch_unwind(AssertUnwindSafe(|| suite(&mut tally)));
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
                "{} of {} checks differ from the integer oracle",
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

// ---------------------------------------------------------------------------
// Integer oracle.
// ---------------------------------------------------------------------------

/// Exact value `(-1)^negative * mantissa * 2^exponent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exact {
    negative: bool,
    mantissa: u128,
    exponent: i32,
}

/// Decompose a finite binary32 bit pattern exactly.
pub fn exact(bits: u32) -> Exact {
    let biased = ((bits >> 23) & 0xff) as i32;
    assert!(biased != 0xff, "oracle input must be finite");
    let fraction = u128::from(bits & 0x007f_ffff);
    let (mantissa, exponent) = if biased == 0 {
        (fraction, -149)
    } else {
        (fraction | 0x0080_0000, biased - 150)
    };
    Exact {
        negative: bits >> 31 != 0,
        mantissa,
        exponent,
    }
}

fn bit_length(value: u128) -> i32 {
    128 - value.leading_zeros() as i32
}

/// Round an exact value once to nearest-even binary32 bits, with gradual
/// underflow; overflow gives infinity (which the reference rejects).
pub fn round(value: Exact) -> u32 {
    let sign = u32::from(value.negative) << 31;
    if value.mantissa == 0 {
        return sign;
    }
    assert!(value.mantissa < 1 << 112, "oracle mantissa out of range");
    let top = value.exponent + bit_length(value.mantissa) - 1;
    let normal = top >= -126;
    let shift = if normal {
        bit_length(value.mantissa) - 24
    } else {
        -149 - value.exponent
    };
    let mut quotient = if shift <= 0 {
        value.mantissa << (-shift) as u32
    } else if shift >= 120 {
        0
    } else {
        let quotient = value.mantissa >> shift as u32;
        let remainder = value.mantissa & ((1u128 << shift as u32) - 1);
        let half = 1u128 << (shift as u32 - 1);
        if remainder > half || (remainder == half && quotient & 1 == 1) {
            quotient + 1
        } else {
            quotient
        }
    };
    if !normal {
        // Raw subnormal bits; a carry to 2^23 is the smallest normal.
        return sign | quotient as u32;
    }
    let mut biased = top + 127;
    if quotient == 1 << 24 {
        quotient >>= 1;
        biased += 1;
    }
    if biased >= 255 {
        return sign | 0x7f80_0000;
    }
    sign | ((biased as u32) << 23) | (quotient as u32 & 0x007f_ffff)
}

/// Exact product.
pub fn product(left: Exact, right: Exact) -> Exact {
    Exact {
        negative: left.negative != right.negative,
        mantissa: left.mantissa * right.mantissa,
        exponent: left.exponent + right.exponent,
    }
}

/// Sum that rounds identically to the exact sum.
///
/// A term more than 60 binary orders below the other only acts as a sticky
/// bit, so it is replaced by a one at 61 orders below, which lies strictly
/// inside the same rounding interval and keeps the sign.
pub fn sum(left: Exact, right: Exact) -> Exact {
    if left.mantissa == 0 && right.mantissa == 0 {
        return Exact {
            negative: left.negative && right.negative,
            mantissa: 0,
            exponent: 0,
        };
    }
    if right.mantissa == 0 {
        return left;
    }
    if left.mantissa == 0 {
        return right;
    }
    let top = |value: Exact| value.exponent + bit_length(value.mantissa);
    let sticky = |tiny: Exact, big: Exact| Exact {
        negative: tiny.negative,
        mantissa: 1,
        exponent: top(big) - 61,
    };
    let (left, right) = if top(right) < top(left) - 60 {
        (left, sticky(right, left))
    } else if top(left) < top(right) - 60 {
        (sticky(left, right), right)
    } else {
        (left, right)
    };
    let exponent = left.exponent.min(right.exponent);
    let signed = |value: Exact| {
        let magnitude = (value.mantissa << (value.exponent - exponent) as u32) as i128;
        if value.negative {
            -magnitude
        } else {
            magnitude
        }
    };
    let total = signed(left) + signed(right);
    Exact {
        // Exact cancellation gives +0 under round-to-nearest-even.
        negative: total < 0,
        mantissa: total.unsigned_abs(),
        exponent,
    }
}

fn oracle_add(left: u32, right: u32) -> u32 {
    round(sum(exact(left), exact(right)))
}

fn oracle_multiply(left: u32, right: u32) -> u32 {
    round(product(exact(left), exact(right)))
}

fn oracle_fused(x: u32, w: u32, accumulator: u32) -> u32 {
    round(sum(product(exact(x), exact(w)), exact(accumulator)))
}

fn oracle_relu(value: u32) -> u32 {
    let negative_or_zero = value >> 31 != 0 || value == 0;
    if negative_or_zero {
        0
    } else {
        value
    }
}

fn oracle_sum(values: &[u32]) -> u32 {
    values.iter().fold(0, |acc, &value| oracle_add(acc, value))
}

// ---------------------------------------------------------------------------
// Inputs and buffers.
// ---------------------------------------------------------------------------

fn err(error: impl std::fmt::Debug) -> String {
    format!("{error:?}")
}

/// Deterministic normal-range bit patterns of mixed sign (2^-9 .. 2^6).
fn values(count: usize, seed: u32) -> Vec<u32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let exponent = 118 + (state >> 28);
            (state & 0x8000_0000) | (exponent << 23) | (state & 0x007f_ffff)
        })
        .collect()
}

/// Signed zeros, subnormals, the smallest normals and large finite values.
const EDGE_BITS: [u32; 12] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x8000_0007,
    0x007f_ffff,
    0x0080_0000,
    0x8080_0001,
    0x3f80_0001,
    0x3f7f_fffe,
    0x4b80_0000,
    0xcb80_0000,
    0x5f00_0000,
];

fn tensor(bits: &[u32]) -> Result<CpuBuffer, String> {
    let device = CpuDevice::new().map_err(err)?;
    let usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let desc = BufferDesc::new(bits.len() as u64 * 4, usages, MemoryClass::Host).map_err(err)?;
    let mut buffer = device.create_buffer(desc).map_err(err)?;
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
    device
        .create_queue()
        .map_err(err)?
        .write_buffer(&mut buffer, 0, &bytes)
        .map_err(err)?;
    Ok(buffer)
}

fn read(buffer: &CpuBuffer) -> Result<Vec<f32>, String> {
    let bytes = CpuDevice::new()
        .map_err(err)?
        .create_queue()
        .map_err(err)?
        .read_buffer(buffer, 0, buffer.len() as u64)
        .map_err(err)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn kernels() -> Result<CpuF32KernelsV1, String> {
    CpuF32KernelsV1::new(1 << 24).map_err(err)
}

// ---------------------------------------------------------------------------
// Suites.
// ---------------------------------------------------------------------------

fn binary(tally: &mut Tally) -> SuiteResult {
    let kernels = kernels()?;
    let mut left = values(2000, 3);
    let mut right = values(2000, 5);
    // Every ordered pair of edge values whose result stays finite.
    for &a in &EDGE_BITS {
        for &b in &EDGE_BITS {
            left.push(a);
            right.push(b);
        }
    }
    let cases: [(CpuF32BinaryOp, BinaryOracle); 2] = [
        (CpuF32BinaryOp::Add, oracle_add),
        (CpuF32BinaryOp::Multiply, oracle_multiply),
    ];
    for (operation, oracle) in cases {
        let (l, r): (Vec<u32>, Vec<u32>) = left
            .iter()
            .zip(&right)
            .filter(|(a, b)| oracle(**a, **b) & 0x7f80_0000 != 0x7f80_0000)
            .map(|(a, b)| (*a, *b))
            .unzip();
        let expected: Vec<u32> = l.iter().zip(&r).map(|(a, b)| oracle(*a, *b)).collect();
        let mut output = tensor(&vec![0; l.len()])?;
        kernels
            .binary(operation, &tensor(&l)?, &tensor(&r)?, &mut output)
            .map_err(err)?;
        tally.bits(&expected, &read(&output)?);
    }
    Ok(())
}

fn relu_gather(tally: &mut Tally) -> SuiteResult {
    let kernels = kernels()?;
    let mut input = EDGE_BITS.to_vec();
    input.extend(values(500, 7));
    let mut output = tensor(&vec![0; input.len()])?;
    kernels.relu(&tensor(&input)?, &mut output).map_err(err)?;
    let expected: Vec<u32> = input.iter().map(|&v| oracle_relu(v)).collect();
    tally.bits(&expected, &read(&output)?);

    let indices: Vec<usize> = (0..700).map(|i| (i * 37 + 5) % input.len()).collect();
    let mut output = tensor(&vec![0; indices.len()])?;
    kernels
        .gather(&tensor(&input)?, &indices, &mut output)
        .map_err(err)?;
    let expected: Vec<u32> = indices.iter().map(|&i| input[i]).collect();
    tally.bits(&expected, &read(&output)?);
    Ok(())
}

fn serial_accumulation(tally: &mut Tally) -> SuiteResult {
    let kernels = kernels()?;
    // Order-sensitive fixtures: 2^24 + 1 - 2^24 is 0, reordered it is 1.
    let mut cases = vec![
        vec![0x4b80_0000, 0x3f80_0000, 0xcb80_0000],
        vec![0x4b80_0000, 0xcb80_0000, 0x3f80_0000],
        vec![0x8000_0000],
        vec![0x0000_0001, 0x8000_0001, 0x007f_ffff, 0x0000_0001],
    ];
    for (count, seed) in [(1, 11), (17, 13), (1000, 17), (4096, 19)] {
        cases.push(values(count, seed));
    }
    for case in &cases {
        let mut output = tensor(&[0])?;
        kernels.sum(&tensor(case)?, &mut output).map_err(err)?;
        tally.bits(&[oracle_sum(case)], &read(&output)?);
    }

    let destination: Vec<u32> = values(40, 23);
    let source = values(600, 29);
    let indices: Vec<usize> = (0..600).map(|i| (i * i + 3) % 31).collect();
    let mut expected = destination.clone();
    for (value, &index) in source.iter().zip(&indices) {
        expected[index] = oracle_add(expected[index], *value);
    }
    let mut output = tensor(&destination)?;
    kernels
        .scatter_add(&tensor(&source)?, &indices, &mut output)
        .map_err(err)?;
    tally.bits(&expected, &read(&output)?);
    Ok(())
}

fn oracle_projection(x: &[u32], w: &[u32], k: usize, n: usize) -> Vec<u32> {
    (0..n)
        .map(|column| (0..k).fold(0, |acc, row| oracle_fused(x[row], w[row * n + column], acc)))
        .collect()
}

fn fused_projection(tally: &mut Tally) -> SuiteResult {
    let kernels = kernels()?;
    // (1 + 2^-23)(1 - 2^-23) - 1 = -2^-46 only with a single rounding.
    let mut cases = vec![(
        vec![0xbf80_0000, 0x3f80_0001],
        vec![0x3f80_0000, 0x3f7f_fffe],
        2,
        1,
    )];
    for (k, n, seed) in [(1, 1, 31), (3, 2, 37), (37, 70, 41), (256, 9, 43)] {
        cases.push((values(k, seed), values(k * n, seed + 1000), k, n));
    }
    for (x, w, k, n) in &cases {
        let mut output = tensor(&vec![0; *n])?;
        kernels
            .project_kn(&tensor(x)?, &tensor(w)?, &mut output, *k, *n)
            .map_err(err)?;
        tally.bits(&oracle_projection(x, w, *k, *n), &read(&output)?);
    }
    Ok(())
}

fn graph(tally: &mut Tally) -> SuiteResult {
    let n = 300usize;
    let gather: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % n).collect();
    let scatter: Vec<usize> = (0..n).map(|i| (i * 11) % n).collect();
    let inputs = [F32ShapeV1::Vector(n as u64), F32ShapeV1::Vector(n as u64)];
    let vector = inputs[0];
    let nodes = [
        F32NodeV1 {
            operation: F32OpV1::Add { left: 0, right: 1 },
            output: vector,
        },
        F32NodeV1 {
            operation: F32OpV1::Relu { input: 2 },
            output: vector,
        },
        F32NodeV1 {
            operation: F32OpV1::Multiply { left: 3, right: 1 },
            output: vector,
        },
        F32NodeV1 {
            operation: F32OpV1::Gather {
                input: 4,
                indices: &gather,
            },
            output: vector,
        },
        F32NodeV1 {
            operation: F32OpV1::ScatterAdd {
                base: 5,
                source: 0,
                indices: &scatter,
            },
            output: vector,
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
    for (seed_a, seed_b) in [(47, 53), (59, 61), (67, 71), (73, 79), (83, 89)] {
        let a = values(n, seed_a);
        let b = values(n, seed_b);
        let inputs = [tensor(&a)?, tensor(&b)?];
        let output = execute_f32_graph(
            &CpuDevice::new().map_err(err)?,
            &graph,
            &[&inputs[0], &inputs[1]],
        )
        .map_err(err)?;
        tally.bits(
            &[oracle_graph(&a, &b, &gather, &scatter)],
            &read(&output.output)?,
        );
        // Inputs are read-only.
        tally.bits(&a, &read(&inputs[0])?);
        tally.bits(&b, &read(&inputs[1])?);
    }
    Ok(())
}

/// Oracle of the graph above: Add, Relu, Multiply, Gather, ScatterAdd, Sum.
fn oracle_graph(a: &[u32], b: &[u32], gather: &[usize], scatter: &[usize]) -> u32 {
    let added: Vec<u32> = a.iter().zip(b).map(|(x, y)| oracle_add(*x, *y)).collect();
    let relu: Vec<u32> = added.iter().map(|&v| oracle_relu(v)).collect();
    let multiplied: Vec<u32> = relu
        .iter()
        .zip(b)
        .map(|(x, y)| oracle_multiply(*x, *y))
        .collect();
    let mut gathered: Vec<u32> = gather.iter().map(|&i| multiplied[i]).collect();
    for (value, &index) in a.iter().zip(scatter) {
        gathered[index] = oracle_add(gathered[index], *value);
    }
    oracle_sum(&gathered)
}

/// FP4 E2M1 magnitudes as integer halves: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
const FP4_HALVES: [u128; 8] = [0, 1, 2, 3, 4, 6, 8, 12];

fn oracle_fp4(code: u8, scale: Exact) -> u32 {
    round(Exact {
        negative: code & 8 != 0,
        mantissa: FP4_HALVES[usize::from(code & 7)] * scale.mantissa,
        exponent: scale.exponent - 1,
    })
}

fn fp4_decode(tally: &mut Tally) -> SuiteResult {
    let e8m0 = |byte: u8| Exact {
        negative: false,
        mantissa: 1,
        exponent: i32::from(byte) - 127,
    };
    let mut f32_scales: Vec<u32> = (0..=600).collect();
    f32_scales.extend([
        0x0080_0000,
        0x3f80_0000,
        0x3fc0_0000,
        0x3eaa_aaab,
        0x7f7f_ffff,
    ]);
    let e8m0_bytes: Vec<u8> = (0..=254).collect();
    let cases: [(Fp4ScaleEncodingV1, Vec<Exact>, Vec<u8>); 2] = [
        (
            Fp4ScaleEncodingV1::E8M0,
            e8m0_bytes.iter().map(|&b| e8m0(b)).collect(),
            e8m0_bytes.clone(),
        ),
        (
            Fp4ScaleEncodingV1::F32,
            f32_scales.iter().map(|&b| exact(b)).collect(),
            f32_scales.iter().flat_map(|b| b.to_le_bytes()).collect(),
        ),
    ];
    for (encoding, scales, scale_bytes) in cases {
        // One 16-value row per scale; codes whose result would overflow are
        // replaced by the zero of the same sign.
        let mut codes = Vec::new();
        let mut expected = Vec::new();
        for scale in &scales {
            let row: Vec<u8> = (0u8..16)
                .map(|code| {
                    if oracle_fp4(code, *scale) & 0x7f80_0000 == 0x7f80_0000 {
                        code & 8
                    } else {
                        code
                    }
                })
                .collect();
            expected.extend(row.iter().map(|&code| oracle_fp4(code, *scale)));
            codes.extend(row.chunks(2).map(|pair| pair[0] | (pair[1] << 4)));
        }
        let layout = Fp4E2M1KvLayoutV1::new(scales.len() as u64, 16, 16, encoding).map_err(err)?;
        let block = CpuFp4E2M1KvBlockV1::from_parts(layout, codes, scale_bytes).map_err(err)?;
        tally.bits(&expected, &block.decode().map_err(err)?);
    }
    Ok(())
}
