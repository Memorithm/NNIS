//! WGSL F32 reference kernels compared with the `nnis-cpu` reference.
//!
//! Without an adapter each test writes an explicit SKIP to stderr and passes;
//! that pass is not evidence. Software adapters exercise the API path only and
//! are not hardware evidence. No timing is measured.

use std::io::Write;
use std::sync::OnceLock;

use nnis_core::{
    BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableError, PortableQueue,
};
use nnis_cpu::numerical::{CpuF32BinaryOp, CpuF32KernelsV1};
use nnis_cpu::{CpuBuffer, CpuDevice};
use nnis_wgpu::numerical::{WgpuF32BinaryOp, WgpuF32KernelsV1, WgpuF32Operation};
use nnis_wgpu::{WgpuBuffer, WgpuDevice};

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

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

struct Pair {
    cpu: CpuDevice,
    wgpu: &'static WgpuDevice,
}

impl Pair {
    fn new(wgpu: &'static WgpuDevice) -> Self {
        Self {
            cpu: CpuDevice::new().unwrap(),
            wgpu,
        }
    }

    fn cpu_buffer(&self, values: &[f32]) -> CpuBuffer {
        let desc = BufferDesc::new(values.len() as u64 * 4, usages(), MemoryClass::Host).unwrap();
        let mut buffer = self.cpu.create_buffer(desc).unwrap();
        self.cpu
            .create_queue()
            .unwrap()
            .write_buffer(&mut buffer, 0, &encode(values))
            .unwrap();
        buffer
    }

    fn wgpu_buffer(&self, values: &[f32]) -> WgpuBuffer {
        let desc =
            BufferDesc::new(values.len() as u64 * 4, usages(), MemoryClass::DeviceLocal).unwrap();
        let mut buffer = self.wgpu.create_buffer(desc).unwrap();
        self.wgpu
            .create_queue()
            .unwrap()
            .write_buffer(&mut buffer, 0, &encode(values))
            .unwrap();
        buffer
    }

    fn cpu_values(&self, buffer: &CpuBuffer) -> Vec<f32> {
        let queue = self.cpu.create_queue().unwrap();
        decode(&queue.read_buffer(buffer, 0, buffer.len() as u64).unwrap())
    }

    fn wgpu_values(&self, buffer: &WgpuBuffer) -> Vec<f32> {
        let queue = self.wgpu.create_queue().unwrap();
        decode(&queue.read_buffer(buffer, 0, buffer.len()).unwrap())
    }
}

fn cpu_kernels() -> CpuF32KernelsV1 {
    CpuF32KernelsV1::new(1 << 24).unwrap()
}

/// Deterministic normal-range values of mixed sign and magnitude.
fn values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let exponent = 112 + (state >> 27); // 2^-15 .. 2^16
            let bits = (state & 0x8000_0000) | (exponent << 23) | (state & 0x007f_ffff);
            f32::from_bits(bits)
        })
        .collect()
}

fn kind(error: &PortableError) -> &'static str {
    match error {
        PortableError::InvalidDescriptor(_) => "InvalidDescriptor",
        PortableError::OutOfBounds { .. } => "OutOfBounds",
        PortableError::Unsupported(_) => "Unsupported",
        PortableError::Backend(_) => "Backend",
    }
}

#[test]
fn binary_add_and_multiply_are_bit_exact() {
    let Some(wgpu) = adapter_or_skip("binary") else {
        return;
    };
    let pair = Pair::new(wgpu);
    let kernels = WgpuF32KernelsV1::new(wgpu).unwrap();
    let mut left = vec![0.0, -0.0, 1.0, 16_777_216.0, 1.5, -2.5];
    let mut right = vec![-0.0, -0.0, f32::EPSILON / 2.0, 1.0, -1.5, 4.0];
    left.extend(values(994, 7));
    right.extend(values(994, 11));
    for (cpu_op, wgpu_op, identity) in [
        (
            CpuF32BinaryOp::Add,
            WgpuF32BinaryOp::Add,
            WgpuF32Operation::Add,
        ),
        (
            CpuF32BinaryOp::Multiply,
            WgpuF32BinaryOp::Multiply,
            WgpuF32Operation::Multiply,
        ),
    ] {
        let mut cpu_out = pair.cpu_buffer(&vec![0.0; left.len()]);
        cpu_kernels()
            .binary(
                cpu_op,
                &pair.cpu_buffer(&left),
                &pair.cpu_buffer(&right),
                &mut cpu_out,
            )
            .unwrap();
        let mut out = pair.wgpu_buffer(&vec![0.0; left.len()]);
        let report = kernels
            .binary(
                wgpu_op,
                &pair.wgpu_buffer(&left),
                &pair.wgpu_buffer(&right),
                &mut out,
            )
            .unwrap();
        assert_eq!(report.operation, identity);
        assert_eq!(report.output_values, left.len());
        let expected = pair.cpu_values(&cpu_out);
        assert!(expected.iter().all(|v| *v == 0.0 || v.is_normal()));
        assert_eq!(
            bits(&pair.wgpu_values(&out)),
            bits(&expected),
            "{identity:?}"
        );
    }
}

#[test]
fn relu_and_gather_are_bitwise_exact_including_subnormals() {
    let Some(wgpu) = adapter_or_skip("relu_gather") else {
        return;
    };
    let pair = Pair::new(wgpu);
    let kernels = WgpuF32KernelsV1::new(wgpu).unwrap();
    let mut input = vec![
        0.0,
        -0.0,
        -1.0,
        2.5,
        f32::from_bits(1),
        -f32::from_bits(5),
        f32::MAX,
    ];
    input.extend(values(193, 3));

    let mut cpu_out = pair.cpu_buffer(&vec![9.0; input.len()]);
    cpu_kernels()
        .relu(&pair.cpu_buffer(&input), &mut cpu_out)
        .unwrap();
    let mut out = pair.wgpu_buffer(&vec![9.0; input.len()]);
    kernels.relu(&pair.wgpu_buffer(&input), &mut out).unwrap();
    assert_eq!(
        bits(&pair.wgpu_values(&out)),
        bits(&pair.cpu_values(&cpu_out))
    );

    let indices: Vec<usize> = (0..150)
        .map(|i| (i * 37 + 5) % input.len())
        .chain([4, 4, 1, 5])
        .collect();
    let mut cpu_out = pair.cpu_buffer(&vec![0.0; indices.len()]);
    cpu_kernels()
        .gather(&pair.cpu_buffer(&input), &indices, &mut cpu_out)
        .unwrap();
    let mut out = pair.wgpu_buffer(&vec![0.0; indices.len()]);
    kernels
        .gather(&pair.wgpu_buffer(&input), &indices, &mut out)
        .unwrap();
    assert_eq!(
        bits(&pair.wgpu_values(&out)),
        bits(&pair.cpu_values(&cpu_out))
    );
}

#[test]
fn serial_sum_and_scatter_add_are_bit_exact() {
    let Some(wgpu) = adapter_or_skip("sum_scatter") else {
        return;
    };
    let pair = Pair::new(wgpu);
    let kernels = WgpuF32KernelsV1::new(wgpu).unwrap();
    for input in [
        vec![-0.0],
        vec![-0.0, -0.0, -0.0],
        vec![16_777_216.0, 1.0, 1.0, -16_777_216.0],
        vec![3.0e38, -3.0e38, 1.0],
        values(1000, 19),
    ] {
        let mut cpu_out = pair.cpu_buffer(&[7.0]);
        cpu_kernels()
            .sum(&pair.cpu_buffer(&input), &mut cpu_out)
            .unwrap();
        let mut out = pair.wgpu_buffer(&[7.0]);
        kernels.sum(&pair.wgpu_buffer(&input), &mut out).unwrap();
        assert_eq!(
            bits(&pair.wgpu_values(&out)),
            bits(&pair.cpu_values(&cpu_out)),
            "sum of {} values",
            input.len()
        );
    }

    let destination = values(40, 23);
    let source = values(300, 29);
    let indices: Vec<usize> = (0..300).map(|i| (i * i + 3) % 31).collect();
    let mut cpu_out = pair.cpu_buffer(&destination);
    cpu_kernels()
        .scatter_add(&pair.cpu_buffer(&source), &indices, &mut cpu_out)
        .unwrap();
    let mut out = pair.wgpu_buffer(&destination);
    kernels
        .scatter_add(&pair.wgpu_buffer(&source), &indices, &mut out)
        .unwrap();
    let expected = pair.cpu_values(&cpu_out);
    assert_eq!(bits(&pair.wgpu_values(&out)), bits(&expected));
    assert_eq!(bits(&expected[31..]), bits(&destination[31..]));
}

#[test]
fn projection_is_within_declared_bound() {
    let Some(wgpu) = adapter_or_skip("project_kn") else {
        return;
    };
    let pair = Pair::new(wgpu);
    let kernels = WgpuF32KernelsV1::new(wgpu).unwrap();
    let (k, n) = (37, 70);
    let input = values(k, 31);
    let weights = values(k * n, 37);
    let mut cpu_out = pair.cpu_buffer(&vec![0.0; n]);
    cpu_kernels()
        .project_kn(
            &pair.cpu_buffer(&input),
            &pair.cpu_buffer(&weights),
            &mut cpu_out,
            k,
            n,
        )
        .unwrap();
    let mut out = pair.wgpu_buffer(&vec![0.0; n]);
    kernels
        .project_kn(
            &pair.wgpu_buffer(&input),
            &pair.wgpu_buffer(&weights),
            &mut out,
            k,
            n,
        )
        .unwrap();
    let expected = pair.cpu_values(&cpu_out);
    let actual = pair.wgpu_values(&out);
    let unit = 2f64.powi(-24);
    let mut exact = 0;
    for column in 0..n {
        let magnitude: f64 = (0..k)
            .map(|row| (f64::from(input[row]) * f64::from(weights[row * n + column])).abs())
            .sum();
        let bound = (3 * k + 1) as f64 * unit * magnitude;
        let difference = (f64::from(actual[column]) - f64::from(expected[column])).abs();
        assert!(
            difference <= bound,
            "column {column}: wgpu {} cpu {} diff {difference} bound {bound}",
            actual[column],
            expected[column]
        );
        exact += usize::from(actual[column].to_bits() == expected[column].to_bits());
    }
    let _ = writeln!(
        std::io::stderr(),
        "nnis-wgpu project_kn: {exact}/{n} outputs bit-identical to the fused CPU reference \
         (declared bound, not bit-exactness, is the contract)"
    );
}

#[test]
fn errors_match_cpu_variants_and_preserve_destinations() {
    let Some(wgpu) = adapter_or_skip("errors") else {
        return;
    };
    let pair = Pair::new(wgpu);
    let kernels = WgpuF32KernelsV1::new(wgpu).unwrap();
    let cpu = cpu_kernels();
    let sentinel = [5.0, 6.0, 7.0];

    // Non-finite unselected gather input.
    let input = [1.0, f32::NAN, 2.0];
    let mut cpu_out = pair.cpu_buffer(&sentinel[..2]);
    let cpu_error = cpu
        .gather(&pair.cpu_buffer(&input), &[0, 2], &mut cpu_out)
        .unwrap_err();
    let mut out = pair.wgpu_buffer(&sentinel[..2]);
    let error = kernels
        .gather(&pair.wgpu_buffer(&input), &[0, 2], &mut out)
        .unwrap_err();
    assert_eq!(kind(&error), kind(&cpu_error));
    assert_eq!(pair.wgpu_values(&out), sentinel[..2]);

    // Overflowing intermediate in sum, add, projection and scatter-add.
    let big = [3.0e38, 3.0e38, -3.0e38];
    let mut out = pair.wgpu_buffer(&sentinel[..1]);
    let error = kernels.sum(&pair.wgpu_buffer(&big), &mut out).unwrap_err();
    let mut cpu_out = pair.cpu_buffer(&sentinel[..1]);
    assert_eq!(
        kind(&error),
        kind(&cpu.sum(&pair.cpu_buffer(&big), &mut cpu_out).unwrap_err())
    );
    assert_eq!(pair.wgpu_values(&out), sentinel[..1]);

    let mut out = pair.wgpu_buffer(&sentinel);
    let error = kernels
        .binary(
            WgpuF32BinaryOp::Add,
            &pair.wgpu_buffer(&big),
            &pair.wgpu_buffer(&big),
            &mut out,
        )
        .unwrap_err();
    assert_eq!(kind(&error), "InvalidDescriptor");
    assert_eq!(pair.wgpu_values(&out), sentinel);

    let mut out = pair.wgpu_buffer(&sentinel[..1]);
    let error = kernels
        .project_kn(
            &pair.wgpu_buffer(&big),
            &pair.wgpu_buffer(&[1.0, 1.0, 1.0]),
            &mut out,
            3,
            1,
        )
        .unwrap_err();
    assert_eq!(kind(&error), "InvalidDescriptor");
    assert_eq!(pair.wgpu_values(&out), sentinel[..1]);

    let mut out = pair.wgpu_buffer(&sentinel);
    let error = kernels
        .scatter_add(
            &pair.wgpu_buffer(&[1.0, 3.0e38, 3.0e38]),
            &[0, 1, 1],
            &mut out,
        )
        .unwrap_err();
    assert_eq!(kind(&error), "InvalidDescriptor");
    assert_eq!(
        pair.wgpu_values(&out),
        sentinel,
        "partial scatter must not commit"
    );

    // Non-finite old scatter destination.
    let mut out = pair.wgpu_buffer(&[1.0, f32::INFINITY]);
    let error = kernels
        .scatter_add(&pair.wgpu_buffer(&[1.0]), &[0], &mut out)
        .unwrap_err();
    assert_eq!(kind(&error), "InvalidDescriptor");

    // Structural errors.
    let three = pair.wgpu_buffer(&sentinel);
    let mut two = pair.wgpu_buffer(&sentinel[..2]);
    assert_eq!(
        kind(&kernels.relu(&three, &mut two).unwrap_err()),
        kind(
            &cpu.relu(
                &pair.cpu_buffer(&sentinel),
                &mut pair.cpu_buffer(&sentinel[..2])
            )
            .unwrap_err()
        )
    );
    assert_eq!(
        kind(&kernels.gather(&three, &[3], &mut two).unwrap_err()),
        "InvalidDescriptor"
    );
    assert_eq!(
        kind(&kernels.gather(&three, &[], &mut two).unwrap_err()),
        "InvalidDescriptor"
    );
    let mut one = pair.wgpu_buffer(&[0.0]);
    assert_eq!(
        kind(
            &kernels
                .project_kn(&three, &three, &mut one, 0, 1)
                .unwrap_err()
        ),
        "InvalidDescriptor"
    );
    let copy_only = BufferDesc::new(
        12,
        BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        MemoryClass::Shared,
    )
    .unwrap();
    let copy_only = wgpu.create_buffer(copy_only).unwrap();
    assert_eq!(
        kind(&kernels.relu(&copy_only, &mut two).unwrap_err()),
        "Unsupported"
    );
    let odd = BufferDesc::new(6, usages(), MemoryClass::Shared).unwrap();
    let odd = wgpu.create_buffer(odd).unwrap();
    assert_eq!(
        kind(&kernels.sum(&odd, &mut one).unwrap_err()),
        "InvalidDescriptor"
    );
    assert_eq!(pair.wgpu_values(&two), sentinel[..2]);
}
