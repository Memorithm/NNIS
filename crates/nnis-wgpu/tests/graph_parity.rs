//! Portable F32 graph execution on WGPU compared with the CPU graph reference.
//!
//! Without an adapter each test writes an explicit SKIP to stderr and passes;
//! that pass is not evidence. Software adapters exercise the API path only and
//! are not hardware evidence. No timing is measured.

use std::io::Write;
use std::sync::OnceLock;

use nnis_core::graph::{
    F32GraphLimitsV1, F32GraphV1, F32NodeV1, F32OpV1, F32ShapeV1, ValidatedF32GraphV1,
    F32_GRAPH_POLICY, F32_GRAPH_VERSION,
};
use nnis_core::{
    BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableError, PortableQueue,
};
use nnis_cpu::graph::execute_f32_graph as cpu_execute;
use nnis_cpu::numerical::{CpuF32BinaryOp, CpuF32KernelsV1};
use nnis_cpu::{CpuBuffer, CpuDevice};
use nnis_wgpu::graph::{execute_f32_graph, execute_f32_graph_traced};
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

fn limits() -> F32GraphLimitsV1 {
    F32GraphLimitsV1 {
        max_inputs: 16,
        max_nodes: 32,
        max_tensor_bytes: 1 << 20,
        max_live_payload_bytes: 1 << 24,
        max_scratch_bytes: 1 << 20,
        max_work_items: 1 << 24,
    }
}

fn plan<'a>(inputs: &'a [F32ShapeV1], nodes: &'a [F32NodeV1<'a>]) -> F32GraphV1<'a> {
    F32GraphV1 {
        schema_version: F32_GRAPH_VERSION,
        numerical_policy: F32_GRAPH_POLICY,
        inputs,
        nodes,
    }
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

fn cpu_tensor(values: &[f32]) -> CpuBuffer {
    let device = CpuDevice::new().unwrap();
    let desc = BufferDesc::new(values.len() as u64 * 4, usages(), MemoryClass::Host).unwrap();
    let mut buffer = device.create_buffer(desc).unwrap();
    device
        .create_queue()
        .unwrap()
        .write_buffer(&mut buffer, 0, &encode(values))
        .unwrap();
    buffer
}

fn cpu_values(buffer: &CpuBuffer) -> Vec<f32> {
    let queue = CpuDevice::new().unwrap().create_queue().unwrap();
    decode(&queue.read_buffer(buffer, 0, buffer.len() as u64).unwrap())
}

fn wgpu_tensor(device: &WgpuDevice, values: &[f32]) -> WgpuBuffer {
    let desc =
        BufferDesc::new(values.len() as u64 * 4, usages(), MemoryClass::DeviceLocal).unwrap();
    let mut buffer = device.create_buffer(desc).unwrap();
    device
        .create_queue()
        .unwrap()
        .write_buffer(&mut buffer, 0, &encode(values))
        .unwrap();
    buffer
}

fn wgpu_values(device: &WgpuDevice, buffer: &WgpuBuffer) -> Vec<f32> {
    decode(
        &device
            .create_queue()
            .unwrap()
            .read_buffer(buffer, 0, buffer.len())
            .unwrap(),
    )
}

fn values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let exponent = 118 + (state >> 28); // 2^-9 .. 2^6
            f32::from_bits((state & 0x8000_0000) | (exponent << 23) | (state & 0x007f_ffff))
        })
        .collect()
}

/// Run on both backends; return (cpu output, wgpu output).
fn run_both(
    device: &WgpuDevice,
    graph: &ValidatedF32GraphV1<'_>,
    inputs: &[Vec<f32>],
) -> (Vec<f32>, Vec<f32>) {
    let cpu_inputs: Vec<CpuBuffer> = inputs.iter().map(|v| cpu_tensor(v)).collect();
    let cpu_refs: Vec<&CpuBuffer> = cpu_inputs.iter().collect();
    let cpu = cpu_execute(&CpuDevice::new().unwrap(), graph, &cpu_refs).unwrap();
    let wgpu_inputs: Vec<WgpuBuffer> = inputs.iter().map(|v| wgpu_tensor(device, v)).collect();
    let wgpu_refs: Vec<&WgpuBuffer> = wgpu_inputs.iter().collect();
    let wgpu = execute_f32_graph(device, graph, &wgpu_refs).unwrap();
    assert_eq!(wgpu.report.executed_nodes, graph.plan().nodes.len());
    assert_eq!(wgpu.report.budget, graph.budget());
    assert_eq!(
        wgpu.report.node_artifact_fingerprints.len(),
        graph.plan().nodes.len()
    );
    for (buffer, expected) in wgpu_inputs.iter().zip(inputs) {
        assert_eq!(
            bits(&wgpu_values(device, buffer)),
            bits(expected),
            "input mutated"
        );
    }
    (cpu_values(&cpu.output), wgpu_values(device, &wgpu.output))
}

#[test]
fn projection_free_graphs_are_bit_exact_end_to_end() {
    let Some(device) = adapter_or_skip("graph_projection_free") else {
        return;
    };
    // Branched dataflow reading one value several times.
    let shapes = [F32ShapeV1::Vector(2)];
    let nodes = [
        F32NodeV1 {
            operation: F32OpV1::Relu { input: 0 },
            output: shapes[0],
        },
        F32NodeV1 {
            operation: F32OpV1::Multiply { left: 1, right: 1 },
            output: shapes[0],
        },
        F32NodeV1 {
            operation: F32OpV1::Add { left: 1, right: 2 },
            output: shapes[0],
        },
    ];
    let graph = plan(&shapes, &nodes).validate(limits()).unwrap();
    let (cpu, wgpu) = run_both(device, &graph, &[vec![-2.0, 3.0]]);
    assert_eq!(bits(&wgpu), bits(&cpu));
    assert_eq!(cpu, [0.0, 12.0]);

    // Every non-projection operation on 300-element vectors.
    let n = 300;
    let gather: Vec<usize> = (0..200).map(|i| (i * 7 + 3) % n).collect();
    let scatter: Vec<usize> = (0..200).map(|i| (i * i + 1) % 150).collect();
    let shapes = [
        F32ShapeV1::Vector(n as u64),
        F32ShapeV1::Vector(n as u64),
        F32ShapeV1::Vector(150),
    ];
    let nodes = [
        F32NodeV1 {
            operation: F32OpV1::Add { left: 0, right: 1 },
            output: shapes[0],
        },
        F32NodeV1 {
            operation: F32OpV1::Multiply { left: 3, right: 1 },
            output: shapes[0],
        },
        F32NodeV1 {
            operation: F32OpV1::Relu { input: 4 },
            output: shapes[0],
        },
        F32NodeV1 {
            operation: F32OpV1::Gather {
                input: 5,
                indices: &gather,
            },
            output: F32ShapeV1::Vector(200),
        },
        F32NodeV1 {
            operation: F32OpV1::ScatterAdd {
                base: 2,
                source: 6,
                indices: &scatter,
            },
            output: F32ShapeV1::Vector(150),
        },
        F32NodeV1 {
            operation: F32OpV1::Sum { input: 7 },
            output: F32ShapeV1::Vector(1),
        },
    ];
    let graph = plan(&shapes, &nodes).validate(limits()).unwrap();
    let inputs = [values(n, 3), values(n, 5), values(150, 7)];
    let (cpu, wgpu) = run_both(device, &graph, &inputs);
    assert!(cpu[0].is_normal());
    assert_eq!(bits(&wgpu), bits(&cpu));
}

fn cpu_node(op: F32OpV1<'_>, operand: &dyn Fn(usize) -> Vec<f32>, output_len: usize) -> Vec<f32> {
    let kernels = CpuF32KernelsV1::new(1 << 24).unwrap();
    let mut out = cpu_tensor(&vec![0.0; output_len]);
    match op {
        F32OpV1::Add { left, right } => kernels
            .binary(
                CpuF32BinaryOp::Add,
                &cpu_tensor(&operand(left)),
                &cpu_tensor(&operand(right)),
                &mut out,
            )
            .map(|_| ()),
        F32OpV1::Multiply { left, right } => kernels
            .binary(
                CpuF32BinaryOp::Multiply,
                &cpu_tensor(&operand(left)),
                &cpu_tensor(&operand(right)),
                &mut out,
            )
            .map(|_| ()),
        F32OpV1::Relu { input } => kernels
            .relu(&cpu_tensor(&operand(input)), &mut out)
            .map(|_| ()),
        F32OpV1::Sum { input } => kernels
            .sum(&cpu_tensor(&operand(input)), &mut out)
            .map(|_| ()),
        F32OpV1::ProjectKn { input, weights } => {
            let x = operand(input);
            kernels
                .project_kn(
                    &cpu_tensor(&x),
                    &cpu_tensor(&operand(weights)),
                    &mut out,
                    x.len(),
                    output_len,
                )
                .map(|_| ())
        }
        F32OpV1::Gather { input, indices } => kernels
            .gather(&cpu_tensor(&operand(input)), indices, &mut out)
            .map(|_| ()),
        F32OpV1::ScatterAdd {
            base,
            source,
            indices,
        } => {
            out = cpu_tensor(&operand(base));
            kernels
                .scatter_add(&cpu_tensor(&operand(source)), indices, &mut out)
                .map(|_| ())
        }
    }
    .unwrap();
    cpu_values(&out)
}

#[test]
fn graphs_with_projections_meet_per_node_declared_tolerances() {
    let Some(device) = adapter_or_skip("graph_projection") else {
        return;
    };
    let (k, h) = (24, 40);
    let shapes = [
        F32ShapeV1::Vector(k),
        F32ShapeV1::Matrix { rows: k, cols: h },
        F32ShapeV1::Vector(h),
        F32ShapeV1::Matrix { rows: h, cols: 8 },
        F32ShapeV1::Vector(8),
    ];
    let nodes = [
        F32NodeV1 {
            operation: F32OpV1::ProjectKn {
                input: 0,
                weights: 1,
            },
            output: F32ShapeV1::Vector(h),
        },
        F32NodeV1 {
            operation: F32OpV1::Add { left: 5, right: 2 },
            output: F32ShapeV1::Vector(h),
        },
        F32NodeV1 {
            operation: F32OpV1::Relu { input: 6 },
            output: F32ShapeV1::Vector(h),
        },
        F32NodeV1 {
            operation: F32OpV1::ProjectKn {
                input: 7,
                weights: 3,
            },
            output: F32ShapeV1::Vector(8),
        },
        F32NodeV1 {
            operation: F32OpV1::Add { left: 8, right: 4 },
            output: F32ShapeV1::Vector(8),
        },
        F32NodeV1 {
            operation: F32OpV1::Gather {
                input: 9,
                indices: &[1, 0, 7, 7, 3],
            },
            output: F32ShapeV1::Vector(5),
        },
        F32NodeV1 {
            operation: F32OpV1::Sum { input: 10 },
            output: F32ShapeV1::Vector(1),
        },
    ];
    let graph = plan(&shapes, &nodes).validate(limits()).unwrap();
    let inputs = [
        values(k as usize, 11),
        values((k * h) as usize, 13),
        values(h as usize, 17),
        values((h * 8) as usize, 19),
        values(8, 23),
    ];
    let wgpu_inputs: Vec<WgpuBuffer> = inputs.iter().map(|v| wgpu_tensor(device, v)).collect();
    let refs: Vec<&WgpuBuffer> = wgpu_inputs.iter().collect();
    let trace = execute_f32_graph_traced(device, &graph, &refs).unwrap();
    let node_values: Vec<Vec<f32>> = trace.nodes.iter().map(|b| wgpu_values(device, b)).collect();
    let operand = |id: usize| -> Vec<f32> {
        if id < inputs.len() {
            inputs[id].clone()
        } else {
            node_values[id - inputs.len()].clone()
        }
    };
    let unit = 2f64.powi(-24);
    for (index, node) in nodes.iter().enumerate() {
        let actual = &node_values[index];
        let expected = cpu_node(node.operation, &operand, actual.len());
        match node.operation {
            F32OpV1::ProjectKn { input, weights } => {
                let x = operand(input);
                let w = operand(weights);
                let n = actual.len();
                for column in 0..n {
                    let magnitude: f64 = (0..x.len())
                        .map(|row| (f64::from(x[row]) * f64::from(w[row * n + column])).abs())
                        .sum();
                    let bound = (3 * x.len() + 1) as f64 * unit * magnitude;
                    let difference =
                        (f64::from(actual[column]) - f64::from(expected[column])).abs();
                    assert!(
                        difference <= bound,
                        "node {index} column {column}: {difference} > {bound}"
                    );
                }
            }
            _ => assert_eq!(
                bits(actual),
                bits(&expected),
                "node {index} {:?}",
                node.operation
            ),
        }
    }
    // End-to-end CPU comparison is informational only for projection graphs.
    let cpu_inputs: Vec<CpuBuffer> = inputs.iter().map(|v| cpu_tensor(v)).collect();
    let cpu_refs: Vec<&CpuBuffer> = cpu_inputs.iter().collect();
    let cpu = cpu_values(
        &cpu_execute(&CpuDevice::new().unwrap(), &graph, &cpu_refs)
            .unwrap()
            .output,
    );
    let _ = writeln!(
        std::io::stderr(),
        "nnis-wgpu graph_projection: end-to-end wgpu {} vs cpu {} (informational; per-node tolerances are the contract)",
        node_values[6][0],
        cpu[0]
    );
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
fn graph_errors_match_cpu_and_preserve_inputs() {
    let Some(device) = adapter_or_skip("graph_errors") else {
        return;
    };
    let shapes = [F32ShapeV1::Vector(2)];
    let nodes = [
        F32NodeV1 {
            operation: F32OpV1::Relu { input: 0 },
            output: shapes[0],
        },
        F32NodeV1 {
            operation: F32OpV1::Add { left: 1, right: 1 },
            output: shapes[0],
        },
    ];
    let graph = plan(&shapes, &nodes).validate(limits()).unwrap();
    let cpu_device = CpuDevice::new().unwrap();
    let check = |values: &[f32]| {
        let cpu_error = cpu_execute(&cpu_device, &graph, &[&cpu_tensor(values)]).unwrap_err();
        let input = wgpu_tensor(device, values);
        let error = execute_f32_graph(device, &graph, &[&input]).unwrap_err();
        assert_eq!(kind(&error), kind(&cpu_error), "{values:?}");
        assert_eq!(bits(&wgpu_values(device, &input)), bits(values));
    };
    check(&[1.0, f32::MAX]); // late arithmetic overflow
    check(&[1.0, f32::NAN]); // non-finite input

    // Unused non-finite input is still rejected, as on the CPU.
    let two = [F32ShapeV1::Vector(2), F32ShapeV1::Vector(1)];
    let one_node = [F32NodeV1 {
        operation: F32OpV1::Relu { input: 0 },
        output: two[0],
    }];
    let graph2 = plan(&two, &one_node).validate(limits()).unwrap();
    let a = wgpu_tensor(device, &[1.0, 2.0]);
    let b = wgpu_tensor(device, &[f32::INFINITY]);
    let cpu_error = cpu_execute(
        &cpu_device,
        &graph2,
        &[&cpu_tensor(&[1.0, 2.0]), &cpu_tensor(&[f32::INFINITY])],
    )
    .unwrap_err();
    assert_eq!(
        kind(&execute_f32_graph(device, &graph2, &[&a, &b]).unwrap_err()),
        kind(&cpu_error)
    );

    // Binding count, byte length and usage.
    assert_eq!(
        kind(&execute_f32_graph(device, &graph2, &[&a]).unwrap_err()),
        "InvalidDescriptor"
    );
    let wrong = wgpu_tensor(device, &[1.0, 2.0, 3.0]);
    assert_eq!(
        kind(&execute_f32_graph(device, &graph, &[&wrong]).unwrap_err()),
        "InvalidDescriptor"
    );
    let copy_only = BufferDesc::new(
        8,
        BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        MemoryClass::Shared,
    )
    .unwrap();
    let copy_only = device.create_buffer(copy_only).unwrap();
    assert_eq!(
        kind(&execute_f32_graph(device, &graph, &[&copy_only]).unwrap_err()),
        "Unsupported"
    );
}
