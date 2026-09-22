use nnis_core::graph::{
    F32GraphLimitsV1, F32GraphV1, F32NodeV1, F32OpV1, F32ShapeV1, F32_GRAPH_POLICY,
    F32_GRAPH_VERSION,
};
use nnis_core::{BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableQueue};
use nnis_cpu::graph::execute_f32_graph;
use nnis_cpu::{CpuBuffer, CpuDevice};

fn limits() -> F32GraphLimitsV1 {
    F32GraphLimitsV1 {
        max_inputs: 16,
        max_nodes: 32,
        max_tensor_bytes: 1024,
        max_live_payload_bytes: 8192,
        max_scratch_bytes: 1024,
        max_work_items: 100000,
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

fn tensor(values: &[f32]) -> CpuBuffer {
    let device = CpuDevice::with_max_buffer_bytes(1024).unwrap();
    let data: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let mut buffer = device
        .create_buffer(BufferDesc::new(data.len() as u64, usage, MemoryClass::Host).unwrap())
        .unwrap();
    device
        .create_queue()
        .unwrap()
        .write_buffer(&mut buffer, 0, &data)
        .unwrap();
    buffer
}

fn bytes(buffer: &CpuBuffer) -> Vec<u8> {
    CpuDevice::new()
        .unwrap()
        .create_queue()
        .unwrap()
        .read_buffer(buffer, 0, buffer.len() as u64)
        .unwrap()
}

fn assert_values(buffer: &CpuBuffer, expected: &[f32]) {
    let expected: Vec<u8> = expected
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(bytes(buffer), expected);
}

#[test]
fn validated_seven_node_graph_executes_and_can_be_reused_with_new_bindings() {
    let shapes = [
        F32ShapeV1::Vector(3),
        F32ShapeV1::Matrix { rows: 3, cols: 2 },
        F32ShapeV1::Vector(2),
        F32ShapeV1::Matrix { rows: 2, cols: 2 },
        F32ShapeV1::Vector(2),
    ];
    let nodes = [
        F32NodeV1 {
            operation: F32OpV1::ProjectKn {
                input: 0,
                weights: 1,
            },
            output: F32ShapeV1::Vector(2),
        },
        F32NodeV1 {
            operation: F32OpV1::Add { left: 5, right: 2 },
            output: F32ShapeV1::Vector(2),
        },
        F32NodeV1 {
            operation: F32OpV1::Relu { input: 6 },
            output: F32ShapeV1::Vector(2),
        },
        F32NodeV1 {
            operation: F32OpV1::ProjectKn {
                input: 7,
                weights: 3,
            },
            output: F32ShapeV1::Vector(2),
        },
        F32NodeV1 {
            operation: F32OpV1::Add { left: 8, right: 4 },
            output: F32ShapeV1::Vector(2),
        },
        F32NodeV1 {
            operation: F32OpV1::Gather {
                input: 9,
                indices: &[1, 0, 1],
            },
            output: F32ShapeV1::Vector(3),
        },
        F32NodeV1 {
            operation: F32OpV1::Sum { input: 10 },
            output: F32ShapeV1::Vector(1),
        },
    ];
    let valid = plan(&shapes, &nodes).validate(limits()).unwrap();
    assert_eq!(valid.budget().input_payload_bytes, 68);
    assert_eq!(valid.budget().node_payload_bytes, 56);
    assert_eq!(valid.budget().max_scratch_payload_bytes, 12);
    assert_eq!(valid.budget().live_payload_bound_bytes, 136);
    let weights1 = tensor(&[1.0, 2.0, -1.0, 4.0, 0.5, -2.0]);
    let bias1 = tensor(&[0.5, 1.0]);
    let weights2 = tensor(&[2.0, -1.0, 4.0, 3.0]);
    let bias2 = tensor(&[1.0, 2.0]);
    let device = CpuDevice::new().unwrap();
    for (input_values, expected) in [([2.0, -1.0, 3.0], 5.0), ([0.0; 3], 15.0)] {
        let input = tensor(&input_values);
        let result = execute_f32_graph(
            &device,
            &valid,
            &[&input, &weights1, &bias1, &weights2, &bias2],
        )
        .unwrap();
        assert_values(&result.output, &[expected]);
        assert_values(&input, &input_values);
        assert_eq!(result.report.executed_nodes, 7);
        assert_eq!(result.report.budget, valid.budget());
        assert!(result.report.retained_node_capacity_bytes >= 56);
        assert!(result.report.max_scratch_capacity_bytes >= 12);
        assert!(result.report.node_handle_capacity_bytes >= 7 * core::mem::size_of::<CpuBuffer>());
    }
    assert_values(&weights1, &[1.0, 2.0, -1.0, 4.0, 0.5, -2.0]);
}

#[test]
fn scatter_add_creates_a_new_value_and_preserves_base_and_source() {
    let shapes = [F32ShapeV1::Vector(2), F32ShapeV1::Vector(3)];
    let nodes = [F32NodeV1 {
        operation: F32OpV1::ScatterAdd {
            base: 0,
            source: 1,
            indices: &[1, 1, 0],
        },
        output: F32ShapeV1::Vector(2),
    }];
    let graph = plan(&shapes, &nodes).validate(limits()).unwrap();
    let base = tensor(&[10.0, 20.0]);
    let source = tensor(&[1.0, 2.0, 3.0]);
    let result = execute_f32_graph(&CpuDevice::new().unwrap(), &graph, &[&base, &source]).unwrap();
    assert_values(&result.output, &[13.0, 23.0]);
    assert_values(&base, &[10.0, 20.0]);
    assert_values(&source, &[1.0, 2.0, 3.0]);
}

#[test]
fn branched_dataflow_can_read_one_prior_value_multiple_times() {
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
    let input = tensor(&[-2.0, 3.0]);
    let result = execute_f32_graph(&CpuDevice::new().unwrap(), &graph, &[&input]).unwrap();
    assert_values(&result.output, &[0.0, 12.0]);
    assert_values(&input, &[-2.0, 3.0]);
}

#[test]
fn late_arithmetic_failure_returns_no_result_and_preserves_all_inputs() {
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
    let input = tensor(&[1.0, f32::MAX]);
    let before = bytes(&input);
    let capacity = input.capacity_bytes();
    let result = execute_f32_graph(&CpuDevice::new().unwrap(), &graph, &[&input]);
    assert!(result.is_err());
    assert_eq!(bytes(&input), before);
    assert_eq!(input.capacity_bytes(), capacity);
    let retry = tensor(&[1.0, 2.0]);
    let result = execute_f32_graph(&CpuDevice::new().unwrap(), &graph, &[&retry]).unwrap();
    assert_values(&result.output, &[2.0, 4.0]);
}

#[test]
fn binding_count_length_usage_and_unused_nonfinite_inputs_are_rejected() {
    let shapes = [F32ShapeV1::Vector(1), F32ShapeV1::Vector(1)];
    let nodes = [F32NodeV1 {
        operation: F32OpV1::Relu { input: 0 },
        output: shapes[0],
    }];
    let graph = plan(&shapes, &nodes).validate(limits()).unwrap();
    let device = CpuDevice::new().unwrap();
    let input = tensor(&[1.0]);
    assert!(execute_f32_graph(&device, &graph, &[&input]).is_err());
    assert!(execute_f32_graph(&device, &graph, &[&input, &tensor(&[1.0, 2.0])]).is_err());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(execute_f32_graph(&device, &graph, &[&input, &tensor(&[bad])]).is_err());
    }
    let no_storage = device
        .create_buffer(BufferDesc::new(4, BufferUsages::COPY_SRC, MemoryClass::Host).unwrap())
        .unwrap();
    assert!(execute_f32_graph(&device, &graph, &[&input, &no_storage]).is_err());
    assert_values(&input, &[1.0]);
}

#[test]
fn backend_capabilities_are_rechecked_after_portable_validation() {
    let shapes = [F32ShapeV1::Vector(2)];
    let nodes = [F32NodeV1 {
        operation: F32OpV1::Relu { input: 0 },
        output: shapes[0],
    }];
    let graph = plan(&shapes, &nodes).validate(limits()).unwrap();
    let input = tensor(&[1.0, 2.0]);
    assert!(execute_f32_graph(
        &CpuDevice::with_max_buffer_bytes(7).unwrap(),
        &graph,
        &[&input]
    )
    .is_err());
    let result = execute_f32_graph(
        &CpuDevice::with_max_buffer_bytes(8).unwrap(),
        &graph,
        &[&input],
    )
    .unwrap();
    assert_values(&result.output, &[1.0, 2.0]);
}

#[test]
fn failed_unused_node_is_not_silently_eliminated() {
    let shapes = [F32ShapeV1::Vector(1)];
    let nodes = [
        F32NodeV1 {
            operation: F32OpV1::Multiply { left: 0, right: 0 },
            output: shapes[0],
        },
        F32NodeV1 {
            operation: F32OpV1::Relu { input: 0 },
            output: shapes[0],
        },
    ];
    let graph = plan(&shapes, &nodes).validate(limits()).unwrap();
    let input = tensor(&[f32::MAX]);
    assert!(execute_f32_graph(&CpuDevice::new().unwrap(), &graph, &[&input]).is_err());
    assert_values(&input, &[f32::MAX]);
}

#[test]
fn aliased_input_bindings_are_read_only_and_conservatively_charged() {
    let shapes = [F32ShapeV1::Vector(1); 2];
    let nodes = [F32NodeV1 {
        operation: F32OpV1::Add { left: 0, right: 1 },
        output: shapes[0],
    }];
    let graph = plan(&shapes, &nodes).validate(limits()).unwrap();
    let input = tensor(&[3.0]);
    let result = execute_f32_graph(&CpuDevice::new().unwrap(), &graph, &[&input, &input]).unwrap();
    assert_values(&result.output, &[6.0]);
    assert_values(&input, &[3.0]);
    assert_eq!(result.report.budget.input_payload_bytes, 8);
}
