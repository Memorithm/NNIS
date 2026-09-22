//! Typed runtime graph, not a trained model or performance benchmark.

use nnis_core::graph::{
    F32GraphLimitsV1, F32GraphV1, F32NodeV1, F32OpV1, F32ShapeV1,
    F32_GRAPH_POLICY, F32_GRAPH_VERSION,
};
use nnis_core::{BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableError, PortableQueue, Result};
use nnis_cpu::graph::execute_f32_graph;
use nnis_cpu::{CpuBuffer, CpuDevice};

fn tensor(device: &CpuDevice, values: &[f32]) -> Result<CpuBuffer> {
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let data: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    let mut buffer = device.create_buffer(BufferDesc::new(data.len() as u64, usage, MemoryClass::Host)?)?;
    device.create_queue()?.write_buffer(&mut buffer, 0, &data)?;
    Ok(buffer)
}

fn run() -> Result<()> {
    let inputs = [
        F32ShapeV1::Vector(3), F32ShapeV1::Matrix { rows: 3, cols: 2 },
        F32ShapeV1::Vector(2), F32ShapeV1::Matrix { rows: 2, cols: 2 },
        F32ShapeV1::Vector(2),
    ];
    let v2 = F32ShapeV1::Vector(2);
    let nodes = [
        F32NodeV1 { operation: F32OpV1::ProjectKn { input: 0, weights: 1 }, output: v2 },
        F32NodeV1 { operation: F32OpV1::Add { left: 5, right: 2 }, output: v2 },
        F32NodeV1 { operation: F32OpV1::Relu { input: 6 }, output: v2 },
        F32NodeV1 { operation: F32OpV1::ProjectKn { input: 7, weights: 3 }, output: v2 },
        F32NodeV1 { operation: F32OpV1::Add { left: 8, right: 4 }, output: v2 },
        F32NodeV1 { operation: F32OpV1::Gather { input: 9, indices: &[1, 0, 1] }, output: F32ShapeV1::Vector(3) },
        F32NodeV1 { operation: F32OpV1::Sum { input: 10 }, output: F32ShapeV1::Vector(1) },
    ];
    let plan = F32GraphV1 {
        schema_version: F32_GRAPH_VERSION, numerical_policy: F32_GRAPH_POLICY,
        inputs: &inputs, nodes: &nodes,
    };
    let limits = F32GraphLimitsV1 {
        max_inputs: 5, max_nodes: 7, max_tensor_bytes: 24,
        max_live_payload_bytes: 136, max_scratch_bytes: 12, max_work_items: 1000,
    };
    if plan.validate(F32GraphLimitsV1 { max_live_payload_bytes: 135, ..limits }).is_ok() {
        return Err(PortableError::Backend("undersized graph budget accepted".to_string()));
    }
    let graph = plan.validate(limits)?;
    let device = CpuDevice::with_max_buffer_bytes(24)?;
    let x = tensor(&device, &[2.0, -1.0, 3.0])?;
    let w1 = tensor(&device, &[1.0, 2.0, -1.0, 4.0, 0.5, -2.0])?;
    let b1 = tensor(&device, &[0.5, 1.0])?;
    let w2 = tensor(&device, &[2.0, -1.0, 4.0, 3.0])?;
    let b2 = tensor(&device, &[1.0, 2.0])?;
    let result = execute_f32_graph(&device, &graph, &[&x, &w1, &b1, &w2, &b2])?;
    let data = device.create_queue()?.read_buffer(&result.output, 0, 4)?;
    if data != 5.0_f32.to_le_bytes() {
        return Err(PortableError::Backend("graph plan output mismatch".to_string()));
    }
    println!(
        "CPU_GRAPH_PLAN_OK version={} nodes={} result=5 live_payload_bound_bytes={} retained_node_capacity_bytes={} max_scratch_capacity_bytes={}",
        result.report.schema_version, result.report.executed_nodes,
        result.report.budget.live_payload_bound_bytes,
        result.report.retained_node_capacity_bytes, result.report.max_scratch_capacity_bytes,
    );
    Ok(())
}

fn main() -> Result<()> {
    run()
}

#[cfg(test)]
mod tests {
    #[test]
    fn graph_plan_matches_analytical_result() {
        super::run().unwrap();
    }
}
