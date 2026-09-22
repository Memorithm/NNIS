//! Retain-all-output scalar executor for the portable built-in F32 graph.
//!
//! No caller buffer is mutated. Intermediates stay private and are dropped on
//! every returned error. Only the final value and software accounting escape
//! after every node succeeds. This is not durable transaction rollback.

use nnis_core::graph::{
    F32GraphBudgetV1, F32OpV1, F32ShapeV1, ValidatedF32GraphV1, F32_GRAPH_POLICY,
    F32_GRAPH_VERSION,
};
use nnis_core::{
    BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableError, Result,
};

use crate::numerical::{CpuF32BinaryOp, CpuF32KernelsV1, CPU_F32_NUMERICAL_POLICY};
use crate::{require_usage, CpuBuffer, CpuDevice};

/// Successful graph evidence. Capacity values exclude allocator metadata, all
/// caller-owned inputs, OS residency and the caller's graph descriptor storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuF32GraphReportV1 {
    pub schema_version: u32,
    pub executed_nodes: usize,
    pub budget: F32GraphBudgetV1,
    /// Sum of retained node-buffer capacities just before returning final output.
    pub retained_node_capacity_bytes: u64,
    pub max_scratch_capacity_bytes: usize,
    /// Retained capacity of the internal buffer-handle vector, not tensor data.
    pub node_handle_capacity_bytes: usize,
}

/// The only escaping tensor is the final output. Other node buffers have been
/// dropped; recorded peaks refer to the just-completed execution.
#[derive(Debug)]
pub struct CpuF32GraphOutputV1 {
    pub output: CpuBuffer,
    pub report: CpuF32GraphReportV1,
}

/// Execute a statically validated graph with immutable external inputs.
///
/// All bindings, host representability, device per-buffer limits and all input
/// values are checked before any node allocation. Arithmetic failures can occur
/// later, but never expose a partially evaluated graph or change an input.
/// Allocation failures returned by Vec are errors; OS termination is outside
/// this contract. No buffer pool, graph rewrite or vendor backend is involved.
pub fn execute_f32_graph(
    device: &CpuDevice,
    graph: &ValidatedF32GraphV1<'_>,
    inputs: &[&CpuBuffer],
) -> Result<CpuF32GraphOutputV1> {
    let plan = graph.plan();
    if plan.numerical_policy != CPU_F32_NUMERICAL_POLICY
        || CPU_F32_NUMERICAL_POLICY != F32_GRAPH_POLICY {
        return Err(invalid("CPU graph numerical policy is unsupported"));
    }
    if inputs.len() != plan.inputs.len() {
        return Err(invalid("CPU graph binding count mismatch"));
    }
    for shape in plan.inputs.iter().copied().chain(plan.nodes.iter().map(|node| node.output)) {
        let bytes = host_bytes(shape)?;
        if bytes as u64 > device.capabilities().max_buffer_bytes {
            return Err(invalid("CPU graph tensor exceeds device per-buffer capability"));
        }
    }
    for (&shape, buffer) in plan.inputs.iter().zip(inputs) {
        require_usage(buffer, BufferUsages::STORAGE, "CPU graph input")?;
        if buffer.len() != host_bytes(shape)? {
            return Err(invalid("CPU graph input byte length mismatch"));
        }
        for word in buffer.bytes.chunks_exact(4) {
            if !f32::from_le_bytes([word[0], word[1], word[2], word[3]]).is_finite() {
                return Err(invalid("CPU graph input contains a non-finite value"));
            }
        }
    }
    let scratch = usize::try_from(graph.budget().max_scratch_payload_bytes)
        .map_err(|_| invalid("CPU graph scratch does not fit host address space"))?;
    let kernels = CpuF32KernelsV1::new(scratch)?;
    let mut computed = Vec::new();
    computed.try_reserve_exact(plan.nodes.len())
        .map_err(|_| invalid("CPU graph node-handle reservation failed"))?;
    let node_handle_capacity_bytes = computed.capacity().checked_mul(core::mem::size_of::<CpuBuffer>())
        .ok_or_else(|| invalid("CPU graph node-handle capacity overflows usize"))?;
    let mut retained_node_capacity_bytes = 0_u64;
    let mut max_scratch_capacity_bytes = 0_usize;

    for node in plan.nodes {
        let mut output = device.create_buffer(BufferDesc::new(
            node.output.bytes()?,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            MemoryClass::Host,
        )?)?;
        let get = |id| value(id, inputs, &computed);
        let report = match node.operation {
            F32OpV1::Add { left, right } => {
                kernels.binary(CpuF32BinaryOp::Add, get(left)?, get(right)?, &mut output)?
            }
            F32OpV1::Multiply { left, right } => {
                kernels.binary(CpuF32BinaryOp::Multiply, get(left)?, get(right)?, &mut output)?
            }
            F32OpV1::Relu { input } => kernels.relu(get(input)?, &mut output)?,
            F32OpV1::Sum { input } => kernels.sum(get(input)?, &mut output)?,
            F32OpV1::ProjectKn { input, weights } => {
                let input = get(input)?;
                let n = host_bytes(node.output)? / 4;
                kernels.project_kn(input, get(weights)?, &mut output, input.len() / 4, n)?
            }
            F32OpV1::Gather { input, indices } => {
                kernels.gather(get(input)?, indices, &mut output)?
            }
            F32OpV1::ScatterAdd { base, source, indices } => {
                // New destination, not an in-place mutation of the base binding.
                output.bytes.copy_from_slice(&get(base)?.bytes);
                kernels.scatter_add(get(source)?, indices, &mut output)?
            }
        };
        let capacity = u64::try_from(output.capacity_bytes())
            .map_err(|_| invalid("CPU graph capacity exceeds u64"))?;
        retained_node_capacity_bytes = retained_node_capacity_bytes.checked_add(capacity)
            .ok_or_else(|| invalid("CPU graph capacity accounting overflows u64"))?;
        max_scratch_capacity_bytes = max_scratch_capacity_bytes.max(report.scratch_capacity_bytes);
        computed.push(output);
    }
    let output = computed.pop().ok_or_else(|| invalid("CPU graph has no output"))?;
    Ok(CpuF32GraphOutputV1 {
        output,
        report: CpuF32GraphReportV1 {
            schema_version: F32_GRAPH_VERSION,
            executed_nodes: plan.nodes.len(),
            budget: graph.budget(),
            retained_node_capacity_bytes,
            max_scratch_capacity_bytes,
            node_handle_capacity_bytes,
        },
    })
}

fn value<'a>(id: usize, inputs: &[&'a CpuBuffer], computed: &'a [CpuBuffer]) -> Result<&'a CpuBuffer> {
    if id < inputs.len() {
        return Ok(inputs[id]);
    }
    computed.get(id - inputs.len()).ok_or_else(|| invalid("CPU graph operand is unavailable"))
}

fn host_bytes(shape: F32ShapeV1) -> Result<usize> {
    let bytes = usize::try_from(shape.bytes()?)
        .map_err(|_| invalid("CPU graph tensor does not fit host address space"))?;
    if bytes > isize::MAX as usize {
        return Err(invalid("CPU graph tensor exceeds host allocation limit"));
    }
    Ok(bytes)
}

fn invalid(message: &'static str) -> PortableError {
    PortableError::InvalidDescriptor(message)
}
