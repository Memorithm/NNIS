//! Portable built-in F32 graph execution on WGPU through [`WgpuF32KernelsV1`].
//!
//! Mirrors `nnis_cpu::graph::execute_f32_graph`: the plan must already be
//! validated; bindings, per-buffer device limits, input usage, input byte
//! lengths and the finiteness of every input are checked before any node
//! buffer is allocated; caller buffers are never mutated (scatter-add writes a
//! fresh copy of its base); intermediates are private and dropped on every
//! error; only the final value escapes.
//!
//! Declared tolerance: each node follows the numerical policy of the WGSL
//! kernel it runs (see [`crate::numerical`]). A graph without `ProjectKn`
//! nodes is bit-exact with the CPU graph reference for normal-range values.
//! Projection outputs are within the declared projection bound of the CPU
//! kernel applied to the *same* inputs; later nodes then consume those
//! values, so end-to-end equality with the CPU graph is not claimed for graphs
//! containing projections. [`execute_f32_graph_traced`] exposes every node
//! output so callers can check each node against the CPU kernel on the WGPU
//! node inputs.

use nnis_core::graph::{
    F32GraphBudgetV1, F32OpV1, ValidatedF32GraphV1, F32_GRAPH_POLICY, F32_GRAPH_VERSION,
};
use nnis_core::{BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableError, Result};

use crate::memory::require_usage;
use crate::numerical::{WgpuF32BinaryOp, WgpuF32KernelsV1};
use crate::{WgpuBuffer, WgpuDevice};

/// Successful graph execution record; not timing or residency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuF32GraphReportV1 {
    pub schema_version: u32,
    pub executed_nodes: usize,
    pub budget: F32GraphBudgetV1,
    /// Fingerprint of the WGSL artifact bound for each node, in node order.
    pub node_artifact_fingerprints: Vec<[u8; 32]>,
}

/// Final output of a graph execution.
#[derive(Debug)]
pub struct WgpuF32GraphOutputV1 {
    pub output: WgpuBuffer,
    pub report: WgpuF32GraphReportV1,
}

/// Every node output (in node order) of a traced execution.
#[derive(Debug)]
pub struct WgpuF32GraphTraceV1 {
    pub nodes: Vec<WgpuBuffer>,
    pub report: WgpuF32GraphReportV1,
}

/// Execute a statically validated graph with immutable external inputs.
pub fn execute_f32_graph(
    device: &WgpuDevice,
    graph: &ValidatedF32GraphV1<'_>,
    inputs: &[&WgpuBuffer],
) -> Result<WgpuF32GraphOutputV1> {
    let mut trace = execute_f32_graph_traced(device, graph, inputs)?;
    let output = trace
        .nodes
        .pop()
        .ok_or_else(|| invalid("WGPU graph has no output"))?;
    Ok(WgpuF32GraphOutputV1 {
        output,
        report: trace.report,
    })
}

/// Execute like [`execute_f32_graph`] but return every node output.
pub fn execute_f32_graph_traced(
    device: &WgpuDevice,
    graph: &ValidatedF32GraphV1<'_>,
    inputs: &[&WgpuBuffer],
) -> Result<WgpuF32GraphTraceV1> {
    let plan = graph.plan();
    if plan.schema_version != F32_GRAPH_VERSION || plan.numerical_policy != F32_GRAPH_POLICY {
        return Err(invalid(
            "WGPU graph schema or numerical policy is unsupported",
        ));
    }
    if inputs.len() != plan.inputs.len() {
        return Err(invalid("WGPU graph binding count mismatch"));
    }
    for shape in plan
        .inputs
        .iter()
        .copied()
        .chain(plan.nodes.iter().map(|node| node.output))
    {
        if shape.bytes()? > device.capabilities().max_buffer_bytes {
            return Err(invalid(
                "WGPU graph tensor exceeds device per-buffer capability",
            ));
        }
    }
    let kernels = WgpuF32KernelsV1::new(device)?;
    for (&shape, buffer) in plan.inputs.iter().zip(inputs) {
        require_usage(buffer, BufferUsages::STORAGE, "WGPU graph input")?;
        if buffer.len() != shape.bytes()? {
            return Err(invalid("WGPU graph input byte length mismatch"));
        }
    }
    for buffer in inputs {
        kernels.check_finite(buffer).map_err(|error| match error {
            PortableError::InvalidDescriptor(_) => {
                invalid("WGPU graph input contains a non-finite value")
            }
            other => other,
        })?;
    }

    let queue = device.create_queue()?;
    let mut computed: Vec<WgpuBuffer> = Vec::with_capacity(plan.nodes.len());
    let mut fingerprints = Vec::with_capacity(plan.nodes.len());
    for node in plan.nodes {
        let mut output = device.create_buffer(BufferDesc::new(
            node.output.bytes()?,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            MemoryClass::DeviceLocal,
        )?)?;
        let get = |id| value(id, inputs, &computed);
        let report = match node.operation {
            F32OpV1::Add { left, right } => {
                kernels.binary(WgpuF32BinaryOp::Add, get(left)?, get(right)?, &mut output)?
            }
            F32OpV1::Multiply { left, right } => kernels.binary(
                WgpuF32BinaryOp::Multiply,
                get(left)?,
                get(right)?,
                &mut output,
            )?,
            F32OpV1::Relu { input } => kernels.relu(get(input)?, &mut output)?,
            F32OpV1::Sum { input } => kernels.sum(get(input)?, &mut output)?,
            F32OpV1::ProjectKn { input, weights } => {
                let input = get(input)?;
                let k = words(input.len())?;
                let n = words(node.output.bytes()?)?;
                kernels.project_kn(input, get(weights)?, &mut output, k, n)?
            }
            F32OpV1::Gather { input, indices } => {
                kernels.gather(get(input)?, indices, &mut output)?
            }
            F32OpV1::ScatterAdd {
                base,
                source,
                indices,
            } => {
                // New destination, not an in-place mutation of the base binding.
                let base = get(base)?;
                queue.copy_unchecked(base, &output, base.len())?;
                kernels.scatter_add(get(source)?, indices, &mut output)?
            }
        };
        fingerprints.push(report.artifact_fingerprint);
        computed.push(output);
    }
    Ok(WgpuF32GraphTraceV1 {
        nodes: computed,
        report: WgpuF32GraphReportV1 {
            schema_version: F32_GRAPH_VERSION,
            executed_nodes: plan.nodes.len(),
            budget: graph.budget(),
            node_artifact_fingerprints: fingerprints,
        },
    })
}

fn value<'a>(
    id: usize,
    inputs: &[&'a WgpuBuffer],
    computed: &'a [WgpuBuffer],
) -> Result<&'a WgpuBuffer> {
    if id < inputs.len() {
        return Ok(inputs[id]);
    }
    computed
        .get(id - inputs.len())
        .ok_or_else(|| invalid("WGPU graph operand is unavailable"))
}

fn words(bytes: u64) -> Result<usize> {
    usize::try_from(bytes / 4).map_err(|_| invalid("WGPU graph tensor exceeds host usize"))
}

fn invalid(message: &'static str) -> PortableError {
    PortableError::InvalidDescriptor(message)
}
