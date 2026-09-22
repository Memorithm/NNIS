//! Backend-neutral built-in F32 graph contract, not a shader/package ABI.
//!
//! Values are numbered as inputs followed by node outputs. Every operand must
//! refer to an earlier value; loops, forward references and in-place writes are
//! rejected. Validation borrows the immutable plan and allocates no heap memory.

use crate::{PortableError, Result};

pub const F32_GRAPH_VERSION: u32 = 1;
pub const F32_GRAPH_POLICY: &str = "finite-f32-le-ordered-fma-v1";

/// Logical shape; a matrix is not interchangeable with a same-sized vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F32ShapeV1 {
    Vector(u64),
    Matrix { rows: u64, cols: u64 },
}

impl F32ShapeV1 {
    pub fn elements(self) -> Result<u64> {
        let count = match self {
            Self::Vector(count) => count,
            Self::Matrix { rows, cols } => {
                if rows == 0 || cols == 0 {
                    return Err(invalid("graph matrix dimensions must be positive"));
                }
                mul(rows, cols)?
            }
        };
        if count == 0 {
            return Err(invalid("graph tensor must be non-empty"));
        }
        Ok(count)
    }

    pub fn bytes(self) -> Result<u64> {
        mul(self.elements()?, 4)
    }

    fn vector(self) -> Result<u64> {
        match self {
            Self::Vector(count) => Ok(count),
            _ => Err(invalid("graph operation requires a vector")),
        }
    }
}

/// Explicit builtin operation. Indices are borrowed, ordered and immutable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F32OpV1<'a> {
    Add {
        left: usize,
        right: usize,
    },
    Multiply {
        left: usize,
        right: usize,
    },
    Relu {
        input: usize,
    },
    Sum {
        input: usize,
    },
    ProjectKn {
        input: usize,
        weights: usize,
    },
    Gather {
        input: usize,
        indices: &'a [usize],
    },
    ScatterAdd {
        base: usize,
        source: usize,
        indices: &'a [usize],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32NodeV1<'a> {
    pub operation: F32OpV1<'a>,
    pub output: F32ShapeV1,
}

/// In-memory plan. No implicit parser, allocation, default policy or mutation.
/// The result of a graph is the last node's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32GraphV1<'a> {
    pub schema_version: u32,
    pub numerical_policy: &'a str,
    pub inputs: &'a [F32ShapeV1],
    pub nodes: &'a [F32NodeV1<'a>],
}

/// Caller-owned admission limits, not estimates of available physical memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32GraphLimitsV1 {
    pub max_inputs: usize,
    pub max_nodes: usize,
    pub max_tensor_bytes: u64,
    pub max_live_payload_bytes: u64,
    pub max_scratch_bytes: u64,
    pub max_work_items: u64,
}

/// Conservative retain-all-output payload bound. Caller input aliases are
/// charged once per binding. No allocator overhead, metadata or RSS is included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F32GraphBudgetV1 {
    pub input_payload_bytes: u64,
    pub node_payload_bytes: u64,
    pub max_scratch_payload_bytes: u64,
    pub live_payload_bound_bytes: u64,
    /// Scalar input validation visits + arithmetic/selection steps + output
    /// writes. Index scans are separately included; this is not elapsed time.
    pub work_items: u64,
}

/// Cannot be constructed or modified without successful validation. Borrowed
/// nodes/indices/shapes cannot be changed while this handle is still in use.
#[derive(Debug, Clone, Copy)]
pub struct ValidatedF32GraphV1<'a> {
    plan: F32GraphV1<'a>,
    budget: F32GraphBudgetV1,
}

impl<'a> ValidatedF32GraphV1<'a> {
    pub fn plan(&self) -> &F32GraphV1<'a> {
        &self.plan
    }

    pub const fn budget(&self) -> F32GraphBudgetV1 {
        self.budget
    }
}

impl<'a> F32GraphV1<'a> {
    /// Validate every node before any backend action. All nodes are executed,
    /// including unused outputs; this v1 contract performs no graph rewrites.
    pub fn validate(self, limits: F32GraphLimitsV1) -> Result<ValidatedF32GraphV1<'a>> {
        if self.schema_version != F32_GRAPH_VERSION || self.numerical_policy != F32_GRAPH_POLICY {
            return Err(invalid("unsupported graph schema or numerical policy"));
        }
        if limits.max_inputs == 0
            || limits.max_nodes == 0
            || limits.max_tensor_bytes == 0
            || limits.max_live_payload_bytes == 0
            || limits.max_scratch_bytes == 0
            || limits.max_work_items == 0
        {
            return Err(invalid("graph limits must be positive"));
        }
        if self.inputs.is_empty()
            || self.inputs.len() > limits.max_inputs
            || self.nodes.is_empty()
            || self.nodes.len() > limits.max_nodes
        {
            return Err(invalid("graph input/node count exceeds admission contract"));
        }
        self.inputs
            .len()
            .checked_add(self.nodes.len())
            .ok_or_else(|| invalid("graph value count overflows usize"))?;
        let mut budget = F32GraphBudgetV1 {
            input_payload_bytes: 0,
            node_payload_bytes: 0,
            max_scratch_payload_bytes: 0,
            live_payload_bound_bytes: 0,
            work_items: 0,
        };
        for &shape in self.inputs {
            let bytes = tensor_bytes(shape, limits.max_tensor_bytes)?;
            budget.input_payload_bytes = add(budget.input_payload_bytes, bytes)?;
            budget.work_items = add(budget.work_items, shape.elements()?)?;
        }
        check_work(budget.work_items, limits.max_work_items)?;
        for (index, node) in self.nodes.iter().enumerate() {
            let bytes = tensor_bytes(node.output, limits.max_tensor_bytes)?;
            if bytes > limits.max_scratch_bytes {
                return Err(invalid("graph node exceeds scratch payload limit"));
            }
            budget.node_payload_bytes = add(budget.node_payload_bytes, bytes)?;
            budget.max_scratch_payload_bytes = budget.max_scratch_payload_bytes.max(bytes);
            let (expected, work, indices, index_limit) = self.infer(node.operation, index)?;
            if expected != node.output {
                return Err(invalid("graph output shape does not match operation"));
            }
            let index_work = u64::try_from(indices.len())
                .map_err(|_| invalid("graph index count exceeds u64"))?;
            budget.work_items = add(budget.work_items, add(work, index_work)?)?;
            check_work(budget.work_items, limits.max_work_items)?;
            // Charge the scan before performing it, including very large lists.
            for &value in indices {
                if u64::try_from(value).map_err(|_| invalid("graph index exceeds u64"))?
                    >= index_limit
                {
                    return Err(invalid("graph index is out of range"));
                }
            }
            budget.live_payload_bound_bytes = add(
                add(budget.input_payload_bytes, budget.node_payload_bytes)?,
                budget.max_scratch_payload_bytes,
            )?;
            if budget.live_payload_bound_bytes > limits.max_live_payload_bytes {
                return Err(invalid("graph exceeds live payload admission limit"));
            }
        }
        Ok(ValidatedF32GraphV1 { plan: self, budget })
    }

    fn shape_before(&self, value: usize, node_index: usize) -> Result<F32ShapeV1> {
        if value < self.inputs.len() {
            return Ok(self.inputs[value]);
        }
        let producer = value - self.inputs.len();
        if producer >= node_index {
            return Err(invalid(
                "graph operand is missing, forward, cyclic or in-place",
            ));
        }
        Ok(self.nodes[producer].output)
    }

    fn infer(
        &self,
        operation: F32OpV1<'a>,
        index: usize,
    ) -> Result<(F32ShapeV1, u64, &'a [usize], u64)> {
        let shape = |id| self.shape_before(id, index);
        match operation {
            F32OpV1::Add { left, right } | F32OpV1::Multiply { left, right } => {
                let left = shape(left)?.vector()?;
                if shape(right)? != F32ShapeV1::Vector(left) {
                    return Err(invalid("graph binary operands have different shapes"));
                }
                Ok((F32ShapeV1::Vector(left), mul(left, 4)?, &[], 0))
            }
            F32OpV1::Relu { input } => {
                let count = shape(input)?.vector()?;
                Ok((F32ShapeV1::Vector(count), mul(count, 3)?, &[], 0))
            }
            F32OpV1::Sum { input } => {
                let count = shape(input)?.vector()?;
                Ok((F32ShapeV1::Vector(1), add(mul(count, 2)?, 1)?, &[], 0))
            }
            F32OpV1::ProjectKn { input, weights } => {
                let k = shape(input)?.vector()?;
                let (rows, cols) = match shape(weights)? {
                    F32ShapeV1::Matrix { rows, cols } => (rows, cols),
                    _ => return Err(invalid("graph projection requires matrix weights")),
                };
                if rows != k {
                    return Err(invalid("graph projection K dimension mismatch"));
                }
                let work = add(add(k, mul(mul(rows, cols)?, 2)?)?, cols)?;
                Ok((F32ShapeV1::Vector(cols), work, &[], 0))
            }
            F32OpV1::Gather { input, indices } => {
                let count = shape(input)?.vector()?;
                let selected = u64::try_from(indices.len())
                    .map_err(|_| invalid("graph index count exceeds u64"))?;
                if selected == 0 {
                    return Err(invalid("graph gather indices must be non-empty"));
                }
                Ok((
                    F32ShapeV1::Vector(selected),
                    add(count, mul(selected, 2)?)?,
                    indices,
                    count,
                ))
            }
            F32OpV1::ScatterAdd {
                base,
                source,
                indices,
            } => {
                let count = shape(base)?.vector()?;
                let source = shape(source)?.vector()?;
                if u64::try_from(indices.len()).ok() != Some(source) {
                    return Err(invalid("graph scatter source/index length mismatch"));
                }
                // Base validation and copy; source validation/add; final write.
                let work = add(mul(count, 3)?, mul(source, 2)?)?;
                Ok((F32ShapeV1::Vector(count), work, indices, count))
            }
        }
    }
}

fn tensor_bytes(shape: F32ShapeV1, limit: u64) -> Result<u64> {
    let bytes = shape.bytes()?;
    if bytes > limit {
        return Err(invalid("graph tensor exceeds per-buffer limit"));
    }
    Ok(bytes)
}

fn check_work(work: u64, limit: u64) -> Result<()> {
    if work > limit {
        return Err(invalid("graph exceeds logical work limit"));
    }
    Ok(())
}

fn invalid(message: &'static str) -> PortableError {
    PortableError::InvalidDescriptor(message)
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| invalid("graph accounting overflows u64"))
}

fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("graph accounting overflows u64"))
}
