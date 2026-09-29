//! CPU reference implementation of the portable autoregressive session.
//!
//! [`CpuPortableSession`] runs a tiny analytical synthetic model over the
//! existing P2b F32 kernels and the host-dense [`PortableKvCacheV1`]. It is the
//! portable counterpart shape to CUDA InferenceSession (encode / decode_one /
//! KV advance / truncate) without touching `nnis-rt` or any NVIDIA path.
//!
//! Synthetic fixtures and unit tests are structural only: they are not
//! model-quality, latency, throughput or hardware evidence.

use nnis_core::graph::{
    F32GraphLimitsV1, F32GraphV1, F32NodeV1, F32OpV1, F32ShapeV1, F32_GRAPH_POLICY,
    F32_GRAPH_VERSION,
};
use nnis_core::session::{
    PortableKvCacheV1, PortableSessionError, PortableSessionV1, SyntheticPortableModelSpecV1,
    PORTABLE_SESSION_POLICY, PORTABLE_SESSION_VERSION,
};
use nnis_core::{
    BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableError, PortableQueue,
};

use crate::graph::execute_f32_graph;
use crate::numerical::{CpuF32BinaryOp, CpuF32KernelsV1};
use crate::{CpuBuffer, CpuDevice};

/// CPU portable session over a synthetic analytical model.
#[derive(Debug)]
pub struct CpuPortableSession {
    spec: SyntheticPortableModelSpecV1,
    device: CpuDevice,
    kernels: CpuF32KernelsV1,
    /// Row-major `[vocab, hidden]` embedding table.
    embedding: CpuBuffer,
    /// Row-major `[hidden, vocab]` LM-head weights (`ProjectKn` layout).
    lm_head: CpuBuffer,
    kv: PortableKvCacheV1,
    running_sum: Vec<f32>,
    pending_row: Option<Vec<f32>>,
    logits: Vec<f32>,
    position: usize,
}

impl CpuPortableSession {
    /// Build a session for [`SyntheticPortableModelSpecV1::tiny`] with identity LM head.
    pub fn tiny() -> Result<Self, PortableSessionError> {
        Self::new(SyntheticPortableModelSpecV1::tiny())
    }

    /// Build a session for `spec` with one-hot embeddings and an identity LM head
    /// when `hidden_size == vocab_size`, otherwise a truncated/padded identity.
    pub fn new(spec: SyntheticPortableModelSpecV1) -> Result<Self, PortableSessionError> {
        if PORTABLE_SESSION_POLICY != "finite-f32-le-ordered-fma-v1" {
            return Err(PortableSessionError::Invalid(
                "portable session policy identity mismatch",
            ));
        }
        let device = CpuDevice::with_max_buffer_bytes(1 << 20).map_err(map_portable)?;
        let kernels = CpuF32KernelsV1::new(spec.hidden_size.max(spec.vocab_size) * 4)
            .map_err(map_portable)?;
        let embedding = one_hot_embedding(&device, &spec)?;
        let lm_head = identity_lm_head(&device, &spec)?;
        let kv = PortableKvCacheV1::new(spec.layers, spec.capacity, spec.hidden_size)?;
        let running_sum = vec![0.0; spec.hidden_size];
        Ok(Self {
            spec,
            device,
            kernels,
            embedding,
            lm_head,
            kv,
            running_sum,
            pending_row: None,
            logits: Vec::new(),
            position: 0,
        })
    }

    pub const fn spec(&self) -> SyntheticPortableModelSpecV1 {
        self.spec
    }

    pub const fn session_version(&self) -> u32 {
        PORTABLE_SESSION_VERSION
    }

    /// Exact host KV capacity payload bytes (allocated), not process RSS.
    pub fn kv_capacity_payload_bytes(&self) -> u64 {
        self.kv.capacity_payload_bytes()
    }

    /// Exact host KV logical payload bytes for the committed length.
    pub fn kv_logical_payload_bytes(&self) -> u64 {
        self.kv.logical_payload_bytes()
    }

    fn validate_token(&self, token: u32) -> Result<usize, PortableSessionError> {
        let index = token as usize;
        if index >= self.spec.vocab_size {
            return Err(PortableSessionError::Invalid(
                "token id exceeds synthetic vocabulary",
            ));
        }
        Ok(index)
    }

    fn gather_embedding(&self, token_index: usize) -> Result<Vec<f32>, PortableSessionError> {
        let hidden = self.spec.hidden_size;
        let mut row = tensor(&self.device, &vec![0.0; hidden])?;
        // Gather hidden contiguous elements starting at token_index * hidden.
        let indices: Vec<usize> = (0..hidden).map(|i| token_index * hidden + i).collect();
        self.kernels
            .gather(&self.embedding, &indices, &mut row)
            .map_err(map_portable)?;
        read_f32(&self.device, &row)
    }

    fn refresh_logits_from_sum(&mut self) -> Result<(), PortableSessionError> {
        let hidden = self.spec.hidden_size;
        let vocab = self.spec.vocab_size;
        let input = tensor(&self.device, &self.running_sum)?;
        let mut output = tensor(&self.device, &vec![0.0; vocab])?;
        self.kernels
            .project_kn(&input, &self.lm_head, &mut output, hidden, vocab)
            .map_err(map_portable)?;
        self.logits = read_f32(&self.device, &output)?;
        Ok(())
    }

    /// Project `(running_sum + pending_row)` through the LM head via the portable
    /// F32 graph executor. Used by tests to prove graph-plan reuse; the hot path
    /// of [`Self::stage_token`] uses the same kernels directly.
    pub fn logits_via_f32_graph(
        &self,
        pending_row: &[f32],
    ) -> Result<Vec<f32>, PortableSessionError> {
        if pending_row.len() != self.spec.hidden_size {
            return Err(PortableSessionError::Invalid(
                "pending row width mismatch for graph projection",
            ));
        }
        let hidden = self.spec.hidden_size;
        let vocab = self.spec.vocab_size;
        let mut context = self.running_sum.clone();
        for (dst, src) in context.iter_mut().zip(pending_row.iter()) {
            let sum = *dst + *src;
            if !sum.is_finite() {
                return Err(PortableSessionError::Invalid(
                    "context sum became non-finite",
                ));
            }
            *dst = sum;
        }
        let context_buf = tensor(&self.device, &context)?;
        let inputs_shapes = [
            F32ShapeV1::Vector(hidden as u64),
            F32ShapeV1::Matrix {
                rows: hidden as u64,
                cols: vocab as u64,
            },
        ];
        let nodes = [F32NodeV1 {
            operation: F32OpV1::ProjectKn {
                input: 0,
                weights: 1,
            },
            output: F32ShapeV1::Vector(vocab as u64),
        }];
        let plan = F32GraphV1 {
            schema_version: F32_GRAPH_VERSION,
            numerical_policy: F32_GRAPH_POLICY,
            inputs: &inputs_shapes,
            nodes: &nodes,
        };
        let limits = F32GraphLimitsV1 {
            max_inputs: 8,
            max_nodes: 8,
            max_tensor_bytes: 4096,
            max_live_payload_bytes: 4096,
            max_scratch_bytes: 4096,
            max_work_items: 1_000_000,
        };
        let validated = plan.validate(limits).map_err(map_portable)?;
        let out = execute_f32_graph(&self.device, &validated, &[&context_buf, &self.lm_head])
            .map_err(map_portable)?;
        read_f32(&self.device, &out.output)
    }
}

impl PortableSessionV1 for CpuPortableSession {
    fn position(&self) -> usize {
        self.position
    }

    fn kv_len(&self) -> usize {
        self.kv.len()
    }

    fn capacity(&self) -> usize {
        self.spec.capacity
    }

    fn vocab_size(&self) -> usize {
        self.spec.vocab_size
    }

    fn hidden_size(&self) -> usize {
        self.spec.hidden_size
    }

    fn has_pending_kv(&self) -> bool {
        self.pending_row.is_some()
    }

    fn logits(&self) -> &[f32] {
        &self.logits
    }

    fn stage_token(&mut self, token: u32) -> Result<&[f32], PortableSessionError> {
        if self.pending_row.is_some() {
            return Err(PortableSessionError::State(
                "portable session already has a pending KV row; advance_kv or truncate first",
            ));
        }
        let token_index = self.validate_token(token)?;
        let next_len = self
            .kv
            .len()
            .checked_add(1)
            .ok_or(PortableSessionError::Invalid(
                "portable session length overflow",
            ))?;
        if next_len > self.spec.capacity {
            return Err(PortableSessionError::State(
                "portable session encode/decode would exceed KV capacity",
            ));
        }
        let row = self.gather_embedding(token_index)?;
        let row_buf = tensor(&self.device, &row)?;
        let sum_buf = tensor(&self.device, &self.running_sum)?;
        let mut out_sum = tensor(&self.device, &vec![0.0; self.spec.hidden_size])?;
        self.kernels
            .binary(CpuF32BinaryOp::Add, &sum_buf, &row_buf, &mut out_sum)
            .map_err(map_portable)?;
        let mut logits_buf = tensor(&self.device, &vec![0.0; self.spec.vocab_size])?;
        self.kernels
            .project_kn(
                &out_sum,
                &self.lm_head,
                &mut logits_buf,
                self.spec.hidden_size,
                self.spec.vocab_size,
            )
            .map_err(map_portable)?;
        self.logits = read_f32(&self.device, &logits_buf)?;
        // Stash the row; running_sum updates only on advance_kv.
        self.pending_row = Some(row);
        Ok(&self.logits)
    }

    fn advance_kv(&mut self) -> Result<(), PortableSessionError> {
        let row = self.pending_row.take().ok_or(PortableSessionError::State(
            "portable session advance_kv requires a staged token",
        ))?;
        // Update running sum with the committed row using the same Add kernel.
        let sum_buf = tensor(&self.device, &self.running_sum)?;
        let row_buf = tensor(&self.device, &row)?;
        let mut out_sum = tensor(&self.device, &vec![0.0; self.spec.hidden_size])?;
        self.kernels
            .binary(CpuF32BinaryOp::Add, &sum_buf, &row_buf, &mut out_sum)
            .map_err(map_portable)?;
        self.running_sum = read_f32(&self.device, &out_sum)?;
        // One row per layer; synthetic v1 uses identical rows across layers.
        let views: Vec<&[f32]> = (0..self.spec.layers).map(|_| row.as_slice()).collect();
        self.kv.append(&views)?;
        self.position = self
            .position
            .checked_add(1)
            .ok_or(PortableSessionError::Invalid(
                "portable session position overflow",
            ))?;
        Ok(())
    }

    fn encode(&mut self, tokens: &[u32]) -> Result<&[f32], PortableSessionError> {
        if tokens.is_empty() {
            return Err(PortableSessionError::Invalid(
                "portable session encode requires a non-empty token list",
            ));
        }
        if self.position != 0 || !self.kv.is_empty() || self.pending_row.is_some() {
            return Err(PortableSessionError::State(
                "portable session encode requires a fresh (reset) session",
            ));
        }
        for &token in tokens {
            self.stage_token(token)?;
            self.advance_kv()?;
        }
        Ok(&self.logits)
    }

    fn decode_one(&mut self, token: u32) -> Result<&[f32], PortableSessionError> {
        if self.pending_row.is_some() {
            return Err(PortableSessionError::State(
                "portable session decode_one requires no pending KV row",
            ));
        }
        self.stage_token(token)?;
        self.advance_kv()?;
        Ok(&self.logits)
    }

    fn truncate(&mut self, length: usize) -> Result<(), PortableSessionError> {
        if length > self.kv.len() {
            return Err(PortableSessionError::Invalid(
                "portable session truncate length exceeds kv_len",
            ));
        }
        self.pending_row = None;
        self.kv.truncate(length)?;
        self.position = length;
        self.running_sum.fill(0.0);
        for index in 0..length {
            let row = self.kv.row(0, index)?;
            for (dst, src) in self.running_sum.iter_mut().zip(row.iter()) {
                let sum = *dst + *src;
                if !sum.is_finite() {
                    return Err(PortableSessionError::Invalid(
                        "portable session truncate recomputed a non-finite sum",
                    ));
                }
                *dst = sum;
            }
        }
        if length == 0 {
            self.logits.clear();
        } else {
            self.refresh_logits_from_sum()?;
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<(), PortableSessionError> {
        self.pending_row = None;
        self.kv.reset();
        self.running_sum.fill(0.0);
        self.logits.clear();
        self.position = 0;
        Ok(())
    }
}

fn map_portable(error: PortableError) -> PortableSessionError {
    PortableSessionError::Backend(error.to_string())
}

fn tensor(device: &CpuDevice, values: &[f32]) -> Result<CpuBuffer, PortableSessionError> {
    if values.is_empty() {
        return Err(PortableSessionError::Invalid(
            "portable session buffer must be non-empty",
        ));
    }
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let descriptor =
        BufferDesc::new(bytes.len() as u64, usage, MemoryClass::Host).map_err(map_portable)?;
    let mut buffer = device.create_buffer(descriptor).map_err(map_portable)?;
    device
        .create_queue()
        .map_err(map_portable)?
        .write_buffer(&mut buffer, 0, &bytes)
        .map_err(map_portable)?;
    Ok(buffer)
}

fn read_f32(device: &CpuDevice, buffer: &CpuBuffer) -> Result<Vec<f32>, PortableSessionError> {
    let bytes = device
        .create_queue()
        .map_err(map_portable)?
        .read_buffer(buffer, 0, buffer.len() as u64)
        .map_err(map_portable)?;
    if bytes.len() % 4 != 0 {
        return Err(PortableSessionError::Invalid(
            "portable session buffer length is not a multiple of 4",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

fn one_hot_embedding(
    device: &CpuDevice,
    spec: &SyntheticPortableModelSpecV1,
) -> Result<CpuBuffer, PortableSessionError> {
    let mut values = vec![0.0; spec.vocab_size * spec.hidden_size];
    for token in 0..spec.vocab_size {
        let axis = token % spec.hidden_size;
        values[token * spec.hidden_size + axis] = 1.0;
    }
    tensor(device, &values)
}

fn identity_lm_head(
    device: &CpuDevice,
    spec: &SyntheticPortableModelSpecV1,
) -> Result<CpuBuffer, PortableSessionError> {
    // ProjectKn weights are row-major [hidden, vocab]: weights[row * vocab + col].
    let mut values = vec![0.0; spec.hidden_size * spec.vocab_size];
    let diag = spec.hidden_size.min(spec.vocab_size);
    for i in 0..diag {
        values[i * spec.vocab_size + i] = 1.0;
    }
    tensor(device, &values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_is_analytical_for_tiny_identity_model() {
        let mut session = CpuPortableSession::tiny().unwrap();
        // Tokens 0,1,0 -> running_sum one-hots [2,1,0,0], identity LM -> same logits.
        let logits = session.encode(&[0, 1, 0]).unwrap().to_vec();
        assert_eq!(logits, vec![2.0, 1.0, 0.0, 0.0]);
        assert_eq!(session.position(), 3);
        assert_eq!(session.kv_len(), 3);
        let logits = session.decode_one(2).unwrap().to_vec();
        assert_eq!(logits, vec![2.0, 1.0, 1.0, 0.0]);
        assert_eq!(session.kv_logical_payload_bytes(), 4 * 4 * 4); // layers=1,len=4,width=4,f32
    }

    #[test]
    fn stage_then_advance_is_explicit_and_truncate_rewinds() {
        let mut session = CpuPortableSession::tiny().unwrap();
        session.encode(&[0, 1]).unwrap();
        assert!(!session.has_pending_kv());
        let staged = session.stage_token(2).unwrap().to_vec();
        assert_eq!(staged, vec![1.0, 1.0, 1.0, 0.0]);
        assert!(session.has_pending_kv());
        assert_eq!(session.kv_len(), 2); // not yet committed
        session.advance_kv().unwrap();
        assert_eq!(session.kv_len(), 3);
        session.truncate(1).unwrap();
        assert_eq!(session.position(), 1);
        assert_eq!(session.kv_len(), 1);
        assert_eq!(session.logits(), &[1.0, 0.0, 0.0, 0.0]);
        let logits = session.decode_one(1).unwrap().to_vec();
        assert_eq!(logits, vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn graph_projection_matches_kernel_path() {
        let mut session = CpuPortableSession::tiny().unwrap();
        session.encode(&[0, 3]).unwrap();
        let pending = session.gather_embedding(1).unwrap();
        let via_graph = session.logits_via_f32_graph(&pending).unwrap();
        session.stage_token(1).unwrap();
        assert_eq!(session.logits(), via_graph.as_slice());
    }

    #[test]
    fn fail_closed_on_capacity_invalid_token_and_double_stage() {
        let mut session = CpuPortableSession::tiny().unwrap();
        assert!(session.encode(&[]).is_err());
        assert!(session.stage_token(99).is_err());
        session.encode(&[0, 1, 2, 3, 0, 1, 2, 3]).unwrap();
        assert_eq!(session.kv_len(), 8);
        assert!(session.decode_one(0).is_err());
        session.reset().unwrap();
        session.stage_token(0).unwrap();
        assert!(session.stage_token(1).is_err());
        assert!(session.encode(&[0]).is_err()); // not fresh
    }
}
