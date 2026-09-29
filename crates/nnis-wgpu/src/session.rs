//! WGPU implementation of the portable autoregressive session.
//!
//! [`WgpuPortableSession`] mirrors [`nnis_cpu::session::CpuPortableSession`] on
//! the same synthetic analytical model, using the existing WGSL F32 kernels
//! for gather / add / project. Host-dense [`PortableKvCacheV1`] remains the KV
//! layout. Without an adapter there is no session to construct; integration
//! tests SKIP explicitly. Software adapters are code-path only, never hardware
//! evidence. CUDA paths are untouched.

use nnis_core::session::{
    PortableKvCacheV1, PortableSessionError, PortableSessionV1, SyntheticPortableModelSpecV1,
    PORTABLE_SESSION_POLICY, PORTABLE_SESSION_VERSION,
};
use nnis_core::{
    BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableError, PortableQueue,
};

use crate::numerical::{WgpuF32BinaryOp, WgpuF32KernelsV1};
use crate::{WgpuBuffer, WgpuDevice};

/// WGPU portable session over the same synthetic model as the CPU reference.
#[derive(Debug)]
pub struct WgpuPortableSession<'a> {
    spec: SyntheticPortableModelSpecV1,
    device: &'a WgpuDevice,
    embedding: WgpuBuffer,
    lm_head: WgpuBuffer,
    kv: PortableKvCacheV1,
    running_sum: Vec<f32>,
    pending_row: Option<Vec<f32>>,
    logits: Vec<f32>,
    position: usize,
}

impl<'a> WgpuPortableSession<'a> {
    /// Build a session on an already-opened device.
    pub fn new(
        device: &'a WgpuDevice,
        spec: SyntheticPortableModelSpecV1,
    ) -> Result<Self, PortableSessionError> {
        if PORTABLE_SESSION_POLICY != "finite-f32-le-ordered-fma-v1" {
            return Err(PortableSessionError::Invalid(
                "portable session policy identity mismatch",
            ));
        }
        let embedding = one_hot_embedding(device, &spec)?;
        let lm_head = identity_lm_head(device, &spec)?;
        let kv = PortableKvCacheV1::new(spec.layers, spec.capacity, spec.hidden_size)?;
        Ok(Self {
            spec,
            device,
            embedding,
            lm_head,
            kv,
            running_sum: vec![0.0; spec.hidden_size],
            pending_row: None,
            logits: Vec::new(),
            position: 0,
        })
    }

    /// Build a tiny synthetic session on `device`.
    pub fn tiny(device: &'a WgpuDevice) -> Result<Self, PortableSessionError> {
        Self::new(device, SyntheticPortableModelSpecV1::tiny())
    }

    pub const fn spec(&self) -> SyntheticPortableModelSpecV1 {
        self.spec
    }

    pub const fn session_version(&self) -> u32 {
        PORTABLE_SESSION_VERSION
    }

    pub fn device(&self) -> &'a WgpuDevice {
        self.device
    }

    pub fn kv_capacity_payload_bytes(&self) -> u64 {
        self.kv.capacity_payload_bytes()
    }

    pub fn kv_logical_payload_bytes(&self) -> u64 {
        self.kv.logical_payload_bytes()
    }

    fn kernels(&self) -> Result<WgpuF32KernelsV1<'_>, PortableSessionError> {
        WgpuF32KernelsV1::new(self.device).map_err(map_portable)
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
        let mut row = tensor(self.device, &vec![0.0; hidden])?;
        let indices: Vec<usize> = (0..hidden).map(|i| token_index * hidden + i).collect();
        self.kernels()?
            .gather(&self.embedding, &indices, &mut row)
            .map_err(map_portable)?;
        read_f32(self.device, &row)
    }

    fn refresh_logits_from_sum(&mut self) -> Result<(), PortableSessionError> {
        let hidden = self.spec.hidden_size;
        let vocab = self.spec.vocab_size;
        let input = tensor(self.device, &self.running_sum)?;
        let mut output = tensor(self.device, &vec![0.0; vocab])?;
        self.kernels()?
            .project_kn(&input, &self.lm_head, &mut output, hidden, vocab)
            .map_err(map_portable)?;
        self.logits = read_f32(self.device, &output)?;
        Ok(())
    }
}

impl<'a> PortableSessionV1 for WgpuPortableSession<'a> {
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
        let row_buf = tensor(self.device, &row)?;
        let sum_buf = tensor(self.device, &self.running_sum)?;
        let mut out_sum = tensor(self.device, &vec![0.0; self.spec.hidden_size])?;
        self.kernels()?
            .binary(WgpuF32BinaryOp::Add, &sum_buf, &row_buf, &mut out_sum)
            .map_err(map_portable)?;
        let mut logits_buf = tensor(self.device, &vec![0.0; self.spec.vocab_size])?;
        self.kernels()?
            .project_kn(
                &out_sum,
                &self.lm_head,
                &mut logits_buf,
                self.spec.hidden_size,
                self.spec.vocab_size,
            )
            .map_err(map_portable)?;
        self.logits = read_f32(self.device, &logits_buf)?;
        self.pending_row = Some(row);
        Ok(&self.logits)
    }

    fn advance_kv(&mut self) -> Result<(), PortableSessionError> {
        let row = self.pending_row.take().ok_or(PortableSessionError::State(
            "portable session advance_kv requires a staged token",
        ))?;
        let sum_buf = tensor(self.device, &self.running_sum)?;
        let row_buf = tensor(self.device, &row)?;
        let mut out_sum = tensor(self.device, &vec![0.0; self.spec.hidden_size])?;
        self.kernels()?
            .binary(WgpuF32BinaryOp::Add, &sum_buf, &row_buf, &mut out_sum)
            .map_err(map_portable)?;
        self.running_sum = read_f32(self.device, &out_sum)?;
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

fn tensor(device: &WgpuDevice, values: &[f32]) -> Result<WgpuBuffer, PortableSessionError> {
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
    let descriptor = BufferDesc::new(bytes.len() as u64, usage, MemoryClass::DeviceLocal)
        .map_err(map_portable)?;
    let mut buffer = device.create_buffer(descriptor).map_err(map_portable)?;
    device
        .create_queue()
        .map_err(map_portable)?
        .write_buffer(&mut buffer, 0, &bytes)
        .map_err(map_portable)?;
    Ok(buffer)
}

fn read_f32(device: &WgpuDevice, buffer: &WgpuBuffer) -> Result<Vec<f32>, PortableSessionError> {
    let bytes = device
        .create_queue()
        .map_err(map_portable)?
        .read_buffer(buffer, 0, buffer.len())
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
    device: &WgpuDevice,
    spec: &SyntheticPortableModelSpecV1,
) -> Result<WgpuBuffer, PortableSessionError> {
    let mut values = vec![0.0; spec.vocab_size * spec.hidden_size];
    for token in 0..spec.vocab_size {
        let axis = token % spec.hidden_size;
        values[token * spec.hidden_size + axis] = 1.0;
    }
    tensor(device, &values)
}

fn identity_lm_head(
    device: &WgpuDevice,
    spec: &SyntheticPortableModelSpecV1,
) -> Result<WgpuBuffer, PortableSessionError> {
    let mut values = vec![0.0; spec.hidden_size * spec.vocab_size];
    let diag = spec.hidden_size.min(spec.vocab_size);
    for i in 0..diag {
        values[i * spec.vocab_size + i] = 1.0;
    }
    tensor(device, &values)
}
