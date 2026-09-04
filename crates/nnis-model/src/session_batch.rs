use crate::{GenerationConfig, InferenceSession, Model, SamplingConfig};
use nnis_rt::{NnisError, Result};

/// One independent sampled-generation request in a [`SampledSessionBatch`].
///
/// Requests own their prompt so the batch can be prepared independently of the
/// execution call. Every item retains its own generation envelope and sampling
/// seed/policy; NNIS does not silently share RNG state across sessions.
#[derive(Debug, Clone, PartialEq)]
pub struct SampledBatchRequest {
    pub input_ids: Vec<u32>,
    pub generation: GenerationConfig,
    pub sampling: SamplingConfig,
}

impl SampledBatchRequest {
    pub fn new(
        input_ids: Vec<u32>,
        generation: GenerationConfig,
        sampling: SamplingConfig,
    ) -> Self {
        Self {
            input_ids,
            generation,
            sampling,
        }
    }
}

/// A bounded collection of fully independent NNIS inference sessions.
///
/// Each entry owns its own CUDA stream, KV cache, decode workspace, position and
/// sampling invocation. The current NNML1 implementation executes batch items in
/// deterministic index order by calling the existing single-session API. It does
/// **not** claim fused batched kernels, overlapping streams, dynamic batching or
/// throughput improvement; those require separate NNML2/NNML4 qualification.
#[derive(Debug)]
pub struct SampledSessionBatch<'model> {
    sessions: Vec<InferenceSession<'model>>,
}

impl Model {
    /// Allocate `session_count` independent sessions for host-orchestrated batch
    /// execution.
    pub fn new_sampled_session_batch(
        &self,
        session_count: usize,
    ) -> Result<SampledSessionBatch<'_>> {
        SampledSessionBatch::new(self, session_count)
    }
}

impl<'model> SampledSessionBatch<'model> {
    fn new(model: &'model Model, session_count: usize) -> Result<Self> {
        if session_count == 0 {
            return Err(NnisError::invalid_input(
                "sampled session batch requires at least one session",
            ));
        }

        let mut sessions = Vec::with_capacity(session_count);
        for _ in 0..session_count {
            sessions.push(model.new_session()?);
        }
        Ok(Self { sessions })
    }

    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Current decoder position of every session, in stable batch-index order.
    pub fn positions(&self) -> Vec<usize> {
        self.sessions.iter().map(InferenceSession::position).collect()
    }

    /// Execute one sampled request per session in deterministic index order.
    ///
    /// A request-count mismatch fails before any session is touched. Once shape
    /// validation succeeds, each item is an independent failure domain: an error
    /// from one request is returned in that item's result and does not prevent
    /// later sessions from executing. This avoids pretending the host-orchestrated
    /// batch is an atomic GPU transaction.
    pub fn generate_sampled(
        &mut self,
        requests: &[SampledBatchRequest],
    ) -> Result<Vec<std::result::Result<Vec<u32>, NnisError>>> {
        if requests.len() != self.sessions.len() {
            return Err(NnisError::invalid_input(format!(
                "sampled session batch has {} sessions but received {} requests",
                self.sessions.len(),
                requests.len()
            )));
        }

        Ok(self
            .sessions
            .iter_mut()
            .zip(requests)
            .map(|(session, request)| {
                session.generate_sampled(
                    &request.input_ids,
                    request.generation,
                    request.sampling,
                )
            })
            .collect())
    }
}
