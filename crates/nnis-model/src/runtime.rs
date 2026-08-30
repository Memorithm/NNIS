use crate::{
    load_model_directory, F32DecoderKernels, F32RuntimeKernels, GenerationConfig, ModelConfig,
    ModelWeights, WeightDType,
};
use nnis_jit::JitCompiler;
use nnis_kernels::{F32Elementwise, F32Gather, F32Gemm, F32TopK, F32TopKWorkspace};
use nnis_rt::{Context, DeviceBuffer, KvAppend, KvCache, KvCacheConfig, NnisError, Result, Stream};
use std::path::Path;
use std::sync::Arc;

/// Compiled, immutable decoder model. Sessions own all mutable execution state.
#[derive(Debug)]
pub struct Model {
    config: ModelConfig,
    weights: ModelWeights,
    context: Arc<Context>,
    gather: F32Gather,
    gemm: F32Gemm,
    elementwise: F32Elementwise,
    top_k: F32TopK,
    decoder: F32DecoderKernels,
    runtime: F32RuntimeKernels,
    rope_cos: DeviceBuffer<f32>,
    rope_sin: DeviceBuffer<f32>,
}

impl Model {
    pub fn new(config: ModelConfig, weights: ModelWeights, stream: &Stream) -> Result<Self> {
        config.validate_execution_support()?;
        if config.weight_dtype != WeightDType::F32 {
            return Err(NnisError::unsupported(
                "the first decoder execution path currently requires f32 model weights",
            ));
        }
        weights.validate(&config)?;
        if !Arc::ptr_eq(weights.context(), stream.ctx()) {
            return Err(NnisError::invalid_input(
                "model weights and construction stream must share one CUDA context",
            ));
        }
        let context = Arc::clone(stream.ctx());
        let compiler = JitCompiler::new();
        let gather = F32Gather::load(&context, &compiler)?;
        let gemm = F32Gemm::load(&context, &compiler)?;
        let elementwise = F32Elementwise::load(&context, &compiler)?;
        let top_k = F32TopK::load(&context, &compiler)?;
        let decoder = F32DecoderKernels::load(&context, &compiler)?;
        let runtime = F32RuntimeKernels::load(&context, &compiler)?;
        let (cos_host, sin_host) = build_rope_cache(&config)?;
        let rope_cos = DeviceBuffer::from_host(&context, stream, &cos_host)?;
        let rope_sin = DeviceBuffer::from_host(&context, stream, &sin_host)?;
        Ok(Self {
            config,
            weights,
            context,
            gather,
            gemm,
            elementwise,
            top_k,
            decoder,
            runtime,
            rope_cos,
            rope_sin,
        })
    }

    pub fn load_directory(
        context: &Arc<Context>,
        stream: &Stream,
        directory: impl AsRef<Path>,
    ) -> Result<Self> {
        let (config, weights) = load_model_directory(context, stream, directory)?;
        Self::new(config, weights, stream)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn new_session(&self) -> Result<InferenceSession<'_>> {
        InferenceSession::new(self)
    }
}

/// Long-lived device allocations reused for every decoder token.
#[derive(Debug)]
struct DecodeWorkspace {
    hidden: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: Arc<DeviceBuffer<f32>>,
    q_rope: DeviceBuffer<f32>,
    k_rope: Arc<DeviceBuffer<f32>>,
    attention: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    residual: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    activated: DeviceBuffer<f32>,
    gated: DeviceBuffer<f32>,
    mlp: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    current_token: DeviceBuffer<u32>,
    top_value: DeviceBuffer<f32>,
    top_k_workspace: F32TopKWorkspace,
}

impl DecodeWorkspace {
    fn new(model: &Model) -> Result<Self> {
        let context = &model.context;
        let hidden = model.config.hidden_size;
        let intermediate = model.config.intermediate_size;
        let vocab = model.config.vocab_size;
        let kv_width = model.config.key_value_width()?;
        Ok(Self {
            hidden: DeviceBuffer::new(context, hidden)?,
            normed: DeviceBuffer::new(context, hidden)?,
            q: DeviceBuffer::new(context, hidden)?,
            k: DeviceBuffer::new(context, kv_width)?,
            v: Arc::new(DeviceBuffer::new(context, kv_width)?),
            q_rope: DeviceBuffer::new(context, hidden)?,
            k_rope: Arc::new(DeviceBuffer::new(context, kv_width)?),
            attention: DeviceBuffer::new(context, hidden)?,
            projected: DeviceBuffer::new(context, hidden)?,
            residual: DeviceBuffer::new(context, hidden)?,
            gate: DeviceBuffer::new(context, intermediate)?,
            up: DeviceBuffer::new(context, intermediate)?,
            activated: DeviceBuffer::new(context, intermediate)?,
            gated: DeviceBuffer::new(context, intermediate)?,
            mlp: DeviceBuffer::new(context, hidden)?,
            logits: DeviceBuffer::new(context, vocab)?,
            current_token: DeviceBuffer::new(context, 1)?,
            top_value: DeviceBuffer::new(context, 1)?,
            top_k_workspace: model.top_k.workspace(context, vocab)?,
        })
    }
}

/// Mutable autoregressive state for one sequence.
///
/// Safe methods require `&mut self`, own every temporary buffer and submit all
/// dependent operations to exactly one stream. Existing NNIS `enqueue_*`
/// primitives are used internally only under this ownership and ordering
/// discipline; no buffer or cache is exposed while work is outstanding.
#[derive(Debug)]
pub struct InferenceSession<'model> {
    model: &'model Model,
    stream: Stream,
    cache: KvCache<f32>,
    workspace: DecodeWorkspace,
    pending_appends: Vec<KvAppend<f32>>,
    position: usize,
}

impl<'model> InferenceSession<'model> {
    fn new(model: &'model Model) -> Result<Self> {
        let stream = Stream::new(&model.context)?;
        let cache = KvCache::new(
            &stream,
            KvCacheConfig::new(
                model.config.num_hidden_layers,
                model.config.num_key_value_heads,
                model.config.head_dim(),
                model.config.max_position_embeddings,
            )?,
        )?;
        let workspace = DecodeWorkspace::new(model)?;
        Ok(Self {
            model,
            stream,
            cache,
            workspace,
            pending_appends: Vec::new(),
            position: 0,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn capacity(&self) -> usize {
        self.model.config.max_position_embeddings
    }

    pub fn reset(&mut self) -> Result<()> {
        self.stream.synchronize()?;
        self.pending_appends.clear();
        self.cache.reset();
        self.position = 0;
        Ok(())
    }

    /// Prefill a complete prompt on one stream and return logits after the last
    /// prompt token. Input IDs are uploaded once; token selection thereafter is
    /// device-resident.
    pub fn prefill(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        self.validate_prompt(input_ids, 0)?;
        self.reset()?;
        let device_ids = DeviceBuffer::from_host(&self.model.context, &self.stream, input_ids)?;
        let enqueue_result = self.enqueue_prefill(&device_ids);
        self.finish(enqueue_result)?;
        self.workspace.logits.to_vec(&self.stream)
    }

    /// Decode one explicit token into the existing session and return its
    /// resulting logits. The H2D token copy and all decoder work share the same
    /// synchronization boundary.
    pub fn decode_one(&mut self, token: u32) -> Result<Vec<f32>> {
        self.validate_token(token)?;
        if self.position >= self.capacity() {
            return Err(NnisError::invalid_input(format!(
                "decode position {} exceeds session capacity {}",
                self.position,
                self.capacity()
            )));
        }
        let host = [token];
        let enqueue_result = (|| {
            // SAFETY: `host` and the session-owned token buffer remain alive
            // until `finish` synchronizes below.
            unsafe {
                self.workspace
                    .current_token
                    .copy_from_host_async(&self.stream, &host)?;
            }
            self.enqueue_current_token()
        })();
        self.finish(enqueue_result)?;
        self.workspace.logits.to_vec(&self.stream)
    }

    /// Greedy generation entirely through NNIS.
    ///
    /// Fixed-length generation remains one device-resident graph. When
    /// `eos_token_id` is configured, NNIS observes one top-1 token on
    /// the host per step so it can stop submitting decoder work after
    /// EOS. Transformer stages themselves remain ordered on one CUDA
    /// stream with no intermediate activation roundtrips.
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        generation: GenerationConfig,
    ) -> Result<Vec<u32>> {
        generation.validate(self.model.config.vocab_size)?;
        self.validate_prompt(input_ids, generation.max_new_tokens)?;
        match generation.eos_token_id {
            Some(eos_token_id) => {
                self.generate_until_eos(input_ids, generation.max_new_tokens, eos_token_id)
            }
            None => self.generate_fixed(input_ids, generation.max_new_tokens),
        }
    }

    fn generate_fixed(&mut self, input_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        self.reset()?;
        let device_ids = DeviceBuffer::from_host(&self.model.context, &self.stream, input_ids)?;
        let generated = DeviceBuffer::<u32>::new(&self.model.context, max_new_tokens)?;

        let enqueue_result = (|| {
            self.enqueue_prefill(&device_ids)?;
            for step in 0..max_new_tokens {
                // SAFETY: session ownership keeps logits, outputs and scratch
                // live; the workspace is used serially on this stream only.
                unsafe {
                    self.model.top_k.enqueue_top_k(
                        &self.stream,
                        &self.workspace.logits,
                        &self.workspace.top_value,
                        &self.workspace.current_token,
                        1,
                        &self.workspace.top_k_workspace,
                    )?;
                    self.model.runtime.enqueue_record_token(
                        &self.stream,
                        &self.workspace.current_token,
                        &generated,
                        step,
                    )?;
                }
                self.enqueue_current_token()?;
            }
            Ok(())
        })();
        self.finish(enqueue_result)?;
        generated.to_vec(&self.stream)
    }

    fn generate_until_eos(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        eos_token_id: u32,
    ) -> Result<Vec<u32>> {
        self.reset()?;
        let device_ids = DeviceBuffer::from_host(&self.model.context, &self.stream, input_ids)?;
        let enqueue_result = self.enqueue_prefill(&device_ids);
        self.finish(enqueue_result)?;

        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let mut token_host = [0_u32; 1];
            let enqueue_result = (|| {
                // SAFETY: top-1 writes exactly one u32 token. The host
                // destination remains alive and untouched until `finish`
                // synchronizes this stream below.
                unsafe {
                    self.model.top_k.enqueue_top_k(
                        &self.stream,
                        &self.workspace.logits,
                        &self.workspace.top_value,
                        &self.workspace.current_token,
                        1,
                        &self.workspace.top_k_workspace,
                    )?;
                    self.workspace
                        .current_token
                        .copy_to_host_async(&self.stream, &mut token_host)?;
                }
                Ok(())
            })();
            self.finish(enqueue_result)?;

            let token = token_host[0];
            generated.push(token);
            let enqueue_result = self.enqueue_current_token();
            if token == eos_token_id || enqueue_result.is_err() {
                self.finish(enqueue_result)?;
                if token == eos_token_id {
                    return Ok(generated);
                }
            }
        }

        // The final non-EOS token has been submitted but no following
        // top-1 observation exists to provide the synchronization
        // boundary, so drain it explicitly before returning.
        self.finish(Ok(()))?;
        Ok(generated)
    }

    fn enqueue_prefill(&mut self, input_ids: &DeviceBuffer<u32>) -> Result<()> {
        for token_position in 0..input_ids.len() {
            // SAFETY: all three buffers are session/call-owned until `finish`;
            // positions were host-validated before the graph was submitted.
            unsafe {
                self.model.runtime.enqueue_select_token(
                    &self.stream,
                    input_ids,
                    &self.workspace.current_token,
                    token_position,
                )?;
            }
            self.enqueue_current_token()?;
        }
        Ok(())
    }

    fn enqueue_current_token(&mut self) -> Result<()> {
        let config = &self.model.config;
        let position = self.position;
        if position >= config.max_position_embeddings {
            return Err(NnisError::invalid_input(format!(
                "decoder position {position} exceeds max_position_embeddings {}",
                config.max_position_embeddings
            )));
        }

        // SAFETY: `current_token` is either host-validated or produced by NNIS
        // top-1 over exactly `vocab_size` logits, so the gather index is in
        // range. All buffers are owned by the model/session and every dependent
        // access is serialized on this one stream.
        unsafe {
            self.model.gather.enqueue_gather(
                &self.stream,
                self.model.weights.token_embedding.tensor().as_f32()?,
                &self.workspace.current_token,
                &self.workspace.hidden,
                config.vocab_size,
                config.hidden_size,
            )?;
        }

        let attention_scale = 1.0_f32 / (config.head_dim() as f32).sqrt();
        let kv_width = config.key_value_width()?;
        for layer_index in 0..config.num_hidden_layers {
            let layer = &self.model.weights.layers[layer_index];
            // SAFETY: the session owns all mutable buffers exclusively and
            // submits every read/write in dependency order on one stream.
            unsafe {
                self.model.decoder.enqueue_weighted_rms_norm(
                    &self.stream,
                    &self.workspace.hidden,
                    layer.input_norm.tensor().as_f32()?,
                    &self.workspace.normed,
                    1,
                    config.hidden_size,
                    config.rms_norm_eps,
                )?;
                self.model.gemm.enqueue_gemm(
                    &self.stream,
                    &self.workspace.normed,
                    layer.q_proj.tensor().as_f32()?,
                    &self.workspace.q,
                    1,
                    config.hidden_size,
                    config.hidden_size,
                )?;
                self.model.gemm.enqueue_gemm(
                    &self.stream,
                    &self.workspace.normed,
                    layer.k_proj.tensor().as_f32()?,
                    &self.workspace.k,
                    1,
                    kv_width,
                    config.hidden_size,
                )?;
                self.model.gemm.enqueue_gemm(
                    &self.stream,
                    &self.workspace.normed,
                    layer.v_proj.tensor().as_f32()?,
                    &self.workspace.v,
                    1,
                    kv_width,
                    config.hidden_size,
                )?;
                self.model.runtime.enqueue_rope_position(
                    &self.stream,
                    &self.workspace.q,
                    &self.model.rope_cos,
                    &self.model.rope_sin,
                    &self.workspace.q_rope,
                    config.num_attention_heads,
                    config.head_dim(),
                    position,
                    config.max_position_embeddings,
                )?;
                self.model.runtime.enqueue_rope_position(
                    &self.stream,
                    &self.workspace.k,
                    &self.model.rope_cos,
                    &self.model.rope_sin,
                    &self.workspace.k_rope,
                    config.num_key_value_heads,
                    config.head_dim(),
                    position,
                    config.max_position_embeddings,
                )?;
            }

            let append = self.cache.append_layer_async(
                layer_index,
                Arc::clone(&self.workspace.k_rope),
                Arc::clone(&self.workspace.v),
                1,
            )?;
            self.pending_appends.push(append);

            // SAFETY: append copies and the attention consumer are ordered on
            // the cache/session stream; the cache and every workspace buffer
            // remain session-owned through the final synchronization.
            unsafe {
                self.model.decoder.enqueue_cached_attention_decode(
                    &self.stream,
                    &self.workspace.q_rope,
                    &self.cache,
                    layer_index,
                    &self.workspace.attention,
                    attention_scale,
                )?;
                self.model.gemm.enqueue_gemm(
                    &self.stream,
                    &self.workspace.attention,
                    layer.o_proj.tensor().as_f32()?,
                    &self.workspace.projected,
                    1,
                    config.hidden_size,
                    config.hidden_size,
                )?;
                self.model.elementwise.enqueue_vector_add(
                    &self.stream,
                    &self.workspace.hidden,
                    &self.workspace.projected,
                    &self.workspace.residual,
                )?;
                self.model.decoder.enqueue_weighted_rms_norm(
                    &self.stream,
                    &self.workspace.residual,
                    layer.post_attention_norm.tensor().as_f32()?,
                    &self.workspace.normed,
                    1,
                    config.hidden_size,
                    config.rms_norm_eps,
                )?;
                self.model.gemm.enqueue_gemm(
                    &self.stream,
                    &self.workspace.normed,
                    layer.gate_proj.tensor().as_f32()?,
                    &self.workspace.gate,
                    1,
                    config.intermediate_size,
                    config.hidden_size,
                )?;
                self.model.gemm.enqueue_gemm(
                    &self.stream,
                    &self.workspace.normed,
                    layer.up_proj.tensor().as_f32()?,
                    &self.workspace.up,
                    1,
                    config.intermediate_size,
                    config.hidden_size,
                )?;
                self.model.elementwise.enqueue_silu(
                    &self.stream,
                    &self.workspace.gate,
                    &self.workspace.activated,
                )?;
                self.model.decoder.enqueue_multiply(
                    &self.stream,
                    &self.workspace.activated,
                    &self.workspace.up,
                    &self.workspace.gated,
                )?;
                self.model.gemm.enqueue_gemm(
                    &self.stream,
                    &self.workspace.gated,
                    layer.down_proj.tensor().as_f32()?,
                    &self.workspace.mlp,
                    1,
                    config.hidden_size,
                    config.intermediate_size,
                )?;
                self.model.elementwise.enqueue_vector_add(
                    &self.stream,
                    &self.workspace.residual,
                    &self.workspace.mlp,
                    &self.workspace.hidden,
                )?;
            }
        }

        // SAFETY: final norm/logit buffers and immutable weights remain alive
        // for the model/session lifetime and are ordered after all layers.
        unsafe {
            self.model.decoder.enqueue_weighted_rms_norm(
                &self.stream,
                &self.workspace.hidden,
                self.model.weights.final_norm.tensor().as_f32()?,
                &self.workspace.normed,
                1,
                config.hidden_size,
                config.rms_norm_eps,
            )?;
            self.model.gemm.enqueue_gemm(
                &self.stream,
                &self.workspace.normed,
                self.model.weights.lm_head.tensor().as_f32()?,
                &self.workspace.logits,
                1,
                config.vocab_size,
                config.hidden_size,
            )?;
        }
        self.position += 1;
        Ok(())
    }

    fn finish(&mut self, enqueue_result: Result<()>) -> Result<()> {
        let synchronize_result = self.stream.synchronize();
        self.pending_appends.clear();
        match enqueue_result {
            Ok(()) => synchronize_result,
            Err(error) => {
                let _ = synchronize_result;
                Err(error)
            }
        }
    }

    fn validate_prompt(&self, input_ids: &[u32], extra_tokens: usize) -> Result<()> {
        if input_ids.is_empty() {
            return Err(NnisError::invalid_input(
                "decoder prefill requires at least one input token",
            ));
        }
        for &token in input_ids {
            self.validate_token(token)?;
        }
        let total = input_ids.len().checked_add(extra_tokens).ok_or_else(|| {
            NnisError::invalid_input("prompt + generation length overflows usize")
        })?;
        if total > self.capacity() {
            return Err(NnisError::invalid_input(format!(
                "prompt + generation requires {total} positions; session capacity is {}",
                self.capacity()
            )));
        }
        Ok(())
    }

    fn validate_token(&self, token: u32) -> Result<()> {
        if token as usize >= self.model.config.vocab_size {
            return Err(NnisError::invalid_input(format!(
                "token id {token} is out of range for vocabulary {}",
                self.model.config.vocab_size
            )));
        }
        Ok(())
    }
}

fn build_rope_cache(config: &ModelConfig) -> Result<(Vec<f32>, Vec<f32>)> {
    let half = config.head_dim() / 2;
    let elements = config
        .max_position_embeddings
        .checked_mul(half)
        .ok_or_else(|| NnisError::invalid_input("RoPE cache shape overflows usize"))?;
    let mut cos = Vec::with_capacity(elements);
    let mut sin = Vec::with_capacity(elements);
    for position in 0..config.max_position_embeddings {
        for pair in 0..half {
            let exponent = (2 * pair) as f32 / config.head_dim() as f32;
            let inverse_frequency = 1.0_f32 / config.rope_theta.powf(exponent);
            let angle = position as f32 * inverse_frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Activation, DecoderLayerWeights, DeviceTensor, MatrixWeight, VectorWeight};
    use nnis_rt::gpu_context;

    fn matrix(
        context: &Arc<Context>,
        stream: &Stream,
        rows: usize,
        cols: usize,
        values: Vec<f32>,
    ) -> MatrixWeight {
        MatrixWeight::new(
            DeviceTensor::F32(Arc::new(
                DeviceBuffer::from_host(context, stream, &values).unwrap(),
            )),
            rows,
            cols,
        )
        .unwrap()
    }

    fn vector(context: &Arc<Context>, stream: &Stream, values: Vec<f32>) -> VectorWeight {
        let len = values.len();
        VectorWeight::new(
            DeviceTensor::F32(Arc::new(
                DeviceBuffer::from_host(context, stream, &values).unwrap(),
            )),
            len,
        )
        .unwrap()
    }

    #[test]
    fn full_decoder_generation_runs_through_nnis_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let construction_stream = Stream::new(&context).unwrap();
        let config = ModelConfig {
            vocab_size: 4,
            eos_token_id: Some(0),
            hidden_size: 4,
            intermediate_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_position_embeddings: 8,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10_000.0,
            activation: Activation::Silu,
            weight_dtype: WeightDType::F32,
        };
        let embedding = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let zeros = vec![0.0_f32; 16];
        let kv_zeros = vec![0.0_f32; 8];
        let weights = ModelWeights {
            token_embedding: matrix(&context, &construction_stream, 4, 4, embedding),
            layers: vec![DecoderLayerWeights {
                input_norm: vector(&context, &construction_stream, vec![1.0; 4]),
                q_proj: matrix(&context, &construction_stream, 4, 4, zeros.clone()),
                k_proj: matrix(&context, &construction_stream, 4, 2, kv_zeros.clone()),
                v_proj: matrix(&context, &construction_stream, 4, 2, kv_zeros),
                o_proj: matrix(&context, &construction_stream, 4, 4, zeros.clone()),
                post_attention_norm: vector(&context, &construction_stream, vec![1.0; 4]),
                gate_proj: matrix(&context, &construction_stream, 4, 4, zeros.clone()),
                up_proj: matrix(&context, &construction_stream, 4, 4, zeros.clone()),
                down_proj: matrix(&context, &construction_stream, 4, 4, zeros.clone()),
            }],
            final_norm: vector(&context, &construction_stream, vec![1.0; 4]),
            lm_head: matrix(&context, &construction_stream, 4, 4, zeros),
        };
        let model = Model::new(config, weights, &construction_stream).unwrap();
        let mut session = model.new_session().unwrap();
        let generated = session
            .generate(&[1, 2], GenerationConfig::greedy(2))
            .unwrap();
        assert_eq!(generated, vec![0, 0]);
        assert_eq!(session.position(), 4);

        let generated = session
            .generate(&[1, 2], GenerationConfig::greedy_until_eos(4, 0))
            .unwrap();
        assert_eq!(generated, vec![0]);
        assert_eq!(session.position(), 3);
    }
}

#[cfg(test)]
#[path = "smollm2_diagnostic.rs"]
mod smollm2_diagnostic;
