//! Explicit F16 decoder runtime used only for NNML5 reference alignment.
//!
//! This path deliberately remains separate from the historical [`crate::Model`]
//! runtime. NNIS model-format-v1 weights are still loaded as the qualified F32
//! base graph, then explicitly materialized as resident IEEE binary16 tensors.
//! Decoder activations and KV storage are binary16, projection/attention
//! accumulators are F32, and LM-head logits are exposed as F32 after the
//! explicit F16 output boundary implemented by [`crate::F16ReferenceKernels`].
//!
//! The path is correctness/qualification infrastructure. It does not change a
//! runtime default and does not by itself authorize cross-runtime performance
//! claims.

use crate::runtime::build_rope_cache;
use crate::{
    load_model_directory, F16ReferenceExecutionPlan, F16ReferenceKernels,
    F16ReferenceProjectionLayout, F16TransposedProjectionCandidate, F32RuntimeKernels,
    GenerationConfig, MatrixWeight, ModelConfig, ModelWeights, WeightDType,
};
use nnis_jit::JitCompiler;
use nnis_kernels::{F32TopK, F32TopKWorkspace};
use nnis_rt::{Context, DeviceBuffer, KvAppend, KvCache, KvCacheConfig, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

pub const F16_REFERENCE_PLAN_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F16ReferenceStorage {
    F16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F16ReferenceAccumulator {
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F16ReferenceLogits {
    F32,
}

/// Versioned, explicit numeric contract for the NNML5 F16 reference path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct F16ReferencePlan {
    pub schema_version: u32,
    pub weight_storage: F16ReferenceStorage,
    pub activation_storage: F16ReferenceStorage,
    pub kv_storage: F16ReferenceStorage,
    pub projection_accumulator: F16ReferenceAccumulator,
    pub attention_accumulator: F16ReferenceAccumulator,
    pub logits_storage: F16ReferenceLogits,
}

impl F16ReferencePlan {
    /// The only F16 reference contract implemented by schema v1.
    pub const fn edge_llm_v0_10_0_alignment() -> Self {
        Self {
            schema_version: F16_REFERENCE_PLAN_VERSION,
            weight_storage: F16ReferenceStorage::F16,
            activation_storage: F16ReferenceStorage::F16,
            kv_storage: F16ReferenceStorage::F16,
            projection_accumulator: F16ReferenceAccumulator::F32,
            attention_accumulator: F16ReferenceAccumulator::F32,
            logits_storage: F16ReferenceLogits::F32,
        }
    }

    pub fn validate(&self, config: &ModelConfig) -> Result<()> {
        if self.schema_version != F16_REFERENCE_PLAN_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported F16 reference-plan schema {}; expected {}",
                self.schema_version, F16_REFERENCE_PLAN_VERSION
            )));
        }
        config.validate_execution_support()?;
        if config.weight_dtype != WeightDType::F32 {
            return Err(NnisError::unsupported(
                "F16 reference runtime requires the model-format-v1 F32 base graph before explicit resident narrowing",
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct F16DecoderLayerWeights {
    input_norm: Arc<DeviceBuffer<u16>>,
    q_proj: Arc<DeviceBuffer<u16>>,
    k_proj: Arc<DeviceBuffer<u16>>,
    v_proj: Arc<DeviceBuffer<u16>>,
    o_proj: Arc<DeviceBuffer<u16>>,
    post_attention_norm: Arc<DeviceBuffer<u16>>,
    gate_proj: Arc<DeviceBuffer<u16>>,
    up_proj: Arc<DeviceBuffer<u16>>,
    down_proj: Arc<DeviceBuffer<u16>>,
}

#[derive(Debug)]
struct F16ModelWeights {
    token_embedding: Arc<DeviceBuffer<u16>>,
    layers: Vec<F16DecoderLayerWeights>,
    final_norm: Arc<DeviceBuffer<u16>>,
    lm_head: Arc<DeviceBuffer<u16>>,
}

impl F16ModelWeights {
    fn from_f32(
        source: &ModelWeights,
        stream: &Stream,
        kernels: &F16ReferenceKernels,
        execution_plan: F16ReferenceExecutionPlan,
        projection_candidate: Option<&F16TransposedProjectionCandidate>,
    ) -> Result<Self> {
        fn narrow(
            stream: &Stream,
            kernels: &F16ReferenceKernels,
            source: &DeviceBuffer<f32>,
        ) -> Result<Arc<DeviceBuffer<u16>>> {
            let output = Arc::new(DeviceBuffer::<u16>::new(stream.ctx(), source.len())?);
            // SAFETY: source/output stay alive across the immediately following
            // stream synchronization, so no async conversion outlives either buffer.
            unsafe { kernels.enqueue_narrow_from_f32(stream, source, &output)? };
            stream.synchronize()?;
            Ok(output)
        }

        fn narrow_projection(
            stream: &Stream,
            kernels: &F16ReferenceKernels,
            source: &MatrixWeight,
            layout: F16ReferenceProjectionLayout,
            projection_candidate: Option<&F16TransposedProjectionCandidate>,
        ) -> Result<Arc<DeviceBuffer<u16>>> {
            let kn = narrow(stream, kernels, source.tensor().as_f32()?)?;
            match layout {
                F16ReferenceProjectionLayout::KnReference => Ok(kn),
                F16ReferenceProjectionLayout::NkTransposedCandidate => {
                    let candidate = projection_candidate.ok_or_else(|| {
                        NnisError::unsupported(
                            "F16 transposed projection plan selected without candidate kernels",
                        )
                    })?;
                    let nk = Arc::new(DeviceBuffer::<u16>::new(stream.ctx(), kn.len())?);
                    // SAFETY: both buffers remain alive through synchronization.
                    unsafe {
                        candidate.enqueue_transpose_kn_to_nk(
                            stream,
                            &kn,
                            &nk,
                            source.rows(),
                            source.cols(),
                        )?;
                    }
                    stream.synchronize()?;
                    Ok(nk)
                }
            }
        }

        let layout = execution_plan.projection_layout;
        let token_embedding = narrow(stream, kernels, source.token_embedding.tensor().as_f32()?)?;
        let mut layers = Vec::with_capacity(source.layers.len());
        for layer in &source.layers {
            layers.push(F16DecoderLayerWeights {
                input_norm: narrow(stream, kernels, layer.input_norm.tensor().as_f32()?)?,
                q_proj: narrow_projection(
                    stream,
                    kernels,
                    &layer.q_proj,
                    layout,
                    projection_candidate,
                )?,
                k_proj: narrow_projection(
                    stream,
                    kernels,
                    &layer.k_proj,
                    layout,
                    projection_candidate,
                )?,
                v_proj: narrow_projection(
                    stream,
                    kernels,
                    &layer.v_proj,
                    layout,
                    projection_candidate,
                )?,
                o_proj: narrow_projection(
                    stream,
                    kernels,
                    &layer.o_proj,
                    layout,
                    projection_candidate,
                )?,
                post_attention_norm: narrow(
                    stream,
                    kernels,
                    layer.post_attention_norm.tensor().as_f32()?,
                )?,
                gate_proj: narrow_projection(
                    stream,
                    kernels,
                    &layer.gate_proj,
                    layout,
                    projection_candidate,
                )?,
                up_proj: narrow_projection(
                    stream,
                    kernels,
                    &layer.up_proj,
                    layout,
                    projection_candidate,
                )?,
                down_proj: narrow_projection(
                    stream,
                    kernels,
                    &layer.down_proj,
                    layout,
                    projection_candidate,
                )?,
            });
        }
        let final_norm = narrow(stream, kernels, source.final_norm.tensor().as_f32()?)?;
        let lm_head = narrow_projection(
            stream,
            kernels,
            &source.lm_head,
            layout,
            projection_candidate,
        )?;
        Ok(Self {
            token_embedding,
            layers,
            final_norm,
            lm_head,
        })
    }
}

/// Immutable F16 reference-alignment model. Sessions own mutable state.
#[derive(Debug)]
pub struct F16ReferenceModel {
    config: ModelConfig,
    execution_plan: F16ReferenceExecutionPlan,
    weights: F16ModelWeights,
    context: Arc<Context>,
    kernels: F16ReferenceKernels,
    projection_candidate: Option<F16TransposedProjectionCandidate>,
    top_k: F32TopK,
    token_runtime: F32RuntimeKernels,
    rope_cos: DeviceBuffer<f32>,
    rope_sin: DeviceBuffer<f32>,
}

impl F16ReferenceModel {
    pub fn new(
        config: ModelConfig,
        weights: ModelWeights,
        stream: &Stream,
        plan: F16ReferencePlan,
    ) -> Result<Self> {
        Self::new_with_execution_plan(
            config,
            weights,
            stream,
            F16ReferenceExecutionPlan::reference(plan),
        )
    }

    pub fn new_with_execution_plan(
        config: ModelConfig,
        weights: ModelWeights,
        stream: &Stream,
        execution_plan: F16ReferenceExecutionPlan,
    ) -> Result<Self> {
        execution_plan.validate(&config)?;
        weights.validate(&config)?;
        if !Arc::ptr_eq(weights.context(), stream.ctx()) {
            return Err(NnisError::invalid_input(
                "F16 reference source weights and construction stream must share one CUDA context",
            ));
        }

        let context = Arc::clone(stream.ctx());
        let compiler = JitCompiler::new();
        let kernels = F16ReferenceKernels::load(&context, &compiler)?;
        let projection_candidate = match execution_plan.projection_layout {
            F16ReferenceProjectionLayout::KnReference => None,
            F16ReferenceProjectionLayout::NkTransposedCandidate => Some(
                F16TransposedProjectionCandidate::load(&context, &compiler)?,
            ),
        };
        let top_k = F32TopK::load(&context, &compiler)?;
        let token_runtime = F32RuntimeKernels::load(&context, &compiler)?;
        let resident_weights = F16ModelWeights::from_f32(
            &weights,
            stream,
            &kernels,
            execution_plan,
            projection_candidate.as_ref(),
        )?;
        let (cos_host, sin_host) = build_rope_cache(&config)?;
        let rope_cos = DeviceBuffer::from_host(&context, stream, &cos_host)?;
        let rope_sin = DeviceBuffer::from_host(&context, stream, &sin_host)?;
        stream.synchronize()?;

        Ok(Self {
            config,
            execution_plan,
            weights: resident_weights,
            context,
            kernels,
            projection_candidate,
            top_k,
            token_runtime,
            rope_cos,
            rope_sin,
        })
    }

    pub fn load_directory(
        context: &Arc<Context>,
        stream: &Stream,
        directory: impl AsRef<Path>,
        plan: F16ReferencePlan,
    ) -> Result<Self> {
        let (config, weights) = load_model_directory(context, stream, directory)?;
        Self::new(config, weights, stream, plan)
    }

    pub fn load_directory_with_execution_plan(
        context: &Arc<Context>,
        stream: &Stream,
        directory: impl AsRef<Path>,
        execution_plan: F16ReferenceExecutionPlan,
    ) -> Result<Self> {
        let (config, weights) = load_model_directory(context, stream, directory)?;
        Self::new_with_execution_plan(config, weights, stream, execution_plan)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    #[must_use]
    pub const fn plan(&self) -> F16ReferencePlan {
        self.execution_plan.numeric
    }

    #[must_use]
    pub const fn execution_plan(&self) -> F16ReferenceExecutionPlan {
        self.execution_plan
    }

    pub fn new_session(&self) -> Result<F16ReferenceSession<'_>> {
        F16ReferenceSession::new(self)
    }

    unsafe fn enqueue_projection(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        weight: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        match self.execution_plan.projection_layout {
            F16ReferenceProjectionLayout::KnReference => unsafe {
                self.kernels
                    .enqueue_project_kn(stream, input, weight, output, k, n)
            },
            F16ReferenceProjectionLayout::NkTransposedCandidate => unsafe {
                self.projection_candidate
                    .as_ref()
                    .ok_or_else(|| {
                        NnisError::unsupported(
                            "F16 transposed projection plan selected without candidate kernels",
                        )
                    })?
                    .enqueue_project_nk(stream, input, weight, output, k, n)
            },
        }
    }

    unsafe fn enqueue_lm_head(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<u16>,
        weight: &DeviceBuffer<u16>,
        output: &DeviceBuffer<f32>,
        k: usize,
        n: usize,
    ) -> Result<()> {
        match self.execution_plan.projection_layout {
            F16ReferenceProjectionLayout::KnReference => unsafe {
                self.kernels
                    .enqueue_lm_head_f32_logits(stream, input, weight, output, k, n)
            },
            F16ReferenceProjectionLayout::NkTransposedCandidate => unsafe {
                self.projection_candidate
                    .as_ref()
                    .ok_or_else(|| {
                        NnisError::unsupported(
                            "F16 transposed projection plan selected without candidate kernels",
                        )
                    })?
                    .enqueue_lm_head_nk_f32_logits(stream, input, weight, output, k, n)
            },
        }
    }
}

#[derive(Debug)]
struct F16DecodeWorkspace {
    hidden: DeviceBuffer<u16>,
    normed: DeviceBuffer<u16>,
    q: DeviceBuffer<u16>,
    k: DeviceBuffer<u16>,
    v: Arc<DeviceBuffer<u16>>,
    q_rope: DeviceBuffer<u16>,
    k_rope: Arc<DeviceBuffer<u16>>,
    attention: DeviceBuffer<u16>,
    projected: DeviceBuffer<u16>,
    residual: DeviceBuffer<u16>,
    gate: DeviceBuffer<u16>,
    up: DeviceBuffer<u16>,
    gated: DeviceBuffer<u16>,
    mlp: DeviceBuffer<u16>,
    logits: DeviceBuffer<f32>,
    current_token: DeviceBuffer<u32>,
    top_value: DeviceBuffer<f32>,
    top_k_workspace: F32TopKWorkspace,
}

impl F16DecodeWorkspace {
    fn new(model: &F16ReferenceModel) -> Result<Self> {
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
            gated: DeviceBuffer::new(context, intermediate)?,
            mlp: DeviceBuffer::new(context, hidden)?,
            logits: DeviceBuffer::new(context, vocab)?,
            current_token: DeviceBuffer::new(context, 1)?,
            top_value: DeviceBuffer::new(context, 1)?,
            top_k_workspace: model.top_k.workspace(context, vocab)?,
        })
    }
}

/// Mutable autoregressive state for the explicit F16 reference model.
#[derive(Debug)]
pub struct F16ReferenceSession<'model> {
    model: &'model F16ReferenceModel,
    stream: Stream,
    cache: KvCache<u16>,
    workspace: F16DecodeWorkspace,
    pending_appends: Vec<KvAppend<u16>>,
    position: usize,
}

/// CUDA-event profile matching the TensorRT Edge-LLM generation-stage metric.
///
/// For `N` generated tokens this path samples token 0 from prefill logits and
/// measures exactly `N - 1` decoder forwards. Top-1 sampling and token recording
/// are deliberately outside every generation-forward CUDA-event interval. The
/// final generated token is not consumed by the model because Edge-LLM likewise
/// reports 32 generated tokens from 31 `llm_generation` stage executions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F16ReferenceGenerationProfile {
    pub schema_version: u32,
    pub metric_definition: String,
    pub generated_tokens: usize,
    pub generation_forward_runs: usize,
    pub prefill_gpu_ms: f64,
    pub generation_forward_gpu_ms: Vec<f64>,
    pub generation_stage_total_gpu_ms: f64,
    pub generation_tokens_per_second_edge_definition: f64,
    pub generated_ids: Vec<u32>,
    pub session_position_after_profile: usize,
    pub sampling_included_in_generation_stage_gpu_time: bool,
    pub final_generated_token_consumed_by_model: bool,
}

impl<'model> F16ReferenceSession<'model> {
    fn new(model: &'model F16ReferenceModel) -> Result<Self> {
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
        let workspace = F16DecodeWorkspace::new(model)?;
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

    pub fn prefill(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        self.validate_prompt(input_ids, 0)?;
        self.reset()?;
        let device_ids = DeviceBuffer::from_host(&self.model.context, &self.stream, input_ids)?;
        let enqueue_result = self.enqueue_prefill(&device_ids);
        self.finish(enqueue_result)?;
        self.workspace.logits.to_vec(&self.stream)
    }

    pub fn decode_one(&mut self, token: u32) -> Result<Vec<f32>> {
        self.validate_token(token)?;
        if self.position >= self.capacity() {
            return Err(NnisError::invalid_input(format!(
                "F16 decode position {} exceeds session capacity {}",
                self.position,
                self.capacity()
            )));
        }
        let host = [token];
        let enqueue_result = (|| {
            // SAFETY: host/current_token stay alive through `finish` below.
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

    /// Profile a fixed greedy request with the same generation-stage accounting
    /// used by the qualified TensorRT Edge-LLM v0.10.0 R1 reference.
    ///
    /// This is a qualification-only metric. It intentionally leaves the session
    /// one token behind normal `generate()` state advancement because the final
    /// generated token is sampled but not consumed by a decoder forward.
    pub fn profile_greedy_edge_generation_semantics(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<F16ReferenceGenerationProfile> {
        if max_new_tokens < 2 {
            return Err(NnisError::invalid_input(
                "Edge-compatible F16 generation profiling requires at least two generated tokens",
            ));
        }
        self.validate_prompt(input_ids, max_new_tokens - 1)?;
        self.reset()?;

        let device_ids = DeviceBuffer::from_host(&self.model.context, &self.stream, input_ids)?;
        let generated = DeviceBuffer::<u32>::new(&self.model.context, max_new_tokens)?;
        let prefill_start = nnis_rt::Event::new(&self.model.context)?;
        let prefill_end = nnis_rt::Event::new(&self.model.context)?;
        let mut generation_events = Vec::with_capacity(max_new_tokens - 1);

        let enqueue_result = (|| {
            prefill_start.record(&self.stream)?;
            self.enqueue_prefill(&device_ids)?;
            prefill_end.record(&self.stream)?;

            // Token 0 is sampled from prefill logits and is outside the
            // generation-stage CUDA timing, matching the NVIDIA definition.
            // SAFETY: buffers remain live through the final stream sync below.
            unsafe {
                self.model.top_k.enqueue_top_k(
                    &self.stream,
                    &self.workspace.logits,
                    &self.workspace.top_value,
                    &self.workspace.current_token,
                    1,
                    &self.workspace.top_k_workspace,
                )?;
                self.model.token_runtime.enqueue_record_token(
                    &self.stream,
                    &self.workspace.current_token,
                    &generated,
                    0,
                )?;
            }

            for step in 1..max_new_tokens {
                let start = nnis_rt::Event::new(&self.model.context)?;
                let end = nnis_rt::Event::new(&self.model.context)?;
                start.record(&self.stream)?;
                self.enqueue_current_token()?;
                end.record(&self.stream)?;

                // SAFETY: top-1 and record are ordered after `end`, so their GPU
                // work is excluded from the decoder-forward timing interval.
                unsafe {
                    self.model.top_k.enqueue_top_k(
                        &self.stream,
                        &self.workspace.logits,
                        &self.workspace.top_value,
                        &self.workspace.current_token,
                        1,
                        &self.workspace.top_k_workspace,
                    )?;
                    self.model.token_runtime.enqueue_record_token(
                        &self.stream,
                        &self.workspace.current_token,
                        &generated,
                        step,
                    )?;
                }
                generation_events.push((start, end));
            }
            Ok(())
        })();
        self.finish(enqueue_result)?;

        prefill_end.synchronize()?;
        let prefill_gpu_ms = prefill_end.elapsed_ms(&prefill_start)?;
        let mut generation_forward_gpu_ms = Vec::with_capacity(generation_events.len());
        for (start, end) in &generation_events {
            end.synchronize()?;
            generation_forward_gpu_ms.push(end.elapsed_ms(start)?);
        }
        let generation_stage_total_gpu_ms: f64 = generation_forward_gpu_ms.iter().sum();
        if !generation_stage_total_gpu_ms.is_finite() || generation_stage_total_gpu_ms <= 0.0 {
            return Err(NnisError::unsupported(
                "F16 generation-stage CUDA events returned non-positive total GPU time",
            ));
        }
        let generated_ids = generated.to_vec(&self.stream)?;
        let generation_tokens_per_second_edge_definition =
            max_new_tokens as f64 / (generation_stage_total_gpu_ms / 1_000.0);

        Ok(F16ReferenceGenerationProfile {
            schema_version: 1,
            metric_definition: "generated_tokens / cumulative GPU time of N-1 decoder forwards after prefill; CUDA events bracket decoder forward only; top-1 sampling excluded; final generated token not consumed"
                .to_string(),
            generated_tokens: max_new_tokens,
            generation_forward_runs: generation_forward_gpu_ms.len(),
            prefill_gpu_ms,
            generation_forward_gpu_ms,
            generation_stage_total_gpu_ms,
            generation_tokens_per_second_edge_definition,
            generated_ids,
            session_position_after_profile: self.position,
            sampling_included_in_generation_stage_gpu_time: false,
            final_generated_token_consumed_by_model: false,
        })
    }

    fn generate_fixed(&mut self, input_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        self.reset()?;
        let device_ids = DeviceBuffer::from_host(&self.model.context, &self.stream, input_ids)?;
        let generated = DeviceBuffer::<u32>::new(&self.model.context, max_new_tokens)?;
        let enqueue_result = (|| {
            self.enqueue_prefill(&device_ids)?;
            for step in 0..max_new_tokens {
                // SAFETY: session-owned buffers remain live and are serialized
                // on one stream for the whole fixed-length graph.
                unsafe {
                    self.model.top_k.enqueue_top_k(
                        &self.stream,
                        &self.workspace.logits,
                        &self.workspace.top_value,
                        &self.workspace.current_token,
                        1,
                        &self.workspace.top_k_workspace,
                    )?;
                    self.model.token_runtime.enqueue_record_token(
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
                // SAFETY: token_host/current_token remain alive until `finish`.
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
        self.finish(Ok(()))?;
        Ok(generated)
    }

    fn enqueue_prefill(&mut self, input_ids: &DeviceBuffer<u32>) -> Result<()> {
        for token_position in 0..input_ids.len() {
            // SAFETY: buffers remain session/call-owned until the final finish.
            unsafe {
                self.model.token_runtime.enqueue_select_token(
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
                "F16 decoder position {position} exceeds max_position_embeddings {}",
                config.max_position_embeddings
            )));
        }

        // SAFETY: current_token is validated or produced by top-1 over exactly
        // vocab_size logits; all dependent accesses are ordered on one stream.
        unsafe {
            self.model.kernels.enqueue_gather(
                &self.stream,
                &self.model.weights.token_embedding,
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
            // SAFETY: one session exclusively owns every mutable tensor and
            // submits the decoder graph in dependency order on one stream.
            unsafe {
                self.model.kernels.enqueue_weighted_rms_norm(
                    &self.stream,
                    &self.workspace.hidden,
                    &layer.input_norm,
                    &self.workspace.normed,
                    1,
                    config.hidden_size,
                    config.rms_norm_eps,
                )?;
                self.model.enqueue_projection(
                    &self.stream,
                    &self.workspace.normed,
                    &layer.q_proj,
                    &self.workspace.q,
                    config.hidden_size,
                    config.hidden_size,
                )?;
                self.model.enqueue_projection(
                    &self.stream,
                    &self.workspace.normed,
                    &layer.k_proj,
                    &self.workspace.k,
                    config.hidden_size,
                    kv_width,
                )?;
                self.model.enqueue_projection(
                    &self.stream,
                    &self.workspace.normed,
                    &layer.v_proj,
                    &self.workspace.v,
                    config.hidden_size,
                    kv_width,
                )?;
                self.model.kernels.enqueue_rope_position(
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
                self.model.kernels.enqueue_rope_position(
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

            // SAFETY: cache append and all consumers share this session stream.
            unsafe {
                self.model.kernels.enqueue_cached_attention_decode(
                    &self.stream,
                    &self.workspace.q_rope,
                    &self.cache,
                    layer_index,
                    &self.workspace.attention,
                    attention_scale,
                )?;
                self.model.enqueue_projection(
                    &self.stream,
                    &self.workspace.attention,
                    &layer.o_proj,
                    &self.workspace.projected,
                    config.hidden_size,
                    config.hidden_size,
                )?;
                self.model.kernels.enqueue_vector_add(
                    &self.stream,
                    &self.workspace.hidden,
                    &self.workspace.projected,
                    &self.workspace.residual,
                )?;
                self.model.kernels.enqueue_weighted_rms_norm(
                    &self.stream,
                    &self.workspace.residual,
                    &layer.post_attention_norm,
                    &self.workspace.normed,
                    1,
                    config.hidden_size,
                    config.rms_norm_eps,
                )?;
                self.model.enqueue_projection(
                    &self.stream,
                    &self.workspace.normed,
                    &layer.gate_proj,
                    &self.workspace.gate,
                    config.hidden_size,
                    config.intermediate_size,
                )?;
                self.model.enqueue_projection(
                    &self.stream,
                    &self.workspace.normed,
                    &layer.up_proj,
                    &self.workspace.up,
                    config.hidden_size,
                    config.intermediate_size,
                )?;
                self.model.kernels.enqueue_silu_multiply(
                    &self.stream,
                    &self.workspace.gate,
                    &self.workspace.up,
                    &self.workspace.gated,
                )?;
                self.model.enqueue_projection(
                    &self.stream,
                    &self.workspace.gated,
                    &layer.down_proj,
                    &self.workspace.mlp,
                    config.intermediate_size,
                    config.hidden_size,
                )?;
                self.model.kernels.enqueue_vector_add(
                    &self.stream,
                    &self.workspace.residual,
                    &self.workspace.mlp,
                    &self.workspace.hidden,
                )?;
            }
        }

        // SAFETY: final norm/logit buffers and immutable resident F16 weights
        // remain alive for the complete model/session lifetime.
        unsafe {
            self.model.kernels.enqueue_weighted_rms_norm(
                &self.stream,
                &self.workspace.hidden,
                &self.model.weights.final_norm,
                &self.workspace.normed,
                1,
                config.hidden_size,
                config.rms_norm_eps,
            )?;
            self.model.enqueue_lm_head(
                &self.stream,
                &self.workspace.normed,
                &self.model.weights.lm_head,
                &self.workspace.logits,
                config.hidden_size,
                config.vocab_size,
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
                "F16 decoder prefill requires at least one input token",
            ));
        }
        for &token in input_ids {
            self.validate_token(token)?;
        }
        let total = input_ids.len().checked_add(extra_tokens).ok_or_else(|| {
            NnisError::invalid_input("F16 prompt + generation length overflows usize")
        })?;
        if total > self.capacity() {
            return Err(NnisError::invalid_input(format!(
                "F16 prompt + generation requires {total} positions; session capacity is {}",
                self.capacity()
            )));
        }
        Ok(())
    }

    fn validate_token(&self, token: u32) -> Result<()> {
        if token as usize >= self.model.config.vocab_size {
            return Err(NnisError::invalid_input(format!(
                "F16 token id {token} is out of range for vocabulary {}",
                self.model.config.vocab_size
            )));
        }
        Ok(())
    }
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

    fn tiny_config() -> ModelConfig {
        ModelConfig {
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
        }
    }

    fn tiny_weights(context: &Arc<Context>, stream: &Stream) -> ModelWeights {
        let embedding = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let zeros = vec![0.0_f32; 16];
        let kv_zeros = vec![0.0_f32; 8];
        ModelWeights {
            token_embedding: matrix(context, stream, 4, 4, embedding),
            layers: vec![DecoderLayerWeights {
                input_norm: vector(context, stream, vec![1.0; 4]),
                q_proj: matrix(context, stream, 4, 4, zeros.clone()),
                k_proj: matrix(context, stream, 4, 2, kv_zeros.clone()),
                v_proj: matrix(context, stream, 4, 2, kv_zeros),
                o_proj: matrix(context, stream, 4, 4, zeros.clone()),
                post_attention_norm: vector(context, stream, vec![1.0; 4]),
                gate_proj: matrix(context, stream, 4, 4, zeros.clone()),
                up_proj: matrix(context, stream, 4, 4, zeros.clone()),
                down_proj: matrix(context, stream, 4, 4, zeros.clone()),
            }],
            final_norm: vector(context, stream, vec![1.0; 4]),
            lm_head: matrix(context, stream, 4, 4, zeros),
        }
    }

    #[test]
    fn reference_plan_is_versioned_and_fail_closed() {
        let config = tiny_config();
        let plan = F16ReferencePlan::edge_llm_v0_10_0_alignment();
        plan.validate(&config).unwrap();
        let encoded = serde_json::to_string(&plan).unwrap();
        assert!(encoded.contains("\"schema_version\":1"));
        assert!(encoded.contains("\"weight_storage\":\"f16\""));
        let mut future = plan;
        future.schema_version = F16_REFERENCE_PLAN_VERSION + 1;
        assert!(future.validate(&config).is_err());

        let mut unsupported = config;
        unsupported.weight_dtype = WeightDType::Bf16;
        assert!(plan.validate(&unsupported).is_err());
    }

    #[test]
    fn full_f16_reference_generation_runs_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let construction_stream = Stream::new(&context).unwrap();
        let model = F16ReferenceModel::new(
            tiny_config(),
            tiny_weights(&context, &construction_stream),
            &construction_stream,
            F16ReferencePlan::edge_llm_v0_10_0_alignment(),
        )
        .unwrap();
        assert_eq!(
            model.execution_plan().projection_layout,
            F16ReferenceProjectionLayout::KnReference
        );
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

    #[test]
    fn transposed_projection_execution_plan_runs_same_tiny_generation_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let construction_stream = Stream::new(&context).unwrap();
        let reference_model = F16ReferenceModel::new(
            tiny_config(),
            tiny_weights(&context, &construction_stream),
            &construction_stream,
            F16ReferencePlan::edge_llm_v0_10_0_alignment(),
        )
        .unwrap();
        let candidate_model = F16ReferenceModel::new_with_execution_plan(
            tiny_config(),
            tiny_weights(&context, &construction_stream),
            &construction_stream,
            F16ReferenceExecutionPlan::edge_llm_v0_10_0_transposed_projection_candidate(),
        )
        .unwrap();
        assert_eq!(
            candidate_model.execution_plan().projection_layout,
            F16ReferenceProjectionLayout::NkTransposedCandidate
        );

        let reference_logits = reference_model
            .new_session()
            .unwrap()
            .prefill(&[1, 2])
            .unwrap();
        let candidate_logits = candidate_model
            .new_session()
            .unwrap()
            .prefill(&[1, 2])
            .unwrap();
        assert_eq!(
            reference_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            candidate_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        let reference_ids = reference_model
            .new_session()
            .unwrap()
            .generate(&[1, 2], GenerationConfig::greedy(2))
            .unwrap();
        let candidate_ids = candidate_model
            .new_session()
            .unwrap()
            .generate(&[1, 2], GenerationConfig::greedy(2))
            .unwrap();
        assert_eq!(reference_ids, candidate_ids);
    }

    #[test]
    fn edge_generation_profile_counts_n_minus_one_forwards() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let construction_stream = Stream::new(&context).unwrap();
        let model = F16ReferenceModel::new(
            tiny_config(),
            tiny_weights(&context, &construction_stream),
            &construction_stream,
            F16ReferencePlan::edge_llm_v0_10_0_alignment(),
        )
        .unwrap();
        let mut session = model.new_session().unwrap();
        let profile = session
            .profile_greedy_edge_generation_semantics(&[1, 2], 2)
            .unwrap();
        assert_eq!(profile.generated_ids, vec![0, 0]);
        assert_eq!(profile.generated_tokens, 2);
        assert_eq!(profile.generation_forward_runs, 1);
        assert_eq!(profile.generation_forward_gpu_ms.len(), 1);
        assert!(profile.prefill_gpu_ms > 0.0);
        assert!(profile.generation_stage_total_gpu_ms > 0.0);
        assert!(profile.generation_tokens_per_second_edge_definition > 0.0);
        assert_eq!(profile.session_position_after_profile, 3);
        assert!(!profile.sampling_included_in_generation_stage_gpu_time);
        assert!(!profile.final_generated_token_consumed_by_model);
    }
}
