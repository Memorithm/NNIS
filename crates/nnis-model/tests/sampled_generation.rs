use nnis_model::{
    Activation, DecoderLayerWeights, DeviceTensor, GenerationConfig, GenerationStreamControl,
    MatrixWeight, Model, ModelConfig, ModelWeights, SampledBatchRequest, SamplingConfig,
    VectorWeight, WeightDType,
};
use nnis_rt::{gpu_context, Context, DeviceBuffer, Stream};
use std::sync::Arc;

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

fn equal_logit_model(context: &Arc<Context>, stream: &Stream) -> Model {
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
    };
    Model::new(config, weights, stream).unwrap()
}

#[test]
fn sampled_generation_is_seed_reproducible_on_gpu() {
    let Some(context) = gpu_context() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let construction_stream = Stream::new(&context).unwrap();
    let model = equal_logit_model(&context, &construction_stream);

    let mut first = model.new_session().unwrap();
    let first_tokens = first
        .generate_sampled(
            &[1, 2],
            GenerationConfig::fixed(3),
            SamplingConfig::seeded(42),
        )
        .unwrap();
    assert_eq!(first_tokens, vec![2, 0, 1]);
    assert_eq!(first.position(), 5);

    let mut second = model.new_session().unwrap();
    let second_tokens = second
        .generate_sampled(
            &[1, 2],
            GenerationConfig::fixed(3),
            SamplingConfig::seeded(42),
        )
        .unwrap();
    assert_eq!(second_tokens, first_tokens);
    assert_eq!(second.position(), 5);
}

#[test]
fn sampled_generation_observes_requested_eos_on_gpu() {
    let Some(context) = gpu_context() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let construction_stream = Stream::new(&context).unwrap();
    let model = equal_logit_model(&context, &construction_stream);
    let mut session = model.new_session().unwrap();

    let generated = session
        .generate_sampled(
            &[1, 2],
            GenerationConfig::until_eos(4, 2),
            SamplingConfig::seeded(42),
        )
        .unwrap();
    assert_eq!(generated, vec![2]);
    assert_eq!(session.position(), 3);
}

#[test]
fn sampled_streaming_delivers_tokens_and_stops_continuation_ready_on_gpu() {
    let Some(context) = gpu_context() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let construction_stream = Stream::new(&context).unwrap();
    let model = equal_logit_model(&context, &construction_stream);
    let mut session = model.new_session().unwrap();
    let mut streamed = Vec::new();

    let generated = session
        .generate_sampled_streaming(
            &[1, 2],
            GenerationConfig::fixed(4),
            SamplingConfig::seeded(42),
            |token| {
                streamed.push(token);
                if streamed.len() == 2 {
                    GenerationStreamControl::Stop
                } else {
                    GenerationStreamControl::Continue
                }
            },
        )
        .unwrap();

    assert_eq!(streamed, vec![2, 0]);
    assert_eq!(generated, streamed);
    assert_eq!(session.position(), 4);

    let continuation_logits = session.decode_one(1).unwrap();
    assert_eq!(continuation_logits.len(), 4);
    assert_eq!(session.position(), 5);
}

#[test]
fn sampled_streaming_reports_eos_before_terminating_on_gpu() {
    let Some(context) = gpu_context() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let construction_stream = Stream::new(&context).unwrap();
    let model = equal_logit_model(&context, &construction_stream);
    let mut session = model.new_session().unwrap();
    let mut streamed = Vec::new();

    let generated = session
        .generate_sampled_streaming(
            &[1, 2],
            GenerationConfig::until_eos(4, 2),
            SamplingConfig::seeded(42),
            |token| {
                streamed.push(token);
                GenerationStreamControl::Continue
            },
        )
        .unwrap();

    assert_eq!(streamed, vec![2]);
    assert_eq!(generated, streamed);
    assert_eq!(session.position(), 3);
}

#[test]
fn sampled_session_batch_preserves_independent_reproducible_state_on_gpu() {
    let Some(context) = gpu_context() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let construction_stream = Stream::new(&context).unwrap();
    let model = equal_logit_model(&context, &construction_stream);
    let mut batch = model.new_sampled_session_batch(2).unwrap();

    let requests = vec![
        SampledBatchRequest::new(
            vec![1, 2],
            GenerationConfig::fixed(3),
            SamplingConfig::seeded(42),
        ),
        SampledBatchRequest::new(
            vec![1, 2],
            GenerationConfig::fixed(3),
            SamplingConfig::seeded(42),
        ),
    ];
    let outcomes = batch.generate_sampled(&requests).unwrap();

    assert_eq!(outcomes.len(), 2);
    assert_eq!(outcomes[0].as_ref().unwrap(), &vec![2, 0, 1]);
    assert_eq!(outcomes[1].as_ref().unwrap(), &vec![2, 0, 1]);
    assert_eq!(batch.positions(), vec![5, 5]);
}

#[test]
fn sampled_session_batch_shape_mismatch_is_atomic_and_item_failure_is_isolated_on_gpu() {
    let Some(context) = gpu_context() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let construction_stream = Stream::new(&context).unwrap();
    let model = equal_logit_model(&context, &construction_stream);
    let mut batch = model.new_sampled_session_batch(2).unwrap();

    let short = vec![SampledBatchRequest::new(
        vec![1, 2],
        GenerationConfig::fixed(1),
        SamplingConfig::seeded(42),
    )];
    assert!(batch.generate_sampled(&short).is_err());
    assert_eq!(batch.positions(), vec![0, 0]);

    let requests = vec![
        SampledBatchRequest::new(
            vec![1, 2],
            GenerationConfig::fixed(2),
            SamplingConfig::seeded(42).with_top_k(0),
        ),
        SampledBatchRequest::new(
            vec![1, 2],
            GenerationConfig::fixed(2),
            SamplingConfig::seeded(42),
        ),
    ];
    let outcomes = batch.generate_sampled(&requests).unwrap();
    assert!(outcomes[0].is_err());
    assert_eq!(outcomes[1].as_ref().unwrap(), &vec![2, 0]);
    assert_eq!(batch.positions()[1], 4);
}
