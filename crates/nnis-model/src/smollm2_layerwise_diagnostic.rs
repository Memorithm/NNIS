use super::*;
use nnis_rt::{gpu_context, DeviceBuffer};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const TRACE_FORMAT: &str = "nnis-smollm2-layerwise";
const TRACE_VERSION: u32 = 1;
const SOURCE_REPO: &str = "HuggingFaceTB/SmolLM2-135M";
const SOURCE_REVISION: &str = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2";
const SOURCE_MODEL_SHA256: &str =
    "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1";

#[derive(Debug, Deserialize)]
struct TraceStage {
    name: String,
    file: String,
    elements: usize,
}

#[derive(Debug, Deserialize)]
struct TraceManifest {
    format: String,
    version: u32,
    source_repo: String,
    source_revision: String,
    source_model_sha256: String,
    transformers_version: String,
    source_weight_dtype: String,
    execution_weight_dtype: String,
    input_ids: Vec<u32>,
    hidden_size: usize,
    num_hidden_layers: usize,
    stages: Vec<TraceStage>,
}

fn load_manifest(reference_dir: &Path) -> TraceManifest {
    let bytes = fs::read(reference_dir.join("trace.json")).expect("read SmolLM2 layerwise trace");
    let manifest: TraceManifest = serde_json::from_slice(&bytes).expect("parse SmolLM2 trace");
    assert_eq!(manifest.format, TRACE_FORMAT);
    assert_eq!(manifest.version, TRACE_VERSION);
    assert_eq!(manifest.source_repo, SOURCE_REPO);
    assert_eq!(manifest.source_revision, SOURCE_REVISION);
    assert_eq!(manifest.source_model_sha256, SOURCE_MODEL_SHA256);
    assert_eq!(manifest.transformers_version, "4.40.1");
    assert_eq!(manifest.source_weight_dtype, "bfloat16");
    assert_eq!(manifest.execution_weight_dtype, "f32");
    assert_eq!(manifest.input_ids, [22007, 6463, 314]);
    assert_eq!(manifest.hidden_size, 576);
    assert_eq!(manifest.num_hidden_layers, 30);
    manifest
}

fn expected_stage(manifest: &TraceManifest, reference_dir: &Path, name: &str) -> Vec<f32> {
    let stage = manifest
        .stages
        .iter()
        .find(|stage| stage.name == name)
        .unwrap_or_else(|| panic!("missing trace stage {name}"));
    assert_eq!(stage.elements, 576);
    let bytes = fs::read(reference_dir.join(&stage.file)).expect("read trace vector");
    assert_eq!(bytes.len(), stage.elements * std::mem::size_of::<f32>());
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn report_stage(
    name: &str,
    actual: &DeviceBuffer<f32>,
    stream: &Stream,
    manifest: &TraceManifest,
    reference_dir: &Path,
) {
    let actual = actual.to_vec(stream).expect("copy NNIS trace vector");
    let expected = expected_stage(manifest, reference_dir, name);
    assert_eq!(actual.len(), expected.len());

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut worst_index = 0_usize;
    let mut squared_sum = 0.0_f64;
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(got.is_finite(), "non-finite NNIS value in {name} at {index}");
        let absolute = (got - want).abs();
        let relative = if want == 0.0 {
            if absolute == 0.0 {
                0.0
            } else {
                f32::INFINITY
            }
        } else {
            absolute / want.abs()
        };
        if absolute > max_abs {
            max_abs = absolute;
            worst_index = index;
        }
        max_rel = max_rel.max(relative);
        squared_sum += f64::from(absolute) * f64::from(absolute);
    }
    let rms = (squared_sum / actual.len() as f64).sqrt();
    println!(
        "{name}: max_abs={max_abs:.8e} max_rel={max_rel:.8e} rms={rms:.8e} worst_index={worst_index}"
    );
}

#[test]
fn smollm2_prefill_layerwise_diagnostic_on_gpu() {
    let Some(model_dir) = std::env::var_os("NNIS_SMOLLM2_MODEL") else {
        eprintln!("skipped: NNIS_SMOLLM2_MODEL is not set");
        return;
    };
    let Some(reference_dir) = std::env::var_os("NNIS_SMOLLM2_LAYERWISE_REFERENCE") else {
        eprintln!("skipped: NNIS_SMOLLM2_LAYERWISE_REFERENCE is not set");
        return;
    };
    let reference_dir = PathBuf::from(reference_dir);
    let manifest = load_manifest(&reference_dir);

    let Some(context) = gpu_context() else {
        eprintln!("skipped: no CUDA device");
        return;
    };
    let construction_stream = Stream::new(&context).unwrap();
    let model = Model::load_directory(&context, &construction_stream, model_dir).unwrap();
    assert_eq!(model.config.hidden_size, 576);
    assert_eq!(model.config.num_hidden_layers, 30);
    assert_eq!(model.config.num_attention_heads, 9);
    assert_eq!(model.config.num_key_value_heads, 3);

    let mut session = model.new_session().unwrap();
    let _ = session.prefill(&[22007, 6463]).unwrap();
    assert_eq!(session.position, 2);

    let token = DeviceBuffer::from_host(&model.context, &session.stream, &[314_u32]).unwrap();
    model
        .gather
        .gather(
            &session.stream,
            model.weights.token_embedding.tensor().as_f32().unwrap(),
            &token,
            &session.workspace.hidden,
            model.config.vocab_size,
            model.config.hidden_size,
        )
        .unwrap();
    report_stage(
        "embedding",
        &session.workspace.hidden,
        &session.stream,
        &manifest,
        &reference_dir,
    );

    let position = session.position;
    let attention_scale = 1.0_f32 / (model.config.head_dim() as f32).sqrt();
    let kv_width = model.config.key_value_width().unwrap();

    for layer_index in 0..model.config.num_hidden_layers {
        let layer = &model.weights.layers[layer_index];
        model
            .decoder
            .weighted_rms_norm(
                &session.stream,
                &session.workspace.hidden,
                layer.input_norm.tensor().as_f32().unwrap(),
                &session.workspace.normed,
                1,
                model.config.hidden_size,
                model.config.rms_norm_eps,
            )
            .unwrap();
        model
            .gemm
            .gemm(
                &session.stream,
                &session.workspace.normed,
                layer.q_proj.tensor().as_f32().unwrap(),
                &session.workspace.q,
                1,
                model.config.hidden_size,
                model.config.hidden_size,
            )
            .unwrap();
        model
            .gemm
            .gemm(
                &session.stream,
                &session.workspace.normed,
                layer.k_proj.tensor().as_f32().unwrap(),
                &session.workspace.k,
                1,
                kv_width,
                model.config.hidden_size,
            )
            .unwrap();
        model
            .gemm
            .gemm(
                &session.stream,
                &session.workspace.normed,
                layer.v_proj.tensor().as_f32().unwrap(),
                &session.workspace.v,
                1,
                kv_width,
                model.config.hidden_size,
            )
            .unwrap();

        unsafe {
            model
                .runtime
                .enqueue_rope_position(
                    &session.stream,
                    &session.workspace.q,
                    &model.rope_cos,
                    &model.rope_sin,
                    &session.workspace.q_rope,
                    model.config.num_attention_heads,
                    model.config.head_dim(),
                    position,
                    model.config.max_position_embeddings,
                )
                .unwrap();
            model
                .runtime
                .enqueue_rope_position(
                    &session.stream,
                    &session.workspace.k,
                    &model.rope_cos,
                    &model.rope_sin,
                    &session.workspace.k_rope,
                    model.config.num_key_value_heads,
                    model.config.head_dim(),
                    position,
                    model.config.max_position_embeddings,
                )
                .unwrap();
        }
        session.stream.synchronize().unwrap();

        let append = session
            .cache
            .append_layer_async(
                layer_index,
                Arc::clone(&session.workspace.k_rope),
                Arc::clone(&session.workspace.v),
                1,
            )
            .unwrap();
        model
            .decoder
            .cached_attention_decode(
                &session.stream,
                &session.workspace.q_rope,
                &session.cache,
                layer_index,
                &session.workspace.attention,
                attention_scale,
            )
            .unwrap();
        drop(append);

        model
            .gemm
            .gemm(
                &session.stream,
                &session.workspace.attention,
                layer.o_proj.tensor().as_f32().unwrap(),
                &session.workspace.projected,
                1,
                model.config.hidden_size,
                model.config.hidden_size,
            )
            .unwrap();
        model
            .elementwise
            .vector_add(
                &session.stream,
                &session.workspace.hidden,
                &session.workspace.projected,
                &session.workspace.residual,
            )
            .unwrap();
        model
            .decoder
            .weighted_rms_norm(
                &session.stream,
                &session.workspace.residual,
                layer.post_attention_norm.tensor().as_f32().unwrap(),
                &session.workspace.normed,
                1,
                model.config.hidden_size,
                model.config.rms_norm_eps,
            )
            .unwrap();
        model
            .gemm
            .gemm(
                &session.stream,
                &session.workspace.normed,
                layer.gate_proj.tensor().as_f32().unwrap(),
                &session.workspace.gate,
                1,
                model.config.intermediate_size,
                model.config.hidden_size,
            )
            .unwrap();
        model
            .gemm
            .gemm(
                &session.stream,
                &session.workspace.normed,
                layer.up_proj.tensor().as_f32().unwrap(),
                &session.workspace.up,
                1,
                model.config.intermediate_size,
                model.config.hidden_size,
            )
            .unwrap();
        model
            .elementwise
            .silu(
                &session.stream,
                &session.workspace.gate,
                &session.workspace.activated,
            )
            .unwrap();
        model
            .decoder
            .multiply(
                &session.stream,
                &session.workspace.activated,
                &session.workspace.up,
                &session.workspace.gated,
            )
            .unwrap();
        model
            .gemm
            .gemm(
                &session.stream,
                &session.workspace.gated,
                layer.down_proj.tensor().as_f32().unwrap(),
                &session.workspace.mlp,
                1,
                model.config.hidden_size,
                model.config.intermediate_size,
            )
            .unwrap();
        model
            .elementwise
            .vector_add(
                &session.stream,
                &session.workspace.residual,
                &session.workspace.mlp,
                &session.workspace.hidden,
            )
            .unwrap();

        report_stage(
            &format!("layer{layer_index:02}.hidden"),
            &session.workspace.hidden,
            &session.stream,
            &manifest,
            &reference_dir,
        );
    }

    model
        .decoder
        .weighted_rms_norm(
            &session.stream,
            &session.workspace.hidden,
            model.weights.final_norm.tensor().as_f32().unwrap(),
            &session.workspace.normed,
            1,
            model.config.hidden_size,
            model.config.rms_norm_eps,
        )
        .unwrap();
    report_stage(
        "final_norm",
        &session.workspace.normed,
        &session.stream,
        &manifest,
        &reference_dir,
    );
}
