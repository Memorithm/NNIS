use super::*;
use nnis_rt::gpu_context;
use std::fs;
use std::path::PathBuf;

fn read_f32_le(path: &PathBuf) -> Vec<f32> {
    let bytes = fs::read(path).expect("read SmolLM2 final hidden reference");
    assert_eq!(bytes.len() % 4, 0, "reference byte length must be f32-aligned");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn smollm2_prefill_final_norm_diagnostic_on_gpu() {
    let Some(model_dir) = std::env::var_os("NNIS_SMOLLM2_MODEL") else {
        eprintln!("skipped: NNIS_SMOLLM2_MODEL is not set");
        return;
    };
    let Some(reference_path) = std::env::var_os("NNIS_SMOLLM2_FINAL_HIDDEN") else {
        eprintln!("skipped: NNIS_SMOLLM2_FINAL_HIDDEN is not set");
        return;
    };
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
    let logits = session.prefill(&[22007, 6463, 314]).unwrap();
    assert_eq!(logits.len(), 49_152);

    let actual = session
        .workspace
        .normed
        .to_vec(&session.stream)
        .expect("copy NNIS final norm to host");
    let expected = read_f32_le(&PathBuf::from(reference_path));
    assert_eq!(actual.len(), 576);
    assert_eq!(expected.len(), 576);

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut worst_index = 0_usize;
    let mut squared_sum = 0.0_f64;
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(got.is_finite(), "non-finite NNIS final hidden at index {index}");
        let absolute = (got - want).abs();
        let relative = if want == 0.0 {
            if absolute == 0.0 { 0.0 } else { f32::INFINITY }
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
        "smollm2_prefill_final_norm: max_abs={max_abs:.8e} max_rel={max_rel:.8e} rms={rms:.8e} worst_index={worst_index}"
    );
}
