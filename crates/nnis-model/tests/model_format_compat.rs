use nnis_model::{ModelManifest, NNIS_MODEL_FORMAT, NNIS_MODEL_VERSION};

#[test]
fn version_one_manifest_without_eos_metadata_remains_readable() {
    let json = r#"
    {
      "format": "nnis-model",
      "version": 1,
      "config": {
        "vocab_size": 8,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 16,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "activation": "silu",
        "weight_dtype": "f32"
      },
      "tensors": []
    }
    "#;

    let manifest: ModelManifest = serde_json::from_str(json).unwrap();
    assert_eq!(manifest.format, NNIS_MODEL_FORMAT);
    assert_eq!(manifest.version, NNIS_MODEL_VERSION);
    assert_eq!(manifest.config.eos_token_id, None);
    manifest.config.validate().unwrap();
}

#[test]
fn version_one_manifest_accepts_explicit_eos_metadata() {
    let json = r#"
    {
      "format": "nnis-model",
      "version": 1,
      "config": {
        "vocab_size": 8,
        "eos_token_id": 2,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "max_position_embeddings": 16,
        "rms_norm_eps": 0.00001,
        "rope_theta": 10000.0,
        "activation": "silu",
        "weight_dtype": "f32"
      },
      "tensors": []
    }
    "#;

    let manifest: ModelManifest = serde_json::from_str(json).unwrap();
    assert_eq!(manifest.config.eos_token_id, Some(2));
    manifest.config.validate().unwrap();
}
