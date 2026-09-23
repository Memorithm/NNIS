use nnis_model::{
    load_model_from_safetensors_f32, validate_finite_runtime_output, GeneratedTokenEvidenceV1,
    GenerationConfig, Int2DenseMaterializedModelV1, Int4DenseMaterializedModelV1,
    PhysicalWeightExecutionObservationV1, SafetensorsLoadConfig, SparseDenseMaterializedModelV1,
    WeightCampaignRecipeV1, WeightFullModelCampaignArtifactV1, WeightFullModelCampaignV1,
    WeightFullModelExecutionEvidenceV1, SMOLLM2_135M_BF16,
};
use nnis_rt::{Context, Device, NnisError, Result, Stream};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, PartialEq)]
struct Args {
    model_dir: PathBuf,
    tokenizer: PathBuf,
    prompt_ids: Vec<u32>,
    max_new_tokens: usize,
    sparse_threshold: f32,
    output: PathBuf,
}

fn parse_prompt_ids(raw: &str) -> std::result::Result<Vec<u32>, String> {
    let mut ids = Vec::new();
    for part in raw.split(',') {
        if part.is_empty() || part.trim() != part {
            return Err("prompt token ids must be comma-separated unsigned integers".to_string());
        }
        ids.push(
            part.parse::<u32>()
                .map_err(|error| format!("invalid prompt token id {part:?}: {error}"))?,
        );
    }
    if ids.is_empty() {
        return Err("at least one prompt token id is required".to_string());
    }
    Ok(ids)
}

fn parse_args<I>(arguments: I) -> std::result::Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut arguments = arguments.into_iter();
    let mut model_dir = None;
    let mut tokenizer = None;
    let mut prompt_ids = None;
    let mut max_new_tokens = 8_usize;
    let mut sparse_threshold = 0.05_f32;
    let mut output = None;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model-dir" => {
                model_dir =
                    Some(PathBuf::from(arguments.next().ok_or_else(|| {
                        "--model-dir requires a directory".to_string()
                    })?));
            }
            "--tokenizer" => {
                tokenizer =
                    Some(PathBuf::from(arguments.next().ok_or_else(|| {
                        "--tokenizer requires a file".to_string()
                    })?));
            }
            "--prompt-ids" => {
                let raw = arguments
                    .next()
                    .ok_or_else(|| "--prompt-ids requires comma-separated token ids".to_string())?;
                prompt_ids = Some(parse_prompt_ids(&raw)?);
            }
            "--max-new-tokens" => {
                let raw = arguments
                    .next()
                    .ok_or_else(|| "--max-new-tokens requires an integer".to_string())?;
                max_new_tokens = raw
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --max-new-tokens {raw:?}: {error}"))?;
                if max_new_tokens == 0 {
                    return Err("--max-new-tokens must be greater than zero".to_string());
                }
            }
            "--sparse-threshold" => {
                let raw = arguments
                    .next()
                    .ok_or_else(|| "--sparse-threshold requires a float".to_string())?;
                sparse_threshold = raw
                    .parse::<f32>()
                    .map_err(|error| format!("invalid --sparse-threshold {raw:?}: {error}"))?;
                if !sparse_threshold.is_finite() || sparse_threshold < 0.0 {
                    return Err("--sparse-threshold must be finite and non-negative".to_string());
                }
            }
            "--output" => {
                output =
                    Some(PathBuf::from(arguments.next().ok_or_else(|| {
                        "--output requires a JSON file".to_string()
                    })?));
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }

    Ok(Args {
        model_dir: model_dir.ok_or_else(|| "missing --model-dir DIR".to_string())?,
        tokenizer: tokenizer.ok_or_else(|| "missing --tokenizer FILE".to_string())?,
        prompt_ids: prompt_ids.ok_or_else(|| "missing --prompt-ids IDS".to_string())?,
        max_new_tokens,
        sparse_threshold,
        output: output.ok_or_else(|| "missing --output FILE".to_string())?,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).map_err(|error| NnisError::io("open tokenizer artifact", error))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| NnisError::io("hash tokenizer artifact", error))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn tokenizer_basename(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .ok_or_else(|| NnisError::invalid_input("tokenizer path has no UTF-8 basename"))
}

fn current_nnis_commit() -> Result<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|error| NnisError::io("run git rev-parse HEAD", error))?;
    if !output.status.success() {
        return Err(NnisError::invalid_input(format!(
            "git rev-parse HEAD failed with status {}",
            output.status
        )));
    }
    let commit = String::from_utf8(output.stdout)
        .map_err(|error| NnisError::invalid_input(format!("git HEAD is not UTF-8: {error}")))?
        .trim()
        .to_string();
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NnisError::invalid_input(format!(
            "git HEAD is not a 40-character lowercase commit: {commit:?}"
        )));
    }
    Ok(commit)
}

fn load_source(
    context: &std::sync::Arc<Context>,
    stream: &Stream,
    model_dir: &Path,
) -> Result<nnis_model::LoadedSafetensorsModel> {
    let config = SafetensorsLoadConfig {
        repo_id: Some(SMOLLM2_135M_BF16.source_repo.to_string()),
        revision: Some(SMOLLM2_135M_BF16.source_revision.to_string()),
        local_dir: model_dir.to_string_lossy().into_owned(),
    };
    let loaded = load_model_from_safetensors_f32(context, stream, &config)?;
    SMOLLM2_135M_BF16.validate_config(&loaded.source_config)?;
    if loaded.execution_config.weight_dtype != nnis_model::WeightDType::F32 {
        return Err(NnisError::invalid_input(
            "SmolLM2 qualification runner requires an actual F32 execution graph",
        ));
    }
    Ok(loaded)
}

fn observe_generation(
    model: &nnis_model::Model,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    nnis_commit: &str,
    runtime_entrypoint: &str,
) -> Result<PhysicalWeightExecutionObservationV1> {
    let mut session = model.new_session()?;
    let logits = session.prefill(prompt_ids)?;
    validate_finite_runtime_output("prefill_logits", &logits)?;
    let generated = session.generate(prompt_ids, GenerationConfig::fixed(max_new_tokens))?;
    let generated_tokens = GeneratedTokenEvidenceV1::from_token_ids(&generated)?;
    PhysicalWeightExecutionObservationV1::new(
        nnis_commit,
        runtime_entrypoint,
        generated_tokens,
        false,
    )
}

fn run_int4(
    context: &std::sync::Arc<Context>,
    stream: &Stream,
    model_dir: &Path,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    commit: &str,
) -> Result<WeightFullModelExecutionEvidenceV1> {
    let loaded = load_source(context, stream, model_dir)?;
    let model = Int4DenseMaterializedModelV1::from_f32_model_weights(
        loaded.execution_config,
        loaded.weights,
        stream,
    )?;
    let observation = observe_generation(
        model.model(),
        prompt_ids,
        max_new_tokens,
        commit,
        "Int4DenseMaterializedModelV1::model",
    )?;
    model.full_model_evidence(&SMOLLM2_135M_BF16, &observation)
}

fn run_int2(
    context: &std::sync::Arc<Context>,
    stream: &Stream,
    model_dir: &Path,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    commit: &str,
) -> Result<WeightFullModelExecutionEvidenceV1> {
    let loaded = load_source(context, stream, model_dir)?;
    let model = Int2DenseMaterializedModelV1::from_f32_model_weights(
        loaded.execution_config,
        loaded.weights,
        stream,
    )?;
    let observation = observe_generation(
        model.model(),
        prompt_ids,
        max_new_tokens,
        commit,
        "Int2DenseMaterializedModelV1::model",
    )?;
    model.full_model_evidence(&SMOLLM2_135M_BF16, &observation)
}

#[allow(clippy::too_many_arguments)]
fn run_sparse(
    context: &std::sync::Arc<Context>,
    stream: &Stream,
    model_dir: &Path,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    sparse_threshold: f32,
    commit: &str,
) -> Result<WeightFullModelExecutionEvidenceV1> {
    let loaded = load_source(context, stream, model_dir)?;
    let model = SparseDenseMaterializedModelV1::from_f32_model_weights(
        loaded.execution_config,
        loaded.weights,
        stream,
        sparse_threshold,
    )?;
    let observation = observe_generation(
        model.model(),
        prompt_ids,
        max_new_tokens,
        commit,
        "SparseDenseMaterializedModelV1::model",
    )?;
    model.full_model_evidence(&SMOLLM2_135M_BF16, &observation)
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args = parse_args(std::env::args().skip(1)).map_err(NnisError::invalid_input)?;
    let model_payload = args.model_dir.join("model.safetensors");
    SMOLLM2_135M_BF16.verify_model_file_sha256(&model_payload)?;
    let commit = current_nnis_commit()?;

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;

    let int4 = run_int4(
        &context,
        &stream,
        &args.model_dir,
        &args.prompt_ids,
        args.max_new_tokens,
        &commit,
    )?;
    let int2 = run_int2(
        &context,
        &stream,
        &args.model_dir,
        &args.prompt_ids,
        args.max_new_tokens,
        &commit,
    )?;
    let sparse = run_sparse(
        &context,
        &stream,
        &args.model_dir,
        &args.prompt_ids,
        args.max_new_tokens,
        args.sparse_threshold,
        &commit,
    )?;

    let campaign = WeightFullModelCampaignV1::new(vec![int4, int2, sparse])?;
    campaign.qualification_bundle()?;
    let recipe = WeightCampaignRecipeV1::new(
        args.prompt_ids.clone(),
        args.max_new_tokens,
        args.sparse_threshold,
    )?;
    let artifact = WeightFullModelCampaignArtifactV1::new(
        campaign,
        recipe,
        tokenizer_basename(&args.tokenizer)?,
        sha256_file(&args.tokenizer)?,
    )?;
    artifact.validate()?;
    let json = serde_json::to_string_pretty(&artifact)?;
    fs::write(&args.output, format!("{json}\n"))?;
    println!("{}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn prompt_id_parser_is_strict() {
        assert_eq!(parse_prompt_ids("1,2,3").unwrap(), vec![1, 2, 3]);
        assert!(parse_prompt_ids("").is_err());
        assert!(parse_prompt_ids("1, 2").is_err());
        assert!(parse_prompt_ids("1,nope").is_err());
    }

    #[test]
    fn runner_args_freeze_generation_and_sparse_controls() {
        let args = parse_args(strings(&[
            "--model-dir",
            "/tmp/model",
            "--tokenizer",
            "/tmp/tokenizer.json",
            "--prompt-ids",
            "1,2",
            "--max-new-tokens",
            "4",
            "--sparse-threshold",
            "0.125",
            "--output",
            "/tmp/campaign.json",
        ]))
        .unwrap();
        assert_eq!(args.model_dir, PathBuf::from("/tmp/model"));
        assert_eq!(args.tokenizer, PathBuf::from("/tmp/tokenizer.json"));
        assert_eq!(args.prompt_ids, vec![1, 2]);
        assert_eq!(args.max_new_tokens, 4);
        assert_eq!(args.sparse_threshold.to_bits(), 0.125_f32.to_bits());
        assert_eq!(args.output, PathBuf::from("/tmp/campaign.json"));

        assert!(parse_args(strings(&[
            "--model-dir",
            "/tmp/model",
            "--tokenizer",
            "/tmp/tokenizer.json",
            "--prompt-ids",
            "1",
            "--max-new-tokens",
            "0",
            "--output",
            "/tmp/out.json",
        ]))
        .is_err());
    }
}
