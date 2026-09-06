use nnis::{Context, Device, GenerationConfig, Stream};
use nnis_model::{
    load_model_from_safetensors, preflight_hf_safetensors_source, HfSafetensorsPreflightReportV1,
    Model, SafetensorsLoadConfig,
};
use std::env;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tokenizers::Tokenizer;

const DEFAULT_DEVICE_ORDINAL: i32 = 0;
const DEFAULT_MAX_NEW_TOKENS: usize = 16;
const CLI_PREFLIGHT_SCHEMA: &str = "nnis.hf-preflight@1";
const USAGE: &str = "Usage:\n  nnis-hf validate --model DIR [--tokenizer FILE] [--json]\n  nnis-hf generate --model DIR --prompt TEXT [--tokenizer FILE] [--device N] [--max-new-tokens N]\n\n`validate` performs a CPU-only fail-closed preflight of config.json, Safetensors shards, recognized tensor names/shapes/dtypes, decoder graph completeness, and tokenizer IDs. `generate` loads the same local directory on CUDA. No network access is performed.";
const SOUP_F32_HINT: &str = "For a dense Soup artifact, produce an NNIS-executable source with: soup merge --adapter ADAPTER --output DIR --dtype float32. Soup's default float16 merge and 4-bit merged formats are not admitted by the current NNIS direct-HF execution path.";

#[derive(Debug, PartialEq, Eq)]
struct ValidateArgs {
    model_dir: PathBuf,
    tokenizer_file: PathBuf,
    json: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct GenerateArgs {
    model_dir: PathBuf,
    tokenizer_file: PathBuf,
    prompt: String,
    device_ordinal: i32,
    max_new_tokens: usize,
}

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Help,
    Validate(ValidateArgs),
    Generate(GenerateArgs),
}

fn required_value<I>(arguments: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    arguments
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_validate<I>(mut arguments: I) -> Result<Command, String>
where
    I: Iterator<Item = String>,
{
    let mut model_dir = None;
    let mut tokenizer_file = None;
    let mut json = false;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model" => {
                model_dir = Some(PathBuf::from(required_value(&mut arguments, "--model")?));
            }
            "--tokenizer" => {
                tokenizer_file = Some(PathBuf::from(required_value(
                    &mut arguments,
                    "--tokenizer",
                )?));
            }
            "--json" => json = true,
            "--help" | "-h" => return Ok(Command::Help),
            other => return Err(format!("unknown validate argument {other:?}\n\n{USAGE}")),
        }
    }

    let model_dir = model_dir.ok_or_else(|| "missing --model DIR".to_string())?;
    let tokenizer_file = tokenizer_file.unwrap_or_else(|| model_dir.join("tokenizer.json"));
    Ok(Command::Validate(ValidateArgs {
        model_dir,
        tokenizer_file,
        json,
    }))
}

fn parse_generate<I>(mut arguments: I) -> Result<Command, String>
where
    I: Iterator<Item = String>,
{
    let mut model_dir = None;
    let mut tokenizer_file = None;
    let mut prompt = None;
    let mut device_ordinal = DEFAULT_DEVICE_ORDINAL;
    let mut max_new_tokens = DEFAULT_MAX_NEW_TOKENS;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model" => {
                model_dir = Some(PathBuf::from(required_value(&mut arguments, "--model")?));
            }
            "--tokenizer" => {
                tokenizer_file = Some(PathBuf::from(required_value(
                    &mut arguments,
                    "--tokenizer",
                )?));
            }
            "--prompt" => {
                prompt = Some(required_value(&mut arguments, "--prompt")?);
            }
            "--device" => {
                let raw = required_value(&mut arguments, "--device")?;
                device_ordinal = raw
                    .parse::<i32>()
                    .map_err(|error| format!("invalid --device {raw:?}: {error}"))?;
                if device_ordinal < 0 {
                    return Err("--device must be a non-negative CUDA device ordinal".to_string());
                }
            }
            "--max-new-tokens" => {
                let raw = required_value(&mut arguments, "--max-new-tokens")?;
                max_new_tokens = raw
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --max-new-tokens {raw:?}: {error}"))?;
                if max_new_tokens == 0 {
                    return Err("--max-new-tokens must be greater than zero".to_string());
                }
            }
            "--help" | "-h" => return Ok(Command::Help),
            other => return Err(format!("unknown generate argument {other:?}\n\n{USAGE}")),
        }
    }

    let model_dir = model_dir.ok_or_else(|| "missing --model DIR".to_string())?;
    let tokenizer_file = tokenizer_file.unwrap_or_else(|| model_dir.join("tokenizer.json"));

    Ok(Command::Generate(GenerateArgs {
        model_dir,
        tokenizer_file,
        prompt: prompt.ok_or_else(|| "missing --prompt TEXT".to_string())?,
        device_ordinal,
        max_new_tokens,
    }))
}

fn parse_args<I>(arguments: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut arguments = arguments.into_iter();
    let Some(command) = arguments.next() else {
        return Ok(Command::Help);
    };
    match command.as_str() {
        "--help" | "-h" => Ok(Command::Help),
        "validate" => parse_validate(arguments),
        "generate" => parse_generate(arguments),
        _ => Err(format!("unknown command {command:?}\n\n{USAGE}")),
    }
}

fn load_tokenizer(tokenizer_file: &Path) -> Result<Tokenizer, String> {
    Tokenizer::from_file(tokenizer_file).map_err(|error| {
        format!(
            "failed to load tokenizer {}: {error}",
            tokenizer_file.display()
        )
    })
}

fn validate_tokenizer(
    tokenizer_file: &Path,
    model_vocab_size: usize,
) -> Result<(usize, u32), String> {
    let tokenizer = load_tokenizer(tokenizer_file)?;
    let vocab = tokenizer.get_vocab(true);
    if vocab.is_empty() {
        return Err("tokenizer vocabulary is empty".to_string());
    }
    let max_token_id = vocab
        .values()
        .copied()
        .max()
        .ok_or_else(|| "tokenizer vocabulary is empty".to_string())?;
    if max_token_id as usize >= model_vocab_size {
        return Err(format!(
            "tokenizer maximum token id {max_token_id} is out of range for model vocabulary {model_vocab_size}"
        ));
    }
    Ok((vocab.len(), max_token_id))
}

fn tokenize_prompt(tokenizer_file: &Path, prompt: &str) -> Result<(Tokenizer, Vec<u32>), String> {
    let tokenizer = load_tokenizer(tokenizer_file)?;
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|error| format!("failed to tokenize prompt: {error}"))?;
    let input_ids = encoding.get_ids().to_vec();
    if input_ids.is_empty() {
        return Err("tokenizer produced no input IDs".to_string());
    }
    Ok((tokenizer, input_ids))
}

fn safetensors_load_config(model_dir: &Path) -> SafetensorsLoadConfig {
    SafetensorsLoadConfig {
        repo_id: None,
        revision: None,
        local_dir: model_dir.to_string_lossy().into_owned(),
    }
}

fn json_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            control if control <= '\u{1f}' => {
                write!(&mut output, "\\u{:04x}", control as u32)
                    .expect("writing to String cannot fail");
            }
            other => output.push(other),
        }
    }
    output.push('"');
    output
}

fn render_preflight_json(
    arguments: &ValidateArgs,
    report: &HfSafetensorsPreflightReportV1,
    tokenizer_vocab_size: usize,
    tokenizer_max_token_id: u32,
) -> Result<String, String> {
    let model_source = report
        .to_json()
        .map_err(|error| format!("failed to serialize model preflight report: {error}"))?;
    let model_directory = json_string(arguments.model_dir.to_string_lossy().as_ref());
    let tokenizer_file = json_string(arguments.tokenizer_file.to_string_lossy().as_ref());
    let blocker = if report.direct_f32_execution_ready {
        "null".to_string()
    } else {
        json_string(SOUP_F32_HINT)
    };
    let mut output = String::new();
    write!(
        &mut output,
        "{{\"schema\":{},\"model_directory\":{},\"tokenizer_file\":{},\"model_source\":{},\"tokenizer\":{{\"vocabulary_size\":{},\"maximum_token_id\":{},\"model_vocabulary_size\":{}}},\"direct_execution_ready\":{},\"direct_execution_blocker\":{}}}",
        json_string(CLI_PREFLIGHT_SCHEMA),
        model_directory,
        tokenizer_file,
        model_source,
        tokenizer_vocab_size,
        tokenizer_max_token_id,
        report.metadata.vocab_size,
        report.direct_f32_execution_ready,
        blocker,
    )
    .expect("writing to String cannot fail");
    Ok(output)
}

fn render_preflight_text(
    arguments: &ValidateArgs,
    report: &HfSafetensorsPreflightReportV1,
    tokenizer_vocab_size: usize,
    tokenizer_max_token_id: u32,
) -> String {
    let state = if report.direct_f32_execution_ready {
        "READY"
    } else {
        "SOURCE_VALID_NOT_DIRECT_EXECUTION_READY"
    };
    let mut output = format!(
        "NNIS_HF_PREFLIGHT_{state}\nschema={}\nmodel={}\ntokenizer={}\narchitecture={}\nmodel_type={}\nweight_dtype={:?}\nweight_files={}\nrecognized_tensors={}\nignored_tensors={}\nlogical_tensors={}\ntied_lm_head_required={}\ntokenizer_vocab_size={}\ntokenizer_max_token_id={}\nmodel_vocab_size={}\ndirect_f32_execution_ready={}",
        report.schema_version,
        arguments.model_dir.display(),
        arguments.tokenizer_file.display(),
        report.metadata.architecture,
        report.metadata.model_type,
        report.metadata.weight_dtype,
        report.weight_files.join(","),
        report.recognized_tensor_count,
        report.ignored_tensor_count,
        report.logical_tensor_count,
        report.tied_lm_head_required,
        tokenizer_vocab_size,
        tokenizer_max_token_id,
        report.metadata.vocab_size,
        report.direct_f32_execution_ready,
    );
    if !report.direct_f32_execution_ready {
        output.push('\n');
        output.push_str(SOUP_F32_HINT);
    }
    output
}

fn validate(arguments: &ValidateArgs) -> Result<(String, bool), String> {
    let load_config = safetensors_load_config(&arguments.model_dir);
    let report = preflight_hf_safetensors_source(&load_config).map_err(|error| {
        format!(
            "failed CPU-only Hugging Face Safetensors preflight for {}: {error}",
            arguments.model_dir.display()
        )
    })?;
    let (tokenizer_vocab_size, tokenizer_max_token_id) =
        validate_tokenizer(&arguments.tokenizer_file, report.metadata.vocab_size)?;
    let output = if arguments.json {
        render_preflight_json(
            arguments,
            &report,
            tokenizer_vocab_size,
            tokenizer_max_token_id,
        )?
    } else {
        render_preflight_text(
            arguments,
            &report,
            tokenizer_vocab_size,
            tokenizer_max_token_id,
        )
    };
    Ok((output, report.direct_f32_execution_ready))
}

fn generate(arguments: &GenerateArgs) -> Result<String, String> {
    let (tokenizer, input_ids) =
        tokenize_prompt(&arguments.tokenizer_file, arguments.prompt.as_str())?;

    let device = Device::get(arguments.device_ordinal).map_err(|error| {
        format!(
            "failed to select CUDA device {}: {error}",
            arguments.device_ordinal
        )
    })?;
    let context =
        Context::new(&device).map_err(|error| format!("failed to create CUDA context: {error}"))?;
    let construction_stream =
        Stream::new(&context).map_err(|error| format!("failed to create CUDA stream: {error}"))?;

    let load_config = safetensors_load_config(&arguments.model_dir);
    let (config, weights) =
        load_model_from_safetensors(&context, &construction_stream, &load_config).map_err(
            |error| {
                format!(
                    "failed to load Hugging Face Safetensors model {}: {error}\n{SOUP_F32_HINT}",
                    arguments.model_dir.display()
                )
            },
        )?;
    let model = Model::new(config, weights, &construction_stream).map_err(|error| {
        format!(
            "failed to construct the qualified NNIS decoder from {}: {error}\n{SOUP_F32_HINT}",
            arguments.model_dir.display()
        )
    })?;

    let requested_positions = input_ids
        .len()
        .checked_add(arguments.max_new_tokens)
        .ok_or_else(|| "prompt plus generation length overflows usize".to_string())?;
    if requested_positions > model.config().max_position_embeddings {
        return Err(format!(
            "prompt has {} tokens and generation requests {} more, exceeding model capacity {}",
            input_ids.len(),
            arguments.max_new_tokens,
            model.config().max_position_embeddings
        ));
    }

    let generation = match model.config().eos_token_id {
        Some(eos_token_id) => {
            GenerationConfig::greedy_until_eos(arguments.max_new_tokens, eos_token_id)
        }
        None => GenerationConfig::greedy(arguments.max_new_tokens),
    };
    let generated = model
        .new_session()
        .and_then(|mut session| session.generate(&input_ids, generation))
        .map_err(|error| format!("NNIS generation failed: {error}"))?;

    tokenizer
        .decode(&generated, true)
        .map_err(|error| format!("failed to decode generated token IDs: {error}"))
}

fn main() -> ExitCode {
    let command = match parse_args(env::args().skip(1)) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };

    match command {
        Command::Help => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Command::Validate(arguments) => match validate(&arguments) {
            Ok((output, true)) => {
                println!("{output}");
                ExitCode::SUCCESS
            }
            Ok((output, false)) => {
                println!("{output}");
                ExitCode::FAILURE
            }
            Err(error) => {
                eprintln!("nnis-hf validate: {error}");
                ExitCode::FAILURE
            }
        },
        Command::Generate(arguments) => match generate(&arguments) {
            Ok(text) => {
                println!("{text}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("nnis-hf generate: {error}");
                ExitCode::FAILURE
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn help_is_available_without_cuda() {
        assert_eq!(parse_args(Vec::<String>::new()).unwrap(), Command::Help);
        assert_eq!(parse_args(strings(&["--help"])).unwrap(), Command::Help);
    }

    #[test]
    fn validate_defaults_to_model_tokenizer_and_text_output() {
        let parsed = parse_args(strings(&["validate", "--model", "/models/soup-merged"])).unwrap();
        assert_eq!(
            parsed,
            Command::Validate(ValidateArgs {
                model_dir: PathBuf::from("/models/soup-merged"),
                tokenizer_file: PathBuf::from("/models/soup-merged/tokenizer.json"),
                json: false,
            })
        );
    }

    #[test]
    fn validate_accepts_explicit_tokenizer_and_json_without_cuda() {
        let parsed = parse_args(strings(&[
            "validate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--json",
        ]))
        .unwrap();
        assert_eq!(
            parsed,
            Command::Validate(ValidateArgs {
                model_dir: PathBuf::from("/model"),
                tokenizer_file: PathBuf::from("/tokenizer.json"),
                json: true,
            })
        );
    }

    #[test]
    fn model_directory_supplies_default_tokenizer_for_generation() {
        let parsed = parse_args(strings(&[
            "generate",
            "--model",
            "/models/soup-merged",
            "--prompt",
            "Hello",
        ]))
        .unwrap();
        assert_eq!(
            parsed,
            Command::Generate(GenerateArgs {
                model_dir: PathBuf::from("/models/soup-merged"),
                tokenizer_file: PathBuf::from("/models/soup-merged/tokenizer.json"),
                prompt: "Hello".to_string(),
                device_ordinal: 0,
                max_new_tokens: 16,
            })
        );
    }

    #[test]
    fn explicit_tokenizer_and_runtime_limits_are_parsed_without_cuda() {
        let parsed = parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "Hello, NNIS!",
            "--device",
            "3",
            "--max-new-tokens",
            "7",
        ]))
        .unwrap();
        assert_eq!(
            parsed,
            Command::Generate(GenerateArgs {
                model_dir: PathBuf::from("/model"),
                tokenizer_file: PathBuf::from("/tokenizer.json"),
                prompt: "Hello, NNIS!".to_string(),
                device_ordinal: 3,
                max_new_tokens: 7,
            })
        );
    }

    #[test]
    fn invalid_arguments_fail_before_cuda() {
        assert!(parse_args(strings(&["validate"])).is_err());
        assert!(parse_args(strings(
            &["validate", "--model", "/model", "--prompt", "x",]
        ))
        .is_err());
        assert!(parse_args(strings(&["generate"])).is_err());
        assert!(parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--prompt",
            "x",
            "--max-new-tokens",
            "0",
        ]))
        .is_err());
        assert!(parse_args(strings(&[
            "generate", "--model", "/model", "--prompt", "x", "--device", "-1",
        ]))
        .is_err());
        assert!(parse_args(strings(&["unknown"])).is_err());
    }

    #[test]
    fn json_string_escapes_control_and_syntax_characters() {
        assert_eq!(json_string("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\"");
    }

    #[test]
    fn soup_hint_names_the_only_current_dense_direct_execution_dtype() {
        assert!(SOUP_F32_HINT.contains("--dtype float32"));
        assert!(SOUP_F32_HINT.contains("float16"));
        assert!(SOUP_F32_HINT.contains("4-bit"));
    }
}
