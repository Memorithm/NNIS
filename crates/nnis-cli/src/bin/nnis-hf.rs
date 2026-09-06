use nnis::{Context, Device, GenerationConfig, Stream};
use nnis_model::{load_model_from_safetensors, Model, SafetensorsLoadConfig};
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tokenizers::Tokenizer;

const DEFAULT_DEVICE_ORDINAL: i32 = 0;
const DEFAULT_MAX_NEW_TOKENS: usize = 16;
const USAGE: &str = "Usage:\n  nnis-hf generate --model DIR --prompt TEXT [--tokenizer FILE] [--device N] [--max-new-tokens N]\n\nLoads an already-materialized local Hugging Face Safetensors directory with NNIS's strict loader. The tokenizer defaults to DIR/tokenizer.json. No network access is performed.";
const SOUP_F32_HINT: &str = "For a dense Soup artifact, produce an NNIS-executable source with: soup merge --adapter ADAPTER --output DIR --dtype float32. Soup's default float16 merge and 4-bit merged formats are not admitted by the current NNIS direct-HF execution path.";

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
    Generate(GenerateArgs),
}

fn parse_args<I>(arguments: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut arguments = arguments.into_iter();
    let Some(command) = arguments.next() else {
        return Ok(Command::Help);
    };
    if matches!(command.as_str(), "--help" | "-h") {
        return Ok(Command::Help);
    }
    if command != "generate" {
        return Err(format!("unknown command {command:?}\n\n{USAGE}"));
    }

    let mut model_dir = None;
    let mut tokenizer_file = None;
    let mut prompt = None;
    let mut device_ordinal = DEFAULT_DEVICE_ORDINAL;
    let mut max_new_tokens = DEFAULT_MAX_NEW_TOKENS;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model" => {
                model_dir = Some(PathBuf::from(
                    arguments
                        .next()
                        .ok_or_else(|| "--model requires a directory".to_string())?,
                ));
            }
            "--tokenizer" => {
                tokenizer_file = Some(PathBuf::from(
                    arguments
                        .next()
                        .ok_or_else(|| "--tokenizer requires a file".to_string())?,
                ));
            }
            "--prompt" => {
                prompt = Some(
                    arguments
                        .next()
                        .ok_or_else(|| "--prompt requires text".to_string())?,
                );
            }
            "--device" => {
                let raw = arguments
                    .next()
                    .ok_or_else(|| "--device requires an integer ordinal".to_string())?;
                device_ordinal = raw
                    .parse::<i32>()
                    .map_err(|error| format!("invalid --device {raw:?}: {error}"))?;
                if device_ordinal < 0 {
                    return Err("--device must be a non-negative CUDA device ordinal".to_string());
                }
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

fn tokenize_prompt(tokenizer_file: &Path, prompt: &str) -> Result<(Tokenizer, Vec<u32>), String> {
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(|error| {
        format!(
            "failed to load tokenizer {}: {error}",
            tokenizer_file.display()
        )
    })?;
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|error| format!("failed to tokenize prompt: {error}"))?;
    let input_ids = encoding.get_ids().to_vec();
    if input_ids.is_empty() {
        return Err("tokenizer produced no input IDs".to_string());
    }
    Ok((tokenizer, input_ids))
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

    let load_config = SafetensorsLoadConfig {
        repo_id: None,
        revision: None,
        local_dir: arguments.model_dir.to_string_lossy().into_owned(),
    };
    let (config, weights) = load_model_from_safetensors(&context, &construction_stream, &load_config)
        .map_err(|error| {
            format!(
                "failed to load Hugging Face Safetensors model {}: {error}\n{SOUP_F32_HINT}",
                arguments.model_dir.display()
            )
        })?;
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
    fn model_directory_supplies_default_tokenizer() {
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
            "generate",
            "--model",
            "/model",
            "--prompt",
            "x",
            "--device",
            "-1",
        ]))
        .is_err());
        assert!(parse_args(strings(&["unknown"])).is_err());
    }

    #[test]
    fn soup_hint_names_the_only_current_dense_direct_execution_dtype() {
        assert!(SOUP_F32_HINT.contains("--dtype float32"));
        assert!(SOUP_F32_HINT.contains("float16"));
        assert!(SOUP_F32_HINT.contains("4-bit"));
    }
}
