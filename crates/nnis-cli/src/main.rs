use nnis::{Context, Device, GenerationConfig, Model, Stream};
use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tokenizers::Tokenizer;

const DEFAULT_MAX_NEW_TOKENS: usize = 16;
const USAGE: &str = "Usage:\n  nnis generate --model DIR --tokenizer FILE --prompt TEXT [--max-new-tokens N]\n\nThe first generation CLI is intentionally narrow: greedy decoding on the first visible CUDA device using an explicit NNIS model directory and Hugging Face tokenizer.json file.";

#[derive(Debug, PartialEq, Eq)]
struct GenerateArgs {
    model_dir: PathBuf,
    tokenizer_file: PathBuf,
    prompt: String,
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

    Ok(Command::Generate(GenerateArgs {
        model_dir: model_dir.ok_or_else(|| "missing --model DIR".to_string())?,
        tokenizer_file: tokenizer_file.ok_or_else(|| "missing --tokenizer FILE".to_string())?,
        prompt: prompt.ok_or_else(|| "missing --prompt TEXT".to_string())?,
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

    let device =
        Device::first().map_err(|error| format!("failed to select CUDA device: {error}"))?;
    let context =
        Context::new(&device).map_err(|error| format!("failed to create CUDA context: {error}"))?;
    let construction_stream =
        Stream::new(&context).map_err(|error| format!("failed to create CUDA stream: {error}"))?;
    let model = Model::load_directory(&context, &construction_stream, &arguments.model_dir)
        .map_err(|error| {
            format!(
                "failed to load NNIS model {}: {error}",
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

    let generated = model
        .new_session()
        .and_then(|mut session| {
            session.generate(
                &input_ids,
                GenerationConfig::greedy(arguments.max_new_tokens),
            )
        })
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
                eprintln!("nnis generate: {error}");
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
    fn generate_arguments_are_parsed_without_touching_cuda() {
        let parsed = parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "Hello, NNIS!",
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
                max_new_tokens: 7,
            })
        );
    }

    #[test]
    fn generate_rejects_missing_and_invalid_arguments() {
        assert!(parse_args(strings(&["generate"])).is_err());
        assert!(parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "x",
            "--max-new-tokens",
            "0",
        ]))
        .is_err());
        assert!(parse_args(strings(&["unknown"])).is_err());
    }

    #[test]
    fn pinned_tiny_llama_tokenizer_matches_transformers_when_available() {
        let Some(path) = std::env::var_os("NNIS_TINY_LLAMA_TOKENIZER") else {
            eprintln!("skipped: NNIS_TINY_LLAMA_TOKENIZER is not set");
            return;
        };
        let (_, input_ids) =
            tokenize_prompt(&PathBuf::from(path), "Hello, NNIS!").expect("tokenize pinned prompt");
        assert_eq!(input_ids, vec![1, 15043, 29892, 405, 29940, 3235, 29991]);
    }
}
