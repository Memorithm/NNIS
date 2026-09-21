use nnis::{
    current_process_gpu_memory, Context, Device, GenerationConfig, GenerationStreamControl, Model,
    NvmlProcessMemorySnapshotV1, SamplingConfig, Stream, NNIS_NVML_PROCESS_MEMORY_SNAPSHOT_VERSION,
};
use serde::Serialize;
use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tokenizers::Tokenizer;

const DEFAULT_DEVICE_ORDINAL: i32 = 0;
const DEFAULT_MAX_NEW_TOKENS: usize = 16;
const USAGE: &str = "Usage:\n  nnis generate --model DIR --tokenizer FILE --prompt TEXT [--device N] [--max-new-tokens N] [--sample --seed U64] [--temperature F] [--top-k N] [--top-p F] [--stream]\n  nnis nvml-process-memory [--device N] [--json]\n\nDefault decoding is greedy (unchanged). Opt-in `--sample` requires `--seed` and uses host-visible NNML1 SamplingConfig. Optional `--temperature`, `--top-k`, and `--top-p` apply only with `--sample`. `--stream` is valid only with `--sample` and prints each decoded token piece as it is emitted. CUDA device ordinal defaults to 0.\n\n`nvml-process-memory` is a fail-closed, read-only NVML process-scoped usedGpuMemory debug surface for this PID on the selected CUDA device (default 0). Human text by default; `--json` emits versioned JSON with schema_version. It does not claim physical residency, weight-only attribution, or performance.";

#[derive(Debug, PartialEq)]
struct GenerateArgs {
    model_dir: PathBuf,
    tokenizer_file: PathBuf,
    prompt: String,
    device_ordinal: i32,
    max_new_tokens: usize,
    sample: bool,
    stream: bool,
    seed: Option<u64>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
}

#[derive(Debug, PartialEq)]
struct NvmlProcessMemoryArgs {
    device_ordinal: i32,
    json: bool,
}

#[derive(Debug, PartialEq)]
enum Command {
    Help,
    Generate(GenerateArgs),
    NvmlProcessMemory(NvmlProcessMemoryArgs),
}

#[derive(Debug, Serialize)]
struct NvmlProcessMemoryJsonV1<'a> {
    schema_version: u32,
    pid: u32,
    device_ordinal: i32,
    device_uuid: &'a str,
    used_gpu_memory_bytes: u64,
}

fn parse_positive_f32(flag: &str, raw: &str) -> Result<f32, String> {
    let value = raw
        .parse::<f32>()
        .map_err(|error| format!("invalid {flag} {raw:?}: {error}"))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{flag} must be finite and positive; got {raw}"));
    }
    Ok(value)
}

fn build_sampling_config(arguments: &GenerateArgs) -> Result<SamplingConfig, String> {
    let seed = arguments
        .seed
        .ok_or_else(|| "--sample requires --seed U64".to_string())?;
    let mut sampling = SamplingConfig::seeded(seed);
    if let Some(temperature) = arguments.temperature {
        sampling = sampling.with_temperature(temperature);
    }
    if let Some(top_k) = arguments.top_k {
        sampling = sampling.with_top_k(top_k);
    }
    if let Some(top_p) = arguments.top_p {
        sampling = sampling.with_top_p(top_p);
    }
    Ok(sampling)
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
    if command == "nvml-process-memory" {
        return parse_nvml_process_memory_args(arguments);
    }
    if command != "generate" {
        return Err(format!("unknown command {command:?}\n\n{USAGE}"));
    }

    let mut model_dir = None;
    let mut tokenizer_file = None;
    let mut prompt = None;
    let mut device_ordinal = DEFAULT_DEVICE_ORDINAL;
    let mut max_new_tokens = DEFAULT_MAX_NEW_TOKENS;
    let mut sample = false;
    let mut stream = false;
    let mut seed = None;
    let mut temperature = None;
    let mut top_k = None;
    let mut top_p = None;

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
            "--sample" => sample = true,
            "--stream" => stream = true,
            "--seed" => {
                let raw = arguments
                    .next()
                    .ok_or_else(|| "--seed requires an unsigned integer".to_string())?;
                seed = Some(
                    raw.parse::<u64>()
                        .map_err(|error| format!("invalid --seed {raw:?}: {error}"))?,
                );
            }
            "--temperature" => {
                let raw = arguments
                    .next()
                    .ok_or_else(|| "--temperature requires a float".to_string())?;
                temperature = Some(parse_positive_f32("--temperature", &raw)?);
            }
            "--top-k" => {
                let raw = arguments
                    .next()
                    .ok_or_else(|| "--top-k requires an integer".to_string())?;
                let value = raw
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --top-k {raw:?}: {error}"))?;
                if value == 0 {
                    return Err("--top-k must be greater than zero".to_string());
                }
                top_k = Some(value);
            }
            "--top-p" => {
                let raw = arguments
                    .next()
                    .ok_or_else(|| "--top-p requires a float".to_string())?;
                let value = parse_positive_f32("--top-p", &raw)?;
                if value > 1.0 {
                    return Err(format!("--top-p must be in (0, 1]; got {raw}"));
                }
                top_p = Some(value);
            }
            "--help" | "-h" => return Ok(Command::Help),
            other => return Err(format!("unknown generate argument {other:?}\n\n{USAGE}")),
        }
    }

    if stream && !sample {
        return Err("--stream requires --sample".to_string());
    }
    if sample && seed.is_none() {
        return Err("--sample requires --seed U64".to_string());
    }
    if !sample {
        if seed.is_some() {
            return Err("--seed requires --sample".to_string());
        }
        if temperature.is_some() {
            return Err("--temperature requires --sample".to_string());
        }
        if top_k.is_some() {
            return Err("--top-k requires --sample".to_string());
        }
        if top_p.is_some() {
            return Err("--top-p requires --sample".to_string());
        }
    }

    Ok(Command::Generate(GenerateArgs {
        model_dir: model_dir.ok_or_else(|| "missing --model DIR".to_string())?,
        tokenizer_file: tokenizer_file.ok_or_else(|| "missing --tokenizer FILE".to_string())?,
        prompt: prompt.ok_or_else(|| "missing --prompt TEXT".to_string())?,
        device_ordinal,
        max_new_tokens,
        sample,
        stream,
        seed,
        temperature,
        top_k,
        top_p,
    }))
}

fn parse_nvml_process_memory_args<I>(arguments: I) -> Result<Command, String>
where
    I: IntoIterator<Item = String>,
{
    let mut device_ordinal = DEFAULT_DEVICE_ORDINAL;
    let mut json = false;
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
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
            "--json" => json = true,
            "--help" | "-h" => return Ok(Command::Help),
            other => {
                return Err(format!(
                    "unknown nvml-process-memory argument {other:?}\n\n{USAGE}"
                ))
            }
        }
    }
    Ok(Command::NvmlProcessMemory(NvmlProcessMemoryArgs {
        device_ordinal,
        json,
    }))
}

fn format_nvml_process_memory_text(snapshot: &NvmlProcessMemorySnapshotV1) -> String {
    format!(
        "NNIS NVML process-memory snapshot v{version}\nschema_version: {schema}\npid: {pid}\ndevice_ordinal: {ordinal}\ndevice_uuid: {uuid}\nused_gpu_memory_bytes: {bytes}\n\nClaim boundary: process-scoped NVML usedGpuMemory for this PID/device only. Not physical residency, weight-only attribution, or performance.",
        version = NNIS_NVML_PROCESS_MEMORY_SNAPSHOT_VERSION,
        schema = snapshot.schema_version,
        pid = snapshot.pid,
        ordinal = snapshot.device_ordinal,
        uuid = snapshot.device_uuid,
        bytes = snapshot.used_gpu_memory_bytes,
    )
}

fn format_nvml_process_memory_json(
    snapshot: &NvmlProcessMemorySnapshotV1,
) -> Result<String, String> {
    let payload = NvmlProcessMemoryJsonV1 {
        schema_version: snapshot.schema_version,
        pid: snapshot.pid,
        device_ordinal: snapshot.device_ordinal,
        device_uuid: snapshot.device_uuid.as_str(),
        used_gpu_memory_bytes: snapshot.used_gpu_memory_bytes,
    };
    serde_json::to_string_pretty(&payload)
        .map_err(|error| format!("failed to serialize NVML snapshot JSON: {error}"))
}

/// Select CUDA device, retain a primary context so this PID is visible to NVML as
/// a compute process, then query the public process-memory snapshot API.
fn probe_nvml_process_memory(
    arguments: &NvmlProcessMemoryArgs,
) -> Result<NvmlProcessMemorySnapshotV1, String> {
    let device = Device::get(arguments.device_ordinal).map_err(|error| {
        format!(
            "failed to select CUDA device {}: {error}",
            arguments.device_ordinal
        )
    })?;
    let _context =
        Context::new(&device).map_err(|error| format!("failed to create CUDA context: {error}"))?;
    current_process_gpu_memory(&device)
        .map_err(|error| format!("NVML process-memory probe failed: {error}"))
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

fn decode_token_piece(tokenizer: &Tokenizer, token: u32) -> Result<String, String> {
    tokenizer
        .decode(&[token], true)
        .map_err(|error| format!("failed to decode generated token ID {token}: {error}"))
}

/// Returns `Some(text)` for buffered output paths; `None` when streaming already printed.
fn generate(arguments: &GenerateArgs) -> Result<Option<String>, String> {
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

    let generation = match model.config().eos_token_id {
        Some(eos_token_id) => {
            GenerationConfig::greedy_until_eos(arguments.max_new_tokens, eos_token_id)
        }
        None => GenerationConfig::greedy(arguments.max_new_tokens),
    };

    if !arguments.sample {
        let generated = model
            .new_session()
            .and_then(|mut session| session.generate(&input_ids, generation))
            .map_err(|error| format!("NNIS generation failed: {error}"))?;
        let text = tokenizer
            .decode(&generated, true)
            .map_err(|error| format!("failed to decode generated token IDs: {error}"))?;
        return Ok(Some(text));
    }

    let sampling = build_sampling_config(arguments)?;
    if arguments.stream {
        let mut stdout = io::stdout();
        let mut decode_error = None;
        model
            .new_session()
            .and_then(|mut session| {
                session.generate_sampled_streaming(&input_ids, generation, sampling, |token| {
                    match decode_token_piece(&tokenizer, token) {
                        Ok(piece) => {
                            let _ = write!(stdout, "{piece}");
                            let _ = stdout.flush();
                            GenerationStreamControl::Continue
                        }
                        Err(error) => {
                            decode_error = Some(error);
                            GenerationStreamControl::Stop
                        }
                    }
                })
            })
            .map_err(|error| format!("NNIS sampled streaming generation failed: {error}"))?;
        if let Some(error) = decode_error {
            return Err(error);
        }
        println!();
        return Ok(None);
    }

    let generated = model
        .new_session()
        .and_then(|mut session| session.generate_sampled(&input_ids, generation, sampling))
        .map_err(|error| format!("NNIS sampled generation failed: {error}"))?;
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|error| format!("failed to decode generated token IDs: {error}"))?;
    Ok(Some(text))
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
            Ok(Some(text)) => {
                println!("{text}");
                ExitCode::SUCCESS
            }
            Ok(None) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("nnis generate: {error}");
                ExitCode::FAILURE
            }
        },
        Command::NvmlProcessMemory(arguments) => match probe_nvml_process_memory(&arguments) {
            Ok(snapshot) => {
                let rendered = if arguments.json {
                    match format_nvml_process_memory_json(&snapshot) {
                        Ok(text) => text,
                        Err(error) => {
                            eprintln!("nnis nvml-process-memory: {error}");
                            return ExitCode::FAILURE;
                        }
                    }
                } else {
                    format_nvml_process_memory_text(&snapshot)
                };
                println!("{rendered}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("nnis nvml-process-memory: {error}");
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

    fn base_generate_args() -> GenerateArgs {
        GenerateArgs {
            model_dir: PathBuf::from("/model"),
            tokenizer_file: PathBuf::from("/tokenizer.json"),
            prompt: "Hello, NNIS!".to_string(),
            device_ordinal: 3,
            max_new_tokens: 7,
            sample: false,
            stream: false,
            seed: None,
            temperature: None,
            top_k: None,
            top_p: None,
        }
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
            "--device",
            "3",
            "--max-new-tokens",
            "7",
        ]))
        .unwrap();
        assert_eq!(parsed, Command::Generate(base_generate_args()));
    }

    #[test]
    fn generate_defaults_to_device_zero() {
        let parsed = parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "x",
        ]))
        .unwrap();
        let Command::Generate(arguments) = parsed else {
            panic!("expected generate command");
        };
        assert_eq!(arguments.device_ordinal, 0);
        assert!(!arguments.sample);
        assert!(!arguments.stream);
    }

    #[test]
    fn sample_requires_seed_and_accepts_builders_without_cuda() {
        let parsed = parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "x",
            "--sample",
            "--seed",
            "42",
            "--temperature",
            "0.8",
            "--top-k",
            "16",
            "--top-p",
            "0.9",
            "--stream",
        ]))
        .unwrap();
        let Command::Generate(arguments) = parsed else {
            panic!("expected generate command");
        };
        assert!(arguments.sample);
        assert!(arguments.stream);
        assert_eq!(arguments.seed, Some(42));
        assert_eq!(arguments.temperature, Some(0.8));
        assert_eq!(arguments.top_k, Some(16));
        assert_eq!(arguments.top_p, Some(0.9));
        let sampling = build_sampling_config(&arguments).unwrap();
        assert_eq!(sampling.seed, 42);
        assert_eq!(sampling.temperature, 0.8);
        assert_eq!(sampling.top_k, Some(16));
        assert_eq!(sampling.top_p, Some(0.9));
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
        assert!(parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "x",
            "--device",
            "-1",
        ]))
        .is_err());
        assert!(parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "x",
            "--device",
            "gpu0",
        ]))
        .is_err());
        assert!(parse_args(strings(&["unknown"])).is_err());
        assert!(parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "x",
            "--sample",
        ]))
        .is_err());
        assert!(parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "x",
            "--stream",
        ]))
        .is_err());
        assert!(parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "x",
            "--seed",
            "1",
        ]))
        .is_err());
        assert!(parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "x",
            "--temperature",
            "0.5",
        ]))
        .is_err());
        assert!(parse_args(strings(&[
            "generate",
            "--model",
            "/model",
            "--tokenizer",
            "/tokenizer.json",
            "--prompt",
            "x",
            "--sample",
            "--seed",
            "1",
            "--top-p",
            "1.5",
        ]))
        .is_err());
    }

    #[test]
    fn nvml_process_memory_defaults_without_cuda() {
        let parsed = parse_args(strings(&["nvml-process-memory"])).unwrap();
        assert_eq!(
            parsed,
            Command::NvmlProcessMemory(NvmlProcessMemoryArgs {
                device_ordinal: 0,
                json: false,
            })
        );
    }

    #[test]
    fn nvml_process_memory_parses_device_and_json_without_cuda() {
        let parsed =
            parse_args(strings(&["nvml-process-memory", "--device", "2", "--json"])).unwrap();
        assert_eq!(
            parsed,
            Command::NvmlProcessMemory(NvmlProcessMemoryArgs {
                device_ordinal: 2,
                json: true,
            })
        );
    }

    #[test]
    fn nvml_process_memory_help_and_rejects_without_cuda() {
        assert_eq!(
            parse_args(strings(&["nvml-process-memory", "--help"])).unwrap(),
            Command::Help
        );
        assert!(parse_args(strings(&["nvml-process-memory", "--device", "-1"])).is_err());
        assert!(parse_args(strings(&["nvml-process-memory", "--device", "gpu0"])).is_err());
        assert!(parse_args(strings(&["nvml-process-memory", "--device"])).is_err());
        assert!(parse_args(strings(&["nvml-process-memory", "--unknown"])).is_err());
        assert!(USAGE.contains("nvml-process-memory"));
        assert!(USAGE.contains("--json"));
    }

    #[test]
    fn nvml_process_memory_text_and_json_formatters_are_versioned() {
        let snapshot = NvmlProcessMemorySnapshotV1 {
            schema_version: NNIS_NVML_PROCESS_MEMORY_SNAPSHOT_VERSION,
            pid: 42,
            device_ordinal: 1,
            device_uuid: "GPU-test-uuid".to_string(),
            used_gpu_memory_bytes: 1_024,
        };
        let text = format_nvml_process_memory_text(&snapshot);
        assert!(text.contains("schema_version: 1"));
        assert!(text.contains("pid: 42"));
        assert!(text.contains("used_gpu_memory_bytes: 1024"));
        assert!(text.contains("Not physical residency"));
        let json = format_nvml_process_memory_json(&snapshot).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["pid"], 42);
        assert_eq!(value["device_ordinal"], 1);
        assert_eq!(value["device_uuid"], "GPU-test-uuid");
        assert_eq!(value["used_gpu_memory_bytes"], 1024);
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
