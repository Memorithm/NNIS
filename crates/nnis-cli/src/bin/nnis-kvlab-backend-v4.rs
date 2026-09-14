use nnis::{Context, Device, Stream};
use nnis_model::{load_model_from_safetensors, Model, SafetensorsLoadConfig};
use serde::Deserialize;
use serde_json::{Number, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

const REQUEST_SCHEMA: &str = "kvlab.prospect-kv-backend-request/v4";
const RESPONSE_SCHEMA: &str = "kvlab.prospect-kv-backend-response/v4";
const TRACE_SCHEMA: &str = "kvlab.prospect-kv-real-model-position-trace/v1";
const ARTIFACT_SCHEMA: &str = "nnis.kvlab-position-teacher-forced-evaluation/v1";
const DEFAULT_RUNTIME_BACKEND: &str = "nnis-kvlab-v4";

const USAGE: &str = "Usage:\n  nnis-kvlab-backend-v4 --model DIR --model-id ID --model-revision REV --tokenizer-revision REV --runtime-revision REV [--runtime-backend ID] [--device N]\n\nReads one canonical KVLab backend request v4 from stdin and writes one canonical response v4 to stdout. KV row identity is the zero-based input-sequence position. The first evaluation token is a bridge token processed after candidate compaction; metrics score only the remaining teacher-forced evaluation tokens.";

#[derive(Debug)]
struct Args {
    model_dir: PathBuf,
    model_id: String,
    model_revision: String,
    tokenizer_revision: String,
    runtime_backend: String,
    runtime_revision: String,
    device_ordinal: i32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestV4 {
    schema: String,
    mode: String,
    experiment_id: String,
    run_repository_revision: String,
    model_id: String,
    model_revision: String,
    tokenizer_revision: String,
    runtime_backend: String,
    runtime_revision: String,
    evaluation_id: String,
    trace_sha256: String,
    seed: u64,
    model_input_token_ids: Vec<u32>,
    retained_positions: Vec<usize>,
    evaluation_token_ids: Vec<u32>,
    bytes_per_token: u64,
    policy: Option<String>,
}

#[derive(Debug)]
struct ScoreStep {
    target_token_id: u32,
    top1_token_id: u32,
    target_nll: f64,
}

fn required_value<I>(arguments: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    arguments
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_args<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut arguments = arguments.into_iter();
    let mut model_dir = None;
    let mut model_id = None;
    let mut model_revision = None;
    let mut tokenizer_revision = None;
    let mut runtime_backend = DEFAULT_RUNTIME_BACKEND.to_string();
    let mut runtime_revision = None;
    let mut device_ordinal = 0_i32;

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model" => {
                model_dir = Some(PathBuf::from(required_value(&mut arguments, "--model")?))
            }
            "--model-id" => model_id = Some(required_value(&mut arguments, "--model-id")?),
            "--model-revision" => {
                model_revision = Some(required_value(&mut arguments, "--model-revision")?)
            }
            "--tokenizer-revision" => {
                tokenizer_revision = Some(required_value(&mut arguments, "--tokenizer-revision")?)
            }
            "--runtime-backend" => {
                runtime_backend = required_value(&mut arguments, "--runtime-backend")?
            }
            "--runtime-revision" => {
                runtime_revision = Some(required_value(&mut arguments, "--runtime-revision")?)
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
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown argument {other:?}\n\n{USAGE}")),
        }
    }

    let args = Args {
        model_dir: model_dir.ok_or_else(|| "missing --model DIR".to_string())?,
        model_id: model_id.ok_or_else(|| "missing --model-id ID".to_string())?,
        model_revision: model_revision.ok_or_else(|| "missing --model-revision REV".to_string())?,
        tokenizer_revision: tokenizer_revision
            .ok_or_else(|| "missing --tokenizer-revision REV".to_string())?,
        runtime_backend,
        runtime_revision: runtime_revision
            .ok_or_else(|| "missing --runtime-revision REV".to_string())?,
        device_ordinal,
    };

    for (name, value) in [
        ("model-id", args.model_id.as_str()),
        ("model-revision", args.model_revision.as_str()),
        ("tokenizer-revision", args.tokenizer_revision.as_str()),
        ("runtime-backend", args.runtime_backend.as_str()),
        ("runtime-revision", args.runtime_revision.as_str()),
    ] {
        require_ascii_text(name, value)?;
    }
    Ok(args)
}

fn parse_request(payload: &str, args: &Args) -> Result<RequestV4, String> {
    let value: Value =
        serde_json::from_str(payload).map_err(|error| format!("invalid request JSON: {error}"))?;
    let canonical = serde_json::to_string(&value)
        .map_err(|error| format!("failed to canonicalize request JSON: {error}"))?;
    if canonical != payload {
        return Err("backend request JSON must be canonical".to_string());
    }
    let request: RequestV4 =
        serde_json::from_value(value).map_err(|error| format!("invalid request schema: {error}"))?;
    validate_request(&request, args)?;
    Ok(request)
}

fn validate_request(request: &RequestV4, args: &Args) -> Result<(), String> {
    if request.schema != REQUEST_SCHEMA {
        return Err(format!("unsupported request schema {:?}", request.schema));
    }
    for (name, value) in [
        ("experiment_id", request.experiment_id.as_str()),
        ("model_id", request.model_id.as_str()),
        ("model_revision", request.model_revision.as_str()),
        ("tokenizer_revision", request.tokenizer_revision.as_str()),
        ("runtime_backend", request.runtime_backend.as_str()),
        ("runtime_revision", request.runtime_revision.as_str()),
        ("evaluation_id", request.evaluation_id.as_str()),
    ] {
        require_ascii_text(name, value)?;
    }
    if !is_lower_hex(&request.run_repository_revision, 40) {
        return Err("run_repository_revision must be a lowercase full Git SHA".to_string());
    }
    if !is_lower_hex(&request.trace_sha256, 64) {
        return Err("trace_sha256 must be a lowercase SHA-256 digest".to_string());
    }
    if request.bytes_per_token == 0 {
        return Err("bytes_per_token must be positive".to_string());
    }
    if request.model_id != args.model_id
        || request.model_revision != args.model_revision
        || request.tokenizer_revision != args.tokenizer_revision
        || request.runtime_backend != args.runtime_backend
        || request.runtime_revision != args.runtime_revision
    {
        return Err(
            "request model/runtime provenance does not match backend configuration".to_string(),
        );
    }
    if request.model_input_token_ids.is_empty() {
        return Err("model_input_token_ids must not be empty".to_string());
    }
    if request.evaluation_token_ids.len() < 2 {
        return Err(
            "NNIS backend requires at least two evaluation tokens: one bridge and one scored target"
                .to_string(),
        );
    }
    validate_retained_positions(
        request.model_input_token_ids.len(),
        &request.retained_positions,
    )?;

    let full_positions: Vec<usize> = (0..request.model_input_token_ids.len()).collect();
    match request.mode.as_str() {
        "baseline" => {
            if request.policy.is_some() {
                return Err("baseline request policy must be null".to_string());
            }
            if request.retained_positions != full_positions {
                return Err("baseline must retain every input position".to_string());
            }
        }
        "candidate" => {
            let policy = request
                .policy
                .as_deref()
                .ok_or_else(|| "candidate request requires a policy".to_string())?;
            require_ascii_text("policy", policy)?;
            if request.retained_positions == full_positions {
                return Err("candidate selection duplicates the full-cache baseline".to_string());
            }
        }
        other => return Err(format!("unsupported request mode {other:?}")),
    }

    let expected_trace_sha256 = trace_sha256(request)?;
    if expected_trace_sha256 != request.trace_sha256 {
        return Err("request trace_sha256 does not match embedded position trace".to_string());
    }
    Ok(())
}

fn validate_retained_positions(input_len: usize, positions: &[usize]) -> Result<(), String> {
    let mut previous = None;
    for &position in positions {
        if position >= input_len {
            return Err(format!(
                "retained_positions contains {position} outside input prefix 0..{input_len}"
            ));
        }
        if previous.is_some_and(|last| position <= last) {
            return Err("retained_positions must be strictly increasing and unique".to_string());
        }
        previous = Some(position);
    }
    Ok(())
}

fn trace_sha256(request: &RequestV4) -> Result<String, String> {
    let mut trace = BTreeMap::<String, Value>::new();
    trace.insert(
        "evaluation_token_ids".to_string(),
        serde_json::to_value(&request.evaluation_token_ids)
            .map_err(|error| format!("failed to serialize evaluation ids: {error}"))?,
    );
    trace.insert(
        "model_input_token_ids".to_string(),
        serde_json::to_value(&request.model_input_token_ids)
            .map_err(|error| format!("failed to serialize model ids: {error}"))?,
    );
    trace.insert(
        "schema".to_string(),
        Value::String(TRACE_SCHEMA.to_string()),
    );
    let payload = serde_json::to_vec(&trace)
        .map_err(|error| format!("failed to serialize canonical trace: {error}"))?;
    Ok(sha256_hex(&payload))
}

fn execute_request(args: &Args, request: &RequestV4) -> Result<(Vec<u8>, f64, f64), String> {
    let device = Device::get(args.device_ordinal).map_err(|error| {
        format!(
            "failed to select CUDA device {}: {error}",
            args.device_ordinal
        )
    })?;
    let context =
        Context::new(&device).map_err(|error| format!("failed to create CUDA context: {error}"))?;
    let construction_stream =
        Stream::new(&context).map_err(|error| format!("failed to create CUDA stream: {error}"))?;
    let load_config = SafetensorsLoadConfig {
        repo_id: None,
        revision: None,
        local_dir: args.model_dir.to_string_lossy().into_owned(),
    };
    let (config, weights) =
        load_model_from_safetensors(&context, &construction_stream, &load_config).map_err(
            |error| format!("failed to load model {}: {error}", args.model_dir.display()),
        )?;
    let model = Model::new(config, weights, &construction_stream)
        .map_err(|error| format!("failed to construct NNIS model: {error}"))?;

    validate_model_token_range(request, model.config().vocab_size)?;
    let decoded_evaluation_tokens = request
        .evaluation_token_ids
        .len()
        .checked_sub(1)
        .ok_or_else(|| "evaluation token length underflow".to_string())?;
    let required_positions = request
        .model_input_token_ids
        .len()
        .checked_add(decoded_evaluation_tokens)
        .ok_or_else(|| "model input plus evaluation length overflows usize".to_string())?;
    if required_positions > model.config().max_position_embeddings {
        return Err(format!(
            "model input plus decoded evaluation requires {required_positions} logical positions but model capacity is {}",
            model.config().max_position_embeddings
        ));
    }

    let mut session = model
        .new_session()
        .map_err(|error| format!("failed to create NNIS inference session: {error}"))?;
    session
        .prefill(&request.model_input_token_ids)
        .map_err(|error| format!("NNIS prefill failed: {error}"))?;

    if request.mode == "candidate" {
        session
            .compact_kv_cache_rows(&request.retained_positions)
            .map_err(|error| format!("NNIS KV compaction failed: {error}"))?;
    }
    let position_after_selection = session.position();
    if position_after_selection != request.model_input_token_ids.len() {
        return Err(format!(
            "logical position changed across selection: expected {}, got {position_after_selection}",
            request.model_input_token_ids.len()
        ));
    }

    let bridge = request.evaluation_token_ids[0];
    let mut logits = session
        .decode_one(bridge)
        .map_err(|error| format!("NNIS bridge decode failed: {error}"))?;
    let mut steps = Vec::with_capacity(request.evaluation_token_ids.len() - 1);

    for (offset, &target) in request.evaluation_token_ids[1..].iter().enumerate() {
        let (nll, top1) = score_logits(&logits, target)?;
        steps.push(ScoreStep {
            target_token_id: target,
            top1_token_id: top1,
            target_nll: nll,
        });
        if offset + 2 < request.evaluation_token_ids.len() {
            logits = session
                .decode_one(target)
                .map_err(|error| format!("NNIS teacher-forced decode failed: {error}"))?;
        }
    }

    let mean_nll = steps.iter().map(|step| step.target_nll).sum::<f64>() / steps.len() as f64;
    let token_accuracy = steps
        .iter()
        .filter(|step| step.target_token_id == step.top1_token_id)
        .count() as f64
        / steps.len() as f64;
    if !mean_nll.is_finite() || !token_accuracy.is_finite() {
        return Err("evaluation produced non-finite metrics".to_string());
    }

    let artifact = render_artifact(request, position_after_selection, bridge, &steps)?;
    Ok((artifact, mean_nll, token_accuracy))
}

fn validate_model_token_range(request: &RequestV4, vocab_size: usize) -> Result<(), String> {
    for (field, tokens) in [
        (
            "model_input_token_ids",
            request.model_input_token_ids.as_slice(),
        ),
        (
            "evaluation_token_ids",
            request.evaluation_token_ids.as_slice(),
        ),
    ] {
        if let Some(&token) = tokens.iter().find(|&&token| token as usize >= vocab_size) {
            return Err(format!(
                "{field} contains token id {token} outside model vocabulary 0..{vocab_size}"
            ));
        }
    }
    Ok(())
}

fn score_logits(logits: &[f32], target: u32) -> Result<(f64, u32), String> {
    if logits.is_empty() {
        return Err("NNIS returned empty logits".to_string());
    }
    if target as usize >= logits.len() {
        return Err(format!(
            "target token {target} is outside logits width {}",
            logits.len()
        ));
    }

    let mut maximum = f64::NEG_INFINITY;
    let mut top1_index = 0_usize;
    let mut top1_value = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("logit {index} is non-finite"));
        }
        maximum = maximum.max(value as f64);
        if value > top1_value {
            top1_value = value;
            top1_index = index;
        }
    }
    let sum_exp = logits
        .iter()
        .map(|&value| ((value as f64) - maximum).exp())
        .sum::<f64>();
    if !sum_exp.is_finite() || sum_exp <= 0.0 {
        return Err("log-sum-exp normalization is invalid".to_string());
    }
    let log_sum_exp = maximum + sum_exp.ln();
    let nll = log_sum_exp - logits[target as usize] as f64;
    if !nll.is_finite() || nll < 0.0 {
        return Err("target negative log-likelihood is invalid".to_string());
    }
    let top1 = u32::try_from(top1_index)
        .map_err(|_| "logits width exceeds u32 token-id range".to_string())?;
    Ok((nll, top1))
}

fn render_artifact(
    request: &RequestV4,
    logical_position: usize,
    bridge_token_id: u32,
    steps: &[ScoreStep],
) -> Result<Vec<u8>, String> {
    let mut root = BTreeMap::<String, Value>::new();
    root.insert(
        "applied_policy".to_string(),
        request.policy.clone().map_or(Value::Null, Value::String),
    );
    root.insert(
        "bridge_token_id".to_string(),
        Value::Number(Number::from(bridge_token_id)),
    );
    root.insert(
        "logical_position_after_selection".to_string(),
        Value::Number(Number::from(logical_position as u64)),
    );
    root.insert("mode".to_string(), Value::String(request.mode.clone()));
    root.insert(
        "retained_positions".to_string(),
        serde_json::to_value(&request.retained_positions)
            .map_err(|error| format!("failed to serialize retained positions: {error}"))?,
    );
    root.insert(
        "schema".to_string(),
        Value::String(ARTIFACT_SCHEMA.to_string()),
    );
    root.insert(
        "seed".to_string(),
        Value::Number(Number::from(request.seed)),
    );
    root.insert(
        "trace_sha256".to_string(),
        Value::String(request.trace_sha256.clone()),
    );

    let mut step_values = Vec::with_capacity(steps.len());
    for step in steps {
        let mut value = BTreeMap::<String, Value>::new();
        value.insert(
            "target_nll".to_string(),
            Value::Number(
                Number::from_f64(step.target_nll)
                    .ok_or_else(|| "target_nll is not JSON-finite".to_string())?,
            ),
        );
        value.insert(
            "target_token_id".to_string(),
            Value::Number(Number::from(step.target_token_id)),
        );
        value.insert(
            "top1_token_id".to_string(),
            Value::Number(Number::from(step.top1_token_id)),
        );
        step_values.push(
            serde_json::to_value(value)
                .map_err(|error| format!("failed to serialize score step: {error}"))?,
        );
    }
    root.insert("steps".to_string(), Value::Array(step_values));
    serde_json::to_vec(&root)
        .map_err(|error| format!("failed to serialize evaluation artefact: {error}"))
}

fn render_response(
    request: &RequestV4,
    request_sha256: &str,
    artifact: &[u8],
    mean_nll: f64,
    token_accuracy: f64,
) -> Result<String, String> {
    let mut root = BTreeMap::<String, Value>::new();
    root.insert(
        "applied_mode".to_string(),
        Value::String(request.mode.clone()),
    );
    root.insert(
        "applied_policy".to_string(),
        request.policy.clone().map_or(Value::Null, Value::String),
    );
    root.insert(
        "applied_retained_positions".to_string(),
        serde_json::to_value(&request.retained_positions)
            .map_err(|error| format!("failed to serialize applied retained positions: {error}"))?,
    );
    root.insert(
        "metrics".to_string(),
        Value::Array(vec![
            metric_value(
                "mean_nll",
                "quality",
                "nat_per_token",
                "lower_is_better",
                mean_nll,
            )?,
            metric_value(
                "token_accuracy",
                "quality",
                "ratio",
                "higher_is_better",
                token_accuracy,
            )?,
        ]),
    );
    root.insert(
        "output_artifact_base64".to_string(),
        Value::String(base64::encode(artifact)),
    );
    root.insert(
        "request_sha256".to_string(),
        Value::String(request_sha256.to_string()),
    );
    root.insert(
        "schema".to_string(),
        Value::String(RESPONSE_SCHEMA.to_string()),
    );
    serde_json::to_string(&root)
        .map_err(|error| format!("failed to serialize backend response: {error}"))
}

fn metric_value(
    name: &str,
    kind: &str,
    unit: &str,
    preference: &str,
    value: f64,
) -> Result<Value, String> {
    let number =
        Number::from_f64(value).ok_or_else(|| format!("metric {name} is not JSON-finite"))?;
    let mut metric = BTreeMap::<String, Value>::new();
    metric.insert("kind".to_string(), Value::String(kind.to_string()));
    metric.insert("name".to_string(), Value::String(name.to_string()));
    metric.insert(
        "preference".to_string(),
        Value::String(preference.to_string()),
    );
    metric.insert("unit".to_string(), Value::String(unit.to_string()));
    metric.insert("value".to_string(), Value::Number(number));
    serde_json::to_value(metric).map_err(|error| format!("failed to serialize metric: {error}"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_ascii_text(name: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{name} must be non-empty"));
    }
    if !value.is_ascii() {
        return Err(format!(
            "{name} must be ASCII for canonical JSON interoperability"
        ));
    }
    Ok(())
}

fn run(args: &Args, payload: &str) -> Result<String, String> {
    let request_sha256 = sha256_hex(payload.as_bytes());
    let request = parse_request(payload, args)?;
    let (artifact, mean_nll, token_accuracy) = execute_request(args, &request)?;
    render_response(
        &request,
        &request_sha256,
        &artifact,
        mean_nll,
        token_accuracy,
    )
}

fn main() -> ExitCode {
    let args = match parse_args(env::args().skip(1)) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let mut payload = String::new();
    if let Err(error) = io::stdin().read_to_string(&mut payload) {
        eprintln!("failed to read backend request: {error}");
        return ExitCode::from(2);
    }
    match run(&args, &payload) {
        Ok(response) => {
            let mut stdout = io::stdout();
            if let Err(error) = stdout.write_all(response.as_bytes()) {
                eprintln!("failed to write backend response: {error}");
                return ExitCode::from(2);
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            model_dir: PathBuf::from("/tmp/model"),
            model_id: "example/model".to_string(),
            model_revision: "model-r1".to_string(),
            tokenizer_revision: "tok-r1".to_string(),
            runtime_backend: DEFAULT_RUNTIME_BACKEND.to_string(),
            runtime_revision: "1fc4dc6d5f45e7d306d8e35823197afc2cf9846c".to_string(),
            device_ordinal: 0,
        }
    }

    fn request_json() -> String {
        let model = vec![42_u32, 7, 42, 9, 7];
        let evaluation = vec![5_u32, 5, 6];
        let mut trace = BTreeMap::<String, Value>::new();
        trace.insert(
            "evaluation_token_ids".to_string(),
            serde_json::to_value(&evaluation).unwrap(),
        );
        trace.insert(
            "model_input_token_ids".to_string(),
            serde_json::to_value(&model).unwrap(),
        );
        trace.insert(
            "schema".to_string(),
            Value::String(TRACE_SCHEMA.to_string()),
        );
        let trace_sha256 = sha256_hex(&serde_json::to_vec(&trace).unwrap());

        let mut request = BTreeMap::<String, Value>::new();
        request.insert(
            "bytes_per_token".to_string(),
            Value::Number(Number::from(64_u64)),
        );
        request.insert(
            "evaluation_id".to_string(),
            Value::String("holdout-1".to_string()),
        );
        request.insert(
            "evaluation_token_ids".to_string(),
            serde_json::to_value(&evaluation).unwrap(),
        );
        request.insert(
            "experiment_id".to_string(),
            Value::String("experiment-1".to_string()),
        );
        request.insert("mode".to_string(), Value::String("candidate".to_string()));
        request.insert(
            "model_id".to_string(),
            Value::String("example/model".to_string()),
        );
        request.insert(
            "model_input_token_ids".to_string(),
            serde_json::to_value(&model).unwrap(),
        );
        request.insert(
            "model_revision".to_string(),
            Value::String("model-r1".to_string()),
        );
        request.insert("policy".to_string(), Value::String("lru".to_string()));
        request.insert(
            "retained_positions".to_string(),
            serde_json::to_value(vec![0_usize, 2, 4]).unwrap(),
        );
        request.insert(
            "run_repository_revision".to_string(),
            Value::String("b".repeat(40)),
        );
        request.insert(
            "runtime_backend".to_string(),
            Value::String(DEFAULT_RUNTIME_BACKEND.to_string()),
        );
        request.insert(
            "runtime_revision".to_string(),
            Value::String("1fc4dc6d5f45e7d306d8e35823197afc2cf9846c".to_string()),
        );
        request.insert(
            "schema".to_string(),
            Value::String(REQUEST_SCHEMA.to_string()),
        );
        request.insert("seed".to_string(), Value::Number(Number::from(7_u64)));
        request.insert(
            "tokenizer_revision".to_string(),
            Value::String("tok-r1".to_string()),
        );
        request.insert("trace_sha256".to_string(), Value::String(trace_sha256));
        serde_json::to_string(&request).unwrap()
    }

    #[test]
    fn request_accepts_repeated_model_tokens_and_position_identity() {
        let payload = request_json();
        let request = parse_request(&payload, &args()).unwrap();
        assert_eq!(request.model_input_token_ids, [42, 7, 42, 9, 7]);
        assert_eq!(request.retained_positions, [0, 2, 4]);
    }

    #[test]
    fn retained_positions_are_strict_and_bounded() {
        assert!(validate_retained_positions(5, &[]).is_ok());
        assert!(validate_retained_positions(5, &[0, 2, 4]).is_ok());
        assert!(validate_retained_positions(5, &[0, 0]).is_err());
        assert!(validate_retained_positions(5, &[2, 1]).is_err());
        assert!(validate_retained_positions(5, &[0, 5]).is_err());
    }

    #[test]
    fn trace_hash_detects_token_drift() {
        let payload = request_json();
        let mut value: Value = serde_json::from_str(&payload).unwrap();
        value["model_input_token_ids"] = serde_json::json!([42, 7, 41, 9, 7]);
        let tampered = serde_json::to_string(&value).unwrap();
        let error = parse_request(&tampered, &args()).unwrap_err();
        assert!(error.contains("trace_sha256"), "{error}");
    }

    #[test]
    fn backend_requires_bridge_plus_scored_target() {
        let payload = request_json();
        let mut value: Value = serde_json::from_str(&payload).unwrap();
        value["evaluation_token_ids"] = serde_json::json!([5]);
        let tampered = serde_json::to_string(&value).unwrap();
        let error = parse_request(&tampered, &args()).unwrap_err();
        assert!(error.contains("at least two evaluation tokens"), "{error}");
    }

    #[test]
    fn noncanonical_request_is_rejected_before_execution() {
        let value: Value = serde_json::from_str(&request_json()).unwrap();
        let pretty = serde_json::to_string_pretty(&value).unwrap();
        let error = parse_request(&pretty, &args()).unwrap_err();
        assert!(error.contains("canonical"), "{error}");
    }

    #[test]
    fn score_logits_matches_direct_softmax_oracle() {
        let logits = [0.0_f32, 1.0, 2.0];
        let (nll, top1) = score_logits(&logits, 1).unwrap();
        let expected = (1.0_f64 + (-1.0_f64).exp() + 1.0_f64.exp()).ln();
        assert!((nll - expected).abs() < 1.0e-12, "{nll} != {expected}");
        assert_eq!(top1, 2);
    }

    #[test]
    fn response_is_sorted_canonical_ascii_json() {
        let request = parse_request(&request_json(), &args()).unwrap();
        let response = render_response(&request, &"c".repeat(64), b"artifact", 0.25, 0.5).unwrap();
        let reparsed: Value = serde_json::from_str(&response).unwrap();
        assert_eq!(serde_json::to_string(&reparsed).unwrap(), response);
        assert!(response.is_ascii());
        assert!(response.contains("kvlab.prospect-kv-backend-response/v4"));
        assert!(response.contains("applied_retained_positions"));
    }
}
