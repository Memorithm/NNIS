use nnis_model::{F16ReferenceModel, F16ReferencePlan, GenerationConfig};
use nnis_rt::{Context, Device, NnisError, Result, Stream};
use serde::Serialize;
use std::env;
use std::path::{Path, PathBuf};

const PROMPT_IDS: [u32; 3] = [22_007, 6_463, 314];
const EXPECTED_GREEDY_IDS: [u32; 32] = [
    260, 3_075, 338, 6_650, 260, 2_591, 284, 260, 8_872, 1_592, 30, 198, 198, 504, 8_872, 314, 253,
    8_304, 282, 260, 2_591, 30, 657, 314, 253, 19_284, 1_248, 338, 21_837, 260, 2_591, 30,
];

#[derive(Debug)]
struct Arguments {
    model_dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct QualificationReport {
    schema_version: u32,
    model: &'static str,
    prompt_ids: Vec<u32>,
    decode_steps: usize,
    f16_plan: F16ReferencePlan,
    generated_ids: Vec<u32>,
    expected_ids: Vec<u32>,
    exact_greedy_32_of_32: bool,
}

fn parse_arguments() -> std::result::Result<Arguments, String> {
    let mut args = env::args().skip(1);
    let mut model_dir = None;
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--model" => {
                model_dir = Some(PathBuf::from(
                    args.next().ok_or("--model requires a directory")?,
                ));
            }
            "--help" | "-h" => {
                return Err("usage: smollm2_f16_reference --model DIR".to_string());
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Arguments {
        model_dir: model_dir.ok_or("missing --model DIR")?,
    })
}

fn validate_model_shape(model: &F16ReferenceModel) -> Result<()> {
    let config = model.config();
    if config.vocab_size != 49_152
        || config.hidden_size != 576
        || config.intermediate_size != 1_536
        || config.num_hidden_layers != 30
        || config.num_attention_heads != 9
        || config.num_key_value_heads != 3
        || config.head_dim() != 64
        || config.max_position_embeddings != 8_192
        || config.eos_token_id != Some(0)
        || config.rope_theta != 100_000.0
    {
        return Err(NnisError::invalid_input(format!(
            "loaded model config does not match pinned SmolLM2-135M: {config:?}"
        )));
    }
    Ok(())
}

fn run(model_dir: &Path) -> Result<()> {
    let device = Device::first()?;
    let context = Context::new(&device)?;
    let construction_stream = Stream::new(&context)?;
    let plan = F16ReferencePlan::edge_llm_v0_10_0_alignment();
    let model = F16ReferenceModel::load_directory(&context, &construction_stream, model_dir, plan)?;
    validate_model_shape(&model)?;

    let generated = model.new_session()?.generate(
        &PROMPT_IDS,
        GenerationConfig::greedy(EXPECTED_GREEDY_IDS.len()),
    )?;
    let exact = generated.as_slice() == EXPECTED_GREEDY_IDS;

    let report = QualificationReport {
        schema_version: 1,
        model: "HuggingFaceTB/SmolLM2-135M@93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        prompt_ids: PROMPT_IDS.to_vec(),
        decode_steps: EXPECTED_GREEDY_IDS.len(),
        f16_plan: plan,
        generated_ids: generated.clone(),
        expected_ids: EXPECTED_GREEDY_IDS.to_vec(),
        exact_greedy_32_of_32: exact,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| NnisError::invalid_input(format!("serialize report: {error}")))?
    );

    if !exact {
        let divergence = generated
            .iter()
            .zip(EXPECTED_GREEDY_IDS.iter())
            .position(|(actual, expected)| actual != expected);
        return Err(NnisError::invalid_input(format!(
            "F16 SmolLM2 greedy trajectory diverged from the qualified R1 trajectory at {divergence:?}"
        )));
    }
    println!("NNIS_F16_SMOLLM2_EXACT_GREEDY_32_OF_32_OK");
    Ok(())
}

fn main() {
    let arguments = match parse_arguments() {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(&arguments.model_dir) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
