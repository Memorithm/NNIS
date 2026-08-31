use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase};
use nnis_jit::JitCompiler;
use nnis_kernels::{F32Gemv, F32LmHeadArgmax, F32TopK};
use nnis_model::load_model_directory;
use nnis_rt::{gpu_context, DeviceBuffer, NnisError, Result, Stream};
use serde_json::json;
use std::path::PathBuf;

const EXPECTED_HIDDEN: usize = 576;
const EXPECTED_VOCAB: usize = 49_152;
const REFERENCE_GEMV_BLOCK_SIZE: u32 = 64;

fn env_usize(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(value) => value.parse::<usize>().map_err(|error| {
            NnisError::invalid_input(format!("invalid {name}={value:?}: {error}"))
        }),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(NnisError::invalid_input(format!(
            "failed to read {name}: {error}"
        ))),
    }
}

fn parse_model_dir() -> Result<PathBuf> {
    let mut args = std::env::args().skip(1);
    let mut model = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model = Some(PathBuf::from(args.next().ok_or_else(|| {
                    NnisError::invalid_input("--model requires a directory")
                })?));
            }
            "--help" | "-h" => {
                return Err(NnisError::invalid_input(
                    "usage: smollm2_lm_head_argmax_candidates --model DIR",
                ));
            }
            other => {
                return Err(NnisError::invalid_input(format!(
                    "unknown argument {other:?}"
                )));
            }
        }
    }
    model.ok_or_else(|| NnisError::invalid_input("--model DIR is required"))
}

fn deterministic_hidden(k: usize) -> Vec<f32> {
    (0..k)
        .map(|index| ((index * 29 % 61) as f32 - 30.0) * 0.015625)
        .collect()
}

fn read_scalar_f32(buffer: &DeviceBuffer<f32>, stream: &Stream) -> Result<f32> {
    Ok(buffer.to_vec(stream)?[0])
}

fn read_scalar_u32(buffer: &DeviceBuffer<u32>, stream: &Stream) -> Result<u32> {
    Ok(buffer.to_vec(stream)?[0])
}

fn assert_same_winner(
    stream: &Stream,
    reference_value: &DeviceBuffer<f32>,
    reference_index: &DeviceBuffer<u32>,
    candidate_value: &DeviceBuffer<f32>,
    candidate_index: &DeviceBuffer<u32>,
) -> Result<(f32, u32)> {
    let reference_value = read_scalar_f32(reference_value, stream)?;
    let reference_index = read_scalar_u32(reference_index, stream)?;
    let candidate_value = read_scalar_f32(candidate_value, stream)?;
    let candidate_index = read_scalar_u32(candidate_index, stream)?;
    if reference_index != candidate_index {
        return Err(NnisError::unsupported(format!(
            "fused LM-head argmax token mismatch: reference {reference_index}, candidate {candidate_index}"
        )));
    }
    if reference_value.to_bits() != candidate_value.to_bits() {
        return Err(NnisError::unsupported(format!(
            "fused LM-head argmax winning value is not bitwise-equivalent: reference {reference_value:e}, candidate {candidate_value:e}"
        )));
    }
    Ok((reference_value, reference_index))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("SmolLM2 LM-head argmax candidate benchmark failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let model_dir = parse_model_dir()?;
    let context = gpu_context().ok_or_else(|| NnisError::unsupported("no CUDA device"))?;
    let stream = Stream::new(&context)?;
    let (config, weights) = load_model_directory(&context, &stream, &model_dir)?;
    if config.hidden_size != EXPECTED_HIDDEN || config.vocab_size != EXPECTED_VOCAB {
        return Err(NnisError::unsupported(format!(
            "E2 benchmark requires SmolLM2 shape hidden={EXPECTED_HIDDEN}, vocab={EXPECTED_VOCAB}; got hidden={}, vocab={}",
            config.hidden_size, config.vocab_size
        )));
    }

    let compiler = JitCompiler::new();
    let reference_gemv =
        F32Gemv::load_with_block_size(&context, &compiler, REFERENCE_GEMV_BLOCK_SIZE)?;
    let top_k = F32TopK::load(&context, &compiler)?;
    let candidate_block_size = u32::try_from(env_usize("NNIS_LM_HEAD_ARGMAX_BLOCK_SIZE", 64)?)
        .map_err(|_| NnisError::invalid_input("NNIS_LM_HEAD_ARGMAX_BLOCK_SIZE exceeds u32"))?;
    let fused = F32LmHeadArgmax::load_with_block_size(&context, &compiler, candidate_block_size)?;

    let warmups = env_usize("NNIS_PROFILE_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_PROFILE_ITERATIONS", 100)?;
    let bench_config = BenchConfig::new(warmups, iterations);
    bench_config.validate()?;

    let hidden_host = deterministic_hidden(config.hidden_size);
    let hidden = DeviceBuffer::from_host(&context, &stream, &hidden_host)?;
    let lm_head = weights.lm_head.tensor().as_f32()?;

    let logits = DeviceBuffer::<f32>::new(&context, config.vocab_size)?;
    let reference_value = DeviceBuffer::<f32>::new(&context, 1)?;
    let reference_index = DeviceBuffer::<u32>::new(&context, 1)?;
    let reference_top_k_workspace = top_k.workspace(&context, config.vocab_size)?;

    let candidate_value = DeviceBuffer::<f32>::new(&context, 1)?;
    let candidate_index = DeviceBuffer::<u32>::new(&context, 1)?;
    let candidate_workspace = fused.workspace(&context, config.vocab_size)?;

    reference_gemv.project_kn(
        &stream,
        &hidden,
        lm_head,
        &logits,
        config.hidden_size,
        config.vocab_size,
    )?;
    top_k.top_k(&stream, &logits, &reference_value, &reference_index, 1)?;
    fused.argmax_kn(
        &stream,
        &hidden,
        lm_head,
        &candidate_value,
        &candidate_index,
        config.hidden_size,
        config.vocab_size,
        &candidate_workspace,
    )?;
    let winner = assert_same_winner(
        &stream,
        &reference_value,
        &reference_index,
        &candidate_value,
        &candidate_index,
    )?;

    let work_items = config.hidden_size as u64 * config.vocab_size as u64;
    let reference_case = BenchmarkCase::new("smollm2_lm_head_gemv64_plus_top1", "f32")
        .with_dimension("k", config.hidden_size as u64)
        .with_dimension("vocab", config.vocab_size as u64)
        .with_work_items(work_items);
    let candidate_case = BenchmarkCase::new("smollm2_lm_head_fused_argmax", "f32")
        .with_dimension("k", config.hidden_size as u64)
        .with_dimension("vocab", config.vocab_size as u64)
        .with_dimension("block_size", candidate_block_size as u64)
        .with_work_items(work_items);

    let reference_report = benchmark_gpu(&context, &stream, reference_case, bench_config, || {
        // SAFETY: buffers/workspaces remain alive through the benchmark;
        // the harness serializes and drains the stream.
        unsafe {
            reference_gemv.enqueue_project_kn(
                &stream,
                &hidden,
                lm_head,
                &logits,
                config.hidden_size,
                config.vocab_size,
            )?;
            top_k.enqueue_top_k(
                &stream,
                &logits,
                &reference_value,
                &reference_index,
                1,
                &reference_top_k_workspace,
            )
        }
    })?;

    let candidate_report = benchmark_gpu(&context, &stream, candidate_case, bench_config, || {
        // SAFETY: buffers/workspace remain alive through the benchmark;
        // the harness serializes and drains the stream.
        unsafe {
            fused.enqueue_argmax_kn(
                &stream,
                &hidden,
                lm_head,
                &candidate_value,
                &candidate_index,
                config.hidden_size,
                config.vocab_size,
                &candidate_workspace,
            )
        }
    })?;

    let winner_after = assert_same_winner(
        &stream,
        &reference_value,
        &reference_index,
        &candidate_value,
        &candidate_index,
    )?;
    if winner.1 != winner_after.1 || winner.0.to_bits() != winner_after.0.to_bits() {
        return Err(NnisError::unsupported(
            "LM-head argmax winner changed across benchmark timing",
        ));
    }

    let report = json!({
        "schema_version": 1,
        "experiment": "E2-f32-lm-head-gemv64-top1-vs-fused-argmax",
        "promotion_state": "candidate-only; model runtime unchanged",
        "correctness_gate": "candidate winning token id and winning f32 value must equal E1.1 GEMV64 + F32TopK(1), with the winning value bit-for-bit identical before and after timing",
        "representation": {
            "logical_weights": "unchanged",
            "execution_weight_dtype": "f32",
            "candidate_changes_representation": false,
        },
        "hardware": {
            "gpu_name": context.props().name,
            "compute_capability_major": context.props().compute_capability.0,
            "compute_capability_minor": context.props().compute_capability.1,
            "multiprocessor_count": context.props().multiprocessor_count,
        },
        "shape": {
            "k": config.hidden_size,
            "vocab": config.vocab_size,
        },
        "benchmark_config": {
            "warmups": warmups,
            "iterations": iterations,
            "reference_gemv_block_size": REFERENCE_GEMV_BLOCK_SIZE,
            "reference_top_k_block_size": top_k.block_size(),
            "candidate_block_size": candidate_block_size,
            "candidate_count": candidate_workspace.candidate_count(),
        },
        "winner": {
            "token_id": winner.1,
            "value": winner.0,
            "value_bits": winner.0.to_bits(),
            "bitwise_equivalent": true,
        },
        "reference": {
            "path": "F32Gemv::project_kn(block=64) -> materialized vocab logits -> F32TopK(k=1)",
            "median_ms": reference_report.statistics.median_ms,
            "p95_ms": reference_report.statistics.p95_ms,
        },
        "candidate": {
            "path": "F32LmHeadArgmax -> block candidates -> deterministic final winner; no vocab logits buffer write",
            "median_ms": candidate_report.statistics.median_ms,
            "p95_ms": candidate_report.statistics.p95_ms,
        },
        "isolated": {
            "latency_speedup_reference_over_candidate": reference_report.statistics.median_ms / candidate_report.statistics.median_ms,
            "latency_ms_saved_per_lm_head_selection": reference_report.statistics.median_ms - candidate_report.statistics.median_ms,
        },
        "limitations": [
            "CUDA-event medians measure device execution and do not include host launch overhead",
            "this is not an end-to-end tok/s result",
            "the runtime still uses the promoted E1.1 LM-head GEMV64 path",
            "end-to-end promotion requires a separate same-session ABBA verification",
            "cross-regime benchmark comparisons are invalid pending environment fingerprint issue #49",
        ],
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|error| {
            NnisError::unsupported(format!("failed to serialize E2 report: {error}"))
        })?
    );
    Ok(())
}
