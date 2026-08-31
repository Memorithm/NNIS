from pathlib import Path

p = Path("crates/nnis-bench/src/lib.rs")
s = p.read_text()
s = s.replace(
    "use std::collections::BTreeMap;\nuse std::path::Path;",
    "use std::collections::BTreeMap;\nuse std::fs;\nuse std::path::Path;",
    1,
)

marker = """/// Hardware and build identity captured with every result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkMetadata {"""
insert = """/// Environment variable used to bind separate benchmark processes into one
/// explicitly declared measurement campaign.
pub const BENCH_RUN_CONTEXT_ENV: &str = \"NNIS_BENCH_RUN_CONTEXT_ID\";

/// Versioned execution-environment evidence used to decide whether benchmark
/// reports may be compared.
///
/// Collection is best-effort, but comparison is deliberately fail-closed: a
/// cross-report comparison must have an explicit run-context id and every
/// platform-specific field required for the detected platform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkEnvironmentFingerprint {
    pub schema_version: u32,
    pub run_context_id: Option<String>,
    pub environment_label: Option<String>,
    pub host_kernel_release: Option<String>,
    pub platform_model: Option<String>,
    pub cuda_visible_devices: Option<String>,
    pub jetson_power_mode: Option<String>,
    pub jetson_clock_state: Option<String>,
}

impl Default for BenchmarkEnvironmentFingerprint {
    fn default() -> Self {
        Self {
            schema_version: 1,
            run_context_id: None,
            environment_label: None,
            host_kernel_release: None,
            platform_model: None,
            cuda_visible_devices: None,
            jetson_power_mode: None,
            jetson_clock_state: None,
        }
    }
}

impl BenchmarkEnvironmentFingerprint {
    fn collect() -> Self {
        let platform_model = read_trimmed(\"/proc/device-tree/model\");
        let is_jetson = platform_model
            .as_deref()
            .map(|model| model.to_ascii_lowercase().contains(\"jetson\"))
            .unwrap_or(false);
        Self {
            schema_version: 1,
            run_context_id: nonempty_env(BENCH_RUN_CONTEXT_ENV),
            environment_label: nonempty_env(\"NNIS_BENCH_ENVIRONMENT_LABEL\"),
            host_kernel_release: command_text(\"uname\", &[\"-r\"]),
            platform_model,
            cuda_visible_devices: nonempty_env(\"CUDA_VISIBLE_DEVICES\"),
            jetson_power_mode: if is_jetson {
                command_text(\"nvpmodel\", &[\"-q\"])
            } else {
                None
            },
            jetson_clock_state: if is_jetson {
                collect_jetson_clock_state()
            } else {
                None
            },
        }
    }
}

/// Hardware and build identity captured with every result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkMetadata {"""
if marker not in s:
    raise SystemExit("BenchmarkMetadata marker missing")
s = s.replace(marker, insert, 1)

field_marker = """    pub driver_version: Option<String>,
    pub nvrtc_version: Option<String>,
}"""
field_insert = """    pub driver_version: Option<String>,
    pub nvrtc_version: Option<String>,
    #[serde(default)]
    pub environment_fingerprint: BenchmarkEnvironmentFingerprint,
}"""
if field_marker not in s:
    raise SystemExit("metadata fields marker missing")
s = s.replace(field_marker, field_insert, 1)

collect_marker = """            nvrtc_version: nnis_sys::nvrtc::version()
                .map(|(major, minor)| format!(\"{major}.{minor}\")),
        }
    }
}"""
collect_insert = """            nvrtc_version: nnis_sys::nvrtc::version()
                .map(|(major, minor)| format!(\"{major}.{minor}\")),
            environment_fingerprint: BenchmarkEnvironmentFingerprint::collect(),
        }
    }

    /// Reject cross-report comparison unless the execution environment is
    /// sufficiently complete and compatible. Code revisions are intentionally
    /// not compared here: candidate-vs-baseline runs may use different SHAs.
    pub fn require_compatible_environment(&self, other: &Self) -> Result<()> {
        require_equal(\"host_arch\", &self.host_arch, &other.host_arch)?;
        require_equal(\"host_os\", &self.host_os, &other.host_os)?;
        require_equal(\"gpu_ordinal\", &self.gpu_ordinal, &other.gpu_ordinal)?;
        require_equal(\"gpu_name\", &self.gpu_name, &other.gpu_name)?;
        require_equal(
            \"compute_capability_major\",
            &self.compute_capability_major,
            &other.compute_capability_major,
        )?;
        require_equal(
            \"compute_capability_minor\",
            &self.compute_capability_minor,
            &other.compute_capability_minor,
        )?;
        require_equal(
            \"multiprocessor_count\",
            &self.multiprocessor_count,
            &other.multiprocessor_count,
        )?;
        require_present_equal(\"gpu_uuid\", self.gpu_uuid.as_deref(), other.gpu_uuid.as_deref())?;
        require_present_equal(
            \"driver_version\",
            self.driver_version.as_deref(),
            other.driver_version.as_deref(),
        )?;
        require_present_equal(
            \"nvrtc_version\",
            self.nvrtc_version.as_deref(),
            other.nvrtc_version.as_deref(),
        )?;

        let left = &self.environment_fingerprint;
        let right = &other.environment_fingerprint;
        require_equal(
            \"environment_fingerprint.schema_version\",
            &left.schema_version,
            &right.schema_version,
        )?;
        if left.schema_version != 1 {
            return Err(NnisError::unsupported(format!(
                \"unsupported benchmark environment fingerprint schema {}\",
                left.schema_version
            )));
        }
        require_present_equal(
            \"run_context_id\",
            left.run_context_id.as_deref(),
            right.run_context_id.as_deref(),
        )?;
        require_present_equal(
            \"host_kernel_release\",
            left.host_kernel_release.as_deref(),
            right.host_kernel_release.as_deref(),
        )?;
        require_optional_equal(
            \"environment_label\",
            left.environment_label.as_deref(),
            right.environment_label.as_deref(),
        )?;
        require_optional_equal(
            \"platform_model\",
            left.platform_model.as_deref(),
            right.platform_model.as_deref(),
        )?;
        require_optional_equal(
            \"cuda_visible_devices\",
            left.cuda_visible_devices.as_deref(),
            right.cuda_visible_devices.as_deref(),
        )?;

        let is_jetson = left
            .platform_model
            .as_deref()
            .map(|model| model.to_ascii_lowercase().contains(\"jetson\"))
            .unwrap_or(false)
            || right
                .platform_model
                .as_deref()
                .map(|model| model.to_ascii_lowercase().contains(\"jetson\"))
                .unwrap_or(false);
        if is_jetson {
            require_present_equal(
                \"jetson_power_mode\",
                left.jetson_power_mode.as_deref(),
                right.jetson_power_mode.as_deref(),
            )?;
            require_present_equal(
                \"jetson_clock_state\",
                left.jetson_clock_state.as_deref(),
                right.jetson_clock_state.as_deref(),
            )?;
        } else {
            require_optional_equal(
                \"jetson_power_mode\",
                left.jetson_power_mode.as_deref(),
                right.jetson_power_mode.as_deref(),
            )?;
            require_optional_equal(
                \"jetson_clock_state\",
                left.jetson_clock_state.as_deref(),
                right.jetson_clock_state.as_deref(),
            )?;
        }
        Ok(())
    }
}"""
if collect_marker not in s:
    raise SystemExit("metadata collect marker missing")
s = s.replace(collect_marker, collect_insert, 1)

helper_marker = "fn git_identity() -> (String, Option<bool>) {"
helpers = """fn nonempty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn read_trimmed(path: &str) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let value = String::from_utf8_lossy(&bytes)
        .trim_matches(char::from(0))
        .trim()
        .to_string();
    (!value.is_empty()).then_some(value)
}

fn command_text(program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program).args(arguments).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let normalized = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(\"\\n\");
    (!normalized.is_empty()).then_some(normalized)
}

fn collect_jetson_clock_state() -> Option<String> {
    let program = if Path::new(\"/usr/bin/jetson_clocks\").exists() {
        \"/usr/bin/jetson_clocks\"
    } else {
        \"jetson_clocks\"
    };
    let output = command_text(program, &[\"--show\"])?;
    let stable = output
        .lines()
        .map(str::trim)
        .filter(|line| {
            line.starts_with(\"cpu\")
                || line.starts_with(\"gpu-\")
                || line.starts_with(\"EMC \")
                || line.starts_with(\"NV Power Mode:\")
        })
        .collect::<Vec<_>>()
        .join(\"\\n\");
    (!stable.is_empty()).then_some(stable)
}

fn require_equal<T>(name: &str, left: &T, right: &T) -> Result<()>
where
    T: PartialEq + std::fmt::Debug,
{
    if left != right {
        return Err(NnisError::invalid_input(format!(
            \"benchmark environments differ at {name}: left={left:?}, right={right:?}\"
        )));
    }
    Ok(())
}

fn require_present_equal(name: &str, left: Option<&str>, right: Option<&str>) -> Result<()> {
    let left = left.ok_or_else(|| {
        NnisError::invalid_input(format!(
            \"benchmark environment is incomplete: left report is missing {name}\"
        ))
    })?;
    let right = right.ok_or_else(|| {
        NnisError::invalid_input(format!(
            \"benchmark environment is incomplete: right report is missing {name}\"
        ))
    })?;
    if left != right {
        return Err(NnisError::invalid_input(format!(
            \"benchmark environments differ at {name}: left={left:?}, right={right:?}\"
        )));
    }
    Ok(())
}

fn require_optional_equal(name: &str, left: Option<&str>, right: Option<&str>) -> Result<()> {
    if left != right {
        return Err(NnisError::invalid_input(format!(
            \"benchmark environments differ at {name}: left={left:?}, right={right:?}\"
        )));
    }
    Ok(())
}

fn git_identity() -> (String, Option<bool>) {"""
if helper_marker not in s:
    raise SystemExit("git_identity marker missing")
s = s.replace(helper_marker, helpers, 1)

test_marker = """    #[test]
    fn config_rejects_zero_measured_iterations() {"""
tests = """    fn metadata_fixture(run_context_id: Option<&str>) -> BenchmarkMetadata {
        BenchmarkMetadata {
            unix_timestamp_seconds: 1,
            git_commit: \"a\".to_string(),
            git_dirty: Some(false),
            nnis_version: \"0.1.0\".to_string(),
            host_arch: \"aarch64\".to_string(),
            host_os: \"linux\".to_string(),
            gpu_ordinal: 0,
            gpu_name: \"GPU\".to_string(),
            gpu_uuid: Some(\"uuid\".to_string()),
            compute_capability_major: 11,
            compute_capability_minor: 0,
            multiprocessor_count: 20,
            driver_version: Some(\"13.0\".to_string()),
            nvrtc_version: Some(\"13.0\".to_string()),
            environment_fingerprint: BenchmarkEnvironmentFingerprint {
                schema_version: 1,
                run_context_id: run_context_id.map(str::to_string),
                environment_label: Some(\"native-thor\".to_string()),
                host_kernel_release: Some(\"6.11\".to_string()),
                platform_model: Some(\"NVIDIA Jetson AGX Thor Developer Kit\".to_string()),
                cuda_visible_devices: None,
                jetson_power_mode: Some(\"NV Power Mode: MAXN\\n0\".to_string()),
                jetson_clock_state: Some(
                    \"gpu-gpc-0 MinFreq=1 MaxFreq=1 CurrentFreq=1\\nEMC MinFreq=2 MaxFreq=2 CurrentFreq=2\"
                        .to_string(),
                ),
            },
        }
    }

    #[test]
    fn environment_comparison_requires_explicit_run_context() {
        let left = metadata_fixture(None);
        let right = metadata_fixture(None);
        let error = left.require_compatible_environment(&right).unwrap_err();
        assert!(error.to_string().contains(\"run_context_id\"));
    }

    #[test]
    fn environment_comparison_rejects_clock_drift() {
        let left = metadata_fixture(Some(\"campaign-a\"));
        let mut right = metadata_fixture(Some(\"campaign-a\"));
        right.environment_fingerprint.jetson_clock_state = Some(\"different\".to_string());
        let error = left.require_compatible_environment(&right).unwrap_err();
        assert!(error.to_string().contains(\"jetson_clock_state\"));
    }

    #[test]
    fn environment_comparison_accepts_complete_same_campaign() {
        let left = metadata_fixture(Some(\"campaign-a\"));
        let mut right = metadata_fixture(Some(\"campaign-a\"));
        right.git_commit = \"candidate-b\".to_string();
        left.require_compatible_environment(&right).unwrap();
    }

    #[test]
    fn config_rejects_zero_measured_iterations() {"""
if test_marker not in s:
    raise SystemExit("test insertion marker missing")
s = s.replace(test_marker, tests, 1)
p.write_text(s)

Path("crates/nnis-bench/examples/compare_smollm2_reports.rs").write_text('''use nnis_bench::{BenchmarkMetadata, TimingStatistics};
use serde::Deserialize;
use serde_json::json;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, PartialEq)]
struct TimingReport {
    statistics: TimingStatistics,
}

#[derive(Debug, Deserialize)]
struct ComparableReport {
    schema_version: u32,
    benchmark: String,
    backend: String,
    measurement: String,
    source_repo: String,
    source_revision: String,
    source_model_sha256: String,
    execution_weight_dtype: String,
    input_ids: Vec<u32>,
    decode_steps: usize,
    warmup_iterations: usize,
    iterations: usize,
    metadata: BenchmarkMetadata,
    model: serde_json::Value,
    generation: TimingReport,
    request_total: TimingReport,
    generated_ids: Vec<u32>,
}

fn read_report(path: &Path) -> Result<ComparableReport, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid report {}: {error}", path.display()))
}

fn require_same<T>(name: &str, left: &T, right: &T) -> Result<(), String>
where
    T: PartialEq + std::fmt::Debug,
{
    if left != right {
        return Err(format!(
            "reports are not workload-compatible at {name}: left={left:?}, right={right:?}"
        ));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let left_path = args
        .next()
        .ok_or("usage: compare_smollm2_reports LEFT.json RIGHT.json")?;
    let right_path = args
        .next()
        .ok_or("usage: compare_smollm2_reports LEFT.json RIGHT.json")?;
    if args.next().is_some() {
        return Err("usage: compare_smollm2_reports LEFT.json RIGHT.json".to_string());
    }
    let left = read_report(Path::new(&left_path))?;
    let right = read_report(Path::new(&right_path))?;

    require_same("schema_version", &left.schema_version, &right.schema_version)?;
    if left.schema_version != 2 {
        return Err(format!(
            "unsupported SmolLM2 report schema {}; expected 2",
            left.schema_version
        ));
    }
    require_same("benchmark", &left.benchmark, &right.benchmark)?;
    require_same("backend", &left.backend, &right.backend)?;
    require_same("measurement", &left.measurement, &right.measurement)?;
    require_same("source_repo", &left.source_repo, &right.source_repo)?;
    require_same("source_revision", &left.source_revision, &right.source_revision)?;
    require_same(
        "source_model_sha256",
        &left.source_model_sha256,
        &right.source_model_sha256,
    )?;
    require_same(
        "execution_weight_dtype",
        &left.execution_weight_dtype,
        &right.execution_weight_dtype,
    )?;
    require_same("input_ids", &left.input_ids, &right.input_ids)?;
    require_same("decode_steps", &left.decode_steps, &right.decode_steps)?;
    require_same(
        "warmup_iterations",
        &left.warmup_iterations,
        &right.warmup_iterations,
    )?;
    require_same("iterations", &left.iterations, &right.iterations)?;
    require_same("model", &left.model, &right.model)?;
    require_same("generated_ids", &left.generated_ids, &right.generated_ids)?;
    left.metadata
        .require_compatible_environment(&right.metadata)
        .map_err(|error| error.to_string())?;

    let left_generation = left.generation.statistics.median_ms;
    let right_generation = right.generation.statistics.median_ms;
    let left_request = left.request_total.statistics.median_ms;
    let right_request = right.request_total.statistics.median_ms;
    if left_generation <= 0.0
        || right_generation <= 0.0
        || left_request <= 0.0
        || right_request <= 0.0
    {
        return Err("report medians must be positive".to_string());
    }

    let report = json!({
        "schema_version": 1,
        "comparable": true,
        "run_context_id": left.metadata.environment_fingerprint.run_context_id,
        "left_git_commit": left.metadata.git_commit,
        "right_git_commit": right.metadata.git_commit,
        "generation": {
            "left_median_ms": left_generation,
            "right_median_ms": right_generation,
            "right_over_left_latency_ratio": right_generation / left_generation,
            "right_over_left_throughput_ratio": left_generation / right_generation,
        },
        "request_total": {
            "left_median_ms": left_request,
            "right_median_ms": right_request,
            "right_over_left_latency_ratio": right_request / left_request,
            "right_over_left_throughput_ratio": left_request / right_request,
        },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize comparison: {error}"))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("SmolLM2 report comparison rejected: {error}");
        std::process::exit(1);
    }
}
''')
