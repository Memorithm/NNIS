//! CUDA-event benchmark harness and reproducible NNIS benchmark reports.
//!
//! Timing is performed on the GPU timeline. Host wall-clock duration is not
//! used as a substitute for asynchronous kernel execution time.

use nnis_rt::{Context, Event, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Warmup and measured iteration counts for one benchmark run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchConfig {
    pub warmup_iterations: usize,
    pub iterations: usize,
}

impl BenchConfig {
    pub const fn new(warmup_iterations: usize, iterations: usize) -> Self {
        Self {
            warmup_iterations,
            iterations,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.iterations == 0 {
            return Err(NnisError::invalid_input(
                "benchmark requires at least one measured iteration",
            ));
        }
        Ok(())
    }
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self::new(10, 100)
    }
}

/// Stable description of the operation being measured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkCase {
    pub name: String,
    pub dtype: String,
    pub dimensions: BTreeMap<String, u64>,
    pub work_items: Option<u64>,
    pub bytes_per_iteration: Option<u64>,
}

impl BenchmarkCase {
    pub fn new(name: impl Into<String>, dtype: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dtype: dtype.into(),
            dimensions: BTreeMap::new(),
            work_items: None,
            bytes_per_iteration: None,
        }
    }

    pub fn with_dimension(mut self, name: impl Into<String>, value: u64) -> Self {
        self.dimensions.insert(name.into(), value);
        self
    }

    pub const fn with_work_items(mut self, work_items: u64) -> Self {
        self.work_items = Some(work_items);
        self
    }

    pub const fn with_bytes_per_iteration(mut self, bytes: u64) -> Self {
        self.bytes_per_iteration = Some(bytes);
        self
    }
}

/// Environment variable used to bind separate benchmark processes into one
/// explicitly declared measurement campaign.
pub const BENCH_RUN_CONTEXT_ENV: &str = "NNIS_BENCH_RUN_CONTEXT_ID";

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
        let platform_model = read_trimmed("/proc/device-tree/model");
        let is_jetson = platform_model
            .as_deref()
            .map(|model| model.to_ascii_lowercase().contains("jetson"))
            .unwrap_or(false);
        Self {
            schema_version: 1,
            run_context_id: nonempty_env(BENCH_RUN_CONTEXT_ENV),
            environment_label: nonempty_env("NNIS_BENCH_ENVIRONMENT_LABEL"),
            host_kernel_release: command_text("uname", &["-r"]),
            platform_model,
            cuda_visible_devices: nonempty_env("CUDA_VISIBLE_DEVICES"),
            jetson_power_mode: if is_jetson {
                command_text("nvpmodel", &["-q"])
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
pub struct BenchmarkMetadata {
    pub unix_timestamp_seconds: u64,
    pub git_commit: String,
    pub git_dirty: Option<bool>,
    pub nnis_version: String,
    pub host_arch: String,
    pub host_os: String,
    pub gpu_ordinal: i32,
    pub gpu_name: String,
    pub gpu_uuid: Option<String>,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub multiprocessor_count: u32,
    pub driver_version: Option<String>,
    pub nvrtc_version: Option<String>,
    #[serde(default)]
    pub environment_fingerprint: BenchmarkEnvironmentFingerprint,
}

impl BenchmarkMetadata {
    pub fn collect(context: &Context) -> Self {
        let properties = context.props();
        let (git_commit, git_dirty) = git_identity();
        Self {
            unix_timestamp_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            git_commit,
            git_dirty,
            nnis_version: env!("CARGO_PKG_VERSION").to_string(),
            host_arch: std::env::consts::ARCH.to_string(),
            host_os: std::env::consts::OS.to_string(),
            gpu_ordinal: context.device_ordinal(),
            gpu_name: properties.name.clone(),
            gpu_uuid: properties.uuid.map(|uuid| format!("{uuid:?}")),
            compute_capability_major: properties.compute_capability.0,
            compute_capability_minor: properties.compute_capability.1,
            multiprocessor_count: properties.multiprocessor_count,
            driver_version: nnis_sys::driver::driver_version()
                .map(|(major, minor)| format!("{major}.{minor}")),
            nvrtc_version: nnis_sys::nvrtc::version()
                .map(|(major, minor)| format!("{major}.{minor}")),
            environment_fingerprint: BenchmarkEnvironmentFingerprint::collect(),
        }
    }

    /// Reject cross-report comparison unless the execution environment is
    /// sufficiently complete and compatible. Code revisions are intentionally
    /// not compared here: candidate-vs-baseline runs may use different SHAs.
    pub fn require_compatible_environment(&self, other: &Self) -> Result<()> {
        require_equal("host_arch", &self.host_arch, &other.host_arch)?;
        require_equal("host_os", &self.host_os, &other.host_os)?;
        require_equal("gpu_ordinal", &self.gpu_ordinal, &other.gpu_ordinal)?;
        require_equal("gpu_name", &self.gpu_name, &other.gpu_name)?;
        require_equal(
            "compute_capability_major",
            &self.compute_capability_major,
            &other.compute_capability_major,
        )?;
        require_equal(
            "compute_capability_minor",
            &self.compute_capability_minor,
            &other.compute_capability_minor,
        )?;
        require_equal(
            "multiprocessor_count",
            &self.multiprocessor_count,
            &other.multiprocessor_count,
        )?;
        require_present_equal(
            "gpu_uuid",
            self.gpu_uuid.as_deref(),
            other.gpu_uuid.as_deref(),
        )?;
        require_present_equal(
            "driver_version",
            self.driver_version.as_deref(),
            other.driver_version.as_deref(),
        )?;
        require_present_equal(
            "nvrtc_version",
            self.nvrtc_version.as_deref(),
            other.nvrtc_version.as_deref(),
        )?;

        let left = &self.environment_fingerprint;
        let right = &other.environment_fingerprint;
        require_equal(
            "environment_fingerprint.schema_version",
            &left.schema_version,
            &right.schema_version,
        )?;
        if left.schema_version != 1 {
            return Err(NnisError::unsupported(format!(
                "unsupported benchmark environment fingerprint schema {}",
                left.schema_version
            )));
        }
        require_present_equal(
            "run_context_id",
            left.run_context_id.as_deref(),
            right.run_context_id.as_deref(),
        )?;
        require_present_equal(
            "host_kernel_release",
            left.host_kernel_release.as_deref(),
            right.host_kernel_release.as_deref(),
        )?;
        require_optional_equal(
            "environment_label",
            left.environment_label.as_deref(),
            right.environment_label.as_deref(),
        )?;
        require_optional_equal(
            "platform_model",
            left.platform_model.as_deref(),
            right.platform_model.as_deref(),
        )?;
        require_optional_equal(
            "cuda_visible_devices",
            left.cuda_visible_devices.as_deref(),
            right.cuda_visible_devices.as_deref(),
        )?;

        let is_jetson = left
            .platform_model
            .as_deref()
            .map(|model| model.to_ascii_lowercase().contains("jetson"))
            .unwrap_or(false)
            || right
                .platform_model
                .as_deref()
                .map(|model| model.to_ascii_lowercase().contains("jetson"))
                .unwrap_or(false);
        if is_jetson {
            require_present_equal(
                "jetson_power_mode",
                left.jetson_power_mode.as_deref(),
                right.jetson_power_mode.as_deref(),
            )?;
            require_present_equal(
                "jetson_clock_state",
                left.jetson_clock_state.as_deref(),
                right.jetson_clock_state.as_deref(),
            )?;
        } else {
            require_optional_equal(
                "jetson_power_mode",
                left.jetson_power_mode.as_deref(),
                right.jetson_power_mode.as_deref(),
            )?;
            require_optional_equal(
                "jetson_clock_state",
                left.jetson_clock_state.as_deref(),
                right.jetson_clock_state.as_deref(),
            )?;
        }
        Ok(())
    }
}

/// Distribution summary over per-iteration CUDA-event durations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingStatistics {
    pub min_ms: f64,
    pub median_ms: f64,
    pub mean_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub stddev_ms: f64,
}

/// Median throughput derived from the operation descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Throughput {
    pub items_per_second: Option<f64>,
    /// Decimal GB/s (`10^9` bytes per second).
    pub gigabytes_per_second: Option<f64>,
}

/// Serializable result of one GPU benchmark case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub case: BenchmarkCase,
    pub config: BenchConfig,
    pub metadata: BenchmarkMetadata,
    pub statistics: TimingStatistics,
    pub throughput: Option<Throughput>,
    pub samples_ms: Vec<f64>,
}

impl BenchmarkReport {
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    pub fn to_json_pretty(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

/// Measure one asynchronous GPU operation with CUDA events.
///
/// `enqueue` must submit the complete operation to `stream` and return without
/// synchronizing it. The harness performs warmups, brackets each measured
/// invocation with events on that same stream, and synchronizes the end event
/// before reading its elapsed time. If submission or event handling fails
/// after work may have been queued, the harness drains the stream before
/// returning so captured asynchronous borrows can unwind safely.
pub fn benchmark_gpu<F>(
    context: &Arc<Context>,
    stream: &Stream,
    case: BenchmarkCase,
    config: BenchConfig,
    mut enqueue: F,
) -> Result<BenchmarkReport>
where
    F: FnMut() -> Result<()>,
{
    config.validate()?;
    if !Arc::ptr_eq(context, stream.ctx()) {
        return Err(NnisError::invalid_input(
            "benchmark context and stream do not match",
        ));
    }

    // Establish a clean stream boundary before warmup and measurement.
    stream.synchronize()?;
    for _ in 0..config.warmup_iterations {
        if let Err(error) = enqueue() {
            let _ = stream.synchronize();
            return Err(error);
        }
    }
    stream.synchronize()?;

    let start = Event::new(context)?;
    let end = Event::new(context)?;
    let mut samples_ms = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        start.record(stream)?;
        if let Err(error) = enqueue() {
            let _ = stream.synchronize();
            return Err(error);
        }
        if let Err(error) = end.record(stream) {
            let _ = stream.synchronize();
            return Err(error);
        }
        if let Err(error) = end.synchronize() {
            let _ = stream.synchronize();
            return Err(error);
        }
        let elapsed_ms = end.elapsed_ms(&start)?;
        if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
            return Err(NnisError::unsupported(format!(
                "CUDA events produced invalid duration {elapsed_ms} ms"
            )));
        }
        samples_ms.push(elapsed_ms);
    }

    let statistics = summarize_samples_ms(&samples_ms)?;
    let throughput = throughput(&case, statistics.median_ms);
    Ok(BenchmarkReport {
        schema_version: 1,
        case,
        config,
        metadata: BenchmarkMetadata::collect(context),
        statistics,
        throughput,
        samples_ms,
    })
}

/// Summarize already-collected millisecond samples with the same distribution
/// convention used by [`benchmark_gpu`].
pub fn summarize_samples_ms(samples: &[f64]) -> Result<TimingStatistics> {
    if samples.is_empty() {
        return Err(NnisError::invalid_input(
            "cannot summarize an empty timing sample",
        ));
    }
    if samples
        .iter()
        .any(|sample| !sample.is_finite() || *sample < 0.0)
    {
        return Err(NnisError::invalid_input(
            "timing samples must be finite and non-negative",
        ));
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let count = sorted.len() as f64;
    let mean_ms = sorted.iter().sum::<f64>() / count;
    let variance = sorted
        .iter()
        .map(|sample| (sample - mean_ms).powi(2))
        .sum::<f64>()
        / count;
    Ok(TimingStatistics {
        min_ms: sorted[0],
        median_ms: percentile(&sorted, 0.5),
        mean_ms,
        p95_ms: percentile(&sorted, 0.95),
        p99_ms: percentile(&sorted, 0.99),
        max_ms: sorted[sorted.len() - 1],
        stddev_ms: variance.sqrt(),
    })
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=1.0).contains(&quantile));
    let position = (sorted.len() - 1) as f64 * quantile;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
    }
}

fn throughput(case: &BenchmarkCase, median_ms: f64) -> Option<Throughput> {
    if median_ms <= 0.0 {
        return None;
    }
    let seconds = median_ms / 1_000.0;
    match (case.work_items, case.bytes_per_iteration) {
        (None, None) => None,
        (items, bytes) => Some(Throughput {
            items_per_second: items.map(|items| items as f64 / seconds),
            gigabytes_per_second: bytes.map(|bytes| bytes as f64 / seconds / 1.0e9),
        }),
    }
}

fn nonempty_env(name: &str) -> Option<String> {
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
        .join("\n");
    (!normalized.is_empty()).then_some(normalized)
}

fn collect_jetson_clock_state() -> Option<String> {
    let program = if Path::new("/usr/bin/jetson_clocks").exists() {
        "/usr/bin/jetson_clocks"
    } else {
        "jetson_clocks"
    };
    let output = command_text(program, &["--show"])?;
    let stable = output
        .lines()
        .map(str::trim)
        .filter(|line| {
            line.starts_with("cpu")
                || line.starts_with("gpu-")
                || line.starts_with("EMC ")
                || line.starts_with("NV Power Mode:")
        })
        .collect::<Vec<_>>()
        .join("\n");
    (!stable.is_empty()).then_some(stable)
}

fn require_equal<T>(name: &str, left: &T, right: &T) -> Result<()>
where
    T: PartialEq + std::fmt::Debug,
{
    if left != right {
        return Err(NnisError::invalid_input(format!(
            "benchmark environments differ at {name}: left={left:?}, right={right:?}"
        )));
    }
    Ok(())
}

fn require_present_equal(name: &str, left: Option<&str>, right: Option<&str>) -> Result<()> {
    let left = left.ok_or_else(|| {
        NnisError::invalid_input(format!(
            "benchmark environment is incomplete: left report is missing {name}"
        ))
    })?;
    let right = right.ok_or_else(|| {
        NnisError::invalid_input(format!(
            "benchmark environment is incomplete: right report is missing {name}"
        ))
    })?;
    if left != right {
        return Err(NnisError::invalid_input(format!(
            "benchmark environments differ at {name}: left={left:?}, right={right:?}"
        )));
    }
    Ok(())
}

fn require_optional_equal(name: &str, left: Option<&str>, right: Option<&str>) -> Result<()> {
    if left != right {
        return Err(NnisError::invalid_input(format!(
            "benchmark environments differ at {name}: left={left:?}, right={right:?}"
        )));
    }
    Ok(())
}

fn git_identity() -> (String, Option<bool>) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let commit = run_git(root, &["rev-parse", "HEAD"]).unwrap_or_else(|| {
        option_env!("NNIS_GIT_COMMIT")
            .unwrap_or("unknown")
            .to_string()
    });
    let dirty = run_git(root, &["status", "--porcelain", "--untracked-files=no"])
        .map(|status| !status.is_empty());
    (commit, dirty)
}

fn run_git(root: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(arguments)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_jit::JitCompiler;
    use nnis_kernels::F32Elementwise;
    use nnis_rt::{gpu_context, DeviceBuffer};

    #[test]
    fn statistics_use_interpolated_percentiles() {
        let statistics = summarize_samples_ms(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(statistics.min_ms, 1.0);
        assert_eq!(statistics.median_ms, 2.5);
        assert_eq!(statistics.mean_ms, 2.5);
        assert!((statistics.p95_ms - 3.85).abs() < 1.0e-12);
        assert!((statistics.p99_ms - 3.97).abs() < 1.0e-12);
        assert_eq!(statistics.max_ms, 4.0);
    }

    fn metadata_fixture(run_context_id: Option<&str>) -> BenchmarkMetadata {
        BenchmarkMetadata {
            unix_timestamp_seconds: 1,
            git_commit: "a".to_string(),
            git_dirty: Some(false),
            nnis_version: "0.1.0".to_string(),
            host_arch: "aarch64".to_string(),
            host_os: "linux".to_string(),
            gpu_ordinal: 0,
            gpu_name: "GPU".to_string(),
            gpu_uuid: Some("uuid".to_string()),
            compute_capability_major: 11,
            compute_capability_minor: 0,
            multiprocessor_count: 20,
            driver_version: Some("13.0".to_string()),
            nvrtc_version: Some("13.0".to_string()),
            environment_fingerprint: BenchmarkEnvironmentFingerprint {
                schema_version: 1,
                run_context_id: run_context_id.map(str::to_string),
                environment_label: Some("native-thor".to_string()),
                host_kernel_release: Some("6.11".to_string()),
                platform_model: Some("NVIDIA Jetson AGX Thor Developer Kit".to_string()),
                cuda_visible_devices: None,
                jetson_power_mode: Some("NV Power Mode: MAXN\n0".to_string()),
                jetson_clock_state: Some(
                    "gpu-gpc-0 MinFreq=1 MaxFreq=1 CurrentFreq=1\nEMC MinFreq=2 MaxFreq=2 CurrentFreq=2"
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
        assert!(error.to_string().contains("run_context_id"));
    }

    #[test]
    fn environment_comparison_rejects_clock_drift() {
        let left = metadata_fixture(Some("campaign-a"));
        let mut right = metadata_fixture(Some("campaign-a"));
        right.environment_fingerprint.jetson_clock_state = Some("different".to_string());
        let error = left.require_compatible_environment(&right).unwrap_err();
        assert!(error.to_string().contains("jetson_clock_state"));
    }

    #[test]
    fn environment_comparison_accepts_complete_same_campaign() {
        let left = metadata_fixture(Some("campaign-a"));
        let mut right = metadata_fixture(Some("campaign-a"));
        right.git_commit = "candidate-b".to_string();
        left.require_compatible_environment(&right).unwrap();
    }

    #[test]
    fn config_rejects_zero_measured_iterations() {
        let error = BenchConfig::new(10, 0).validate().unwrap_err();
        assert!(error.to_string().contains("at least one"));
    }

    #[test]
    fn gpu_events_measure_real_elementwise_kernel() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let elements = 1usize << 20;
        let compiler = JitCompiler::new();
        let kernels = F32Elementwise::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let host = (0..elements)
            .map(|index| index as f32 * 0.000_125 - 3.0)
            .collect::<Vec<_>>();
        let input = DeviceBuffer::from_host(&context, &stream, &host).unwrap();
        let output = DeviceBuffer::<f32>::new(&context, elements).unwrap();
        let scale = -0.75_f32;
        let case = BenchmarkCase::new("nnis_scale_f32", "f32")
            .with_dimension("elements", elements as u64)
            .with_work_items(elements as u64)
            .with_bytes_per_iteration((elements * 2 * core::mem::size_of::<f32>()) as u64);

        let report = benchmark_gpu(&context, &stream, case, BenchConfig::new(3, 9), || {
            // SAFETY: the harness synchronizes every measured launch and
            // all captured objects outlive the complete benchmark.
            unsafe { kernels.enqueue_scale(&stream, &input, &output, scale) }
        })
        .unwrap();

        assert_eq!(report.samples_ms.len(), 9);
        assert!(report.statistics.median_ms > 0.0);
        assert!(report.throughput.is_some());
        assert_eq!(report.metadata.gpu_name, context.props().name);
        let json = report.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["case"]["dtype"], "f32");

        let actual = output.to_vec(&stream).unwrap();
        for (index, (&actual, &input)) in actual.iter().zip(&host).enumerate() {
            assert_eq!(actual, input * scale, "mismatch at element {index}");
        }

        let mut calls = 0;
        let failure_case = BenchmarkCase::new("synthetic_enqueue_failure", "f32");
        let error = benchmark_gpu(
            &context,
            &stream,
            failure_case,
            BenchConfig::new(1, 1),
            || {
                // SAFETY: the harness must retain and drain this submission
                // even when the closure reports an error immediately after it.
                unsafe { kernels.enqueue_scale(&stream, &input, &output, scale)? };
                calls += 1;
                if calls == 2 {
                    Err(NnisError::invalid_input("synthetic enqueue failure"))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("synthetic enqueue failure"));
        assert!(stream.query().unwrap(), "error path must drain the stream");
        println!("{}", report.to_json_pretty().unwrap());
    }
}
