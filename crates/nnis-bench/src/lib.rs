//! CUDA-event benchmark harness and reproducible NNIS benchmark reports.
//!
//! Timing is performed on the GPU timeline. Host wall-clock duration is not
//! used as a substitute for asynchronous kernel execution time.

use nnis_rt::{Context, Event, NnisError, Result, Stream};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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
        }
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
/// before reading its elapsed time.
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
        enqueue()?;
    }
    stream.synchronize()?;

    let start = Event::new(context)?;
    let end = Event::new(context)?;
    let mut samples_ms = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        start.record(stream)?;
        enqueue()?;
        end.record(stream)?;
        end.synchronize()?;
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
            .with_bytes_per_iteration((elements * 2 * size_of::<f32>()) as u64);

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
        println!("{}", report.to_json_pretty().unwrap());
    }
}
