use nnis_bench::{benchmark_gpu, BenchConfig, BenchmarkCase, BenchmarkReport};
use nnis_jit::{JitCompiler, OccupancyRecommendation};
use nnis_kernels::F32Elementwise;
use nnis_rt::{Context, Device, DeviceBuffer, NnisError, Stream};
use serde::Serialize;

const DEFAULT_BLOCK_SIZES: &[u32] = &[128, 256, 512, 768, 1024];

#[derive(Debug, Serialize)]
struct ScaleOccupancy {
    recommended_block_size: u32,
    minimum_grid_size: u32,
    active_blocks_per_multiprocessor_at_recommendation: u32,
}

impl From<OccupancyRecommendation> for ScaleOccupancy {
    fn from(value: OccupancyRecommendation) -> Self {
        Self {
            recommended_block_size: value.block_size,
            minimum_grid_size: value.minimum_grid_size,
            active_blocks_per_multiprocessor_at_recommendation: value
                .active_blocks_per_multiprocessor,
        }
    }
}

#[derive(Debug, Serialize)]
struct BlockSizeSweep {
    schema_version: u32,
    occupancy: ScaleOccupancy,
    reports: Vec<BenchmarkReport>,
    correctness_validated: bool,
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    match std::env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn block_sizes() -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let Ok(value) = std::env::var("NNIS_BENCH_BLOCK_SIZES") else {
        return Ok(DEFAULT_BLOCK_SIZES.to_vec());
    };
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(str::parse)
        .collect::<Result<Vec<u32>, _>>()?;
    if values.is_empty() {
        return Err("NNIS_BENCH_BLOCK_SIZES must contain at least one width".into());
    }
    Ok(values)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let elements = env_usize("NNIS_BENCH_ELEMENTS", 1 << 24)?;
    let warmups = env_usize("NNIS_BENCH_WARMUPS", 20)?;
    let iterations = env_usize("NNIS_BENCH_ITERATIONS", 100)?;
    let block_sizes = block_sizes()?;
    if elements == 0 {
        return Err("NNIS_BENCH_ELEMENTS must be non-zero".into());
    }

    let device = Device::first()?;
    let context = Context::new(&device)?;
    let stream = Stream::new(&context)?;
    let compiler = JitCompiler::new();
    let host = (0..elements)
        .map(|index| (index % 4_093) as f32 * 0.000_25 - 0.5)
        .collect::<Vec<_>>();
    let input = DeviceBuffer::from_host(&context, &stream, &host)?;
    let output = DeviceBuffer::<f32>::new(&context, elements)?;
    let scale = -0.75_f32;
    let bytes = (elements as u64)
        .checked_mul(2 * std::mem::size_of::<f32>() as u64)
        .ok_or("benchmark byte count overflow")?;
    let mut reports = Vec::with_capacity(block_sizes.len());
    let mut expected_occupancy = None;

    for block_size in block_sizes {
        let kernels = F32Elementwise::load_with_block_size(&context, &compiler, block_size)?;
        let occupancy = kernels.occupancy()?.scale;
        if let Some(expected) = expected_occupancy {
            if occupancy != expected {
                return Err(NnisError::unsupported(
                    "occupancy recommendation changed across identical module loads",
                )
                .into());
            }
        } else {
            expected_occupancy = Some(occupancy);
        }
        let active_blocks = kernels.active_blocks_per_multiprocessor()?.scale;
        let case = BenchmarkCase::new("nnis_scale_f32_block_sweep", "f32")
            .with_dimension("elements", elements as u64)
            .with_dimension("block_size", u64::from(block_size))
            .with_dimension("active_blocks_per_multiprocessor", u64::from(active_blocks))
            .with_work_items(elements as u64)
            .with_bytes_per_iteration(bytes);
        let report = benchmark_gpu(
            &context,
            &stream,
            case,
            BenchConfig::new(warmups, iterations),
            || {
                // SAFETY: all captured objects outlive this harness, which
                // waits for the end event after every measured launch.
                unsafe { kernels.enqueue_scale(&stream, &input, &output, scale) }
            },
        )?;

        let actual = output.to_vec(&stream)?;
        for (index, (&actual, &input)) in actual.iter().zip(&host).enumerate() {
            if actual != input * scale {
                return Err(
                    format!("block size {block_size} result mismatch at element {index}").into(),
                );
            }
        }
        reports.push(report);
    }

    let result = BlockSizeSweep {
        schema_version: 1,
        occupancy: expected_occupancy
            .expect("at least one block size was validated")
            .into(),
        reports,
        correctness_validated: true,
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
