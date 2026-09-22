//! Fixed tiny graph with analytical weights, not a trained-model benchmark.

use nnis_core::{
    BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableError, PortableQueue, Result,
};
use nnis_cpu::numerical::{CpuF32BinaryOp, CpuF32KernelsV1, CPU_F32_NUMERICAL_POLICY};
use nnis_cpu::{CpuBuffer, CpuDevice};

fn tensor(device: &CpuDevice, values: &[f32]) -> Result<CpuBuffer> {
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    let descriptor = BufferDesc::new(bytes.len() as u64, usage, MemoryClass::Host)?;
    let mut buffer = device.create_buffer(descriptor)?;
    device.create_queue()?.write_buffer(&mut buffer, 0, &bytes)?;
    Ok(buffer)
}

fn check(device: &CpuDevice, buffer: &CpuBuffer, expected: &[f32]) -> Result<()> {
    let actual = device.create_queue()?.read_buffer(buffer, 0, buffer.len() as u64)?;
    let expected: Vec<u8> = expected.iter().flat_map(|value| value.to_le_bytes()).collect();
    if actual != expected {
        return Err(PortableError::Backend("fixed CPU graph output mismatch".to_string()));
    }
    Ok(())
}

fn run() -> Result<()> {
    let device = CpuDevice::with_max_buffer_bytes(256)?;
    let kernels = CpuF32KernelsV1::new(32)?;
    let input = tensor(&device, &[2.0, -1.0, 3.0])?;
    let first_weight = tensor(&device, &[1.0, 2.0, -1.0, 4.0, 0.5, -2.0])?;
    let first_bias = tensor(&device, &[0.5, 1.0])?;
    let second_weight = tensor(&device, &[2.0, -1.0, 4.0, 3.0])?;
    let second_bias = tensor(&device, &[1.0, 2.0])?;
    let mut projected = tensor(&device, &[0.0; 2])?;
    let mut biased = tensor(&device, &[0.0; 2])?;
    let mut hidden = tensor(&device, &[0.0; 2])?;
    let mut logits = tensor(&device, &[0.0; 2])?;
    let mut selected = tensor(&device, &[0.0; 3])?;
    let mut scalar = tensor(&device, &[0.0])?;

    let reports = [
        kernels.project_kn(&input, &first_weight, &mut projected, 3, 2)?,
        kernels.binary(CpuF32BinaryOp::Add, &projected, &first_bias, &mut biased)?,
        kernels.relu(&biased, &mut hidden)?,
        kernels.project_kn(&hidden, &second_weight, &mut projected, 2, 2)?,
        kernels.binary(CpuF32BinaryOp::Add, &projected, &second_bias, &mut logits)?,
        kernels.gather(&logits, &[1, 0, 1], &mut selected)?,
        kernels.sum(&selected, &mut scalar)?,
    ];
    // Exact dyadic arithmetic: hidden=[5,0], logits=[11,-3], sum([-3,11,-3])=5.
    check(&device, &hidden, &[5.0, 0.0])?;
    check(&device, &logits, &[11.0, -3.0])?;
    check(&device, &selected, &[-3.0, 11.0, -3.0])?;
    check(&device, &scalar, &[5.0])?;
    let largest_scratch = reports.iter().map(|report| report.scratch_payload_bytes).max().unwrap_or(0);
    let largest_capacity = reports.iter().map(|report| report.scratch_capacity_bytes).max().unwrap_or(0);
    println!(
        "CPU_F32_GRAPH_OK policy={} operations={} logits=[11,-3] result=5 max_scratch_payload_bytes={} max_scratch_capacity_bytes={}",
        CPU_F32_NUMERICAL_POLICY,
        reports.len(),
        largest_scratch,
        largest_capacity,
    );
    Ok(())
}

fn main() -> Result<()> {
    run()
}

#[cfg(test)]
mod tests {
    #[test]
    fn fixed_graph_has_exact_analytical_output() {
        super::run().unwrap();
    }
}
