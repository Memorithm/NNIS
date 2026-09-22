//! Executable memory smoke, not a model or adaptive-policy benchmark.

use nnis_core::{
    BufferDesc, BufferUsages, PortableDevice, PortableError, PortableFence, PortableQueue,
    Result, MemoryClass,
};
use nnis_cpu::CpuDevice;

fn require(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(PortableError::Backend(message.to_string()));
    }
    Ok(())
}

fn main() -> Result<()> {
    let device = CpuDevice::with_max_buffer_bytes(64)?;
    let queue = device.create_queue()?;
    let usage = BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let descriptor = BufferDesc::new(16, usage, MemoryClass::Host)?;
    let mut current = device.create_buffer(descriptor)?;
    let mut checkpoint = device.create_buffer(descriptor)?;
    let mut candidate = device.create_buffer(descriptor)?;

    queue.write_buffer(&mut current, 0, &[11; 16])?.wait()?;
    queue.copy_buffer(&current, 0, &mut checkpoint, 0, 16)?.wait()?;
    queue.write_buffer(&mut candidate, 0, &[22; 16])?.wait()?;
    queue.copy_buffer(&candidate, 0, &mut current, 0, 16)?.wait()?;
    require(queue.read_buffer(&current, 0, 16)? == [22; 16], "copy mismatch")?;

    let rejected = queue.copy_buffer(&candidate, 0, &mut current, 15, 2);
    require(
        matches!(rejected, Err(PortableError::OutOfBounds { .. })),
        "out-of-bounds copy was not rejected",
    )?;
    require(
        queue.read_buffer(&current, 0, 16)? == [22; 16],
        "rejected copy changed the destination",
    )?;

    queue.copy_buffer(&checkpoint, 0, &mut current, 0, 16)?.wait()?;
    require(
        queue.read_buffer(&current, 0, 16)? == [11; 16],
        "explicit checkpoint restoration failed",
    )?;
    let oversized = BufferDesc::new(65, usage, MemoryClass::Host)?;
    require(
        matches!(device.create_buffer(oversized), Err(PortableError::Unsupported(_))),
        "per-buffer admission limit was not enforced",
    )?;
    println!(
        "CPU_BUFFER_SMOKE_OK backend={} payload_bytes={} capacity_bytes={} restore_ok=true",
        device.backend_id().name(),
        current.len(),
        current.capacity_bytes(),
    );
    Ok(())
}
