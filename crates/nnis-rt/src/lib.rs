//! `nnis-rt`: safe NVIDIA driver-runtime layer for NNIS.
//!
//! Provides device discovery, primary-context ownership, streams, events and
//! memory management on top of the raw FFI in [`nnis_sys`]. All CUDA state is
//! context-routed; see [`context::Context`] for the threading model.

pub mod bf16;
pub mod context;
pub mod device;
pub mod error;
pub mod memory;
pub mod pool;
pub mod stream_event;

pub use bf16::{bf16_bits_to_f32, f32_to_bf16_rne};
pub use context::{gpu_context, Context};
pub use device::{Device, DeviceProps};
pub use error::{ErrorKind, NnisError, Result};
pub use memory::{DeviceBuffer, DevicePod, PinnedBuffer};
pub use pool::{PooledBuffer, StreamOrderedAllocator};
pub use stream_event::{Event, Stream};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_enumeration_matches_count() {
        let count = match Device::count() {
            Ok(n) => n,
            Err(e) => {
                // Distinguish "no driver" from real failures; without a GPU
                // enumeration must fail with an explicit unsupported/library
                // error, not a panic.
                assert!(
                    matches!(
                        e.kind(),
                        ErrorKind::Library(_)
                            | ErrorKind::Driver { .. }
                            | ErrorKind::Unsupported(_)
                    ),
                    "unexpected enumeration error: {e}"
                );
                eprintln!("skipped: no CUDA driver/device ({e})");
                return;
            }
        };
        let devs = Device::enumerate().expect("enumeration should succeed when count > 0");
        assert_eq!(devs.len(), count);
        for d in &devs {
            let p = d.props().unwrap();
            assert!(!p.name.is_empty());
            assert!(p.multiprocessor_count > 0);
            assert!(p.warp_size >= 1);
            assert_eq!(p.ordinal, d.ordinal());
            println!("device {}: {}", d.ordinal(), p.fingerprint());
        }
    }

    #[test]
    fn gpu_context_and_mem_info() {
        let Some(ctx) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let (free, total) = ctx.mem_info().unwrap();
        assert!(total > 0);
        assert!(free <= total);
        println!(
            "mem: {:.1} GiB free / {:.1} GiB total",
            free as f64 / (1 << 30) as f64,
            total as f64 / (1 << 30) as f64
        );
    }
}
