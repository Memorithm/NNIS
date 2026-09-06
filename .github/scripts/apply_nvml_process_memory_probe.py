from pathlib import Path

sys_lib = Path('crates/nnis-sys/src/lib.rs')
text = sys_lib.read_text()
text = text.replace(
    '//! Raw, dynamically-loaded FFI surface for the CUDA driver API and NVRTC.\n',
    '//! Raw, dynamically-loaded FFI surface for the CUDA driver API, NVRTC and optional NVML.\n',
    1,
)
text = text.replace(
    '//! * No link-time dependency on `libcuda` / `libnvrtc`: both libraries are\n//!   resolved at runtime via `dlopen`, so NNIS builds on machines without a\n//!   CUDA toolkit and degrades to a typed "unsupported" state instead of a\n//!   link error.\n',
    '//! * No link-time dependency on `libcuda`, `libnvrtc` or optional `libnvidia-ml`:\n//!   native libraries are resolved at runtime via `dlopen`, so NNIS builds on\n//!   machines without the management library and preserves a capability-negative\n//!   state instead of a link error.\n',
    1,
)
if 'pub mod nvml;' not in text:
    text = text.replace('pub mod driver;\npub mod nvrtc;\n', 'pub mod driver;\npub mod nvml;\npub mod nvrtc;\n', 1)
sys_lib.write_text(text)

rt_lib = Path('crates/nnis-rt/src/lib.rs')
text = rt_lib.read_text()
if 'pub mod process_memory;' not in text:
    text = text.replace('pub mod pool;\n', 'pub mod pool;\npub mod process_memory;\n', 1)
export_anchor = 'pub use pool::{PooledBuffer, StreamOrderedAllocator};\n'
export = '''pub use process_memory::{
    ProcessGpuMemoryProbeV1, ProcessGpuMemorySnapshotV1, ProcessGpuMemorySourceV1,
    ProcessGpuMemoryUnavailableReasonV1, ProcessGpuMemoryUnavailableV1,
    NNIS_PROCESS_GPU_MEMORY_PROBE_VERSION,
};
'''
if 'NNIS_PROCESS_GPU_MEMORY_PROBE_VERSION' not in text:
    if export_anchor not in text:
        raise SystemExit('nnis-rt export anchor missing')
    text = text.replace(export_anchor, export_anchor + export, 1)
rt_lib.write_text(text)

context = Path('crates/nnis-rt/src/context.rs')
text = context.read_text()
anchor = '''    pub fn mem_info(&self) -> Result<(u64, u64)> {
        self.set_current()?;
        let api = driver::api()?;
        let (mut free, mut total): (usize, usize) = (0, 0);
        // SAFETY: out-pointers valid; context is current.
        let rc = unsafe { (api.cuMemGetInfo)(&mut free, &mut total) };
        if rc != 0 {
            return Err(NnisError::driver("cuMemGetInfo", rc));
        }
        Ok((free as u64, total as u64))
    }
'''
method = anchor + '''
    /// Probe NVML for memory attributed to this compute process on the exact
    /// CUDA device UUID backing the context.
    ///
    /// This intentionally has no `cuMemGetInfo` fallback. Missing NVML,
    /// unsupported device queries, permission failures and unavailable process
    /// memory values remain explicit `Unavailable` capability states.
    pub fn process_gpu_memory_probe_v1(
        &self,
    ) -> crate::process_memory::ProcessGpuMemoryProbeV1 {
        crate::process_memory::probe(self)
    }
'''
if 'pub fn process_gpu_memory_probe_v1' not in text:
    if anchor not in text:
        raise SystemExit('Context::mem_info anchor missing')
    text = text.replace(anchor, method, 1)
context.write_text(text)

nvml = Path('crates/nnis-sys/src/nvml.rs')
text = nvml.read_text()
if 'NVML_ERROR_ALREADY_INITIALIZED' not in text:
    text = text.replace(
        'pub const NVML_ERROR_NO_PERMISSION: nvmlReturn_t = 4;\n',
        'pub const NVML_ERROR_NO_PERMISSION: nvmlReturn_t = 4;\n'
        '/// Legacy status returned by older NVML versions when already initialized.\n'
        'pub const NVML_ERROR_ALREADY_INITIALIZED: nvmlReturn_t = 5;\n',
        1,
    )
    text = text.replace(
        '        assert_eq!(NVML_ERROR_NO_PERMISSION, 4);\n',
        '        assert_eq!(NVML_ERROR_NO_PERMISSION, 4);\n'
        '        assert_eq!(NVML_ERROR_ALREADY_INITIALIZED, 5);\n',
        1,
    )
nvml.write_text(text)

process_memory = Path('crates/nnis-rt/src/process_memory.rs')
text = process_memory.read_text()
old = '    if init_status != nvml::NVML_SUCCESS {\n'
new = '''    if init_status != nvml::NVML_SUCCESS
        && init_status != nvml::NVML_ERROR_ALREADY_INITIALIZED
    {
'''
if old in text:
    text = text.replace(old, new, 1)
process_memory.write_text(text)
