//! Portable WGPU execution session for Pascal Vector Projection (PVP).
//!
//! This is an independent NNIS implementation of the frozen PVP logical
//! contract. It consumes the backend-neutral NNIS-PVP0 WGPU-u32 projection,
//! keeps the state resident in one WGPU storage buffer, executes one exact
//! subset-zeta stage per dispatch, and revalidates readback snapshots against
//! the NNIS-PVP0 representation contract.
//!
//! It deliberately does not depend on FLAT-ATTENTION. FLAT owns specialized
//! GPU kernel research; this module is an independent portable runtime carrier
//! used for cross-session replication against NNIS-PVP1 CPU execution.
//!
//! Software adapters are correctness evidence only. Nothing here establishes a
//! physical-GPU latency, throughput, or bandwidth claim.

use core::fmt;
use std::borrow::Cow;
use std::sync::mpsc;

use nnis_core::pvp::{PvpLayoutAdapterV1, PvpLayoutError, PvpPhysicalWordV1, PvpWgpuU32V1};

use crate::{block_on, WgpuAdapterClassV1, WgpuBackendError, WgpuDevice};

/// Stable identity of the NNIS portable WGPU PVP execution contract.
pub const NNIS_PVP_WGPU_EXECUTION_SCHEMA_V1: &str = "nnis.pvp-wgpu-session.v1";
/// Workgroup width of the independent scalar-u32 WGPU reference.
pub const NNIS_PVP_WGPU_WORKGROUP_SIZE: u32 = 64;
/// Entry point of the NNIS PVP WGSL stage kernel.
pub const NNIS_PVP_WGPU_ENTRY_POINT: &str = "pvp_subset_zeta_stage";

/// Independent NNIS scalar-u32 WGSL realization of one PVP butterfly stage.
pub const NNIS_PVP_WGPU_WGSL: &str = "\
struct Params {
    addresses: u32,
    words_per_address: u32,
    stride: u32,
    pair_count: u32,
};

@group(0) @binding(0) var<storage, read_write> state_words: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn pvp_subset_zeta_stage(@builtin(global_invocation_id) id: vec3<u32>) {
    let linear = id.x;
    let total = params.pair_count * params.words_per_address;
    if (linear >= total) {
        return;
    }

    let word = linear % params.words_per_address;
    let pair_index = linear / params.words_per_address;
    let block = pair_index / params.stride;
    let offset = pair_index % params.stride;
    let source_address = block * (2u * params.stride) + offset;
    let target_address = source_address + params.stride;

    if (target_address >= params.addresses) {
        return;
    }

    let source_index = source_address * params.words_per_address + word;
    let target_index = target_address * params.words_per_address + word;
    state_words[target_index] = state_words[target_index] ^ state_words[source_index];
}
";

/// Fail-closed NNIS PVP WGPU runtime error.
#[derive(Debug)]
pub enum WgpuPvpErrorV1 {
    /// NNIS-PVP0 layout or canonical-padding rejection.
    Layout(PvpLayoutError),
    /// WGPU adapter/device/runtime failure.
    Backend(WgpuBackendError),
    /// Requested PVP geometry cannot be represented by this WGPU contract.
    Geometry(&'static str),
    /// WGPU validation, allocation, or readback failure.
    Execution(String),
}

impl From<PvpLayoutError> for WgpuPvpErrorV1 {
    fn from(error: PvpLayoutError) -> Self {
        Self::Layout(error)
    }
}

impl From<WgpuBackendError> for WgpuPvpErrorV1 {
    fn from(error: WgpuBackendError) -> Self {
        Self::Backend(error)
    }
}

impl fmt::Display for WgpuPvpErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(error) => write!(formatter, "PVP layout: {error}"),
            Self::Backend(error) => write!(formatter, "PVP WGPU backend: {error}"),
            Self::Geometry(message) => write!(formatter, "PVP WGPU geometry: {message}"),
            Self::Execution(message) => write!(formatter, "PVP WGPU execution: {message}"),
        }
    }
}

impl std::error::Error for WgpuPvpErrorV1 {}

/// Exact structural accounting for one complete WGPU PVP transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuPvpExecutionStatsV1 {
    pub stages: u32,
    pub logical_gate_xor_ops: u128,
    pub packed_u32_updates: u128,
    pub logical_dispatches: u32,
    pub workgroups_per_stage: u32,
    pub state_words: usize,
    pub state_bytes: usize,
    pub scratch_state_words: usize,
    pub execution_index: u64,
    pub adapter_class: WgpuAdapterClassV1,
}

/// Device-resident portable WGPU PVP session.
pub struct WgpuPvpSessionV1<'a> {
    device: &'a WgpuDevice,
    layout: PvpLayoutAdapterV1,
    state: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    executions: u64,
}

impl fmt::Debug for WgpuPvpSessionV1<'_> {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output
            .debug_struct("WgpuPvpSessionV1")
            .field("layout", &self.layout)
            .field("adapter", self.device.adapter())
            .field("executions", &self.executions)
            .finish_non_exhaustive()
    }
}

impl<'a> WgpuPvpSessionV1<'a> {
    /// Upload one validated NNIS-PVP0 WGPU-u32 state and create the reference
    /// compute pipeline. The state remains resident until the session is
    /// dropped.
    pub fn new(
        device: &'a WgpuDevice,
        initial: &PvpWgpuU32V1,
    ) -> Result<Self, WgpuPvpErrorV1> {
        let layout = initial.layout();
        let state_bytes = layout.storage_bytes(PvpPhysicalWordV1::WgpuU32)?;
        let state_bytes_u64 = u64::try_from(state_bytes)
            .map_err(|_| WgpuPvpErrorV1::Geometry("state bytes exceed u64"))?;
        let limits = device.limits();
        if state_bytes_u64 > limits.max_buffer_size {
            return Err(WgpuPvpErrorV1::Geometry(
                "state exceeds WGPU max_buffer_size",
            ));
        }
        if state_bytes_u64 > u64::from(limits.max_storage_buffer_binding_size) {
            return Err(WgpuPvpErrorV1::Geometry(
                "state exceeds WGPU max_storage_buffer_binding_size",
            ));
        }

        device.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let module = device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nnis.pvp-wgpu-session.v1"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(NNIS_PVP_WGPU_WGSL)),
        });
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("nnis.pvp-wgpu-session.v1"),
                layout: None,
                module: &module,
                entry_point: Some(NNIS_PVP_WGPU_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let state = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nnis.pvp-wgpu-state"),
            size: state_bytes_u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&state, 0, &u32_to_le_bytes(initial.words()));

        let validation = block_on(device.device.pop_error_scope());
        let out_of_memory = block_on(device.device.pop_error_scope());
        if let Some(error) = validation.or(out_of_memory) {
            return Err(WgpuPvpErrorV1::Execution(error.to_string()));
        }

        Ok(Self {
            device,
            layout,
            state,
            pipeline,
            executions: 0,
        })
    }

    #[must_use]
    pub const fn layout(&self) -> PvpLayoutAdapterV1 {
        self.layout
    }

    #[must_use]
    pub const fn execution_count(&self) -> u64 {
        self.executions
    }

    #[must_use]
    pub fn adapter_class(&self) -> WgpuAdapterClassV1 {
        self.device.adapter().class
    }

    /// Execute every PVP subset-zeta stage over the resident state.
    pub fn execute_subset_zeta(&mut self) -> Result<WgpuPvpExecutionStatsV1, WgpuPvpErrorV1> {
        let addresses = u32::try_from(self.layout.addresses()).map_err(|_| {
            WgpuPvpErrorV1::Geometry("address count exceeds WGSL u32 index space")
        })?;
        let words_per_address_usize = self
            .layout
            .words_per_address(PvpPhysicalWordV1::WgpuU32)?;
        let words_per_address = u32::try_from(words_per_address_usize).map_err(|_| {
            WgpuPvpErrorV1::Geometry("words per address exceed WGSL u32 index space")
        })?;
        let pair_count = addresses / 2;
        let invocations = pair_count
            .checked_mul(words_per_address)
            .ok_or(WgpuPvpErrorV1::Geometry(
                "dispatch invocation count overflows u32",
            ))?;
        let workgroups = invocations.div_ceil(NNIS_PVP_WGPU_WORKGROUP_SIZE);
        if workgroups > self.device.limits().max_compute_workgroups_per_dimension {
            return Err(WgpuPvpErrorV1::Geometry(
                "dispatch exceeds max_compute_workgroups_per_dimension",
            ));
        }

        self.device
            .device
            .push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        self.device
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);

        let mut encoder =
            self.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("nnis.pvp-wgpu-execute"),
                });

        let mut stride = 1_u32;
        while stride < addresses {
            let params = [addresses, words_per_address, stride, pair_count];
            let uniform = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nnis.pvp-wgpu-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.device
                .queue
                .write_buffer(&uniform, 0, &u32_to_le_bytes(&params));
            let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nnis.pvp-wgpu-bind-group"),
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.state.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("nnis.pvp-wgpu-stage"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            stride = stride
                .checked_mul(2)
                .ok_or(WgpuPvpErrorV1::Geometry("stride overflow"))?;
        }

        self.device.queue.submit(Some(encoder.finish()));

        let validation = block_on(self.device.device.pop_error_scope());
        let out_of_memory = block_on(self.device.device.pop_error_scope());
        if let Some(error) = validation.or(out_of_memory) {
            return Err(WgpuPvpErrorV1::Execution(error.to_string()));
        }

        self.executions = self
            .executions
            .checked_add(1)
            .ok_or(WgpuPvpErrorV1::Geometry("execution counter overflow"))?;

        let pairs =
            (self.layout.addresses() / 2) as u128 * u128::from(self.layout.stages());
        Ok(WgpuPvpExecutionStatsV1 {
            stages: self.layout.stages(),
            logical_gate_xor_ops: pairs * self.layout.gates() as u128,
            packed_u32_updates: pairs * words_per_address_usize as u128,
            logical_dispatches: self.layout.stages(),
            workgroups_per_stage: workgroups,
            state_words: self
                .layout
                .storage_words(PvpPhysicalWordV1::WgpuU32)?,
            state_bytes: self
                .layout
                .storage_bytes(PvpPhysicalWordV1::WgpuU32)?,
            scratch_state_words: 0,
            execution_index: self.executions,
            adapter_class: self.device.adapter().class,
        })
    }

    /// Read back and revalidate the resident state as NNIS-PVP0 WGPU-u32.
    pub fn snapshot(&self) -> Result<PvpWgpuU32V1, WgpuPvpErrorV1> {
        let bytes = self
            .layout
            .storage_bytes(PvpPhysicalWordV1::WgpuU32)?;
        let bytes_u64 = u64::try_from(bytes)
            .map_err(|_| WgpuPvpErrorV1::Geometry("readback size exceeds u64"))?;
        let readback = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nnis.pvp-wgpu-readback"),
            size: bytes_u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("nnis.pvp-wgpu-readback"),
                });
        encoder.copy_buffer_to_buffer(&self.state, 0, &readback, 0, bytes_u64);
        self.device.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| WgpuPvpErrorV1::Execution("readback callback dropped".into()))?
            .map_err(|error| WgpuPvpErrorV1::Execution(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        if mapped.len() != bytes {
            return Err(WgpuPvpErrorV1::Execution(
                "readback byte length does not match PVP layout".into(),
            ));
        }
        let words = mapped
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        drop(mapped);
        readback.unmap();
        Ok(PvpWgpuU32V1::new(self.layout, words)?)
    }

    /// Stable runtime record tied to the adapter identity and logical layout.
    pub fn canonical_record(&self) -> Result<String, WgpuPvpErrorV1> {
        Ok(format!(
            "{};backend={:?};adapter_class={:?};executions={};{}",
            NNIS_PVP_WGPU_EXECUTION_SCHEMA_V1,
            self.device.backend_id(),
            self.device.adapter().class,
            self.executions,
            self.layout
                .canonical_record(PvpPhysicalWordV1::WgpuU32)?
        ))
    }
}

fn u32_to_le_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_contract_names_the_expected_entry_and_bindings() {
        assert_eq!(NNIS_PVP_WGPU_ENTRY_POINT, "pvp_subset_zeta_stage");
        assert!(NNIS_PVP_WGPU_WGSL.contains("@binding(0)"));
        assert!(NNIS_PVP_WGPU_WGSL.contains("@binding(1)"));
        assert!(NNIS_PVP_WGPU_WGSL.contains("state_words[target_index]"));
        assert!(NNIS_PVP_WGPU_WGSL.contains("^ state_words[source_index]"));
    }
}
