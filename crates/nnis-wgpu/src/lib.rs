//! Portable WGPU backend qualification surface for NNIS (NNIS-P3, first slice).
//!
//! This crate is the only NNIS crate that depends on `wgpu`; `nnis-core` and
//! `nnis-cpu` stay dependency-free. It provides:
//!
//! - adapter/device discovery that returns `None` when no adapter exists;
//! - a mapping from the granted WGPU device limits and features to the NNIS
//!   [`CapabilitySet`] consumed by
//!   [`KernelArtifactV1::bind`](nnis_core::kernel_artifact::KernelArtifactV1::bind),
//!   plus the WGPU-specific per-binding limits that set does not express;
//! - one tiny WGSL kernel (finite F32 elementwise add) carried as a P4 kernel
//!   artifact and executed only after fail-closed binding.
//!
//! An adapter whose device type is CPU, or whose name identifies a known
//! software rasterizer, is reported as software. Software adapters exercise
//! the API path only and are never hardware evidence. Nothing here measures or
//! claims performance.

#![forbid(unsafe_code)]

use core::fmt;
use std::borrow::Cow;
use std::future::Future;
use std::pin::pin;
use std::sync::{mpsc, Arc};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

use nnis_core::kernel_artifact::{
    BoundKernelArtifactV1, KernelArtifactError, KernelArtifactFieldsV1, KernelArtifactV1,
    KernelBindingKindV1, KernelBindingV1, KernelElementTypeV1, KernelSourceKindV1,
};
use nnis_core::{BackendFamily, BackendId, CapabilitySet, PortableError};

pub use wgpu;

/// Exact `wgpu` major line this crate is written against.
pub const WGPU_API_LINE: &str = "23";

/// Workgroup width of the elementwise add kernel.
pub const ADD_F32_WORKGROUP_SIZE: u32 = 64;

/// Numerical policy implemented by the WGSL elementwise add kernel.
///
/// WGSL requires correctly rounded binary32 addition but permits flushing
/// subnormals, so the declared tolerance is bit-exact equality with the CPU
/// reference for finite inputs whose operands and exact sums are normal or
/// zero. Subnormal operands or results are outside the declared contract.
pub const WGSL_F32_ADD_NUMERICAL_POLICY: &str = "wgsl-f32-add-correctly-rounded-normal-range-v1";

/// WGSL source of the elementwise add kernel artifact.
pub const ADD_F32_WGSL: &str = "\
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn add_f32(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index < arrayLength(&output)) {
        output[index] = left[index] + right[index];
    }
}
";

/// Entry point of [`ADD_F32_WGSL`].
pub const ADD_F32_ENTRY_POINT: &str = "add_f32";

/// Kind of adapter reported by WGPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WgpuAdapterClassV1 {
    /// Discrete, integrated, or virtual GPU reported by the driver.
    Hardware,
    /// CPU device type or a known software implementation.
    Software,
    /// Device type not reported.
    Unknown,
}

/// Identity of a discovered adapter, as reported by the driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgpuAdapterReportV1 {
    /// Adapter name.
    pub name: String,
    /// WGPU backend API (Vulkan, Metal, Dx12, Gl, ...).
    pub backend: String,
    /// Reported device type.
    pub device_type: String,
    /// PCI vendor id, or 0 when unknown.
    pub vendor_id: u32,
    /// PCI device id, or 0 when unknown.
    pub device_id: u32,
    /// Driver name.
    pub driver: String,
    /// Driver information string.
    pub driver_info: String,
    /// Classification; `Software` is never hardware evidence.
    pub class: WgpuAdapterClassV1,
}

impl WgpuAdapterReportV1 {
    /// Build a report from WGPU adapter information.
    pub fn from_info(info: &wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            vendor_id: info.vendor,
            device_id: info.device,
            driver: info.driver.clone(),
            driver_info: info.driver_info.clone(),
            class: classify_adapter(info),
        }
    }

    /// Whether this adapter may be cited as hardware (still requires separate
    /// physical qualification evidence).
    pub fn is_hardware(&self) -> bool {
        self.class == WgpuAdapterClassV1::Hardware
    }
}

/// Classify an adapter; known software implementations are `Software` even
/// when they report a GPU device type.
pub fn classify_adapter(info: &wgpu::AdapterInfo) -> WgpuAdapterClassV1 {
    const SOFTWARE_MARKERS: [&str; 5] = ["llvmpipe", "lavapipe", "swiftshader", "softpipe", "warp"];
    let name = info.name.to_ascii_lowercase();
    let driver = info.driver.to_ascii_lowercase();
    if info.device_type == wgpu::DeviceType::Cpu
        || SOFTWARE_MARKERS
            .iter()
            .any(|marker| name.contains(marker) || driver.contains(marker))
    {
        return WgpuAdapterClassV1::Software;
    }
    match info.device_type {
        wgpu::DeviceType::DiscreteGpu
        | wgpu::DeviceType::IntegratedGpu
        | wgpu::DeviceType::VirtualGpu => WgpuAdapterClassV1::Hardware,
        _ => WgpuAdapterClassV1::Unknown,
    }
}

/// Map granted WGPU device limits and features to NNIS capability limits.
///
/// `max_buffer_bytes` is the smaller of the buffer-size and storage-binding
/// limits, because every NNIS kernel binding checked against it may be a
/// storage binding. `max_bindings` is the per-stage storage plus uniform
/// buffer budget, capped by the bind-group budget. Uniform-specific and
/// per-group limits are checked separately by [`check_wgpu_binding_limits`].
/// Timestamps are reported only if the device was granted timestamp queries.
pub fn capabilities_from_limits(
    limits: &wgpu::Limits,
    features: wgpu::Features,
) -> Result<CapabilitySet, PortableError> {
    let bindings_per_stage = limits
        .max_storage_buffers_per_shader_stage
        .saturating_add(limits.max_uniform_buffers_per_shader_stage);
    let bindings_in_groups = limits
        .max_bindings_per_bind_group
        .saturating_mul(limits.max_bind_groups);
    CapabilitySet {
        max_buffer_bytes: limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size)),
        max_workgroup_invocations: limits.max_compute_invocations_per_workgroup,
        max_workgroup_size: [
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
        ],
        max_bindings: bindings_per_stage.min(bindings_in_groups),
        supports_f16: features.contains(wgpu::Features::SHADER_F16),
        supports_timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
    }
    .validate()
}

/// WGPU binding limit named by a failed [`check_wgpu_binding_limits`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WgpuBindingLimitV1 {
    /// Bind group index at or above `max_bind_groups`.
    BindGroupIndex,
    /// Binding index at or above `max_bindings_per_bind_group`.
    BindingIndex,
    /// More storage bindings than `max_storage_buffers_per_shader_stage`.
    StorageBuffersPerStage,
    /// More uniform bindings than `max_uniform_buffers_per_shader_stage`.
    UniformBuffersPerStage,
    /// Uniform binding larger than `max_uniform_buffer_binding_size`.
    UniformBindingSize,
}

/// Check artifact bindings against WGPU limits not expressed by
/// [`CapabilitySet`], failing closed.
pub fn check_wgpu_binding_limits(
    artifact: &KernelArtifactV1,
    limits: &wgpu::Limits,
) -> Result<(), WgpuBindingLimitV1> {
    let mut storage = 0u32;
    let mut uniform = 0u32;
    for binding in &artifact.fields().bindings {
        if binding.group >= limits.max_bind_groups {
            return Err(WgpuBindingLimitV1::BindGroupIndex);
        }
        if binding.binding >= limits.max_bindings_per_bind_group {
            return Err(WgpuBindingLimitV1::BindingIndex);
        }
        match binding.kind {
            KernelBindingKindV1::StorageReadOnly | KernelBindingKindV1::StorageReadWrite => {
                storage += 1;
            }
            KernelBindingKindV1::Uniform => {
                uniform += 1;
                if binding.min_size_bytes > u64::from(limits.max_uniform_buffer_binding_size) {
                    return Err(WgpuBindingLimitV1::UniformBindingSize);
                }
            }
        }
    }
    if storage > limits.max_storage_buffers_per_shader_stage {
        return Err(WgpuBindingLimitV1::StorageBuffersPerStage);
    }
    if uniform > limits.max_uniform_buffers_per_shader_stage {
        return Err(WgpuBindingLimitV1::UniformBuffersPerStage);
    }
    Ok(())
}

/// Build the P4 artifact for [`ADD_F32_WGSL`] over `elements` F32 values.
pub fn add_f32_artifact(elements: u64) -> Result<KernelArtifactV1, WgpuBackendError> {
    let bytes = elements.checked_mul(4).filter(|&bytes| bytes > 0).ok_or(
        WgpuBackendError::InvalidInput("element count must be non-zero and representable"),
    )?;
    let storage = |binding, kind| KernelBindingV1 {
        group: 0,
        binding,
        kind,
        element: KernelElementTypeV1::F32,
        min_size_bytes: bytes,
    };
    Ok(KernelArtifactV1::new(KernelArtifactFieldsV1 {
        artifact_id: "nnis.wgpu.add_f32".into(),
        artifact_revision: 1,
        source_kind: KernelSourceKindV1::Wgsl,
        source: ADD_F32_WGSL.as_bytes().to_vec(),
        entry_point: ADD_F32_ENTRY_POINT.into(),
        bindings: vec![
            storage(0, KernelBindingKindV1::StorageReadOnly),
            storage(1, KernelBindingKindV1::StorageReadOnly),
            storage(2, KernelBindingKindV1::StorageReadWrite),
        ],
        workgroup_size: [ADD_F32_WORKGROUP_SIZE, 1, 1],
        numerical_policy: WGSL_F32_ADD_NUMERICAL_POLICY.into(),
        qualification_evidence_sha256: None,
    })?)
}

/// Discovered WGPU device with its NNIS identity and capability limits.
pub struct WgpuDevice {
    adapter: WgpuAdapterReportV1,
    backend_id: BackendId,
    capabilities: CapabilitySet,
    limits: wgpu::Limits,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl fmt::Debug for WgpuDevice {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        output
            .debug_struct("WgpuDevice")
            .field("adapter", &self.adapter)
            .field("backend_id", &self.backend_id)
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

impl WgpuDevice {
    /// Discover the default adapter for the primary native backends (or those
    /// selected by `WGPU_BACKEND`) and create a device with the adapter's full
    /// limits. Returns `Ok(None)` when no adapter exists.
    pub fn discover() -> Result<Option<Self>, WgpuBackendError> {
        let backends = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let adapter = match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })) {
            Some(adapter) => adapter,
            None => return Ok(None),
        };
        let report = WgpuAdapterReportV1::from_info(&adapter.get_info());
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("nnis-wgpu"),
                required_features: adapter.features() & wgpu::Features::SHADER_F16,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|error| WgpuBackendError::Device(error.to_string()))?;
        let limits = device.limits();
        let capabilities = capabilities_from_limits(&limits, device.features())?;
        let backend_id = BackendId::new(
            BackendFamily::Wgpu,
            format!("wgpu-{WGPU_API_LINE}:{}:{}", report.backend, report.name),
        )?;
        Ok(Some(Self {
            adapter: report,
            backend_id,
            capabilities,
            limits,
            device,
            queue,
        }))
    }

    /// Adapter identity.
    pub fn adapter(&self) -> &WgpuAdapterReportV1 {
        &self.adapter
    }

    /// NNIS backend identity (`Wgpu` family).
    pub fn backend_id(&self) -> &BackendId {
        &self.backend_id
    }

    /// NNIS capability limits derived from the granted device limits.
    pub fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    /// Granted WGPU device limits.
    pub fn limits(&self) -> &wgpu::Limits {
        &self.limits
    }

    /// Bind an artifact to this device: P4 fingerprint, family and capability
    /// checks, then the WGPU per-binding limits.
    pub fn bind(
        &self,
        artifact: &KernelArtifactV1,
        expected_fingerprint: &[u8; 32],
    ) -> Result<BoundKernelArtifactV1, WgpuBackendError> {
        let bound = artifact.bind(&self.backend_id, &self.capabilities, expected_fingerprint)?;
        check_wgpu_binding_limits(artifact, &self.limits)?;
        Ok(bound)
    }

    /// Run the WGSL elementwise add kernel on finite equal-length inputs.
    ///
    /// The artifact is built, fingerprint-checked and bound before any GPU
    /// object is created. Returns the output values; no timing is recorded.
    pub fn add_f32(&self, left: &[f32], right: &[f32]) -> Result<Vec<f32>, WgpuBackendError> {
        if left.len() != right.len() {
            return Err(WgpuBackendError::InvalidInput(
                "inputs must have equal length",
            ));
        }
        if left.iter().chain(right).any(|value| !value.is_finite()) {
            return Err(WgpuBackendError::InvalidInput("inputs must be finite"));
        }
        let elements = left.len() as u64;
        let artifact = add_f32_artifact(elements)?;
        let fingerprint = *add_f32_artifact(elements)?.artifact_fingerprint();
        self.bind(&artifact, &fingerprint)?;
        let workgroups = u32::try_from(elements.div_ceil(u64::from(ADD_F32_WORKGROUP_SIZE)))
            .ok()
            .filter(|&groups| groups <= self.limits.max_compute_workgroups_per_dimension)
            .ok_or(WgpuBackendError::InvalidInput(
                "element count exceeds dispatch limit",
            ))?;
        let bytes = elements * 4;

        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let source = core::str::from_utf8(&artifact.fields().source)
            .map_err(|_| WgpuBackendError::InvalidInput("artifact source is not UTF-8"))?;
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("nnis.wgpu.add_f32"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("nnis.wgpu.add_f32"),
                layout: None,
                module: &module,
                entry_point: Some(&artifact.fields().entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
        let storage = |label, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        let input_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let left_buffer = storage("left", input_usage);
        let right_buffer = storage("right", input_usage);
        let output_buffer = storage(
            "output",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = storage(
            "readback",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        self.queue.write_buffer(&left_buffer, 0, &to_le_bytes(left));
        self.queue
            .write_buffer(&right_buffer, 0, &to_le_bytes(right));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nnis.wgpu.add_f32"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &left_buffer),
                entry(1, &right_buffer),
                entry(2, &output_buffer),
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("nnis.wgpu.add_f32"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nnis.wgpu.add_f32"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let validation = block_on(self.device.pop_error_scope());
        let out_of_memory = block_on(self.device.pop_error_scope());
        if let Some(error) = validation.or(out_of_memory) {
            return Err(WgpuBackendError::Execution(error.to_string()));
        }

        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| WgpuBackendError::Execution("readback callback dropped".into()))?
            .map_err(|error| WgpuBackendError::Execution(error.to_string()))?;
        let values = slice
            .get_mapped_range()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        readback.unmap();
        Ok(values)
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn to_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Block the current thread on a future (WGPU native futures resolve
/// promptly; this avoids an executor dependency).
fn block_on<F: Future>(future: F) -> F::Output {
    struct ThreadWaker(Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => thread::park(),
        }
    }
}

/// Fail-closed WGPU backend errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgpuBackendError {
    /// Invalid caller input.
    InvalidInput(&'static str),
    /// Portable contract violation (capability or identity).
    Portable(PortableError),
    /// P4 artifact validation or binding failure.
    Artifact(KernelArtifactError),
    /// WGPU-specific binding limit exceeded.
    BindingLimit(WgpuBindingLimitV1),
    /// Device request failed.
    Device(String),
    /// Validation, allocation, or readback failure during execution.
    Execution(String),
}

impl From<PortableError> for WgpuBackendError {
    fn from(error: PortableError) -> Self {
        Self::Portable(error)
    }
}

impl From<KernelArtifactError> for WgpuBackendError {
    fn from(error: KernelArtifactError) -> Self {
        Self::Artifact(error)
    }
}

impl From<WgpuBindingLimitV1> for WgpuBackendError {
    fn from(limit: WgpuBindingLimitV1) -> Self {
        Self::BindingLimit(limit)
    }
}

impl fmt::Display for WgpuBackendError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(reason) => write!(output, "invalid input: {reason}"),
            Self::Portable(error) => write!(output, "portable contract: {error}"),
            Self::Artifact(error) => write!(output, "kernel artifact: {error}"),
            Self::BindingLimit(limit) => write!(output, "WGPU binding limit exceeded: {limit:?}"),
            Self::Device(reason) => write!(output, "WGPU device request failed: {reason}"),
            Self::Execution(reason) => write!(output, "WGPU execution failed: {reason}"),
        }
    }
}

impl std::error::Error for WgpuBackendError {}
