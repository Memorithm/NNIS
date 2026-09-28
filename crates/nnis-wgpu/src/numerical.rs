//! Finite-F32 WGSL reference kernels mirroring `nnis-cpu` numerical operations.
//!
//! Each operation is a P4 [`KernelArtifactV1`] bound to the device (fingerprint,
//! family, capabilities, WGPU binding limits) before any pipeline is created.
//! Semantics follow `CpuF32KernelsV1`:
//!
//! - every buffer needs `STORAGE` and a non-zero multiple of four bytes;
//! - every input value (including unselected gather inputs and the whole old
//!   scatter destination) and every arithmetic intermediate must be finite;
//! - the output is staged in a scratch buffer and copied to the destination
//!   only on success, so every returned error preserves the destination.
//!
//! Loads, stores, finiteness checks, ReLU and gather operate on raw `u32`
//! bits, so they cannot flush subnormals or canonicalize values.
//! Arithmetic follows the declared policies:
//!
//! - [`WGSL_F32_EXACT_NUMERICAL_POLICY`] (add, multiply, sum, scatter-add):
//!   WGSL binary32 `+`/`*` are correctly rounded, and sum and scatter-add run
//!   serially in the CPU order from `+0`. Results must be bit-exact with the
//!   CPU reference for normal-range operands and results. Implementations may
//!   flush subnormal arithmetic, so subnormal operands or results are outside
//!   the contract.
//! - [`WGSL_F32_PROJECTION_NUMERICAL_POLICY`] (project_kn): WGSL `fma`
//!   accuracy is inherited from `a * b + c`, so a device may round twice
//!   where the CPU reference fuses. Each output must satisfy
//!   `|wgpu - cpu| <= (3k + 1) * 2^-24 * sum_r |x_r * w_rj|`. The sign of a
//!   zero result is not compared.
//!
//! When several errors apply at once, the variant reported can differ from
//! the CPU reference, because structural checks run on the host before
//! finiteness checks run on the device. Serial kernels (sum, scatter-add) run
//! in one invocation and are reference implementations, not performance
//! paths. No timing is recorded.

use std::borrow::Cow;

use nnis_core::kernel_artifact::{
    KernelArtifactFieldsV1, KernelArtifactV1, KernelBindingKindV1, KernelBindingV1,
    KernelElementTypeV1, KernelSourceKindV1,
};
use nnis_core::{BufferUsages, PortableDevice, PortableError, Result};

use crate::memory::{require_usage, WgpuBuffer, WgpuQueue};
use crate::{block_on, WgpuDevice};

/// Numerical contract version of the WGSL F32 kernels.
pub const WGSL_F32_CONTRACT_VERSION: u32 = 1;

/// Policy of the bit-exact kernels (normal range).
pub const WGSL_F32_EXACT_NUMERICAL_POLICY: &str =
    "wgsl-f32-finite-serial-correctly-rounded-normal-range-v1";

/// Policy of the projection kernel (declared error bound).
pub const WGSL_F32_PROJECTION_NUMERICAL_POLICY: &str =
    "wgsl-f32-finite-fma-inherited-bound-3k-plus-1-ulp-v1";

/// Policy of the bit-manipulation kernels (ReLU, gather, finiteness).
pub const WGSL_F32_BITWISE_NUMERICAL_POLICY: &str = "wgsl-f32-finite-bitwise-exact-v1";

const WORKGROUP: u32 = 64;

const COMMON: &str = "
fn nnis_finite(bits: u32) -> bool {
    return (bits & 0x7f800000u) != 0x7f800000u;
}
";

/// Elementwise binary operation; no implicit broadcasting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuF32BinaryOp {
    Add,
    Multiply,
}

/// Operation identity recorded after successful execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuF32Operation {
    Add,
    Multiply,
    Relu,
    Sum,
    ProjectKn,
    Gather,
    ScatterAdd,
}

/// Successful execution record; not timing, residency or performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuF32ReportV1 {
    pub schema_version: u32,
    pub operation: WgpuF32Operation,
    pub output_values: usize,
    /// Fingerprint of the bound operation artifact.
    pub artifact_fingerprint: [u8; 32],
}

/// WGSL reference kernels on one device and its queue.
#[derive(Debug)]
pub struct WgpuF32KernelsV1<'a> {
    device: &'a WgpuDevice,
    queue: WgpuQueue,
}

pub(crate) struct Binding<'b> {
    pub(crate) buffer: &'b wgpu::Buffer,
    pub(crate) kind: KernelBindingKindV1,
    pub(crate) element: KernelElementTypeV1,
    pub(crate) bytes: u64,
}

impl<'a> WgpuF32KernelsV1<'a> {
    /// Create kernels submitting on a fresh queue of `device`.
    pub fn new(device: &'a WgpuDevice) -> Result<Self> {
        Ok(Self {
            device,
            queue: device.create_queue()?,
        })
    }

    /// Add or multiply equal-length vectors without broadcasting.
    pub fn binary(
        &self,
        operation: WgpuF32BinaryOp,
        left: &WgpuBuffer,
        right: &WgpuBuffer,
        output: &mut WgpuBuffer,
    ) -> Result<WgpuF32ReportV1> {
        let n = self.words(left)?;
        require_equal(self.words(right)?, n)?;
        require_equal(self.words(output)?, n)?;
        let (symbol, identity, id) = match operation {
            WgpuF32BinaryOp::Add => ("+", WgpuF32Operation::Add, "nnis.wgpu.f32.add"),
            WgpuF32BinaryOp::Multiply => {
                ("*", WgpuF32Operation::Multiply, "nnis.wgpu.f32.multiply")
            }
        };
        let source = format!(
            "{COMMON}
@group(0) @binding(0) var<storage, read> left: array<u32>;
@group(0) @binding(1) var<storage, read> right: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= arrayLength(&output)) {{ return; }}
    let bits = bitcast<u32>(bitcast<f32>(left[i]) {symbol} bitcast<f32>(right[i]));
    if (!nnis_finite(bits)) {{ atomicOr(&status, 1u); }}
    output[i] = bits;
}}
"
        );
        self.execute(
            id,
            WGSL_F32_EXACT_NUMERICAL_POLICY,
            &source,
            [WORKGROUP, 1, 1],
            n,
            &[left, right],
            &[read(left), read(right)],
            output,
            false,
            identity,
        )
    }

    /// ReLU maps negative values and both signed zeros to positive zero.
    pub fn relu(&self, input: &WgpuBuffer, output: &mut WgpuBuffer) -> Result<WgpuF32ReportV1> {
        let n = self.words(input)?;
        require_equal(self.words(output)?, n)?;
        let source = format!(
            "{COMMON}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= arrayLength(&output)) {{ return; }}
    let bits = input[i];
    output[i] = select(0u, bits, (bits & 0x80000000u) == 0u && bits != 0u);
}}
"
        );
        self.execute(
            "nnis.wgpu.f32.relu",
            WGSL_F32_BITWISE_NUMERICAL_POLICY,
            &source,
            [WORKGROUP, 1, 1],
            n,
            &[input],
            &[read(input)],
            output,
            false,
            WgpuF32Operation::Relu,
        )
    }

    /// Reduce serially in increasing index order from +0. Output holds one F32.
    pub fn sum(&self, input: &WgpuBuffer, output: &mut WgpuBuffer) -> Result<WgpuF32ReportV1> {
        self.words(input)?;
        require_equal(self.words(output)?, 1)?;
        let source = format!(
            "{COMMON}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> status: atomic<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {{
    var sum: f32 = 0.0;
    let n = arrayLength(&input);
    for (var i = 0u; i < n; i = i + 1u) {{
        sum = sum + bitcast<f32>(input[i]);
        if (!nnis_finite(bitcast<u32>(sum))) {{
            atomicOr(&status, 1u);
            break;
        }}
    }}
    output[0] = bitcast<u32>(sum);
}}
"
        );
        self.execute(
            "nnis.wgpu.f32.sum",
            WGSL_F32_EXACT_NUMERICAL_POLICY,
            &source,
            [1, 1, 1],
            1,
            &[input],
            &[read(input)],
            output,
            false,
            WgpuF32Operation::Sum,
        )
    }

    /// Project `[1,K] x [K,N] -> [1,N]` with row-major weights, without bias.
    ///
    /// Each output starts at +0 and applies WGSL `fma` in increasing K order;
    /// see [`WGSL_F32_PROJECTION_NUMERICAL_POLICY`] for the declared bound.
    pub fn project_kn(
        &self,
        input: &WgpuBuffer,
        weights: &WgpuBuffer,
        output: &mut WgpuBuffer,
        k: usize,
        n: usize,
    ) -> Result<WgpuF32ReportV1> {
        if k == 0 || n == 0 {
            return Err(PortableError::InvalidDescriptor(
                "WGPU F32 matrix dimensions must be positive",
            ));
        }
        let elements = k.checked_mul(n).ok_or(PortableError::InvalidDescriptor(
            "WGPU F32 matrix shape overflows usize",
        ))?;
        require_equal(self.words(input)?, k)?;
        require_equal(self.words(weights)?, elements)?;
        require_equal(self.words(output)?, n)?;
        let source = format!(
            "{COMMON}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let column = id.x;
    let n = arrayLength(&output);
    if (column >= n) {{ return; }}
    let k = arrayLength(&input);
    var sum: f32 = 0.0;
    for (var row = 0u; row < k; row = row + 1u) {{
        sum = fma(bitcast<f32>(input[row]), bitcast<f32>(weights[row * n + column]), sum);
        if (!nnis_finite(bitcast<u32>(sum))) {{
            atomicOr(&status, 1u);
            break;
        }}
    }}
    output[column] = bitcast<u32>(sum);
}}
"
        );
        self.execute(
            "nnis.wgpu.f32.project_kn",
            WGSL_F32_PROJECTION_NUMERICAL_POLICY,
            &source,
            [WORKGROUP, 1, 1],
            n,
            &[input, weights],
            &[read(input), read(weights)],
            output,
            false,
            WgpuF32Operation::ProjectKn,
        )
    }

    /// Gather selected elements in index-list order. Repeated indices are valid.
    ///
    /// The whole input is validated, including unselected values.
    pub fn gather(
        &self,
        input: &WgpuBuffer,
        indices: &[usize],
        output: &mut WgpuBuffer,
    ) -> Result<WgpuF32ReportV1> {
        let limit = self.words(input)?;
        let indices = validate_indices(indices, limit)?;
        require_equal(self.words(output)?, indices.len())?;
        let index_buffer = self.upload_indices(&indices)?;
        let source = format!(
            "{COMMON}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= arrayLength(&output)) {{ return; }}
    output[i] = input[indices[i]];
}}
"
        );
        let index_binding = Binding {
            buffer: &index_buffer,
            kind: KernelBindingKindV1::StorageReadOnly,
            element: KernelElementTypeV1::U32,
            bytes: indices.len() as u64 * 4,
        };
        self.execute(
            "nnis.wgpu.f32.gather",
            WGSL_F32_BITWISE_NUMERICAL_POLICY,
            &source,
            [WORKGROUP, 1, 1],
            indices.len(),
            &[input],
            &[read(input), index_binding],
            output,
            false,
            WgpuF32Operation::Gather,
        )
    }

    /// Add `source[i]` to `output[indices[i]]` serially in index-list order.
    ///
    /// Unselected output values are preserved. The complete old destination
    /// must be finite. An empty list is invalid.
    pub fn scatter_add(
        &self,
        source: &WgpuBuffer,
        indices: &[usize],
        output: &mut WgpuBuffer,
    ) -> Result<WgpuF32ReportV1> {
        let values = self.words(source)?;
        require_equal(values, indices.len())?;
        let limit = self.words(output)?;
        let indices = validate_indices(indices, limit)?;
        let index_buffer = self.upload_indices(&indices)?;
        let wgsl = format!(
            "{COMMON}
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {{
    let n = arrayLength(&indices);
    for (var i = 0u; i < n; i = i + 1u) {{
        let destination = indices[i];
        let bits = bitcast<u32>(bitcast<f32>(output[destination]) + bitcast<f32>(source[i]));
        if (!nnis_finite(bits)) {{
            atomicOr(&status, 1u);
            break;
        }}
        output[destination] = bits;
    }}
}}
"
        );
        let index_binding = Binding {
            buffer: &index_buffer,
            kind: KernelBindingKindV1::StorageReadOnly,
            element: KernelElementTypeV1::U32,
            bytes: indices.len() as u64 * 4,
        };
        self.execute(
            "nnis.wgpu.f32.scatter_add",
            WGSL_F32_EXACT_NUMERICAL_POLICY,
            &wgsl,
            [1, 1, 1],
            1,
            &[source],
            &[read(source), index_binding],
            output,
            true,
            WgpuF32Operation::ScatterAdd,
        )
    }

    /// Fail with `InvalidDescriptor` if any F32 in `buffer` is non-finite.
    pub(crate) fn check_finite(&self, buffer: &WgpuBuffer) -> Result<()> {
        let words = self.words(buffer)?;
        let device = self.queue.device();
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nnis.wgpu.f32.status"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        pop_scopes(device, "WGPU F32 status allocation")?;
        let bindings = [read(buffer), status_binding(&status)];
        let artifact = self.artifact(
            "nnis.wgpu.f32.validate_finite",
            WGSL_F32_BITWISE_NUMERICAL_POLICY,
            &validate_source(),
            &bindings,
            [WORKGROUP, 1, 1],
        )?;
        let groups = self.groups(words, WORKGROUP)?;
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nnis.wgpu.f32.validate_finite"),
        });
        self.dispatch(&mut encoder, &artifact, &bindings, groups);
        self.queue.queue().submit(Some(encoder.finish()));
        pop_scopes(device, "WGPU F32 finiteness check")?;
        if self.queue.read_wgpu(&status, 0, 4)? != [0, 0, 0, 0] {
            return Err(PortableError::InvalidDescriptor(
                "WGPU F32 input or arithmetic intermediate is non-finite",
            ));
        }
        Ok(())
    }

    fn words(&self, buffer: &WgpuBuffer) -> Result<usize> {
        self.queue.check_owner(buffer)?;
        require_usage(buffer, BufferUsages::STORAGE, "WGPU F32 execution")?;
        if buffer.len() % 4 != 0 {
            return Err(PortableError::InvalidDescriptor(
                "WGPU F32 buffer must contain a non-zero multiple of four bytes",
            ));
        }
        usize::try_from(buffer.len() / 4)
            .map_err(|_| PortableError::Unsupported("buffer exceeds host usize".to_string()))
    }

    fn upload_indices(&self, indices: &[u32]) -> Result<wgpu::Buffer> {
        let bytes: Vec<u8> = indices
            .iter()
            .flat_map(|index| index.to_le_bytes())
            .collect();
        let device = self.queue.device();
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nnis.wgpu.f32.indices"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.queue().write_buffer(&buffer, 0, &bytes);
        pop_scopes(device, "WGPU index upload")?;
        Ok(buffer)
    }

    pub(crate) fn queue(&self) -> &WgpuQueue {
        &self.queue
    }

    pub(crate) fn groups(&self, invocations: usize, workgroup_x: u32) -> Result<u32> {
        let groups = (invocations as u64).div_ceil(u64::from(workgroup_x));
        u32::try_from(groups)
            .ok()
            .filter(|&groups| groups <= self.device.limits().max_compute_workgroups_per_dimension)
            .ok_or_else(|| {
                PortableError::Unsupported("WGPU F32 dispatch exceeds workgroup limit".to_string())
            })
    }

    /// Bind the artifact, validate inputs, run the kernel into scratch, and
    /// commit scratch to `output` only if no non-finite value was observed.
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        artifact_id: &str,
        policy: &str,
        source: &str,
        workgroup_size: [u32; 3],
        invocations: usize,
        validate: &[&WgpuBuffer],
        inputs: &[Binding<'_>],
        output: &mut WgpuBuffer,
        seed_from_output: bool,
        operation: WgpuF32Operation,
    ) -> Result<WgpuF32ReportV1> {
        let uses_status = source.contains("var<storage, read_write> status");
        let groups = self.groups(invocations, workgroup_size[0])?;
        let device = self.queue.device();
        let output_bytes = output.len();

        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nnis.wgpu.f32.scratch"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let status = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nnis.wgpu.f32.status"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        pop_scopes(device, "WGPU F32 scratch allocation")?;

        let mut bindings: Vec<Binding<'_>> = inputs
            .iter()
            .map(|binding| Binding {
                buffer: binding.buffer,
                kind: binding.kind,
                element: binding.element,
                bytes: binding.bytes,
            })
            .collect();
        bindings.push(Binding {
            buffer: &scratch,
            kind: KernelBindingKindV1::StorageReadWrite,
            element: KernelElementTypeV1::F32,
            bytes: output_bytes,
        });
        if uses_status {
            bindings.push(status_binding(&status));
        }
        let artifact = self.artifact(artifact_id, policy, source, &bindings, workgroup_size)?;

        let mut validators = Vec::new();
        let mut validated: Vec<&wgpu::Buffer> =
            validate.iter().map(|buffer| buffer.raw()).collect();
        if seed_from_output {
            validated.push(output.raw());
        }
        let validate_source = validate_source();
        for buffer in &validated {
            let bytes = buffer.size();
            let validator_bindings = [
                Binding {
                    buffer,
                    kind: KernelBindingKindV1::StorageReadOnly,
                    element: KernelElementTypeV1::F32,
                    bytes,
                },
                status_binding(&status),
            ];
            let validator = self.artifact(
                "nnis.wgpu.f32.validate_finite",
                WGSL_F32_BITWISE_NUMERICAL_POLICY,
                &validate_source,
                &validator_bindings,
                [WORKGROUP, 1, 1],
            )?;
            let groups = self.groups((bytes / 4) as usize, WORKGROUP)?;
            validators.push((validator, validator_bindings, groups));
        }

        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(artifact_id),
        });
        if seed_from_output {
            encoder.copy_buffer_to_buffer(output.raw(), 0, &scratch, 0, output_bytes);
        }
        for (validator, validator_bindings, groups) in &validators {
            self.dispatch(&mut encoder, validator, validator_bindings, *groups);
        }
        self.dispatch(&mut encoder, &artifact, &bindings, groups);
        self.queue.queue().submit(Some(encoder.finish()));
        pop_scopes(device, "WGPU F32 execution")?;

        let flag = self.queue.read_wgpu(&status, 0, 4)?;
        if flag != [0, 0, 0, 0] {
            return Err(PortableError::InvalidDescriptor(
                "WGPU F32 input or arithmetic intermediate is non-finite",
            ));
        }
        let mut commit = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nnis.wgpu.f32.commit"),
        });
        commit.copy_buffer_to_buffer(&scratch, 0, output.raw(), 0, output_bytes);
        nnis_core::PortableFence::wait(&self.queue.submit_checked(commit)?)?;
        Ok(WgpuF32ReportV1 {
            schema_version: WGSL_F32_CONTRACT_VERSION,
            operation,
            output_values: (output_bytes / 4) as usize,
            artifact_fingerprint: *artifact.artifact_fingerprint(),
        })
    }

    pub(crate) fn artifact(
        &self,
        artifact_id: &str,
        policy: &str,
        source: &str,
        bindings: &[Binding<'_>],
        workgroup_size: [u32; 3],
    ) -> Result<KernelArtifactV1> {
        let artifact = KernelArtifactV1::new(KernelArtifactFieldsV1 {
            artifact_id: artifact_id.into(),
            artifact_revision: 1,
            source_kind: KernelSourceKindV1::Wgsl,
            source: source.as_bytes().to_vec(),
            entry_point: "main".into(),
            bindings: bindings
                .iter()
                .enumerate()
                .map(|(index, binding)| KernelBindingV1 {
                    group: 0,
                    binding: index as u32,
                    kind: binding.kind,
                    element: binding.element,
                    min_size_bytes: binding.bytes,
                })
                .collect(),
            workgroup_size,
            numerical_policy: policy.into(),
            qualification_evidence_sha256: None,
        })
        .map_err(|error| PortableError::Unsupported(format!("kernel artifact: {error}")))?;
        let expected = *artifact.artifact_fingerprint();
        self.device
            .bind(&artifact, &expected)
            .map_err(|error| PortableError::Unsupported(error.to_string()))?;
        Ok(artifact)
    }

    pub(crate) fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        artifact: &KernelArtifactV1,
        bindings: &[Binding<'_>],
        groups: u32,
    ) {
        let device = self.queue.device();
        let source = String::from_utf8_lossy(&artifact.fields().source);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&artifact.fields().artifact_id),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source.into_owned())),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&artifact.fields().artifact_id),
            layout: None,
            module: &module,
            entry_point: Some(&artifact.fields().entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
        let entries: Vec<wgpu::BindGroupEntry<'_>> = bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: binding.buffer.as_entire_binding(),
            })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&artifact.fields().artifact_id),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&artifact.fields().artifact_id),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

fn validate_source() -> String {
    format!(
        "{COMMON}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> status: atomic<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= arrayLength(&input)) {{ return; }}
    if (!nnis_finite(input[i])) {{ atomicOr(&status, 1u); }}
}}
"
    )
}

fn read(buffer: &WgpuBuffer) -> Binding<'_> {
    Binding {
        buffer: buffer.raw(),
        kind: KernelBindingKindV1::StorageReadOnly,
        element: KernelElementTypeV1::F32,
        bytes: buffer.len(),
    }
}

fn status_binding(buffer: &wgpu::Buffer) -> Binding<'_> {
    Binding {
        buffer,
        kind: KernelBindingKindV1::StorageReadWrite,
        element: KernelElementTypeV1::U32,
        bytes: 4,
    }
}

fn require_equal(actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(PortableError::InvalidDescriptor("WGPU F32 shape mismatch"));
    }
    Ok(())
}

fn validate_indices(indices: &[usize], limit: usize) -> Result<Vec<u32>> {
    let error = PortableError::InvalidDescriptor("WGPU F32 index list is empty or out of range");
    if indices.is_empty() {
        return Err(error);
    }
    indices
        .iter()
        .map(|&index| {
            if index < limit {
                u32::try_from(index).map_err(|_| error.clone())
            } else {
                Err(error.clone())
            }
        })
        .collect()
}

pub(crate) fn pop_scopes(device: &wgpu::Device, operation: &str) -> Result<()> {
    let validation = block_on(device.pop_error_scope());
    let out_of_memory = block_on(device.pop_error_scope());
    match validation.or(out_of_memory) {
        None => Ok(()),
        Some(error) => Err(PortableError::Backend(format!(
            "{operation} failed: {error}"
        ))),
    }
}
