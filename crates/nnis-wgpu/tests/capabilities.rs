//! Limit mapping and artifact binding without an adapter.
//!
//! Uses the WebGPU default and downlevel limit tables published by `wgpu`;
//! no device is created, so nothing here depends on the host.

use nnis_core::kernel_artifact::{
    KernelArtifactError, KernelArtifactFieldsV1, KernelArtifactV1, KernelBindingKindV1,
    KernelBindingV1, KernelCapabilityV1, KernelElementTypeV1, KernelSourceKindV1,
};
use nnis_core::{BackendFamily, BackendId, CapabilitySet};
use nnis_wgpu::wgpu;
use nnis_wgpu::{
    add_f32_artifact, capabilities_from_limits, check_wgpu_binding_limits, classify_adapter,
    WgpuAdapterClassV1, WgpuBindingLimitV1, ADD_F32_WORKGROUP_SIZE,
};

fn wgpu_backend() -> BackendId {
    BackendId::new(BackendFamily::Wgpu, "wgpu-limits-table").unwrap()
}

#[test]
fn webgpu_default_limits_map_to_capability_set() {
    let limits = wgpu::Limits::default();
    let capabilities = capabilities_from_limits(&limits, wgpu::Features::empty()).unwrap();
    assert_eq!(
        capabilities,
        CapabilitySet {
            max_buffer_bytes: u64::from(limits.max_storage_buffer_binding_size)
                .min(limits.max_buffer_size),
            max_workgroup_invocations: limits.max_compute_invocations_per_workgroup,
            max_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            max_bindings: (limits.max_storage_buffers_per_shader_stage
                + limits.max_uniform_buffers_per_shader_stage)
                .min(limits.max_bindings_per_bind_group * limits.max_bind_groups),
            supports_f16: false,
            supports_timestamps: false,
        }
    );
    let with_features = capabilities_from_limits(
        &limits,
        wgpu::Features::SHADER_F16 | wgpu::Features::TIMESTAMP_QUERY,
    )
    .unwrap();
    assert!(with_features.supports_f16 && with_features.supports_timestamps);
}

#[test]
fn zero_compute_limits_are_rejected() {
    let limits = wgpu::Limits::downlevel_webgl2_defaults();
    assert_eq!(limits.max_compute_invocations_per_workgroup, 0);
    assert!(capabilities_from_limits(&limits, wgpu::Features::empty()).is_err());
}

#[test]
fn add_artifact_binds_to_downlevel_limits_and_fails_closed_beyond_them() {
    let limits = wgpu::Limits::downlevel_defaults();
    let capabilities = capabilities_from_limits(&limits, wgpu::Features::empty()).unwrap();
    let artifact = add_f32_artifact(1024).unwrap();
    assert_eq!(
        artifact.requirements().workgroup_size,
        [ADD_F32_WORKGROUP_SIZE, 1, 1]
    );
    let bound = artifact
        .bind(
            &wgpu_backend(),
            &capabilities,
            artifact.artifact_fingerprint(),
        )
        .unwrap();
    assert_eq!(bound.capabilities, capabilities);
    check_wgpu_binding_limits(&artifact, &limits).unwrap();

    let too_large = add_f32_artifact(capabilities.max_buffer_bytes / 4 + 1).unwrap();
    assert_eq!(
        too_large.bind(
            &wgpu_backend(),
            &capabilities,
            too_large.artifact_fingerprint()
        ),
        Err(KernelArtifactError::CapabilityMissing {
            capability: KernelCapabilityV1::BufferBytes
        })
    );
    let cpu = BackendId::new(BackendFamily::Cpu, "cpu").unwrap();
    assert_eq!(
        artifact.bind(&cpu, &capabilities, artifact.artifact_fingerprint()),
        Err(KernelArtifactError::BackendFamilyMismatch)
    );
    assert_eq!(
        artifact.bind(&wgpu_backend(), &capabilities, &[0; 32]),
        Err(KernelArtifactError::FingerprintMismatch)
    );
    assert!(add_f32_artifact(0).is_err());
}

fn wgsl_artifact(bindings: Vec<KernelBindingV1>) -> KernelArtifactV1 {
    KernelArtifactV1::new(KernelArtifactFieldsV1 {
        artifact_id: "nnis.wgpu.limits-probe".into(),
        artifact_revision: 1,
        source_kind: KernelSourceKindV1::Wgsl,
        source: b"// limits probe, never compiled".to_vec(),
        entry_point: "main".into(),
        bindings,
        workgroup_size: [1, 1, 1],
        numerical_policy: "none".into(),
        qualification_evidence_sha256: None,
    })
    .unwrap()
}

fn binding(group: u32, binding: u32, kind: KernelBindingKindV1, bytes: u64) -> KernelBindingV1 {
    KernelBindingV1 {
        group,
        binding,
        kind,
        element: KernelElementTypeV1::U32,
        min_size_bytes: bytes,
    }
}

#[test]
fn wgpu_specific_binding_limits_fail_closed() {
    let limits = wgpu::Limits::downlevel_defaults();
    let rw = KernelBindingKindV1::StorageReadWrite;
    assert_eq!(
        check_wgpu_binding_limits(
            &wgsl_artifact(vec![binding(limits.max_bind_groups, 0, rw, 4)]),
            &limits
        ),
        Err(WgpuBindingLimitV1::BindGroupIndex)
    );
    assert_eq!(
        check_wgpu_binding_limits(
            &wgsl_artifact(vec![binding(0, limits.max_bindings_per_bind_group, rw, 4)]),
            &limits
        ),
        Err(WgpuBindingLimitV1::BindingIndex)
    );
    let storage: Vec<_> = (0..=limits.max_storage_buffers_per_shader_stage)
        .map(|index| binding(0, index, rw, 4))
        .collect();
    assert_eq!(
        check_wgpu_binding_limits(&wgsl_artifact(storage), &limits),
        Err(WgpuBindingLimitV1::StorageBuffersPerStage)
    );
    let oversized_uniform = u64::from(limits.max_uniform_buffer_binding_size) + 4;
    assert_eq!(
        check_wgpu_binding_limits(
            &wgsl_artifact(vec![
                binding(0, 0, KernelBindingKindV1::Uniform, oversized_uniform),
                binding(0, 1, rw, 4),
            ]),
            &limits
        ),
        Err(WgpuBindingLimitV1::UniformBindingSize)
    );
}

fn info(name: &str, device_type: wgpu::DeviceType) -> wgpu::AdapterInfo {
    wgpu::AdapterInfo {
        name: name.into(),
        vendor: 0,
        device: 0,
        device_type,
        driver: String::new(),
        driver_info: String::new(),
        backend: wgpu::Backend::Vulkan,
    }
}

#[test]
fn software_adapters_are_never_classified_as_hardware() {
    assert_eq!(
        classify_adapter(&info("any", wgpu::DeviceType::Cpu)),
        WgpuAdapterClassV1::Software
    );
    assert_eq!(
        classify_adapter(&info(
            "llvmpipe (LLVM 17.0.6, 256 bits)",
            wgpu::DeviceType::Other
        )),
        WgpuAdapterClassV1::Software
    );
    assert_eq!(
        classify_adapter(&info("SwiftShader Device", wgpu::DeviceType::IntegratedGpu)),
        WgpuAdapterClassV1::Software
    );
    assert_eq!(
        classify_adapter(&info("Some GPU", wgpu::DeviceType::DiscreteGpu)),
        WgpuAdapterClassV1::Hardware
    );
    assert_eq!(
        classify_adapter(&info("Some GPU", wgpu::DeviceType::Other)),
        WgpuAdapterClassV1::Unknown
    );
}
