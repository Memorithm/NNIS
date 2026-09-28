//! P4 artifact binding against the declared CPU reference capabilities.
//!
//! Binding checks identity and declared limits only; it executes nothing.

use nnis_core::kernel_artifact::{
    KernelArtifactError, KernelArtifactFieldsV1, KernelArtifactV1, KernelBindingKindV1,
    KernelBindingV1, KernelCapabilityV1, KernelElementTypeV1, KernelSourceKindV1,
};
use nnis_core::PortableDevice;
use nnis_cpu::CpuDevice;

fn builtin(bindings: Vec<KernelBindingV1>, workgroup_size: [u32; 3]) -> KernelArtifactV1 {
    KernelArtifactV1::new(KernelArtifactFieldsV1 {
        artifact_id: "nnis.cpu.relu".into(),
        artifact_revision: 1,
        source_kind: KernelSourceKindV1::CpuBuiltin,
        source: b"nnis-cpu.f32.relu".to_vec(),
        entry_point: "relu".into(),
        bindings,
        workgroup_size,
        numerical_policy: nnis_cpu::numerical::CPU_F32_NUMERICAL_POLICY.into(),
        qualification_evidence_sha256: None,
    })
    .unwrap()
}

fn storage(binding: u32, kind: KernelBindingKindV1) -> KernelBindingV1 {
    KernelBindingV1 {
        group: 0,
        binding,
        kind,
        element: KernelElementTypeV1::F32,
        min_size_bytes: 64,
    }
}

#[test]
fn single_binding_builtin_binds_to_declared_cpu_capabilities() {
    let device = CpuDevice::new().unwrap();
    let artifact = builtin(
        vec![storage(0, KernelBindingKindV1::StorageReadWrite)],
        [1, 1, 1],
    );
    let bound = artifact
        .bind(
            device.backend_id(),
            device.capabilities(),
            artifact.artifact_fingerprint(),
        )
        .unwrap();
    assert_eq!(&bound.backend, device.backend_id());
    assert_eq!(&bound.capabilities, device.capabilities());
}

#[test]
fn cpu_declared_limits_reject_wider_artifacts() {
    let device = CpuDevice::new().unwrap();
    let two_bindings = builtin(
        vec![
            storage(0, KernelBindingKindV1::StorageReadOnly),
            storage(1, KernelBindingKindV1::StorageReadWrite),
        ],
        [1, 1, 1],
    );
    assert_eq!(
        two_bindings.bind(
            device.backend_id(),
            device.capabilities(),
            two_bindings.artifact_fingerprint()
        ),
        Err(KernelArtifactError::CapabilityMissing {
            capability: KernelCapabilityV1::Bindings
        })
    );
    let wide = builtin(
        vec![storage(0, KernelBindingKindV1::StorageReadWrite)],
        [4, 1, 1],
    );
    assert_eq!(
        wide.bind(
            device.backend_id(),
            device.capabilities(),
            wide.artifact_fingerprint()
        ),
        Err(KernelArtifactError::CapabilityMissing {
            capability: KernelCapabilityV1::WorkgroupSize { axis: 0 }
        })
    );
}
