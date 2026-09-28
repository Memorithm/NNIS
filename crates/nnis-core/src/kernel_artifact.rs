//! Versioned portable kernel artifact description, binding, and capability identity.
//!
//! NNIS-P4 replaces runtime-compilation assumptions with an explicit artifact:
//! source (WGSL, validated portable IR, or a named CPU built-in), entry point,
//! binding schema, workgroup geometry, capability requirements, numerical
//! policy, and SHA-256 fingerprints of the source bytes and of the canonical
//! artifact descriptor.
//!
//! Binding an artifact to a backend verifies the expected artifact fingerprint,
//! the source-kind/backend-family pairing, and every declared capability
//! requirement against the backend's [`CapabilitySet`], failing closed. Binding
//! compiles, dispatches, and executes nothing, and a bound artifact is not
//! evidence of numerical correctness on that backend.

use core::fmt;

use crate::sha256::{sha256, to_hex, Sha256};
use crate::{BackendFamily, BackendId, BufferUsages, CapabilitySet};

/// Version of the portable kernel artifact contract.
pub const NNIS_PORTABLE_KERNEL_ARTIFACT_VERSION: u32 = 1;

/// Maximum UTF-8 bytes for artifact ids, entry points, and policies.
pub const MAX_KERNEL_ARTIFACT_ID_BYTES: usize = 128;

/// Maximum bindings in one artifact.
pub const MAX_KERNEL_ARTIFACT_BINDINGS: usize = 64;

/// Maximum source bytes in one artifact.
pub const MAX_KERNEL_ARTIFACT_SOURCE_BYTES: usize = 16 * 1024 * 1024;

/// Kind of kernel source carried by an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelSourceKindV1 {
    /// UTF-8 WGSL source for the portable WGPU backend.
    Wgsl,
    /// Pre-validated portable IR bytes for the portable WGPU backend.
    PortableIr,
    /// UTF-8 name of a CPU reference built-in operation.
    CpuBuiltin,
}

impl KernelSourceKindV1 {
    /// Backend family able to consume this source kind.
    pub const fn backend_family(self) -> BackendFamily {
        match self {
            Self::Wgsl | Self::PortableIr => BackendFamily::Wgpu,
            Self::CpuBuiltin => BackendFamily::Cpu,
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::Wgsl => 1,
            Self::PortableIr => 2,
            Self::CpuBuiltin => 3,
        }
    }
}

/// Access kind of one kernel binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelBindingKindV1 {
    /// Read-only storage buffer.
    StorageReadOnly,
    /// Read-write storage buffer.
    StorageReadWrite,
    /// Uniform buffer.
    Uniform,
}

impl KernelBindingKindV1 {
    /// Buffer usage a bound buffer must carry.
    pub const fn required_usage(self) -> BufferUsages {
        match self {
            Self::StorageReadOnly | Self::StorageReadWrite => BufferUsages::STORAGE,
            Self::Uniform => BufferUsages::UNIFORM,
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::StorageReadOnly => 1,
            Self::StorageReadWrite => 2,
            Self::Uniform => 3,
        }
    }
}

/// Element type of one binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelElementTypeV1 {
    /// 32-bit float.
    F32,
    /// 16-bit float; requires backend F16 support.
    F16,
    /// 32-bit unsigned integer.
    U32,
}

impl KernelElementTypeV1 {
    /// Bytes per element.
    pub const fn bytes(self) -> u64 {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::F16 => 2,
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::F32 => 1,
            Self::F16 => 2,
            Self::U32 => 3,
        }
    }
}

/// One entry of the binding schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelBindingV1 {
    /// Bind group index.
    pub group: u32,
    /// Binding index inside the group.
    pub binding: u32,
    /// Access kind.
    pub kind: KernelBindingKindV1,
    /// Element type.
    pub element: KernelElementTypeV1,
    /// Minimum buffer size in bytes (non-zero, element aligned).
    pub min_size_bytes: u64,
}

/// Unvalidated artifact fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArtifactFieldsV1 {
    /// Artifact identifier.
    pub artifact_id: String,
    /// Artifact revision (non-zero).
    pub artifact_revision: u32,
    /// Source kind.
    pub source_kind: KernelSourceKindV1,
    /// Exact source bytes.
    pub source: Vec<u8>,
    /// Entry point identifier.
    pub entry_point: String,
    /// Binding schema, strictly increasing by `(group, binding)`.
    pub bindings: Vec<KernelBindingV1>,
    /// Workgroup geometry.
    pub workgroup_size: [u32; 3],
    /// Numerical policy identifier the kernel implements.
    pub numerical_policy: String,
    /// Optional lowercase-hex SHA-256 of qualification evidence; `None` means
    /// the artifact is unqualified.
    pub qualification_evidence_sha256: Option<String>,
}

/// Capability requirements derived from an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelRequirementsV1 {
    /// Required backend family.
    pub backend_family: BackendFamily,
    /// Workgroup geometry.
    pub workgroup_size: [u32; 3],
    /// Invocations per workgroup.
    pub workgroup_invocations: u32,
    /// Bindings required.
    pub bindings: u32,
    /// Largest minimum binding size.
    pub max_binding_bytes: u64,
    /// F16 support required.
    pub requires_f16: bool,
}

/// Validated, immutable portable kernel artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArtifactV1 {
    fields: KernelArtifactFieldsV1,
    requirements: KernelRequirementsV1,
    source_sha256: [u8; 32],
    artifact_fingerprint: [u8; 32],
}

impl KernelArtifactV1 {
    /// Validate artifact fields and compute its fingerprints.
    pub fn new(fields: KernelArtifactFieldsV1) -> Result<Self, KernelArtifactError> {
        validate_id("artifact_id", &fields.artifact_id)?;
        if fields.artifact_revision == 0 {
            return Err(KernelArtifactError::ZeroArtifactRevision);
        }
        if fields.source.is_empty() || fields.source.len() > MAX_KERNEL_ARTIFACT_SOURCE_BYTES {
            return Err(KernelArtifactError::InvalidSourceSize {
                bytes: fields.source.len(),
            });
        }
        if matches!(
            fields.source_kind,
            KernelSourceKindV1::Wgsl | KernelSourceKindV1::CpuBuiltin
        ) && core::str::from_utf8(&fields.source).is_err()
        {
            return Err(KernelArtifactError::SourceNotUtf8);
        }
        validate_entry_point(&fields.entry_point)?;
        validate_id("numerical_policy", &fields.numerical_policy)?;
        if let Some(evidence) = &fields.qualification_evidence_sha256 {
            if evidence.len() != 64
                || !evidence
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(KernelArtifactError::InvalidEvidenceFingerprint);
            }
        }
        if fields.bindings.is_empty() || fields.bindings.len() > MAX_KERNEL_ARTIFACT_BINDINGS {
            return Err(KernelArtifactError::InvalidBindingCount {
                bindings: fields.bindings.len(),
            });
        }
        let mut max_binding_bytes = 0u64;
        let mut requires_f16 = false;
        for (index, binding) in fields.bindings.iter().enumerate() {
            if index > 0 {
                let previous = fields.bindings[index - 1];
                if (previous.group, previous.binding) >= (binding.group, binding.binding) {
                    return Err(KernelArtifactError::UnorderedOrDuplicateBinding {
                        group: binding.group,
                        binding: binding.binding,
                    });
                }
            }
            if binding.min_size_bytes == 0 || binding.min_size_bytes % binding.element.bytes() != 0
            {
                return Err(KernelArtifactError::InvalidBindingSize {
                    group: binding.group,
                    binding: binding.binding,
                });
            }
            max_binding_bytes = max_binding_bytes.max(binding.min_size_bytes);
            requires_f16 |= binding.element == KernelElementTypeV1::F16;
        }
        if !fields
            .bindings
            .iter()
            .any(|binding| binding.kind == KernelBindingKindV1::StorageReadWrite)
        {
            return Err(KernelArtifactError::NoWritableBinding);
        }
        if fields.workgroup_size.contains(&0) {
            return Err(KernelArtifactError::InvalidWorkgroupSize);
        }
        let workgroup_invocations = fields
            .workgroup_size
            .iter()
            .try_fold(1u32, |product, &dimension| product.checked_mul(dimension))
            .ok_or(KernelArtifactError::InvalidWorkgroupSize)?;
        let requirements = KernelRequirementsV1 {
            backend_family: fields.source_kind.backend_family(),
            workgroup_size: fields.workgroup_size,
            workgroup_invocations,
            bindings: fields.bindings.len() as u32,
            max_binding_bytes,
            requires_f16,
        };
        let source_sha256 = sha256(&fields.source);
        let artifact_fingerprint = descriptor_fingerprint(&fields, &source_sha256);
        Ok(Self {
            fields,
            requirements,
            source_sha256,
            artifact_fingerprint,
        })
    }

    /// Validated fields.
    pub const fn fields(&self) -> &KernelArtifactFieldsV1 {
        &self.fields
    }

    /// Derived capability requirements.
    pub const fn requirements(&self) -> &KernelRequirementsV1 {
        &self.requirements
    }

    /// SHA-256 of the exact source bytes.
    pub const fn source_sha256(&self) -> &[u8; 32] {
        &self.source_sha256
    }

    /// SHA-256 of the canonical v1 descriptor encoding (includes the source
    /// digest and every other field).
    pub const fn artifact_fingerprint(&self) -> &[u8; 32] {
        &self.artifact_fingerprint
    }

    /// Lowercase-hex artifact fingerprint.
    pub fn artifact_fingerprint_hex(&self) -> String {
        to_hex(&self.artifact_fingerprint)
    }

    /// Whether qualification evidence is referenced (not whether it is valid).
    pub fn references_qualification_evidence(&self) -> bool {
        self.fields.qualification_evidence_sha256.is_some()
    }

    /// Check every derived requirement against backend capabilities.
    pub fn check_capabilities(
        &self,
        capabilities: &CapabilitySet,
    ) -> Result<(), KernelArtifactError> {
        let required = &self.requirements;
        for axis in 0..3 {
            if required.workgroup_size[axis] > capabilities.max_workgroup_size[axis] {
                return Err(KernelArtifactError::CapabilityMissing {
                    capability: KernelCapabilityV1::WorkgroupSize { axis: axis as u8 },
                });
            }
        }
        if required.workgroup_invocations > capabilities.max_workgroup_invocations {
            return Err(KernelArtifactError::CapabilityMissing {
                capability: KernelCapabilityV1::WorkgroupInvocations,
            });
        }
        if required.bindings > capabilities.max_bindings {
            return Err(KernelArtifactError::CapabilityMissing {
                capability: KernelCapabilityV1::Bindings,
            });
        }
        if required.max_binding_bytes > capabilities.max_buffer_bytes {
            return Err(KernelArtifactError::CapabilityMissing {
                capability: KernelCapabilityV1::BufferBytes,
            });
        }
        if required.requires_f16 && !capabilities.supports_f16 {
            return Err(KernelArtifactError::CapabilityMissing {
                capability: KernelCapabilityV1::F16,
            });
        }
        Ok(())
    }

    /// Bind this artifact to one backend instance, fail closed.
    ///
    /// Requires the caller's expected artifact fingerprint, a backend of the
    /// family the source kind targets, and satisfied capabilities.
    pub fn bind(
        &self,
        backend: &BackendId,
        capabilities: &CapabilitySet,
        expected_fingerprint: &[u8; 32],
    ) -> Result<BoundKernelArtifactV1, KernelArtifactError> {
        if &self.artifact_fingerprint != expected_fingerprint {
            return Err(KernelArtifactError::FingerprintMismatch);
        }
        if backend.family() != self.requirements.backend_family {
            return Err(KernelArtifactError::BackendFamilyMismatch);
        }
        self.check_capabilities(capabilities)?;
        Ok(BoundKernelArtifactV1 {
            artifact_id: self.fields.artifact_id.clone(),
            artifact_revision: self.fields.artifact_revision,
            artifact_fingerprint: self.artifact_fingerprint,
            backend: backend.clone(),
            capabilities: *capabilities,
        })
    }
}

/// Artifact identity bound to one backend and the capabilities it was checked
/// against. Not a correctness or performance result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundKernelArtifactV1 {
    /// Artifact identifier.
    pub artifact_id: String,
    /// Artifact revision.
    pub artifact_revision: u32,
    /// Verified artifact fingerprint.
    pub artifact_fingerprint: [u8; 32],
    /// Backend instance.
    pub backend: BackendId,
    /// Capabilities the requirements were checked against.
    pub capabilities: CapabilitySet,
}

/// Capability named by a failed requirement check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelCapabilityV1 {
    /// Workgroup size on one axis.
    WorkgroupSize { axis: u8 },
    /// Invocations per workgroup.
    WorkgroupInvocations,
    /// Number of bindings.
    Bindings,
    /// Per-buffer byte limit.
    BufferBytes,
    /// F16 support.
    F16,
}

/// Fail-closed portable kernel artifact errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelArtifactError {
    /// A required identifier was empty.
    EmptyField { field: &'static str },
    /// An identifier was non-canonical or too long.
    InvalidField { field: &'static str },
    /// Artifact revision was zero.
    ZeroArtifactRevision,
    /// Source was empty or exceeded the bound.
    InvalidSourceSize { bytes: usize },
    /// WGSL or CPU built-in source was not UTF-8.
    SourceNotUtf8,
    /// Entry point was not an ASCII identifier.
    InvalidEntryPoint,
    /// Qualification evidence fingerprint was not 64 lowercase hex digits.
    InvalidEvidenceFingerprint,
    /// Binding count was zero or exceeded the bound.
    InvalidBindingCount { bindings: usize },
    /// Bindings were not strictly increasing by `(group, binding)`.
    UnorderedOrDuplicateBinding { group: u32, binding: u32 },
    /// A binding size was zero or not element aligned.
    InvalidBindingSize { group: u32, binding: u32 },
    /// No read-write storage binding was declared.
    NoWritableBinding,
    /// Workgroup size had a zero dimension or overflowed.
    InvalidWorkgroupSize,
    /// Backend lacked a required capability.
    CapabilityMissing { capability: KernelCapabilityV1 },
    /// Expected fingerprint differed from the artifact.
    FingerprintMismatch,
    /// Backend family cannot consume the source kind.
    BackendFamilyMismatch,
}

impl fmt::Display for KernelArtifactError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField { field } => write!(output, "{field} must not be empty"),
            Self::InvalidField { field } => write!(output, "{field} is non-canonical or too long"),
            Self::ZeroArtifactRevision => output.write_str("artifact revision must be non-zero"),
            Self::InvalidSourceSize { bytes } => write!(
                output,
                "artifact source has {bytes} bytes, expected 1..={MAX_KERNEL_ARTIFACT_SOURCE_BYTES}"
            ),
            Self::SourceNotUtf8 => output.write_str("artifact source must be UTF-8"),
            Self::InvalidEntryPoint => output.write_str("entry point must be an ASCII identifier"),
            Self::InvalidEvidenceFingerprint => output
                .write_str("qualification evidence fingerprint must be 64 lowercase hex digits"),
            Self::InvalidBindingCount { bindings } => write!(
                output,
                "artifact declares {bindings} bindings, expected 1..={MAX_KERNEL_ARTIFACT_BINDINGS}"
            ),
            Self::UnorderedOrDuplicateBinding { group, binding } => write!(
                output,
                "binding ({group}, {binding}) is out of order or duplicated"
            ),
            Self::InvalidBindingSize { group, binding } => write!(
                output,
                "binding ({group}, {binding}) size must be non-zero and element aligned"
            ),
            Self::NoWritableBinding => {
                output.write_str("artifact needs a read-write storage binding")
            }
            Self::InvalidWorkgroupSize => output.write_str("workgroup size is zero or overflows"),
            Self::CapabilityMissing { capability } => {
                write!(output, "backend lacks required capability {capability:?}")
            }
            Self::FingerprintMismatch => {
                output.write_str("artifact fingerprint does not match expectation")
            }
            Self::BackendFamilyMismatch => {
                output.write_str("backend family cannot consume this artifact source kind")
            }
        }
    }
}

impl std::error::Error for KernelArtifactError {}

fn validate_id(field: &'static str, value: &str) -> Result<(), KernelArtifactError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(KernelArtifactError::EmptyField { field });
    }
    if trimmed != value || value.len() > MAX_KERNEL_ARTIFACT_ID_BYTES {
        return Err(KernelArtifactError::InvalidField { field });
    }
    Ok(())
}

fn validate_entry_point(value: &str) -> Result<(), KernelArtifactError> {
    let mut bytes = value.bytes();
    let valid_first =
        matches!(bytes.next(), Some(byte) if byte.is_ascii_alphabetic() || byte == b'_');
    if !valid_first
        || value.len() > MAX_KERNEL_ARTIFACT_ID_BYTES
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(KernelArtifactError::InvalidEntryPoint);
    }
    Ok(())
}

/// Canonical v1 descriptor: domain tag, then fixed-order fields with
/// little-endian integers and u32 length-prefixed strings.
fn descriptor_fingerprint(fields: &KernelArtifactFieldsV1, source_sha256: &[u8; 32]) -> [u8; 32] {
    fn text(hasher: &mut Sha256, value: &str) {
        hasher.update(&(value.len() as u32).to_le_bytes());
        hasher.update(value.as_bytes());
    }
    let mut hasher = Sha256::new();
    hasher.update(b"nnis.portable-kernel-artifact.v1\0");
    hasher.update(&NNIS_PORTABLE_KERNEL_ARTIFACT_VERSION.to_le_bytes());
    text(&mut hasher, &fields.artifact_id);
    hasher.update(&fields.artifact_revision.to_le_bytes());
    hasher.update(&[fields.source_kind.tag()]);
    hasher.update(&(fields.source.len() as u64).to_le_bytes());
    hasher.update(source_sha256);
    text(&mut hasher, &fields.entry_point);
    hasher.update(&(fields.bindings.len() as u32).to_le_bytes());
    for binding in &fields.bindings {
        hasher.update(&binding.group.to_le_bytes());
        hasher.update(&binding.binding.to_le_bytes());
        hasher.update(&[binding.kind.tag(), binding.element.tag()]);
        hasher.update(&binding.min_size_bytes.to_le_bytes());
    }
    for dimension in fields.workgroup_size {
        hasher.update(&dimension.to_le_bytes());
    }
    text(&mut hasher, &fields.numerical_policy);
    match &fields.qualification_evidence_sha256 {
        None => hasher.update(&[0]),
        Some(evidence) => {
            hasher.update(&[1]);
            text(&mut hasher, evidence);
        }
    }
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    const WGSL: &str = "@group(0) @binding(0) var<storage, read_write> out: array<f32>;\n\
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) { out[id.x] = 0.0; }\n";

    fn fields() -> KernelArtifactFieldsV1 {
        KernelArtifactFieldsV1 {
            artifact_id: "nnis.portable.zero-fill".into(),
            artifact_revision: 1,
            source_kind: KernelSourceKindV1::Wgsl,
            source: WGSL.as_bytes().to_vec(),
            entry_point: "main".into(),
            bindings: vec![KernelBindingV1 {
                group: 0,
                binding: 0,
                kind: KernelBindingKindV1::StorageReadWrite,
                element: KernelElementTypeV1::F32,
                min_size_bytes: 256,
            }],
            workgroup_size: [64, 1, 1],
            numerical_policy: "finite-f32-le-ordered-fma-v1".into(),
            qualification_evidence_sha256: None,
        }
    }

    fn capabilities() -> CapabilitySet {
        CapabilitySet {
            max_buffer_bytes: 1 << 20,
            max_workgroup_invocations: 256,
            max_workgroup_size: [256, 256, 64],
            max_bindings: 8,
            supports_f16: false,
            supports_timestamps: false,
        }
    }

    fn wgpu() -> BackendId {
        BackendId::new(BackendFamily::Wgpu, "declared-wgpu-adapter").unwrap()
    }

    #[test]
    fn artifact_derives_requirements_and_binds_with_exact_fingerprint() {
        let artifact = KernelArtifactV1::new(fields()).unwrap();
        assert_eq!(
            artifact.requirements(),
            &KernelRequirementsV1 {
                backend_family: BackendFamily::Wgpu,
                workgroup_size: [64, 1, 1],
                workgroup_invocations: 64,
                bindings: 1,
                max_binding_bytes: 256,
                requires_f16: false,
            }
        );
        assert_eq!(artifact.source_sha256(), &sha256(WGSL.as_bytes()));
        assert!(!artifact.references_qualification_evidence());
        let expected = *artifact.artifact_fingerprint();
        let bound = artifact.bind(&wgpu(), &capabilities(), &expected).unwrap();
        assert_eq!(bound.artifact_fingerprint, expected);
        assert_eq!(bound.backend, wgpu());
        assert_eq!(
            artifact.bind(&wgpu(), &capabilities(), &[0; 32]),
            Err(KernelArtifactError::FingerprintMismatch)
        );
    }

    #[test]
    fn fingerprint_is_stable_and_covers_every_field() {
        let base = KernelArtifactV1::new(fields()).unwrap();
        assert_eq!(
            KernelArtifactV1::new(fields())
                .unwrap()
                .artifact_fingerprint(),
            base.artifact_fingerprint()
        );
        // Pins the canonical v1 descriptor encoding.
        assert_eq!(
            base.artifact_fingerprint_hex(),
            "d5f9901548ca2d760b72cf6e05b746b3cfb1c07579cf8340711bd8db1ab21b81"
        );
        type Edit = fn(&mut KernelArtifactFieldsV1);
        let edits: [Edit; 9] = [
            |f| f.artifact_id.push('x'),
            |f| f.artifact_revision = 2,
            |f| f.source.push(b' '),
            |f| f.entry_point = "main2".into(),
            |f| f.bindings[0].min_size_bytes = 512,
            |f| f.workgroup_size = [32, 2, 1],
            |f| f.numerical_policy = "other-policy".into(),
            |f| f.qualification_evidence_sha256 = Some("0".repeat(64)),
            |f| f.source_kind = KernelSourceKindV1::PortableIr,
        ];
        for edit in edits {
            let mut changed = fields();
            edit(&mut changed);
            let changed = KernelArtifactV1::new(changed).unwrap();
            assert_ne!(changed.artifact_fingerprint(), base.artifact_fingerprint());
        }
    }

    #[test]
    fn missing_capabilities_fail_closed() {
        let artifact = KernelArtifactV1::new(fields()).unwrap();
        let expected = *artifact.artifact_fingerprint();
        type CapEdit = fn(&mut CapabilitySet);
        let cases: [(CapEdit, KernelCapabilityV1); 4] = [
            (
                |c| c.max_workgroup_size[0] = 32,
                KernelCapabilityV1::WorkgroupSize { axis: 0 },
            ),
            (
                |c| c.max_workgroup_invocations = 32,
                KernelCapabilityV1::WorkgroupInvocations,
            ),
            (
                |c| c.max_buffer_bytes = 128,
                KernelCapabilityV1::BufferBytes,
            ),
            (|c| c.max_bindings = 0, KernelCapabilityV1::Bindings),
        ];
        for (edit, capability) in cases {
            let mut caps = capabilities();
            edit(&mut caps);
            assert_eq!(
                artifact.bind(&wgpu(), &caps, &expected),
                Err(KernelArtifactError::CapabilityMissing { capability })
            );
        }
        let mut f16 = fields();
        f16.bindings[0].element = KernelElementTypeV1::F16;
        let f16 = KernelArtifactV1::new(f16).unwrap();
        assert!(f16.requirements().requires_f16);
        assert_eq!(
            f16.check_capabilities(&capabilities()),
            Err(KernelArtifactError::CapabilityMissing {
                capability: KernelCapabilityV1::F16
            })
        );
        let mut with_f16 = capabilities();
        with_f16.supports_f16 = true;
        f16.check_capabilities(&with_f16).unwrap();
    }

    #[test]
    fn source_kind_must_match_backend_family() {
        let artifact = KernelArtifactV1::new(fields()).unwrap();
        let cpu = BackendId::new(BackendFamily::Cpu, "nnis-cpu-reference").unwrap();
        assert_eq!(
            artifact.bind(&cpu, &capabilities(), artifact.artifact_fingerprint()),
            Err(KernelArtifactError::BackendFamilyMismatch)
        );
        let mut builtin = fields();
        builtin.source_kind = KernelSourceKindV1::CpuBuiltin;
        builtin.source = b"nnis-cpu.f32.zero-fill".to_vec();
        builtin.workgroup_size = [1, 1, 1];
        let builtin = KernelArtifactV1::new(builtin).unwrap();
        builtin
            .bind(&cpu, &capabilities(), builtin.artifact_fingerprint())
            .unwrap();
        assert_eq!(
            builtin.bind(&wgpu(), &capabilities(), builtin.artifact_fingerprint()),
            Err(KernelArtifactError::BackendFamilyMismatch)
        );
    }

    #[test]
    fn malformed_artifacts_fail_closed() {
        let check = |edit: fn(&mut KernelArtifactFieldsV1), expected| {
            let mut f = fields();
            edit(&mut f);
            assert_eq!(KernelArtifactV1::new(f), Err(expected));
        };
        check(
            |f| f.artifact_id.clear(),
            KernelArtifactError::EmptyField {
                field: "artifact_id",
            },
        );
        check(
            |f| f.artifact_revision = 0,
            KernelArtifactError::ZeroArtifactRevision,
        );
        check(
            |f| f.source.clear(),
            KernelArtifactError::InvalidSourceSize { bytes: 0 },
        );
        check(
            |f| f.source = vec![0xFF, 0xFE],
            KernelArtifactError::SourceNotUtf8,
        );
        check(
            |f| f.entry_point = "1main".into(),
            KernelArtifactError::InvalidEntryPoint,
        );
        check(
            |f| f.entry_point = "ma-in".into(),
            KernelArtifactError::InvalidEntryPoint,
        );
        check(
            |f| f.numerical_policy = " p".into(),
            KernelArtifactError::InvalidField {
                field: "numerical_policy",
            },
        );
        check(
            |f| f.qualification_evidence_sha256 = Some("ABC".into()),
            KernelArtifactError::InvalidEvidenceFingerprint,
        );
        check(
            |f| f.bindings.clear(),
            KernelArtifactError::InvalidBindingCount { bindings: 0 },
        );
        check(
            |f| f.bindings.push(f.bindings[0]),
            KernelArtifactError::UnorderedOrDuplicateBinding {
                group: 0,
                binding: 0,
            },
        );
        check(
            |f| f.bindings[0].min_size_bytes = 6,
            KernelArtifactError::InvalidBindingSize {
                group: 0,
                binding: 0,
            },
        );
        check(
            |f| f.bindings[0].kind = KernelBindingKindV1::StorageReadOnly,
            KernelArtifactError::NoWritableBinding,
        );
        check(
            |f| f.workgroup_size = [64, 0, 1],
            KernelArtifactError::InvalidWorkgroupSize,
        );
        check(
            |f| f.workgroup_size = [u32::MAX, 2, 1],
            KernelArtifactError::InvalidWorkgroupSize,
        );
        // Portable IR need not be UTF-8.
        let mut ir = fields();
        ir.source_kind = KernelSourceKindV1::PortableIr;
        ir.source = vec![0xFF, 0x00, 0x7F];
        KernelArtifactV1::new(ir).unwrap();
        assert_eq!(
            KernelBindingKindV1::Uniform.required_usage(),
            BufferUsages::UNIFORM
        );
    }
}
