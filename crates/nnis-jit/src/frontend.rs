//! GPU-kernel frontend identity and qualification boundaries.
//!
//! NNIS historically compiles CUDA C++ source through NVRTC. Native Rust
//! frontends can feed the same execution stack only after their artifact and
//! qualification boundaries are explicit. This module deliberately does not
//! invoke an external compiler; it records what an integration is allowed to
//! hand to the existing loader and keeps unqualified frontends fail-closed.

/// Kernel-authoring frontend that produced an executable artifact or owns its
/// runtime compilation path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelFrontend {
    /// Existing NNIS CUDA C++ -> NVRTC path.
    NvrtcCudaCpp,
    /// Native Rust SIMT frontend producing PTX before NNIS module loading.
    CudaRustSimt,
    /// Native Rust tile frontend whose JIT/runtime owns the compiled artifact.
    CudaRustTile,
}

/// Boundary at which a frontend can currently hand work to NNIS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrontendArtifactBoundary {
    /// PTX text can be passed to the existing CUDA module loader.
    Ptx,
    /// Architecture-specific CUBIN can be passed to the existing loader.
    Cubin,
    /// The frontend retains ownership of compilation/loading and requires a
    /// dedicated adapter rather than pretending to be PTX/CUBIN.
    RuntimeManaged,
}

/// Qualification state is intentionally independent of frontend identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrontendQualification {
    /// Covered by the repository's existing compilation/load/launch tests.
    Qualified,
    /// Recognized as a research integration target but not yet qualified for
    /// production routing in NNIS.
    Experimental,
}

/// Immutable declaration used by adapters and future qualification evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelFrontendContract {
    frontend: KernelFrontend,
    boundary: FrontendArtifactBoundary,
    qualification: FrontendQualification,
}

impl KernelFrontendContract {
    /// Existing production-qualified NVRTC -> PTX route.
    pub const NVRTC_PTX: Self = Self {
        frontend: KernelFrontend::NvrtcCudaCpp,
        boundary: FrontendArtifactBoundary::Ptx,
        qualification: FrontendQualification::Qualified,
    };

    /// Existing production-qualified NVRTC -> CUBIN route.
    pub const NVRTC_CUBIN: Self = Self {
        frontend: KernelFrontend::NvrtcCudaCpp,
        boundary: FrontendArtifactBoundary::Cubin,
        qualification: FrontendQualification::Qualified,
    };

    /// Research target for native Rust SIMT frontends that emit PTX.
    pub const CUDA_RUST_SIMT_PTX: Self = Self {
        frontend: KernelFrontend::CudaRustSimt,
        boundary: FrontendArtifactBoundary::Ptx,
        qualification: FrontendQualification::Experimental,
    };

    /// Research target for tile-based Rust frontends whose JIT owns loading.
    pub const CUDA_RUST_TILE_RUNTIME: Self = Self {
        frontend: KernelFrontend::CudaRustTile,
        boundary: FrontendArtifactBoundary::RuntimeManaged,
        qualification: FrontendQualification::Experimental,
    };

    #[must_use]
    pub const fn frontend(self) -> KernelFrontend {
        self.frontend
    }

    #[must_use]
    pub const fn boundary(self) -> FrontendArtifactBoundary {
        self.boundary
    }

    #[must_use]
    pub const fn qualification(self) -> FrontendQualification {
        self.qualification
    }

    /// Production routing is allowed only for an explicitly qualified
    /// contract. Merely recognizing a frontend never promotes it.
    #[must_use]
    pub const fn production_routing_allowed(self) -> bool {
        matches!(self.qualification, FrontendQualification::Qualified)
    }

    /// Validate whether a byte artifact can enter NNIS' existing module-loader
    /// boundary without a dedicated frontend adapter.
    #[must_use]
    pub const fn accepts_loader_artifact(self, kind: crate::CodeKind) -> bool {
        matches!(
            (self.boundary, kind),
            (FrontendArtifactBoundary::Ptx, crate::CodeKind::Ptx)
                | (FrontendArtifactBoundary::Cubin, crate::CodeKind::Cubin)
        )
    }
}

/// Evidence collected by a qualification harness for one exact frontend
/// artifact. This record is intentionally not a capability token: successful
/// validation makes evidence reviewable but never changes a contract's
/// [`FrontendQualification`] or enables production routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontendQualificationEvidence<'a> {
    pub frontend: KernelFrontend,
    pub artifact_kind: crate::CodeKind,
    /// Immutable toolchain identity, including version/channel.
    pub toolchain_id: &'a str,
    /// Upstream toolchain source revision used to build the artifact.
    pub toolchain_commit: &'a str,
    /// Lowercase or uppercase hexadecimal SHA-256 of the exact emitted artifact.
    pub artifact_sha256: &'a str,
    /// Exact device identity used for execution evidence.
    pub device_identity: &'a str,
    /// Independent correctness oracle/reference implementation identity.
    pub oracle_id: &'a str,
    /// Stable identifier for the qualification run/evidence bundle.
    pub run_id: &'a str,
    /// Whether the candidate matched the declared correctness oracle.
    pub correction_passed: bool,
    /// Whether declared negative/fail-closed tests passed.
    pub negative_tests_passed: bool,
}

/// Fail-closed reasons for rejecting qualification evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrontendQualificationEvidenceError {
    FrontendMismatch,
    ArtifactBoundaryMismatch,
    MissingToolchainIdentity,
    MissingToolchainCommit,
    InvalidArtifactSha256,
    MissingDeviceIdentity,
    MissingOracleIdentity,
    MissingRunIdentity,
    CorrectionFailed,
    NegativeTestsFailed,
}

impl FrontendQualificationEvidence<'_> {
    /// Validate that an evidence record is complete and bound to the exact
    /// frontend contract/artifact boundary under review.
    ///
    /// A successful result does **not** promote an experimental frontend. The
    /// repository must still change the immutable frontend contract in a
    /// separately reviewed change after destination-owned qualification gates
    /// are satisfied.
    pub fn validate_for(
        &self,
        contract: KernelFrontendContract,
    ) -> Result<(), FrontendQualificationEvidenceError> {
        if self.frontend != contract.frontend() {
            return Err(FrontendQualificationEvidenceError::FrontendMismatch);
        }
        if !contract.accepts_loader_artifact(self.artifact_kind) {
            return Err(FrontendQualificationEvidenceError::ArtifactBoundaryMismatch);
        }
        if self.toolchain_id.trim().is_empty() {
            return Err(FrontendQualificationEvidenceError::MissingToolchainIdentity);
        }
        if self.toolchain_commit.trim().is_empty() {
            return Err(FrontendQualificationEvidenceError::MissingToolchainCommit);
        }
        if !is_sha256_hex(self.artifact_sha256) {
            return Err(FrontendQualificationEvidenceError::InvalidArtifactSha256);
        }
        if self.device_identity.trim().is_empty() {
            return Err(FrontendQualificationEvidenceError::MissingDeviceIdentity);
        }
        if self.oracle_id.trim().is_empty() {
            return Err(FrontendQualificationEvidenceError::MissingOracleIdentity);
        }
        if self.run_id.trim().is_empty() {
            return Err(FrontendQualificationEvidenceError::MissingRunIdentity);
        }
        if !self.correction_passed {
            return Err(FrontendQualificationEvidenceError::CorrectionFailed);
        }
        if !self.negative_tests_passed {
            return Err(FrontendQualificationEvidenceError::NegativeTestsFailed);
        }
        Ok(())
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CodeKind;

    fn valid_simt_evidence() -> FrontendQualificationEvidence<'static> {
        FrontendQualificationEvidence {
            frontend: KernelFrontend::CudaRustSimt,
            artifact_kind: CodeKind::Ptx,
            toolchain_id: "cuda-oxide/nightly-2026-04-03",
            toolchain_commit: "0123456789abcdef",
            artifact_sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            device_identity: "gpu-uuid-and-sm-version",
            oracle_id: "nnis-vector-add-reference-v1",
            run_id: "qualification-run-001",
            correction_passed: true,
            negative_tests_passed: true,
        }
    }

    #[test]
    fn existing_nvrtc_paths_remain_qualified() {
        assert!(KernelFrontendContract::NVRTC_PTX.production_routing_allowed());
        assert!(KernelFrontendContract::NVRTC_PTX.accepts_loader_artifact(CodeKind::Ptx));
        assert!(!KernelFrontendContract::NVRTC_PTX.accepts_loader_artifact(CodeKind::Cubin));

        assert!(KernelFrontendContract::NVRTC_CUBIN.production_routing_allowed());
        assert!(KernelFrontendContract::NVRTC_CUBIN.accepts_loader_artifact(CodeKind::Cubin));
    }

    #[test]
    fn native_rust_frontends_are_fail_closed_until_qualified() {
        for contract in [
            KernelFrontendContract::CUDA_RUST_SIMT_PTX,
            KernelFrontendContract::CUDA_RUST_TILE_RUNTIME,
        ] {
            assert_eq!(
                contract.qualification(),
                FrontendQualification::Experimental
            );
            assert!(!contract.production_routing_allowed());
        }
    }

    #[test]
    fn simt_and_tile_artifact_boundaries_cannot_be_conflated() {
        let simt = KernelFrontendContract::CUDA_RUST_SIMT_PTX;
        assert!(simt.accepts_loader_artifact(CodeKind::Ptx));
        assert!(!simt.accepts_loader_artifact(CodeKind::Cubin));

        let tile = KernelFrontendContract::CUDA_RUST_TILE_RUNTIME;
        assert_eq!(tile.boundary(), FrontendArtifactBoundary::RuntimeManaged);
        assert!(!tile.accepts_loader_artifact(CodeKind::Ptx));
        assert!(!tile.accepts_loader_artifact(CodeKind::Cubin));
    }

    #[test]
    fn qualification_evidence_binds_exact_frontend_and_artifact() {
        let evidence = valid_simt_evidence();
        assert_eq!(
            evidence.validate_for(KernelFrontendContract::CUDA_RUST_SIMT_PTX),
            Ok(())
        );
        assert!(!KernelFrontendContract::CUDA_RUST_SIMT_PTX.production_routing_allowed());

        let mut wrong_frontend = evidence;
        wrong_frontend.frontend = KernelFrontend::CudaRustTile;
        assert_eq!(
            wrong_frontend.validate_for(KernelFrontendContract::CUDA_RUST_SIMT_PTX),
            Err(FrontendQualificationEvidenceError::FrontendMismatch)
        );

        let mut wrong_artifact = evidence;
        wrong_artifact.artifact_kind = CodeKind::Cubin;
        assert_eq!(
            wrong_artifact.validate_for(KernelFrontendContract::CUDA_RUST_SIMT_PTX),
            Err(FrontendQualificationEvidenceError::ArtifactBoundaryMismatch)
        );
    }

    #[test]
    fn qualification_evidence_rejects_incomplete_or_failed_runs() {
        let evidence = valid_simt_evidence();

        let mut bad_hash = evidence;
        bad_hash.artifact_sha256 = "not-a-sha256";
        assert_eq!(
            bad_hash.validate_for(KernelFrontendContract::CUDA_RUST_SIMT_PTX),
            Err(FrontendQualificationEvidenceError::InvalidArtifactSha256)
        );

        let mut correction_failed = evidence;
        correction_failed.correction_passed = false;
        assert_eq!(
            correction_failed.validate_for(KernelFrontendContract::CUDA_RUST_SIMT_PTX),
            Err(FrontendQualificationEvidenceError::CorrectionFailed)
        );

        let mut negative_tests_failed = evidence;
        negative_tests_failed.negative_tests_passed = false;
        assert_eq!(
            negative_tests_failed.validate_for(KernelFrontendContract::CUDA_RUST_SIMT_PTX),
            Err(FrontendQualificationEvidenceError::NegativeTestsFailed)
        );
    }
}
