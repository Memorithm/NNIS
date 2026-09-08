//! GPU-kernel frontend identity and qualification boundaries.
//!
//! NNIS historically compiles CUDA C++ source through NVRTC.  Native Rust
//! frontends can feed the same execution stack only after their artifact and
//! qualification boundaries are explicit.  This module deliberately does not
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CodeKind;

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
}
