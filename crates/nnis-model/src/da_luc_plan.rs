use crate::ModelConfig;
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

/// NNIS-owned schema for the research-only DA-LUC selection/binding plan.
pub const NNIS_DA_LUC_PLAN_VERSION: u32 = 1;

/// Exact FLAT research DA-LUC view schema supported by DAL0.
pub const SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION: u16 = 1;

/// Execution policy remains separate from all existing F16/F32 attention plans.
///
/// `Default` deliberately preserves the dense reference path. Merely constructing
/// a DA-LUC candidate never changes runtime behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "policy", content = "candidate", rename_all = "snake_case")]
pub enum NnisKvExecutionPolicy {
    DenseReference,
    DaLucCandidate(NnisDalucCandidatePlan),
}

impl Default for NnisKvExecutionPolicy {
    fn default() -> Self {
        Self::DenseReference
    }
}

impl NnisKvExecutionPolicy {
    pub fn validate(
        &self,
        config: &ModelConfig,
        backend: &NnisDalucBackendCapabilities,
    ) -> Result<()> {
        match self {
            Self::DenseReference => config.validate_execution_support(),
            Self::DaLucCandidate(plan) => plan.validate(config, backend),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NnisDalucFloatDType {
    F16,
    Bf16,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NnisDalucBitOrder {
    Lsb0,
    Msb0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NnisDalucCodebookScope {
    SharedAcrossKvHeads,
    PerKvHead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NnisDalucResidualSemantics {
    None,
    SparseCoordinates {
        value_dtype: NnisDalucFloatDType,
        index_bits: u8,
        bit_order: NnisDalucBitOrder,
        max_entries_per_vector: usize,
    },
    SparseBitmap {
        value_dtype: NnisDalucFloatDType,
        bit_order: NnisDalucBitOrder,
        max_entries_per_vector: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnisDalucKeyRepresentation {
    pub subspace_dim: usize,
    pub codebook_entries: usize,
    pub codebook_dtype: NnisDalucFloatDType,
    pub codebook_scope: NnisDalucCodebookScope,
    pub index_bits: u8,
    pub index_bit_order: NnisDalucBitOrder,
    pub residual: NnisDalucResidualSemantics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NnisDalucZeroPointStorage {
    None,
    U8,
    U16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NnisDalucValueRepresentation {
    Dense {
        dtype: NnisDalucFloatDType,
    },
    GroupwiseAffine {
        storage_bits: u8,
        group_size: usize,
        scale_dtype: NnisDalucFloatDType,
        zero_point: NnisDalucZeroPointStorage,
        bit_order: NnisDalucBitOrder,
        residual: NnisDalucResidualSemantics,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnisDalucHeadGeometry {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NnisDalucRowOrder {
    BatchTokenHead,
    BatchHeadToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NnisDalucStorageTopology {
    Contiguous { capacity_tokens: usize },
    Paged {
        page_size: usize,
        physical_pages_per_batch: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NnisDalucPaddingRule {
    None,
    ZeroFilledToAlignment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnisDalucViewLayout {
    pub row_order: NnisDalucRowOrder,
    pub topology: NnisDalucStorageTopology,
    pub plane_alignment_bytes: usize,
    pub padding: NnisDalucPaddingRule,
}

/// The only DAL0 direct-consumption mode. It describes a property to validate,
/// not a performance or "zero dequantization" claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NnisDalucConsumptionMode {
    DirectCompressedNoDenseKvMaterialization,
}

/// NVIDIA-local physical realization identity.
///
/// This is intentionally *not* part of [`NnisDalucCandidatePlan::semantic_fingerprint`].
/// A future CUDA packing may change without redefining the FLAT view semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NnisDalucCudaPhysicalLayout {
    Fdal3CompatibleWordPackedV1,
}

/// NNIS DAL0 research-only binding to the FLAT DA-LUC attention-facing view.
///
/// The metadata mirrors only the fields NNIS must validate before a later CUDA
/// backend may bind storage. FLAT remains the semantic/view contract owner.
/// No payload, codebook value, residual value, token data, credential, pointer,
/// stream handle or measured performance value is represented here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnisDalucCandidatePlan {
    pub schema_version: u32,
    pub flat_view_schema_version: u16,
    pub geometry: NnisDalucHeadGeometry,
    pub keys: NnisDalucKeyRepresentation,
    pub values: NnisDalucValueRepresentation,
    pub layout: NnisDalucViewLayout,
    pub consumption: NnisDalucConsumptionMode,
    pub physical_layout: NnisDalucCudaPhysicalLayout,
}

/// Caller-supplied backend capabilities for direct compressed consumption.
///
/// DAL0 does not infer physical capability from a device name or compute
/// capability. A future runtime adapter must build this record from verified
/// backend/device support; missing capability fails closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnisDalucBackendCapabilities {
    pub direct_compressed_decode: bool,
    pub supports_contiguous: bool,
    pub supports_batch_head_token: bool,
    pub supports_zero_filled_padding: bool,
    pub supports_f32_codebooks: bool,
    pub supports_lsb0_u8_key_indices: bool,
    pub supports_groupwise_affine_u8_values: bool,
    pub supports_f32_value_scales: bool,
    pub supports_u8_zero_points: bool,
    pub maximum_head_dim: usize,
    pub maximum_lut_entries: usize,
    pub minimum_plane_alignment_bytes: usize,
}

#[derive(Serialize)]
struct SemanticIdentity<'a> {
    flat_view_schema_version: u16,
    geometry: &'a NnisDalucHeadGeometry,
    keys: &'a NnisDalucKeyRepresentation,
    values: &'a NnisDalucValueRepresentation,
    layout: &'a NnisDalucViewLayout,
    consumption: NnisDalucConsumptionMode,
}

impl NnisDalucCandidatePlan {
    /// Construct the exact narrow FLAT FDAL3-compatible metadata subset intended
    /// for the first future NNIS CUDA oracle. This is explicit opt-in only.
    pub fn fdal3_compatible_v1(
        config: &ModelConfig,
        capacity_tokens: usize,
        subspace_dim: usize,
        codebook_entries: usize,
        codebook_scope: NnisDalucCodebookScope,
        value_group_size: usize,
    ) -> Self {
        let head_dim = config.head_dim();
        Self {
            schema_version: NNIS_DA_LUC_PLAN_VERSION,
            flat_view_schema_version: SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION,
            geometry: NnisDalucHeadGeometry {
                q_heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                key_head_dim: head_dim,
                value_head_dim: head_dim,
            },
            keys: NnisDalucKeyRepresentation {
                subspace_dim,
                codebook_entries,
                codebook_dtype: NnisDalucFloatDType::F32,
                codebook_scope,
                index_bits: 8,
                index_bit_order: NnisDalucBitOrder::Lsb0,
                residual: NnisDalucResidualSemantics::None,
            },
            values: NnisDalucValueRepresentation::GroupwiseAffine {
                storage_bits: 8,
                group_size: value_group_size,
                scale_dtype: NnisDalucFloatDType::F32,
                zero_point: NnisDalucZeroPointStorage::U8,
                bit_order: NnisDalucBitOrder::Lsb0,
                residual: NnisDalucResidualSemantics::None,
            },
            layout: NnisDalucViewLayout {
                row_order: NnisDalucRowOrder::BatchHeadToken,
                topology: NnisDalucStorageTopology::Contiguous { capacity_tokens },
                plane_alignment_bytes: 4,
                padding: NnisDalucPaddingRule::ZeroFilledToAlignment,
            },
            consumption: NnisDalucConsumptionMode::DirectCompressedNoDenseKvMaterialization,
            physical_layout: NnisDalucCudaPhysicalLayout::Fdal3CompatibleWordPackedV1,
        }
    }

    pub fn validate(
        &self,
        config: &ModelConfig,
        backend: &NnisDalucBackendCapabilities,
    ) -> Result<()> {
        config.validate_execution_support()?;
        if self.schema_version != NNIS_DA_LUC_PLAN_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported NNIS DA-LUC plan schema {}; expected {}",
                self.schema_version, NNIS_DA_LUC_PLAN_VERSION
            )));
        }
        if self.flat_view_schema_version != SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported FLAT DA-LUC view schema {}; expected {}",
                self.flat_view_schema_version, SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION
            )));
        }
        self.validate_geometry(config)?;
        self.validate_fdal3_subset()?;
        self.validate_backend(backend)
    }

    fn validate_geometry(&self, config: &ModelConfig) -> Result<()> {
        let expected_head_dim = config.head_dim();
        if self.geometry.q_heads != config.num_attention_heads
            || self.geometry.kv_heads != config.num_key_value_heads
            || self.geometry.key_head_dim != expected_head_dim
            || self.geometry.value_head_dim != expected_head_dim
        {
            return Err(NnisError::invalid_input(
                "DA-LUC head geometry does not match the NNIS model configuration",
            ));
        }
        if self.geometry.q_heads == 0
            || self.geometry.kv_heads == 0
            || self.geometry.q_heads % self.geometry.kv_heads != 0
        {
            return Err(NnisError::invalid_input(
                "DA-LUC q_heads must be an integral non-zero multiple of kv_heads",
            ));
        }
        Ok(())
    }

    fn validate_fdal3_subset(&self) -> Result<()> {
        if self.geometry.key_head_dim == 0
            || self.geometry.value_head_dim == 0
            || self.geometry.key_head_dim > 128
            || self.geometry.value_head_dim > 128
        {
            return Err(NnisError::unsupported(
                "DAL0 admits only the FLAT FDAL3 v1 head-dimension subset (1..=128)",
            ));
        }

        if self.keys.subspace_dim == 0
            || self.geometry.key_head_dim % self.keys.subspace_dim != 0
        {
            return Err(NnisError::invalid_input(
                "DA-LUC K subspace_dim must exactly partition key_head_dim",
            ));
        }
        if self.keys.codebook_entries < 2 || self.keys.codebook_entries > 256 {
            return Err(NnisError::unsupported(
                "DAL0 admits only 2..=256 K codebook entries for 8-bit indices",
            ));
        }
        if self.keys.codebook_dtype != NnisDalucFloatDType::F32
            || self.keys.index_bits != 8
            || self.keys.index_bit_order != NnisDalucBitOrder::Lsb0
            || self.keys.residual != NnisDalucResidualSemantics::None
        {
            return Err(NnisError::unsupported(
                "DAL0 admits only FDAL3 v1 F32 K codebooks, 8-bit LSB0 indices and no K residual",
            ));
        }
        let subspaces = self.geometry.key_head_dim / self.keys.subspace_dim;
        let lut_entries = subspaces
            .checked_mul(self.keys.codebook_entries)
            .ok_or_else(|| NnisError::invalid_input("DA-LUC LUT size overflows usize"))?;
        if lut_entries > 2048 {
            return Err(NnisError::unsupported(
                "DAL0 admits at most 2048 query-local K LUT entries",
            ));
        }

        let NnisDalucValueRepresentation::GroupwiseAffine {
            storage_bits,
            group_size,
            scale_dtype,
            zero_point,
            bit_order,
            residual,
        } = self.values
        else {
            return Err(NnisError::unsupported(
                "DAL0 admits only FDAL3 v1 groupwise-affine V",
            ));
        };
        if storage_bits != 8
            || group_size == 0
            || self.geometry.value_head_dim % group_size != 0
            || scale_dtype != NnisDalucFloatDType::F32
            || zero_point != NnisDalucZeroPointStorage::U8
            || bit_order != NnisDalucBitOrder::Lsb0
            || residual != NnisDalucResidualSemantics::None
        {
            return Err(NnisError::unsupported(
                "DAL0 admits only 8-bit LSB0 groupwise V with F32 scales, U8 zero-points, exact grouping and no V residual",
            ));
        }

        if self.layout.row_order != NnisDalucRowOrder::BatchHeadToken
            || self.layout.padding != NnisDalucPaddingRule::ZeroFilledToAlignment
        {
            return Err(NnisError::unsupported(
                "DAL0 admits only FDAL3 v1 BatchHeadToken rows with zero-filled alignment padding",
            ));
        }
        let NnisDalucStorageTopology::Contiguous { capacity_tokens } = self.layout.topology else {
            return Err(NnisError::unsupported(
                "DAL0 does not admit paged DA-LUC storage in the first CUDA candidate",
            ));
        };
        if capacity_tokens == 0 {
            return Err(NnisError::invalid_input(
                "DA-LUC contiguous capacity_tokens must be non-zero",
            ));
        }
        if self.layout.plane_alignment_bytes < 4
            || !self.layout.plane_alignment_bytes.is_power_of_two()
        {
            return Err(NnisError::unsupported(
                "DAL0 requires power-of-two DA-LUC plane alignment of at least 4 bytes",
            ));
        }
        Ok(())
    }

    fn validate_backend(&self, backend: &NnisDalucBackendCapabilities) -> Result<()> {
        if backend.minimum_plane_alignment_bytes == 0
            || !backend.minimum_plane_alignment_bytes.is_power_of_two()
        {
            return Err(NnisError::invalid_input(
                "DA-LUC backend minimum plane alignment must be a non-zero power of two",
            ));
        }
        let required = [
            (backend.direct_compressed_decode, "direct compressed decode"),
            (backend.supports_contiguous, "contiguous storage"),
            (
                backend.supports_batch_head_token,
                "BatchHeadToken row order",
            ),
            (
                backend.supports_zero_filled_padding,
                "zero-filled alignment padding",
            ),
            (backend.supports_f32_codebooks, "F32 K codebooks"),
            (
                backend.supports_lsb0_u8_key_indices,
                "8-bit LSB0 K indices",
            ),
            (
                backend.supports_groupwise_affine_u8_values,
                "groupwise-affine U8 V",
            ),
            (backend.supports_f32_value_scales, "F32 V scales"),
            (backend.supports_u8_zero_points, "U8 V zero-points"),
        ];
        for (available, name) in required {
            if !available {
                return Err(NnisError::unsupported(format!(
                    "DA-LUC backend lacks required capability: {name}"
                )));
            }
        }
        if self.geometry.key_head_dim > backend.maximum_head_dim
            || self.geometry.value_head_dim > backend.maximum_head_dim
        {
            return Err(NnisError::unsupported(
                "DA-LUC plan head dimension exceeds backend capability",
            ));
        }
        let lut_entries = (self.geometry.key_head_dim / self.keys.subspace_dim)
            .checked_mul(self.keys.codebook_entries)
            .ok_or_else(|| NnisError::invalid_input("DA-LUC LUT size overflows usize"))?;
        if lut_entries > backend.maximum_lut_entries {
            return Err(NnisError::unsupported(
                "DA-LUC query-local LUT exceeds backend capability",
            ));
        }
        if self.layout.plane_alignment_bytes < backend.minimum_plane_alignment_bytes
            || self.layout.plane_alignment_bytes % backend.minimum_plane_alignment_bytes != 0
        {
            return Err(NnisError::unsupported(
                "DA-LUC plane alignment is incompatible with backend capability",
            ));
        }
        Ok(())
    }

    /// Deterministic JSON for evidence joins. Fixed Rust struct field order and
    /// tagged enums make the v1 encoding deterministic; unknown fields are
    /// rejected on decode.
    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|err| {
            NnisError::invalid_input(format!("failed to serialize DA-LUC plan: {err}"))
        })
    }

    /// SHA-256 of the complete NNIS plan metadata, including the NVIDIA-local
    /// physical-layout identity. This is provenance/cache identity, not auth.
    pub fn fingerprint(&self) -> Result<String> {
        Ok(sha256_hex(&self.canonical_json()?))
    }

    /// SHA-256 of only the FLAT-facing logical/view semantics plus the direct
    /// consumption requirement. The NVIDIA physical layout is deliberately
    /// excluded so implementation identity cannot redefine semantic identity.
    pub fn semantic_fingerprint(&self) -> Result<String> {
        let semantic = SemanticIdentity {
            flat_view_schema_version: self.flat_view_schema_version,
            geometry: &self.geometry,
            keys: &self.keys,
            values: &self.values,
            layout: &self.layout,
            consumption: self.consumption,
        };
        let bytes = serde_json::to_vec(&semantic).map_err(|err| {
            NnisError::invalid_input(format!(
                "failed to serialize DA-LUC semantic identity: {err}"
            ))
        })?;
        Ok(sha256_hex(&bytes))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Activation, WeightDType};

    fn config() -> ModelConfig {
        ModelConfig {
            vocab_size: 128,
            eos_token_id: Some(2),
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 256,
            rms_norm_eps: 1.0e-5,
            rope_theta: 10_000.0,
            activation: Activation::Silu,
            weight_dtype: WeightDType::F32,
        }
    }

    fn backend() -> NnisDalucBackendCapabilities {
        NnisDalucBackendCapabilities {
            direct_compressed_decode: true,
            supports_contiguous: true,
            supports_batch_head_token: true,
            supports_zero_filled_padding: true,
            supports_f32_codebooks: true,
            supports_lsb0_u8_key_indices: true,
            supports_groupwise_affine_u8_values: true,
            supports_f32_value_scales: true,
            supports_u8_zero_points: true,
            maximum_head_dim: 128,
            maximum_lut_entries: 2048,
            minimum_plane_alignment_bytes: 4,
        }
    }

    fn candidate() -> NnisDalucCandidatePlan {
        NnisDalucCandidatePlan::fdal3_compatible_v1(
            &config(),
            256,
            8,
            64,
            NnisDalucCodebookScope::PerKvHead,
            8,
        )
    }

    #[test]
    fn default_execution_policy_remains_dense_reference() {
        assert_eq!(NnisKvExecutionPolicy::default(), NnisKvExecutionPolicy::DenseReference);
        NnisKvExecutionPolicy::default()
            .validate(&config(), &backend())
            .unwrap();
    }

    #[test]
    fn exact_fdal3_compatible_subset_is_accepted() {
        let plan = candidate();
        plan.validate(&config(), &backend()).unwrap();
        assert_eq!(plan.flat_view_schema_version, 1);
        assert_eq!(plan.geometry.q_heads, 4);
        assert_eq!(plan.geometry.kv_heads, 2);
        assert_eq!(
            plan.consumption,
            NnisDalucConsumptionMode::DirectCompressedNoDenseKvMaterialization
        );
    }

    #[test]
    fn unknown_schema_and_geometry_mismatch_fail_closed() {
        let mut plan = candidate();
        plan.flat_view_schema_version += 1;
        assert!(plan.validate(&config(), &backend()).is_err());

        let mut plan = candidate();
        plan.geometry.kv_heads = 1;
        assert!(plan.validate(&config(), &backend()).is_err());
    }

    #[test]
    fn unsupported_k_v_representation_and_packing_fail_closed() {
        let mut plan = candidate();
        plan.keys.index_bits = 4;
        assert!(plan.validate(&config(), &backend()).is_err());

        let mut plan = candidate();
        plan.keys.codebook_dtype = NnisDalucFloatDType::F16;
        assert!(plan.validate(&config(), &backend()).is_err());

        let mut plan = candidate();
        plan.values = NnisDalucValueRepresentation::Dense {
            dtype: NnisDalucFloatDType::F32,
        };
        assert!(plan.validate(&config(), &backend()).is_err());

        let mut plan = candidate();
        plan.layout.topology = NnisDalucStorageTopology::Paged {
            page_size: 16,
            physical_pages_per_batch: 16,
        };
        assert!(plan.validate(&config(), &backend()).is_err());
    }

    #[test]
    fn backend_capability_mismatch_fails_closed() {
        let plan = candidate();
        let mut unsupported = backend();
        unsupported.direct_compressed_decode = false;
        assert!(plan.validate(&config(), &unsupported).is_err());

        let mut unsupported = backend();
        unsupported.maximum_lut_entries = 128;
        assert!(plan.validate(&config(), &unsupported).is_err());
    }

    #[test]
    fn canonical_serialization_and_fingerprints_are_deterministic() {
        let plan = candidate();
        assert_eq!(plan.canonical_json().unwrap(), plan.canonical_json().unwrap());
        assert_eq!(plan.fingerprint().unwrap(), plan.fingerprint().unwrap());
        assert_eq!(
            plan.semantic_fingerprint().unwrap(),
            plan.semantic_fingerprint().unwrap()
        );
        assert_eq!(plan.fingerprint().unwrap().len(), 64);
        assert_eq!(plan.semantic_fingerprint().unwrap().len(), 64);
    }

    #[test]
    fn semantic_identity_is_independent_from_nnis_plan_schema_and_physical_identity() {
        let plan = candidate();
        let semantic = plan.semantic_fingerprint().unwrap();

        let mut changed_plan_version = plan.clone();
        changed_plan_version.schema_version += 1;
        assert_eq!(changed_plan_version.semantic_fingerprint().unwrap(), semantic);
        assert_ne!(changed_plan_version.fingerprint().unwrap(), plan.fingerprint().unwrap());

        // v1 currently has one physical layout variant; prove the semantic encoder
        // omits the field by comparing against the independently serialized view.
        let semantic_bytes = serde_json::to_vec(&SemanticIdentity {
            flat_view_schema_version: plan.flat_view_schema_version,
            geometry: &plan.geometry,
            keys: &plan.keys,
            values: &plan.values,
            layout: &plan.layout,
            consumption: plan.consumption,
        })
        .unwrap();
        assert_eq!(sha256_hex(&semantic_bytes), semantic);
    }

    #[test]
    fn every_semantically_relevant_field_changes_semantic_identity() {
        let plan = candidate();
        let baseline = plan.semantic_fingerprint().unwrap();

        let mut changed = plan.clone();
        changed.keys.codebook_entries += 1;
        assert_ne!(changed.semantic_fingerprint().unwrap(), baseline);

        let mut changed = plan.clone();
        changed.geometry.q_heads = 8;
        assert_ne!(changed.semantic_fingerprint().unwrap(), baseline);

        let mut changed = plan.clone();
        changed.layout.plane_alignment_bytes = 8;
        assert_ne!(changed.semantic_fingerprint().unwrap(), baseline);
    }

    #[test]
    fn serialized_metadata_contains_no_payload_or_secret_surface() {
        let json = String::from_utf8(candidate().canonical_json().unwrap()).unwrap();
        for forbidden in [
            "token_id",
            "token_ids",
            "codebook_values",
            "residual_values",
            "cuda_pointer",
            "stream_handle",
            "credential",
            "secret",
        ] {
            assert!(!json.contains(forbidden));
        }
    }

    #[test]
    fn unknown_json_fields_fail_closed() {
        let mut value = serde_json::to_value(candidate()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("future_field".to_string(), serde_json::json!(true));
        assert!(serde_json::from_value::<NnisDalucCandidatePlan>(value).is_err());
    }
}
