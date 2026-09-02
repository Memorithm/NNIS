use crate::{
    NnisDalucCandidatePlan, NnisDalucFloatDType, NnisDalucHeadGeometry,
    NnisDalucKeyRepresentation, NnisDalucPaddingRule, NnisDalucStorageTopology,
    NnisDalucValueRepresentation, NnisDalucViewLayout,
    SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION,
};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

/// NNIS schema for evidence imported from the FLAT DA-LUC host oracle.
pub const NNIS_DA_LUC_ORACLE_EVIDENCE_VERSION: u32 = 1;
/// FLAT host-oracle payload version audited by DAL1.
pub const SUPPORTED_FLAT_DA_LUC_ORACLE_PAYLOAD_VERSION: u16 = 1;
/// Canonical upstream repository for the evidence producer.
pub const FLAT_DA_LUC_ORACLE_REPOSITORY: &str = "Memorithm/FLAT-ATTENTION";

/// Exact FLAT-facing view snapshot carried by one DAL1 evidence record.
///
/// `batch` and `kv_len` are runtime-fixture dimensions absent from the static
/// NNIS candidate plan. All remaining fields must equal the DAL0 plan exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnisDalucFlatViewSnapshot {
    pub schema_version: u16,
    pub batch: usize,
    pub kv_len: usize,
    pub geometry: NnisDalucHeadGeometry,
    pub keys: NnisDalucKeyRepresentation,
    pub values: NnisDalucValueRepresentation,
    pub layout: NnisDalucViewLayout,
}

impl NnisDalucFlatViewSnapshot {
    fn canonical_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|error| {
            NnisError::invalid_input(format!(
                "failed to serialize DA-LUC FLAT view snapshot: {error}"
            ))
        })
    }

    /// NNIS adapter identity for the imported FLAT view snapshot.
    ///
    /// This is not presented as a FLAT-native contract hash; the authoritative
    /// FLAT identity remains the exact source commit plus schema/payload versions.
    pub fn adapter_fingerprint(&self) -> Result<String> {
        Ok(sha256_hex(&self.canonical_json()?))
    }
}

/// Byte-exact storage report projected from FLAT `DalucOracleStorageReport`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnisDalucOracleStorageEvidence {
    pub logical_kv_scalar_count: usize,
    pub key_codebook_payload_bytes: usize,
    pub key_index_payload_bytes: usize,
    pub key_residual_value_payload_bytes: usize,
    pub key_residual_index_payload_bytes: usize,
    pub value_payload_bytes: usize,
    pub value_scale_payload_bytes: usize,
    pub value_zero_point_payload_bytes: usize,
    pub value_residual_value_payload_bytes: usize,
    pub value_residual_index_payload_bytes: usize,
    pub page_metadata_payload_bytes: usize,
    pub packing_tail_padding_bits: usize,
    pub alignment_padding_bytes: usize,
    pub external_metadata_bytes: usize,
    pub total_representation_bytes: usize,
    pub dense_baseline_dtype: NnisDalucFloatDType,
    pub dense_baseline_bytes: usize,
    pub effective_bits_per_value: f64,
    pub compression_ratio_against_dense: f64,
}

/// One finite reconstruction-error summary projected from the FLAT oracle.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnisDalucOracleErrorStats {
    pub samples: usize,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub rmse: f64,
}

/// K and V remain separate evidence axes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnisDalucOracleReconstructionEvidence {
    pub keys: NnisDalucOracleErrorStats,
    pub values: NnisDalucOracleErrorStats,
}

/// Versioned DAL1 evidence record imported from the FLAT-owned host oracle.
///
/// The record contains evidence and identity only. It never carries codebook,
/// index, residual, K/V payload, pointer, stream, credential, or measured
/// performance data and cannot select a runtime path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NnisDalucFlatOracleEvidence {
    pub schema_version: u32,
    pub flat_repository: String,
    pub flat_source_commit: String,
    pub flat_view_schema_version: u16,
    pub flat_oracle_payload_version: u16,
    pub nnis_plan_fingerprint: String,
    pub nnis_semantic_fingerprint: String,
    pub adapter_view_fingerprint: String,
    pub view: NnisDalucFlatViewSnapshot,
    pub storage: NnisDalucOracleStorageEvidence,
    pub reconstruction: NnisDalucOracleReconstructionEvidence,
}

impl NnisDalucFlatOracleEvidence {
    /// Validate one imported FLAT oracle report against the exact NNIS plan.
    pub fn validate_against_plan(&self, plan: &NnisDalucCandidatePlan) -> Result<()> {
        if self.schema_version != NNIS_DA_LUC_ORACLE_EVIDENCE_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported NNIS DA-LUC oracle evidence schema {}; expected {}",
                self.schema_version, NNIS_DA_LUC_ORACLE_EVIDENCE_VERSION
            )));
        }
        if self.flat_repository != FLAT_DA_LUC_ORACLE_REPOSITORY {
            return Err(NnisError::invalid_input(
                "DA-LUC oracle evidence names an unexpected producer repository",
            ));
        }
        validate_commit_sha(&self.flat_source_commit)?;
        if self.flat_view_schema_version != SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION
            || self.view.schema_version != SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION
            || plan.flat_view_schema_version != SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION
        {
            return Err(NnisError::unsupported(
                "DA-LUC oracle evidence uses an unsupported FLAT view schema",
            ));
        }
        if self.flat_oracle_payload_version != SUPPORTED_FLAT_DA_LUC_ORACLE_PAYLOAD_VERSION {
            return Err(NnisError::unsupported(format!(
                "unsupported FLAT DA-LUC oracle payload version {}; expected {}",
                self.flat_oracle_payload_version, SUPPORTED_FLAT_DA_LUC_ORACLE_PAYLOAD_VERSION
            )));
        }

        let plan_fingerprint = plan.fingerprint()?;
        let semantic_fingerprint = plan.semantic_fingerprint()?;
        if self.nnis_plan_fingerprint != plan_fingerprint {
            return Err(NnisError::invalid_input(
                "DA-LUC oracle evidence does not match the NNIS plan fingerprint",
            ));
        }
        if self.nnis_semantic_fingerprint != semantic_fingerprint {
            return Err(NnisError::invalid_input(
                "DA-LUC oracle evidence does not match the NNIS semantic fingerprint",
            ));
        }
        if self.adapter_view_fingerprint != self.view.adapter_fingerprint()? {
            return Err(NnisError::invalid_input(
                "DA-LUC oracle evidence view fingerprint is inconsistent",
            ));
        }

        self.validate_view(plan)?;
        self.validate_storage()?;
        self.validate_reconstruction()?;
        Ok(())
    }

    /// Deterministic NNIS evidence identity for provenance joins.
    pub fn fingerprint(&self) -> Result<String> {
        let canonical = serde_json::to_vec(self).map_err(|error| {
            NnisError::invalid_input(format!(
                "failed to serialize DA-LUC oracle evidence: {error}"
            ))
        })?;
        Ok(sha256_hex(&canonical))
    }

    fn validate_view(&self, plan: &NnisDalucCandidatePlan) -> Result<()> {
        if self.view.batch == 0 || self.view.kv_len == 0 {
            return Err(NnisError::invalid_input(
                "DA-LUC oracle evidence batch and kv_len must be non-zero",
            ));
        }
        if self.view.geometry != plan.geometry
            || self.view.keys != plan.keys
            || self.view.values != plan.values
            || self.view.layout != plan.layout
        {
            return Err(NnisError::invalid_input(
                "DA-LUC oracle evidence FLAT view drifts from NNIS plan semantics",
            ));
        }
        match self.view.layout.topology {
            NnisDalucStorageTopology::Contiguous { capacity_tokens } => {
                if self.view.kv_len > capacity_tokens {
                    return Err(NnisError::invalid_input(
                        "DA-LUC oracle evidence kv_len exceeds contiguous capacity",
                    ));
                }
                if self.storage.page_metadata_payload_bytes != 0 {
                    return Err(NnisError::invalid_input(
                        "contiguous DA-LUC evidence must not carry page-table bytes",
                    ));
                }
            }
            NnisDalucStorageTopology::Paged { .. } => {
                return Err(NnisError::unsupported(
                    "DAL1 currently binds only the DAL0 contiguous FLAT subset",
                ));
            }
        }
        Ok(())
    }

    fn validate_storage(&self) -> Result<()> {
        let expected_scalars = checked_mul_many(&[
            self.view.batch,
            self.view.geometry.kv_heads,
            self.view.kv_len,
            self.view
                .geometry
                .key_head_dim
                .checked_add(self.view.geometry.value_head_dim)
                .ok_or_else(|| NnisError::invalid_input("DA-LUC scalar count overflows"))?,
        ])?;
        if self.storage.logical_kv_scalar_count != expected_scalars || expected_scalars == 0 {
            return Err(NnisError::invalid_input(
                "DA-LUC exact-storage logical scalar count is inconsistent",
            ));
        }

        let payload_bytes = checked_add_many(&[
            self.storage.key_codebook_payload_bytes,
            self.storage.key_index_payload_bytes,
            self.storage.key_residual_value_payload_bytes,
            self.storage.key_residual_index_payload_bytes,
            self.storage.value_payload_bytes,
            self.storage.value_scale_payload_bytes,
            self.storage.value_zero_point_payload_bytes,
            self.storage.value_residual_value_payload_bytes,
            self.storage.value_residual_index_payload_bytes,
            self.storage.page_metadata_payload_bytes,
            self.storage.alignment_padding_bytes,
            self.storage.external_metadata_bytes,
        ])?;
        if payload_bytes != self.storage.total_representation_bytes || payload_bytes == 0 {
            return Err(NnisError::invalid_input(
                "DA-LUC exact-storage byte breakdown does not equal total representation bytes",
            ));
        }
        if self.storage.packing_tail_padding_bits > 70 {
            return Err(NnisError::invalid_input(
                "DA-LUC packing tail exceeds the ten-plane FLAT oracle bound",
            ));
        }

        let expected_dense = self.expected_dense_baseline_bytes()?;
        if self.storage.dense_baseline_bytes != expected_dense || expected_dense == 0 {
            return Err(NnisError::invalid_input(
                "DA-LUC dense baseline byte count is inconsistent with the FLAT view",
            ));
        }

        let expected_effective = self.storage.total_representation_bytes as f64 * 8.0
            / self.storage.logical_kv_scalar_count as f64;
        let expected_ratio = self.storage.dense_baseline_bytes as f64
            / self.storage.total_representation_bytes as f64;
        if !self.storage.effective_bits_per_value.is_finite()
            || !self.storage.compression_ratio_against_dense.is_finite()
            || self.storage.effective_bits_per_value.to_bits() != expected_effective.to_bits()
            || self.storage.compression_ratio_against_dense.to_bits() != expected_ratio.to_bits()
        {
            return Err(NnisError::invalid_input(
                "DA-LUC derived storage metrics are inconsistent with exact byte counts",
            ));
        }
        Ok(())
    }

    fn expected_dense_baseline_bytes(&self) -> Result<usize> {
        let capacity_tokens = match self.view.layout.topology {
            NnisDalucStorageTopology::Contiguous { capacity_tokens } => capacity_tokens,
            NnisDalucStorageTopology::Paged { .. } => {
                return Err(NnisError::unsupported(
                    "DAL1 dense-baseline verification currently supports contiguous views only",
                ));
            }
        };
        let physical_rows = checked_mul_many(&[
            self.view.batch,
            self.view.geometry.kv_heads,
            capacity_tokens,
        ])?;
        let scalar_bytes = dtype_bytes(self.storage.dense_baseline_dtype);
        let key_raw = checked_mul_many(&[
            physical_rows,
            self.view.geometry.key_head_dim,
            scalar_bytes,
        ])?;
        let value_raw = checked_mul_many(&[
            physical_rows,
            self.view.geometry.value_head_dim,
            scalar_bytes,
        ])?;
        let key_plane = padded_plane_bytes(key_raw, &self.view.layout)?;
        let value_plane = padded_plane_bytes(value_raw, &self.view.layout)?;
        checked_add_many(&[
            key_plane,
            value_plane,
            self.storage.page_metadata_payload_bytes,
            self.storage.external_metadata_bytes,
        ])
    }

    fn validate_reconstruction(&self) -> Result<()> {
        let key_samples = checked_mul_many(&[
            self.view.batch,
            self.view.geometry.kv_heads,
            self.view.kv_len,
            self.view.geometry.key_head_dim,
        ])?;
        let value_samples = checked_mul_many(&[
            self.view.batch,
            self.view.geometry.kv_heads,
            self.view.kv_len,
            self.view.geometry.value_head_dim,
        ])?;
        validate_error_stats("K", self.reconstruction.keys, key_samples)?;
        validate_error_stats("V", self.reconstruction.values, value_samples)
    }
}

fn validate_error_stats(
    side: &'static str,
    stats: NnisDalucOracleErrorStats,
    expected_samples: usize,
) -> Result<()> {
    if stats.samples != expected_samples || stats.samples == 0 {
        return Err(NnisError::invalid_input(format!(
            "DA-LUC {side} reconstruction sample count is inconsistent"
        )));
    }
    if !stats.max_abs.is_finite()
        || !stats.mean_abs.is_finite()
        || !stats.rmse.is_finite()
        || stats.max_abs < 0.0
        || stats.mean_abs < 0.0
        || stats.rmse < 0.0
        || stats.mean_abs > stats.max_abs
        || stats.rmse > stats.max_abs
    {
        return Err(NnisError::invalid_input(format!(
            "DA-LUC {side} reconstruction statistics are malformed"
        )));
    }
    Ok(())
}

fn validate_commit_sha(value: &str) -> Result<()> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NnisError::invalid_input(
            "DA-LUC FLAT source commit must be a lowercase 40-character SHA-1",
        ));
    }
    Ok(())
}

fn dtype_bytes(dtype: NnisDalucFloatDType) -> usize {
    match dtype {
        NnisDalucFloatDType::F16 | NnisDalucFloatDType::Bf16 => 2,
        NnisDalucFloatDType::F32 => 4,
    }
}

fn padded_plane_bytes(logical_bytes: usize, layout: &NnisDalucViewLayout) -> Result<usize> {
    match layout.padding {
        NnisDalucPaddingRule::None => Ok(logical_bytes),
        NnisDalucPaddingRule::ZeroFilledToAlignment => {
            let alignment = layout.plane_alignment_bytes;
            if alignment == 0 || !alignment.is_power_of_two() {
                return Err(NnisError::invalid_input(
                    "DA-LUC evidence carries invalid plane alignment",
                ));
            }
            if logical_bytes == 0 {
                return Ok(0);
            }
            let mask = alignment - 1;
            logical_bytes
                .checked_add(mask)
                .map(|value| value & !mask)
                .ok_or_else(|| NnisError::invalid_input("DA-LUC alignment arithmetic overflows"))
        }
    }
}

fn checked_mul_many(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, value| {
        acc.checked_mul(*value)
            .ok_or_else(|| NnisError::invalid_input("DA-LUC evidence arithmetic overflows"))
    })
}

fn checked_add_many(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |acc, value| {
        acc.checked_add(*value)
            .ok_or_else(|| NnisError::invalid_input("DA-LUC evidence arithmetic overflows"))
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NnisDalucBitOrder, NnisDalucCodebookScope, NnisDalucConsumptionMode,
        NnisDalucCudaPhysicalLayout, NnisDalucResidualSemantics, NnisDalucRowOrder,
        NnisDalucZeroPointStorage, NNIS_DA_LUC_PLAN_VERSION,
    };

    const FLAT_HEAD: &str = "c35b044c5324963a300ff50da0f7ec10dcc6db71";

    fn plan() -> NnisDalucCandidatePlan {
        NnisDalucCandidatePlan {
            schema_version: NNIS_DA_LUC_PLAN_VERSION,
            flat_view_schema_version: SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION,
            geometry: NnisDalucHeadGeometry {
                q_heads: 2,
                kv_heads: 1,
                key_head_dim: 4,
                value_head_dim: 4,
            },
            keys: NnisDalucKeyRepresentation {
                subspace_dim: 2,
                codebook_entries: 4,
                codebook_dtype: NnisDalucFloatDType::F32,
                codebook_scope: NnisDalucCodebookScope::SharedAcrossKvHeads,
                index_bits: 8,
                index_bit_order: NnisDalucBitOrder::Lsb0,
                residual: NnisDalucResidualSemantics::None,
            },
            values: NnisDalucValueRepresentation::GroupwiseAffine {
                storage_bits: 8,
                group_size: 2,
                scale_dtype: NnisDalucFloatDType::F32,
                zero_point: NnisDalucZeroPointStorage::U8,
                bit_order: NnisDalucBitOrder::Lsb0,
                residual: NnisDalucResidualSemantics::None,
            },
            layout: NnisDalucViewLayout {
                row_order: NnisDalucRowOrder::BatchHeadToken,
                topology: NnisDalucStorageTopology::Contiguous { capacity_tokens: 4 },
                plane_alignment_bytes: 4,
                padding: NnisDalucPaddingRule::ZeroFilledToAlignment,
            },
            consumption: NnisDalucConsumptionMode::DirectCompressedNoDenseKvMaterialization,
            physical_layout: NnisDalucCudaPhysicalLayout::Fdal3CompatibleWordPackedV1,
        }
    }

    fn evidence(plan: &NnisDalucCandidatePlan) -> NnisDalucFlatOracleEvidence {
        let view = NnisDalucFlatViewSnapshot {
            schema_version: SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION,
            batch: 1,
            kv_len: 2,
            geometry: plan.geometry,
            keys: plan.keys,
            values: plan.values,
            layout: plan.layout,
        };
        let total = 144usize;
        let scalars = 16usize;
        NnisDalucFlatOracleEvidence {
            schema_version: NNIS_DA_LUC_ORACLE_EVIDENCE_VERSION,
            flat_repository: FLAT_DA_LUC_ORACLE_REPOSITORY.to_owned(),
            flat_source_commit: FLAT_HEAD.to_owned(),
            flat_view_schema_version: SUPPORTED_FLAT_DA_LUC_VIEW_SCHEMA_VERSION,
            flat_oracle_payload_version: SUPPORTED_FLAT_DA_LUC_ORACLE_PAYLOAD_VERSION,
            nnis_plan_fingerprint: plan.fingerprint().unwrap(),
            nnis_semantic_fingerprint: plan.semantic_fingerprint().unwrap(),
            adapter_view_fingerprint: view.adapter_fingerprint().unwrap(),
            view,
            storage: NnisDalucOracleStorageEvidence {
                logical_kv_scalar_count: scalars,
                key_codebook_payload_bytes: 64,
                key_index_payload_bytes: 8,
                key_residual_value_payload_bytes: 0,
                key_residual_index_payload_bytes: 0,
                value_payload_bytes: 16,
                value_scale_payload_bytes: 32,
                value_zero_point_payload_bytes: 8,
                value_residual_value_payload_bytes: 0,
                value_residual_index_payload_bytes: 0,
                page_metadata_payload_bytes: 0,
                packing_tail_padding_bits: 0,
                alignment_padding_bytes: 0,
                external_metadata_bytes: 16,
                total_representation_bytes: total,
                dense_baseline_dtype: NnisDalucFloatDType::F32,
                dense_baseline_bytes: 144,
                effective_bits_per_value: total as f64 * 8.0 / scalars as f64,
                compression_ratio_against_dense: 1.0,
            },
            reconstruction: NnisDalucOracleReconstructionEvidence {
                keys: NnisDalucOracleErrorStats {
                    samples: 8,
                    max_abs: 0.25,
                    mean_abs: 0.125,
                    rmse: 0.20,
                },
                values: NnisDalucOracleErrorStats {
                    samples: 8,
                    max_abs: 0.10,
                    mean_abs: 0.05,
                    rmse: 0.08,
                },
            },
        }
    }

    #[test]
    fn valid_flat_oracle_evidence_binds_to_plan() {
        let plan = plan();
        let evidence = evidence(&plan);
        evidence.validate_against_plan(&plan).unwrap();
        assert_eq!(evidence.fingerprint().unwrap().len(), 64);
    }

    #[test]
    fn semantic_view_drift_is_rejected_even_with_rehashed_snapshot() {
        let plan = plan();
        let mut evidence = evidence(&plan);
        evidence.view.keys.codebook_entries += 1;
        evidence.adapter_view_fingerprint = evidence.view.adapter_fingerprint().unwrap();
        assert!(evidence.validate_against_plan(&plan).is_err());
    }

    #[test]
    fn exact_storage_tampering_is_rejected() {
        let plan = plan();
        let mut evidence = evidence(&plan);
        evidence.storage.total_representation_bytes += 1;
        assert!(evidence.validate_against_plan(&plan).is_err());

        let mut evidence = evidence(&plan);
        evidence.storage.effective_bits_per_value = 1.0;
        assert!(evidence.validate_against_plan(&plan).is_err());
    }

    #[test]
    fn source_and_oracle_version_are_fail_closed() {
        let plan = plan();
        let mut evidence = evidence(&plan);
        evidence.flat_source_commit = "not-a-commit".into();
        assert!(evidence.validate_against_plan(&plan).is_err());

        let mut evidence = evidence(&plan);
        evidence.flat_oracle_payload_version += 1;
        assert!(evidence.validate_against_plan(&plan).is_err());
    }

    #[test]
    fn non_finite_reconstruction_evidence_is_rejected() {
        let plan = plan();
        let mut evidence = evidence(&plan);
        evidence.reconstruction.keys.rmse = f64::NAN;
        assert!(evidence.validate_against_plan(&plan).is_err());
    }

    #[test]
    fn evidence_serialization_contains_no_payload_or_performance_surface() {
        let plan = plan();
        let json = serde_json::to_string(&evidence(&plan)).unwrap();
        for forbidden in [
            "codebook_values",
            "residual_values",
            "token_ids",
            "device_pointer",
            "stream_handle",
            "latency_ms",
            "tokens_per_second",
            "secret",
        ] {
            assert!(!json.contains(forbidden));
        }
    }
}
