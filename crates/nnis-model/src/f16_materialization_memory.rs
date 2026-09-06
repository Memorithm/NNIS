use crate::{F16ReferenceExecutionPlan, WeightAllocationSummaryV1};
use nnis_rt::{NnisError, Result};
use serde::{Deserialize, Serialize};

pub const NNIS_F16_WEIGHT_MATERIALIZATION_MEMORY_EVIDENCE_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum F16WeightMaterializationEventKindV1 {
    AllocateResident,
    AllocateTemporary,
    ReleaseTemporary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct F16WeightMaterializationEventV1 {
    pub sequence: u32,
    pub logical_name: String,
    pub kind: F16WeightMaterializationEventKindV1,
    pub bytes: u64,
    pub live_f16_allocation_bytes: u64,
    pub live_temporary_f16_allocation_bytes: u64,
    pub scoped_owned_allocation_bytes: u64,
}

/// Exact allocation-lifetime evidence for one successful F32 -> resident-F16
/// weight materialization.
///
/// The scoped byte counts include only the source `ModelWeights` allocations,
/// which remain live throughout the materialization call, plus F16 weight and
/// conversion buffers explicitly allocated by that call. They are not physical
/// page residency, process-wide VRAM, CUDA allocator overhead, RoPE/KV/session
/// memory, kernel/module storage, or a serving-memory measurement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct F16WeightMaterializationMemoryEvidenceV1 {
    pub schema_version: u32,
    pub execution_plan: F16ReferenceExecutionPlan,
    pub source_weight_allocations: WeightAllocationSummaryV1,
    pub steady_state_f16_weight_allocations: WeightAllocationSummaryV1,
    pub peak_live_f16_allocation_bytes: u64,
    pub peak_live_temporary_f16_allocation_bytes: u64,
    pub peak_scoped_owned_allocation_bytes: u64,
    pub final_scoped_owned_allocation_bytes: u64,
    pub events: Vec<F16WeightMaterializationEventV1>,
}

#[derive(Debug)]
pub(crate) struct F16WeightMaterializationTracker {
    source_owned_allocation_bytes: u64,
    live_f16_allocation_bytes: u64,
    live_temporary_f16_allocation_bytes: u64,
    peak_live_f16_allocation_bytes: u64,
    peak_live_temporary_f16_allocation_bytes: u64,
    peak_scoped_owned_allocation_bytes: u64,
    events: Vec<F16WeightMaterializationEventV1>,
}

impl F16WeightMaterializationTracker {
    pub(crate) fn new(source_owned_allocation_bytes: u64) -> Result<Self> {
        if source_owned_allocation_bytes == 0 {
            return Err(NnisError::invalid_input(
                "F16 materialization requires a non-empty live source weight graph",
            ));
        }
        Ok(Self {
            source_owned_allocation_bytes,
            live_f16_allocation_bytes: 0,
            live_temporary_f16_allocation_bytes: 0,
            peak_live_f16_allocation_bytes: 0,
            peak_live_temporary_f16_allocation_bytes: 0,
            peak_scoped_owned_allocation_bytes: source_owned_allocation_bytes,
            events: Vec::new(),
        })
    }

    pub(crate) fn allocate_resident(&mut self, logical_name: &str, bytes: u64) -> Result<()> {
        self.allocate(
            logical_name,
            bytes,
            F16WeightMaterializationEventKindV1::AllocateResident,
            false,
        )
    }

    pub(crate) fn allocate_temporary(&mut self, logical_name: &str, bytes: u64) -> Result<()> {
        self.allocate(
            logical_name,
            bytes,
            F16WeightMaterializationEventKindV1::AllocateTemporary,
            true,
        )
    }

    fn allocate(
        &mut self,
        logical_name: &str,
        bytes: u64,
        kind: F16WeightMaterializationEventKindV1,
        temporary: bool,
    ) -> Result<()> {
        if bytes == 0 {
            return Err(NnisError::invalid_input(format!(
                "F16 materialization allocation {logical_name:?} has zero bytes"
            )));
        }
        self.live_f16_allocation_bytes = self
            .live_f16_allocation_bytes
            .checked_add(bytes)
            .ok_or_else(|| {
                NnisError::invalid_input("live F16 materialization bytes overflow u64")
            })?;
        if temporary {
            self.live_temporary_f16_allocation_bytes = self
                .live_temporary_f16_allocation_bytes
                .checked_add(bytes)
                .ok_or_else(|| NnisError::invalid_input("live temporary F16 bytes overflow u64"))?;
        }
        self.record(logical_name, kind, bytes)
    }

    pub(crate) fn release_temporary(&mut self, logical_name: &str, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Err(NnisError::invalid_input(format!(
                "F16 materialization release {logical_name:?} has zero bytes"
            )));
        }
        self.live_temporary_f16_allocation_bytes = self
            .live_temporary_f16_allocation_bytes
            .checked_sub(bytes)
            .ok_or_else(|| NnisError::invalid_input("temporary F16 release exceeds live bytes"))?;
        self.live_f16_allocation_bytes = self
            .live_f16_allocation_bytes
            .checked_sub(bytes)
            .ok_or_else(|| {
                NnisError::invalid_input("F16 release exceeds live materialization bytes")
            })?;
        self.record(
            logical_name,
            F16WeightMaterializationEventKindV1::ReleaseTemporary,
            bytes,
        )
    }

    fn record(
        &mut self,
        logical_name: &str,
        kind: F16WeightMaterializationEventKindV1,
        bytes: u64,
    ) -> Result<()> {
        let scoped = self
            .source_owned_allocation_bytes
            .checked_add(self.live_f16_allocation_bytes)
            .ok_or_else(|| NnisError::invalid_input("scoped materialization bytes overflow u64"))?;
        self.peak_live_f16_allocation_bytes = self
            .peak_live_f16_allocation_bytes
            .max(self.live_f16_allocation_bytes);
        self.peak_live_temporary_f16_allocation_bytes = self
            .peak_live_temporary_f16_allocation_bytes
            .max(self.live_temporary_f16_allocation_bytes);
        self.peak_scoped_owned_allocation_bytes =
            self.peak_scoped_owned_allocation_bytes.max(scoped);
        let sequence = u32::try_from(self.events.len())
            .map_err(|_| NnisError::invalid_input("F16 materialization event count exceeds u32"))?;
        self.events.push(F16WeightMaterializationEventV1 {
            sequence,
            logical_name: logical_name.to_string(),
            kind,
            bytes,
            live_f16_allocation_bytes: self.live_f16_allocation_bytes,
            live_temporary_f16_allocation_bytes: self.live_temporary_f16_allocation_bytes,
            scoped_owned_allocation_bytes: scoped,
        });
        Ok(())
    }

    pub(crate) fn finish(
        self,
        execution_plan: F16ReferenceExecutionPlan,
        source_weight_allocations: WeightAllocationSummaryV1,
        steady_state_f16_weight_allocations: WeightAllocationSummaryV1,
    ) -> Result<F16WeightMaterializationMemoryEvidenceV1> {
        if source_weight_allocations.owned_device_allocation_bytes
            != self.source_owned_allocation_bytes
        {
            return Err(NnisError::invalid_input(
                "source weight summary changed during F16 materialization accounting",
            ));
        }
        if self.live_temporary_f16_allocation_bytes != 0 {
            return Err(NnisError::invalid_input(
                "F16 materialization completed with temporary allocations still accounted live",
            ));
        }
        if self.live_f16_allocation_bytes
            != steady_state_f16_weight_allocations.owned_device_allocation_bytes
        {
            return Err(NnisError::invalid_input(
                "tracked final F16 bytes disagree with resident F16 weight summary",
            ));
        }
        let final_scoped_owned_allocation_bytes = self
            .source_owned_allocation_bytes
            .checked_add(self.live_f16_allocation_bytes)
            .ok_or_else(|| {
                NnisError::invalid_input("final scoped materialization bytes overflow u64")
            })?;
        if self.peak_scoped_owned_allocation_bytes < final_scoped_owned_allocation_bytes {
            return Err(NnisError::invalid_input(
                "F16 materialization peak is below final scoped allocation bytes",
            ));
        }
        Ok(F16WeightMaterializationMemoryEvidenceV1 {
            schema_version: NNIS_F16_WEIGHT_MATERIALIZATION_MEMORY_EVIDENCE_VERSION,
            execution_plan,
            source_weight_allocations,
            steady_state_f16_weight_allocations,
            peak_live_f16_allocation_bytes: self.peak_live_f16_allocation_bytes,
            peak_live_temporary_f16_allocation_bytes: self.peak_live_temporary_f16_allocation_bytes,
            peak_scoped_owned_allocation_bytes: self.peak_scoped_owned_allocation_bytes,
            final_scoped_owned_allocation_bytes,
            events: self.events,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        F16ReferencePlan, F16ReferenceProjectionLayout, WeightAllocationDTypeV1,
        WeightAllocationSegmentV1, NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION,
    };

    fn summary(bytes: u64, dtype: WeightAllocationDTypeV1) -> WeightAllocationSummaryV1 {
        WeightAllocationSummaryV1 {
            schema_version: NNIS_WEIGHT_ALLOCATION_SUMMARY_VERSION,
            logical_tensor_references: 1,
            logical_element_references: bytes,
            unique_device_allocations: 1,
            unique_device_elements: bytes,
            owned_device_allocation_bytes: bytes,
            segments: vec![WeightAllocationSegmentV1 {
                allocation_index: 0,
                dtype,
                elements: bytes,
                bytes,
                logical_names: vec!["weights".to_string()],
            }],
        }
    }

    fn plan(layout: F16ReferenceProjectionLayout) -> F16ReferenceExecutionPlan {
        F16ReferenceExecutionPlan {
            schema_version: crate::F16_REFERENCE_EXECUTION_PLAN_VERSION,
            numeric: F16ReferencePlan::edge_llm_v0_10_0_alignment(),
            projection_layout: layout,
        }
    }

    #[test]
    fn reference_materialization_peak_tracks_source_plus_live_resident_bytes() {
        let mut tracker = F16WeightMaterializationTracker::new(100).unwrap();
        tracker.allocate_resident("a", 20).unwrap();
        tracker.allocate_resident("b", 30).unwrap();
        let evidence = tracker
            .finish(
                plan(F16ReferenceProjectionLayout::KnReference),
                summary(100, WeightAllocationDTypeV1::F32),
                summary(50, WeightAllocationDTypeV1::F16),
            )
            .unwrap();
        assert_eq!(evidence.peak_live_f16_allocation_bytes, 50);
        assert_eq!(evidence.peak_live_temporary_f16_allocation_bytes, 0);
        assert_eq!(evidence.peak_scoped_owned_allocation_bytes, 150);
        assert_eq!(evidence.final_scoped_owned_allocation_bytes, 150);
    }

    #[test]
    fn transposed_materialization_records_real_temporary_overlap() {
        let mut tracker = F16WeightMaterializationTracker::new(100).unwrap();
        tracker.allocate_resident("prior", 20).unwrap();
        tracker.allocate_temporary("proj.kn_temporary", 30).unwrap();
        tracker.allocate_resident("proj", 30).unwrap();
        tracker.release_temporary("proj.kn_temporary", 30).unwrap();
        let evidence = tracker
            .finish(
                plan(F16ReferenceProjectionLayout::NkTransposedCandidate),
                summary(100, WeightAllocationDTypeV1::F32),
                summary(50, WeightAllocationDTypeV1::F16),
            )
            .unwrap();
        assert_eq!(evidence.peak_live_f16_allocation_bytes, 80);
        assert_eq!(evidence.peak_live_temporary_f16_allocation_bytes, 30);
        assert_eq!(evidence.peak_scoped_owned_allocation_bytes, 180);
        assert_eq!(evidence.final_scoped_owned_allocation_bytes, 150);
        assert_eq!(evidence.events.len(), 4);
    }

    #[test]
    fn temporary_release_underflow_fails_closed() {
        let mut tracker = F16WeightMaterializationTracker::new(100).unwrap();
        let error = tracker.release_temporary("missing", 1).unwrap_err();
        assert!(error.to_string().contains("release exceeds"));
    }

    #[test]
    fn final_resident_mismatch_fails_closed() {
        let mut tracker = F16WeightMaterializationTracker::new(100).unwrap();
        tracker.allocate_resident("a", 20).unwrap();
        let error = tracker
            .finish(
                plan(F16ReferenceProjectionLayout::KnReference),
                summary(100, WeightAllocationDTypeV1::F32),
                summary(19, WeightAllocationDTypeV1::F16),
            )
            .unwrap_err();
        assert!(error.to_string().contains("tracked final F16 bytes"));
    }
}
