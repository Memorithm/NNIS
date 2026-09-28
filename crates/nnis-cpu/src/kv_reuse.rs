//! DSV41-2 CPU consumer of a model-supplied cross-layer KV reuse plan.
//!
//! [`CpuCrossLayerKvBindingV1`] binds exactly one dense
//! [`CpuReplaySourceV1`] to every owning (layer, component) pair of a
//! validated [`CrossLayerKvReusePlanV1`] and resolves reads for reusing layers
//! to their declared owner. It never infers, extends, or repairs reuse; a
//! missing, duplicate, or undeclared owner source fails closed. All bound
//! sources must cover the identical logical position range so that a reusing
//! layer reads exactly the positions its owner holds.
//!
//! Sharing is logical: fewer bound sources is not a physical memory, quality,
//! latency, or throughput result.

use core::fmt;

use nnis_core::kv_reuse_plan::{
    CrossLayerKvReusePlanV1, KvComponent, KvLayerSourceV1, KvReusePlanError,
};
use nnis_core::replay_state::{ReplayStateProviderV1, ReplayWindowRequestV1};

use crate::replay::{CpuReplayError, CpuReplaySourceV1};

/// Owner replay sources bound to a validated cross-layer KV reuse plan.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuCrossLayerKvBindingV1 {
    plan: CrossLayerKvReusePlanV1,
    keys: Vec<Option<CpuReplaySourceV1>>,
    values: Vec<Option<CpuReplaySourceV1>>,
}

impl CpuCrossLayerKvBindingV1 {
    /// Bind one owner source per owning layer for each component.
    ///
    /// `key_sources` and `value_sources` are `(owning_layer, source)` pairs.
    /// Every owning layer of the plan must appear exactly once per component,
    /// no reusing layer may be given a source, and all sources must declare
    /// the same logical start and end positions.
    pub fn new(
        plan: CrossLayerKvReusePlanV1,
        key_sources: Vec<(u32, CpuReplaySourceV1)>,
        value_sources: Vec<(u32, CpuReplaySourceV1)>,
    ) -> Result<Self, CpuKvReuseError> {
        let keys = bind_component(&plan, KvComponent::Key, key_sources)?;
        let values = bind_component(&plan, KvComponent::Value, value_sources)?;
        let mut range = None;
        for source in keys.iter().chain(values.iter()).flatten() {
            let identity = source.replay_source_identity();
            let bound = (
                identity.logical_start_position(),
                identity.logical_end_position(),
            );
            match range {
                None => range = Some(bound),
                Some(expected) if expected != bound => {
                    return Err(CpuKvReuseError::SourceRangeMismatch {
                        expected_start: expected.0,
                        expected_end: expected.1,
                        actual_start: bound.0,
                        actual_end: bound.1,
                    });
                }
                Some(_) => {}
            }
        }
        Ok(Self { plan, keys, values })
    }

    /// Validated plan this binding consumes.
    pub const fn plan(&self) -> &CrossLayerKvReusePlanV1 {
        &self.plan
    }

    /// Owner source read by `layer` for `component`.
    pub fn source(
        &self,
        layer: u32,
        component: KvComponent,
    ) -> Result<&CpuReplaySourceV1, CpuKvReuseError> {
        let owner = self.plan.owner(layer, component)?;
        let slots = match component {
            KvComponent::Key => &self.keys,
            KvComponent::Value => &self.values,
        };
        slots.get(owner as usize).and_then(Option::as_ref).ok_or(
            CpuKvReuseError::MissingOwnerSource {
                layer: owner,
                component,
            },
        )
    }

    /// Replay a validated window for `layer`/`component` from its owner.
    pub fn replay_window(
        &self,
        layer: u32,
        component: KvComponent,
        request: &ReplayWindowRequestV1,
    ) -> Result<Vec<f32>, CpuKvReuseError> {
        Ok(self.source(layer, component)?.replay_window(request)?)
    }

    /// Replay the caller-supplied most recent window for `layer`/`component`.
    pub fn replay_recent_window(
        &self,
        layer: u32,
        component: KvComponent,
        window_items: u64,
    ) -> Result<(ReplayWindowRequestV1, Vec<f32>), CpuKvReuseError> {
        Ok(self
            .source(layer, component)?
            .replay_recent_window(window_items)?)
    }

    /// Number of bound owner sources for `component` (logical, not physical).
    pub fn bound_source_count(&self, component: KvComponent) -> usize {
        let slots = match component {
            KvComponent::Key => &self.keys,
            KvComponent::Value => &self.values,
        };
        slots.iter().filter(|slot| slot.is_some()).count()
    }
}

fn bind_component(
    plan: &CrossLayerKvReusePlanV1,
    component: KvComponent,
    sources: Vec<(u32, CpuReplaySourceV1)>,
) -> Result<Vec<Option<CpuReplaySourceV1>>, CpuKvReuseError> {
    let mut slots: Vec<Option<CpuReplaySourceV1>> = vec![None; plan.layer_count()];
    for (layer, source) in sources {
        if plan.layer(layer)?.source(component) != KvLayerSourceV1::Own {
            return Err(CpuKvReuseError::SourceForReusingLayer { layer, component });
        }
        let slot = &mut slots[layer as usize];
        if slot.is_some() {
            return Err(CpuKvReuseError::DuplicateOwnerSource { layer, component });
        }
        *slot = Some(source);
    }
    for layer in plan.owning_layers(component) {
        if slots[layer as usize].is_none() {
            return Err(CpuKvReuseError::MissingOwnerSource { layer, component });
        }
    }
    Ok(slots)
}

/// Fail-closed CPU cross-layer KV binding errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuKvReuseError {
    /// Plan validation or lookup failed.
    Plan(KvReusePlanError),
    /// Replay of the owner source failed.
    Replay(CpuReplayError),
    /// An owning layer had no bound source for this component.
    MissingOwnerSource { layer: u32, component: KvComponent },
    /// An owning layer was given more than one source for this component.
    DuplicateOwnerSource { layer: u32, component: KvComponent },
    /// A source was supplied for a layer that reuses this component.
    SourceForReusingLayer { layer: u32, component: KvComponent },
    /// Bound sources did not declare identical logical ranges.
    SourceRangeMismatch {
        expected_start: u64,
        expected_end: u64,
        actual_start: u64,
        actual_end: u64,
    },
}

impl From<KvReusePlanError> for CpuKvReuseError {
    fn from(error: KvReusePlanError) -> Self {
        Self::Plan(error)
    }
}

impl From<CpuReplayError> for CpuKvReuseError {
    fn from(error: CpuReplayError) -> Self {
        Self::Replay(error)
    }
}

impl fmt::Display for CpuKvReuseError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(error) => write!(output, "KV reuse plan: {error}"),
            Self::Replay(error) => write!(output, "KV replay: {error}"),
            Self::MissingOwnerSource { layer, component } => write!(
                output,
                "owning layer {layer} has no bound {} source",
                component.name()
            ),
            Self::DuplicateOwnerSource { layer, component } => write!(
                output,
                "owning layer {layer} has more than one {} source",
                component.name()
            ),
            Self::SourceForReusingLayer { layer, component } => write!(
                output,
                "layer {layer} reuses its {} cache and must not be given a source",
                component.name()
            ),
            Self::SourceRangeMismatch {
                expected_start,
                expected_end,
                actual_start,
                actual_end,
            } => write!(
                output,
                "bound source range {actual_start}..={actual_end} differs from {expected_start}..={expected_end}"
            ),
        }
    }
}

impl std::error::Error for CpuKvReuseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Plan(error) => Some(error),
            Self::Replay(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_core::kv_reuse_plan::KvLayerReuseV1;
    use nnis_core::replay_state::{
        ReplayIdentityError, ReplayRepresentationIdentityV1, ReplaySourceIdentityV1,
    };
    use KvLayerSourceV1::{Own, ReuseFrom};

    const WIDTH: usize = 2;

    fn source(tag: &str, start: u64, end: u64, fill: f32) -> CpuReplaySourceV1 {
        let identity = ReplaySourceIdentityV1::new(
            "nnis-cpu-reference",
            tag,
            1,
            ReplayRepresentationIdentityV1::new("dense.f32.rows", 1, 0).unwrap(),
            start,
            end,
        )
        .unwrap();
        let items = (end - start + 1) as usize;
        let rows = (0..items * WIDTH)
            .map(|index| fill + index as f32)
            .collect();
        CpuReplaySourceV1::new(identity, WIDTH, rows).unwrap()
    }

    /// Layer 1 reuses layer 0's key but owns its value; layer 2 reuses both.
    fn plan() -> CrossLayerKvReusePlanV1 {
        CrossLayerKvReusePlanV1::new(
            "fixture.kv-share.v1",
            "fixture-model",
            1,
            vec![
                KvLayerReuseV1::OWN,
                KvLayerReuseV1 {
                    key: ReuseFrom(0),
                    value: Own,
                },
                KvLayerReuseV1 {
                    key: ReuseFrom(0),
                    value: ReuseFrom(1),
                },
            ],
        )
        .unwrap()
    }

    fn binding() -> CpuCrossLayerKvBindingV1 {
        CpuCrossLayerKvBindingV1::new(
            plan(),
            vec![(0, source("k0", 0, 7, 0.0))],
            vec![
                (0, source("v0", 0, 7, 100.0)),
                (1, source("v1", 0, 7, 200.0)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn reusing_layers_read_their_declared_owner_bit_exactly() {
        let binding = binding();
        let (_, owner_key) = binding
            .replay_recent_window(0, KvComponent::Key, 2)
            .unwrap();
        let (_, reused_key) = binding
            .replay_recent_window(2, KvComponent::Key, 2)
            .unwrap();
        assert_eq!(owner_key, reused_key);
        assert_eq!(owner_key, vec![12.0, 13.0, 14.0, 15.0]);

        let (_, layer1_value) = binding
            .replay_recent_window(1, KvComponent::Value, 1)
            .unwrap();
        let (_, layer2_value) = binding
            .replay_recent_window(2, KvComponent::Value, 1)
            .unwrap();
        assert_eq!(layer1_value, layer2_value);
        assert_eq!(layer1_value, vec![214.0, 215.0]);

        let (_, layer0_value) = binding
            .replay_recent_window(0, KvComponent::Value, 1)
            .unwrap();
        assert_eq!(layer0_value, vec![114.0, 115.0]);

        assert_eq!(binding.bound_source_count(KvComponent::Key), 1);
        assert_eq!(binding.bound_source_count(KvComponent::Value), 2);
    }

    #[test]
    fn explicit_window_request_must_match_owner_identity() {
        let binding = binding();
        let owner = binding.source(2, KvComponent::Key).unwrap();
        let request =
            ReplayWindowRequestV1::new(owner.replay_source_identity().clone(), 3, 4).unwrap();
        assert_eq!(
            binding
                .replay_window(2, KvComponent::Key, &request)
                .unwrap(),
            vec![6.0, 7.0, 8.0, 9.0]
        );
        // A request bound to layer 1's own value source is not layer 0's key.
        let value_owner = binding.source(1, KvComponent::Value).unwrap();
        let foreign =
            ReplayWindowRequestV1::new(value_owner.replay_source_identity().clone(), 3, 4).unwrap();
        assert_eq!(
            binding.replay_window(2, KvComponent::Key, &foreign),
            Err(CpuKvReuseError::Replay(CpuReplayError::Identity(
                ReplayIdentityError::SourceIdentityMismatch
            )))
        );
    }

    #[test]
    fn missing_duplicate_or_undeclared_sources_fail_closed() {
        assert_eq!(
            CpuCrossLayerKvBindingV1::new(
                plan(),
                vec![(0, source("k0", 0, 7, 0.0))],
                vec![(0, source("v0", 0, 7, 0.0))],
            ),
            Err(CpuKvReuseError::MissingOwnerSource {
                layer: 1,
                component: KvComponent::Value,
            })
        );
        assert_eq!(
            CpuCrossLayerKvBindingV1::new(
                plan(),
                vec![(0, source("k0", 0, 7, 0.0)), (0, source("k0b", 0, 7, 0.0))],
                Vec::new(),
            ),
            Err(CpuKvReuseError::DuplicateOwnerSource {
                layer: 0,
                component: KvComponent::Key,
            })
        );
        assert_eq!(
            CpuCrossLayerKvBindingV1::new(
                plan(),
                vec![(0, source("k0", 0, 7, 0.0)), (1, source("k1", 0, 7, 0.0))],
                Vec::new(),
            ),
            Err(CpuKvReuseError::SourceForReusingLayer {
                layer: 1,
                component: KvComponent::Key,
            })
        );
        assert_eq!(
            CpuCrossLayerKvBindingV1::new(plan(), vec![(3, source("k3", 0, 7, 0.0))], Vec::new(),),
            Err(CpuKvReuseError::Plan(KvReusePlanError::LayerOutOfRange {
                layer: 3,
                layers: 3,
            }))
        );
    }

    #[test]
    fn owner_sources_must_share_one_logical_range() {
        assert_eq!(
            CpuCrossLayerKvBindingV1::new(
                plan(),
                vec![(0, source("k0", 0, 7, 0.0))],
                vec![(0, source("v0", 0, 7, 0.0)), (1, source("v1", 1, 8, 0.0))],
            ),
            Err(CpuKvReuseError::SourceRangeMismatch {
                expected_start: 0,
                expected_end: 7,
                actual_start: 1,
                actual_end: 8,
            })
        );
    }

    #[test]
    fn queries_outside_the_plan_fail_closed() {
        assert_eq!(
            binding().source(3, KvComponent::Value).err(),
            Some(CpuKvReuseError::Plan(KvReusePlanError::LayerOutOfRange {
                layer: 3,
                layers: 3,
            }))
        );
    }
}
