//! Backend-neutral consumer contract for model-supplied cross-layer KV reuse.
//!
//! DSV41-2 validates and resolves a versioned plan that states, for every
//! layer, whether its key and value caches are owned by that layer or reuse
//! the same component of one strictly earlier owning layer. The plan is
//! supplied by the model/domain owner; NNIS never infers, extends, or repairs
//! reuse semantics. Key and value may differ only because the plan says so.
//!
//! The contract is logical: it moves no bytes, allocates nothing, and an owner
//! count is not a physical-memory, quality, latency, or throughput result.

use core::fmt;

/// Version of the NNIS cross-layer KV reuse-plan consumer contract.
pub const NNIS_CROSS_LAYER_KV_REUSE_PLAN_VERSION: u32 = 1;

/// Maximum number of layers accepted by one plan.
pub const MAX_KV_REUSE_PLAN_LAYERS: usize = 4096;

/// Maximum UTF-8 bytes accepted for plan and model identities.
pub const MAX_KV_REUSE_PLAN_ID_BYTES: usize = 128;

/// KV cache component addressed by a reuse decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvComponent {
    /// Key cache.
    Key,
    /// Value cache.
    Value,
}

impl KvComponent {
    /// Stable lowercase name used in diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Key => "key",
            Self::Value => "value",
        }
    }
}

/// Source of one layer's key or value cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvLayerSourceV1 {
    /// The layer owns and produces this component.
    Own,
    /// The layer reads this component from the given earlier owning layer.
    ReuseFrom(u32),
}

/// Model-declared key/value sources for one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvLayerReuseV1 {
    /// Key-cache source.
    pub key: KvLayerSourceV1,
    /// Value-cache source.
    pub value: KvLayerSourceV1,
}

impl KvLayerReuseV1 {
    /// Layer owning both its key and value caches.
    pub const OWN: Self = Self {
        key: KvLayerSourceV1::Own,
        value: KvLayerSourceV1::Own,
    };

    /// Source declared for one component.
    pub const fn source(&self, component: KvComponent) -> KvLayerSourceV1 {
        match component {
            KvComponent::Key => self.key,
            KvComponent::Value => self.value,
        }
    }
}

/// Validated, immutable cross-layer KV reuse plan supplied by a model owner.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrossLayerKvReusePlanV1 {
    plan_id: String,
    model_id: String,
    plan_schema_version: u32,
    layers: Vec<KvLayerReuseV1>,
}

impl CrossLayerKvReusePlanV1 {
    /// Validate a model-supplied plan.
    ///
    /// Every reuse must reference a strictly earlier layer that owns the same
    /// component. Chained, self, forward, or out-of-range reuse fails closed;
    /// the plan is never rewritten to make it valid.
    pub fn new(
        plan_id: impl Into<String>,
        model_id: impl Into<String>,
        plan_schema_version: u32,
        layers: Vec<KvLayerReuseV1>,
    ) -> Result<Self, KvReusePlanError> {
        let plan_id = plan_id.into();
        let model_id = model_id.into();
        validate_id("plan_id", &plan_id)?;
        validate_id("model_id", &model_id)?;
        if plan_schema_version == 0 {
            return Err(KvReusePlanError::ZeroPlanSchemaVersion);
        }
        if layers.is_empty() {
            return Err(KvReusePlanError::NoLayers);
        }
        if layers.len() > MAX_KV_REUSE_PLAN_LAYERS {
            return Err(KvReusePlanError::TooManyLayers {
                layers: layers.len(),
            });
        }
        for (index, layer) in layers.iter().enumerate() {
            // Bounded by MAX_KV_REUSE_PLAN_LAYERS, so the index fits u32.
            let layer_index = index as u32;
            for component in [KvComponent::Key, KvComponent::Value] {
                if let KvLayerSourceV1::ReuseFrom(producer) = layer.source(component) {
                    if producer >= layer_index {
                        return Err(KvReusePlanError::NonCausalReuse {
                            layer: layer_index,
                            component,
                            producer,
                        });
                    }
                    if layers[producer as usize].source(component) != KvLayerSourceV1::Own {
                        return Err(KvReusePlanError::ChainedReuse {
                            layer: layer_index,
                            component,
                            producer,
                        });
                    }
                }
            }
        }
        Ok(Self {
            plan_id,
            model_id,
            plan_schema_version,
            layers,
        })
    }

    /// Model-owner plan identity.
    pub fn plan_id(&self) -> &str {
        &self.plan_id
    }

    /// Model identity the plan was declared for.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Model-owner plan schema version.
    pub const fn plan_schema_version(&self) -> u32 {
        self.plan_schema_version
    }

    /// Number of layers declared by the plan.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Declared sources for every layer, in layer order.
    pub fn layers(&self) -> &[KvLayerReuseV1] {
        &self.layers
    }

    /// Declared sources for one layer.
    pub fn layer(&self, layer: u32) -> Result<KvLayerReuseV1, KvReusePlanError> {
        self.layers
            .get(layer as usize)
            .copied()
            .ok_or(KvReusePlanError::LayerOutOfRange {
                layer,
                layers: self.layers.len(),
            })
    }

    /// Layer that owns the cache read by `layer` for `component`.
    pub fn owner(&self, layer: u32, component: KvComponent) -> Result<u32, KvReusePlanError> {
        Ok(match self.layer(layer)?.source(component) {
            KvLayerSourceV1::Own => layer,
            KvLayerSourceV1::ReuseFrom(producer) => producer,
        })
    }

    /// Layers owning a cache for `component`, in increasing order.
    pub fn owning_layers(&self, component: KvComponent) -> Vec<u32> {
        self.layers
            .iter()
            .enumerate()
            .filter(|(_, layer)| layer.source(component) == KvLayerSourceV1::Own)
            .map(|(index, _)| index as u32)
            .collect()
    }

    /// Number of logical caches the plan declares for `component`.
    ///
    /// This is a logical owner count, not a physical memory measurement.
    pub fn owned_count(&self, component: KvComponent) -> usize {
        self.layers
            .iter()
            .filter(|layer| layer.source(component) == KvLayerSourceV1::Own)
            .count()
    }
}

/// Fail-closed cross-layer KV reuse-plan errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvReusePlanError {
    /// One identity was blank.
    EmptyId { field: &'static str },
    /// One identity was not in canonical trimmed form.
    NonCanonicalId { field: &'static str },
    /// One identity exceeded the bounded contract.
    IdTooLong { field: &'static str, bytes: usize },
    /// Plan schema version zero is not valid.
    ZeroPlanSchemaVersion,
    /// The plan declared no layers.
    NoLayers,
    /// The plan declared more than [`MAX_KV_REUSE_PLAN_LAYERS`] layers.
    TooManyLayers { layers: usize },
    /// A layer reused itself or a later layer.
    NonCausalReuse {
        layer: u32,
        component: KvComponent,
        producer: u32,
    },
    /// A layer reused a producer that does not own that component.
    ChainedReuse {
        layer: u32,
        component: KvComponent,
        producer: u32,
    },
    /// A queried layer is not declared by the plan.
    LayerOutOfRange { layer: u32, layers: usize },
}

impl fmt::Display for KvReusePlanError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId { field } => write!(output, "{field} must not be empty"),
            Self::NonCanonicalId { field } => {
                write!(
                    output,
                    "{field} must not have leading or trailing whitespace"
                )
            }
            Self::IdTooLong { field, bytes } => write!(
                output,
                "{field} uses {bytes} bytes, maximum is {MAX_KV_REUSE_PLAN_ID_BYTES}"
            ),
            Self::ZeroPlanSchemaVersion => {
                output.write_str("KV reuse plan schema version must be non-zero")
            }
            Self::NoLayers => output.write_str("KV reuse plan must declare at least one layer"),
            Self::TooManyLayers { layers } => write!(
                output,
                "KV reuse plan declares {layers} layers, maximum is {MAX_KV_REUSE_PLAN_LAYERS}"
            ),
            Self::NonCausalReuse {
                layer,
                component,
                producer,
            } => write!(
                output,
                "layer {layer} {} cache reuses non-earlier layer {producer}",
                component.name()
            ),
            Self::ChainedReuse {
                layer,
                component,
                producer,
            } => write!(
                output,
                "layer {layer} {} cache reuses layer {producer}, which does not own it",
                component.name()
            ),
            Self::LayerOutOfRange { layer, layers } => {
                write!(output, "layer {layer} is outside the {layers}-layer plan")
            }
        }
    }
}

impl std::error::Error for KvReusePlanError {}

fn validate_id(field: &'static str, value: &str) -> Result<(), KvReusePlanError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(KvReusePlanError::EmptyId { field });
    }
    if trimmed != value {
        return Err(KvReusePlanError::NonCanonicalId { field });
    }
    if value.len() > MAX_KV_REUSE_PLAN_ID_BYTES {
        return Err(KvReusePlanError::IdTooLong {
            field,
            bytes: value.len(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use KvLayerSourceV1::{Own, ReuseFrom};

    fn layer(key: KvLayerSourceV1, value: KvLayerSourceV1) -> KvLayerReuseV1 {
        KvLayerReuseV1 { key, value }
    }

    fn plan(layers: Vec<KvLayerReuseV1>) -> Result<CrossLayerKvReusePlanV1, KvReusePlanError> {
        CrossLayerKvReusePlanV1::new("model.kv-share.v1", "fixture-model", 1, layers)
    }

    #[test]
    fn asymmetric_plan_resolves_declared_owners() {
        let plan = plan(vec![
            KvLayerReuseV1::OWN,
            layer(ReuseFrom(0), Own),
            layer(ReuseFrom(0), ReuseFrom(1)),
            KvLayerReuseV1::OWN,
            layer(ReuseFrom(3), ReuseFrom(3)),
        ])
        .unwrap();
        assert_eq!(plan.layer_count(), 5);
        assert_eq!(plan.owner(0, KvComponent::Key).unwrap(), 0);
        assert_eq!(plan.owner(2, KvComponent::Key).unwrap(), 0);
        assert_eq!(plan.owner(2, KvComponent::Value).unwrap(), 1);
        assert_eq!(plan.owner(4, KvComponent::Value).unwrap(), 3);
        assert_eq!(plan.owning_layers(KvComponent::Key), vec![0, 3]);
        assert_eq!(plan.owning_layers(KvComponent::Value), vec![0, 1, 3]);
        assert_eq!(plan.owned_count(KvComponent::Key), 2);
        assert_eq!(plan.owned_count(KvComponent::Value), 3);
    }

    #[test]
    fn plan_without_reuse_owns_every_layer() {
        let plan = plan(vec![KvLayerReuseV1::OWN; 4]).unwrap();
        for index in 0..4 {
            assert_eq!(plan.owner(index, KvComponent::Key).unwrap(), index);
            assert_eq!(plan.owner(index, KvComponent::Value).unwrap(), index);
        }
        assert_eq!(plan.owned_count(KvComponent::Key), 4);
    }

    #[test]
    fn self_forward_and_first_layer_reuse_fail_closed() {
        assert_eq!(
            plan(vec![layer(ReuseFrom(0), Own)]),
            Err(KvReusePlanError::NonCausalReuse {
                layer: 0,
                component: KvComponent::Key,
                producer: 0,
            })
        );
        assert_eq!(
            plan(vec![layer(Own, ReuseFrom(1)), KvLayerReuseV1::OWN]),
            Err(KvReusePlanError::NonCausalReuse {
                layer: 0,
                component: KvComponent::Value,
                producer: 1,
            })
        );
        assert_eq!(
            plan(vec![KvLayerReuseV1::OWN, layer(Own, ReuseFrom(7))]),
            Err(KvReusePlanError::NonCausalReuse {
                layer: 1,
                component: KvComponent::Value,
                producer: 7,
            })
        );
    }

    #[test]
    fn chained_or_cross_component_ownership_is_not_inferred() {
        assert_eq!(
            plan(vec![
                KvLayerReuseV1::OWN,
                layer(ReuseFrom(0), Own),
                layer(ReuseFrom(1), Own),
            ]),
            Err(KvReusePlanError::ChainedReuse {
                layer: 2,
                component: KvComponent::Key,
                producer: 1,
            })
        );
        // Layer 1 owns only its value; reusing its key is not valid.
        assert_eq!(
            plan(vec![
                KvLayerReuseV1::OWN,
                layer(ReuseFrom(0), Own),
                layer(ReuseFrom(1), ReuseFrom(1)),
            ]),
            Err(KvReusePlanError::ChainedReuse {
                layer: 2,
                component: KvComponent::Key,
                producer: 1,
            })
        );
    }

    #[test]
    fn malformed_identity_version_and_size_are_rejected() {
        assert_eq!(
            CrossLayerKvReusePlanV1::new(" plan", "model", 1, vec![KvLayerReuseV1::OWN]),
            Err(KvReusePlanError::NonCanonicalId { field: "plan_id" })
        );
        assert_eq!(
            CrossLayerKvReusePlanV1::new("plan", "", 1, vec![KvLayerReuseV1::OWN]),
            Err(KvReusePlanError::EmptyId { field: "model_id" })
        );
        assert_eq!(
            CrossLayerKvReusePlanV1::new("p".repeat(129), "model", 1, vec![KvLayerReuseV1::OWN]),
            Err(KvReusePlanError::IdTooLong {
                field: "plan_id",
                bytes: 129,
            })
        );
        assert_eq!(
            CrossLayerKvReusePlanV1::new("plan", "model", 0, vec![KvLayerReuseV1::OWN]),
            Err(KvReusePlanError::ZeroPlanSchemaVersion)
        );
        assert_eq!(plan(Vec::new()), Err(KvReusePlanError::NoLayers));
        assert_eq!(
            plan(vec![KvLayerReuseV1::OWN; MAX_KV_REUSE_PLAN_LAYERS + 1]),
            Err(KvReusePlanError::TooManyLayers {
                layers: MAX_KV_REUSE_PLAN_LAYERS + 1,
            })
        );
        assert!(plan(vec![KvLayerReuseV1::OWN; MAX_KV_REUSE_PLAN_LAYERS]).is_ok());
    }

    #[test]
    fn out_of_range_queries_fail_closed() {
        let plan = plan(vec![KvLayerReuseV1::OWN; 2]).unwrap();
        assert_eq!(
            plan.owner(2, KvComponent::Key),
            Err(KvReusePlanError::LayerOutOfRange {
                layer: 2,
                layers: 2
            })
        );
    }
}
