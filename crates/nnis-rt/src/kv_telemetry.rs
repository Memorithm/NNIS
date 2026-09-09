//! Read-only logical telemetry for NNIS device-resident KV caches.
//!
//! This module exposes metadata only. It does not read K/V bytes back to the
//! host, alter placement, or define an eviction/tiering policy. The intent is
//! to let experiment harnesses compare cache state across backends while NNIS
//! remains the owner of NVIDIA/CUDA allocation and movement semantics.

use crate::{DevicePod, KvCache, Result};

/// Snapshot of one fixed-capacity NNIS KV cache's logical occupancy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheTelemetry {
    /// Valid token count for each layer, in layer order.
    pub layer_lengths: Vec<usize>,
    /// Sum of valid token positions across all layers.
    pub total_live_tokens: usize,
    /// Sum of available token positions across all layers.
    pub total_capacity_tokens: usize,
    /// Number of cache layers.
    pub layers: usize,
    /// Native K/V head count retained by the CUDA cache.
    pub heads: usize,
    /// Elements per head vector.
    pub head_dim: usize,
}

/// Observe logical occupancy of `cache` without touching K/V payload bytes.
///
/// # Errors
///
/// Propagates cache indexing errors and fails if aggregate token counts would
/// overflow `usize`.
pub fn observe_kv_cache<T: DevicePod>(cache: &KvCache<T>) -> Result<KvCacheTelemetry> {
    let config = cache.config();
    let mut layer_lengths = Vec::with_capacity(config.layers);
    let mut total_live_tokens = 0usize;

    for layer in 0..config.layers {
        let length = cache.len(layer)?;
        total_live_tokens = total_live_tokens
            .checked_add(length)
            .ok_or_else(|| crate::NnisError::invalid_input("KV telemetry live-token sum overflows usize"))?;
        layer_lengths.push(length);
    }

    let total_capacity_tokens = config
        .layers
        .checked_mul(config.capacity)
        .ok_or_else(|| crate::NnisError::invalid_input("KV telemetry capacity sum overflows usize"))?;

    Ok(KvCacheTelemetry {
        layer_lengths,
        total_live_tokens,
        total_capacity_tokens,
        layers: config.layers,
        heads: config.heads,
        head_dim: config.head_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_type_preserves_explicit_zero_state() {
        let telemetry = KvCacheTelemetry {
            layer_lengths: vec![0, 0],
            total_live_tokens: 0,
            total_capacity_tokens: 16,
            layers: 2,
            heads: 4,
            head_dim: 8,
        };
        assert_eq!(telemetry.layer_lengths, vec![0, 0]);
        assert_eq!(telemetry.total_live_tokens, 0);
        assert_eq!(telemetry.total_capacity_tokens, 16);
    }
}
