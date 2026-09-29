//! Backend-neutral portable autoregressive session surface (v1).
//!
//! This contract is the portable counterpart to the CUDA-only InferenceSession
//! shape: encode (prefill),
//! decode_one, explicit KV advance, and truncate. It owns no vendor API and
//! invents no model science. Concrete backends (CPU reference, later WGPU)
//! implement [`PortableSessionV1`] over host-visible dense F32 KV storage
//! described by [`PortableKvCacheV1`].
//!
//! The synthetic model dimensions in [`SyntheticPortableModelSpecV1`] exist so
//! unit tests can exercise the session without downloading weights. Synthetic
//! fixtures are never model-quality or performance evidence.
//!
//! CUDA nnis-rt KvCache and the existing NVIDIA InferenceSession path
//! are intentionally untouched by this module.

use core::fmt;

/// Version of the portable session surface contract.
pub const PORTABLE_SESSION_VERSION: u32 = 1;

/// Numerical policy identity shared with the portable F32 graph / CPU kernels.
pub const PORTABLE_SESSION_POLICY: &str = "finite-f32-le-ordered-fma-v1";

/// Stable identity for the host-dense F32 KV layout used by portable sessions.
pub const PORTABLE_KV_LAYOUT_ID: &str = "nnis.kv.host-dense-f32.v1";

/// Largest accepted vocabulary, hidden size, capacity or layer count in v1.
pub const MAX_SYNTHETIC_SESSION_DIM: usize = 1024;

/// Fail-closed errors for the portable session surface and host KV layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortableSessionError {
    /// A dimension or token id is zero, out of range, or overflows.
    Invalid(&'static str),
    /// Session state does not allow the requested transition.
    State(&'static str),
    /// Backend / arithmetic failure described by the implementor.
    Backend(String),
}

impl fmt::Display for PortableSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => f.write_str(message),
            Self::State(message) => f.write_str(message),
            Self::Backend(message) => write!(f, "portable session backend: {message}"),
        }
    }
}

impl std::error::Error for PortableSessionError {}

/// Fixed tiny analytical model dimensions for portable session tests.
///
/// Values are structural only. They do not identify a trained checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyntheticPortableModelSpecV1 {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub capacity: usize,
    pub layers: usize,
}

impl SyntheticPortableModelSpecV1 {
    /// Validate positive dimensions within the v1 admission ceiling.
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        capacity: usize,
        layers: usize,
    ) -> Result<Self, PortableSessionError> {
        for (name, value) in [
            ("vocab_size", vocab_size),
            ("hidden_size", hidden_size),
            ("capacity", capacity),
            ("layers", layers),
        ] {
            if value == 0 {
                return Err(PortableSessionError::Invalid(
                    "synthetic portable model dimensions must be positive",
                ));
            }
            if value > MAX_SYNTHETIC_SESSION_DIM {
                let _ = name;
                return Err(PortableSessionError::Invalid(
                    "synthetic portable model dimension exceeds v1 maximum",
                ));
            }
        }
        let _ = vocab_size
            .checked_mul(hidden_size)
            .and_then(|v| v.checked_mul(capacity))
            .and_then(|v| v.checked_mul(layers))
            .ok_or(PortableSessionError::Invalid(
                "synthetic portable model storage shape overflows usize",
            ))?;
        Ok(Self {
            vocab_size,
            hidden_size,
            capacity,
            layers,
        })
    }

    /// Default tiny fixture used by CPU/WGPU portable session tests.
    pub fn tiny() -> Self {
        Self::new(4, 4, 8, 1).expect("static tiny synthetic spec is valid")
    }
}

/// Host-side dense F32 KV cache: `[layer][capacity][row_width]`.
///
/// Logical length is tracked once and applied to every layer (transformer-style
/// shared sequence length). Append writes one row per layer. Truncate is
/// all-or-nothing across layers. This is a layout and accounting object, not a
/// CUDA device allocation and not a compression format.
#[derive(Debug, Clone, PartialEq)]
pub struct PortableKvCacheV1 {
    layers: usize,
    capacity: usize,
    row_width: usize,
    length: usize,
    storage: Vec<f32>,
}

impl PortableKvCacheV1 {
    /// Allocate zero-filled storage for the declared layout.
    pub fn new(
        layers: usize,
        capacity: usize,
        row_width: usize,
    ) -> Result<Self, PortableSessionError> {
        if layers == 0 || capacity == 0 || row_width == 0 {
            return Err(PortableSessionError::Invalid(
                "portable KV dimensions must be positive",
            ));
        }
        if layers > MAX_SYNTHETIC_SESSION_DIM
            || capacity > MAX_SYNTHETIC_SESSION_DIM
            || row_width > MAX_SYNTHETIC_SESSION_DIM
        {
            return Err(PortableSessionError::Invalid(
                "portable KV dimension exceeds v1 maximum",
            ));
        }
        let total = layers
            .checked_mul(capacity)
            .and_then(|v| v.checked_mul(row_width))
            .ok_or(PortableSessionError::Invalid(
                "portable KV storage shape overflows usize",
            ))?;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(total)
            .map_err(|_| PortableSessionError::Invalid("portable KV storage reservation failed"))?;
        storage.resize(total, 0.0);
        Ok(Self {
            layers,
            capacity,
            row_width,
            length: 0,
            storage,
        })
    }

    pub const fn layers(&self) -> usize {
        self.layers
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub const fn row_width(&self) -> usize {
        self.row_width
    }

    pub const fn len(&self) -> usize {
        self.length
    }

    pub const fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Exact host payload bytes currently addressed by the logical length.
    pub fn logical_payload_bytes(&self) -> u64 {
        (self.layers as u64)
            .saturating_mul(self.length as u64)
            .saturating_mul(self.row_width as u64)
            .saturating_mul(4)
    }

    /// Exact host payload bytes of the full allocated capacity.
    pub fn capacity_payload_bytes(&self) -> u64 {
        (self.layers as u64)
            .saturating_mul(self.capacity as u64)
            .saturating_mul(self.row_width as u64)
            .saturating_mul(4)
    }

    /// Immutable view of one stored row.
    pub fn row(&self, layer: usize, index: usize) -> Result<&[f32], PortableSessionError> {
        if layer >= self.layers {
            return Err(PortableSessionError::Invalid(
                "portable KV layer index out of range",
            ));
        }
        if index >= self.length {
            return Err(PortableSessionError::Invalid(
                "portable KV row index out of range",
            ));
        }
        Ok(self.row_slot(layer, index))
    }

    /// Append one row to every layer. `rows.len()` must equal `layers`.
    pub fn append(&mut self, rows: &[&[f32]]) -> Result<(), PortableSessionError> {
        if rows.len() != self.layers {
            return Err(PortableSessionError::Invalid(
                "portable KV append requires one row slice per layer",
            ));
        }
        if self.length >= self.capacity {
            return Err(PortableSessionError::State(
                "portable KV append exceeds capacity",
            ));
        }
        for (layer, row) in rows.iter().enumerate() {
            if row.len() != self.row_width {
                return Err(PortableSessionError::Invalid(
                    "portable KV append row width mismatch",
                ));
            }
            if row.iter().any(|value| !value.is_finite()) {
                return Err(PortableSessionError::Invalid(
                    "portable KV append rejects non-finite values",
                ));
            }
            self.row_slot_mut(layer, self.length).copy_from_slice(row);
        }
        self.length += 1;
        Ok(())
    }

    /// Shorten every layer to `length`. Fail closed when `length > len`.
    pub fn truncate(&mut self, length: usize) -> Result<(), PortableSessionError> {
        if length > self.length {
            return Err(PortableSessionError::Invalid(
                "portable KV truncate length exceeds current length",
            ));
        }
        self.length = length;
        Ok(())
    }

    /// Reset logical length to zero without releasing storage.
    pub fn reset(&mut self) {
        self.length = 0;
    }

    fn row_offset(&self, layer: usize, index: usize) -> usize {
        ((layer * self.capacity) + index) * self.row_width
    }

    fn row_slot(&self, layer: usize, index: usize) -> &[f32] {
        let start = self.row_offset(layer, index);
        &self.storage[start..start + self.row_width]
    }

    fn row_slot_mut(&mut self, layer: usize, index: usize) -> &mut [f32] {
        let start = self.row_offset(layer, index);
        let end = start + self.row_width;
        &mut self.storage[start..end]
    }
}

/// Portable autoregressive session operations.
///
/// `stage_token` writes a pending KV row and refreshes logits without changing
/// [`PortableSessionV1::kv_len`]. [`PortableSessionV1::advance_kv`] commits that
/// row. [`PortableSessionV1::encode`] and [`PortableSessionV1::decode_one`] are
/// the caller-facing helpers that stage then advance for each token.
pub trait PortableSessionV1 {
    fn position(&self) -> usize;
    fn kv_len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn has_pending_kv(&self) -> bool;
    fn logits(&self) -> &[f32];

    fn stage_token(&mut self, token: u32) -> Result<&[f32], PortableSessionError>;
    fn advance_kv(&mut self) -> Result<(), PortableSessionError>;
    fn encode(&mut self, tokens: &[u32]) -> Result<&[f32], PortableSessionError>;
    fn decode_one(&mut self, token: u32) -> Result<&[f32], PortableSessionError>;
    fn truncate(&mut self, length: usize) -> Result<(), PortableSessionError>;
    fn reset(&mut self) -> Result<(), PortableSessionError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_spec_and_kv_round_trip() {
        let spec = SyntheticPortableModelSpecV1::tiny();
        assert_eq!(spec.vocab_size, 4);
        let mut kv = PortableKvCacheV1::new(spec.layers, spec.capacity, spec.hidden_size).unwrap();
        assert!(kv.is_empty());
        kv.append(&[&[1.0, 0.0, 0.0, 0.0]]).unwrap();
        kv.append(&[&[0.0, 1.0, 0.0, 0.0]]).unwrap();
        assert_eq!(kv.len(), 2);
        assert_eq!(kv.row(0, 0).unwrap(), &[1.0, 0.0, 0.0, 0.0]);
        kv.truncate(1).unwrap();
        assert_eq!(kv.len(), 1);
        assert!(kv.truncate(2).is_err());
        assert!(kv.append(&[&[0.0, 0.0, 1.0]]).is_err());
    }

    #[test]
    fn kv_rejects_capacity_overflow_and_non_finite() {
        let mut kv = PortableKvCacheV1::new(1, 1, 2).unwrap();
        kv.append(&[&[1.0, 2.0]]).unwrap();
        assert!(kv.append(&[&[3.0, 4.0]]).is_err());
        kv.reset();
        assert!(kv.append(&[&[f32::NAN, 0.0]]).is_err());
    }
}
