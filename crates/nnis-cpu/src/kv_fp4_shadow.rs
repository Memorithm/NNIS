//! Opt-in DSV41-3 FP4 E2M1 shadow storage for [`PortableKvCacheV1`].
//!
//! The dense F32 cache remains the semantic reference. This module encodes a
//! per-layer FP4 shadow with exact [`Fp4KvStorageV1`] accounting. It is not a
//! quality, memory, latency or throughput claim.

use nnis_core::kv_fp4::{Fp4E2M1KvLayoutV1, Fp4KvStorageV1, Fp4ScaleEncodingV1};
use nnis_core::session::{
    PortableKvCacheV1, PortableKvStorageModeV1, PortableKvStorageTelemetryV1, PortableSessionError,
};

use crate::fp4_kv::CpuFp4E2M1KvBlockV1;

/// Per-layer FP4 shadow mirroring a dense portable KV cache.
#[derive(Debug, Clone, PartialEq)]
pub struct PortableFp4KvShadowV1 {
    group_size: u32,
    scale_encoding: Fp4ScaleEncodingV1,
    layers: usize,
    capacity: usize,
    row_width: usize,
    length: usize,
    /// Packed codes per layer for the committed length (empty when length == 0).
    layer_codes: Vec<Vec<u8>>,
    /// Scale bytes per layer for the committed length.
    layer_scales: Vec<Vec<u8>>,
}

impl PortableFp4KvShadowV1 {
    /// Build an empty shadow for `mode` matching `cache` geometry.
    pub fn new(
        cache: &PortableKvCacheV1,
        mode: PortableKvStorageModeV1,
    ) -> Result<Option<Self>, PortableSessionError> {
        let PortableKvStorageModeV1::Fp4E2M1 {
            group_size,
            scale_encoding,
        } = mode
        else {
            return Ok(None);
        };
        mode.validate_for_row_width(cache.row_width())?;
        Ok(Some(Self {
            group_size,
            scale_encoding,
            layers: cache.layers(),
            capacity: cache.capacity(),
            row_width: cache.row_width(),
            length: 0,
            layer_codes: vec![Vec::new(); cache.layers()],
            layer_scales: vec![Vec::new(); cache.layers()],
        }))
    }

    pub const fn length(&self) -> usize {
        self.length
    }

    pub const fn group_size(&self) -> u32 {
        self.group_size
    }

    pub const fn scale_encoding(&self) -> Fp4ScaleEncodingV1 {
        self.scale_encoding
    }

    /// Re-encode every committed dense row into the FP4 shadow.
    pub fn sync_from_dense(
        &mut self,
        cache: &PortableKvCacheV1,
    ) -> Result<(), PortableSessionError> {
        if cache.layers() != self.layers
            || cache.capacity() != self.capacity
            || cache.row_width() != self.row_width
        {
            return Err(PortableSessionError::Invalid(
                "FP4 shadow geometry does not match dense portable KV cache",
            ));
        }
        let length = cache.len();
        if length == 0 {
            self.length = 0;
            for codes in &mut self.layer_codes {
                codes.clear();
            }
            for scales in &mut self.layer_scales {
                scales.clear();
            }
            return Ok(());
        }
        let rows = u64::try_from(length)
            .map_err(|_| PortableSessionError::Invalid("portable FP4 shadow length exceeds u64"))?;
        let width = u32::try_from(self.row_width).map_err(|_| {
            PortableSessionError::Invalid("portable FP4 shadow row width exceeds u32")
        })?;
        let layout = Fp4E2M1KvLayoutV1::new(rows, width, self.group_size, self.scale_encoding)
            .map_err(|error| {
                PortableSessionError::Backend(format!("FP4 shadow layout failed: {error}"))
            })?;
        for layer in 0..self.layers {
            let values = cache.layer_prefix(layer)?;
            let block = CpuFp4E2M1KvBlockV1::encode(layout, values).map_err(|error| {
                PortableSessionError::Backend(format!("FP4 shadow encode failed: {error}"))
            })?;
            self.layer_codes[layer] = block.codes().to_vec();
            self.layer_scales[layer] = block.scales().to_vec();
        }
        self.length = length;
        Ok(())
    }

    /// Decode one layer's shadow back to dense F32 (CPU oracle).
    pub fn decode_layer(&self, layer: usize) -> Result<Vec<f32>, PortableSessionError> {
        if layer >= self.layers {
            return Err(PortableSessionError::Invalid(
                "FP4 shadow layer index out of range",
            ));
        }
        if self.length == 0 {
            return Ok(Vec::new());
        }
        let rows = self.length as u64;
        let width = self.row_width as u32;
        let layout = Fp4E2M1KvLayoutV1::new(rows, width, self.group_size, self.scale_encoding)
            .map_err(|error| {
                PortableSessionError::Backend(format!("FP4 shadow layout failed: {error}"))
            })?;
        let block = CpuFp4E2M1KvBlockV1::from_parts(
            layout,
            self.layer_codes[layer].clone(),
            self.layer_scales[layer].clone(),
        )
        .map_err(|error| {
            PortableSessionError::Backend(format!("FP4 shadow from_parts failed: {error}"))
        })?;
        block.decode().map_err(|error| {
            PortableSessionError::Backend(format!("FP4 shadow decode failed: {error}"))
        })
    }

    /// Codes for one layer's committed shadow.
    pub fn layer_codes(&self, layer: usize) -> Result<&[u8], PortableSessionError> {
        self.layer_codes
            .get(layer)
            .map(Vec::as_slice)
            .ok_or(PortableSessionError::Invalid(
                "FP4 shadow layer index out of range",
            ))
    }

    /// Scales for one layer's committed shadow.
    pub fn layer_scales(&self, layer: usize) -> Result<&[u8], PortableSessionError> {
        self.layer_scales
            .get(layer)
            .map(Vec::as_slice)
            .ok_or(PortableSessionError::Invalid(
                "FP4 shadow layer index out of range",
            ))
    }

    fn storage_for_rows(
        &self,
        rows: usize,
    ) -> Result<Option<Fp4KvStorageV1>, PortableSessionError> {
        if rows == 0 {
            return Ok(None);
        }
        let layout = Fp4E2M1KvLayoutV1::new(
            rows as u64,
            self.row_width as u32,
            self.group_size,
            self.scale_encoding,
        )
        .map_err(|error| {
            PortableSessionError::Backend(format!("FP4 shadow layout failed: {error}"))
        })?;
        let one = layout.storage().map_err(|error| {
            PortableSessionError::Backend(format!("FP4 shadow storage failed: {error}"))
        })?;
        // Sum identical per-layer blocks.
        let mut total = one;
        for _ in 1..self.layers {
            total.logical_values = total.logical_values.checked_add(one.logical_values).ok_or(
                PortableSessionError::Invalid("FP4 shadow logical_values overflow"),
            )?;
            total.groups = total
                .groups
                .checked_add(one.groups)
                .ok_or(PortableSessionError::Invalid("FP4 shadow groups overflow"))?;
            total.code_bytes = total.code_bytes.checked_add(one.code_bytes).ok_or(
                PortableSessionError::Invalid("FP4 shadow code_bytes overflow"),
            )?;
            total.scale_bytes = total.scale_bytes.checked_add(one.scale_bytes).ok_or(
                PortableSessionError::Invalid("FP4 shadow scale_bytes overflow"),
            )?;
            total.padding_values = total.padding_values.checked_add(one.padding_values).ok_or(
                PortableSessionError::Invalid("FP4 shadow padding_values overflow"),
            )?;
            total.metadata_bytes = total.metadata_bytes.checked_add(one.metadata_bytes).ok_or(
                PortableSessionError::Invalid("FP4 shadow metadata_bytes overflow"),
            )?;
            total.total_bytes = total.total_bytes.checked_add(one.total_bytes).ok_or(
                PortableSessionError::Invalid("FP4 shadow total_bytes overflow"),
            )?;
        }
        Ok(Some(total))
    }

    /// Compose dense + FP4 telemetry for `cache` and this shadow.
    pub fn telemetry(
        &self,
        cache: &PortableKvCacheV1,
    ) -> Result<PortableKvStorageTelemetryV1, PortableSessionError> {
        let mut telemetry = cache.dense_telemetry();
        telemetry.mode = PortableKvStorageModeV1::Fp4E2M1 {
            group_size: self.group_size,
            scale_encoding: self.scale_encoding,
        };
        telemetry.fp4_logical = self.storage_for_rows(self.length)?;
        telemetry.fp4_capacity = self.storage_for_rows(self.capacity)?;
        Ok(telemetry)
    }
}
