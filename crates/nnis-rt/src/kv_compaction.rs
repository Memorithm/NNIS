//! Correctness-first physical compaction for the owned KV cache.
//!
//! A compaction keeps an explicit ordered subset of the currently active KV
//! rows. K/V values are copied to device scratch first, then the selected rows
//! are written back as the new contiguous active prefix through the public
//! `KvCache` append path. This module does not reinterpret logical token
//! positions or RoPE positions; callers remain responsible for preserving any
//! higher-level logical-position state.

use crate::{DeviceBuffer, DevicePod, KvCache, NnisError, Result};
use nnis_sys::{driver, CUdeviceptr};
use std::mem::size_of;
use std::sync::Arc;

/// Compact one KV-cache layer to the explicit active-row positions supplied by
/// the caller.
///
/// `retained_positions` must be strictly increasing, unique, and address rows
/// in the layer's current active prefix. Empty input drops every active row for
/// the layer. Supplying the exact identity selection is a synchronized no-op.
///
/// The operation is synchronous and correctness-first. Selected K/V rows are
/// staged into separate device allocations before the cache prefix is rebuilt,
/// so source and destination regions never overlap during selection capture.
pub fn compact_kv_cache_layer<T: DevicePod>(
    cache: &mut KvCache<T>,
    layer: usize,
    retained_positions: &[usize],
) -> Result<()> {
    let active = cache.len(layer)?;
    validate_retained_positions(active, retained_positions)?;

    if retained_positions.len() == active && retained_positions.iter().copied().eq(0..active) {
        return cache.stream().synchronize();
    }

    if retained_positions.is_empty() {
        cache.stream().synchronize()?;
        return cache.reset_layer(layer);
    }

    let config = cache.config();
    let retained = retained_positions.len();
    let scratch_elements = compaction_scratch_elements(config.heads, retained, config.head_dim)?;
    let scratch_keys = Arc::new(DeviceBuffer::<T>::new(
        cache.stream().ctx(),
        scratch_elements,
    )?);
    let scratch_values = Arc::new(DeviceBuffer::<T>::new(
        cache.stream().ctx(),
        scratch_elements,
    )?);

    stage_selected_rows(
        cache,
        layer,
        retained_positions,
        &scratch_keys,
        &scratch_values,
    )?;

    // The scratch capture is complete before logical length changes. Rebuild
    // the active prefix through the existing append path so cache accounting
    // remains owned by KvCache rather than duplicated here.
    cache.reset_layer(layer)?;
    cache.append_layer(layer, scratch_keys, scratch_values, retained)
}

/// Compact every layer of a KV cache to the same explicit active-row selection.
///
/// All layers must begin with the same active length. For non-empty selections,
/// a complete replacement cache is built first and swapped into place only
/// after every layer has been staged and appended successfully. Ordinary
/// validation or CUDA failures before that swap leave the original cache
/// logically unchanged.
///
/// This operation changes active physical KV rows only. It does not represent
/// or mutate any model-level logical token/RoPE position.
pub fn compact_kv_cache<T: DevicePod>(
    cache: &mut KvCache<T>,
    retained_positions: &[usize],
) -> Result<()> {
    let config = cache.config();
    let active = cache.len(0)?;
    for layer in 1..config.layers {
        let layer_active = cache.len(layer)?;
        if layer_active != active {
            return Err(NnisError::invalid_input(format!(
                "whole-cache KV compaction requires uniform active lengths; layer 0 has {active} rows but layer {layer} has {layer_active}"
            )));
        }
    }
    validate_retained_positions(active, retained_positions)?;

    if retained_positions.len() == active && retained_positions.iter().copied().eq(0..active) {
        return cache.stream().synchronize();
    }

    if retained_positions.is_empty() {
        cache.stream().synchronize()?;
        cache.reset();
        return Ok(());
    }

    let retained = retained_positions.len();
    let scratch_elements = compaction_scratch_elements(config.heads, retained, config.head_dim)?;
    let mut replacement = KvCache::<T>::new(cache.stream(), config)?;

    for layer in 0..config.layers {
        let scratch_keys = Arc::new(DeviceBuffer::<T>::new(
            cache.stream().ctx(),
            scratch_elements,
        )?);
        let scratch_values = Arc::new(DeviceBuffer::<T>::new(
            cache.stream().ctx(),
            scratch_elements,
        )?);
        stage_selected_rows(
            cache,
            layer,
            retained_positions,
            &scratch_keys,
            &scratch_values,
        )?;
        replacement.append_layer(layer, scratch_keys, scratch_values, retained)?;
    }

    replacement.stream().synchronize()?;
    std::mem::swap(cache, &mut replacement);
    Ok(())
}

fn compaction_scratch_elements(heads: usize, retained: usize, head_dim: usize) -> Result<usize> {
    heads
        .checked_mul(retained)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| NnisError::invalid_input("KV compaction scratch shape overflows usize"))
}

fn validate_retained_positions(active: usize, retained_positions: &[usize]) -> Result<()> {
    let mut previous = None;
    for &position in retained_positions {
        if position >= active {
            return Err(NnisError::invalid_input(format!(
                "KV compaction position {position} is outside active prefix 0..{active}"
            )));
        }
        if previous.is_some_and(|last| position <= last) {
            return Err(NnisError::invalid_input(
                "KV compaction positions must be strictly increasing and unique",
            ));
        }
        previous = Some(position);
    }
    Ok(())
}

fn stage_selected_rows<T: DevicePod>(
    cache: &KvCache<T>,
    layer: usize,
    retained_positions: &[usize],
    scratch_keys: &Arc<DeviceBuffer<T>>,
    scratch_values: &Arc<DeviceBuffer<T>>,
) -> Result<()> {
    let config = cache.config();
    let row_bytes = config
        .head_dim
        .checked_mul(size_of::<T>())
        .ok_or_else(|| NnisError::invalid_input("KV compaction row size overflows usize"))?;
    let retained = retained_positions.len();
    let stream = cache.stream();
    let context = stream.ctx();
    context.set_current()?;
    let api = driver::api()?;

    for head in 0..config.heads {
        for (destination_position, &source_position) in retained_positions.iter().enumerate() {
            let source_element = layer
                .checked_mul(config.heads)
                .and_then(|value| value.checked_add(head))
                .and_then(|value| value.checked_mul(config.capacity))
                .and_then(|value| value.checked_add(source_position))
                .and_then(|value| value.checked_mul(config.head_dim))
                .ok_or_else(|| {
                    NnisError::invalid_input("KV compaction source offset overflows usize")
                })?;
            let destination_element = head
                .checked_mul(retained)
                .and_then(|value| value.checked_add(destination_position))
                .and_then(|value| value.checked_mul(config.head_dim))
                .ok_or_else(|| {
                    NnisError::invalid_input("KV compaction destination offset overflows usize")
                })?;

            let source_key = device_address(cache.keys(), source_element)?;
            let source_value = device_address(cache.values(), source_element)?;
            let destination_key = device_address(scratch_keys.as_ref(), destination_element)?;
            let destination_value = device_address(scratch_values.as_ref(), destination_element)?;

            // SAFETY: all offsets were checked against live allocations, the
            // owning CUDA context is current, and scratch/cache buffers remain
            // borrowed until the synchronization below.
            let key_rc = unsafe {
                (api.cuMemcpyAsync)(destination_key, source_key, row_bytes, stream.raw())
            };
            if key_rc != 0 {
                let error = NnisError::driver("cuMemcpyAsync(KV compaction key stage)", key_rc)
                    .with("layer", layer)
                    .with("head", head)
                    .with("source_position", source_position)
                    .with("destination_position", destination_position);
                return stage_error(stream, scratch_keys, scratch_values, error);
            }
            // SAFETY: same proof as the key transfer above.
            let value_rc = unsafe {
                (api.cuMemcpyAsync)(destination_value, source_value, row_bytes, stream.raw())
            };
            if value_rc != 0 {
                let error = NnisError::driver("cuMemcpyAsync(KV compaction value stage)", value_rc)
                    .with("layer", layer)
                    .with("head", head)
                    .with("source_position", source_position)
                    .with("destination_position", destination_position);
                return stage_error(stream, scratch_keys, scratch_values, error);
            }
        }
    }

    if let Err(error) = stream.synchronize() {
        retain_scratch_on_failed_sync(scratch_keys, scratch_values);
        return Err(error.with("operation", "KV compaction staging synchronization"));
    }
    Ok(())
}

fn stage_error<T: DevicePod>(
    stream: &crate::Stream,
    scratch_keys: &Arc<DeviceBuffer<T>>,
    scratch_values: &Arc<DeviceBuffer<T>>,
    error: NnisError,
) -> Result<()> {
    if stream.synchronize().is_err() {
        retain_scratch_on_failed_sync(scratch_keys, scratch_values);
        return Err(error.with("synchronization", "failed after compaction staging error"));
    }
    Err(error)
}

fn retain_scratch_on_failed_sync<T: DevicePod>(
    scratch_keys: &Arc<DeviceBuffer<T>>,
    scratch_values: &Arc<DeviceBuffer<T>>,
) {
    // CUDA did not prove prior transfers stopped touching these allocations.
    // Leak one retained ownership reference for each destination rather than
    // risk freeing device memory still referenced by the driver.
    std::mem::forget(Arc::clone(scratch_keys));
    std::mem::forget(Arc::clone(scratch_values));
}

fn device_address<T>(buffer: &DeviceBuffer<T>, element_offset: usize) -> Result<CUdeviceptr> {
    if element_offset > buffer.len() {
        return Err(NnisError::invalid_input(format!(
            "KV compaction element offset {element_offset} exceeds buffer length {}",
            buffer.len()
        )));
    }
    let byte_offset = element_offset
        .checked_mul(size_of::<T>())
        .ok_or_else(|| NnisError::invalid_input("KV compaction byte offset overflows usize"))?;
    let byte_offset = u64::try_from(byte_offset)
        .map_err(|_| NnisError::invalid_input("KV compaction byte offset exceeds u64"))?;
    buffer
        .device_ptr()
        .checked_add(byte_offset)
        .map(|address| address as CUdeviceptr)
        .ok_or_else(|| NnisError::invalid_input("KV compaction device address overflows u64"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gpu_context, KvCacheConfig, Stream};

    #[test]
    fn retained_positions_are_strict_and_bounded() {
        assert!(validate_retained_positions(4, &[]).is_ok());
        assert!(validate_retained_positions(4, &[0, 2, 3]).is_ok());
        assert!(validate_retained_positions(4, &[0, 0]).is_err());
        assert!(validate_retained_positions(4, &[2, 1]).is_err());
        assert!(validate_retained_positions(4, &[0, 4]).is_err());
    }

    #[test]
    fn compaction_preserves_selected_device_rows_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let mut cache =
            KvCache::<f32>::new(&stream, KvCacheConfig::new(1, 2, 2, 4).unwrap()).unwrap();
        let keys = Arc::new(
            DeviceBuffer::from_host(
                &context,
                &stream,
                &[
                    10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 110.0, 111.0, 120.0, 121.0,
                    130.0, 131.0, 140.0, 141.0,
                ],
            )
            .unwrap(),
        );
        let values = Arc::new(
            DeviceBuffer::from_host(
                &context,
                &stream,
                &[
                    1010.0, 1011.0, 1020.0, 1021.0, 1030.0, 1031.0, 1040.0, 1041.0, 1110.0, 1111.0,
                    1120.0, 1121.0, 1130.0, 1131.0, 1140.0, 1141.0,
                ],
            )
            .unwrap(),
        );
        cache.append_layer(0, keys, values, 4).unwrap();

        compact_kv_cache_layer(&mut cache, 0, &[0, 2]).unwrap();
        assert_eq!(cache.len(0).unwrap(), 2);

        let keys = cache.keys().to_vec(&stream).unwrap();
        let values = cache.values().to_vec(&stream).unwrap();
        assert_eq!(&keys[0..4], &[10.0, 11.0, 30.0, 31.0]);
        assert_eq!(&keys[8..12], &[110.0, 111.0, 130.0, 131.0]);
        assert_eq!(&values[0..4], &[1010.0, 1011.0, 1030.0, 1031.0]);
        assert_eq!(&values[8..12], &[1110.0, 1111.0, 1130.0, 1131.0]);
    }

    #[test]
    fn whole_cache_compaction_swaps_only_after_all_layers_are_built_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let mut cache =
            KvCache::<f32>::new(&stream, KvCacheConfig::new(2, 1, 2, 4).unwrap()).unwrap();

        for layer in 0..2 {
            let base = layer as f32 * 100.0;
            let keys = Arc::new(
                DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &[
                        base + 10.0,
                        base + 11.0,
                        base + 20.0,
                        base + 21.0,
                        base + 30.0,
                        base + 31.0,
                        base + 40.0,
                        base + 41.0,
                    ],
                )
                .unwrap(),
            );
            let values = Arc::new(
                DeviceBuffer::from_host(
                    &context,
                    &stream,
                    &[
                        base + 1010.0,
                        base + 1011.0,
                        base + 1020.0,
                        base + 1021.0,
                        base + 1030.0,
                        base + 1031.0,
                        base + 1040.0,
                        base + 1041.0,
                    ],
                )
                .unwrap(),
            );
            cache.append_layer(layer, keys, values, 4).unwrap();
        }

        compact_kv_cache(&mut cache, &[1, 3]).unwrap();
        assert_eq!(cache.len(0).unwrap(), 2);
        assert_eq!(cache.len(1).unwrap(), 2);

        let keys = cache.keys().to_vec(&stream).unwrap();
        let values = cache.values().to_vec(&stream).unwrap();
        assert_eq!(&keys[0..4], &[20.0, 21.0, 40.0, 41.0]);
        assert_eq!(&keys[8..12], &[120.0, 121.0, 140.0, 141.0]);
        assert_eq!(&values[0..4], &[1020.0, 1021.0, 1040.0, 1041.0]);
        assert_eq!(&values[8..12], &[1120.0, 1121.0, 1140.0, 1141.0]);
    }
}
