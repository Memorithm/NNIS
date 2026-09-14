//! Correctness-first physical compaction for the owned KV cache.
//!
//! Compaction keeps an explicit ordered subset of already materialized K/V
//! rows. The source cache is never mutated while the replacement is built: rows
//! are staged through separate device allocations and appended into a fresh
//! cache. Only a fully constructed replacement may be committed by a caller.
//!
//! This module operates on physical cache rows. It does not reinterpret token
//! identities or RoPE positions; higher-level runtimes must preserve those
//! logical-position semantics separately.

use crate::{DeviceBuffer, DevicePod, KvCache, NnisError, Result};
use nnis_sys::{driver, CUdeviceptr};
use std::mem::size_of;
use std::sync::Arc;

/// Build a new cache containing only `retained_positions` from every layer.
///
/// Positions address the current active prefix and must be strictly increasing
/// and unique. The source cache is left untouched on both success and failure.
/// Empty input produces an empty replacement cache with the same capacity and
/// stream binding.
pub fn compact_kv_cache<T: DevicePod>(
    cache: &KvCache<T>,
    retained_positions: &[usize],
) -> Result<KvCache<T>> {
    let config = cache.config();
    let mut selections = Vec::with_capacity(config.layers);
    for layer in 0..config.layers {
        let active = cache.len(layer)?;
        validate_retained_positions(active, retained_positions)?;
        selections.push(retained_positions.to_vec());
    }
    rebuild_cache(cache, &selections)
}

/// Transactionally compact one cache layer while preserving every other layer.
///
/// The complete replacement cache is constructed first. `cache` is swapped only
/// after every selected row has been copied and synchronized successfully, so a
/// failed compaction never leaves the original cache with a shortened logical
/// length.
pub fn compact_kv_cache_layer<T: DevicePod>(
    cache: &mut KvCache<T>,
    layer: usize,
    retained_positions: &[usize],
) -> Result<()> {
    let config = cache.config();
    if layer >= config.layers {
        return Err(NnisError::invalid_input(format!(
            "KV compaction layer {layer} is out of range for {} layers",
            config.layers
        )));
    }

    let mut selections = Vec::with_capacity(config.layers);
    for current_layer in 0..config.layers {
        let active = cache.len(current_layer)?;
        let selection = if current_layer == layer {
            validate_retained_positions(active, retained_positions)?;
            retained_positions.to_vec()
        } else {
            (0..active).collect()
        };
        selections.push(selection);
    }

    let replacement = rebuild_cache(cache, &selections)?;
    *cache = replacement;
    Ok(())
}

fn rebuild_cache<T: DevicePod>(
    source: &KvCache<T>,
    selections: &[Vec<usize>],
) -> Result<KvCache<T>> {
    let config = source.config();
    if selections.len() != config.layers {
        return Err(NnisError::invalid_input(format!(
            "KV compaction requires one selection per layer; got {} for {} layers",
            selections.len(),
            config.layers
        )));
    }

    // Finish all prior work touching the source before capturing rows. The
    // source remains valid and unchanged if any later allocation/copy fails.
    source.stream().synchronize()?;
    let mut replacement = KvCache::<T>::new(source.stream(), config)?;

    for (layer, retained_positions) in selections.iter().enumerate() {
        let active = source.len(layer)?;
        validate_retained_positions(active, retained_positions)?;
        if retained_positions.is_empty() {
            continue;
        }

        let retained = retained_positions.len();
        let scratch_elements = config
            .heads
            .checked_mul(retained)
            .and_then(|value| value.checked_mul(config.head_dim))
            .ok_or_else(|| {
                NnisError::invalid_input("KV compaction scratch shape overflows usize")
            })?;
        let scratch_keys = Arc::new(DeviceBuffer::<T>::new(
            source.stream().ctx(),
            scratch_elements,
        )?);
        let scratch_values = Arc::new(DeviceBuffer::<T>::new(
            source.stream().ctx(),
            scratch_elements,
        )?);

        stage_selected_rows(
            source,
            layer,
            retained_positions,
            &scratch_keys,
            &scratch_values,
        )?;

        // `append_layer` owns cache-length accounting. It writes only into the
        // fresh replacement, so a failure cannot corrupt the source cache.
        replacement.append_layer(layer, scratch_keys, scratch_values, retained)?;
    }

    Ok(replacement)
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
    let row_elements = config.head_dim;
    let row_bytes = row_elements
        .checked_mul(size_of::<T>())
        .ok_or_else(|| NnisError::invalid_input("KV compaction row size overflows usize"))?;
    let retained = retained_positions.len();
    let stream = cache.stream();
    let context = stream.ctx();

    // Resolve and range-check every transfer before submitting the first CUDA
    // operation. Ordinary validation failures therefore cannot strand a
    // partially submitted selection plan.
    let mut copies = Vec::with_capacity(config.heads.saturating_mul(retained));
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
            copies.push(KvRowCopy {
                source_key: device_region_address(cache.keys(), source_element, row_elements)?,
                source_value: device_region_address(cache.values(), source_element, row_elements)?,
                destination_key: device_region_address(
                    scratch_keys.as_ref(),
                    destination_element,
                    row_elements,
                )?,
                destination_value: device_region_address(
                    scratch_values.as_ref(),
                    destination_element,
                    row_elements,
                )?,
                head,
                source_position,
                destination_position,
            });
        }
    }

    context.set_current()?;
    let api = driver::api()?;
    for copy in copies {
        // SAFETY: every region was range-validated before submission, all
        // allocations remain borrowed here, and `stream` belongs to `context`.
        let key_rc = unsafe {
            (api.cuMemcpyAsync)(
                copy.destination_key,
                copy.source_key,
                row_bytes,
                stream.raw(),
            )
        };
        if key_rc != 0 {
            let error = NnisError::driver("cuMemcpyAsync(KV compaction key stage)", key_rc)
                .with("layer", layer)
                .with("head", copy.head)
                .with("source_position", copy.source_position)
                .with("destination_position", copy.destination_position);
            return stage_error(stream, scratch_keys, scratch_values, error);
        }

        // SAFETY: same proof as the key transfer above.
        let value_rc = unsafe {
            (api.cuMemcpyAsync)(
                copy.destination_value,
                copy.source_value,
                row_bytes,
                stream.raw(),
            )
        };
        if value_rc != 0 {
            let error = NnisError::driver("cuMemcpyAsync(KV compaction value stage)", value_rc)
                .with("layer", layer)
                .with("head", copy.head)
                .with("source_position", copy.source_position)
                .with("destination_position", copy.destination_position);
            return stage_error(stream, scratch_keys, scratch_values, error);
        }
    }

    if let Err(error) = stream.synchronize() {
        retain_scratch_on_failed_sync(scratch_keys, scratch_values);
        return Err(error.with("operation", "KV compaction staging synchronization"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct KvRowCopy {
    source_key: CUdeviceptr,
    source_value: CUdeviceptr,
    destination_key: CUdeviceptr,
    destination_value: CUdeviceptr,
    head: usize,
    source_position: usize,
    destination_position: usize,
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

fn device_region_address<T>(
    buffer: &DeviceBuffer<T>,
    element_offset: usize,
    elements: usize,
) -> Result<CUdeviceptr> {
    let end = element_offset
        .checked_add(elements)
        .ok_or_else(|| NnisError::invalid_input("KV compaction region overflows usize"))?;
    if end > buffer.len() {
        return Err(NnisError::invalid_input(format!(
            "KV compaction region {element_offset}..{end} exceeds buffer length {}",
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
    fn replacement_compaction_preserves_source_and_selected_rows_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let mut source =
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
        source.append_layer(0, keys, values, 4).unwrap();

        let compacted = compact_kv_cache(&source, &[0, 2]).unwrap();
        assert_eq!(source.len(0).unwrap(), 4);
        assert_eq!(compacted.len(0).unwrap(), 2);

        let compacted_keys = compacted.keys().to_vec(&stream).unwrap();
        let compacted_values = compacted.values().to_vec(&stream).unwrap();
        assert_eq!(&compacted_keys[0..4], &[10.0, 11.0, 30.0, 31.0]);
        assert_eq!(&compacted_keys[8..12], &[110.0, 111.0, 130.0, 131.0]);
        assert_eq!(&compacted_values[0..4], &[1010.0, 1011.0, 1030.0, 1031.0]);
        assert_eq!(&compacted_values[8..12], &[1110.0, 1111.0, 1130.0, 1131.0]);

        let source_keys = source.keys().to_vec(&stream).unwrap();
        assert_eq!(&source_keys[0..8], &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0]);
    }
}