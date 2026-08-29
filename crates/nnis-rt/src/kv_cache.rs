//! Device-resident key/value cache for autoregressive decoder execution.
//!
//! Storage is laid out as `[layer][head][capacity][head_dim]`. Each head owns
//! one fixed-capacity contiguous region, so appending new tokens only copies the
//! new suffix into place; existing cache contents are never recopied.

use crate::async_work::PendingGpuWork;
use crate::{DeviceBuffer, DevicePod, NnisError, Result, Stream};
use nnis_sys::{driver, CUdeviceptr};
use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;

/// Shape of an owned device-resident KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheConfig {
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub capacity: usize,
}

impl KvCacheConfig {
    pub fn new(layers: usize, heads: usize, head_dim: usize, capacity: usize) -> Result<Self> {
        let config = Self {
            layers,
            heads,
            head_dim,
            capacity,
        };
        config.validate()?;
        Ok(config)
    }

    /// Total elements in either the K or V allocation.
    pub fn elements_per_side(&self) -> Result<usize> {
        self.layers
            .checked_mul(self.heads)
            .and_then(|value| value.checked_mul(self.capacity))
            .and_then(|value| value.checked_mul(self.head_dim))
            .ok_or_else(|| NnisError::invalid_input("KV cache shape overflows usize"))
    }

    fn validate(&self) -> Result<()> {
        if self.layers == 0 || self.heads == 0 || self.head_dim == 0 || self.capacity == 0 {
            return Err(NnisError::invalid_input(format!(
                "KV cache dimensions must be non-zero; got layers={}, heads={}, head_dim={}, capacity={}",
                self.layers, self.heads, self.head_dim, self.capacity
            )));
        }
        let _ = self.elements_per_side()?;
        Ok(())
    }
}

/// Fixed-capacity, device-resident KV storage bound to one CUDA stream.
///
/// The cache owns one K and one V allocation for its full lifetime. Logical
/// lengths are tracked independently per layer. Appends are stream ordered and
/// copy only the newly produced `[heads][tokens][head_dim]` suffix.
pub struct KvCache<T: DevicePod> {
    config: KvCacheConfig,
    stream: Stream,
    keys: Arc<DeviceBuffer<T>>,
    values: Arc<DeviceBuffer<T>>,
    lengths: Vec<usize>,
}

impl<T: DevicePod> core::fmt::Debug for KvCache<T> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("KvCache")
            .field("config", &self.config)
            .field("lengths", &self.lengths)
            .finish_non_exhaustive()
    }
}

impl<T: DevicePod> KvCache<T> {
    /// Allocate an uninitialized cache on `stream`'s context.
    ///
    /// Unwritten capacity is intentionally left uninitialized: the logical
    /// per-layer lengths are the authority for what may be consumed.
    pub fn new(stream: &Stream, config: KvCacheConfig) -> Result<Self> {
        config.validate()?;
        let elements = config.elements_per_side()?;
        let keys = Arc::new(DeviceBuffer::new(stream.ctx(), elements)?);
        let values = Arc::new(DeviceBuffer::new(stream.ctx(), elements)?);
        Ok(Self {
            config,
            stream: stream.clone(),
            keys,
            values,
            lengths: vec![0; config.layers],
        })
    }

    pub fn config(&self) -> KvCacheConfig {
        self.config
    }

    /// Stream that orders every append into this cache.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    pub fn keys(&self) -> &DeviceBuffer<T> {
        &self.keys
    }

    pub fn values(&self) -> &DeviceBuffer<T> {
        &self.values
    }

    /// Shared ownership for higher-level owned asynchronous operation records.
    pub fn keys_owned(&self) -> Arc<DeviceBuffer<T>> {
        Arc::clone(&self.keys)
    }

    /// Shared ownership for higher-level owned asynchronous operation records.
    pub fn values_owned(&self) -> Arc<DeviceBuffer<T>> {
        Arc::clone(&self.values)
    }

    /// Number of valid token positions currently stored for `layer`.
    pub fn len(&self, layer: usize) -> Result<usize> {
        self.lengths.get(layer).copied().ok_or_else(|| {
            NnisError::invalid_input(format!("KV cache layer {layer} is out of range"))
        })
    }

    pub fn remaining(&self, layer: usize) -> Result<usize> {
        Ok(self.config.capacity - self.len(layer)?)
    }

    /// Active contiguous element range for one layer/head in the flat cache.
    pub fn head_range(&self, layer: usize, head: usize) -> Result<Range<usize>> {
        let length = self.len(layer)?;
        if head >= self.config.heads {
            return Err(NnisError::invalid_input(format!(
                "KV cache head {head} is out of range for {} heads",
                self.config.heads
            )));
        }
        let start = self.head_base_elements(layer, head)?;
        let active = length
            .checked_mul(self.config.head_dim)
            .ok_or_else(|| NnisError::invalid_input("KV cache active range overflows usize"))?;
        let end = start
            .checked_add(active)
            .ok_or_else(|| NnisError::invalid_input("KV cache active range overflows usize"))?;
        Ok(start..end)
    }

    /// Reset logical lengths without clearing device memory.
    pub fn reset(&mut self) {
        self.lengths.fill(0);
    }

    /// Reset one layer without clearing device memory.
    pub fn reset_layer(&mut self, layer: usize) -> Result<()> {
        let length = self.lengths.get_mut(layer).ok_or_else(|| {
            NnisError::invalid_input(format!("KV cache layer {layer} is out of range"))
        })?;
        *length = 0;
        Ok(())
    }

    /// Append packed `[heads][tokens][head_dim]` K/V tensors and wait.
    pub fn append_layer(
        &mut self,
        layer: usize,
        keys: Arc<DeviceBuffer<T>>,
        values: Arc<DeviceBuffer<T>>,
        tokens: usize,
    ) -> Result<()> {
        self.append_layer_async(layer, keys, values, tokens)?.wait()
    }

    /// Append packed `[heads][tokens][head_dim]` K/V tensors without a host
    /// synchronization.
    ///
    /// The returned [`KvAppend`] owns both source tensors plus shared ownership
    /// of the destination cache allocations until a completion event recorded
    /// after every D2D copy has fired. Dropping the handle early waits rather
    /// than freeing memory still referenced by CUDA.
    pub fn append_layer_async(
        &mut self,
        layer: usize,
        keys: Arc<DeviceBuffer<T>>,
        values: Arc<DeviceBuffer<T>>,
        tokens: usize,
    ) -> Result<KvAppend<T>> {
        let start = self.len(layer)?;
        let end = start
            .checked_add(tokens)
            .ok_or_else(|| NnisError::invalid_input("KV cache position overflows usize"))?;
        if end > self.config.capacity {
            return Err(NnisError::invalid_input(format!(
                "KV cache overflow at layer {layer}: {start} + {tokens} tokens exceeds capacity {}",
                self.config.capacity
            )));
        }
        let expected = self
            .config
            .heads
            .checked_mul(tokens)
            .and_then(|value| value.checked_mul(self.config.head_dim))
            .ok_or_else(|| NnisError::invalid_input("KV append shape overflows usize"))?;
        if keys.len() != expected || values.len() != expected {
            return Err(NnisError::invalid_input(format!(
                "KV append source lengths are {}/{} elements; {} heads x {tokens} tokens x {} dimensions requires {expected}",
                keys.len(),
                values.len(),
                self.config.heads,
                self.config.head_dim
            )));
        }
        let context = self.stream.ctx();
        if !Arc::ptr_eq(context, keys.ctx()) || !Arc::ptr_eq(context, values.ctx()) {
            return Err(NnisError::invalid_input(
                "KV cache and append sources must belong to one CUDA context",
            ));
        }
        for source in [&*keys, &*values] {
            if source.device_ptr() == self.keys.device_ptr()
                || source.device_ptr() == self.values.device_ptr()
            {
                return Err(NnisError::invalid_input(
                    "KV append sources must not alias the cache allocations",
                ));
            }
        }

        let resources = KvAppendResources {
            source_keys: keys,
            source_values: values,
            cache_keys: Arc::clone(&self.keys),
            cache_values: Arc::clone(&self.values),
        };

        let elements_per_head = tokens
            .checked_mul(self.config.head_dim)
            .ok_or_else(|| NnisError::invalid_input("KV append head size overflows usize"))?;
        let bytes_per_head = elements_per_head
            .checked_mul(size_of::<T>())
            .ok_or_else(|| NnisError::invalid_input("KV append byte size overflows usize"))?;

        // Build and validate the entire transfer plan before submitting the
        // first CUDA operation. After submission begins, no ordinary Rust
        // validation error is allowed to drop the ownership graph early.
        let mut copies = Vec::with_capacity(self.config.heads);
        for head in 0..self.config.heads {
            let source_offset = head
                .checked_mul(elements_per_head)
                .ok_or_else(|| NnisError::invalid_input("KV source offset overflows usize"))?;
            let destination_offset = self.append_base_elements(layer, head, start)?;
            copies.push(KvHeadCopy {
                source_key: Self::device_region_address(
                    &resources.source_keys,
                    source_offset,
                    elements_per_head,
                )?,
                source_value: Self::device_region_address(
                    &resources.source_values,
                    source_offset,
                    elements_per_head,
                )?,
                destination_key: Self::device_region_address(
                    &resources.cache_keys,
                    destination_offset,
                    elements_per_head,
                )?,
                destination_value: Self::device_region_address(
                    &resources.cache_values,
                    destination_offset,
                    elements_per_head,
                )?,
            });
        }

        if bytes_per_head != 0 {
            context.set_current()?;
            let api = driver::api()?;
            for (head, copy) in copies.into_iter().enumerate() {
                // SAFETY: the complete plan was range-validated before any
                // submission. All addresses refer to allocations retained by
                // `resources`, the context is current, and the stream belongs
                // to that context.
                let key_rc = unsafe {
                    (api.cuMemcpyAsync)(
                        copy.destination_key,
                        copy.source_key,
                        bytes_per_head,
                        self.stream.raw(),
                    )
                };
                if key_rc != 0 {
                    return Self::append_error(
                        &self.stream,
                        resources,
                        NnisError::driver("cuMemcpyAsync(KV key append)", key_rc)
                            .with("layer", layer)
                            .with("head", head)
                            .with("tokens", tokens),
                    );
                }
                // SAFETY: same proof as the key copy above.
                let value_rc = unsafe {
                    (api.cuMemcpyAsync)(
                        copy.destination_value,
                        copy.source_value,
                        bytes_per_head,
                        self.stream.raw(),
                    )
                };
                if value_rc != 0 {
                    return Self::append_error(
                        &self.stream,
                        resources,
                        NnisError::driver("cuMemcpyAsync(KV value append)", value_rc)
                            .with("layer", layer)
                            .with("head", head)
                            .with("tokens", tokens),
                    );
                }
            }
        }

        // SAFETY: `resources` owns the source K/V buffers and cloned ownership
        // of both destination cache allocations referenced by every copy just
        // enqueued above. `PendingGpuWork` also retains the bound stream.
        let work = unsafe { PendingGpuWork::from_enqueued(&self.stream, resources)? };
        self.lengths[layer] = end;
        Ok(KvAppend {
            work,
            layer,
            start,
            tokens,
        })
    }

    fn head_base_elements(&self, layer: usize, head: usize) -> Result<usize> {
        if layer >= self.config.layers {
            return Err(NnisError::invalid_input(format!(
                "KV cache layer {layer} is out of range for {} layers",
                self.config.layers
            )));
        }
        if head >= self.config.heads {
            return Err(NnisError::invalid_input(format!(
                "KV cache head {head} is out of range for {} heads",
                self.config.heads
            )));
        }
        layer
            .checked_mul(self.config.heads)
            .and_then(|value| value.checked_add(head))
            .and_then(|value| value.checked_mul(self.config.capacity))
            .and_then(|value| value.checked_mul(self.config.head_dim))
            .ok_or_else(|| NnisError::invalid_input("KV cache offset overflows usize"))
    }

    fn append_base_elements(&self, layer: usize, head: usize, position: usize) -> Result<usize> {
        self.head_base_elements(layer, head)?
            .checked_add(position.checked_mul(self.config.head_dim).ok_or_else(|| {
                NnisError::invalid_input("KV cache position offset overflows usize")
            })?)
            .ok_or_else(|| NnisError::invalid_input("KV cache append offset overflows usize"))
    }

    fn device_region_address(
        buffer: &DeviceBuffer<T>,
        element_offset: usize,
        elements: usize,
    ) -> Result<CUdeviceptr> {
        let end = element_offset
            .checked_add(elements)
            .ok_or_else(|| NnisError::invalid_input("device element range overflows usize"))?;
        if end > buffer.len() {
            return Err(NnisError::invalid_input(format!(
                "device region {element_offset}..{end} exceeds buffer length {}",
                buffer.len()
            )));
        }
        let byte_offset = element_offset
            .checked_mul(size_of::<T>())
            .ok_or_else(|| NnisError::invalid_input("device byte offset overflows usize"))?;
        let byte_offset = u64::try_from(byte_offset)
            .map_err(|_| NnisError::invalid_input("device byte offset exceeds u64"))?;
        let address = buffer
            .device_ptr()
            .checked_add(byte_offset)
            .ok_or_else(|| NnisError::invalid_input("device address overflows u64"))?;
        Ok(address as CUdeviceptr)
    }

    fn append_error(
        stream: &Stream,
        resources: KvAppendResources<T>,
        error: NnisError,
    ) -> Result<KvAppend<T>> {
        if stream.synchronize().is_err() {
            // The driver did not prove that earlier copies stopped using these
            // allocations. Leak the ownership graph rather than risk device
            // use-after-free while returning the original submission error.
            std::mem::forget(resources);
        }
        Err(error)
    }
}

#[derive(Clone, Copy)]
struct KvHeadCopy {
    source_key: CUdeviceptr,
    source_value: CUdeviceptr,
    destination_key: CUdeviceptr,
    destination_value: CUdeviceptr,
}

struct KvAppendResources<T: DevicePod> {
    source_keys: Arc<DeviceBuffer<T>>,
    source_values: Arc<DeviceBuffer<T>>,
    cache_keys: Arc<DeviceBuffer<T>>,
    cache_values: Arc<DeviceBuffer<T>>,
}

/// Completion handle for one asynchronous KV-cache append.
pub struct KvAppend<T: DevicePod> {
    work: PendingGpuWork<KvAppendResources<T>>,
    layer: usize,
    start: usize,
    tokens: usize,
}

impl<T: DevicePod> core::fmt::Debug for KvAppend<T> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("KvAppend")
            .field("layer", &self.layer)
            .field("start", &self.start)
            .field("tokens", &self.tokens)
            .finish_non_exhaustive()
    }
}

impl<T: DevicePod> KvAppend<T> {
    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn start_position(&self) -> usize {
        self.start
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub fn query(&self) -> Result<bool> {
        self.work.query()
    }

    pub fn wait(self) -> Result<()> {
        let _resources = self.work.wait()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_context;

    #[test]
    fn config_rejects_zero_and_overflow_shapes() {
        for config in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
            assert!(KvCacheConfig::new(config.0, config.1, config.2, config.3).is_err());
        }
        assert!(KvCacheConfig::new(usize::MAX, 2, 2, 2).is_err());
    }

    #[test]
    fn append_grows_in_place_and_reset_reuses_capacity_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let stream = Stream::new(&context).unwrap();
        let config = KvCacheConfig::new(2, 2, 3, 4).unwrap();
        let mut cache = KvCache::<f32>::new(&stream, config).unwrap();

        let first_keys = Arc::new(
            DeviceBuffer::from_host(
                &context,
                &stream,
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                ],
            )
            .unwrap(),
        );
        let first_values = Arc::new(
            DeviceBuffer::from_host(
                &context,
                &stream,
                &[
                    101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 111.0, 112.0, 113.0, 114.0, 115.0,
                    116.0,
                ],
            )
            .unwrap(),
        );
        let pending = cache
            .append_layer_async(0, first_keys, first_values, 2)
            .unwrap();
        assert_eq!(pending.start_position(), 0);
        assert_eq!(pending.tokens(), 2);
        assert_eq!(cache.len(0).unwrap(), 2);
        pending.wait().unwrap();

        let second_keys = Arc::new(
            DeviceBuffer::from_host(&context, &stream, &[7.0, 8.0, 9.0, 17.0, 18.0, 19.0]).unwrap(),
        );
        let second_values = Arc::new(
            DeviceBuffer::from_host(
                &context,
                &stream,
                &[107.0, 108.0, 109.0, 117.0, 118.0, 119.0],
            )
            .unwrap(),
        );
        cache
            .append_layer(0, second_keys, second_values, 1)
            .unwrap();
        assert_eq!(cache.len(0).unwrap(), 3);
        assert_eq!(cache.remaining(0).unwrap(), 1);

        let keys = cache.keys().to_vec(&stream).unwrap();
        let values = cache.values().to_vec(&stream).unwrap();
        // Layer 0, head 0 occupies elements 0..12; only positions 0..3 are valid.
        assert_eq!(&keys[0..9], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(
            &values[0..9],
            &[101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
        );
        // Layer 0, head 1 starts after head 0's full capacity of 4x3 elements.
        assert_eq!(
            &keys[12..21],
            &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        );
        assert_eq!(cache.head_range(0, 1).unwrap(), 12..21);

        let overflow_keys = Arc::new(DeviceBuffer::<f32>::new(&context, 12).unwrap());
        let overflow_values = Arc::new(DeviceBuffer::<f32>::new(&context, 12).unwrap());
        let error = cache
            .append_layer_async(0, overflow_keys, overflow_values, 2)
            .unwrap_err();
        assert!(error.to_string().contains("KV cache overflow"), "{error}");
        assert_eq!(cache.len(0).unwrap(), 3);

        cache.reset_layer(0).unwrap();
        assert_eq!(cache.len(0).unwrap(), 0);
        cache.reset();
        assert_eq!(cache.len(1).unwrap(), 0);
    }
}
