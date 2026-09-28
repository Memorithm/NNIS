//! DSV41-1/DSV41-2 WGPU counterparts of the CPU replay and KV-reuse references.
//!
//! [`WgpuReplaySourceV1`] owns an immutable device-resident dense row-major
//! F32 payload for one exact [`ReplaySourceIdentityV1`]. A replay validates
//! the request against the provider's current identity exactly like
//! `nnis_cpu::replay::CpuReplaySourceV1`, then copies the requested inclusive
//! logical window into a fresh device buffer with a WGPU buffer-to-buffer copy.
//! Row offsets are whole F32 words, so the copy is 4-byte aligned and never
//! staged through the host; no shader touches the values, so the replayed
//! bits (including `-0.0` and subnormals) equal the source bits.
//!
//! [`WgpuCrossLayerKvBindingV1`] mirrors `CpuCrossLayerKvBindingV1`: it binds
//! exactly one owner source to each owning (layer, component) of a validated
//! [`CrossLayerKvReusePlanV1`] and resolves reads for reusing layers to their
//! declared owner. It never infers, extends, or repairs reuse.
//!
//! Sharing is logical. Fewer bound sources is not a physical memory,
//! residency, quality, latency, or throughput result, and nothing here
//! performs compression, eviction, paging, or model-specific interpretation.

use core::fmt;

use nnis_core::kv_reuse_plan::{
    CrossLayerKvReusePlanV1, KvComponent, KvLayerSourceV1, KvReusePlanError,
};
use nnis_core::replay_state::{
    ReplayIdentityError, ReplaySourceIdentityV1, ReplayStateProviderV1, ReplayWindowRequestV1,
};
use nnis_core::{
    BufferDesc, BufferUsages, MemoryClass, PortableBuffer, PortableDevice, PortableError,
    PortableFence, PortableQueue,
};

use crate::memory::{WgpuBuffer, WgpuQueue};
use crate::numerical::WgpuF32KernelsV1;
use crate::WgpuDevice;

/// Buffer usages of replay payloads and replayed windows.
fn replay_usages() -> BufferUsages {
    BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
}

/// Immutable device-resident replay source bound to one exact source identity.
///
/// Row `i` of the payload holds the `row_width` values for logical position
/// `identity.logical_start_position() + i`. The payload buffer is private and
/// only exposed by shared reference, so it cannot be written after binding.
#[derive(Debug)]
pub struct WgpuReplaySourceV1 {
    identity: ReplaySourceIdentityV1,
    row_width: usize,
    rows: WgpuBuffer,
    queue: WgpuQueue,
}

impl WgpuReplaySourceV1 {
    /// Upload a dense finite row-major host payload bound to `identity`.
    ///
    /// Validation matches the CPU reference in order and variant: zero row
    /// width, logical-range overflow, payload length, then the first
    /// non-finite flat index.
    pub fn from_host(
        device: &WgpuDevice,
        identity: ReplaySourceIdentityV1,
        row_width: usize,
        rows: &[f32],
    ) -> Result<Self, WgpuReplayError> {
        let expected = expected_values(&identity, row_width)?;
        if rows.len() != expected {
            return Err(WgpuReplayError::PayloadLengthMismatch {
                expected,
                actual: rows.len(),
            });
        }
        if let Some(index) = rows.iter().position(|value| !value.is_finite()) {
            return Err(WgpuReplayError::NonFiniteValue { index });
        }
        let queue = device.create_queue()?;
        let mut buffer = device.create_buffer(replay_desc(expected)?)?;
        let bytes: Vec<u8> = rows.iter().flat_map(|value| value.to_le_bytes()).collect();
        queue.write_buffer(&mut buffer, 0, &bytes)?.wait()?;
        Ok(Self {
            identity,
            row_width,
            rows: buffer,
            queue,
        })
    }

    /// Bind an existing device buffer, taking ownership of it.
    ///
    /// The buffer must belong to `device`, carry `STORAGE | COPY_SRC`, hold
    /// exactly `items * row_width` F32 values, and contain only finite values
    /// (checked on the device; a failure reports
    /// [`WgpuReplayError::NonFiniteDeviceValue`] without an index).
    pub fn from_buffer(
        device: &WgpuDevice,
        identity: ReplaySourceIdentityV1,
        row_width: usize,
        rows: WgpuBuffer,
    ) -> Result<Self, WgpuReplayError> {
        let expected = expected_values(&identity, row_width)?;
        let queue = device.create_queue()?;
        queue.check_owner(&rows)?;
        let usages = rows.descriptor().usages;
        if !usages.contains(BufferUsages::STORAGE | BufferUsages::COPY_SRC) {
            return Err(WgpuReplayError::Portable(PortableError::Unsupported(
                "WGPU replay source buffer requires STORAGE and COPY_SRC usages".to_string(),
            )));
        }
        let expected_bytes = byte_len(expected)?;
        if rows.len() != expected_bytes {
            return Err(WgpuReplayError::PayloadByteLengthMismatch {
                expected_bytes,
                actual_bytes: rows.len(),
            });
        }
        match WgpuF32KernelsV1::new(device)?.check_finite(&rows) {
            Ok(()) => {}
            Err(PortableError::InvalidDescriptor(_)) => {
                return Err(WgpuReplayError::NonFiniteDeviceValue)
            }
            Err(error) => return Err(error.into()),
        }
        Ok(Self {
            identity,
            row_width,
            rows,
            queue,
        })
    }

    /// Number of F32 values stored per logical position.
    pub const fn row_width(&self) -> usize {
        self.row_width
    }

    /// Device payload for the full declared source range (read-only).
    pub const fn rows(&self) -> &WgpuBuffer {
        &self.rows
    }

    /// Copy the rows of a validated replay window into a new device buffer.
    ///
    /// Fails closed if the request was built for a different source identity,
    /// including any provider, source, generation, representation, epoch, or
    /// range drift. The returned buffer belongs to this source's device and
    /// has `STORAGE | COPY_SRC | COPY_DST` usages.
    pub fn replay_window(
        &self,
        request: &ReplayWindowRequestV1,
    ) -> Result<WgpuBuffer, WgpuReplayError> {
        self.validate_replay_window(request)?;
        let first_row = request
            .logical_start_position()
            .checked_sub(self.identity.logical_start_position())
            .ok_or(ReplayIdentityError::PositionOverflow)?;
        let first_row = host_index(first_row)?;
        let row_count = host_index(request.logical_items()?)?;
        let start = first_row
            .checked_mul(self.row_width)
            .ok_or(WgpuReplayError::HostIndexOverflow)?;
        let len = row_count
            .checked_mul(self.row_width)
            .ok_or(WgpuReplayError::HostIndexOverflow)?;
        let end = start
            .checked_add(len)
            .ok_or(WgpuReplayError::HostIndexOverflow)?;
        if byte_len(end)? > self.rows.len() {
            return Err(WgpuReplayError::HostIndexOverflow);
        }
        let mut window = self
            .queue
            .create_bounded_buffer(replay_desc(len)?, &self.rows)?;
        self.queue
            .copy_buffer(&self.rows, byte_len(start)?, &mut window, 0, byte_len(len)?)?
            .wait()?;
        Ok(window)
    }

    /// Resolve and copy the most recent `window_items` logical positions.
    ///
    /// `window_items` is caller-supplied policy; it is neither chosen nor
    /// clamped here.
    pub fn replay_recent_window(
        &self,
        window_items: u64,
    ) -> Result<(ReplayWindowRequestV1, WgpuBuffer), WgpuReplayError> {
        let request = ReplayWindowRequestV1::recent(self.identity.clone(), window_items)?;
        let rows = self.replay_window(&request)?;
        Ok((request, rows))
    }

    /// Read a replayed (or payload) buffer of this source's device to host F32.
    pub fn read_f32(&self, buffer: &WgpuBuffer) -> Result<Vec<f32>, WgpuReplayError> {
        let bytes = self.queue.read_buffer(buffer, 0, buffer.len())?;
        Ok(bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }
}

impl ReplayStateProviderV1 for WgpuReplaySourceV1 {
    fn replay_source_identity(&self) -> &ReplaySourceIdentityV1 {
        &self.identity
    }
}

fn expected_values(
    identity: &ReplaySourceIdentityV1,
    row_width: usize,
) -> Result<usize, WgpuReplayError> {
    if row_width == 0 {
        return Err(WgpuReplayError::ZeroRowWidth);
    }
    let items = host_index(identity.logical_items()?)?;
    items
        .checked_mul(row_width)
        .ok_or(WgpuReplayError::HostIndexOverflow)
}

fn byte_len(values: usize) -> Result<u64, WgpuReplayError> {
    u64::try_from(values)
        .ok()
        .and_then(|values| values.checked_mul(4))
        .ok_or(WgpuReplayError::HostIndexOverflow)
}

fn replay_desc(values: usize) -> Result<BufferDesc, WgpuReplayError> {
    Ok(BufferDesc::new(
        byte_len(values)?,
        replay_usages(),
        MemoryClass::DeviceLocal,
    )?)
}

fn host_index(value: u64) -> Result<usize, WgpuReplayError> {
    usize::try_from(value).map_err(|_| WgpuReplayError::HostIndexOverflow)
}

/// Fail-closed WGPU replay errors.
///
/// Variants shared with `CpuReplayError` carry identical payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgpuReplayError {
    /// Backend-neutral identity or window validation failed.
    Identity(ReplayIdentityError),
    /// A replay source must store at least one value per logical position.
    ZeroRowWidth,
    /// Host payload length did not match declared items times row width.
    PayloadLengthMismatch { expected: usize, actual: usize },
    /// Device payload byte length did not match declared items times row width.
    PayloadByteLengthMismatch {
        expected_bytes: u64,
        actual_bytes: u64,
    },
    /// Host payload contained a NaN or infinity at this flat index.
    NonFiniteValue { index: usize },
    /// Device payload contained a NaN or infinity (index not reported).
    NonFiniteDeviceValue,
    /// A logical count or offset does not fit host `usize` indexing.
    HostIndexOverflow,
    /// Portable WGPU allocation, copy, or readback failed.
    Portable(PortableError),
}

impl From<ReplayIdentityError> for WgpuReplayError {
    fn from(error: ReplayIdentityError) -> Self {
        Self::Identity(error)
    }
}

impl From<PortableError> for WgpuReplayError {
    fn from(error: PortableError) -> Self {
        Self::Portable(error)
    }
}

impl fmt::Display for WgpuReplayError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identity(error) => write!(output, "replay identity: {error}"),
            Self::ZeroRowWidth => output.write_str("WGPU replay row width must be non-zero"),
            Self::PayloadLengthMismatch { expected, actual } => write!(
                output,
                "WGPU replay payload has {actual} values, expected {expected}"
            ),
            Self::PayloadByteLengthMismatch {
                expected_bytes,
                actual_bytes,
            } => write!(
                output,
                "WGPU replay buffer has {actual_bytes} bytes, expected {expected_bytes}"
            ),
            Self::NonFiniteValue { index } => {
                write!(output, "WGPU replay payload value {index} is not finite")
            }
            Self::NonFiniteDeviceValue => {
                output.write_str("WGPU replay device payload contains a non-finite value")
            }
            Self::HostIndexOverflow => {
                output.write_str("WGPU replay logical range does not fit host indexing")
            }
            Self::Portable(error) => write!(output, "WGPU replay: {error}"),
        }
    }
}

impl std::error::Error for WgpuReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Identity(error) => Some(error),
            Self::Portable(error) => Some(error),
            _ => None,
        }
    }
}

/// Owner replay sources bound to a validated cross-layer KV reuse plan.
#[derive(Debug)]
pub struct WgpuCrossLayerKvBindingV1 {
    plan: CrossLayerKvReusePlanV1,
    keys: Vec<Option<WgpuReplaySourceV1>>,
    values: Vec<Option<WgpuReplaySourceV1>>,
}

impl WgpuCrossLayerKvBindingV1 {
    /// Bind one owner source per owning layer for each component.
    ///
    /// `key_sources` and `value_sources` are `(owning_layer, source)` pairs.
    /// Every owning layer must appear exactly once per component, no reusing
    /// layer may be given a source, and all sources must declare the same
    /// logical start and end positions. Checks run in the CPU reference order.
    pub fn new(
        plan: CrossLayerKvReusePlanV1,
        key_sources: Vec<(u32, WgpuReplaySourceV1)>,
        value_sources: Vec<(u32, WgpuReplaySourceV1)>,
    ) -> Result<Self, WgpuKvReuseError> {
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
                    return Err(WgpuKvReuseError::SourceRangeMismatch {
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
    ) -> Result<&WgpuReplaySourceV1, WgpuKvReuseError> {
        let owner = self.plan.owner(layer, component)?;
        let slots = match component {
            KvComponent::Key => &self.keys,
            KvComponent::Value => &self.values,
        };
        slots.get(owner as usize).and_then(Option::as_ref).ok_or(
            WgpuKvReuseError::MissingOwnerSource {
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
    ) -> Result<WgpuBuffer, WgpuKvReuseError> {
        Ok(self.source(layer, component)?.replay_window(request)?)
    }

    /// Replay the caller-supplied most recent window for `layer`/`component`.
    pub fn replay_recent_window(
        &self,
        layer: u32,
        component: KvComponent,
        window_items: u64,
    ) -> Result<(ReplayWindowRequestV1, WgpuBuffer), WgpuKvReuseError> {
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
    sources: Vec<(u32, WgpuReplaySourceV1)>,
) -> Result<Vec<Option<WgpuReplaySourceV1>>, WgpuKvReuseError> {
    let mut slots: Vec<Option<WgpuReplaySourceV1>> =
        (0..plan.layer_count()).map(|_| None).collect();
    for (layer, source) in sources {
        if plan.layer(layer)?.source(component) != KvLayerSourceV1::Own {
            return Err(WgpuKvReuseError::SourceForReusingLayer { layer, component });
        }
        let slot = &mut slots[layer as usize];
        if slot.is_some() {
            return Err(WgpuKvReuseError::DuplicateOwnerSource { layer, component });
        }
        *slot = Some(source);
    }
    for layer in plan.owning_layers(component) {
        if slots[layer as usize].is_none() {
            return Err(WgpuKvReuseError::MissingOwnerSource { layer, component });
        }
    }
    Ok(slots)
}

/// Fail-closed WGPU cross-layer KV binding errors (mirrors `CpuKvReuseError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgpuKvReuseError {
    /// Plan validation or lookup failed.
    Plan(KvReusePlanError),
    /// Replay of the owner source failed.
    Replay(WgpuReplayError),
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

impl From<KvReusePlanError> for WgpuKvReuseError {
    fn from(error: KvReusePlanError) -> Self {
        Self::Plan(error)
    }
}

impl From<WgpuReplayError> for WgpuKvReuseError {
    fn from(error: WgpuReplayError) -> Self {
        Self::Replay(error)
    }
}

impl fmt::Display for WgpuKvReuseError {
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

impl std::error::Error for WgpuKvReuseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Plan(error) => Some(error),
            Self::Replay(error) => Some(error),
            _ => None,
        }
    }
}
