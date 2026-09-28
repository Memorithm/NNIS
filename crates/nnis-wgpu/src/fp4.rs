//! DSV41-3 WGPU decode of FP4 E2M1 group-scaled KV blocks.
//!
//! [`WgpuFp4E2M1KvBlockV1`] holds the packed codes and raw scale bytes of one
//! [`Fp4E2M1KvLayoutV1`] block in device buffers. Construction validates the
//! parts on the host in the same order, and with the same variants, as
//! `nnis_cpu::fp4_kv::CpuFp4E2M1KvBlockV1::from_parts`: code and scale
//! lengths, then row by row and position by position zero padding nibbles,
//! valid scales (F32 finite and not sign-negative; E8M0 not `0xFF`) and
//! finite decoded values.
//!
//! [`WgpuFp4E2M1KvBlockV1::decode`] runs one WGSL kernel, bound as a P4
//! artifact, that decodes every logical value with **integer arithmetic
//! only**. The E2M1 magnitude `m / 2` (`m` in `0,1,2,3,4,6,8,12`) times the
//! scale `S * 2^E` (F32 significand and exponent, or `2^(k - 127)` for E8M0)
//! is the exact integer product `m * S` (below `2^28`) at exponent `E - 1`,
//! rounded once to binary32 with round-to-nearest-even, including subnormal
//! results. That equals the CPU reference, which forms the product exactly in
//! F64 and rounds once to F32. No floating-point operation runs on the device,
//! so subnormal flushing cannot change the result. The sign nibble bit is
//! applied last, so `-0.0` is preserved.
//!
//! Declared tolerance: bit-exact with the CPU reference decode. Encoding stays
//! a CPU reference operation. This is a correctness path only: no memory,
//! residency, model-quality, latency or throughput claim.

use core::fmt;

use nnis_core::kernel_artifact::{KernelBindingKindV1, KernelElementTypeV1};
use nnis_core::kv_fp4::{
    Fp4E2M1KvLayoutV1, Fp4KvStorageV1, Fp4LayoutError, Fp4ScaleEncodingV1, FP4_E2M1_MAGNITUDES,
};
use nnis_core::{
    BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableError, PortableFence,
    PortableQueue,
};

use crate::memory::WgpuBuffer;
use crate::numerical::{pop_scopes, Binding, WgpuF32KernelsV1};
use crate::WgpuDevice;

/// Contract version of the WGSL FP4 decode.
pub const WGSL_FP4_DECODE_CONTRACT_VERSION: u32 = 1;

/// Numerical policy of the WGSL FP4 decode artifact.
pub const WGSL_FP4_DECODE_NUMERICAL_POLICY: &str =
    "wgsl-fp4-e2m1-integer-exact-product-single-rne-rounding-v1";

const WORKGROUP: u32 = 64;
const PARAM_WORDS: usize = 6;

const DECODE_SOURCE: &str = "
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

fn code_byte(index: u32) -> u32 {
    return (codes[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu;
}

fn scale_byte(index: u32) -> u32 {
    return (scales[index >> 2u] >> ((index & 3u) * 8u)) & 0xffu;
}

// Round p * 2^e (p < 2^28) once to binary32 bits, nearest-even.
fn round_bits(p: u32, e: i32) -> u32 {
    if (p == 0u) { return 0u; }
    let msb = i32(31u - countLeadingZeros(p));
    var x = msb + e;
    if (x >= -126) {
        let shift = msb - 23;
        var sig = 0u;
        if (shift <= 0) {
            sig = p << u32(-shift);
        } else {
            let s = u32(shift);
            sig = p >> s;
            let rem = p & ((1u << s) - 1u);
            let half = 1u << (s - 1u);
            if (rem > half || (rem == half && (sig & 1u) == 1u)) { sig = sig + 1u; }
            if (sig == 0x1000000u) { sig = sig >> 1u; x = x + 1; }
        }
        if (x > 127) { return 0x7f800000u; }
        return (u32(x + 127) << 23u) | (sig & 0x7fffffu);
    }
    let shift = -(e + 149);
    if (shift <= 0) { return p << u32(-shift); }
    if (shift >= 29) { return 0u; }
    let s = u32(shift);
    var q = p >> s;
    let rem = p & ((1u << s) - 1u);
    let half = 1u << (s - 1u);
    if (rem > half || (rem == half && (q & 1u) == 1u)) { q = q + 1u; }
    return q;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let row_width = params[0];
    let code_bytes_per_row = params[1];
    let groups_per_row = params[2];
    let group_size = params[3];
    let total = params[4];
    let encoding = params[5];
    if (i >= total) { return; }
    let row = i / row_width;
    let position = i % row_width;
    let packed = code_byte(row * code_bytes_per_row + position / 2u);
    var code = packed & 0xfu;
    if ((position & 1u) == 1u) { code = packed >> 4u; }
    let group = row * groups_per_row + position / group_size;
    var table = array<u32, 8>(0u, 1u, 2u, 3u, 4u, 6u, 8u, 12u);
    let m = table[code & 7u];
    var p = 0u;
    var e = 0;
    if (encoding == 0u) {
        let bits = scales[group];
        let biased = (bits >> 23u) & 0xffu;
        let mantissa = bits & 0x7fffffu;
        if (biased == 0u) {
            p = m * mantissa;
            e = -150;
        } else {
            p = m * (mantissa | 0x800000u);
            e = i32(biased) - 151;
        }
    } else {
        p = m;
        e = i32(scale_byte(group)) - 128;
    }
    var result = round_bits(p, e);
    if ((code & 8u) != 0u) { result = result | 0x80000000u; }
    output[i] = result;
}
";

/// Successful decode record; not timing, residency or performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuFp4DecodeReportV1 {
    pub schema_version: u32,
    pub decoded_values: usize,
    /// Fingerprint of the bound decode artifact.
    pub artifact_fingerprint: [u8; 32],
}

/// Decoded dense row-major F32 values in a fresh device buffer.
#[derive(Debug)]
pub struct WgpuFp4DecodeOutputV1 {
    /// `rows * row_width` F32 values (padding excluded), usages
    /// `STORAGE | COPY_SRC | COPY_DST`.
    pub values: WgpuBuffer,
    pub report: WgpuFp4DecodeReportV1,
}

/// FP4 E2M1 group-scaled KV block resident in WGPU buffers.
#[derive(Debug)]
pub struct WgpuFp4E2M1KvBlockV1 {
    layout: Fp4E2M1KvLayoutV1,
    storage: Fp4KvStorageV1,
    codes: WgpuBuffer,
    scales: WgpuBuffer,
    params: WgpuBuffer,
    values: usize,
}

impl WgpuFp4E2M1KvBlockV1 {
    /// Validate packed codes and raw scale bytes on the host, then upload them.
    ///
    /// Fails closed exactly like `CpuFp4E2M1KvBlockV1::from_parts`. A layout
    /// whose value count, code bytes or scale bytes exceed `u32` indexing is
    /// rejected with [`WgpuFp4Error::ExceedsDeviceIndexing`].
    pub fn from_parts(
        device: &WgpuDevice,
        layout: Fp4E2M1KvLayoutV1,
        codes: &[u8],
        scales: &[u8],
    ) -> Result<Self, WgpuFp4Error> {
        let storage = validate_parts(&layout, codes, scales)?;
        let values = host_len(storage.logical_values)?;
        let device_index = |value: u64| u32::try_from(value).is_ok();
        if !(device_index(storage.logical_values)
            && device_index(storage.code_bytes)
            && device_index(storage.scale_bytes))
        {
            return Err(WgpuFp4Error::ExceedsDeviceIndexing);
        }
        let encoding = match layout.scale_encoding() {
            Fp4ScaleEncodingV1::F32 => 0u32,
            Fp4ScaleEncodingV1::E8M0 => 1u32,
        };
        let params: [u32; PARAM_WORDS] = [
            layout.row_width(),
            layout.code_bytes_per_row() as u32,
            layout.groups_per_row(),
            layout.group_size(),
            values as u32,
            encoding,
        ];
        let param_bytes: Vec<u8> = params.iter().flat_map(|word| word.to_le_bytes()).collect();
        Ok(Self {
            layout,
            storage,
            codes: upload_words(device, codes)?,
            scales: upload_words(device, scales)?,
            params: upload_words(device, &param_bytes)?,
            values,
        })
    }

    /// Validated layout.
    pub const fn layout(&self) -> &Fp4E2M1KvLayoutV1 {
        &self.layout
    }

    /// Exact storage breakdown for this block's layout.
    pub const fn storage(&self) -> &Fp4KvStorageV1 {
        &self.storage
    }

    /// Decode the block on `device` (which must own it) into a fresh buffer.
    pub fn decode(&self, device: &WgpuDevice) -> Result<WgpuFp4DecodeOutputV1, WgpuFp4Error> {
        let kernels = WgpuF32KernelsV1::new(device)?;
        let queue = kernels.queue();
        queue.check_owner(&self.codes)?;
        let output_bytes = (self.values as u64)
            .checked_mul(4)
            .ok_or(WgpuFp4Error::HostIndexOverflow)?;
        let output = device.create_buffer(BufferDesc::new(
            output_bytes,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            MemoryClass::DeviceLocal,
        )?)?;
        let bindings = [
            read_u32(&self.params),
            read_u32(&self.codes),
            read_u32(&self.scales),
            Binding {
                buffer: output.raw(),
                kind: KernelBindingKindV1::StorageReadWrite,
                element: KernelElementTypeV1::F32,
                bytes: output_bytes,
            },
        ];
        let artifact = kernels.artifact(
            "nnis.wgpu.fp4_e2m1.decode",
            WGSL_FP4_DECODE_NUMERICAL_POLICY,
            DECODE_SOURCE,
            &bindings,
            [WORKGROUP, 1, 1],
        )?;
        let groups = kernels.groups(self.values, WORKGROUP)?;
        let raw = queue.device();
        raw.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        raw.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = raw.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nnis.wgpu.fp4_e2m1.decode"),
        });
        kernels.dispatch(&mut encoder, &artifact, &bindings, groups);
        let submission = queue.queue().submit(Some(encoder.finish()));
        pop_scopes(raw, "WGPU FP4 decode")?;
        queue.fence_for(submission).wait()?;
        Ok(WgpuFp4DecodeOutputV1 {
            values: output,
            report: WgpuFp4DecodeReportV1 {
                schema_version: WGSL_FP4_DECODE_CONTRACT_VERSION,
                decoded_values: self.values,
                artifact_fingerprint: *artifact.artifact_fingerprint(),
            },
        })
    }

    /// Decode on `device` and read the dense row-major F32 values back.
    pub fn decode_to_host(&self, device: &WgpuDevice) -> Result<Vec<f32>, WgpuFp4Error> {
        let output = self.decode(device)?;
        let bytes = device
            .create_queue()?
            .read_buffer(&output.values, 0, output.values.len())?;
        Ok(bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }
}

fn read_u32(buffer: &WgpuBuffer) -> Binding<'_> {
    Binding {
        buffer: buffer.raw(),
        kind: KernelBindingKindV1::StorageReadOnly,
        element: KernelElementTypeV1::U32,
        bytes: buffer.len(),
    }
}

/// Upload bytes zero-padded to whole `u32` words.
fn upload_words(device: &WgpuDevice, bytes: &[u8]) -> Result<WgpuBuffer, WgpuFp4Error> {
    let mut padded = Vec::new();
    padded
        .try_reserve_exact(bytes.len() + 4)
        .map_err(|_| WgpuFp4Error::AllocationFailed)?;
    padded.extend_from_slice(bytes);
    padded.resize(bytes.len().div_ceil(4).max(1) * 4, 0);
    let queue = device.create_queue()?;
    let mut buffer = device.create_buffer(BufferDesc::new(
        padded.len() as u64,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        MemoryClass::DeviceLocal,
    )?)?;
    queue.write_buffer(&mut buffer, 0, &padded)?.wait()?;
    Ok(buffer)
}

/// Host validation in the order and with the variants of the CPU reference.
fn validate_parts(
    layout: &Fp4E2M1KvLayoutV1,
    codes: &[u8],
    scales: &[u8],
) -> Result<Fp4KvStorageV1, WgpuFp4Error> {
    let storage = layout.storage()?;
    let expected_codes = host_len(storage.code_bytes)?;
    if codes.len() != expected_codes {
        return Err(WgpuFp4Error::CodeBytesMismatch {
            expected: expected_codes,
            actual: codes.len(),
        });
    }
    let expected_scales = host_len(storage.scale_bytes)?;
    if scales.len() != expected_scales {
        return Err(WgpuFp4Error::ScaleBytesMismatch {
            expected: expected_scales,
            actual: scales.len(),
        });
    }
    let row_width = layout.row_width() as usize;
    let padded = host_len(layout.padded_row_width())?;
    let code_bytes_per_row = host_len(layout.code_bytes_per_row())?;
    let group_size = layout.group_size() as usize;
    let groups_per_row = layout.groups_per_row() as usize;
    for row in 0..host_len(layout.rows())? {
        let row_codes = &codes[row * code_bytes_per_row..(row + 1) * code_bytes_per_row];
        for position in 0..padded {
            let byte = row_codes[position / 2];
            let code = if position % 2 == 0 {
                byte & 0x0F
            } else {
                byte >> 4
            };
            if position >= row_width {
                if code != 0 {
                    return Err(WgpuFp4Error::NonZeroPadding { row, position });
                }
                continue;
            }
            let group = row * groups_per_row + position / group_size;
            let scale = host_scale(layout.scale_encoding(), scales, group)?;
            let magnitude = f64::from(FP4_E2M1_MAGNITUDES[usize::from(code & 0x7)]) * scale;
            if !(magnitude as f32).is_finite() {
                return Err(WgpuFp4Error::DecodeOverflow { group });
            }
        }
    }
    Ok(storage)
}

fn host_scale(
    encoding: Fp4ScaleEncodingV1,
    scales: &[u8],
    group: usize,
) -> Result<f64, WgpuFp4Error> {
    match encoding {
        Fp4ScaleEncodingV1::F32 => {
            let bytes = scales
                .get(group * 4..group * 4 + 4)
                .ok_or(WgpuFp4Error::HostIndexOverflow)?;
            let scale = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if !scale.is_finite() || scale.is_sign_negative() {
                return Err(WgpuFp4Error::InvalidScale { group });
            }
            Ok(f64::from(scale))
        }
        Fp4ScaleEncodingV1::E8M0 => {
            let exponent = *scales.get(group).ok_or(WgpuFp4Error::HostIndexOverflow)?;
            if exponent == u8::MAX {
                return Err(WgpuFp4Error::InvalidScale { group });
            }
            Ok(f64::from_bits(
                ((i64::from(exponent) - 127 + 1023) as u64) << 52,
            ))
        }
    }
}

fn host_len(value: u64) -> Result<usize, WgpuFp4Error> {
    usize::try_from(value).map_err(|_| WgpuFp4Error::HostIndexOverflow)
}

/// Fail-closed WGPU FP4 errors.
///
/// Variants shared with `CpuFp4Error` carry identical payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WgpuFp4Error {
    /// Layout validation failed.
    Layout(Fp4LayoutError),
    /// Packed code length did not match the layout.
    CodeBytesMismatch { expected: usize, actual: usize },
    /// Scale byte length did not match the layout.
    ScaleBytesMismatch { expected: usize, actual: usize },
    /// A group scale was invalid for its encoding.
    InvalidScale { group: usize },
    /// A padding nibble was non-zero.
    NonZeroPadding { row: usize, position: usize },
    /// A value in this group would decode to a non-finite F32.
    DecodeOverflow { group: usize },
    /// A size does not fit host indexing.
    HostIndexOverflow,
    /// Host allocation failed.
    AllocationFailed,
    /// Value count, code bytes or scale bytes exceed `u32` device indexing.
    ExceedsDeviceIndexing,
    /// Portable WGPU allocation, binding, dispatch or readback failed.
    Portable(PortableError),
}

impl From<Fp4LayoutError> for WgpuFp4Error {
    fn from(error: Fp4LayoutError) -> Self {
        Self::Layout(error)
    }
}

impl From<PortableError> for WgpuFp4Error {
    fn from(error: PortableError) -> Self {
        Self::Portable(error)
    }
}

impl fmt::Display for WgpuFp4Error {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layout(error) => write!(output, "FP4 layout: {error}"),
            Self::CodeBytesMismatch { expected, actual } => {
                write!(output, "FP4 code bytes {actual} do not match {expected}")
            }
            Self::ScaleBytesMismatch { expected, actual } => {
                write!(output, "FP4 scale bytes {actual} do not match {expected}")
            }
            Self::InvalidScale { group } => write!(output, "FP4 group {group} scale is invalid"),
            Self::NonZeroPadding { row, position } => write!(
                output,
                "FP4 padding code at row {row} position {position} is non-zero"
            ),
            Self::DecodeOverflow { group } => write!(
                output,
                "FP4 group {group} would decode to a non-finite value"
            ),
            Self::HostIndexOverflow => output.write_str("FP4 size does not fit host indexing"),
            Self::AllocationFailed => output.write_str("FP4 host allocation failed"),
            Self::ExceedsDeviceIndexing => {
                output.write_str("FP4 block exceeds u32 WGPU device indexing")
            }
            Self::Portable(error) => write!(output, "WGPU FP4: {error}"),
        }
    }
}

impl std::error::Error for WgpuFp4Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Layout(error) => Some(error),
            Self::Portable(error) => Some(error),
            _ => None,
        }
    }
}
