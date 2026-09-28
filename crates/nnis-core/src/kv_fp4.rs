//! Backend-neutral FP4 E2M1 group-scaled KV layout and exact storage accounting.
//!
//! DSV41-3 declares a versioned layout for KV rows stored as 4-bit E2M1
//! floating-point codes (1 sign, 2 exponent with bias 1, 1 mantissa bit; the
//! OCP MX FP4 element encoding) with one explicit scale per group of
//! consecutive values in a row. FP4 E2M1 is a non-uniform floating-point grid
//! and is a distinct format, with distinct evidence, from the NNIS INT4
//! integer representations.
//!
//! This module performs no encoding or execution. It fixes the code table and
//! reports every storage component (codes, scales, row padding, descriptor
//! metadata) so an effective bits-per-value figure always includes all
//! overheads. Nominal 4-bit width is never a compression result by itself.

use core::fmt;

/// Version of the NNIS FP4 E2M1 group-scaled KV layout contract.
pub const NNIS_FP4_E2M1_KV_LAYOUT_VERSION: u32 = 1;

/// Stable representation identity for this layout.
pub const FP4_E2M1_KV_REPRESENTATION_ID: &str = "nnis.kv.fp4-e2m1.group-scaled";

/// Bits per stored FP4 code.
pub const FP4_CODE_BITS: u64 = 4;

/// Largest finite E2M1 magnitude.
pub const FP4_E2M1_MAX_MAGNITUDE: f32 = 6.0;

/// Magnitudes of the eight non-negative E2M1 codes (`code & 0x7`).
pub const FP4_E2M1_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Serialized v1 layout descriptor size counted as metadata:
/// version `u32`, rows `u64`, row width `u32`, group size `u32`,
/// scale encoding `u8`, and 3 reserved bytes.
pub const FP4_KV_DESCRIPTOR_BYTES: u64 = 24;

/// Largest accepted group size.
pub const MAX_FP4_GROUP_SIZE: u32 = 4096;

/// Decode one E2M1 code (low 4 bits) to its unscaled value.
///
/// Code `0x8` decodes to negative zero.
pub fn fp4_e2m1_value(code: u8) -> Result<f32, Fp4LayoutError> {
    if code > 0xF {
        return Err(Fp4LayoutError::InvalidCode { code });
    }
    let magnitude = FP4_E2M1_MAGNITUDES[usize::from(code & 0x7)];
    Ok(if code & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    })
}

/// Explicit per-group scale encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp4ScaleEncodingV1 {
    /// Finite non-negative little-endian IEEE-754 F32 scale (4 bytes).
    F32,
    /// Unsigned power-of-two exponent `2^(e - 127)`, `e` in `0..=254` (1 byte).
    /// Byte `255` is invalid.
    E8M0,
}

impl Fp4ScaleEncodingV1 {
    /// Stored bytes per group scale.
    pub const fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::E8M0 => 1,
        }
    }

    /// Stable descriptor tag.
    pub const fn tag(self) -> u8 {
        match self {
            Self::F32 => 1,
            Self::E8M0 => 2,
        }
    }
}

/// Dense baseline declared for compression-ratio reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenseKvBaselineV1 {
    /// 32-bit float.
    F32,
    /// 16-bit IEEE half.
    F16,
    /// 16-bit bfloat.
    BF16,
}

impl DenseKvBaselineV1 {
    /// Bits per dense scalar.
    pub const fn bits(self) -> u64 {
        match self {
            Self::F32 => 32,
            Self::F16 | Self::BF16 => 16,
        }
    }
}

/// Validated FP4 E2M1 group-scaled KV layout.
///
/// Groups are formed within a row and never cross rows. The final group of a
/// row is padded with zero codes when `row_width` is not a multiple of
/// `group_size`; padding is reported separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp4E2M1KvLayoutV1 {
    rows: u64,
    row_width: u32,
    group_size: u32,
    scale_encoding: Fp4ScaleEncodingV1,
}

impl Fp4E2M1KvLayoutV1 {
    /// Validate a layout with an even group size so groups are byte aligned.
    pub fn new(
        rows: u64,
        row_width: u32,
        group_size: u32,
        scale_encoding: Fp4ScaleEncodingV1,
    ) -> Result<Self, Fp4LayoutError> {
        if rows == 0 {
            return Err(Fp4LayoutError::ZeroRows);
        }
        if row_width == 0 {
            return Err(Fp4LayoutError::ZeroRowWidth);
        }
        if group_size == 0 || group_size % 2 != 0 || group_size > MAX_FP4_GROUP_SIZE {
            return Err(Fp4LayoutError::InvalidGroupSize { group_size });
        }
        let layout = Self {
            rows,
            row_width,
            group_size,
            scale_encoding,
        };
        layout.storage()?;
        Ok(layout)
    }

    /// Logical rows (positions).
    pub const fn rows(&self) -> u64 {
        self.rows
    }

    /// Logical values per row.
    pub const fn row_width(&self) -> u32 {
        self.row_width
    }

    /// Values per scale group.
    pub const fn group_size(&self) -> u32 {
        self.group_size
    }

    /// Scale encoding.
    pub const fn scale_encoding(&self) -> Fp4ScaleEncodingV1 {
        self.scale_encoding
    }

    /// Scale groups per row, including a final partially filled group.
    pub const fn groups_per_row(&self) -> u32 {
        // Rounding up never exceeds `row_width`, so the result fits `u32`.
        (self.row_width as u64).div_ceil(self.group_size as u64) as u32
    }

    /// Stored codes per row, including padding codes.
    pub const fn padded_row_width(&self) -> u64 {
        self.groups_per_row() as u64 * self.group_size as u64
    }

    /// Packed code bytes per row (two codes per byte).
    pub const fn code_bytes_per_row(&self) -> u64 {
        self.padded_row_width() / 2
    }

    /// Scale bytes per row.
    pub const fn scale_bytes_per_row(&self) -> u64 {
        self.groups_per_row() as u64 * self.scale_encoding.bytes()
    }

    /// Exact storage breakdown for this layout.
    pub fn storage(&self) -> Result<Fp4KvStorageV1, Fp4LayoutError> {
        let logical_values = self
            .rows
            .checked_mul(u64::from(self.row_width))
            .ok_or(Fp4LayoutError::SizeOverflow)?;
        let padding_values_per_row = self.padded_row_width() - u64::from(self.row_width);
        let padding_values = self
            .rows
            .checked_mul(padding_values_per_row)
            .ok_or(Fp4LayoutError::SizeOverflow)?;
        let code_bytes = self
            .rows
            .checked_mul(self.code_bytes_per_row())
            .ok_or(Fp4LayoutError::SizeOverflow)?;
        let groups = self
            .rows
            .checked_mul(u64::from(self.groups_per_row()))
            .ok_or(Fp4LayoutError::SizeOverflow)?;
        let scale_bytes = groups
            .checked_mul(self.scale_encoding.bytes())
            .ok_or(Fp4LayoutError::SizeOverflow)?;
        let total_bytes = code_bytes
            .checked_add(scale_bytes)
            .and_then(|bytes| bytes.checked_add(FP4_KV_DESCRIPTOR_BYTES))
            .ok_or(Fp4LayoutError::SizeOverflow)?;
        Ok(Fp4KvStorageV1 {
            logical_values,
            groups,
            code_bytes,
            scale_bytes,
            padding_values,
            metadata_bytes: FP4_KV_DESCRIPTOR_BYTES,
            total_bytes,
        })
    }
}

/// Exact resident storage breakdown of one FP4 E2M1 KV block.
///
/// Codebook and residual overheads are structurally zero for this layout and
/// are reported explicitly so reports stay comparable with other formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fp4KvStorageV1 {
    /// Logical KV scalars represented (rows x row width).
    pub logical_values: u64,
    /// Scale groups stored.
    pub groups: u64,
    /// Packed code bytes, including padding codes.
    pub code_bytes: u64,
    /// Group-scale bytes.
    pub scale_bytes: u64,
    /// Padding codes stored to complete final row groups.
    pub padding_values: u64,
    /// Descriptor metadata bytes.
    pub metadata_bytes: u64,
    /// Total resident bytes: codes + scales + metadata.
    pub total_bytes: u64,
}

impl Fp4KvStorageV1 {
    /// Codebook overhead bytes (none for FP4 E2M1).
    pub const fn codebook_bytes(&self) -> u64 {
        0
    }

    /// Residual/outlier overhead bytes (none in v1).
    pub const fn residual_bytes(&self) -> u64 {
        0
    }

    /// Bits spent on padding codes.
    pub fn padding_bits(&self) -> u128 {
        u128::from(self.padding_values) * u128::from(FP4_CODE_BITS)
    }

    /// Total resident bits.
    pub fn total_bits(&self) -> u128 {
        u128::from(self.total_bytes) * 8
    }

    /// Exact effective bits per logical value as `(numerator, denominator)`.
    pub fn effective_bits_per_value_ratio(&self) -> (u128, u128) {
        (self.total_bits(), u128::from(self.logical_values))
    }

    /// Effective bits per logical value, including every overhead.
    pub fn effective_bits_per_value(&self) -> f64 {
        self.total_bits() as f64 / self.logical_values as f64
    }

    /// Dense baseline bits for the same logical values.
    pub fn dense_bits(&self, baseline: DenseKvBaselineV1) -> u128 {
        u128::from(self.logical_values) * u128::from(baseline.bits())
    }

    /// Exact storage ratio `dense_bits / total_bits` as `(numerator, denominator)`.
    ///
    /// This is a storage-accounting ratio only, not a measured memory result.
    pub fn compression_ratio_against(&self, baseline: DenseKvBaselineV1) -> (u128, u128) {
        (self.dense_bits(baseline), self.total_bits())
    }
}

/// Fail-closed FP4 layout errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fp4LayoutError {
    /// Layout declared zero rows.
    ZeroRows,
    /// Layout declared zero values per row.
    ZeroRowWidth,
    /// Group size was zero, odd, or larger than [`MAX_FP4_GROUP_SIZE`].
    InvalidGroupSize { group_size: u32 },
    /// A code did not fit in 4 bits.
    InvalidCode { code: u8 },
    /// Storage arithmetic overflowed.
    SizeOverflow,
}

impl fmt::Display for Fp4LayoutError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroRows => output.write_str("FP4 KV layout must have at least one row"),
            Self::ZeroRowWidth => output.write_str("FP4 KV row width must be non-zero"),
            Self::InvalidGroupSize { group_size } => write!(
                output,
                "FP4 group size {group_size} must be even and in 2..={MAX_FP4_GROUP_SIZE}"
            ),
            Self::InvalidCode { code } => write!(output, "FP4 code {code:#x} exceeds 4 bits"),
            Self::SizeOverflow => output.write_str("FP4 KV storage arithmetic overflow"),
        }
    }
}

impl std::error::Error for Fp4LayoutError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_code_table_matches_sign_exponent_mantissa_definition() {
        for code in 0u8..8 {
            let exponent = (code >> 1) & 0x3;
            let mantissa = f32::from(code & 0x1);
            let expected = if exponent == 0 {
                mantissa * 0.5
            } else {
                (1.0 + mantissa * 0.5) * f32::powi(2.0, i32::from(exponent) - 1)
            };
            assert_eq!(fp4_e2m1_value(code).unwrap(), expected);
            assert_eq!(fp4_e2m1_value(code | 0x8).unwrap(), -expected);
        }
        assert!(fp4_e2m1_value(0x8).unwrap().is_sign_negative());
        assert_eq!(
            fp4_e2m1_value(0x10),
            Err(Fp4LayoutError::InvalidCode { code: 0x10 })
        );
        // Non-uniform grid: FP4 E2M1 is not a scaled INT4 lattice.
        assert_ne!(
            FP4_E2M1_MAGNITUDES[7] - FP4_E2M1_MAGNITUDES[6],
            FP4_E2M1_MAGNITUDES[1] - FP4_E2M1_MAGNITUDES[0]
        );
    }

    #[test]
    fn aligned_e8m0_storage_is_exact() {
        let layout = Fp4E2M1KvLayoutV1::new(128, 64, 32, Fp4ScaleEncodingV1::E8M0).unwrap();
        let storage = layout.storage().unwrap();
        assert_eq!(storage.logical_values, 8192);
        assert_eq!(storage.groups, 256);
        assert_eq!(storage.code_bytes, 4096);
        assert_eq!(storage.scale_bytes, 256);
        assert_eq!(storage.padding_values, 0);
        assert_eq!(storage.metadata_bytes, 24);
        assert_eq!(storage.total_bytes, 4096 + 256 + 24);
        assert_eq!(storage.codebook_bytes(), 0);
        assert_eq!(storage.residual_bytes(), 0);
        // (4096 + 256 + 24) * 8 / 8192 = 4.2734375 bits per value.
        assert_eq!(storage.effective_bits_per_value_ratio(), (35008, 8192));
        assert_eq!(storage.effective_bits_per_value(), 4.2734375);
        assert_eq!(
            storage.compression_ratio_against(DenseKvBaselineV1::F16),
            (131072, 35008)
        );
    }

    #[test]
    fn f32_scales_and_row_padding_are_charged() {
        let layout = Fp4E2M1KvLayoutV1::new(3, 10, 4, Fp4ScaleEncodingV1::F32).unwrap();
        assert_eq!(layout.groups_per_row(), 3);
        assert_eq!(layout.padded_row_width(), 12);
        assert_eq!(layout.code_bytes_per_row(), 6);
        assert_eq!(layout.scale_bytes_per_row(), 12);
        let storage = layout.storage().unwrap();
        assert_eq!(storage.logical_values, 30);
        assert_eq!(storage.padding_values, 6);
        assert_eq!(storage.padding_bits(), 24);
        assert_eq!(storage.code_bytes, 18);
        assert_eq!(storage.scale_bytes, 36);
        assert_eq!(storage.total_bytes, 18 + 36 + 24);
        // Tiny blocks are dominated by overhead: 20.8 bits per value, not 4.
        assert_eq!(storage.effective_bits_per_value_ratio(), (624, 30));
        assert!(storage.effective_bits_per_value() > 20.0);
        assert_eq!(
            storage.compression_ratio_against(DenseKvBaselineV1::F32),
            (960, 624)
        );
    }

    #[test]
    fn invalid_layouts_fail_closed() {
        let e8 = Fp4ScaleEncodingV1::E8M0;
        assert_eq!(
            Fp4E2M1KvLayoutV1::new(0, 8, 8, e8),
            Err(Fp4LayoutError::ZeroRows)
        );
        assert_eq!(
            Fp4E2M1KvLayoutV1::new(1, 0, 8, e8),
            Err(Fp4LayoutError::ZeroRowWidth)
        );
        for group_size in [0, 3, MAX_FP4_GROUP_SIZE + 2] {
            assert_eq!(
                Fp4E2M1KvLayoutV1::new(1, 8, group_size, e8),
                Err(Fp4LayoutError::InvalidGroupSize { group_size })
            );
        }
        assert_eq!(
            Fp4E2M1KvLayoutV1::new(u64::MAX, u32::MAX, 2, e8),
            Err(Fp4LayoutError::SizeOverflow)
        );
    }

    #[test]
    fn scale_encodings_have_distinct_tags_and_sizes() {
        assert_eq!(Fp4ScaleEncodingV1::F32.bytes(), 4);
        assert_eq!(Fp4ScaleEncodingV1::E8M0.bytes(), 1);
        assert_ne!(
            Fp4ScaleEncodingV1::F32.tag(),
            Fp4ScaleEncodingV1::E8M0.tag()
        );
        assert_eq!(DenseKvBaselineV1::BF16.bits(), 16);
    }
}
