//! Backend-neutral Pascal Vector Projection (PVP) layout adapter.
//!
//! This module carries only the versioned representation contract required to
//! move one address-major PVP bitplane bank between the qualified CPU u64
//! physical layout and the portable WGPU u32 physical layout.
//!
//! It deliberately owns no Pascal/ANF model semantics and no compute kernel.
//! SML-GENIUS owns model meaning, SciRust owns reusable CPU/SIMD execution, and
//! FLAT-ATTENTION owns specialized WGPU kernel qualification.

use core::fmt;

pub const PVP_LOGICAL_SCHEMA_V1: &str = "pvp-bitplanes/v1";
pub const NNIS_PVP_LAYOUT_ADAPTER_SCHEMA_V1: &str = "nnis.pvp-layout-adapter.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PvpPhysicalWordV1 {
    CpuU64,
    WgpuU32,
}

impl PvpPhysicalWordV1 {
    pub const fn bits(self) -> usize {
        match self {
            Self::CpuU64 => 64,
            Self::WgpuU32 => 32,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::CpuU64 => "cpu-u64",
            Self::WgpuU32 => "wgpu-u32",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PvpLayoutError {
    ZeroAddresses,
    AddressesNotPowerOfTwo { addresses: usize },
    ZeroGates,
    ArithmeticOverflow,
    StorageLengthMismatch {
        physical: PvpPhysicalWordV1,
        expected_words: usize,
        actual_words: usize,
    },
    NonZeroPadding {
        physical: PvpPhysicalWordV1,
        address: usize,
    },
}

impl fmt::Display for PvpLayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroAddresses => formatter.write_str("PVP address count must be non-zero"),
            Self::AddressesNotPowerOfTwo { addresses } => {
                write!(formatter, "PVP address count {addresses} is not a power of two")
            }
            Self::ZeroGates => formatter.write_str("PVP gate count must be non-zero"),
            Self::ArithmeticOverflow => formatter.write_str("PVP layout arithmetic overflow"),
            Self::StorageLengthMismatch {
                physical,
                expected_words,
                actual_words,
            } => write!(
                formatter,
                "PVP {} storage has {actual_words} words, expected {expected_words}",
                physical.label()
            ),
            Self::NonZeroPadding { physical, address } => write!(
                formatter,
                "PVP {} row {address} contains non-zero canonical padding",
                physical.label()
            ),
        }
    }
}

impl std::error::Error for PvpLayoutError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PvpLayoutAdapterV1 {
    addresses: usize,
    gates: usize,
}

impl PvpLayoutAdapterV1 {
    pub fn new(addresses: usize, gates: usize) -> Result<Self, PvpLayoutError> {
        if addresses == 0 {
            return Err(PvpLayoutError::ZeroAddresses);
        }
        if !addresses.is_power_of_two() {
            return Err(PvpLayoutError::AddressesNotPowerOfTwo { addresses });
        }
        if gates == 0 {
            return Err(PvpLayoutError::ZeroGates);
        }
        addresses
            .checked_mul(gates)
            .ok_or(PvpLayoutError::ArithmeticOverflow)?;
        Ok(Self { addresses, gates })
    }

    pub const fn addresses(self) -> usize {
        self.addresses
    }

    pub const fn gates(self) -> usize {
        self.gates
    }

    pub const fn stages(self) -> u32 {
        self.addresses.trailing_zeros()
    }

    pub fn logical_bits(self) -> Result<usize, PvpLayoutError> {
        self.addresses
            .checked_mul(self.gates)
            .ok_or(PvpLayoutError::ArithmeticOverflow)
    }

    pub fn words_per_address(
        self,
        physical: PvpPhysicalWordV1,
    ) -> Result<usize, PvpLayoutError> {
        checked_ceil_div(self.gates, physical.bits())
    }

    pub fn storage_words(
        self,
        physical: PvpPhysicalWordV1,
    ) -> Result<usize, PvpLayoutError> {
        self.addresses
            .checked_mul(self.words_per_address(physical)?)
            .ok_or(PvpLayoutError::ArithmeticOverflow)
    }

    pub fn storage_bytes(
        self,
        physical: PvpPhysicalWordV1,
    ) -> Result<usize, PvpLayoutError> {
        let bytes_per_word = physical.bits() / 8;
        self.storage_words(physical)?
            .checked_mul(bytes_per_word)
            .ok_or(PvpLayoutError::ArithmeticOverflow)
    }

    pub fn padding_bits(
        self,
        physical: PvpPhysicalWordV1,
    ) -> Result<usize, PvpLayoutError> {
        let physical_bits = self
            .storage_words(physical)?
            .checked_mul(physical.bits())
            .ok_or(PvpLayoutError::ArithmeticOverflow)?;
        physical_bits
            .checked_sub(self.logical_bits()?)
            .ok_or(PvpLayoutError::ArithmeticOverflow)
    }

    pub fn canonical_record(
        self,
        physical: PvpPhysicalWordV1,
    ) -> Result<String, PvpLayoutError> {
        Ok(format!(
            "{};source_logical={};addresses={};gates={};physical={};word_bits={};order=address-major-gate-word;words_per_address={};storage_bytes={};padding_bits={}",
            NNIS_PVP_LAYOUT_ADAPTER_SCHEMA_V1,
            PVP_LOGICAL_SCHEMA_V1,
            self.addresses,
            self.gates,
            physical.label(),
            physical.bits(),
            self.words_per_address(physical)?,
            self.storage_bytes(physical)?,
            self.padding_bits(physical)?
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvpCpuU64V1 {
    layout: PvpLayoutAdapterV1,
    words: Vec<u64>,
}

impl PvpCpuU64V1 {
    pub fn new(layout: PvpLayoutAdapterV1, words: Vec<u64>) -> Result<Self, PvpLayoutError> {
        validate_u64(layout, &words)?;
        Ok(Self { layout, words })
    }

    pub const fn layout(&self) -> PvpLayoutAdapterV1 {
        self.layout
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn to_wgpu_u32(&self) -> Result<PvpWgpuU32V1, PvpLayoutError> {
        let source_words = self
            .layout
            .words_per_address(PvpPhysicalWordV1::CpuU64)?;
        let target_words = self
            .layout
            .words_per_address(PvpPhysicalWordV1::WgpuU32)?;
        let mut output = vec![
            0_u32;
            self.layout
                .storage_words(PvpPhysicalWordV1::WgpuU32)?
        ];

        for address in 0..self.layout.addresses() {
            let source_base = address
                .checked_mul(source_words)
                .ok_or(PvpLayoutError::ArithmeticOverflow)?;
            let target_base = address
                .checked_mul(target_words)
                .ok_or(PvpLayoutError::ArithmeticOverflow)?;
            for target_word in 0..target_words {
                let source_word = target_word / 2;
                let shift = (target_word % 2) * 32;
                output[target_base + target_word] =
                    (self.words[source_base + source_word] >> shift) as u32;
            }
        }

        PvpWgpuU32V1::new(self.layout, output)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvpWgpuU32V1 {
    layout: PvpLayoutAdapterV1,
    words: Vec<u32>,
}

impl PvpWgpuU32V1 {
    pub fn new(layout: PvpLayoutAdapterV1, words: Vec<u32>) -> Result<Self, PvpLayoutError> {
        validate_u32(layout, &words)?;
        Ok(Self { layout, words })
    }

    pub const fn layout(&self) -> PvpLayoutAdapterV1 {
        self.layout
    }

    pub fn words(&self) -> &[u32] {
        &self.words
    }

    pub fn to_cpu_u64(&self) -> Result<PvpCpuU64V1, PvpLayoutError> {
        let source_words = self
            .layout
            .words_per_address(PvpPhysicalWordV1::WgpuU32)?;
        let target_words = self
            .layout
            .words_per_address(PvpPhysicalWordV1::CpuU64)?;
        let mut output = vec![
            0_u64;
            self.layout
                .storage_words(PvpPhysicalWordV1::CpuU64)?
        ];

        for address in 0..self.layout.addresses() {
            let source_base = address
                .checked_mul(source_words)
                .ok_or(PvpLayoutError::ArithmeticOverflow)?;
            let target_base = address
                .checked_mul(target_words)
                .ok_or(PvpLayoutError::ArithmeticOverflow)?;
            for source_word in 0..source_words {
                let target_word = source_word / 2;
                let shift = (source_word % 2) * 32;
                output[target_base + target_word] |=
                    u64::from(self.words[source_base + source_word]) << shift;
            }
        }

        PvpCpuU64V1::new(self.layout, output)
    }
}

fn checked_ceil_div(value: usize, divisor: usize) -> Result<usize, PvpLayoutError> {
    value
        .checked_add(divisor - 1)
        .map(|sum| sum / divisor)
        .ok_or(PvpLayoutError::ArithmeticOverflow)
}

fn validate_u64(layout: PvpLayoutAdapterV1, words: &[u64]) -> Result<(), PvpLayoutError> {
    let physical = PvpPhysicalWordV1::CpuU64;
    let expected = layout.storage_words(physical)?;
    if words.len() != expected {
        return Err(PvpLayoutError::StorageLengthMismatch {
            physical,
            expected_words: expected,
            actual_words: words.len(),
        });
    }
    let tail = layout.gates() % 64;
    if tail == 0 {
        return Ok(());
    }
    let mask = !((1_u64 << tail) - 1);
    let row_words = layout.words_per_address(physical)?;
    let last = row_words - 1;
    for address in 0..layout.addresses() {
        if words[address * row_words + last] & mask != 0 {
            return Err(PvpLayoutError::NonZeroPadding { physical, address });
        }
    }
    Ok(())
}

fn validate_u32(layout: PvpLayoutAdapterV1, words: &[u32]) -> Result<(), PvpLayoutError> {
    let physical = PvpPhysicalWordV1::WgpuU32;
    let expected = layout.storage_words(physical)?;
    if words.len() != expected {
        return Err(PvpLayoutError::StorageLengthMismatch {
            physical,
            expected_words: expected,
            actual_words: words.len(),
        });
    }
    let tail = layout.gates() % 32;
    if tail == 0 {
        return Ok(());
    }
    let mask = !((1_u32 << tail) - 1);
    let row_words = layout.words_per_address(physical)?;
    let last = row_words - 1;
    for address in 0..layout.addresses() {
        if words[address * row_words + last] & mask != 0 {
            return Err(PvpLayoutError::NonZeroPadding { physical, address });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_fixture(layout: PvpLayoutAdapterV1) -> PvpCpuU64V1 {
        let row_words = layout
            .words_per_address(PvpPhysicalWordV1::CpuU64)
            .unwrap();
        let mut words = vec![
            0_u64;
            layout
                .storage_words(PvpPhysicalWordV1::CpuU64)
                .unwrap()
        ];
        for address in 0..layout.addresses() {
            for gate in 0..layout.gates() {
                if ((address * 13 + gate * 17 + (address ^ gate)) % 11) < 5 {
                    words[address * row_words + gate / 64] |= 1_u64 << (gate % 64);
                }
            }
        }
        PvpCpuU64V1::new(layout, words).unwrap()
    }

    #[test]
    fn layout_preserves_one_logical_schema_across_two_physical_projections() {
        let layout = PvpLayoutAdapterV1::new(128, 65).unwrap();
        assert_eq!(layout.stages(), 7);
        assert_eq!(layout.logical_bits().unwrap(), 8_320);
        assert_eq!(
            layout.words_per_address(PvpPhysicalWordV1::CpuU64).unwrap(),
            2
        );
        assert_eq!(
            layout.words_per_address(PvpPhysicalWordV1::WgpuU32).unwrap(),
            3
        );
        assert_eq!(
            layout.storage_bytes(PvpPhysicalWordV1::CpuU64).unwrap(),
            2_048
        );
        assert_eq!(
            layout.storage_bytes(PvpPhysicalWordV1::WgpuU32).unwrap(),
            1_536
        );
        assert_eq!(
            layout.padding_bits(PvpPhysicalWordV1::CpuU64).unwrap(),
            8_064
        );
        assert_eq!(
            layout.padding_bits(PvpPhysicalWordV1::WgpuU32).unwrap(),
            4_032
        );
        assert!(
            layout
                .canonical_record(PvpPhysicalWordV1::CpuU64)
                .unwrap()
                .contains(PVP_LOGICAL_SCHEMA_V1)
        );
    }

    #[test]
    fn cpu_u64_and_wgpu_u32_round_trip_exactly_for_tail_geometries() {
        for gates in [1, 31, 32, 33, 63, 64, 65, 97, 129] {
            let layout = PvpLayoutAdapterV1::new(64, gates).unwrap();
            let cpu = cpu_fixture(layout);
            let wgpu = cpu.to_wgpu_u32().unwrap();
            let round_trip = wgpu.to_cpu_u64().unwrap();
            assert_eq!(round_trip, cpu, "gates={gates}");
        }
    }

    #[test]
    fn nonzero_padding_fails_closed_on_both_projections() {
        let layout = PvpLayoutAdapterV1::new(8, 65).unwrap();

        let mut cpu = vec![
            0_u64;
            layout
                .storage_words(PvpPhysicalWordV1::CpuU64)
                .unwrap()
        ];
        cpu[1] = 1_u64 << 63;
        assert!(matches!(
            PvpCpuU64V1::new(layout, cpu),
            Err(PvpLayoutError::NonZeroPadding {
                physical: PvpPhysicalWordV1::CpuU64,
                address: 0
            })
        ));

        let mut wgpu = vec![
            0_u32;
            layout
                .storage_words(PvpPhysicalWordV1::WgpuU32)
                .unwrap()
        ];
        wgpu[2] = 1_u32 << 31;
        assert!(matches!(
            PvpWgpuU32V1::new(layout, wgpu),
            Err(PvpLayoutError::NonZeroPadding {
                physical: PvpPhysicalWordV1::WgpuU32,
                address: 0
            })
        ));
    }

    #[test]
    fn invalid_shapes_and_lengths_are_rejected() {
        assert_eq!(
            PvpLayoutAdapterV1::new(0, 1),
            Err(PvpLayoutError::ZeroAddresses)
        );
        assert_eq!(
            PvpLayoutAdapterV1::new(3, 1),
            Err(PvpLayoutError::AddressesNotPowerOfTwo { addresses: 3 })
        );
        assert_eq!(PvpLayoutAdapterV1::new(8, 0), Err(PvpLayoutError::ZeroGates));

        let layout = PvpLayoutAdapterV1::new(8, 8).unwrap();
        assert!(matches!(
            PvpCpuU64V1::new(layout, vec![]),
            Err(PvpLayoutError::StorageLengthMismatch { .. })
        ));
        assert!(matches!(
            PvpWgpuU32V1::new(layout, vec![]),
            Err(PvpLayoutError::StorageLengthMismatch { .. })
        ));
    }
}
