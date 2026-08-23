//! Minimal `bfloat16` helpers (no external crates).
//!
//! NNIS's numeric policy for bf16 storage: **compute in `f32`, store in
//! bf16**. Conversion uses round-to-nearest-even, bit-identical to CUDA's
//! `__float2bfloat16_rn`, so host and device agree exactly.

/// Round an `f32` to bf16 precision, returning the raw 16-bit pattern
/// (round-to-nearest-even, matching `__float2bfloat16_rn`).
#[must_use]
pub fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    // NaN must stay NaN without rounding into infinity.
    if bits & 0x7FFF_FFFFu32 > 0x7F80_0000u32 {
        return ((bits | 0x0040_0000u32) >> 16) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

/// Widen a raw bf16 pattern to `f32` (exact).
#[must_use]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_round_trip_and_specials() {
        assert_eq!(bf16_bits_to_f32(f32_to_bf16_rne(1.0)), 1.0);
        assert_eq!(bf16_bits_to_f32(f32_to_bf16_rne(0.0)), 0.0);
        assert_eq!(bf16_bits_to_f32(f32_to_bf16_rne(-2.5)), -2.5);
        // Exact RNE tie (tail 0x8000) with even kept LSB rounds back down.
        let exact_tie = f32::from_bits(1.0_f32.to_bits() + 0x8000);
        assert_eq!(bf16_bits_to_f32(f32_to_bf16_rne(exact_tie)), 1.0);
        // Just past the tie rounds up to the next bf16 value.
        let just_above = f32::from_bits(1.0_f32.to_bits() + 0x8001);
        assert!(bf16_bits_to_f32(f32_to_bf16_rne(just_above)) > 1.0);
        // Infinities saturate rather than wrap.
        assert_eq!(
            bf16_bits_to_f32(f32_to_bf16_rne(f32::INFINITY)),
            f32::INFINITY
        );
        // NaN stays NaN.
        assert!(bf16_bits_to_f32(f32_to_bf16_rne(f32::NAN)).is_nan());
    }
}
