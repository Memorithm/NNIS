//! Integer-only binary32 oracle used by the CPU host qualification suites.
//!
//! Each binary32 is decomposed into an exact integer mantissa and
//! power-of-two exponent, sums and products are computed exactly in 128-bit
//! integers, and [`round`] rounds once to nearest-even binary32 with gradual
//! underflow. Nothing here uses the host floating-point unit, fused
//! multiply-add or float conversion, so it can judge the host's arithmetic.

/// Exact value `(-1)^negative * mantissa * 2^exponent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exact {
    /// Sign; a zero mantissa with `negative` set is negative zero.
    pub negative: bool,
    /// Exact integer magnitude; must stay below 2^112 when rounded.
    pub mantissa: u128,
    /// Power-of-two exponent of the least significant mantissa bit.
    pub exponent: i32,
}

/// Decompose a finite binary32 bit pattern exactly.
pub fn exact(bits: u32) -> Exact {
    let biased = ((bits >> 23) & 0xff) as i32;
    assert!(biased != 0xff, "oracle input must be finite");
    let fraction = u128::from(bits & 0x007f_ffff);
    let (mantissa, exponent) = if biased == 0 {
        (fraction, -149)
    } else {
        (fraction | 0x0080_0000, biased - 150)
    };
    Exact {
        negative: bits >> 31 != 0,
        mantissa,
        exponent,
    }
}

fn bit_length(value: u128) -> i32 {
    128 - value.leading_zeros() as i32
}

/// Round an exact value once to nearest-even binary32 bits, with gradual
/// underflow; overflow gives infinity (which the reference rejects).
pub fn round(value: Exact) -> u32 {
    let sign = u32::from(value.negative) << 31;
    if value.mantissa == 0 {
        return sign;
    }
    assert!(value.mantissa < 1 << 112, "oracle mantissa out of range");
    let top = value.exponent + bit_length(value.mantissa) - 1;
    let normal = top >= -126;
    let shift = if normal {
        bit_length(value.mantissa) - 24
    } else {
        -149 - value.exponent
    };
    let mut quotient = if shift <= 0 {
        value.mantissa << (-shift) as u32
    } else if shift >= 120 {
        0
    } else {
        let quotient = value.mantissa >> shift as u32;
        let remainder = value.mantissa & ((1u128 << shift as u32) - 1);
        let half = 1u128 << (shift as u32 - 1);
        if remainder > half || (remainder == half && quotient & 1 == 1) {
            quotient + 1
        } else {
            quotient
        }
    };
    if !normal {
        // Raw subnormal bits; a carry to 2^23 is the smallest normal.
        return sign | quotient as u32;
    }
    let mut biased = top + 127;
    if quotient == 1 << 24 {
        quotient >>= 1;
        biased += 1;
    }
    if biased >= 255 {
        return sign | 0x7f80_0000;
    }
    sign | ((biased as u32) << 23) | (quotient as u32 & 0x007f_ffff)
}

/// Exact product.
pub fn product(left: Exact, right: Exact) -> Exact {
    Exact {
        negative: left.negative != right.negative,
        mantissa: left.mantissa * right.mantissa,
        exponent: left.exponent + right.exponent,
    }
}

/// Sum that rounds identically to the exact sum.
///
/// A term more than 60 binary orders below the other only acts as a sticky
/// bit, so it is replaced by a one at 61 orders below, which lies strictly
/// inside the same rounding interval and keeps the sign.
pub fn sum(left: Exact, right: Exact) -> Exact {
    if left.mantissa == 0 && right.mantissa == 0 {
        return Exact {
            negative: left.negative && right.negative,
            mantissa: 0,
            exponent: 0,
        };
    }
    if right.mantissa == 0 {
        return left;
    }
    if left.mantissa == 0 {
        return right;
    }
    let top = |value: Exact| value.exponent + bit_length(value.mantissa);
    let sticky = |tiny: Exact, big: Exact| Exact {
        negative: tiny.negative,
        mantissa: 1,
        exponent: top(big) - 61,
    };
    let (left, right) = if top(right) < top(left) - 60 {
        (left, sticky(right, left))
    } else if top(left) < top(right) - 60 {
        (sticky(left, right), right)
    } else {
        (left, right)
    };
    let exponent = left.exponent.min(right.exponent);
    let signed = |value: Exact| {
        let magnitude = (value.mantissa << (value.exponent - exponent) as u32) as i128;
        if value.negative {
            -magnitude
        } else {
            magnitude
        }
    };
    let total = signed(left) + signed(right);
    Exact {
        // Exact cancellation gives +0 under round-to-nearest-even.
        negative: total < 0,
        mantissa: total.unsigned_abs(),
        exponent,
    }
}
