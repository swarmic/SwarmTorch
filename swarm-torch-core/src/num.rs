//! Deterministic, `no_std`-compatible floating-point math helpers.
//!
//! Under `no_std`, Rust `core` does not provide `f64::sqrt` or `f32::sqrt`.
//! [`sqrt_f64`] and [`sqrt_f32`] provide IEEE-754 exponent-halving + Newton-Raphson
//! implementations that preserve `+0.0`, `-0.0`, `+Inf`, and `NaN` semantics and achieve
//! full 53-bit (`f64`) / 24-bit (`f32`) precision across the entire finite domain,
//! including subnormals and `f64::MAX`.

/// Software IEEE-754 square root over `f64` used under `no_std` and exposed for
/// deterministic cross-target verification.
#[inline]
pub fn sqrt_f64_soft(x: f64) -> f64 {
    if x < 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return x;
    }

    // Normalize IEEE-754 subnormals (x < 2^-1022) by scaling by 2^54 so the biased
    // exponent is non-zero before bit-level exponent halving.
    let (x_norm, post_scale) = if x < f64::MIN_POSITIVE {
        // 2^54 = 18014398509481984.0, 2^-27 = 1.0 / 134217728.0
        (
            x * f64::from_bits(0x4350_0000_0000_0000),
            f64::from_bits(0x3e40_0000_0000_0000),
        )
    } else {
        (x, 1.0)
    };

    // Bit-level initial seed: halves the unbiased exponent and approximates sqrt(1 + m)
    // within < 6.1% relative error across all normal f64 values.
    let bits = x_norm.to_bits();
    let mut y = f64::from_bits((bits >> 1) + 0x1ff8_0000_0000_0000);

    // 6 Newton-Raphson iterations reduce relative error from < 2^-4 to < 2^-150 (<< 2^-53).
    for _ in 0..6 {
        y = 0.5 * (y + x_norm / y);
    }

    y * post_scale
}

/// Computes $\sqrt{x}$ for `f64` in both `std` and `no_std` builds.
#[inline]
pub fn sqrt_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        sqrt_f64_soft(x)
    }
}

/// Computes $\sqrt{x}$ for `f32` in both `std` and `no_std` builds via widened `f64` evaluation.
#[inline]
pub fn sqrt_f32(x: f32) -> f32 {
    if x < 0.0 || x.is_nan() {
        return f32::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return x;
    }
    sqrt_f64(x as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqrt_f64_soft_special_values_preserve_ieee754_semantics() {
        assert!(sqrt_f64_soft(-1.0).is_nan());
        assert!(sqrt_f64_soft(f64::NEG_INFINITY).is_nan());
        assert!(sqrt_f64_soft(f64::NAN).is_nan());

        let pos_zero = sqrt_f64_soft(0.0);
        assert_eq!(pos_zero.to_bits(), 0.0f64.to_bits());

        let neg_zero = sqrt_f64_soft(-0.0);
        assert_eq!(neg_zero.to_bits(), (-0.0f64).to_bits());

        assert_eq!(sqrt_f64_soft(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn sqrt_f64_soft_matches_hardware_across_full_exponent_range() {
        let samples = [
            1e-320_f64, // deep subnormal
            f64::from_bits(1),
            f64::MIN_POSITIVE * 0.5,
            f64::MIN_POSITIVE,
            1e-300,
            1e-100,
            1e-12,
            0.25,
            0.5,
            1.0,
            2.0,
            3.0,
            100.0,
            1e12,
            1e38,
            1e100,
            1e300,
            f64::MAX,
        ];

        for &x in &samples {
            let got = sqrt_f64_soft(x);
            let expected = x.sqrt();
            let rel_err = if expected == 0.0 {
                (got - expected).abs()
            } else {
                ((got - expected) / expected).abs()
            };
            assert!(
                rel_err <= 2.0 * f64::EPSILON,
                "sqrt_f64_soft({x:e}) = {got:e}, expected {expected:e}, rel_err = {rel_err:e}"
            );
        }
    }
}
