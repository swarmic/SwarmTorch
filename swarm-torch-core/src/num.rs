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

/// Software IEEE-754 natural exponential $e^x$ over `f64` using base-2 range reduction
/// $x = k \ln 2 + r$ ($|r| \le \frac{\ln 2}{2}$) and an 11th-order Horner Taylor polynomial.
///
/// Exposed for `no_std` builds and cross-target determinism verification.
#[inline]
pub fn exp_f64_soft(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 709.782_712_893_384 {
        return f64::INFINITY;
    }
    if x <= -745.133_219_101_941_1 {
        return 0.0;
    }
    // Exact fdlibm / Sun Microsystems 33-bit high + 53-bit low split of ln(2):
    // LN2_HI = 0x3fe62e42_fee00000 (low 20 mantissa bits zero so k_f * LN2_HI is exact for |k| <= 1075)
    // LN2_LO = 0x3dea39ef_35793c76
    let ln2_hi = f64::from_bits(0x3fe6_2e42_fee0_0000);
    let ln2_lo = f64::from_bits(0x3dea_39ef_3579_3c76);
    const INV_LN2: f64 = core::f64::consts::LOG2_E;

    // Nearest integer k to x / ln(2) using exact Cody-Waite range reduction
    let k_f = if x >= 0.0 {
        (x * INV_LN2 + 0.5) as i64 as f64
    } else {
        (x * INV_LN2 - 0.5) as i64 as f64
    };
    let k = k_f as i32;
    let r = (x - k_f * ln2_hi) - k_f * ln2_lo;

    // 12-term Horner polynomial for exp(r) on |r| <= ln(2)/2 (~0.3466)
    let mut poly = 1.0 / 479_001_600.0; // 1/12!
    poly = poly * r + 1.0 / 39_916_800.0; // 1/11!
    poly = poly * r + 1.0 / 3_628_800.0; // 1/10!
    poly = poly * r + 1.0 / 362_880.0; // 1/9!
    poly = poly * r + 1.0 / 40_320.0; // 1/8!
    poly = poly * r + 1.0 / 5_040.0; // 1/7!
    poly = poly * r + 1.0 / 720.0; // 1/6!
    poly = poly * r + 1.0 / 120.0; // 1/5!
    poly = poly * r + 1.0 / 24.0; // 1/4!
    poly = poly * r + 1.0 / 6.0; // 1/3!
    poly = poly * r + 0.5; // 1/2!
    poly = poly * r + 1.0;
    poly = poly * r + 1.0;

    // Scale by 2^k via IEEE-754 exponent bit manipulation (handling subnormal k < -1022).
    if k <= -1022 {
        let k1 = -1022i64;
        let k2 = (k as i64) - k1; // in [-53, 0]
        let scale1 = f64::from_bits(((k1 + 1023) as u64) << 52);
        let scale2 = f64::from_bits(((k2 + 1023) as u64) << 52);
        (poly * scale1) * scale2
    } else {
        let scale = f64::from_bits((((k as i64) + 1023) as u64) << 52);
        poly * scale
    }
}

/// Computes $e^x$ for `f64` in both `std` and `no_std` builds.
#[inline]
pub fn exp_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.exp()
    }
    #[cfg(not(feature = "std"))]
    {
        exp_f64_soft(x)
    }
}

/// Software IEEE-754 natural logarithm $\ln(x)$ over `f64` using mantissa/exponent
/// decomposition $x = m \cdot 2^e$ ($m \in [\frac{\sqrt{2}}{2}, \sqrt{2})$) and an odd-power
/// $\operatorname{atanh}$ series for $\ln(m) = 2 \sum_{j=0}^{7} \frac{u^{2j+1}}{2j+1}$ where
/// $u = \frac{m - 1}{m + 1}$.
#[inline]
pub fn ln_f64_soft(x: f64) -> f64 {
    if x < 0.0 || x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }

    // Normalize subnormals (x < 2^-1022) by scaling by 2^54 before bit extraction.
    let (x_norm, exp_offset) = if x < f64::MIN_POSITIVE {
        (x * f64::from_bits(0x4350_0000_0000_0000), -54i64)
    } else {
        (x, 0i64)
    };

    let bits = x_norm.to_bits();
    let mut exp = ((bits >> 52) & 0x7ff) as i64 - 1023 + exp_offset;
    let mant_bits = (bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000;
    let mut m = f64::from_bits(mant_bits); // m in [1.0, 2.0)

    if m > core::f64::consts::SQRT_2 {
        m *= 0.5;
        exp += 1;
    }

    // u = (m - 1) / (m + 1), |u| <= (sqrt(2) - 1) / (sqrt(2) + 1) ~ 0.17157
    let u = (m - 1.0) / (m + 1.0);
    let u2 = u * u;
    let mut series = 1.0 / 17.0;
    series = series * u2 + 1.0 / 15.0;
    series = series * u2 + 1.0 / 13.0;
    series = series * u2 + 1.0 / 11.0;
    series = series * u2 + 1.0 / 9.0;
    series = series * u2 + 1.0 / 7.0;
    series = series * u2 + 1.0 / 5.0;
    series = series * u2 + 1.0 / 3.0;
    series = series * u2 + 1.0;
    let ln_m = 2.0 * u * series;

    (exp as f64) * core::f64::consts::LN_2 + ln_m
}

/// Computes $\ln(x)$ for `f64` in both `std` and `no_std` builds.
#[inline]
pub fn ln_f64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.ln()
    }
    #[cfg(not(feature = "std"))]
    {
        ln_f64_soft(x)
    }
}

/// Numerically stable two-term log-sum-exp $\ln(e^a + e^b) = \max(a, b) + \ln(1 + e^{-|a - b|})$.
#[inline]
pub fn log_sum_exp_pair_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max_val = if a >= b { a } else { b };
    if max_val.is_infinite() {
        return max_val;
    }
    let diff = if a >= b { b - a } else { a - b };
    max_val + ln_f64(1.0 + exp_f64(diff))
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

    #[test]
    fn exp_and_ln_f64_soft_match_std_across_domain() {
        let exp_samples = [-500.0, -50.0, -2.5, -0.5, 0.0, 0.25, 1.0, 2.5, 20.0, 200.0];
        for &x in &exp_samples {
            let got = exp_f64_soft(x);
            let expected = x.exp();
            let rel_err = ((got - expected) / expected).abs();
            assert!(
                rel_err <= 8.0 * f64::EPSILON,
                "exp_f64_soft({x}) = {got:e}, expected {expected:e}, rel_err = {rel_err:e}"
            );
        }

        let ln_samples = [
            1e-310_f64,
            f64::MIN_POSITIVE,
            1e-50,
            1e-6,
            0.1,
            0.5,
            1.0,
            core::f64::consts::E,
            10.0,
            1e50,
            1e300,
        ];
        for &x in &ln_samples {
            let got = ln_f64_soft(x);
            let expected = x.ln();
            let abs_or_rel = if expected == 0.0 {
                (got - expected).abs()
            } else {
                ((got - expected) / expected).abs()
            };
            assert!(
                abs_or_rel <= 8.0 * f64::EPSILON,
                "ln_f64_soft({x:e}) = {got:e}, expected {expected:e}, err = {abs_or_rel:e}"
            );
        }

        let lse = log_sum_exp_pair_f64(1000.0, 1000.0);
        assert!((lse - (1000.0 + core::f64::consts::LN_2)).abs() < 1e-12);
    }
}
