//! Rényi Differential Privacy (`RDP`) accounting and exact Discrete-Gaussian `DpFedAvgV1`
//! mechanism (`ADR-0023` §4.2, `W12-03`, `ST-02`, `Wave 16B`).
//!
//! This module is `no_std` compatible (with `alloc`-gated cohort/vector mechanism helpers and
//! `std`-gated OS CSPRNG entropy constructors) and provides:
//! - [`RdpAccountantV1`] over the canonical 16-order Rényi grid $\Lambda$ (`Mironov, Talwar, &
//!   Zhang 2019` SGM binomial expansion + `Mironov 2017` / `Balle et al. 2020` conversions),
//! - Exact integer-lattice $\ell_2$ clipping (`u128`-verified $\|\hat{\Delta}_i\|_2^2 \le B^2$)
//!   and exact associative `i128` summation, eliminating floating-point sensitivity underestimation
//!   (`Casacuberta et al. CCS 2022`, `arXiv:2112.05307`),
//! - Exact floating-point-free Discrete Gaussian sampling $\mathcal{N}_{\mathbb{Z}}(0, \sigma_{\mathbb{Z}}^2)$
//!   (`Canonne, Kamath, & Steinke NeurIPS 2020`, `arXiv:2004.00010` Algorithms 1–3) over exact
//!   `u128` rational Bernoulli-exponential rejection,
//! - Sealed `SampledDpCohortV1`, [`DpNoiseEntropyV1`], and [`VerifiedDpMechanismRoundV1`] proof
//!   tokens binding run, round, parent digest, profile, OS CSPRNG provenance, and sanitized output.

#[cfg(feature = "alloc")]
use alloc::collections::BTreeSet;
#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::crypto::{
    sha256_slices, CryptoError, PrivacyAccountantHook, RdpBudgetDescriptorV1, Sha256CounterDrbgV1,
};
#[cfg(feature = "alloc")]
use crate::num::sqrt_f64_soft;
use crate::num::{abs_f64, ln_f64_soft, log_sum_exp_pair_f64};
#[cfg(feature = "alloc")]
use crate::traits::{GradientUpdate, PeerId};

/// Domain tag for deriving per-round Poisson cohort sampling seeds (`Wave 16B`).
pub const DP_POISSON_SEED_DOMAIN_V1: &[u8] = b"swarmtorch.dp_fedavg.poisson.seed.v1";

/// Domain tag for `Sha256CounterDrbgV1` during Poisson cohort Bernoulli trials (`Wave 16B`).
pub const DP_POISSON_DRBG_DOMAIN_V1: &[u8] = b"swarmtorch.dp_fedavg.poisson.drbg.v1";

/// Domain tag for committing to authorized/sampled cohort sets (`Wave 16B`).
pub const DP_COHORT_DIGEST_DOMAIN_V1: &[u8] = b"swarmtorch.dp_fedavg.cohort.digest.v1";

/// Domain tag for deriving per-round Discrete Gaussian noise DRBG seeds (`Wave 16B`).
pub const DP_GAUSSIAN_SEED_DOMAIN_V1: &[u8] = b"swarmtorch.dp_fedavg.dgauss.seed.v1";

/// Domain tag for `Sha256CounterDrbgV1` during Discrete Gaussian sampling (`Wave 16B`).
pub const DP_GAUSSIAN_DRBG_DOMAIN_V1: &[u8] = b"swarmtorch.dp_fedavg.dgauss.drbg.v1";

/// Domain tag for one-way noise entropy seed commitment inside [`VerifiedDpMechanismRoundV1`].
pub const DP_NOISE_COMMITMENT_DOMAIN_V1: &[u8] = b"swarmtorch.dp_fedavg.noise.commit.v1";

/// Domain tag for binding the sanitized output vector inside [`VerifiedDpMechanismRoundV1`].
pub const DP_OUTPUT_DIGEST_DOMAIN_V1: &[u8] = b"swarmtorch.dp_fedavg.output.digest.v1";

/// Fixed integer lattice $\ell_2$ clipping radius $B = 2^{16} = 65{,}536$ (`Wave 16B`).
///
/// Every client update quantized onto $\mathbb{Z}^d$ is strictly checked in exact `u128` arithmetic
/// to satisfy $\sum_j \hat{\Delta}_{i,j}^2 \le B^2 = 4{,}294{,}967{,}296$.
pub const DP_DISCRETE_LATTICE_RADIUS_V1: u64 = 1 << 16;

/// Maximum supported integer discrete Gaussian scale $\sigma_{\mathbb{Z}} \le 2^{24}$ (`Wave 16B`),
/// supporting $\sigma = \sigma_{\mathbb{Z}} / B \in (0, 256.0]$ while keeping $2\sigma_{\mathbb{Z}}^2 (\sigma_{\mathbb{Z}} + 1)^2 < 2^{98} \ll 2^{128}$.
pub const DP_MAX_DISCRETE_SIGMA_Z_V1: u64 = 1 << 24;

/// Number of Rényi divergence orders in [`RDP_ALPHA_GRID_V1`].
pub const RDP_GRID_LEN: usize = 16;

/// Canonical 16-order Rényi grid $\Lambda$ (`ADR-0023` §4.2).
pub const RDP_ALPHA_GRID_V1: [f64; RDP_GRID_LEN] = [
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0,
];

/// Unit of privacy protected by the mechanism (`ADR-0023` §4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub enum DpUnitV1 {
    /// Protects an entire participating client's contribution in a federated round.
    ClientLevel,
    /// Protects a single training example within a local or central dataset.
    ExampleLevel,
}

/// Neighboring-dataset adjacency relation (`ADR-0023` §4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub enum DpAdjacencyV1 {
    /// Adding or removing one unit ($\Delta_2 = \tau$).
    AddRemoveOne,
    /// Replacing one unit with another ($\Delta_2 = 2\tau$, halving the effective noise multiplier).
    ReplaceOne,
}

/// Conversion formula from Rényi DP $(\alpha, \rho(\alpha))$ to $(\varepsilon, \delta)$-DP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub enum DpConversionFormulaV1 {
    /// Standard Mironov (2017) Proposition 3 bound:
    /// $\varepsilon(\delta) = \min_{\alpha} \left(\rho(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}\right)$.
    Mironov2017,
    /// Tight Balle et al. (AISTATS 2020) Theorem 21 bound:
    /// $\varepsilon(\delta) = \min_{\alpha} \max\left(0, \rho(\alpha) + \frac{\ln(1/\delta) + (\alpha - 1)\ln(1 - 1/\alpha) - \ln(\alpha)}{\alpha - 1}\right)$.
    Balle2020Tight,
}

/// Explicit privacy guarantee classification for [`DpMechanismProfileV1`] and [`RdpAccountantV1`] (`ST-02`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "alloc", serde(rename_all = "snake_case"))]
pub enum PrivacyAccountingClassificationV1 {
    /// Experimental hypothetical RDP budget accounting only.
    ///
    /// **No** Poisson client subsampling, calibrated Gaussian noise injection, or formal Differential
    /// Privacy protection is claimed or enforced.
    #[default]
    ExperimentalHypotheticalBudgetOnly,
    /// Certified classification backed by `DpFedAvgMechanismV1` (exact integer-lattice $\ell_2$
    /// clipping, Poisson client subsampling, exact Discrete Gaussian noise, and sealed
    /// [`VerifiedDpMechanismRoundV1`] proof accounting). Still fails closed in standalone
    /// [`RdpAccountantV1::new`] and [`PrivacyAccountantHook::record_clipped_round`].
    VerifiedMechanismProtected,
}

/// Normative configuration profile for Differential Privacy accounting (`ADR-0023` §4.2, `ST-02`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct DpMechanismProfileV1 {
    /// Unit of privacy (`ClientLevel` or `ExampleLevel`).
    pub unit: DpUnitV1,
    /// Adjacency relation (`AddRemoveOne` or `ReplaceOne`).
    pub adjacency: DpAdjacencyV1,
    /// Per-unit $\ell_2$ clipping threshold $\tau > 0$.
    pub clipping_radius_tau: f32,
    /// Gaussian noise multiplier $\sigma = \sigma_{\text{std}} / \tau > 0$.
    pub noise_multiplier: f32,
    /// Target $(\varepsilon, \delta)$-DP failure probability $\delta \in (0, 1)$.
    pub target_delta: f64,
    /// Hard ceiling $\varepsilon_{\max} > 0$ above which rounds are rejected with [`CryptoError::PrivacyBudgetExhausted`].
    pub max_epsilon: f64,
    /// Conversion formula from RDP to $(\varepsilon, \delta)$-DP.
    pub conversion_formula: DpConversionFormulaV1,
    /// Accounting classification (`ExperimentalHypotheticalBudgetOnly` when used as a standalone budget tracker).
    #[cfg_attr(feature = "alloc", serde(default))]
    pub classification: PrivacyAccountingClassificationV1,
}

impl DpMechanismProfileV1 {
    /// Whether this profile claims genuine noise-injected Differential Privacy guarantees
    /// (`false` for [`PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly`]).
    #[inline]
    pub const fn guarantees_differential_privacy(&self) -> bool {
        matches!(
            self.classification,
            PrivacyAccountingClassificationV1::VerifiedMechanismProtected
        )
    }

    /// Validate all privacy mechanism parameters.
    pub fn validate(&self) -> Result<(), CryptoError> {
        if !self.clipping_radius_tau.is_finite()
            || self.clipping_radius_tau <= 0.0
            || !self.noise_multiplier.is_finite()
            || self.noise_multiplier <= 0.0
            || !self.target_delta.is_finite()
            || self.target_delta <= 0.0
            || self.target_delta >= 1.0
            || !self.max_epsilon.is_finite()
            || self.max_epsilon <= 0.0
        {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        Ok(())
    }

    /// Effective noise-to-sensitivity ratio $\sigma_{\text{eff}} = \sigma_{\text{std}} / \Delta_2$.
    ///
    /// Under `AddRemoveOne`, $\Delta_2 = \tau$ so $\sigma_{\text{eff}} = \sigma$.
    /// Under `ReplaceOne`, $\Delta_2 = 2\tau$ so $\sigma_{\text{eff}} = \sigma / 2$.
    #[inline]
    pub fn effective_noise_multiplier(&self) -> f64 {
        match self.adjacency {
            DpAdjacencyV1::AddRemoveOne => self.noise_multiplier as f64,
            DpAdjacencyV1::ReplaceOne => (self.noise_multiplier as f64) * 0.5,
        }
    }
}

/// Computes $\ln \binom{k}{j}$ for integer $0 \le j \le k \le 64$ using exact integer arithmetic
/// before taking [`ln_f64_soft`].
#[inline]
fn ln_binomial_u32(k: u32, j: u32) -> f64 {
    if j == 0 || j == k {
        return 0.0;
    }
    let j_sym = if j > k - j { k - j } else { j };
    let mut ln_acc = 0.0f64;
    for i in 1..=j_sym {
        let num = (k - j_sym + i) as f64;
        let den = i as f64;
        ln_acc += ln_f64_soft(num) - ln_f64_soft(den);
    }
    ln_acc
}

/// Computes the exact integer-order Rényi divergence $\rho_{\text{SGM}}(k, q, \sigma_{\text{eff}})$
/// for $k \in \{2, \dots, 64\}$ under Poisson subsampling (`Mironov et al. 2019` §3):
///
/// $$\rho_{\text{SGM}}(k, q, \sigma) = \frac{1}{k - 1} \ln \left( \sum_{j=0}^{k} \binom{k}{j} (1 - q)^{k - j} q^j \exp\left(\frac{j(j - 1)}{2\sigma^2}\right) \right)$$
pub fn compute_sgm_rdp_integer_order(k: u32, q: f64, sigma_eff: f64) -> Result<f64, CryptoError> {
    if k < 2
        || !q.is_finite()
        || !(0.0..=1.0).contains(&q)
        || !sigma_eff.is_finite()
        || sigma_eff <= 0.0
    {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    if q == 0.0 {
        return Ok(0.0);
    }
    let full_gaussian = (k as f64) / (2.0 * sigma_eff * sigma_eff);
    if q == 1.0 {
        return Ok(full_gaussian);
    }

    let ln_q = ln_f64_soft(q);
    let ln_one_minus_q = ln_f64_soft(1.0 - q);
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma_eff * sigma_eff);

    let mut log_moment = f64::NEG_INFINITY;
    for j in 0..=k {
        let ln_comb = ln_binomial_u32(k, j);
        let term = ln_comb
            + ((k - j) as f64) * ln_one_minus_q
            + (j as f64) * ln_q
            + ((j as f64) * ((j as f64) - 1.0)) * inv_two_sigma_sq;
        log_moment = log_sum_exp_pair_f64(log_moment, term);
    }

    let rdp = log_moment / ((k - 1) as f64);
    if !rdp.is_finite() {
        return Ok(full_gaussian);
    }
    // Enforce non-negativity and subsampling monotonicity (rdp <= full_gaussian).
    Ok(rdp.clamp(0.0, full_gaussian))
}

/// Computes the 16-order Rényi divergence increment $\Delta\rho(\alpha)$ across [`RDP_ALPHA_GRID_V1`]
/// for a single round with cohort subsampling rate $q \in (0, 1]$ and effective noise multiplier $\sigma_{\text{eff}} > 0$.
pub fn compute_round_rdp_schedule_v1(
    q: f64,
    sigma_eff: f64,
) -> Result<[f64; RDP_GRID_LEN], CryptoError> {
    if !q.is_finite() || q <= 0.0 || q > 1.0 || !sigma_eff.is_finite() || sigma_eff <= 0.0 {
        return Err(CryptoError::InvalidPrivacyParameter);
    }

    let mut out = [0.0f64; RDP_GRID_LEN];
    if q == 1.0 {
        let denom = 2.0 * sigma_eff * sigma_eff;
        for (i, &alpha) in RDP_ALPHA_GRID_V1.iter().enumerate() {
            out[i] = alpha / denom;
        }
        return Ok(out);
    }

    // First compute exact integer orders in the grid:
    // indices: 3->2.0, 5->3.0, 6->4.0, 7->5.0, 8->6.0, 9->8.0, 10->12.0, 11->16.0, 12->24.0, 13->32.0, 14->48.0, 15->64.0
    let integer_slots: [(usize, u32); 12] = [
        (3, 2),
        (5, 3),
        (6, 4),
        (7, 5),
        (8, 6),
        (9, 8),
        (10, 12),
        (11, 16),
        (12, 24),
        (13, 32),
        (14, 48),
        (15, 64),
    ];
    for &(slot, k) in &integer_slots {
        out[slot] = compute_sgm_rdp_integer_order(k, q, sigma_eff)?;
    }

    // Fractional orders via log-convexity of the cumulant generating function K(alpha) = (alpha - 1) * rho(alpha):
    // - For alpha in {1.25, 1.5, 1.75} in (1, 2): monotonicity of rho(alpha) implies rho(alpha) <= min(rho(2), alpha / (2*sigma^2)).
    let rho_2 = out[3];
    let rho_3 = out[5];
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma_eff * sigma_eff);
    for slot in 0..3 {
        let alpha = RDP_ALPHA_GRID_V1[slot];
        let full_gauss = alpha * inv_two_sigma_sq;
        out[slot] = if rho_2 < full_gauss {
            rho_2
        } else {
            full_gauss
        };
    }

    // - For alpha = 2.5 in (2, 3): K(2.5) <= 0.5 * K(2) + 0.5 * K(3) => 1.5 * rho(2.5) <= 0.5 * rho(2) + 1.0 * rho(3)
    //   => rho(2.5) <= (rho(2) + 2 * rho(3)) / 3.
    let interp_2_5 = (rho_2 + 2.0 * rho_3) / 3.0;
    let full_2_5 = 2.5 * inv_two_sigma_sq;
    out[4] = if interp_2_5 < full_2_5 {
        interp_2_5
    } else {
        full_2_5
    };

    Ok(out)
}

/// Converts a 16-order Rényi divergence schedule `rdp_orders` into the tightest $(\varepsilon(\delta), \alpha^*)$ pair
/// under `formula` (`ADR-0023` §4.2).
pub fn convert_rdp_to_approx_dp_v1(
    rdp_orders: &[f64; RDP_GRID_LEN],
    target_delta: f64,
    formula: DpConversionFormulaV1,
) -> Result<(f64, f64), CryptoError> {
    if !target_delta.is_finite() || target_delta <= 0.0 || target_delta >= 1.0 {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    let ln_inv_delta = -ln_f64_soft(target_delta);

    let mut best_eps = f64::INFINITY;
    let mut best_alpha = RDP_ALPHA_GRID_V1[0];

    for (i, &alpha) in RDP_ALPHA_GRID_V1.iter().enumerate() {
        let rho = rdp_orders[i];
        if !rho.is_finite() || rho < 0.0 {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        let eps_candidate = match formula {
            DpConversionFormulaV1::Mironov2017 => rho + ln_inv_delta / (alpha - 1.0),
            DpConversionFormulaV1::Balle2020Tight => {
                let term = (ln_inv_delta + (alpha - 1.0) * ln_f64_soft(1.0 - 1.0 / alpha)
                    - ln_f64_soft(alpha))
                    / (alpha - 1.0);
                let raw = rho + term;
                if raw > 0.0 {
                    raw
                } else {
                    0.0
                }
            }
        };
        if eps_candidate < best_eps {
            best_eps = eps_candidate;
            best_alpha = alpha;
        }
    }

    if !best_eps.is_finite() {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    Ok((best_eps, best_alpha))
}

/// Stateful Rényi Differential Privacy accountant (`ADR-0023` §4.2, `W12-03`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct RdpAccountantV1 {
    profile: DpMechanismProfileV1,
    cumulative_rdp_orders: [f64; RDP_GRID_LEN],
    committed_rounds: u64,
    last_optimal_alpha: f64,
    last_converted_epsilon: f64,
}

impl RdpAccountantV1 {
    /// Construct a new hypothetical [`RdpAccountantV1`] after validating `profile`.
    ///
    /// Fails closed with [`CryptoError::InvalidPrivacyParameter`] if `profile.classification` is
    /// [`PrivacyAccountingClassificationV1::VerifiedMechanismProtected`], because a standalone
    /// accountant does not execute Poisson subsampling or calibrated Gaussian noise injection (`ST-02`).
    pub fn new(profile: DpMechanismProfileV1) -> Result<Self, CryptoError> {
        profile.validate()?;
        if profile.classification
            != PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly
        {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        Ok(Self {
            profile,
            cumulative_rdp_orders: [0.0; RDP_GRID_LEN],
            committed_rounds: 0,
            last_optimal_alpha: RDP_ALPHA_GRID_V1[0],
            last_converted_epsilon: 0.0,
        })
    }

    /// Construct a certified [`RdpAccountantV1`] paired with `DpFedAvgMechanismV1` (`ST-02`, `Wave 16B`).
    ///
    /// Requires `profile.classification == PrivacyAccountingClassificationV1::VerifiedMechanismProtected`
    /// and `profile.unit == DpUnitV1::ClientLevel`. Subsequent rounds can ONLY be recorded via
    /// [`Self::record_verified_mechanism_round`] with a sealed [`VerifiedDpMechanismRoundV1`] proof;
    /// calling [`PrivacyAccountantHook::record_clipped_round`] fails closed.
    pub fn new_verified_mechanism(profile: DpMechanismProfileV1) -> Result<Self, CryptoError> {
        profile.validate()?;
        if profile.classification != PrivacyAccountingClassificationV1::VerifiedMechanismProtected
            || profile.unit != DpUnitV1::ClientLevel
        {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        let _ = compute_discrete_sigma_z_v1(profile.noise_multiplier)?;
        Ok(Self {
            profile,
            cumulative_rdp_orders: [0.0; RDP_GRID_LEN],
            committed_rounds: 0,
            last_optimal_alpha: RDP_ALPHA_GRID_V1[0],
            last_converted_epsilon: 0.0,
        })
    }

    /// Reconstruct a certified [`RdpAccountantV1`] from persisted durable ledger state (`dp/ledger.ndjson`).
    ///
    /// Requires `VerifiedMechanismProtected` + `ClientLevel`, finite/non-negative cumulative RDP
    /// orders, consistent `(committed_rounds == 0) == all_zero`, and converted $\varepsilon \le \varepsilon_{\max}$.
    pub fn restore_verified_mechanism_from_orders(
        profile: DpMechanismProfileV1,
        cumulative_rdp_orders: [f64; RDP_GRID_LEN],
        committed_rounds: u64,
    ) -> Result<Self, CryptoError> {
        profile.validate()?;
        if profile.classification != PrivacyAccountingClassificationV1::VerifiedMechanismProtected
            || profile.unit != DpUnitV1::ClientLevel
        {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        let _ = compute_discrete_sigma_z_v1(profile.noise_multiplier)?;
        Self::restore_orders_inner(profile, cumulative_rdp_orders, committed_rounds)
    }

    fn restore_orders_inner(
        profile: DpMechanismProfileV1,
        cumulative_rdp_orders: [f64; RDP_GRID_LEN],
        committed_rounds: u64,
    ) -> Result<Self, CryptoError> {
        for &rho in &cumulative_rdp_orders {
            if !rho.is_finite() || rho < 0.0 {
                return Err(CryptoError::InvalidPrivacyParameter);
            }
        }
        let all_zero = cumulative_rdp_orders.iter().all(|&r| r == 0.0);
        if (committed_rounds == 0) != all_zero {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        let (converted_eps, opt_alpha) = if committed_rounds == 0 && all_zero {
            (0.0, RDP_ALPHA_GRID_V1[0])
        } else {
            let (eps, alpha) = convert_rdp_to_approx_dp_v1(
                &cumulative_rdp_orders,
                profile.target_delta,
                profile.conversion_formula,
            )?;
            if eps > profile.max_epsilon {
                return Err(CryptoError::PrivacyBudgetExhausted);
            }
            (eps, alpha)
        };
        Ok(Self {
            profile,
            cumulative_rdp_orders,
            committed_rounds,
            last_optimal_alpha: opt_alpha,
            last_converted_epsilon: converted_eps,
        })
    }

    /// Reconstruct an [`RdpAccountantV1`] from persisted durable ledger state (`dp/ledger.ndjson`).
    ///
    /// Verifies that `profile` and every persisted `cumulative_rdp_orders[i]` are finite and $\ge 0$,
    /// recomputes $(\varepsilon(\delta), \alpha^*)$, and fails closed with [`CryptoError::PrivacyBudgetExhausted`]
    /// if the replayed cumulative spend already exceeds `profile.max_epsilon`, or [`CryptoError::InvalidPrivacyParameter`]
    /// if `profile.classification` claims [`PrivacyAccountingClassificationV1::VerifiedMechanismProtected`].
    pub fn restore_from_orders(
        profile: DpMechanismProfileV1,
        cumulative_rdp_orders: [f64; RDP_GRID_LEN],
        committed_rounds: u64,
    ) -> Result<Self, CryptoError> {
        profile.validate()?;
        if profile.classification
            != PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly
        {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        Self::restore_orders_inner(profile, cumulative_rdp_orders, committed_rounds)
    }

    /// Return the configured [`DpMechanismProfileV1`].
    pub const fn profile(&self) -> &DpMechanismProfileV1 {
        &self.profile
    }

    /// Whether this accountant backs an active certified Differential Privacy mechanism.
    ///
    /// Always returns `false` for [`PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly`] (`ST-02`).
    pub const fn guarantees_differential_privacy(&self) -> bool {
        matches!(
            self.profile.classification,
            PrivacyAccountingClassificationV1::VerifiedMechanismProtected
        )
    }

    /// Return the cumulative Rényi divergence array $\sum_t \rho_t(\alpha)$ across [`RDP_ALPHA_GRID_V1`].
    pub const fn cumulative_rdp_orders(&self) -> &[f64; RDP_GRID_LEN] {
        &self.cumulative_rdp_orders
    }

    /// Return the number of rounds debited from the accountant.
    pub const fn committed_rounds(&self) -> u64 {
        self.committed_rounds
    }

    /// Return the current converted $(\varepsilon(\delta), \delta)$ privacy expenditure and optimal order $\alpha^*$.
    pub const fn current_approx_dp(&self) -> (f64, f64, f64) {
        (
            self.last_converted_epsilon,
            self.profile.target_delta,
            self.last_optimal_alpha,
        )
    }

    /// Preview the privacy budget after one additional round with clipping radius `tau` and
    /// subsampling rate `subsampling_rate_q` **without** mutating state.
    ///
    /// Fails closed with:
    /// - [`CryptoError::InvalidPrivacyParameter`] if `tau` is non-finite, $\le 0$, or exceeds
    ///   `profile.clipping_radius_tau` (which would violate the calibrated noise ratio $\sigma$),
    ///   or if `subsampling_rate_q` $\notin (0, 1]$.
    /// - [`CryptoError::PrivacyBudgetExhausted`] if the resulting converted $\varepsilon(\delta)$
    ///   exceeds `profile.max_epsilon`.
    pub fn preview_clipped_round(
        &self,
        tau: f32,
        subsampling_rate_q: f64,
    ) -> Result<(RdpBudgetDescriptorV1, [f64; RDP_GRID_LEN], f64), CryptoError> {
        if !tau.is_finite() || tau <= 0.0 || tau > self.profile.clipping_radius_tau {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        let sigma_eff = self.profile.effective_noise_multiplier();
        let step_rdp = compute_round_rdp_schedule_v1(subsampling_rate_q, sigma_eff)?;

        let mut next_orders = [0.0f64; RDP_GRID_LEN];
        for i in 0..RDP_GRID_LEN {
            next_orders[i] = self.cumulative_rdp_orders[i] + step_rdp[i];
            if !next_orders[i].is_finite() {
                return Err(CryptoError::PrivacyBudgetExhausted);
            }
        }

        let (converted_eps, opt_alpha) = convert_rdp_to_approx_dp_v1(
            &next_orders,
            self.profile.target_delta,
            self.profile.conversion_formula,
        )?;
        if converted_eps > self.profile.max_epsilon {
            return Err(CryptoError::PrivacyBudgetExhausted);
        }

        let opt_idx = RDP_ALPHA_GRID_V1
            .iter()
            .position(|&a| abs_f64(a - opt_alpha) < 1e-12)
            .unwrap_or(0);

        let desc = RdpBudgetDescriptorV1 {
            alpha: opt_alpha,
            epsilon_alpha: next_orders[opt_idx],
            target_delta: self.profile.target_delta,
            clipping_radius_tau: self.profile.clipping_radius_tau,
            noise_multiplier: self.profile.noise_multiplier,
        };
        Ok((desc, next_orders, converted_eps))
    }

    /// Preview a certified `DpFedAvgMechanismV1` round without mutating accountant state,
    /// requiring exact profile agreement with the sealed [`VerifiedDpMechanismRoundV1`] proof.
    pub fn preview_verified_mechanism_round(
        &self,
        proof: &VerifiedDpMechanismRoundV1,
    ) -> Result<(RdpBudgetDescriptorV1, [f64; RDP_GRID_LEN], f64), CryptoError> {
        if self.profile.classification
            != PrivacyAccountingClassificationV1::VerifiedMechanismProtected
            || proof.profile != self.profile
        {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        self.preview_clipped_round(proof.profile.clipping_radius_tau, proof.subsampling_rate_q)
    }

    /// Commit a certified `DpFedAvgMechanismV1` round into this accountant using a sealed
    /// [`VerifiedDpMechanismRoundV1`] proof token (`ST-02`, `Wave 16B`).
    pub fn record_verified_mechanism_round(
        &mut self,
        proof: &VerifiedDpMechanismRoundV1,
    ) -> Result<(), CryptoError> {
        let (desc, next_orders, converted_eps) = self.preview_verified_mechanism_round(proof)?;
        let next_rounds = self
            .committed_rounds
            .checked_add(1)
            .ok_or(CryptoError::PrivacyBudgetExhausted)?;
        self.cumulative_rdp_orders = next_orders;
        self.committed_rounds = next_rounds;
        self.last_optimal_alpha = desc.alpha;
        self.last_converted_epsilon = converted_eps;
        Ok(())
    }
}

impl PrivacyAccountantHook for RdpAccountantV1 {
    fn current_budget(&self) -> RdpBudgetDescriptorV1 {
        let opt_idx = RDP_ALPHA_GRID_V1
            .iter()
            .position(|&a| abs_f64(a - self.last_optimal_alpha) < 1e-12)
            .unwrap_or(0);
        RdpBudgetDescriptorV1 {
            alpha: self.last_optimal_alpha,
            epsilon_alpha: self.cumulative_rdp_orders[opt_idx],
            target_delta: self.profile.target_delta,
            clipping_radius_tau: self.profile.clipping_radius_tau,
            noise_multiplier: self.profile.noise_multiplier,
        }
    }

    fn record_clipped_round(
        &mut self,
        tau: f32,
        subsampling_rate_q: f64,
    ) -> Result<(), CryptoError> {
        if self.profile.classification
            != PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly
        {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        let (desc, next_orders, converted_eps) =
            self.preview_clipped_round(tau, subsampling_rate_q)?;
        let next_rounds = self
            .committed_rounds
            .checked_add(1)
            .ok_or(CryptoError::PrivacyBudgetExhausted)?;
        self.cumulative_rdp_orders = next_orders;
        self.committed_rounds = next_rounds;
        self.last_optimal_alpha = desc.alpha;
        self.last_converted_epsilon = converted_eps;
        Ok(())
    }
}

/// Entropy source provenance for certified DP noise generation (`Wave 16B`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "alloc", serde(rename_all = "snake_case"))]
pub enum DpEntropyProvenanceV1 {
    /// Seeded directly from the host OS CSPRNG via `getrandom::getrandom`.
    OsCsprng,
    /// Caller-supplied explicit seed for deterministic testing / `no_std` simulation ONLY.
    /// Rejected by production durable certified DP ledger debits.
    ExplicitSeedNonProduction,
}

/// Entropy source provenance for Poisson cohort sampling (`Wave 16B`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "alloc", serde(rename_all = "snake_case"))]
pub enum DpSamplingProvenanceV1 {
    /// Sampled using host OS CSPRNG entropy (`getrandom::getrandom`).
    OsCsprng,
    /// Deterministic full cohort inclusion (`q == 1.0`, every authorized peer included with probability `1`).
    DeterministicFullCohort,
    /// Sampled using caller-supplied explicit seed for deterministic testing / `no_std` simulation ONLY.
    ExplicitSeedNonProduction,
}

/// Single-use, non-`Clone` sealed 256-bit noise entropy handle for `DpFedAvgMechanismV1` (`Wave 16B`).
///
/// Production code under `std` constructs `Self::generate_os_csprng`, which cannot be spoofed
/// from outside `swarm_torch_core::dp` and cannot be cloned or reused across rounds. Its `Debug`
/// implementation redacts the raw seed bytes.
pub struct DpNoiseEntropyV1 {
    seed: [u8; 32],
    provenance: DpEntropyProvenanceV1,
}

impl core::fmt::Debug for DpNoiseEntropyV1 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DpNoiseEntropyV1")
            .field("seed", &"[REDACTED]")
            .field("provenance", &self.provenance)
            .finish()
    }
}

impl DpNoiseEntropyV1 {
    /// Draw 256 bits of fresh cryptographically secure entropy from the host OS CSPRNG (`getrandom`).
    #[cfg(feature = "std")]
    pub fn generate_os_csprng() -> Result<Self, CryptoError> {
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed).map_err(|_| CryptoError::VerificationFailed)?;
        if seed == [0u8; 32] {
            return Err(CryptoError::AllZeroSeed);
        }
        Ok(Self {
            seed,
            provenance: DpEntropyProvenanceV1::OsCsprng,
        })
    }

    /// Construct a deterministic explicit-seed entropy handle for unit tests and `no_std` simulation.
    ///
    /// Always stamped [`DpEntropyProvenanceV1::ExplicitSeedNonProduction`] and rejects `[0u8; 32]`
    /// with [`CryptoError::AllZeroSeed`].
    pub fn from_explicit_seed_for_testing(seed: [u8; 32]) -> Result<Self, CryptoError> {
        if seed == [0u8; 32] {
            return Err(CryptoError::AllZeroSeed);
        }
        Ok(Self {
            seed,
            provenance: DpEntropyProvenanceV1::ExplicitSeedNonProduction,
        })
    }

    /// Return the entropy provenance tag.
    pub const fn provenance(&self) -> DpEntropyProvenanceV1 {
        self.provenance
    }

    /// Return a one-way SHA-256 commitment to the entropy handle without exposing the raw seed.
    pub fn commitment_sha256(&self) -> [u8; 32] {
        sha256_slices(&[DP_NOISE_COMMITMENT_DOMAIN_V1, b"handle", &self.seed])
    }
}

/// Compute the conservative integer Discrete Gaussian scale $\sigma_{\mathbb{Z}} = \lceil \sigma \cdot B \rceil$
/// with $B = \text{DP\_DISCRETE\_LATTICE\_RADIUS\_V1}$, guaranteeing $\sigma_{\mathbb{Z}} / B \ge \sigma$
/// without relying on `std`-only `f64::ceil`.
pub fn compute_discrete_sigma_z_v1(noise_multiplier: f32) -> Result<u64, CryptoError> {
    let sigma_f64 = noise_multiplier as f64;
    if !sigma_f64.is_finite() || sigma_f64 <= 0.0 {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    let raw = sigma_f64 * (DP_DISCRETE_LATTICE_RADIUS_V1 as f64);
    if !raw.is_finite() || raw <= 0.0 || raw > (DP_MAX_DISCRETE_SIGMA_Z_V1 as f64) {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    let trunc = raw as u64;
    let sigma_z = if (trunc as f64) < raw {
        trunc
            .checked_add(1)
            .ok_or(CryptoError::InvalidPrivacyParameter)?
    } else if trunc == 0 {
        1
    } else {
        trunc
    };
    if sigma_z == 0 || sigma_z > DP_MAX_DISCRETE_SIGMA_Z_V1 || (sigma_z as f64) < raw {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    Ok(sigma_z)
}

#[inline]
fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Exact floating-point-free $\text{Bernoulli}(\exp(-\text{num}/\text{den}))$ sampler for
/// $0 \le \text{num} \le \text{den}$, $\text{den} \ge 1$ (`Canonne, Kamath, & Steinke 2020`, Algorithm 1).
pub fn sample_bernoulli_exp1_u128(
    mut num: u128,
    mut den: u128,
    drbg: &mut Sha256CounterDrbgV1,
) -> Result<bool, CryptoError> {
    if den == 0 || num > den {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    if num == 0 {
        return Ok(true);
    }
    let g = gcd_u128(num, den);
    num /= g;
    den /= g;

    let mut k: u128 = 1;
    loop {
        // Factor Bernoulli(num / (k * den)) as Bernoulli(1 / k) AND Bernoulli(num / den),
        // eliminating `k * den` overflow across the entire u128 denominator range `1..=u128::MAX`.
        let step_pass =
            (k == 1 || drbg.uniform_u128_bounded(k) == 0) && drbg.uniform_u128_bounded(den) < num;
        if step_pass {
            k = k
                .checked_add(1)
                .ok_or(CryptoError::InvalidPrivacyParameter)?;
        } else {
            return Ok((k & 1) == 1);
        }
    }
}

/// Exact floating-point-free $\text{Bernoulli}(\exp(-\text{num}/\text{den}))$ sampler for arbitrary
/// non-negative rational $\text{num}/\text{den}$ (`Canonne, Kamath, & Steinke 2020`, Algorithm 1).
pub fn sample_bernoulli_exp_u128(
    num: u128,
    den: u128,
    drbg: &mut Sha256CounterDrbgV1,
) -> Result<bool, CryptoError> {
    if den == 0 {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    if num == 0 {
        return Ok(true);
    }
    let integer_part = num / den;
    let rem = num % den;
    for _ in 0..integer_part {
        if !sample_bernoulli_exp1_u128(1, 1, drbg)? {
            return Ok(false);
        }
    }
    sample_bernoulli_exp1_u128(rem, den, drbg)
}

/// Exact $\text{Geometric}(1 - e^{-1/t})$ sampler for integer scale $t \ge 1$
/// (`Canonne, Kamath, & Steinke 2020`, `sample_geometric_exp_fast(1/t)`).
pub fn sample_geometric_exp_fast_u128(
    t: u128,
    drbg: &mut Sha256CounterDrbgV1,
) -> Result<u128, CryptoError> {
    if t == 0 {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    let u = loop {
        let cand = drbg.uniform_u128_bounded(t);
        if sample_bernoulli_exp1_u128(cand, t, drbg)? {
            break cand;
        }
    };
    let mut v: u128 = 0;
    while sample_bernoulli_exp1_u128(1, 1, drbg)? {
        v = v
            .checked_add(1)
            .ok_or(CryptoError::InvalidPrivacyParameter)?;
    }
    v.checked_mul(t)
        .and_then(|vt| vt.checked_add(u))
        .ok_or(CryptoError::InvalidPrivacyParameter)
}

/// Exact Discrete Laplace $\text{Lap}_{\mathbb{Z}}(t)$ sampler for integer scale $t \ge 1$
/// with $\Pr[X = x] \propto \exp(-|x| / t)$ (`Canonne, Kamath, & Steinke 2020`, Algorithm 2).
pub fn sample_dlaplace_u128(t: u128, drbg: &mut Sha256CounterDrbgV1) -> Result<i128, CryptoError> {
    if t == 0 {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    loop {
        let sign_neg = (drbg.next_u32() & 1) == 1;
        let mag = sample_geometric_exp_fast_u128(t, drbg)?;
        if sign_neg && mag == 0 {
            continue;
        }
        let mag_i128 = i128::try_from(mag).map_err(|_| CryptoError::InvalidPrivacyParameter)?;
        return Ok(if sign_neg { -mag_i128 } else { mag_i128 });
    }
}

/// Exact floating-point-free Discrete Gaussian $\mathcal{N}_{\mathbb{Z}}(0, \sigma_{\mathbb{Z}}^2)$
/// sampler (`Canonne, Kamath, & Steinke NeurIPS 2020`, `arXiv:2004.00010` Algorithm 3).
///
/// Every arithmetic step uses exact integer operations in `u128`/`i128`, eliminating
/// floating-point rounding and LSB discretization side channels (`Mironov 2012`, `Jin et al. 2022`).
pub fn sample_dgauss_u128(
    sigma_z: u64,
    drbg: &mut Sha256CounterDrbgV1,
) -> Result<i128, CryptoError> {
    if sigma_z == 0 || sigma_z > DP_MAX_DISCRETE_SIGMA_Z_V1 {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    let s = sigma_z as u128;
    let s_sq = s * s;
    let t = s + 1;
    let denom = 2u128 * s_sq * t * t;

    loop {
        let candidate = sample_dlaplace_u128(t, drbg)?;
        let abs_c = candidate.unsigned_abs();
        let abs_c_t = abs_c
            .checked_mul(t)
            .ok_or(CryptoError::InvalidPrivacyParameter)?;
        let diff = abs_c_t.abs_diff(s_sq);
        let num = diff
            .checked_mul(diff)
            .ok_or(CryptoError::InvalidPrivacyParameter)?;
        if sample_bernoulli_exp_u128(num, denom, drbg)? {
            return Ok(candidate);
        }
    }
}

/// Exact 128-bit dyadic Bernoulli($q$) trial for any IEEE-754 `f64` $q \in [2^{-76}, 1.0]$
/// (`Wave 16B`). Zero rounding bias across all 53 mantissa bits of `q`.
pub fn sample_bernoulli_f64_exact_v1(
    q: f64,
    drbg: &mut Sha256CounterDrbgV1,
) -> Result<bool, CryptoError> {
    if !q.is_finite() || q <= 0.0 || q > 1.0 {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    if q == 1.0 {
        return Ok(true);
    }
    let bits = q.to_bits();
    let biased_exp = ((bits >> 52) & 0x7ff) as i32;
    if biased_exp == 0 {
        // Subnormal q < 2^-1022 cannot satisfy q * N >= 1 for any realistic cohort.
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    let unbiased_exp = biased_exp - 1023;
    if unbiased_exp >= 0 {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    let shift = (-unbiased_exp) as u32;
    if shift > 76 {
        return Err(CryptoError::InvalidPrivacyParameter);
    }
    let mantissa = (1u128 << 52) | ((bits & 0x000f_ffff_ffff_ffff) as u128);
    let threshold = mantissa << (76 - shift);
    Ok(drbg.next_u128() < threshold)
}

/// Project `update` onto the closed $\ell_2$ ball of radius `tau` and quantize onto the exact
/// integer lattice $\mathbb{Z}^d$ with radius $B = \text{DP\_DISCRETE\_LATTICE\_RADIUS\_V1}$
/// (`Wave 16B`, `arXiv:2112.05307`).
///
/// Returns `(quantized_i64_vec, original_l2_norm_f64, was_clipped)` and strictly enforces
/// $\sum_{j=0}^{d-1} \hat{\Delta}_j^2 \le B^2$ in exact `u128` arithmetic before returning.
#[cfg(feature = "alloc")]
pub fn clip_and_quantize_l2_lattice_v1(
    update: &[f32],
    tau: f32,
) -> Result<(Vec<i64>, f64, bool), CryptoError> {
    let tau_f64 = tau as f64;
    if update.is_empty() || !tau_f64.is_finite() || tau_f64 <= 0.0 {
        return Err(CryptoError::InvalidPrivacyParameter);
    }

    let mut norm_sq_f64 = 0.0f64;
    for &v in update {
        let vf = v as f64;
        if !vf.is_finite() {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        norm_sq_f64 += vf * vf;
    }
    let norm_f64 = sqrt_f64_soft(norm_sq_f64);
    if !norm_f64.is_finite() {
        return Err(CryptoError::InvalidPrivacyParameter);
    }

    let was_clipped = norm_f64 > tau_f64;
    let b_i64 = DP_DISCRETE_LATTICE_RADIUS_V1 as i64;
    let b_f64 = DP_DISCRETE_LATTICE_RADIUS_V1 as f64;
    let b_sq_u128 =
        (DP_DISCRETE_LATTICE_RADIUS_V1 as u128) * (DP_DISCRETE_LATTICE_RADIUS_V1 as u128);

    let denom = if was_clipped { norm_f64 } else { tau_f64 };
    let scale = b_f64 / denom;
    if !scale.is_finite() {
        return Err(CryptoError::InvalidPrivacyParameter);
    }

    let mut quantized = Vec::with_capacity(update.len());
    let mut norm_sq_u128 = 0u128;
    for &v in update {
        let scaled = (v as f64) * scale;
        if !scaled.is_finite() {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        // Cast `as i64` truncates toward zero (|q_j| <= |scaled|), avoiding std-only f64::trunc.
        let q_j = (scaled.clamp(-b_f64, b_f64) as i64).clamp(-b_i64, b_i64);
        let mag = q_j.unsigned_abs() as u128;
        norm_sq_u128 = norm_sq_u128
            .checked_add(mag * mag)
            .ok_or(CryptoError::InvalidPrivacyParameter)?;
        quantized.push(q_j);
    }

    // Hard exact-integer norm projection: eliminate any worst-case float rounding overshoot so
    // `sum(q_j^2) <= B^2` holds as an unconditional exact u128 inequality.
    while norm_sq_u128 > b_sq_u128 {
        let mut best_idx = None;
        let mut best_mag = 0u64;
        for (idx, &q_j) in quantized.iter().enumerate() {
            let mag = q_j.unsigned_abs();
            if mag > best_mag {
                best_mag = mag;
                best_idx = Some(idx);
            }
        }
        let idx = best_idx.ok_or(CryptoError::InvalidPrivacyParameter)?;
        let old_abs = quantized[idx].unsigned_abs() as u128;
        quantized[idx] -= quantized[idx].signum();
        let delta = 2u128 * old_abs - 1u128;
        norm_sq_u128 -= delta;
    }

    Ok((quantized, norm_f64, was_clipped))
}

#[cfg(feature = "alloc")]
fn compute_peer_set_digest_v1(tag: &[u8], peers: &BTreeSet<PeerId>) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(DP_COHORT_DIGEST_DOMAIN_V1);
    hasher.update(tag);
    hasher.update((peers.len() as u64).to_le_bytes());
    for peer in peers {
        hasher.update(peer.as_bytes());
    }
    hasher.finalize().into()
}

/// Sealed Poisson-sampled round cohort receipt (`Wave 16B`).
///
/// Private fields guarantee that [`DpFedAvgMechanismV1::execute_round_v1`] can only operate on
/// a cohort produced by exact Bernoulli($q$) trials (or deterministic full cohort when $q = 1.0$),
/// binding `run_id`, `round_id`, `authorized_cohort_size`, and `subsampling_rate_q`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct SampledDpCohortV1 {
    run_id_bytes: [u8; 16],
    round_id: u64,
    authorized_cohort_size: usize,
    authorized_cohort_digest_sha256: [u8; 32],
    subsampling_rate_q: f64,
    sampled_peers: BTreeSet<PeerId>,
    sampled_cohort_digest_sha256: [u8; 32],
    sampling_seed_commitment_sha256: [u8; 32],
    sampling_provenance: DpSamplingProvenanceV1,
}

#[cfg(feature = "alloc")]
impl SampledDpCohortV1 {
    /// Sample an independent Poisson($q$) round cohort using host OS CSPRNG (`getrandom`).
    ///
    /// When `subsampling_rate_q == 1.0`, every authorized peer is deterministically included
    /// (`DpSamplingProvenanceV1::DeterministicFullCohort`). Fails closed on empty or duplicate
    /// `authorized_peers`, invalid `q`, or $q \cdot N < 1.0$.
    #[cfg(feature = "std")]
    pub fn sample_poisson_os_rng_v1(
        authorized_peers: &[PeerId],
        subsampling_rate_q: f64,
        run_id_bytes: [u8; 16],
        round_id: u64,
    ) -> Result<Self, CryptoError> {
        if subsampling_rate_q == 1.0 {
            return Self::full_cohort_v1(authorized_peers, run_id_bytes, round_id);
        }
        let mut seed = [0u8; 32];
        getrandom::getrandom(&mut seed).map_err(|_| CryptoError::VerificationFailed)?;
        if seed == [0u8; 32] {
            return Err(CryptoError::AllZeroSeed);
        }
        Self::sample_poisson_inner(
            authorized_peers,
            subsampling_rate_q,
            seed,
            run_id_bytes,
            round_id,
            DpSamplingProvenanceV1::OsCsprng,
        )
    }

    /// Construct a deterministic full-participation ($q = 1.0$) cohort receipt.
    pub fn full_cohort_v1(
        authorized_peers: &[PeerId],
        run_id_bytes: [u8; 16],
        round_id: u64,
    ) -> Result<Self, CryptoError> {
        let unique = Self::validate_unique_authorized(authorized_peers)?;
        let auth_digest = compute_peer_set_digest_v1(b"authorized", &unique);
        let sampled_digest = compute_peer_set_digest_v1(b"sampled", &unique);
        let commitment = sha256_slices(&[
            DP_POISSON_SEED_DOMAIN_V1,
            b"full_cohort_q1",
            &run_id_bytes,
            &round_id.to_le_bytes(),
            &auth_digest,
        ]);
        Ok(Self {
            run_id_bytes,
            round_id,
            authorized_cohort_size: unique.len(),
            authorized_cohort_digest_sha256: auth_digest,
            subsampling_rate_q: 1.0,
            sampled_peers: unique,
            sampled_cohort_digest_sha256: sampled_digest,
            sampling_seed_commitment_sha256: commitment,
            sampling_provenance: DpSamplingProvenanceV1::DeterministicFullCohort,
        })
    }

    /// Deterministic explicit-seed Poisson($q$) sampler for unit tests and `no_std` simulation ONLY.
    ///
    /// Always stamps [`DpSamplingProvenanceV1::ExplicitSeedNonProduction`] and rejects `[0u8; 32]`
    /// with [`CryptoError::AllZeroSeed`].
    pub fn sample_poisson_with_explicit_seed_v1(
        authorized_peers: &[PeerId],
        subsampling_rate_q: f64,
        round_entropy_seed: [u8; 32],
        run_id_bytes: [u8; 16],
        round_id: u64,
    ) -> Result<Self, CryptoError> {
        if round_entropy_seed == [0u8; 32] {
            return Err(CryptoError::AllZeroSeed);
        }
        Self::sample_poisson_inner(
            authorized_peers,
            subsampling_rate_q,
            round_entropy_seed,
            run_id_bytes,
            round_id,
            DpSamplingProvenanceV1::ExplicitSeedNonProduction,
        )
    }

    fn validate_unique_authorized(
        authorized_peers: &[PeerId],
    ) -> Result<BTreeSet<PeerId>, CryptoError> {
        if authorized_peers.is_empty() {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        let mut unique = BTreeSet::new();
        for &peer in authorized_peers {
            if !unique.insert(peer) {
                return Err(CryptoError::InvalidPrivacyParameter);
            }
        }
        Ok(unique)
    }

    fn sample_poisson_inner(
        authorized_peers: &[PeerId],
        subsampling_rate_q: f64,
        round_entropy_seed: [u8; 32],
        run_id_bytes: [u8; 16],
        round_id: u64,
        provenance: DpSamplingProvenanceV1,
    ) -> Result<Self, CryptoError> {
        if !subsampling_rate_q.is_finite() || subsampling_rate_q <= 0.0 || subsampling_rate_q > 1.0
        {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        let unique = Self::validate_unique_authorized(authorized_peers)?;
        let expected_denom = subsampling_rate_q * (unique.len() as f64);
        if !expected_denom.is_finite() || expected_denom < 1.0 {
            return Err(CryptoError::InvalidPrivacyParameter);
        }

        let auth_digest = compute_peer_set_digest_v1(b"authorized", &unique);
        let derived_seed = sha256_slices(&[
            DP_POISSON_SEED_DOMAIN_V1,
            &round_entropy_seed,
            &run_id_bytes,
            &round_id.to_le_bytes(),
            &auth_digest,
        ]);
        let commitment = sha256_slices(&[DP_POISSON_SEED_DOMAIN_V1, b"commit", &derived_seed]);
        let mut drbg = Sha256CounterDrbgV1::new(
            DP_POISSON_DRBG_DOMAIN_V1,
            derived_seed,
            round_id,
            unique.len() as u64,
        );

        let mut sampled_peers = BTreeSet::new();
        for &peer in &unique {
            if sample_bernoulli_f64_exact_v1(subsampling_rate_q, &mut drbg)? {
                sampled_peers.insert(peer);
            }
        }
        let sampled_digest = {
            let mut raw = compute_peer_set_digest_v1(b"sampled", &sampled_peers);
            raw = sha256_slices(&[DP_COHORT_DIGEST_DOMAIN_V1, b"salted", &derived_seed, &raw]);
            raw
        };
        Ok(Self {
            run_id_bytes,
            round_id,
            authorized_cohort_size: unique.len(),
            authorized_cohort_digest_sha256: auth_digest,
            subsampling_rate_q,
            sampled_peers,
            sampled_cohort_digest_sha256: sampled_digest,
            sampling_seed_commitment_sha256: commitment,
            sampling_provenance: provenance,
        })
    }

    /// Run identifier bytes bound to this cohort sample.
    pub const fn run_id_bytes(&self) -> &[u8; 16] {
        &self.run_id_bytes
    }

    /// Round identifier bound to this cohort sample.
    pub const fn round_id(&self) -> u64 {
        self.round_id
    }

    /// Total authorized cohort size $N$ (used for fixed denominator $m = q N$).
    pub const fn authorized_cohort_size(&self) -> usize {
        self.authorized_cohort_size
    }

    /// SHA-256 digest committing to the canonical sorted authorized cohort.
    pub const fn authorized_cohort_digest_sha256(&self) -> &[u8; 32] {
        &self.authorized_cohort_digest_sha256
    }

    /// Configured Poisson subsampling rate $q \in (0, 1]$.
    pub const fn subsampling_rate_q(&self) -> f64 {
        self.subsampling_rate_q
    }

    /// Realized sampled peers $\mathcal{S}_t$ (trusted coordinator internal state — NEVER publish
    /// $|\mathcal{S}_t|$ to external untrusted bundle observers!).
    pub const fn sampled_peers(&self) -> &BTreeSet<PeerId> {
        &self.sampled_peers
    }

    /// Whether `peer` was selected into $\mathcal{S}_t$.
    pub fn contains_peer(&self, peer: &PeerId) -> bool {
        self.sampled_peers.contains(peer)
    }

    /// SHA-256 digest committing to the realized sampled peer set.
    pub const fn sampled_cohort_digest_sha256(&self) -> &[u8; 32] {
        &self.sampled_cohort_digest_sha256
    }

    /// One-way commitment to the sampling entropy seed.
    pub const fn sampling_seed_commitment_sha256(&self) -> &[u8; 32] {
        &self.sampling_seed_commitment_sha256
    }

    /// Entropy provenance of this cohort sample.
    pub const fn sampling_provenance(&self) -> DpSamplingProvenanceV1 {
        self.sampling_provenance
    }
}

/// Sealed proof token minted exclusively by `DpFedAvgMechanismV1::execute_round_v1` (`ST-02`, `Wave 16B`).
///
/// Private fields (and deliberate absence of `Deserialize`) ensure no external caller can forge
/// a proof or bypass exact lattice clipping and Discrete Gaussian noise injection.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedDpMechanismRoundV1 {
    run_id_bytes: [u8; 16],
    round_id: u64,
    parent_digest_bytes: [u8; 32],
    profile: DpMechanismProfileV1,
    subsampling_rate_q: f64,
    authorized_cohort_size: usize,
    fixed_sensitivity_denominator: f64,
    discrete_lattice_radius_b: u64,
    discrete_sigma_z: u64,
    per_coordinate_noise_stddev: f64,
    sampled_cohort_digest_sha256: [u8; 32],
    sanitized_output_digest_sha256: [u8; 32],
    noise_seed_commitment_sha256: [u8; 32],
    sampling_provenance: DpSamplingProvenanceV1,
    noise_provenance: DpEntropyProvenanceV1,
}

impl VerifiedDpMechanismRoundV1 {
    /// Bound run identifier bytes.
    pub const fn run_id_bytes(&self) -> &[u8; 16] {
        &self.run_id_bytes
    }

    /// Bound round identifier.
    pub const fn round_id(&self) -> u64 {
        self.round_id
    }

    /// Bound parent checkpoint SHA-256 digest bytes.
    pub const fn parent_digest_bytes(&self) -> &[u8; 32] {
        &self.parent_digest_bytes
    }

    /// Exact DP mechanism profile enforced during execution.
    pub const fn profile(&self) -> &DpMechanismProfileV1 {
        &self.profile
    }

    /// Enforced Poisson subsampling rate $q \in (0, 1]$.
    pub const fn subsampling_rate_q(&self) -> f64 {
        self.subsampling_rate_q
    }

    /// Total authorized cohort size $N$.
    pub const fn authorized_cohort_size(&self) -> usize {
        self.authorized_cohort_size
    }

    /// Fixed sensitivity denominator $m = q N \ge 1.0$.
    pub const fn fixed_sensitivity_denominator(&self) -> f64 {
        self.fixed_sensitivity_denominator
    }

    /// Integer lattice clipping radius $B$.
    pub const fn discrete_lattice_radius_b(&self) -> u64 {
        self.discrete_lattice_radius_b
    }

    /// Integer Discrete Gaussian scale $\sigma_{\mathbb{Z}} = \lceil \sigma \cdot B \rceil$.
    pub const fn discrete_sigma_z(&self) -> u64 {
        self.discrete_sigma_z
    }

    /// Effective post-division per-coordinate noise standard deviation $(\sigma_{\mathbb{Z}} / B) \cdot \tau / (q N)$.
    pub const fn per_coordinate_noise_stddev(&self) -> f64 {
        self.per_coordinate_noise_stddev
    }

    /// SHA-256 commitment to the realized sampled cohort.
    pub const fn sampled_cohort_digest_sha256(&self) -> &[u8; 32] {
        &self.sampled_cohort_digest_sha256
    }

    /// SHA-256 commitment to the emitted sanitized `f32` mean update vector.
    pub const fn sanitized_output_digest_sha256(&self) -> &[u8; 32] {
        &self.sanitized_output_digest_sha256
    }

    /// One-way SHA-256 commitment to the round noise seed (never exposing the raw seed).
    pub const fn noise_seed_commitment_sha256(&self) -> &[u8; 32] {
        &self.noise_seed_commitment_sha256
    }

    /// Sampling entropy provenance.
    pub const fn sampling_provenance(&self) -> DpSamplingProvenanceV1 {
        self.sampling_provenance
    }

    /// Noise entropy provenance.
    pub const fn noise_provenance(&self) -> DpEntropyProvenanceV1 {
        self.noise_provenance
    }

    /// Whether both sampling and Discrete Gaussian noise used production-certified entropy
    /// (`OsCsprng` for noise, and `OsCsprng` or `DeterministicFullCohort` for sampling).
    pub const fn is_production_certified_entropy(&self) -> bool {
        matches!(self.noise_provenance, DpEntropyProvenanceV1::OsCsprng)
            && matches!(
                self.sampling_provenance,
                DpSamplingProvenanceV1::OsCsprng | DpSamplingProvenanceV1::DeterministicFullCohort
            )
    }
}

/// Inputs to [`DpFedAvgMechanismV1::execute_round_v1`] (`Wave 16B`).
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub struct DpFedAvgRoundInputV1<'a> {
    /// Bound training run identifier bytes.
    pub run_id_bytes: [u8; 16],
    /// Target training round identifier.
    pub round_id: u64,
    /// Parent checkpoint SHA-256 digest bytes.
    pub parent_digest_bytes: [u8; 32],
    /// Model parameter dimension $d \ge 1$ (required even when `client_updates` is empty!).
    pub parameter_dim: usize,
    /// Sealed Poisson cohort sampling receipt for this `(run_id, round_id)`.
    pub sampled_cohort: &'a SampledDpCohortV1,
    /// Sealed single-use 256-bit Discrete Gaussian noise entropy handle (consumed by move).
    pub noise_entropy: DpNoiseEntropyV1,
    /// Admitted updates from peers in `sampled_cohort` (empty slice permitted for empty Poisson draws / dropouts).
    pub client_updates: &'a [GradientUpdate],
}

/// Sanitized output and sealed proof from [`DpFedAvgMechanismV1::execute_round_v1`] (`Wave 16B`).
///
/// **Observer Boundary Notice:** `clipped_count` and `participant_count` are trusted-coordinator-only
/// diagnostics and MUST NOT be published into external run bundle telemetry or untrusted observers.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct DpFedAvgSanitizedOutcomeV1 {
    /// Differentially private mean update vector $\frac{\tau}{B \cdot q N}\left(\sum_{i} \hat{\Delta}_i + Z\right)$.
    pub noisy_mean_update: Vec<f32>,
    /// Coordinator-internal count of updates whose original $\ell_2$ norm exceeded $\tau$.
    pub clipped_count: usize,
    /// Coordinator-internal count of admitted updates summed in this round (`0` on empty draw).
    pub participant_count: usize,
    /// Fixed expected cohort denominator $m = q N \ge 1.0$.
    pub fixed_sensitivity_denominator: f64,
    /// Integer lattice radius $B$.
    pub discrete_lattice_radius_b: u64,
    /// Integer Discrete Gaussian scale $\sigma_{\mathbb{Z}} = \lceil \sigma \cdot B \rceil$.
    pub discrete_sigma_z: u64,
    /// Effective post-division per-coordinate noise standard deviation.
    pub per_coordinate_noise_stddev: f64,
    /// Sealed proof required to advance [`RdpAccountantV1`] and debit `dp/ledger.ndjson`.
    pub proof: VerifiedDpMechanismRoundV1,
}

/// Certified exact Discrete-Gaussian `DpFedAvgV1` mechanism (`ST-02`, `Wave 16B`).
#[cfg(feature = "alloc")]
pub struct DpFedAvgMechanismV1;

#[cfg(feature = "alloc")]
impl DpFedAvgMechanismV1 {
    /// Execute one client-level `DpFedAvgV1` round:
    /// 1. Validate `profile` (`VerifiedMechanismProtected` + `ClientLevel`), `sampled_cohort`
    ///    binding (`run_id`, `round_id`, $q N \ge 1.0$), and unique sampled-peer membership.
    /// 2. Clip and quantize every client update onto the exact integer lattice
    ///    $\|\hat{\Delta}_i\|_2^2 \le B^2$ (`u128`-verified) and sum associatively in `i128`.
    /// 3. Sample exact floating-point-free Discrete Gaussian noise
    ///    $Z_j \sim \mathcal{N}_{\mathbb{Z}}(0, \sigma_{\mathbb{Z}}^2)$ (`Canonne et al. 2020`)
    ///    with $\sigma_{\mathbb{Z}} = \lceil \sigma \cdot B \rceil$, adding to the exact `i128` sum.
    /// 4. Post-process $\tilde{S}_j \in \mathbb{Z}$ to `f32` via fixed expected denominator $m = q N$
    ///    and mint [`VerifiedDpMechanismRoundV1`].
    pub fn execute_round_v1(
        profile: DpMechanismProfileV1,
        input: DpFedAvgRoundInputV1<'_>,
    ) -> Result<DpFedAvgSanitizedOutcomeV1, CryptoError> {
        profile.validate()?;
        if profile.classification != PrivacyAccountingClassificationV1::VerifiedMechanismProtected
            || profile.unit != DpUnitV1::ClientLevel
            || input.parameter_dim == 0
        {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        if input.sampled_cohort.run_id_bytes != input.run_id_bytes
            || input.sampled_cohort.round_id != input.round_id
        {
            return Err(CryptoError::InvalidPrivacyParameter);
        }
        if input.noise_entropy.seed == [0u8; 32] {
            return Err(CryptoError::AllZeroSeed);
        }

        let q = input.sampled_cohort.subsampling_rate_q;
        let cohort_size = input.sampled_cohort.authorized_cohort_size;
        let fixed_denom = q * (cohort_size as f64);
        if !q.is_finite()
            || q <= 0.0
            || q > 1.0
            || cohort_size == 0
            || !fixed_denom.is_finite()
            || fixed_denom < 1.0
            || input.client_updates.len() > input.sampled_cohort.sampled_peers.len()
        {
            return Err(CryptoError::InvalidPrivacyParameter);
        }

        let sigma_z = compute_discrete_sigma_z_v1(profile.noise_multiplier)?;
        let mut seen_senders = BTreeSet::new();
        let mut exact_sum_i128 = vec![0i128; input.parameter_dim];
        let mut clipped_count = 0usize;

        for upd in input.client_updates {
            if upd.round_id != input.round_id || upd.gradients.len() != input.parameter_dim {
                return Err(CryptoError::InvalidPrivacyParameter);
            }
            let peer = PeerId::new(upd.sender);
            if !input.sampled_cohort.sampled_peers.contains(&peer) {
                return Err(CryptoError::InvalidPrivacyParameter);
            }
            if !seen_senders.insert(peer) {
                return Err(CryptoError::InvalidPrivacyParameter);
            }
            let (quantized, _orig_norm, was_clipped) =
                clip_and_quantize_l2_lattice_v1(&upd.gradients, profile.clipping_radius_tau)?;
            if was_clipped {
                clipped_count += 1;
            }
            for (acc, &q_j) in exact_sum_i128.iter_mut().zip(quantized.iter()) {
                *acc = acc
                    .checked_add(q_j as i128)
                    .ok_or(CryptoError::InvalidPrivacyParameter)?;
            }
        }

        // Derive context-bound 256-bit DRBG seed from noise entropy.
        let derived_noise_seed = sha256_slices(&[
            DP_GAUSSIAN_SEED_DOMAIN_V1,
            &input.noise_entropy.seed,
            &input.run_id_bytes,
            &input.round_id.to_le_bytes(),
            &input.parent_digest_bytes,
            &input.sampled_cohort.sampled_cohort_digest_sha256,
        ]);
        let noise_commitment = sha256_slices(&[
            DP_NOISE_COMMITMENT_DOMAIN_V1,
            &input.run_id_bytes,
            &input.round_id.to_le_bytes(),
            &input.parent_digest_bytes,
            &derived_noise_seed,
        ]);
        let mut drbg = Sha256CounterDrbgV1::new(
            DP_GAUSSIAN_DRBG_DOMAIN_V1,
            derived_noise_seed,
            input.round_id,
            input.parameter_dim as u64,
        );

        // Exact integer perturbation + deterministic f32 post-processing (`Dwork & Roth 2014` Prop. 2.1).
        let tau_f64 = profile.clipping_radius_tau as f64;
        let post_scale = tau_f64 / ((DP_DISCRETE_LATTICE_RADIUS_V1 as f64) * fixed_denom);
        if !post_scale.is_finite() || post_scale <= 0.0 {
            return Err(CryptoError::InvalidPrivacyParameter);
        }

        let mut noisy_mean_update = Vec::with_capacity(input.parameter_dim);
        let mut out_hasher = {
            use sha2::Digest;
            let mut h = sha2::Sha256::new();
            h.update(DP_OUTPUT_DIGEST_DOMAIN_V1);
            h.update(input.run_id_bytes);
            h.update(input.round_id.to_le_bytes());
            h.update((input.parameter_dim as u64).to_le_bytes());
            h
        };

        for &sum_j in &exact_sum_i128 {
            let z_j = sample_dgauss_u128(sigma_z, &mut drbg)?;
            let noisy_int = sum_j
                .checked_add(z_j)
                .ok_or(CryptoError::InvalidPrivacyParameter)?;
            // Total deterministic post-processing (clamping to finite f32 range eliminates
            // data-dependent overflow aborts and canonicalizes `-0.0` to `+0.0`).
            let raw_f32 =
                ((noisy_int as f64) * post_scale).clamp(-(f32::MAX as f64), f32::MAX as f64) as f32;
            let val_f32 = if raw_f32 == 0.0 { 0.0 } else { raw_f32 };
            {
                use sha2::Digest;
                out_hasher.update(val_f32.to_bits().to_le_bytes());
            }
            noisy_mean_update.push(val_f32);
        }

        let sanitized_output_digest_sha256: [u8; 32] = {
            use sha2::Digest;
            out_hasher.finalize().into()
        };
        let per_coordinate_noise_stddev =
            ((sigma_z as f64) / (DP_DISCRETE_LATTICE_RADIUS_V1 as f64)) * tau_f64 / fixed_denom;

        let proof = VerifiedDpMechanismRoundV1 {
            run_id_bytes: input.run_id_bytes,
            round_id: input.round_id,
            parent_digest_bytes: input.parent_digest_bytes,
            profile,
            subsampling_rate_q: q,
            authorized_cohort_size: cohort_size,
            fixed_sensitivity_denominator: fixed_denom,
            discrete_lattice_radius_b: DP_DISCRETE_LATTICE_RADIUS_V1,
            discrete_sigma_z: sigma_z,
            per_coordinate_noise_stddev,
            sampled_cohort_digest_sha256: input.sampled_cohort.sampled_cohort_digest_sha256,
            sanitized_output_digest_sha256,
            noise_seed_commitment_sha256: noise_commitment,
            sampling_provenance: input.sampled_cohort.sampling_provenance,
            noise_provenance: input.noise_entropy.provenance,
        };

        Ok(DpFedAvgSanitizedOutcomeV1 {
            noisy_mean_update,
            clipped_count,
            participant_count: input.client_updates.len(),
            fixed_sensitivity_denominator: fixed_denom,
            discrete_lattice_radius_b: DP_DISCRETE_LATTICE_RADIUS_V1,
            discrete_sigma_z: sigma_z,
            per_coordinate_noise_stddev,
            proof,
        })
    }
}

#[cfg(test)]
#[path = "dp_tests.rs"]
mod tests;
