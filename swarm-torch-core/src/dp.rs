//! Rényi Differential Privacy (`RDP`) accounting and Poisson-Subsampled Gaussian Mechanism (`SGM`)
//! composition (`ADR-0023` §4.2, `W12-03`).
//!
//! This module is `no_std` compatible and implements [`RdpAccountantV1`] over the canonical
//! 16-order Rényi grid $\Lambda$, supporting:
//! - Exact analytical Gaussian Rényi divergence ($q = 1.0$),
//! - Log-domain binomial moment expansion for the Poisson-Subsampled Gaussian Mechanism
//!   (`Mironov, Talwar, & Zhang 2019`, $0 < q < 1$),
//! - Both `Mironov (2017)` standard and `Balle et al. (2020)` tight $(\alpha, \rho(\alpha)) \to (\varepsilon, \delta)$-DP
//!   conversions, and
//! - Fail-closed budget gating via [`PrivacyAccountantHook`].

use crate::crypto::{CryptoError, PrivacyAccountantHook, RdpBudgetDescriptorV1};
use crate::num::{ln_f64_soft, log_sum_exp_pair_f64};

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

/// Normative configuration profile for a Differential Privacy mechanism (`ADR-0023` §4.2).
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
}

impl DpMechanismProfileV1 {
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
    /// Construct a new [`RdpAccountantV1`] after validating `profile`.
    pub fn new(profile: DpMechanismProfileV1) -> Result<Self, CryptoError> {
        profile.validate()?;
        Ok(Self {
            profile,
            cumulative_rdp_orders: [0.0; RDP_GRID_LEN],
            committed_rounds: 0,
            last_optimal_alpha: RDP_ALPHA_GRID_V1[0],
            last_converted_epsilon: 0.0,
        })
    }

    /// Reconstruct an [`RdpAccountantV1`] from persisted durable ledger state (`dp/ledger.ndjson`).
    ///
    /// Verifies that `profile` and every persisted `cumulative_rdp_orders[i]` are finite and $\ge 0$,
    /// recomputes $(\varepsilon(\delta), \alpha^*)$, and fails closed with [`CryptoError::PrivacyBudgetExhausted`]
    /// if the replayed cumulative spend already exceeds `profile.max_epsilon`.
    pub fn restore_from_orders(
        profile: DpMechanismProfileV1,
        cumulative_rdp_orders: [f64; RDP_GRID_LEN],
        committed_rounds: u64,
    ) -> Result<Self, CryptoError> {
        profile.validate()?;
        for &rho in &cumulative_rdp_orders {
            if !rho.is_finite() || rho < 0.0 {
                return Err(CryptoError::InvalidPrivacyParameter);
            }
        }
        let (converted_eps, opt_alpha) =
            if committed_rounds == 0 && cumulative_rdp_orders.iter().all(|&r| r == 0.0) {
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

    /// Return the configured [`DpMechanismProfileV1`].
    pub const fn profile(&self) -> &DpMechanismProfileV1 {
        &self.profile
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
            .position(|&a| (a - opt_alpha).abs() < 1e-12)
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
}

impl PrivacyAccountantHook for RdpAccountantV1 {
    fn current_budget(&self) -> RdpBudgetDescriptorV1 {
        let opt_idx = RDP_ALPHA_GRID_V1
            .iter()
            .position(|&a| (a - self.last_optimal_alpha).abs() < 1e-12)
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
        let (desc, next_orders, converted_eps) =
            self.preview_clipped_round(tau, subsampling_rate_q)?;
        self.cumulative_rdp_orders = next_orders;
        self.committed_rounds = self.committed_rounds.saturating_add(1);
        self.last_optimal_alpha = desc.alpha;
        self.last_converted_epsilon = converted_eps;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_gaussian_rdp_and_composition_match_analytical_closed_form() {
        let profile = DpMechanismProfileV1 {
            unit: DpUnitV1::ClientLevel,
            adjacency: DpAdjacencyV1::AddRemoveOne,
            clipping_radius_tau: 1.0,
            noise_multiplier: 2.0, // sigma = 2.0 => rho(alpha) = alpha / 8
            target_delta: 1e-5,
            max_epsilon: 6.0,
            conversion_formula: DpConversionFormulaV1::Mironov2017,
        };
        let mut acc = RdpAccountantV1::new(profile).unwrap();

        // Record 4 full-participation rounds (q = 1.0): cumulative rho(alpha) = 4 * alpha / 8 = alpha / 2.
        for _ in 0..4 {
            acc.record_clipped_round(1.0, 1.0).unwrap();
        }
        assert_eq!(acc.committed_rounds(), 4);
        let orders = acc.cumulative_rdp_orders();
        for (i, &alpha) in RDP_ALPHA_GRID_V1.iter().enumerate() {
            assert!((orders[i] - 0.5 * alpha).abs() < 1e-12);
        }

        // At alpha = 6.0: rho(6) = 3.0, ln(1e5)/5 = 11.512925465 / 5 = 2.302585093 => eps = 5.302585093
        let (eps, delta, opt_alpha) = acc.current_approx_dp();
        assert_eq!(delta, 1e-5);
        assert_eq!(opt_alpha, 6.0);
        assert!((eps - (3.0 + 11.512_925_464_970_229 / 5.0)).abs() < 1e-9);
    }

    #[test]
    fn subsampled_gaussian_amplification_and_budget_exhaustion_gate() {
        let profile = DpMechanismProfileV1 {
            unit: DpUnitV1::ClientLevel,
            adjacency: DpAdjacencyV1::AddRemoveOne,
            clipping_radius_tau: 1.0,
            noise_multiplier: 1.5,
            target_delta: 1e-5,
            max_epsilon: 2.5,
            conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        };
        let mut acc = RdpAccountantV1::new(profile).unwrap();

        // Subsampling at q = 0.05 yields strictly smaller RDP than q = 1.0
        let sgm_sched = compute_round_rdp_schedule_v1(0.05, 1.5).unwrap();
        let full_sched = compute_round_rdp_schedule_v1(1.0, 1.5).unwrap();
        for i in 0..RDP_GRID_LEN {
            assert!(
                sgm_sched[i] < full_sched[i],
                "order {}: sgm {} >= full {}",
                RDP_ALPHA_GRID_V1[i],
                sgm_sched[i],
                full_sched[i]
            );
        }

        // Passing a clipping radius larger than calibrated profile.clipping_radius_tau fails closed
        assert_eq!(
            acc.record_clipped_round(1.5, 0.05).unwrap_err(),
            CryptoError::InvalidPrivacyParameter
        );

        // Repeated subsampled rounds succeed under amplification, then eventually hit PrivacyBudgetExhausted
        acc.record_clipped_round(1.0, 0.05).unwrap();
        let rounds_before = acc.committed_rounds();
        assert_eq!(rounds_before, 1);
        let mut hit_exhausted = false;
        for _ in 0..40 {
            if let Err(err) = acc.record_clipped_round(1.0, 0.25) {
                assert_eq!(err, CryptoError::PrivacyBudgetExhausted);
                hit_exhausted = true;
                break;
            }
        }
        assert!(hit_exhausted);
        assert!(acc.committed_rounds() >= rounds_before);
        let (eps, _delta, _alpha) = acc.current_approx_dp();
        assert!(eps <= profile.max_epsilon);
    }
}
