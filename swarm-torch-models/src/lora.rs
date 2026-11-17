//! Low-Rank Adaptation (`LoRA`) and Heterogeneous `FLoRA` Factor Aggregation (`ADR-0024` §4, `W12-01`).
//!
//! Resolves the bilinear factor-averaging cross-term bias
//! ($\bar{B}\bar{A} \ne \frac{1}{n}\sum_i B_i A_i$) via three explicit, auditable modes:
//! 1. [`LoraAggregationModeV1::FreezeASingleFactorV1`]: Freezes $A_0$ across the cohort (verified via
//!    canonical SHA-256 commitment `swarmtorch.lora.a.v1`) and aggregates $B_i \in \mathbb{R}^{d_{\text{out}} \times r}$
//!    using any [`RobustAggregator`] (`FedAvg`, `CoordinateMedian`, `TrimmedMean`, `Krum`, `MultiKrum`,
//!    `Bulyan`, or `CenteredClipping`) with **zero cross-term error** in $\mathcal{O}(d_{\text{out}} r)$ space.
//! 2. [`LoraAggregationModeV1::FloraStackingV1`]: Implements **FLoRA** (`Wang et al., NeurIPS 2024`)
//!    block-diagonal factor stacking ($\tilde{B} \in \mathbb{R}^{d_{\text{out}} \times R}$,
//!    $\tilde{A} \in \mathbb{R}^{R \times d_{\text{in}}}$, $R = \sum_i r_i$) supporting **heterogeneous**
//!    client ranks $r_i$ and exact base-weight merge $W \leftarrow W + \tilde{B}\tilde{A}$ with **zero
//!    cross-term noise**.
//! 3. [`LoraAggregationModeV1::BiasedFactorFedAvgV1`]: Explicit opt-in homogeneous-rank averaging of
//!    $\bar{B}$ and $\bar{A}$ that computes the exact Frobenius cross-term drift residual
//!    $\|\frac{\alpha}{r}\bar{B}\bar{A} - \frac{1}{n}\sum_i \frac{\alpha_i}{r_i} B_i A_i\|_F$ for audit reporting.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use swarm_torch_core::aggregation::RobustAggregator;
use swarm_torch_core::crypto::sha256_bytes;
use swarm_torch_core::num::{abs_f64, sqrt_f64_soft};
use swarm_torch_core::traits::{GradientUpdate, PeerId};

use crate::safetensors_v1::LoRaAdapterDescriptorV1;

/// Domain separation tag for canonical frozen-$A$ factor commitments (`ADR-0024` §4).
pub const LORA_FROZEN_A_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.lora.a.v1";

/// Maximum default stacked rank $R = \sum_i r_i$ for `FloraStackingV1`.
pub const DEFAULT_MAX_FLORA_STACKED_RANK: u32 = 4_096;

/// Relative tolerance for comparing LoRA scaling factors `alpha` across homogeneous-scale
/// aggregation modes (`FreezeASingleFactorV1` and `BiasedFactorFedAvgV1`).
pub const DEFAULT_ALPHA_MISMATCH_TOLERANCE: f32 = 1e-6;

/// Errors returned during LoRA / FLoRA adapter validation, aggregation, or base-weight merging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraError {
    /// Rank $r == 0$, matrix dimension $d_{\text{in}} == 0$ or $d_{\text{out}} == 0$, or slice length mismatch.
    InvalidRankOrDimension,
    /// Scaling factor $\alpha$ is non-finite or $\le 0$.
    InvalidAlpha,
    /// Adapter or base weight matrix contains `NaN` or `±Inf`.
    NonFiniteWeight,
    /// Empty peer cohort or mismatched `target_module` / `(d_in, d_out)` across submissions.
    CohortMismatch,
    /// Peer's $A_i$ factor digest does not match `expected_a_digest` under `FreezeASingleFactorV1`.
    FrozenADigestMismatch,
    /// Heterogeneous ranks $r_i \ne r_j$ submitted to a homogeneous-rank aggregation mode.
    HeterogeneousRankUnsupportedInMode,
    /// Total stacked rank $R = \sum_i r_i$ exceeds `max_stacked_rank`.
    StackedRankExceedsLimit,
    /// Same [`PeerId`] submitted more than one adapter update in a single round cohort.
    DuplicatePeerSubmission(PeerId),
    /// Underlying [`RobustAggregator`] returned an error.
    AggregationFailed(swarm_torch_core::Error),
}

impl core::fmt::Display for LoraError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidRankOrDimension => {
                write!(f, "invalid LoRA rank or matrix dimension mismatch")
            }
            Self::InvalidAlpha => write!(f, "LoRA scaling factor alpha must be finite and > 0"),
            Self::NonFiniteWeight => write!(f, "non-finite (NaN or Inf) weight in LoRA adapter"),
            Self::CohortMismatch => {
                write!(
                    f,
                    "empty LoRA cohort or mismatched target_module / matrix dimensions"
                )
            }
            Self::FrozenADigestMismatch => {
                write!(
                    f,
                    "peer LoRA A factor digest does not match expected frozen A digest"
                )
            }
            Self::HeterogeneousRankUnsupportedInMode => {
                write!(
                    f,
                    "heterogeneous LoRA ranks require FloraStackingV1 aggregation mode"
                )
            }
            Self::StackedRankExceedsLimit => {
                write!(f, "FLoRA total stacked rank exceeds max_stacked_rank")
            }
            Self::DuplicatePeerSubmission(peer) => {
                write!(f, "duplicate LoRA submission from peer {peer:?}")
            }
            Self::AggregationFailed(err) => write!(f, "LoRA robust aggregation failed: {err}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LoraError {}

/// Validated low-rank adapter matrix pair ($A \in \mathbb{R}^{r \times d_{\text{in}}}$,
/// $B \in \mathbb{R}^{d_{\text{out}} \times r}$) for a target linear layer $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
/// (`ADR-0024` §4, `W12-01`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LoraAdapterPairV1 {
    /// Adapter metadata (`target_module`, `tensor_a_name`, `tensor_b_name`, `rank`, `alpha`).
    pub descriptor: LoRaAdapterDescriptorV1,
    /// Input dimension $d_{\text{in}} \ge 1$.
    pub d_in: usize,
    /// Output dimension $d_{\text{out}} \ge 1$.
    pub d_out: usize,
    /// Row-major down-projection matrix $A \in \mathbb{R}^{r \times d_{\text{in}}}$.
    pub a_weights: Vec<f32>,
    /// Row-major up-projection matrix $B \in \mathbb{R}^{d_{\text{out}} \times r}$.
    pub b_weights: Vec<f32>,
}

impl LoraAdapterPairV1 {
    /// Construct and validate a [`LoraAdapterPairV1`].
    pub fn try_new(
        descriptor: LoRaAdapterDescriptorV1,
        d_in: usize,
        d_out: usize,
        a_weights: Vec<f32>,
        b_weights: Vec<f32>,
    ) -> Result<Self, LoraError> {
        let pair = Self {
            descriptor,
            d_in,
            d_out,
            a_weights,
            b_weights,
        };
        pair.validate()?;
        Ok(pair)
    }

    /// Validate matrix dimensions, buffer lengths, scaling factor `alpha`, and finiteness.
    pub fn validate(&self) -> Result<(), LoraError> {
        if self.descriptor.target_module.is_empty()
            || self.descriptor.rank == 0
            || self.d_in == 0
            || self.d_out == 0
        {
            return Err(LoraError::InvalidRankOrDimension);
        }
        if !self.descriptor.alpha.is_finite() || self.descriptor.alpha <= 0.0 {
            return Err(LoraError::InvalidAlpha);
        }
        let r =
            usize::try_from(self.descriptor.rank).map_err(|_| LoraError::InvalidRankOrDimension)?;
        let expected_a = r
            .checked_mul(self.d_in)
            .ok_or(LoraError::InvalidRankOrDimension)?;
        let expected_b = self
            .d_out
            .checked_mul(r)
            .ok_or(LoraError::InvalidRankOrDimension)?;
        if self.a_weights.len() != expected_a || self.b_weights.len() != expected_b {
            return Err(LoraError::InvalidRankOrDimension);
        }
        if self.a_weights.iter().any(|v| !v.is_finite())
            || self.b_weights.iter().any(|v| !v.is_finite())
        {
            return Err(LoraError::NonFiniteWeight);
        }
        Ok(())
    }

    /// Compute the canonical SHA-256 commitment over the down-projection factor $A$
    /// (`swarmtorch.lora.a.v1`), normalizing `-0.0` to `+0.0`.
    pub fn frozen_a_digest(&self) -> [u8; 32] {
        let mod_bytes = self.descriptor.target_module.as_bytes();
        let mut buf = Vec::with_capacity(
            LORA_FROZEN_A_DOMAIN_TAG_V1.len() + 16 + mod_bytes.len() + self.a_weights.len() * 4,
        );
        buf.extend_from_slice(LORA_FROZEN_A_DOMAIN_TAG_V1);
        buf.extend_from_slice(&(mod_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(mod_bytes);
        buf.extend_from_slice(&self.descriptor.rank.to_le_bytes());
        buf.extend_from_slice(&(self.d_in as u64).to_le_bytes());
        for &val in &self.a_weights {
            let norm = if val == 0.0 { 0.0f32 } else { val };
            buf.extend_from_slice(&norm.to_le_bytes());
        }
        sha256_bytes(&buf)
    }

    /// Materialize the dense scaled update $\Delta W = \frac{\alpha}{r} B A \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
    /// in widened `f64` arithmetic.
    pub fn materialize_scaled_delta(&self) -> Result<Vec<f32>, LoraError> {
        self.validate()?;
        let total = self
            .d_out
            .checked_mul(self.d_in)
            .ok_or(LoraError::InvalidRankOrDimension)?;
        let mut delta = vec![0.0f32; total];
        self.apply_to_base_weights(&mut delta)?;
        Ok(delta)
    }

    /// Transactionally accumulate $W \leftarrow W + \frac{\alpha}{r} B A$ into `base_w`
    /// ($\mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ row-major) using widened `f64` dot products.
    pub fn apply_to_base_weights(&self, base_w: &mut [f32]) -> Result<(), LoraError> {
        self.validate()?;
        let r =
            usize::try_from(self.descriptor.rank).map_err(|_| LoraError::InvalidRankOrDimension)?;
        let scale = (self.descriptor.alpha as f64) / (r as f64);
        apply_scaled_lora_product_to_base_weights(
            self.d_out,
            self.d_in,
            r,
            scale,
            &self.b_weights,
            &self.a_weights,
            base_w,
        )
    }
}

/// Shared widened `f64` transactional base-weight merge helper for both single-adapter and stacked
/// FLoRA products: validates dimension/buffer invariants and stages all outputs before mutating
/// `base_w`.
fn apply_scaled_lora_product_to_base_weights(
    d_out: usize,
    d_in: usize,
    rank: usize,
    scale: f64,
    b_weights: &[f32],
    a_weights: &[f32],
    base_w: &mut [f32],
) -> Result<(), LoraError> {
    if d_out == 0 || d_in == 0 || rank == 0 {
        return Err(LoraError::InvalidRankOrDimension);
    }
    let expected_len = d_out
        .checked_mul(d_in)
        .ok_or(LoraError::InvalidRankOrDimension)?;
    let expected_b = d_out
        .checked_mul(rank)
        .ok_or(LoraError::InvalidRankOrDimension)?;
    let expected_a = rank
        .checked_mul(d_in)
        .ok_or(LoraError::InvalidRankOrDimension)?;
    if base_w.len() != expected_len
        || b_weights.len() != expected_b
        || a_weights.len() != expected_a
    {
        return Err(LoraError::InvalidRankOrDimension);
    }
    if base_w.iter().any(|v| !v.is_finite())
        || b_weights.iter().any(|v| !v.is_finite())
        || a_weights.iter().any(|v| !v.is_finite())
    {
        return Err(LoraError::NonFiniteWeight);
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(LoraError::InvalidAlpha);
    }

    let mut staged = Vec::with_capacity(expected_len);
    for row in 0..d_out {
        let b_row_offset = row * rank;
        let w_row_offset = row * d_in;
        for col in 0..d_in {
            let mut dot = 0.0f64;
            for k in 0..rank {
                let b_val = b_weights[b_row_offset + k] as f64;
                let a_val = a_weights[k * d_in + col] as f64;
                dot += b_val * a_val;
            }
            let updated = (base_w[w_row_offset + col] as f64) + scale * dot;
            let updated_f32 = updated as f32;
            if !updated_f32.is_finite() {
                return Err(LoraError::NonFiniteWeight);
            }
            staged.push(updated_f32);
        }
    }

    base_w.copy_from_slice(&staged);
    Ok(())
}

/// Block-diagonally stacked FLoRA factor representation (`Wang et al., NeurIPS 2024`, `ADR-0024` §4)
/// supporting heterogeneous client ranks $r_i$ with **zero cross-term error**:
///
/// $$\tilde{B} = \left[w_1 \frac{\alpha_1}{r_1} B_1, \dots, w_n \frac{\alpha_n}{r_n} B_n\right] \in \mathbb{R}^{d_{\text{out}} \times R}, \qquad \tilde{A} = \begin{bmatrix} A_1 \\ \vdots \\ A_n \end{bmatrix} \in \mathbb{R}^{R \times d_{\text{in}}}$$
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FloraStackedDeltaV1 {
    /// Target module identifier (e.g. `"attn.q_proj"`).
    pub target_module: String,
    /// Input dimension $d_{\text{in}}$.
    pub d_in: usize,
    /// Output dimension $d_{\text{out}}$.
    pub d_out: usize,
    /// Total stacked rank $R = \sum_{i=1}^n r_i$.
    pub total_stacked_rank: usize,
    /// Individual client ranks $r_1, \dots, r_n$.
    pub client_ranks: Vec<u32>,
    /// Row-major stacked up-projection matrix $\tilde{B} \in \mathbb{R}^{d_{\text{out}} \times R}$.
    pub stacked_b: Vec<f32>,
    /// Row-major stacked down-projection matrix $\tilde{A} \in \mathbb{R}^{R \times d_{\text{in}}}$.
    pub stacked_a: Vec<f32>,
}

impl FloraStackedDeltaV1 {
    /// Validate matrix dimensions, positive `client_ranks` sum matching `total_stacked_rank`,
    /// exact buffer lengths, and element finiteness before indexing or allocating.
    pub fn validate(&self) -> Result<(), LoraError> {
        if self.target_module.is_empty()
            || self.d_in == 0
            || self.d_out == 0
            || self.total_stacked_rank == 0
            || self.client_ranks.is_empty()
        {
            return Err(LoraError::InvalidRankOrDimension);
        }
        let mut rank_sum = 0usize;
        for &r in &self.client_ranks {
            if r == 0 {
                return Err(LoraError::InvalidRankOrDimension);
            }
            let r_usize = usize::try_from(r).map_err(|_| LoraError::InvalidRankOrDimension)?;
            rank_sum = rank_sum
                .checked_add(r_usize)
                .ok_or(LoraError::InvalidRankOrDimension)?;
        }
        if rank_sum != self.total_stacked_rank {
            return Err(LoraError::InvalidRankOrDimension);
        }
        let expected_b = self
            .d_out
            .checked_mul(self.total_stacked_rank)
            .ok_or(LoraError::InvalidRankOrDimension)?;
        let expected_a = self
            .total_stacked_rank
            .checked_mul(self.d_in)
            .ok_or(LoraError::InvalidRankOrDimension)?;
        if self.stacked_b.len() != expected_b || self.stacked_a.len() != expected_a {
            return Err(LoraError::InvalidRankOrDimension);
        }
        if self.stacked_b.iter().any(|v| !v.is_finite())
            || self.stacked_a.iter().any(|v| !v.is_finite())
        {
            return Err(LoraError::NonFiniteWeight);
        }
        Ok(())
    }

    /// Materialize the exact dense update $\Delta W = \tilde{B}\tilde{A} = \sum_i w_i \frac{\alpha_i}{r_i} B_i A_i$.
    pub fn materialize_exact_delta(&self) -> Result<Vec<f32>, LoraError> {
        self.validate()?;
        let total = self
            .d_out
            .checked_mul(self.d_in)
            .ok_or(LoraError::InvalidRankOrDimension)?;
        let mut delta = vec![0.0f32; total];
        self.apply_exact_to_base_weights(&mut delta)?;
        Ok(delta)
    }

    /// Transactionally merge $W \leftarrow W + \tilde{B}\tilde{A}$ into `base_w` in widened `f64`.
    pub fn apply_exact_to_base_weights(&self, base_w: &mut [f32]) -> Result<(), LoraError> {
        self.validate()?;
        apply_scaled_lora_product_to_base_weights(
            self.d_out,
            self.d_in,
            self.total_stacked_rank,
            1.0,
            &self.stacked_b,
            &self.stacked_a,
            base_w,
        )
    }
}

/// Normative LoRA aggregation mode (`ADR-0024` §4, `W12-01`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LoraAggregationModeV1 {
    /// Freezes $A_0$ (verified against `expected_a_digest`) and aggregates only $B_i$ via the
    /// caller-supplied [`RobustAggregator`]. Exact linear aggregation ($\text{cross-term} = 0$).
    FreezeASingleFactorV1 {
        /// Expected SHA-256 digest from [`LoraAdapterPairV1::frozen_a_digest`].
        expected_a_digest: [u8; 32],
    },
    /// Exact block-diagonal FLoRA stacking (`Wang et al., NeurIPS 2024`) supporting both
    /// homogeneous and heterogeneous client ranks $r_i$ with zero cross-term noise.
    FloraStackingV1 {
        /// Maximum permitted sum of ranks $R = \sum_i r_i$.
        max_stacked_rank: u32,
    },
    /// Explicit opt-in separate factor averaging ($\bar{B} = \frac{1}{n}\sum_i B_i$,
    /// $\bar{A} = \frac{1}{n}\sum_i A_i$) with audited Frobenius cross-term drift bound.
    BiasedFactorFedAvgV1,
}

/// Output of [`aggregate_lora_adapters_v1`].
#[derive(Debug, Clone, PartialEq)]
pub enum LoraAggregationOutcomeV1 {
    /// Exact single-factor (`FreezeASingleFactorV1`) aggregated adapter pair.
    SingleFactorAggregated {
        /// Resulting adapter with frozen $A_0$ and robustly aggregated $B$.
        adapter: LoraAdapterPairV1,
    },
    /// Exact block-diagonal FLoRA stacked delta (`FloraStackingV1`).
    FloraStacked {
        /// Stacked $\tilde{B}$ and $\tilde{A}$ matrices ready for exact base-weight merge.
        stacked: FloraStackedDeltaV1,
    },
    /// Biased separate-factor average (`BiasedFactorFedAvgV1`) with audited Frobenius cross-term residual.
    BiasedFactorFedAvg {
        /// Averaged $(\bar{A}, \bar{B})$ adapter pair.
        adapter: LoraAdapterPairV1,
        /// Exact Frobenius norm $\|\frac{\alpha}{r}\bar{B}\bar{A} - \frac{1}{n}\sum_i \frac{\alpha_i}{r_i} B_i A_i\|_F$.
        cross_term_frobenius_drift_bound: f64,
    },
}

fn alpha_within_tolerance(lhs: f32, rhs: f32, tolerance: f32) -> bool {
    if !lhs.is_finite() || !rhs.is_finite() || !tolerance.is_finite() || tolerance < 0.0 {
        return false;
    }
    let l = lhs as f64;
    let r = rhs as f64;
    let scale = abs_f64(l).max(abs_f64(r));
    if scale == 0.0 {
        return true;
    }
    abs_f64(l - r) <= (tolerance as f64) * scale
}

/// Verify uniform rank and order-invariant cohort-wide relative `alpha` agreement
/// (`max_alpha - min_alpha <= DEFAULT_ALPHA_MISMATCH_TOLERANCE * max(|min_alpha|, |max_alpha|)`).
fn validate_homogeneous_rank_and_alpha(
    submissions: &[(PeerId, LoraAdapterPairV1)],
) -> Result<(), LoraError> {
    let r0 = submissions[0].1.descriptor.rank;
    let mut min_alpha = submissions[0].1.descriptor.alpha;
    let mut max_alpha = min_alpha;
    for (_, pair) in submissions {
        if pair.descriptor.rank != r0 {
            return Err(LoraError::HeterogeneousRankUnsupportedInMode);
        }
        let alpha = pair.descriptor.alpha;
        if alpha < min_alpha {
            min_alpha = alpha;
        }
        if alpha > max_alpha {
            max_alpha = alpha;
        }
    }
    if !alpha_within_tolerance(min_alpha, max_alpha, DEFAULT_ALPHA_MISMATCH_TOLERANCE) {
        return Err(LoraError::CohortMismatch);
    }
    Ok(())
}

/// Aggregate a cohort of peer [`LoraAdapterPairV1`] submissions under `mode` (`ADR-0024` §4, `W12-01`).
pub fn aggregate_lora_adapters_v1<A: RobustAggregator + ?Sized>(
    submissions: &[(PeerId, LoraAdapterPairV1)],
    mode: LoraAggregationModeV1,
    b_factor_aggregator: &A,
) -> Result<LoraAggregationOutcomeV1, LoraError> {
    if submissions.is_empty() {
        return Err(LoraError::CohortMismatch);
    }

    let first = &submissions[0].1;
    let target_module = &first.descriptor.target_module;
    let d_in = first.d_in;
    let d_out = first.d_out;
    let mut seen_peers = BTreeSet::new();

    for (peer, pair) in submissions {
        if !seen_peers.insert(*peer) {
            return Err(LoraError::DuplicatePeerSubmission(*peer));
        }
        pair.validate()?;
        if &pair.descriptor.target_module != target_module
            || pair.d_in != d_in
            || pair.d_out != d_out
        {
            return Err(LoraError::CohortMismatch);
        }
    }

    match mode {
        LoraAggregationModeV1::FreezeASingleFactorV1 { expected_a_digest } => {
            validate_homogeneous_rank_and_alpha(submissions)?;
            let mut b_updates = Vec::with_capacity(submissions.len());

            for (idx, (peer, pair)) in submissions.iter().enumerate() {
                if pair.frozen_a_digest() != expected_a_digest {
                    return Err(LoraError::FrozenADigestMismatch);
                }
                b_updates.push(GradientUpdate {
                    sender: *peer.as_bytes(),
                    sequence: idx as u64,
                    gradients: pair.b_weights.clone(),
                    round_id: 0,
                });
            }

            let agg_b = b_factor_aggregator
                .aggregate(&b_updates)
                .map_err(LoraError::AggregationFailed)?;
            let adapter = LoraAdapterPairV1::try_new(
                first.descriptor.clone(),
                d_in,
                d_out,
                first.a_weights.clone(),
                agg_b,
            )?;
            Ok(LoraAggregationOutcomeV1::SingleFactorAggregated { adapter })
        }
        LoraAggregationModeV1::FloraStackingV1 { max_stacked_rank } => {
            if max_stacked_rank == 0 {
                return Err(LoraError::StackedRankExceedsLimit);
            }
            let mut total_rank_u64 = 0u64;
            let mut client_ranks = Vec::with_capacity(submissions.len());
            for (_, pair) in submissions {
                total_rank_u64 = total_rank_u64
                    .checked_add(pair.descriptor.rank as u64)
                    .ok_or(LoraError::StackedRankExceedsLimit)?;
                client_ranks.push(pair.descriptor.rank);
            }
            if total_rank_u64 > (max_stacked_rank as u64) {
                return Err(LoraError::StackedRankExceedsLimit);
            }
            let total_stacked_rank =
                usize::try_from(total_rank_u64).map_err(|_| LoraError::StackedRankExceedsLimit)?;
            let b_len = d_out
                .checked_mul(total_stacked_rank)
                .ok_or(LoraError::InvalidRankOrDimension)?;
            let a_len = total_stacked_rank
                .checked_mul(d_in)
                .ok_or(LoraError::InvalidRankOrDimension)?;

            let mut stacked_b = vec![0.0f32; b_len];
            let mut stacked_a = Vec::with_capacity(a_len);
            // Normative FLoRA stacking applies uniform 1/n peer weighting across unique participants.
            let weight_i = 1.0f64 / (submissions.len() as f64);

            let mut col_cursor = 0usize;
            for (_, pair) in submissions {
                let r_i = usize::try_from(pair.descriptor.rank)
                    .map_err(|_| LoraError::InvalidRankOrDimension)?;
                let scale_i = weight_i * ((pair.descriptor.alpha as f64) / (r_i as f64));
                for row in 0..d_out {
                    let src_offset = row * r_i;
                    let dst_offset = row * total_stacked_rank + col_cursor;
                    for k in 0..r_i {
                        let val = ((pair.b_weights[src_offset + k] as f64) * scale_i) as f32;
                        if !val.is_finite() {
                            return Err(LoraError::NonFiniteWeight);
                        }
                        stacked_b[dst_offset + k] = val;
                    }
                }
                stacked_a.extend_from_slice(&pair.a_weights);
                col_cursor += r_i;
            }

            let stacked = FloraStackedDeltaV1 {
                target_module: target_module.clone(),
                d_in,
                d_out,
                total_stacked_rank,
                client_ranks,
                stacked_b,
                stacked_a,
            };
            stacked.validate()?;
            Ok(LoraAggregationOutcomeV1::FloraStacked { stacked })
        }
        LoraAggregationModeV1::BiasedFactorFedAvgV1 => {
            validate_homogeneous_rank_and_alpha(submissions)?;

            let n_inv = 1.0f64 / (submissions.len() as f64);
            let avg_a = mean_factor_slice(submissions, first.a_weights.len(), |p| &p.a_weights);
            let avg_b = mean_factor_slice(submissions, first.b_weights.len(), |p| &p.b_weights);

            let biased_adapter =
                LoraAdapterPairV1::try_new(first.descriptor.clone(), d_in, d_out, avg_a, avg_b)?;
            let biased_delta = biased_adapter.materialize_scaled_delta()?;

            // Compute exact average delta (1/n) * sum_i (alpha_i / r_i) B_i A_i to measure Frobenius drift
            let total_elems = d_out
                .checked_mul(d_in)
                .ok_or(LoraError::InvalidRankOrDimension)?;
            let mut exact_delta = vec![0.0f64; total_elems];
            for (_, pair) in submissions {
                let d_i = pair.materialize_scaled_delta()?;
                for (acc, &val) in exact_delta.iter_mut().zip(d_i.iter()) {
                    *acc += (val as f64) * n_inv;
                }
            }

            let mut frob_sq = 0.0f64;
            for (b_val, e_val) in biased_delta.iter().zip(exact_delta.iter()) {
                let diff = (*b_val as f64) - *e_val;
                frob_sq += diff * diff;
            }
            let cross_term_frobenius_drift_bound = sqrt_f64_soft(frob_sq);

            Ok(LoraAggregationOutcomeV1::BiasedFactorFedAvg {
                adapter: biased_adapter,
                cross_term_frobenius_drift_bound,
            })
        }
    }
}

fn mean_factor_slice<'a, F>(
    submissions: &'a [(PeerId, LoraAdapterPairV1)],
    len: usize,
    extract: F,
) -> Vec<f32>
where
    F: Fn(&'a LoraAdapterPairV1) -> &'a [f32],
{
    let n_inv = 1.0f64 / (submissions.len() as f64);
    let mut acc = vec![0.0f64; len];
    for (_, pair) in submissions {
        for (dst, &val) in acc.iter_mut().zip(extract(pair).iter()) {
            *dst += val as f64;
        }
    }
    acc.into_iter().map(|s| (s * n_inv) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_torch_core::aggregation::{CoordinateMedian, FedAvg};

    fn sample_descriptor(rank: u32, alpha: f32) -> LoRaAdapterDescriptorV1 {
        LoRaAdapterDescriptorV1 {
            target_module: String::from("attn.q_proj"),
            tensor_a_name: String::from("attn.q_proj.lora_a"),
            tensor_b_name: String::from("attn.q_proj.lora_b"),
            rank,
            alpha,
        }
    }

    #[test]
    fn freeze_a_single_factor_eliminates_cross_term_and_tolerates_byzantine_peer() {
        let d_in = 3;
        let d_out = 2;
        let rank = 2;
        let desc = sample_descriptor(rank, 4.0); // scale = alpha / r = 2.0
        let a_shared = vec![1.0, 0.5, -0.5, -0.25, 0.75, 1.0]; // 2 x 3

        let p1 = LoraAdapterPairV1::try_new(
            desc.clone(),
            d_in,
            d_out,
            a_shared.clone(),
            vec![0.2, 0.4, -0.1, 0.3], // 2 x 2
        )
        .unwrap();
        let p2 = LoraAdapterPairV1::try_new(
            desc.clone(),
            d_in,
            d_out,
            a_shared.clone(),
            vec![0.4, 0.6, -0.3, 0.5], // 2 x 2
        )
        .unwrap();
        // Peer 3 is Byzantine in B
        let p3_byz = LoraAdapterPairV1::try_new(
            desc,
            d_in,
            d_out,
            a_shared,
            vec![100.0, -100.0, 100.0, -100.0],
        )
        .unwrap();

        let expected_a_digest = p1.frozen_a_digest();
        let submissions = vec![
            (PeerId::new([1u8; 32]), p1.clone()),
            (PeerId::new([2u8; 32]), p2.clone()),
            (PeerId::new([3u8; 32]), p3_byz),
        ];

        // CoordinateMedian selects exact coordinate-wise medians [0.4, 0.4, -0.1, 0.3], immune to +-100!
        let outcome = aggregate_lora_adapters_v1(
            &submissions,
            LoraAggregationModeV1::FreezeASingleFactorV1 { expected_a_digest },
            &CoordinateMedian,
        )
        .unwrap();

        match outcome {
            LoraAggregationOutcomeV1::SingleFactorAggregated { adapter } => {
                assert_eq!(adapter.b_weights, vec![0.4, 0.4, -0.1, 0.3]);
                let mut base_w = vec![1.0f32; d_out * d_in];
                adapter.apply_to_base_weights(&mut base_w).unwrap();
                let direct = adapter.materialize_scaled_delta().unwrap();
                for (w, d) in base_w.iter().zip(direct.iter()) {
                    assert!((*w - (1.0 + *d)).abs() < 1e-6);
                }
            }
            _ => panic!("expected SingleFactorAggregated"),
        }

        // Conflicting alpha inside FreezeASingleFactorV1 fails closed to prevent scale inflation
        let bad_alpha = LoraAdapterPairV1::try_new(
            sample_descriptor(rank, 40.0),
            d_in,
            d_out,
            p1.a_weights.clone(),
            p1.b_weights.clone(),
        )
        .unwrap();
        assert_eq!(
            aggregate_lora_adapters_v1(
                &[
                    (PeerId::new([1u8; 32]), p1.clone()),
                    (PeerId::new([2u8; 32]), bad_alpha),
                ],
                LoraAggregationModeV1::FreezeASingleFactorV1 { expected_a_digest },
                &FedAvg,
            )
            .unwrap_err(),
            LoraError::CohortMismatch
        );

        // Tampering with A_0 fails closed with FrozenADigestMismatch
        let mut bad_a = p1.clone();
        bad_a.a_weights[0] += 0.01;
        let bad_subs = vec![(PeerId::new([1u8; 32]), bad_a)];
        assert_eq!(
            aggregate_lora_adapters_v1(
                &bad_subs,
                LoraAggregationModeV1::FreezeASingleFactorV1 { expected_a_digest },
                &FedAvg,
            )
            .unwrap_err(),
            LoraError::FrozenADigestMismatch
        );
    }

    #[test]
    fn flora_stacking_supports_heterogeneous_ranks_with_zero_cross_term_error() {
        let d_in = 3;
        let d_out = 2;

        // Client 1 has rank r_1 = 1, alpha_1 = 2.0
        let c1 = LoraAdapterPairV1::try_new(
            sample_descriptor(1, 2.0),
            d_in,
            d_out,
            vec![1.0, -2.0, 0.5], // 1 x 3
            vec![0.5, -1.5],      // 2 x 1
        )
        .unwrap();

        // Client 2 has rank r_2 = 2, alpha_2 = 4.0
        let c2 = LoraAdapterPairV1::try_new(
            sample_descriptor(2, 4.0),
            d_in,
            d_out,
            vec![0.25, 0.5, -0.75, -1.0, 0.5, 0.25], // 2 x 3
            vec![1.0, 0.5, -0.5, 2.0],               // 2 x 2
        )
        .unwrap();

        let submissions = vec![
            (PeerId::new([1u8; 32]), c1.clone()),
            (PeerId::new([2u8; 32]), c2.clone()),
        ];

        let outcome = aggregate_lora_adapters_v1(
            &submissions,
            LoraAggregationModeV1::FloraStackingV1 {
                max_stacked_rank: 16,
            },
            &FedAvg,
        )
        .unwrap();

        let stacked = match outcome {
            LoraAggregationOutcomeV1::FloraStacked { stacked } => stacked,
            _ => panic!("expected FloraStacked"),
        };
        assert_eq!(stacked.total_stacked_rank, 3);
        assert_eq!(stacked.client_ranks, vec![1, 2]);

        let flora_delta = stacked.materialize_exact_delta().unwrap();
        let d1 = c1.materialize_scaled_delta().unwrap();
        let d2 = c2.materialize_scaled_delta().unwrap();

        for j in 0..(d_out * d_in) {
            let exact_avg = 0.5 * (d1[j] + d2[j]);
            assert!(
                (flora_delta[j] - exact_avg).abs() < 1e-6,
                "element {j}: flora {} != exact {}",
                flora_delta[j],
                exact_avg
            );
        }
    }

    #[test]
    fn biased_factor_fedavg_quantifies_non_zero_cross_term_drift() {
        let d_in = 2;
        let d_out = 2;
        let desc = sample_descriptor(1, 1.0);

        let c1 =
            LoraAdapterPairV1::try_new(desc.clone(), d_in, d_out, vec![1.0, 0.0], vec![1.0, 0.0])
                .unwrap();
        let c2 =
            LoraAdapterPairV1::try_new(desc, d_in, d_out, vec![0.0, 1.0], vec![0.0, 1.0]).unwrap();

        let submissions = vec![(PeerId::new([1u8; 32]), c1), (PeerId::new([2u8; 32]), c2)];
        let outcome = aggregate_lora_adapters_v1(
            &submissions,
            LoraAggregationModeV1::BiasedFactorFedAvgV1,
            &FedAvg,
        )
        .unwrap();

        match outcome {
            LoraAggregationOutcomeV1::BiasedFactorFedAvg {
                cross_term_frobenius_drift_bound,
                ..
            } => {
                // Exact average is [[0.5, 0.0], [0.0, 0.5]], whereas bar_B * bar_A is [[0.25, 0.25], [0.25, 0.25]].
                // Difference matrix has four +-0.25 entries => Frobenius norm = sqrt(4 * 0.0625) = 0.5!
                assert!((cross_term_frobenius_drift_bound - 0.5).abs() < 1e-6);
            }
            _ => panic!("expected BiasedFactorFedAvg"),
        }
    }

    #[test]
    fn flora_stacked_validate_rejects_malformed_dimensions_lengths_and_non_finite_without_mutation()
    {
        let valid = FloraStackedDeltaV1 {
            target_module: String::from("attn.q_proj"),
            d_in: 2,
            d_out: 2,
            total_stacked_rank: 3,
            client_ranks: vec![1, 2],
            stacked_b: vec![0.1; 6],
            stacked_a: vec![0.2; 6],
        };
        assert!(valid.validate().is_ok());
        let mut base = vec![1.0f32; 4];

        let mut short_b = valid.clone();
        short_b.stacked_b.pop();
        assert_eq!(short_b.validate(), Err(LoraError::InvalidRankOrDimension));
        assert_eq!(
            short_b.apply_exact_to_base_weights(&mut base),
            Err(LoraError::InvalidRankOrDimension)
        );
        assert_eq!(
            short_b.materialize_exact_delta(),
            Err(LoraError::InvalidRankOrDimension)
        );

        let mut short_a = valid.clone();
        short_a.stacked_a.pop();
        assert_eq!(short_a.validate(), Err(LoraError::InvalidRankOrDimension));

        let mut zero_rank = valid.clone();
        zero_rank.client_ranks = vec![0, 3];
        assert_eq!(
            zero_rank.apply_exact_to_base_weights(&mut base),
            Err(LoraError::InvalidRankOrDimension)
        );

        let mut mismatched_rank_sum = valid.clone();
        mismatched_rank_sum.client_ranks = vec![1, 1];
        assert_eq!(
            mismatched_rank_sum.validate(),
            Err(LoraError::InvalidRankOrDimension)
        );

        let mut empty_rank = valid.clone();
        empty_rank.total_stacked_rank = 0;
        empty_rank.client_ranks.clear();
        empty_rank.stacked_b.clear();
        empty_rank.stacked_a.clear();
        assert_eq!(
            empty_rank.apply_exact_to_base_weights(&mut base),
            Err(LoraError::InvalidRankOrDimension)
        );

        let mut nan_b = valid.clone();
        nan_b.stacked_b[5] = f32::NAN;
        assert_eq!(
            nan_b.apply_exact_to_base_weights(&mut base),
            Err(LoraError::NonFiniteWeight)
        );
        let mut nan_a = valid.clone();
        nan_a.stacked_a[5] = f32::INFINITY;
        assert_eq!(
            nan_a.apply_exact_to_base_weights(&mut base),
            Err(LoraError::NonFiniteWeight)
        );
        assert_eq!(base, vec![1.0f32; 4]);
    }

    #[test]
    fn aggregate_lora_rejects_duplicate_peers_across_all_modes_and_enforces_alpha_symmetry() {
        let d_in = 2;
        let d_out = 2;
        let p_a = PeerId::new([1u8; 32]);
        let p_b = PeerId::new([2u8; 32]);
        let pair_a = LoraAdapterPairV1::try_new(
            sample_descriptor(1, 2.0),
            d_in,
            d_out,
            vec![1.0, 0.5],
            vec![0.2, -0.3],
        )
        .unwrap();
        let digest = pair_a.frozen_a_digest();

        // Duplicate PeerId must be rejected identically in all 3 modes.
        let dup_subs = vec![(p_a, pair_a.clone()), (p_a, pair_a.clone())];
        for mode in [
            LoraAggregationModeV1::FreezeASingleFactorV1 {
                expected_a_digest: digest,
            },
            LoraAggregationModeV1::FloraStackingV1 {
                max_stacked_rank: 8,
            },
            LoraAggregationModeV1::BiasedFactorFedAvgV1,
        ] {
            assert_eq!(
                aggregate_lora_adapters_v1(&dup_subs, mode, &FedAvg).unwrap_err(),
                LoraError::DuplicatePeerSubmission(p_a)
            );
        }

        // Alpha mismatch with identical rank rejects identically as CohortMismatch in BOTH
        // FreezeASingleFactorV1 and BiasedFactorFedAvgV1.
        let mismatched_alpha = LoraAdapterPairV1::try_new(
            sample_descriptor(1, 2.5),
            d_in,
            d_out,
            pair_a.a_weights.clone(),
            pair_a.b_weights.clone(),
        )
        .unwrap();
        assert_eq!(
            aggregate_lora_adapters_v1(
                &[(p_a, pair_a.clone()), (p_b, mismatched_alpha.clone())],
                LoraAggregationModeV1::FreezeASingleFactorV1 {
                    expected_a_digest: digest,
                },
                &FedAvg,
            )
            .unwrap_err(),
            LoraError::CohortMismatch
        );
        assert_eq!(
            aggregate_lora_adapters_v1(
                &[(p_a, pair_a), (p_b, mismatched_alpha)],
                LoraAggregationModeV1::BiasedFactorFedAvgV1,
                &FedAvg,
            )
            .unwrap_err(),
            LoraError::CohortMismatch
        );
    }

    #[test]
    fn flora_and_lora_enforce_dimension_overflow_base_rollback_and_mean_overflow_safety() {
        let mut base = vec![1.0f32, f32::MAX, -1.0, 2.0];
        let base_before = base.clone();

        // Overflow during merge on element 1 must leave ALL elements of `base` (including element 0) untouched.
        let pair_overflow = LoraAdapterPairV1::try_new(
            sample_descriptor(1, 2.0),
            2,
            2,
            vec![1.0, 1.0],
            vec![1.0e38, 0.0],
        )
        .unwrap();
        assert_eq!(
            pair_overflow.apply_to_base_weights(&mut base),
            Err(LoraError::NonFiniteWeight)
        );
        assert_eq!(base, base_before);

        // FloraStackedDeltaV1::validate rejects empty target_module, zero dims, and product overflow.
        let empty_mod = FloraStackedDeltaV1 {
            target_module: String::new(),
            d_in: 2,
            d_out: 2,
            total_stacked_rank: 1,
            client_ranks: vec![1],
            stacked_b: vec![0.1; 2],
            stacked_a: vec![0.1; 2],
        };
        assert_eq!(empty_mod.validate(), Err(LoraError::InvalidRankOrDimension));

        let zero_dim = FloraStackedDeltaV1 {
            target_module: String::from("attn.q_proj"),
            d_in: 0,
            d_out: 2,
            total_stacked_rank: 1,
            client_ranks: vec![1],
            stacked_b: vec![0.1; 2],
            stacked_a: vec![],
        };
        assert_eq!(zero_dim.validate(), Err(LoraError::InvalidRankOrDimension));

        let overflow_dim = FloraStackedDeltaV1 {
            target_module: String::from("attn.q_proj"),
            d_in: 2,
            d_out: usize::MAX,
            total_stacked_rank: 2,
            client_ranks: vec![1, 1],
            stacked_b: vec![],
            stacked_a: vec![0.1; 4],
        };
        assert_eq!(
            overflow_dim.validate(),
            Err(LoraError::InvalidRankOrDimension)
        );

        let stacked_overflow = FloraStackedDeltaV1 {
            target_module: String::from("attn.q_proj"),
            d_in: 2,
            d_out: 2,
            total_stacked_rank: 1,
            client_ranks: vec![1],
            stacked_b: vec![1.0e38, 0.0],
            stacked_a: vec![1.0, 1.0],
        };
        assert_eq!(
            stacked_overflow.apply_exact_to_base_weights(&mut base),
            Err(LoraError::NonFiniteWeight)
        );
        assert_eq!(base, base_before);

        // BiasedFactorFedAvgV1 / mean_factor_slice accumulates in f64 so individual peer weights
        // summing beyond f32::MAX (3e38 + 3e38 = 6e38) remain finite without overflow.
        let p1 = PeerId::new([1u8; 32]);
        let p2 = PeerId::new([2u8; 32]);
        let big1 =
            LoraAdapterPairV1::try_new(sample_descriptor(1, 1.0), 1, 1, vec![3.0e38], vec![0.5])
                .unwrap();
        let big2 =
            LoraAdapterPairV1::try_new(sample_descriptor(1, 1.0), 1, 1, vec![3.0e38], vec![0.5])
                .unwrap();
        match aggregate_lora_adapters_v1(
            &[(p1, big1), (p2, big2)],
            LoraAggregationModeV1::BiasedFactorFedAvgV1,
            &FedAvg,
        )
        .unwrap()
        {
            LoraAggregationOutcomeV1::BiasedFactorFedAvg { adapter, .. } => {
                assert!(adapter.a_weights[0].is_finite());
                assert!((adapter.a_weights[0] - 3.0e38).abs() / 3.0e38 < 1e-6);
            }
            _ => panic!("expected BiasedFactorFedAvg"),
        }

        // FloraStackingV1 rejects max_stacked_rank == 0, exceeded rank, and scaling overflow.
        let small = LoraAdapterPairV1::try_new(
            sample_descriptor(2, 1.0),
            1,
            1,
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        )
        .unwrap();
        assert_eq!(
            aggregate_lora_adapters_v1(
                &[(p1, small.clone())],
                LoraAggregationModeV1::FloraStackingV1 {
                    max_stacked_rank: 0
                },
                &FedAvg,
            )
            .unwrap_err(),
            LoraError::StackedRankExceedsLimit
        );
        assert_eq!(
            aggregate_lora_adapters_v1(
                &[(p1, small)],
                LoraAggregationModeV1::FloraStackingV1 {
                    max_stacked_rank: 1
                },
                &FedAvg,
            )
            .unwrap_err(),
            LoraError::StackedRankExceedsLimit
        );
        let scale_overflow =
            LoraAdapterPairV1::try_new(sample_descriptor(1, 10.0), 1, 1, vec![1.0], vec![f32::MAX])
                .unwrap();
        assert_eq!(
            aggregate_lora_adapters_v1(
                &[(p1, scale_overflow)],
                LoraAggregationModeV1::FloraStackingV1 {
                    max_stacked_rank: 4
                },
                &FedAvg,
            )
            .unwrap_err(),
            LoraError::NonFiniteWeight
        );
    }

    #[test]
    fn t_agent_alpha_within_tolerance_rejects_small_alpha_amplification_and_is_order_invariant() {
        let p_a = PeerId::new([1u8; 32]);
        let p_b = PeerId::new([2u8; 32]);
        let p_c = PeerId::new([3u8; 32]);

        // 10x alpha mismatch (1e-7 vs 1e-6) MUST be rejected as CohortMismatch in BOTH
        // FreezeASingleFactorV1 and BiasedFactorFedAvgV1 under relative tolerance 1e-6.
        let low_alpha =
            LoraAdapterPairV1::try_new(sample_descriptor(1, 1e-7), 1, 1, vec![1.0], vec![1.0])
                .unwrap();
        let inflated_alpha =
            LoraAdapterPairV1::try_new(sample_descriptor(1, 1e-6), 1, 1, vec![1.0], vec![1.0])
                .unwrap();
        let digest = low_alpha.frozen_a_digest();

        assert_eq!(
            aggregate_lora_adapters_v1(
                &[(p_a, low_alpha.clone()), (p_b, inflated_alpha.clone())],
                LoraAggregationModeV1::FreezeASingleFactorV1 {
                    expected_a_digest: digest,
                },
                &FedAvg,
            )
            .unwrap_err(),
            LoraError::CohortMismatch
        );
        assert_eq!(
            aggregate_lora_adapters_v1(
                &[(p_a, low_alpha), (p_b, inflated_alpha)],
                LoraAggregationModeV1::BiasedFactorFedAvgV1,
                &FedAvg,
            )
            .unwrap_err(),
            LoraError::CohortMismatch
        );

        // Cohort alpha validation must also be invariant to submission slice ordering.
        let a_mid =
            LoraAdapterPairV1::try_new(sample_descriptor(1, 100.0), 1, 1, vec![1.0], vec![1.0])
                .unwrap();
        let b_hi = LoraAdapterPairV1::try_new(
            sample_descriptor(1, 100.0 + 9e-5),
            1,
            1,
            vec![1.0],
            vec![1.0],
        )
        .unwrap();
        let c_lo = LoraAdapterPairV1::try_new(
            sample_descriptor(1, 100.0 - 9e-5),
            1,
            1,
            vec![1.0],
            vec![1.0],
        )
        .unwrap();
        let res_mid_first = aggregate_lora_adapters_v1(
            &[
                (p_a, a_mid.clone()),
                (p_b, b_hi.clone()),
                (p_c, c_lo.clone()),
            ],
            LoraAggregationModeV1::BiasedFactorFedAvgV1,
            &FedAvg,
        );
        let res_hi_first = aggregate_lora_adapters_v1(
            &[(p_b, b_hi), (p_a, a_mid), (p_c, c_lo)],
            LoraAggregationModeV1::BiasedFactorFedAvgV1,
            &FedAvg,
        );
        assert_eq!(res_mid_first.unwrap_err(), LoraError::CohortMismatch);
        assert_eq!(res_hi_first.unwrap_err(), LoraError::CohortMismatch);
    }
}
