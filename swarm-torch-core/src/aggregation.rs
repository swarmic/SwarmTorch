//! Robust aggregation algorithms for Byzantine-resilient swarm learning (`ADR-0019`, `W9-01`).
//!
//! This module provides aggregation strategies that tolerate malicious or faulty
//! participants in the swarm with:
//! - strict pre-aggregation rejection of `NaN` (`Error::NanDetected`) and `±Inf` (`Error::InfDetected`),
//! - total-order sorting (`f32::total_cmp` / `f64::total_cmp`) and canonical peer tie-breaking,
//! - widened `f64` internal accumulation and distance computation,
//! - checked arithmetic and bounded scratch/work budgets (`AggregationBudget`),
//! - one-pass `Multi-Krum` (`Krum::multi`) and `Bulyan` (`BulyanProfile::IterativeKrumV1`).

#[cfg(feature = "alloc")]
use alloc::string::ToString;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use crate::compression::{
    sample_shared_indices_v1, verify_shared_seed_openings_v1, CompressedGradient, CompressionError,
    CompressionMethod, MerkleOpeningBatchV1, PreSeedVectorCommitmentV1, SharedSeedSparsePayloadV1,
};
#[cfg(feature = "alloc")]
use crate::dataops::TransformAuditV0;
#[cfg(feature = "alloc")]
use crate::traits::UpdateTransform;
use crate::traits::{GradientUpdate, PeerId};
use crate::{Error, Result};
#[cfg(feature = "alloc")]
use sha2::{Digest, Sha256};

/// Maximum gradient vector dimension accepted by default (`10,000,000` coordinates).
pub const MAX_GRADIENT_DIM: usize = 10_000_000;

/// Default maximum peers for pairwise $O(n^2 d)$ distance aggregators (`Krum`, `Bulyan`).
pub const MAX_KRUM_PEERS: usize = 50;

/// Explicit resource and work budget for aggregation (`ADR-0019`, `W9-01`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct AggregationBudget {
    /// Maximum allowed number of peer updates (`n`).
    pub max_peers: usize,
    /// Maximum allowed gradient vector dimension (`d`).
    pub max_dimension: usize,
    /// Maximum allowed scratch memory bytes (`n * n * 8` for pairwise or `n * 4` for coordinate-wise).
    pub max_scratch_bytes: usize,
    /// Maximum allowed elementary `f64` operations (`n * n * d` for pairwise or `n * d` for linear).
    pub max_work_ops: usize,
}

impl AggregationBudget {
    /// Default budget for pairwise quadratic/cubic aggregators (`Krum`, `Bulyan`).
    pub const fn default_pairwise() -> Self {
        Self {
            max_peers: MAX_KRUM_PEERS,
            max_dimension: MAX_GRADIENT_DIM,
            max_scratch_bytes: 64 * 1024 * 1024,
            max_work_ops: 250_000_000,
        }
    }

    /// Validate `(n, dim)` against this budget using checked arithmetic.
    pub fn check(&self, n: usize, dim: usize, pairwise: bool) -> Result<()> {
        if n > self.max_peers || dim > self.max_dimension {
            return Err(Error::ResourceBudgetExceeded);
        }
        let scratch_bytes = if pairwise {
            n.checked_mul(n)
                .and_then(|nn| nn.checked_mul(core::mem::size_of::<f64>()))
                .ok_or(Error::ResourceExhausted)?
        } else {
            n.checked_mul(core::mem::size_of::<f32>())
                .ok_or(Error::ResourceExhausted)?
        };
        if scratch_bytes > self.max_scratch_bytes {
            return Err(Error::ResourceBudgetExceeded);
        }
        let work_ops = if pairwise {
            n.checked_mul(n)
                .and_then(|nn| nn.checked_mul(dim))
                .ok_or(Error::ResourceExhausted)?
        } else {
            n.checked_mul(dim).ok_or(Error::ResourceExhausted)?
        };
        if work_ops > self.max_work_ops {
            return Err(Error::ResourceBudgetExceeded);
        }
        Ok(())
    }
}

impl Default for AggregationBudget {
    fn default() -> Self {
        Self::default_pairwise()
    }
}

#[cfg(feature = "alloc")]
fn validate_gradient_shapes_and_finiteness(updates: &[GradientUpdate]) -> Result<usize> {
    if updates.is_empty() {
        return Err(Error::InsufficientUpdates);
    }

    let dim = updates[0].gradients.len();
    if dim == 0 || dim > MAX_GRADIENT_DIM {
        return Err(Error::InvalidGradient);
    }

    for update in updates {
        if update.gradients.len() != dim {
            return Err(Error::InvalidGradient);
        }
        for &val in &update.gradients {
            if val.is_nan() {
                return Err(Error::NanDetected);
            }
            if val.is_infinite() {
                return Err(Error::InfDetected);
            }
        }
    }

    Ok(dim)
}

#[cfg(feature = "alloc")]
fn validate_finite_output(values: &[f32]) -> Result<()> {
    for &val in values {
        if val.is_nan() {
            return Err(Error::NanDetected);
        }
        if val.is_infinite() {
            return Err(Error::InfDetected);
        }
    }
    Ok(())
}

/// Trait for robust aggregation algorithms
pub trait RobustAggregator: Send + Sync {
    /// Aggregate multiple gradient updates into one
    #[cfg(feature = "alloc")]
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>>;

    /// Fraction of Byzantine nodes this aggregator tolerates
    fn byzantine_tolerance(&self) -> f32;

    /// Computational complexity class
    fn complexity(&self) -> AggregatorComplexity;
}

/// Computational complexity classification for aggregators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregatorComplexity {
    /// O(n) - e.g., Trimmed Mean
    Linear,
    /// O(n²) - e.g., Krum
    Quadratic,
    /// O(n³) - e.g., Bulyan
    Cubic,
}

/// Pipeline error for composable aggregation execution.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub enum AggregationPipelineError {
    Aggregation(Error),
    Compression(CompressionError),
}

/// Composable aggregation pipeline over a concrete robust aggregator.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct AggregationPipeline<A: RobustAggregator> {
    aggregator: A,
    compression: Option<CompressionMethod>,
}

/// Integer-estimate trace output for pipeline planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregationTraceResult {
    pub estimated_memory_bytes: u64,
    pub estimated_compressed_bytes: Option<u64>,
    pub estimated_flops: u64,
    pub num_peers: usize,
    pub gradient_len: usize,
}

#[cfg(feature = "alloc")]
impl<A: RobustAggregator> AggregationPipeline<A> {
    pub fn new(aggregator: A) -> Self {
        Self {
            aggregator,
            compression: None,
        }
    }

    pub fn with_compression(mut self, method: CompressionMethod) -> Self {
        self.compression = Some(method);
        self
    }

    /// Run aggregation and optional compression roundtrip.
    ///
    /// Pre-transforms are applied upstream via `apply_update_transforms`.
    pub fn run(
        &self,
        updates: &[GradientUpdate],
    ) -> core::result::Result<Vec<f32>, AggregationPipelineError> {
        let aggregated = self
            .aggregator
            .aggregate(updates)
            .map_err(AggregationPipelineError::Aggregation)?;

        if let Some(method) = self.compression.clone() {
            let compressed = CompressedGradient::compress(&aggregated, method)
                .map_err(AggregationPipelineError::Compression)?;
            return compressed
                .decompress()
                .map_err(AggregationPipelineError::Compression);
        }
        Ok(aggregated)
    }

    /// Estimate memory/compression/FLOP costs for a candidate execution shape.
    pub fn trace(&self, gradient_len: usize, num_peers: usize) -> AggregationTraceResult {
        let estimated_memory_bytes = (gradient_len as u64)
            .saturating_mul(num_peers as u64)
            .saturating_mul(4);
        let estimated_flops = (gradient_len as u64).saturating_mul(num_peers as u64);

        let estimated_compressed_bytes =
            self.compression.as_ref().and_then(|method| match method {
                CompressionMethod::TopK { k_ratio } => {
                    if gradient_len == 0 || num_peers == 0 {
                        return Some(0);
                    }
                    let raw = (gradient_len as f32) * *k_ratio;
                    let mut k = raw as u64;
                    if (k as f32) < raw {
                        k = k.saturating_add(1);
                    }
                    let k = k.max(1).min(gradient_len as u64);
                    // sparse wire estimate: u32 index + f32 value per kept element
                    Some(k.saturating_mul(8))
                }
                _ => None,
            });

        AggregationTraceResult {
            estimated_memory_bytes,
            estimated_compressed_bytes,
            estimated_flops,
            num_peers,
            gradient_len,
        }
    }
}

/// Apply an update transform and return transformed updates plus audit metadata.
///
/// Provenance fields (`sender`, `sequence`, `round_id`) are preserved from input updates.
#[cfg(feature = "alloc")]
pub fn apply_update_transforms<T: UpdateTransform>(
    updates: Vec<GradientUpdate>,
    transform: &T,
) -> (Vec<GradientUpdate>, TransformAuditV0) {
    let round_id = updates.first().map(|u| u.round_id).unwrap_or(0);
    let transformed = updates
        .into_iter()
        .map(|update| {
            let sender = update.sender;
            let sequence = update.sequence;
            let round_id = update.round_id;
            let mut out = transform.transform(update);
            out.sender = sender;
            out.sequence = sequence;
            out.round_id = round_id;
            out
        })
        .collect();

    let audit = TransformAuditV0 {
        transform_name: transform.name().to_string(),
        core_trusted: transform.is_core_trusted(),
        round_id,
    };

    (transformed, audit)
}

/// Simple averaging aggregator (no Byzantine protection), widened to `f64` internal accumulation.
#[derive(Debug, Clone, Default)]
pub struct FedAvg;

impl RobustAggregator for FedAvg {
    #[cfg(feature = "alloc")]
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>> {
        let dim = validate_gradient_shapes_and_finiteness(updates)?;
        let inv_n = 1.0f64 / (updates.len() as f64);
        let mut sums = alloc::vec![0.0f64; dim];

        for update in updates {
            for (slot, &gradient) in sums.iter_mut().zip(update.gradients.iter()) {
                *slot += gradient as f64;
            }
        }

        let mut result = alloc::vec![0.0f32; dim];
        for (out, &sum) in result.iter_mut().zip(sums.iter()) {
            *out = (sum * inv_n) as f32;
        }
        validate_finite_output(&result)?;
        Ok(result)
    }

    fn byzantine_tolerance(&self) -> f32 {
        0.0
    }

    fn complexity(&self) -> AggregatorComplexity {
        AggregatorComplexity::Linear
    }
}

/// Trimmed mean aggregator — discards top/bottom `k%` of values per coordinate.
#[derive(Debug, Clone)]
pub struct TrimmedMean {
    /// Fraction of values to trim from each end (e.g., `0.2` for 20%).
    pub trim_ratio: f32,
}

impl TrimmedMean {
    /// Create a new `TrimmedMean` aggregator (clamping `trim_ratio` to `[0.0, 0.49]`).
    pub fn new(trim_ratio: f32) -> Self {
        let clamped = if trim_ratio.is_nan() {
            0.0
        } else {
            trim_ratio.clamp(0.0, 0.49)
        };
        Self {
            trim_ratio: clamped,
        }
    }

    /// Fallible constructor that rejects non-finite or out-of-range (`< 0.0` or `>= 0.5`) trim ratios.
    pub fn try_new(trim_ratio: f32) -> Result<Self> {
        if !trim_ratio.is_finite() || !(0.0..0.5).contains(&trim_ratio) {
            return Err(Error::InvalidThreshold);
        }
        Ok(Self { trim_ratio })
    }
}

impl Default for TrimmedMean {
    fn default() -> Self {
        Self::new(0.2)
    }
}

impl RobustAggregator for TrimmedMean {
    #[cfg(feature = "alloc")]
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>> {
        if !self.trim_ratio.is_finite() || self.trim_ratio < 0.0 {
            return Err(Error::InvalidThreshold);
        }
        let dim = validate_gradient_shapes_and_finiteness(updates)?;
        let n = updates.len();
        let trim_count = ((n as f64) * (self.trim_ratio as f64)) as usize;

        if trim_count
            .checked_mul(2)
            .map(|two_k| n <= two_k)
            .unwrap_or(true)
        {
            return Err(Error::InsufficientUpdates);
        }

        let mut result = alloc::vec![0.0f32; dim];
        let mut values = alloc::vec![0.0f32; n];
        let kept_len = n - 2 * trim_count;
        let inv_kept = 1.0f64 / (kept_len as f64);

        for (i, slot) in result.iter_mut().enumerate().take(dim) {
            for (dst, u) in values.iter_mut().zip(updates.iter()) {
                *dst = u.gradients[i];
            }
            values.sort_by(f32::total_cmp);

            let trimmed = &values[trim_count..n - trim_count];
            let sum: f64 = trimmed.iter().map(|&v| v as f64).sum();
            *slot = (sum * inv_kept) as f32;
        }

        validate_finite_output(&result)?;
        Ok(result)
    }

    fn byzantine_tolerance(&self) -> f32 {
        self.trim_ratio
    }

    fn complexity(&self) -> AggregatorComplexity {
        AggregatorComplexity::Linear
    }
}

/// Coordinate-wise median aggregator with widened `f64` even-count midpoint calculation.
#[derive(Debug, Clone, Default)]
pub struct CoordinateMedian;

impl RobustAggregator for CoordinateMedian {
    #[cfg(feature = "alloc")]
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>> {
        let dim = validate_gradient_shapes_and_finiteness(updates)?;
        let n = updates.len();
        let mut result = alloc::vec![0.0f32; dim];
        let mut values = alloc::vec![0.0f32; n];

        for (i, slot) in result.iter_mut().enumerate().take(dim) {
            for (dst, u) in values.iter_mut().zip(updates.iter()) {
                *dst = u.gradients[i];
            }
            values.sort_by(f32::total_cmp);

            *slot = if n % 2 == 0 {
                (((values[n / 2 - 1] as f64) + (values[n / 2] as f64)) * 0.5) as f32
            } else {
                values[n / 2]
            };
        }

        validate_finite_output(&result)?;
        Ok(result)
    }

    fn byzantine_tolerance(&self) -> f32 {
        0.5
    }

    fn complexity(&self) -> AggregatorComplexity {
        AggregatorComplexity::Linear
    }
}

#[cfg(all(feature = "alloc", feature = "krum"))]
fn compute_pairwise_sq_distances_f64(updates: &[GradientUpdate], dim: usize) -> Result<Vec<f64>> {
    let n = updates.len();
    let nn = n.checked_mul(n).ok_or(Error::ResourceExhausted)?;
    let _ = nn.checked_mul(dim).ok_or(Error::ResourceExhausted)?;
    let mut distances = alloc::vec![0.0f64; nn];

    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist = 0.0f64;
            for (&a, &b) in updates[i].gradients.iter().zip(updates[j].gradients.iter()) {
                let diff = (a as f64) - (b as f64);
                dist += diff * diff;
            }
            if !dist.is_finite() {
                return Err(Error::InfDetected);
            }
            distances[i * n + j] = dist;
            distances[j * n + i] = dist;
        }
    }
    Ok(distances)
}

/// Krum and Multi-Krum aggregator (`ADR-0019`, `W9-01`).
///
/// When `num_selected == 1`, selects the single update minimizing the sum of squared
/// Euclidean distances to its `n - f - 2` closest neighbors.
/// When `num_selected > 1` (`Multi-Krum`), selects the top `num_selected` updates from
/// a single Krum scoring pass (`1 <= num_selected <= n - f - 2`) and returns their
/// coordinate-wise mean accumulated in `f64`.
#[cfg(feature = "krum")]
#[derive(Debug, Clone)]
pub struct Krum {
    /// Number of updates to select (`1` for standard Krum, `m > 1` for Multi-Krum).
    pub num_selected: usize,
    /// Expected number of Byzantine nodes (`f`).
    pub num_byzantine: usize,
}

#[cfg(feature = "krum")]
impl Krum {
    /// Create a standard single-winner Krum aggregator (`num_selected = 1`).
    pub const fn new(num_byzantine: usize) -> Self {
        Self {
            num_selected: 1,
            num_byzantine,
        }
    }

    /// Create a Multi-Krum aggregator selecting the top `num_selected` candidates in one pass.
    pub const fn multi(num_byzantine: usize, num_selected: usize) -> Self {
        Self {
            num_selected,
            num_byzantine,
        }
    }

    /// Fallible Multi-Krum constructor that validates `num_selected >= 1`.
    pub fn try_multi(num_byzantine: usize, num_selected: usize) -> Result<Self> {
        if num_selected == 0 {
            return Err(Error::InvalidThreshold);
        }
        Ok(Self::multi(num_byzantine, num_selected))
    }
}

#[cfg(feature = "krum")]
impl RobustAggregator for Krum {
    #[cfg(feature = "alloc")]
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>> {
        if self.num_selected == 0 {
            return Err(Error::InvalidThreshold);
        }
        let dim = validate_gradient_shapes_and_finiteness(updates)?;
        let n = updates.len();
        let f = self.num_byzantine;

        let min_peers = f
            .checked_mul(2)
            .and_then(|two_f| two_f.checked_add(3))
            .ok_or(Error::ResourceExhausted)?;
        if n < min_peers {
            return Err(Error::InsufficientUpdates);
        }
        if n > MAX_KRUM_PEERS {
            return Err(Error::ResourceExhausted);
        }

        let k = n - f - 2;
        if self.num_selected > k {
            return Err(Error::InvalidThreshold);
        }

        let distances = compute_pairwise_sq_distances_f64(updates, dim)?;
        let mut neighbor_dists = alloc::vec![0.0f64; n - 1];
        let mut scores: Vec<(usize, f64)> = Vec::with_capacity(n);

        for i in 0..n {
            let mut slot = 0usize;
            for j in 0..n {
                if i != j {
                    neighbor_dists[slot] = distances[i * n + j];
                    slot += 1;
                }
            }
            neighbor_dists.sort_by(f64::total_cmp);
            let score: f64 = neighbor_dists[..k].iter().copied().sum();
            if !score.is_finite() {
                return Err(Error::InfDetected);
            }
            scores.push((i, score));
        }

        scores.sort_by(|&(idx_a, score_a), &(idx_b, score_b)| {
            score_a
                .total_cmp(&score_b)
                .then_with(|| updates[idx_a].sender.cmp(&updates[idx_b].sender))
                .then_with(|| idx_a.cmp(&idx_b))
        });

        if self.num_selected == 1 {
            return Ok(updates[scores[0].0].gradients.clone());
        }

        let m = self.num_selected;
        let inv_m = 1.0f64 / (m as f64);
        let mut out = alloc::vec![0.0f32; dim];
        for (coord, slot) in out.iter_mut().enumerate().take(dim) {
            let mut sum = 0.0f64;
            for &(idx, _) in scores.iter().take(m) {
                sum += updates[idx].gradients[coord] as f64;
            }
            *slot = (sum * inv_m) as f32;
        }
        validate_finite_output(&out)?;
        Ok(out)
    }

    fn byzantine_tolerance(&self) -> f32 {
        0.33
    }

    fn complexity(&self) -> AggregatorComplexity {
        AggregatorComplexity::Quadratic
    }
}

/// Versioned Bulyan execution profile (`ADR-0019`, `W9-01`).
#[cfg(feature = "bulyan")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub enum BulyanProfile {
    /// El Mhamdi et al. (*Distributed Momentum for Byzantine-resilient Learning*, §2.2.3)
    /// specialized to `m = 1` ordinary Krum:
    /// - requires `f >= 1` and `n >= 4f + 3`,
    /// - selects `theta = n - 2f - 2` candidates via iterative fixed-`f` single-winner Krum
    ///   without replacement (`k = r - f - 2` at remaining pool size `r`),
    /// - averages the `beta = theta - 2f = n - 4f - 2` values closest to the coordinate median.
    #[default]
    IterativeKrumV1,
}

/// Stage 1 of [`BulyanProfile::IterativeKrumV1`]: selects `theta = n - 2f - 2` candidate indices
/// (in selection order, indexing into `updates`) via iterative fixed-`f` Krum without replacement.
#[cfg(all(feature = "alloc", feature = "bulyan"))]
pub fn bulyan_stage1_iterative_krum_v1(
    updates: &[GradientUpdate],
    max_byzantine: usize,
    budget: &AggregationBudget,
) -> Result<Vec<usize>> {
    if max_byzantine == 0 {
        return Err(Error::InvalidThreshold);
    }
    let min_peers = max_byzantine
        .checked_mul(4)
        .and_then(|four_f| four_f.checked_add(3))
        .ok_or(Error::ResourceExhausted)?;
    if updates.len() < min_peers {
        return Err(Error::InsufficientPeers);
    }

    let dim = validate_gradient_shapes_and_finiteness(updates)?;
    let n = updates.len();
    budget.check(n, dim, true)?;

    let distances = compute_pairwise_sq_distances_f64(updates, dim)?;
    let theta = n - 2 * max_byzantine - 2;
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut selected: Vec<usize> = Vec::with_capacity(theta);
    let mut neighbor_dists: Vec<f64> = Vec::with_capacity(n);

    for _ in 0..theta {
        let r = remaining.len();
        let k = r - max_byzantine - 2;

        let mut best_pos = 0usize;
        let mut best_orig = remaining[0];
        let mut best_score = f64::INFINITY;

        for (pos, &orig_i) in remaining.iter().enumerate() {
            neighbor_dists.clear();
            for &orig_j in &remaining {
                if orig_i != orig_j {
                    neighbor_dists.push(distances[orig_i * n + orig_j]);
                }
            }
            neighbor_dists.sort_by(f64::total_cmp);
            let score: f64 = neighbor_dists[..k].iter().copied().sum();
            if !score.is_finite() {
                return Err(Error::InfDetected);
            }

            let is_better = if pos == 0 {
                true
            } else {
                score
                    .total_cmp(&best_score)
                    .then_with(|| updates[orig_i].sender.cmp(&updates[best_orig].sender))
                    .then_with(|| orig_i.cmp(&best_orig))
                    .is_lt()
            };

            if is_better {
                best_pos = pos;
                best_orig = orig_i;
                best_score = score;
            }
        }

        selected.push(best_orig);
        remaining.remove(best_pos);
    }

    Ok(selected)
}

/// Internal Stage 2 kernel parameterized by candidate `(PeerId, &[f32])` pairs (`ADR-0019`, `ST-11`).
///
/// Uses the lower observed coordinate median `sorted[(theta - 1) / 2]` and breaks distance ties
/// strictly by `PeerId` (never by coordinate magnitude), preserving permutation invariance across
/// unique-peer cohorts. Because `sorted[(theta - 1) / 2]` selects the lower median on even `theta`
/// and ties break by `PeerId`, universal sign-flip equivariance `B(-x) == -B(x)` does not hold
/// for even-`theta` asymmetric or tied inputs (`ADR-0019` Q9 amendment).
#[cfg(all(feature = "alloc", feature = "bulyan"))]
fn bulyan_stage2_core<F>(
    theta: usize,
    dim: usize,
    max_byzantine: usize,
    mut coord_at: F,
) -> Result<Vec<f32>>
where
    F: FnMut(usize, usize) -> (PeerId, f32),
{
    let beta = theta - 2 * max_byzantine;
    let inv_beta = 1.0f64 / (beta as f64);
    let mut out = alloc::vec![0.0f32; dim];
    let mut sorted = alloc::vec![0.0f32; theta];
    let mut ranked: Vec<(PeerId, usize, f32, f64)> = Vec::with_capacity(theta);
    let lower_median_idx = (theta - 1) / 2;

    for (coord, slot) in out.iter_mut().enumerate().take(dim) {
        for (idx, elem) in sorted.iter_mut().enumerate().take(theta) {
            *elem = coord_at(idx, coord).1;
        }
        sorted.sort_by(f32::total_cmp);

        // Q9 ratified lower observed median (`sorted[(theta - 1) / 2]`), never arithmetic mean.
        let median = sorted[lower_median_idx] as f64;

        ranked.clear();
        for idx in 0..theta {
            let (peer, val) = coord_at(idx, coord);
            let dist = crate::num::abs_f64((val as f64) - median);
            ranked.push((peer, idx, val, dist));
        }

        ranked.sort_by(
            |&(peer_a, idx_a, _val_a, dist_a), &(peer_b, idx_b, _val_b, dist_b)| {
                dist_a
                    .total_cmp(&dist_b)
                    .then_with(|| peer_a.cmp(&peer_b))
                    .then_with(|| idx_a.cmp(&idx_b))
            },
        );

        let sum: f64 = ranked
            .iter()
            .take(beta)
            .map(|&(_, _, val, _)| val as f64)
            .sum();
        *slot = (sum * inv_beta) as f32;
    }

    validate_finite_output(&out)?;
    Ok(out)
}

#[cfg(all(feature = "alloc", feature = "bulyan"))]
fn validate_stage2_inputs<'a>(
    count: usize,
    max_byzantine: usize,
    first_len: Option<usize>,
    slices: impl Iterator<Item = &'a [f32]>,
) -> Result<usize> {
    if max_byzantine == 0 {
        return Err(Error::InvalidThreshold);
    }
    let min_theta = max_byzantine
        .checked_mul(2)
        .and_then(|two_f| two_f.checked_add(1))
        .ok_or(Error::ResourceExhausted)?;
    if count < min_theta {
        return Err(Error::InsufficientPeers);
    }
    let dim = first_len.unwrap_or(0);
    if dim == 0 || dim > MAX_GRADIENT_DIM {
        return Err(Error::InvalidGradient);
    }
    for slice in slices {
        if slice.len() != dim {
            return Err(Error::InvalidGradient);
        }
        for &val in slice {
            if val.is_nan() {
                return Err(Error::NanDetected);
            }
            if val.is_infinite() {
                return Err(Error::InfDetected);
            }
        }
    }
    Ok(dim)
}

/// Stage 2 of Bulyan (`ADR-0019`, `W9-01`, `ST-11`) preserving explicit `(PeerId, &[f32])` identity.
///
/// Computes the lower observed median `sorted[(theta - 1) / 2]` per coordinate and averages the
/// $\beta = \theta - 2f$ nearest values ranked by `(|v - M|, PeerId)`.
#[cfg(all(feature = "alloc", feature = "bulyan"))]
pub fn bulyan_stage2_with_peers(
    selected_updates: &[(PeerId, &[f32])],
    max_byzantine: usize,
) -> Result<Vec<f32>> {
    let dim = validate_stage2_inputs(
        selected_updates.len(),
        max_byzantine,
        selected_updates.first().map(|(_, s)| s.len()),
        selected_updates.iter().map(|(_, s)| *s),
    )?;
    bulyan_stage2_core(selected_updates.len(), dim, max_byzantine, |idx, coord| {
        (selected_updates[idx].0, selected_updates[idx].1[coord])
    })
}

/// Stage 2 of Bulyan (`ADR-0019`, `W9-01`, `ST-11`) over coordinate slices using caller order for ties.
///
/// Computes each coordinate's lower observed median `sorted[(theta - 1) / 2]` and averages the
/// $\beta = \theta - 2f$ values minimizing $|v - m_j|$ in `f64` (prefer [`bulyan_stage2_with_peers`]
/// when canonical `PeerId` permutation invariance under exact distance ties is required).
#[cfg(all(feature = "alloc", feature = "bulyan"))]
pub fn bulyan_stage2_nearest_to_median(
    selected_updates: &[&[f32]],
    max_byzantine: usize,
) -> Result<Vec<f32>> {
    let dim = validate_stage2_inputs(
        selected_updates.len(),
        max_byzantine,
        selected_updates.first().map(|s| s.len()),
        selected_updates.iter().copied(),
    )?;
    bulyan_stage2_core(selected_updates.len(), dim, max_byzantine, |idx, coord| {
        let mut key = [0u8; 32];
        key[24..32].copy_from_slice(&(idx as u64).to_be_bytes());
        (PeerId::new(key), selected_updates[idx][coord])
    })
}

/// Bulyan Byzantine-resilient aggregator (`ADR-0019`, `W9-01`).
#[cfg(feature = "bulyan")]
#[derive(Debug, Clone)]
pub struct Bulyan {
    /// Maximum tolerated Byzantine peers (`f >= 1`).
    pub max_byzantine: usize,
    /// Versioned execution profile.
    pub profile: BulyanProfile,
    /// Checked resource/work budget.
    pub budget: AggregationBudget,
}

#[cfg(feature = "bulyan")]
impl Bulyan {
    /// Create a new Bulyan aggregator using [`BulyanProfile::IterativeKrumV1`] and the default budget.
    pub const fn new(max_byzantine: usize) -> Self {
        Self {
            max_byzantine,
            profile: BulyanProfile::IterativeKrumV1,
            budget: AggregationBudget::default_pairwise(),
        }
    }

    /// Fallible constructor that rejects `max_byzantine == 0`.
    pub fn try_new(max_byzantine: usize, profile: BulyanProfile) -> Result<Self> {
        if max_byzantine == 0 {
            return Err(Error::InvalidThreshold);
        }
        Ok(Self {
            max_byzantine,
            profile,
            budget: AggregationBudget::default_pairwise(),
        })
    }

    /// Override the aggregation resource budget.
    pub const fn with_budget(mut self, budget: AggregationBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Run Stage 1 and return the selected original peer indices in selection order.
    #[cfg(feature = "alloc")]
    pub fn select_indices(&self, updates: &[GradientUpdate]) -> Result<Vec<usize>> {
        match self.profile {
            BulyanProfile::IterativeKrumV1 => {
                bulyan_stage1_iterative_krum_v1(updates, self.max_byzantine, &self.budget)
            }
        }
    }
}

#[cfg(feature = "bulyan")]
impl RobustAggregator for Bulyan {
    #[cfg(feature = "alloc")]
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>> {
        let selected_indices = self.select_indices(updates)?;
        let selected_with_peers: Vec<(PeerId, &[f32])> = selected_indices
            .iter()
            .map(|&idx| {
                (
                    PeerId::new(updates[idx].sender),
                    updates[idx].gradients.as_slice(),
                )
            })
            .collect();
        bulyan_stage2_with_peers(&selected_with_peers, self.max_byzantine)
    }

    fn byzantine_tolerance(&self) -> f32 {
        0.25
    }

    fn complexity(&self) -> AggregatorComplexity {
        AggregatorComplexity::Cubic
    }
}

/// Maximum inner refinement iterations allowed for [`CenteredClipping`] (`ADR-0019` §4, `W11-01`).
pub const MAX_CCLIP_ITERATIONS: usize = 32;

/// Domain separation tag for deterministic `$s$-Bucketing` cohort permutation (`ADR-0019` §4, `W11-01`).
pub const S_BUCKETING_PERM_DOMAIN_V1: &[u8] = b"swarmtorch.sbucketing.perm.v1";

/// Domain separation tag for synthetic bucket provenance identifiers (`ADR-0019` §4, `W11-01`).
pub const S_BUCKETING_ID_DOMAIN_V1: &[u8] = b"swarmtorch.bucket.id.v1";

/// Stateful / iterative Centered Clipping (`CClip`) aggregator
/// (Karimireddy et al., ICML 2021; `ADR-0019` §4, `W11-01`).
///
/// Computes $L = \text{max\_iterations}$ fixed-point refinement steps starting from the previous
/// round's center $v_0$:
/// $$v_{\ell+1} = v_\ell + \frac{1}{n} \sum_{i=1}^n (x_i - v_\ell) \min\!\left(1, \frac{\tau}{\|x_i - v_\ell\|_2}\right)$$
/// with widened `f64` norm and coordinate accumulation.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct CenteredClipping {
    /// Strictly positive finite clipping radius ($\tau > 0$).
    pub tau: f32,
    /// Number of inner refinement iterations ($1 \le L \le 32$).
    pub max_iterations: usize,
    /// Default center $v_0$ used when invoked via the stateless [`RobustAggregator::aggregate`]
    /// trait method. If empty, defaults to $\mathbf{0}^d$.
    pub initial_center: Vec<f32>,
}

#[cfg(feature = "alloc")]
impl CenteredClipping {
    /// Fallible constructor enforcing all normative `ADR-0019` §4 domain invariants.
    pub fn try_new(tau: f32, max_iterations: usize, initial_center: Vec<f32>) -> Result<Self> {
        if tau.is_nan() {
            return Err(Error::NanDetected);
        }
        if tau.is_infinite() {
            return Err(Error::InfDetected);
        }
        if tau <= 0.0 || max_iterations == 0 || max_iterations > MAX_CCLIP_ITERATIONS {
            return Err(Error::InvalidThreshold);
        }
        for &c in &initial_center {
            if c.is_nan() {
                return Err(Error::NanDetected);
            }
            if c.is_infinite() {
                return Err(Error::InfDetected);
            }
        }
        Ok(Self {
            tau,
            max_iterations,
            initial_center,
        })
    }

    /// Create a `CenteredClipping` aggregator with a zero default center ($\mathbf{0}^d$).
    ///
    /// # Panics
    ///
    /// Panics if `tau` is non-finite or $\le 0.0$, or if `max_iterations` is outside `1..=32`.
    pub fn new(tau: f32, max_iterations: usize) -> Self {
        Self::try_new(tau, max_iterations, Vec::new())
            .expect("CenteredClipping requires finite tau > 0 and 1 <= max_iterations <= 32")
    }

    /// Attach an explicit initial center $v_0$.
    pub fn with_center(mut self, initial_center: Vec<f32>) -> Result<Self> {
        for &c in &initial_center {
            if c.is_nan() {
                return Err(Error::NanDetected);
            }
            if c.is_infinite() {
                return Err(Error::InfDetected);
            }
        }
        self.initial_center = initial_center;
        Ok(self)
    }

    fn validate_config(&self) -> Result<()> {
        if self.tau.is_nan() {
            return Err(Error::NanDetected);
        }
        if self.tau.is_infinite() {
            return Err(Error::InfDetected);
        }
        if self.tau <= 0.0 || self.max_iterations == 0 || self.max_iterations > MAX_CCLIP_ITERATIONS
        {
            return Err(Error::InvalidThreshold);
        }
        Ok(())
    }

    /// Execute $L$-step centered clipping around an explicit reference center $v_0$ (`center`)
    /// with bounded work validation against `budget`.
    pub fn aggregate_with_center_bounded(
        &self,
        updates: &[GradientUpdate],
        center: &[f32],
        budget: &AggregationBudget,
    ) -> Result<Vec<f32>> {
        self.validate_config()?;
        let dim = validate_gradient_shapes_and_finiteness(updates)?;
        if center.len() != dim {
            return Err(Error::InvalidGradient);
        }
        for &c in center {
            if c.is_nan() {
                return Err(Error::NanDetected);
            }
            if c.is_infinite() {
                return Err(Error::InfDetected);
            }
        }

        let n = updates.len();
        if n > budget.max_peers || dim > budget.max_dimension {
            return Err(Error::ResourceBudgetExceeded);
        }
        // Workspace is O(d) f64 (two d-vectors: v and delta_sum).
        let scratch_bytes = dim
            .checked_mul(2)
            .and_then(|two_d| two_d.checked_mul(core::mem::size_of::<f64>()))
            .ok_or(Error::ResourceExhausted)?;
        if scratch_bytes > budget.max_scratch_bytes {
            return Err(Error::ResourceBudgetExceeded);
        }
        let work_ops = self
            .max_iterations
            .checked_mul(n)
            .and_then(|ln| ln.checked_mul(dim))
            .ok_or(Error::ResourceExhausted)?;
        if work_ops > budget.max_work_ops {
            return Err(Error::ResourceBudgetExceeded);
        }

        let tau_f64 = self.tau as f64;
        let inv_n = 1.0f64 / (n as f64);
        let mut v: Vec<f64> = center.iter().map(|&c| c as f64).collect();
        let mut delta_sum: Vec<f64> = alloc::vec![0.0f64; dim];

        for _ in 0..self.max_iterations {
            delta_sum.fill(0.0);

            for update in updates {
                let mut dist_sq = 0.0f64;
                for (&x_j, &v_j) in update.gradients.iter().zip(v.iter()) {
                    let diff = (x_j as f64) - v_j;
                    dist_sq += diff * diff;
                }
                if dist_sq.is_nan() {
                    return Err(Error::NanDetected);
                }
                if dist_sq.is_infinite() {
                    return Err(Error::InfDetected);
                }

                let dist = crate::num::sqrt_f64(dist_sq);
                // Explicit zero-norm and within-radius branch: no division when dist <= tau.
                let scale = if dist == 0.0 || dist <= tau_f64 {
                    1.0f64
                } else {
                    tau_f64 / dist
                };

                for ((slot, &x_j), &v_j) in delta_sum
                    .iter_mut()
                    .zip(update.gradients.iter())
                    .zip(v.iter())
                {
                    *slot += ((x_j as f64) - v_j) * scale;
                }
            }

            for (v_j, &d_j) in v.iter_mut().zip(delta_sum.iter()) {
                *v_j += d_j * inv_n;
                if v_j.is_nan() {
                    return Err(Error::NanDetected);
                }
                if v_j.is_infinite() {
                    return Err(Error::InfDetected);
                }
            }
        }

        let out: Vec<f32> = v.into_iter().map(|val| val as f32).collect();
        validate_finite_output(&out)?;
        Ok(out)
    }

    /// Execute $L$-step centered clipping around an explicit reference center $v_0$ (`center`)
    /// using the default aggregation budget.
    pub fn aggregate_with_center(
        &self,
        updates: &[GradientUpdate],
        center: &[f32],
    ) -> Result<Vec<f32>> {
        let budget = AggregationBudget {
            max_peers: 10_000,
            max_dimension: MAX_GRADIENT_DIM,
            max_scratch_bytes: 256 * 1024 * 1024,
            max_work_ops: 500_000_000,
        };
        self.aggregate_with_center_bounded(updates, center, &budget)
    }
}

#[cfg(feature = "alloc")]
impl RobustAggregator for CenteredClipping {
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>> {
        self.validate_config()?;
        let dim = validate_gradient_shapes_and_finiteness(updates)?;
        if self.initial_center.is_empty() {
            let zero_center = alloc::vec![0.0f32; dim];
            self.aggregate_with_center(updates, &zero_center)
        } else {
            self.aggregate_with_center(updates, &self.initial_center)
        }
    }

    fn byzantine_tolerance(&self) -> f32 {
        0.5
    }

    fn complexity(&self) -> AggregatorComplexity {
        AggregatorComplexity::Linear
    }
}

/// Logical identifier for an `$s$-Bucketing` pre-aggregation bucket (`ADR-0019` §4, `W11-01`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct BucketId(pub u32);

/// Remainder handling policy when cohort size $n$ is not an exact multiple of bucket size $s$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub enum BucketRemainderPolicy {
    /// Strict contract (`ADR-0019` §4 default): require `n % s == 0` and reject otherwise.
    #[default]
    StrictExactMultiple,
    /// Deterministically shuffle the canonical cohort, form $\lfloor n / s \rfloor$ full buckets
    /// of exact size $s$, and drop the remaining $n \bmod s$ tail peers (recording them in
    /// [`BucketExecutionReport::dropped_remainder_peers`]).
    DropRemainderDeterministic,
}

/// Configuration for conservative `$s$-Bucketing` pre-aggregation (Karimireddy et al., ICLR 2022;
/// `ADR-0019` §4, `W11-01`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct BucketingConfig {
    /// Number of peers averaged per bucket ($s \ge 1$).
    pub bucket_size: usize,
    /// Maximum tolerated Byzantine peers in the pre-bucketing cohort ($f$).
    pub byzantine_count: usize,
    /// Deterministic round seed binding the bucket permutation.
    pub round_seed: [u8; 32],
    /// Policy when $n \bmod s \ne 0$.
    pub remainder_policy: BucketRemainderPolicy,
}

/// Provenance record for a single pre-aggregation bucket (`ADR-0019` §4, `W11-01`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct BucketProvenance {
    /// Logical bucket index.
    pub bucket_id: BucketId,
    /// Authenticated peer IDs assigned to this bucket.
    pub members: Vec<PeerId>,
    /// Uniform intra-bucket weight ($1 / s$) applied to every member.
    pub uniform_weight: f64,
}

/// Audit report emitted by `$s$-Bucketing` pre-aggregation (`ADR-0019` §4, `W11-01`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct BucketExecutionReport {
    /// Number of buckets formed ($B = \lfloor n / s \rfloor$).
    pub num_buckets: usize,
    /// Bucket size ($s$).
    pub bucket_size: usize,
    /// Conservative post-bucketing Byzantine bound $f_B = \min(f, B)$.
    pub effective_byzantine_bound: usize,
    /// Per-bucket membership provenance.
    pub buckets: Vec<BucketProvenance>,
    /// Peers dropped when [`BucketRemainderPolicy::DropRemainderDeterministic`] is active.
    pub dropped_remainder_peers: Vec<PeerId>,
}

/// Downstream aggregation rule applied to bucket means after `$s$-Bucketing` (`ADR-0019` §4).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub enum PostBucketingRule {
    /// Single-winner Krum ($B \ge 2f_B + 3$).
    #[cfg(feature = "krum")]
    Krum,
    /// Multi-Krum ($B \ge 2f_B + 3$ and $1 \le m \le B - f_B - 2$).
    #[cfg(feature = "krum")]
    MultiKrum { m: usize },
    /// Iterative Krum + Trimmed Median Bulyan ($f_B \ge 1$ and $B \ge 4f_B + 3$).
    #[cfg(feature = "bulyan")]
    Bulyan,
    /// Coordinate-wise Trimmed Mean ($\lfloor B \cdot \text{trim\_ratio} \rfloor \ge f_B$).
    TrimmedMean { trim_ratio: f32 },
    /// Coordinate-wise Median ($B > 2f_B$).
    CoordinateMedian,
    /// Stateful / iterative Centered Clipping ($B > 2f_B$).
    CenteredClipping { tau: f32, max_iterations: usize },
}

#[cfg(feature = "alloc")]
struct DeterministicSha256WordStream {
    domain: &'static [u8],
    seed: [u8; 32],
    param_a: u64,
    param_b: u64,
    counter: u64,
    block: [u8; 32],
    offset: usize,
}

#[cfg(feature = "alloc")]
impl DeterministicSha256WordStream {
    fn new(domain: &'static [u8], seed: [u8; 32], param_a: u64, param_b: u64) -> Self {
        Self {
            domain,
            seed,
            param_a,
            param_b,
            counter: 0,
            block: [0u8; 32],
            offset: 32,
        }
    }

    fn next_u32(&mut self) -> u32 {
        if self.offset + 4 > 32 {
            let mut hasher = Sha256::new();
            hasher.update(self.domain);
            hasher.update(self.seed);
            hasher.update(self.param_a.to_le_bytes());
            hasher.update(self.param_b.to_le_bytes());
            hasher.update(self.counter.to_le_bytes());
            self.counter = self.counter.wrapping_add(1);
            let digest = hasher.finalize();
            self.block.copy_from_slice(&digest);
            self.offset = 0;
        }
        let word = u32::from_le_bytes([
            self.block[self.offset],
            self.block[self.offset + 1],
            self.block[self.offset + 2],
            self.block[self.offset + 3],
        ]);
        self.offset += 4;
        word
    }

    /// Unbiased integer in `0..bound` ($bound \ge 1$) via Lemire's divisionless rejection method.
    fn uniform_u32_bounded(&mut self, bound: u32) -> u32 {
        debug_assert!(bound >= 1);
        if bound == 1 {
            return 0;
        }
        let s = bound as u64;
        let threshold = s.wrapping_neg() % s;
        loop {
            let x = self.next_u32() as u64;
            let m = x * s;
            let l = m as u32 as u64;
            if l >= threshold {
                return (m >> 32) as u32;
            }
        }
    }
}

/// Validate that the post-bucketing bucket count $B$ satisfies the downstream aggregator's
/// quorum bound under the conservative worst-case fault bound $f_B = \min(f, B)$ (`ADR-0019` §4).
pub fn validate_post_bucketing_quorum(
    num_buckets: usize,
    effective_f: usize,
    rule: PostBucketingRule,
) -> Result<()> {
    if num_buckets == 0 {
        return Err(Error::InsufficientPeers);
    }
    match rule {
        #[cfg(feature = "krum")]
        PostBucketingRule::Krum => {
            let min_b = effective_f
                .checked_mul(2)
                .and_then(|v| v.checked_add(3))
                .ok_or(Error::ResourceExhausted)?;
            if num_buckets < min_b {
                return Err(Error::InsufficientPeers);
            }
            Ok(())
        }
        #[cfg(feature = "krum")]
        PostBucketingRule::MultiKrum { m } => {
            if m == 0 {
                return Err(Error::InvalidThreshold);
            }
            let min_b = effective_f
                .checked_mul(2)
                .and_then(|v| v.checked_add(3))
                .ok_or(Error::ResourceExhausted)?;
            if num_buckets < min_b {
                return Err(Error::InsufficientPeers);
            }
            let max_m = num_buckets - effective_f - 2;
            if m > max_m {
                return Err(Error::InsufficientPeers);
            }
            Ok(())
        }
        #[cfg(feature = "bulyan")]
        PostBucketingRule::Bulyan => {
            if effective_f == 0 {
                return Err(Error::InvalidThreshold);
            }
            let min_b = effective_f
                .checked_mul(4)
                .and_then(|v| v.checked_add(3))
                .ok_or(Error::ResourceExhausted)?;
            if num_buckets < min_b {
                return Err(Error::InsufficientPeers);
            }
            Ok(())
        }
        PostBucketingRule::TrimmedMean { trim_ratio } => {
            if !trim_ratio.is_finite() || !(0.0..0.5).contains(&trim_ratio) {
                return Err(Error::InvalidThreshold);
            }
            let trim_count = ((num_buckets as f32) * trim_ratio) as usize;
            let two_trim = trim_count.checked_mul(2).ok_or(Error::ResourceExhausted)?;
            if trim_count < effective_f || two_trim >= num_buckets {
                return Err(Error::InsufficientPeers);
            }
            Ok(())
        }
        PostBucketingRule::CoordinateMedian => {
            if effective_f > 0 {
                let min_b = effective_f
                    .checked_mul(2)
                    .and_then(|v| v.checked_add(1))
                    .ok_or(Error::ResourceExhausted)?;
                if num_buckets < min_b {
                    return Err(Error::InsufficientPeers);
                }
            }
            Ok(())
        }
        PostBucketingRule::CenteredClipping {
            tau,
            max_iterations,
        } => {
            if tau.is_nan() {
                return Err(Error::NanDetected);
            }
            if tau.is_infinite() {
                return Err(Error::InfDetected);
            }
            if tau <= 0.0 || max_iterations == 0 || max_iterations > MAX_CCLIP_ITERATIONS {
                return Err(Error::InvalidThreshold);
            }
            if effective_f > 0 {
                let min_b = effective_f
                    .checked_mul(2)
                    .and_then(|v| v.checked_add(1))
                    .ok_or(Error::ResourceExhausted)?;
                if num_buckets < min_b {
                    return Err(Error::InsufficientPeers);
                }
            }
            Ok(())
        }
    }
}

/// Partition authenticated peer updates into $B = \lfloor n / s \rfloor$ uniform-weight buckets
/// of exact size $s$ using a canonical `PeerId`-ordered deterministic shuffle (`ADR-0019` §4, `W11-01`).
///
/// Key security properties (`F-02`):
/// 1. Input updates are canonically ordered by `(PeerId, original_index)` before applying the
///    `round_seed` permutation, making bucket membership strictly invariant to network arrival order.
/// 2. Every peer inside a bucket receives exact uniform weight $1 / s$; self-reported
///    `num_samples` on the wire are ignored so a Byzantine peer cannot hijack a bucket mean by
///    claiming `u32::MAX` training examples.
/// 3. The post-bucketing Byzantine bound is conservatively set to $f_B = \min(f, B)$.
#[cfg(feature = "alloc")]
pub fn bucket_updates_v1(
    updates: &[GradientUpdate],
    config: &BucketingConfig,
) -> Result<(Vec<GradientUpdate>, BucketExecutionReport)> {
    if config.bucket_size == 0 {
        return Err(Error::InvalidThreshold);
    }
    let dim = validate_gradient_shapes_and_finiteness(updates)?;
    let n = updates.len();
    let s = config.bucket_size;

    if n < s {
        return Err(Error::InsufficientPeers);
    }
    if n > u32::MAX as usize {
        return Err(Error::ResourceExhausted);
    }
    if config.remainder_policy == BucketRemainderPolicy::StrictExactMultiple && (n % s != 0) {
        return Err(Error::InvalidGradient);
    }

    // 1. Canonical sort by PeerId (with index tie-break) for arrival-order invariance.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        updates[a]
            .sender
            .cmp(&updates[b].sender)
            .then_with(|| a.cmp(&b))
    });

    // 2. Deterministic Fisher-Yates shuffle seeded by (S_BUCKETING_PERM_DOMAIN_V1, round_seed, n, s).
    let mut rng = DeterministicSha256WordStream::new(
        S_BUCKETING_PERM_DOMAIN_V1,
        config.round_seed,
        n as u64,
        s as u64,
    );
    for i in (1..n).rev() {
        let j = rng.uniform_u32_bounded((i + 1) as u32) as usize;
        order.swap(i, j);
    }

    let num_buckets = n / s;
    let used_peers = num_buckets * s;
    let inv_s = 1.0f64 / (s as f64);
    let mut bucket_updates = Vec::with_capacity(num_buckets);
    let mut buckets_meta = Vec::with_capacity(num_buckets);

    for b in 0..num_buckets {
        let slice = &order[(b * s)..((b + 1) * s)];
        let mut mean = alloc::vec![0.0f32; dim];
        let mut members = Vec::with_capacity(s);

        for &idx in slice {
            members.push(PeerId::new(updates[idx].sender));
        }

        for (coord, slot) in mean.iter_mut().enumerate().take(dim) {
            let mut sum = 0.0f64;
            for &idx in slice {
                sum += updates[idx].gradients[coord] as f64;
            }
            *slot = (sum * inv_s) as f32;
        }
        validate_finite_output(&mean)?;

        let mut id_hasher = Sha256::new();
        id_hasher.update(S_BUCKETING_ID_DOMAIN_V1);
        id_hasher.update(config.round_seed);
        id_hasher.update((b as u32).to_le_bytes());
        let id_digest = id_hasher.finalize();
        let mut pseudo_peer = [0u8; 32];
        pseudo_peer.copy_from_slice(&id_digest);

        bucket_updates.push(GradientUpdate {
            sender: pseudo_peer,
            sequence: b as u64,
            gradients: mean,
            round_id: updates[slice[0]].round_id,
        });

        buckets_meta.push(BucketProvenance {
            bucket_id: BucketId(b as u32),
            members,
            uniform_weight: inv_s,
        });
    }

    let dropped_remainder_peers: Vec<PeerId> = order[used_peers..]
        .iter()
        .map(|&idx| PeerId::new(updates[idx].sender))
        .collect();

    let effective_byzantine_bound = config.byzantine_count.min(num_buckets);
    let report = BucketExecutionReport {
        num_buckets,
        bucket_size: s,
        effective_byzantine_bound,
        buckets: buckets_meta,
        dropped_remainder_peers,
    };

    Ok((bucket_updates, report))
}

/// Execute conservative `$s$-Bucketing` followed by quorum-verified downstream robust aggregation
/// (`ADR-0019` §4, `W11-01`).
#[cfg(feature = "alloc")]
pub fn aggregate_with_bucketing_v1(
    updates: &[GradientUpdate],
    config: &BucketingConfig,
    rule: PostBucketingRule,
    cclip_center: Option<&[f32]>,
    budget: &AggregationBudget,
) -> Result<(Vec<f32>, BucketExecutionReport)> {
    let (bucket_updates, report) = bucket_updates_v1(updates, config)?;
    let f_b = report.effective_byzantine_bound;
    validate_post_bucketing_quorum(report.num_buckets, f_b, rule)?;

    let dim = bucket_updates[0].gradients.len();
    let aggregated = match rule {
        #[cfg(feature = "krum")]
        PostBucketingRule::Krum => {
            budget.check(report.num_buckets, dim, true)?;
            Krum::new(f_b).aggregate(&bucket_updates)?
        }
        #[cfg(feature = "krum")]
        PostBucketingRule::MultiKrum { m } => {
            budget.check(report.num_buckets, dim, true)?;
            Krum::try_multi(f_b, m)?.aggregate(&bucket_updates)?
        }
        #[cfg(feature = "bulyan")]
        PostBucketingRule::Bulyan => Bulyan::try_new(f_b, BulyanProfile::IterativeKrumV1)?
            .with_budget(*budget)
            .aggregate(&bucket_updates)?,
        PostBucketingRule::TrimmedMean { trim_ratio } => {
            budget.check(report.num_buckets, dim, false)?;
            TrimmedMean::try_new(trim_ratio)?.aggregate(&bucket_updates)?
        }
        PostBucketingRule::CoordinateMedian => {
            budget.check(report.num_buckets, dim, false)?;
            CoordinateMedian.aggregate(&bucket_updates)?
        }
        PostBucketingRule::CenteredClipping {
            tau,
            max_iterations,
        } => {
            let cclip = CenteredClipping::try_new(tau, max_iterations, Vec::new())?;
            if let Some(center) = cclip_center {
                cclip.aggregate_with_center_bounded(&bucket_updates, center, budget)?
            } else {
                let zero_center = alloc::vec![0.0f32; dim];
                cclip.aggregate_with_center_bounded(&bucket_updates, &zero_center, budget)?
            }
        }
    };

    Ok((aggregated, report))
}

/// Execute robust aggregation directly in the $k$-dimensional shared-seed subspace
/// (`ADR-0021` §3, `W11-02`).
///
/// Because every peer projects onto the identical ascending coordinate subset
/// $S = \text{sample\_shared\_indices\_v1}(d, k, \text{seed})$, pairwise distances and
/// coordinate-wise order statistics on the $d$-dimensional sparse vectors are preserved
/// identically on the $k$-dimensional dense value slices. This reduces Krum/Multi-Krum/Bulyan
/// work from $\mathcal{O}(n^2 d)$ to $\mathcal{O}(n^2 k)$ and `CenteredClipping` work from
/// $\mathcal{O}(Lnd)$ to $\mathcal{O}(Lnk)$ without allocating $n$ full $d$-vectors.
#[cfg(feature = "alloc")]
pub fn aggregate_shared_seed_sparse_v1(
    submissions: &[(PeerId, SharedSeedSparsePayloadV1)],
    aggregator: &dyn RobustAggregator,
    budget: &AggregationBudget,
) -> Result<Vec<f32>> {
    if submissions.is_empty() {
        return Err(Error::InsufficientUpdates);
    }

    let first = &submissions[0].1;
    let d = first.original_len as usize;
    let k = first.k as usize;
    if d == 0 || d > MAX_GRADIENT_DIM || k == 0 || k > d {
        return Err(Error::InvalidGradient);
    }
    if d > budget.max_dimension {
        return Err(Error::ResourceBudgetExceeded);
    }

    let indices =
        sample_shared_indices_v1(d, k, &first.seed).map_err(|_| Error::InvalidGradient)?;
    let n = submissions.len();
    let pairwise = matches!(
        aggregator.complexity(),
        AggregatorComplexity::Quadratic | AggregatorComplexity::Cubic
    );
    // Budget check uses compressed dimension k for work_ops and scratch_bytes!
    budget.check(n, k, pairwise)?;

    let mut subspace_updates = Vec::with_capacity(n);
    for (idx, (peer, payload)) in submissions.iter().enumerate() {
        if payload.original_len != first.original_len
            || payload.k != first.k
            || payload.seed != first.seed
            || payload.scaling != first.scaling
            || payload.values.len() != k
        {
            return Err(Error::InvalidGradient);
        }
        for &val in &payload.values {
            if val.is_nan() {
                return Err(Error::NanDetected);
            }
            if val.is_infinite() {
                return Err(Error::InfDetected);
            }
        }
        subspace_updates.push(GradientUpdate {
            sender: *peer.as_bytes(),
            sequence: idx as u64,
            gradients: payload.values.clone(),
            round_id: 0,
        });
    }

    let subspace_aggregate = aggregator.aggregate(&subspace_updates)?;
    if subspace_aggregate.len() != k {
        return Err(Error::AggregationFailed);
    }

    let mut dense_out = alloc::vec![0.0f32; d];
    for (coord_idx, &val) in indices.into_iter().zip(subspace_aggregate.iter()) {
        dense_out[coord_idx as usize] = val;
    }
    validate_finite_output(&dense_out)?;
    Ok(dense_out)
}

/// Verify each peer's pre-seed [`PreSeedVectorCommitmentV1`] and [`MerkleOpeningBatchV1`] against
/// `expected_round_seed` before executing $\mathcal{O}(n^2 k)$ / $\mathcal{O}(Lnk)$ compressed-subspace
/// robust aggregation (`ADR-0021` §3, `W11-02`).
#[cfg(feature = "alloc")]
pub fn aggregate_verifiable_shared_seed_sparse_v1(
    submissions: &[(
        PeerId,
        PreSeedVectorCommitmentV1,
        SharedSeedSparsePayloadV1,
        MerkleOpeningBatchV1,
    )],
    expected_round_seed: &[u8; 32],
    aggregator: &dyn RobustAggregator,
    budget: &AggregationBudget,
) -> Result<Vec<f32>> {
    if submissions.is_empty() {
        return Err(Error::InsufficientUpdates);
    }

    let mut verified_payloads = Vec::with_capacity(submissions.len());
    for (peer, commitment, payload, openings) in submissions {
        if &payload.seed != expected_round_seed {
            return Err(Error::VerificationFailed);
        }
        verify_shared_seed_openings_v1(commitment, payload, openings).map_err(|err| match err {
            CompressionError::NonFiniteValue => Error::NanDetected,
            _ => Error::VerificationFailed,
        })?;
        verified_payloads.push((*peer, payload.clone()));
    }

    aggregate_shared_seed_sparse_v1(&verified_payloads, aggregator, budget)
}

/// Wave 12 (`W12-03` / `ADR-0023` §4.1) descriptor for bounded-staleness asynchronous aggregation
/// (`FedAsync` polynomial decay mixing profile).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct FedAsyncStalenessDescriptorV1 {
    /// Base mixing step size $\alpha_0 \in (0, 1]$.
    pub base_alpha: f32,
    /// Maximum admitted round staleness $\tau_{\max} = r_{\text{current}} - r_{\text{base}}$.
    pub max_staleness_rounds: u64,
    /// Polynomial decay exponent $a \ge 0$ for $s(\tau) = (1 + \tau)^{-a}$.
    pub polynomial_decay_exponent: f32,
}

impl FedAsyncStalenessDescriptorV1 {
    /// Validate policy hyperparameters before admitting asynchronous updates.
    pub fn validate(&self) -> Result<()> {
        if !self.base_alpha.is_finite()
            || self.base_alpha <= 0.0
            || self.base_alpha > 1.0
            || !self.polynomial_decay_exponent.is_finite()
            || self.polynomial_decay_exponent < 0.0
        {
            return Err(Error::InvalidThreshold);
        }
        Ok(())
    }

    /// Compute the effective mixing weight $\alpha(\tau)$ with an optional hinge threshold
    /// $\tau_{\text{hinge}}$ (`Xie et al. 2019`):
    /// - For $\tau \le \tau_{\text{hinge}}$, $\alpha(\tau) = \alpha_0$.
    /// - For $\tau > \tau_{\text{hinge}}$, $\alpha(\tau) = \alpha_0 (1 + (\tau - \tau_{\text{hinge}}))^{-a}$.
    pub fn compute_hinge_mixing_alpha(
        &self,
        base_round_id: u64,
        current_round_id: u64,
        hinge_threshold_rounds: u64,
    ) -> Result<f32> {
        self.validate()?;
        if base_round_id > current_round_id {
            return Err(Error::InvalidThreshold);
        }
        let tau = current_round_id - base_round_id;
        if tau > self.max_staleness_rounds {
            return Err(Error::InvalidThreshold);
        }
        if tau <= hinge_threshold_rounds || self.polynomial_decay_exponent == 0.0 {
            return Ok(self.base_alpha);
        }
        let excess_tau = (tau - hinge_threshold_rounds) as f64;
        let decay = crate::num::exp_f64_soft(
            -(self.polynomial_decay_exponent as f64) * crate::num::ln_f64_soft(1.0 + excess_tau),
        );
        let alpha = ((self.base_alpha as f64) * decay) as f32;
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(Error::InvalidThreshold);
        }
        Ok(alpha)
    }
}

/// Wave 12 (`W12-03`) trait for staleness-weighted asynchronous model mixing.
pub trait StalenessWeightingPolicy {
    /// Compute the effective mixing weight $\alpha(\tau) \in [0, 1]$ for an update trained against
    /// `base_round_id` arriving at `current_round_id`, or `Err(Error::InvalidThreshold)` if
    /// staleness exceeds policy bounds.
    fn compute_mixing_alpha(&self, base_round_id: u64, current_round_id: u64) -> Result<f32>;
}

impl StalenessWeightingPolicy for FedAsyncStalenessDescriptorV1 {
    fn compute_mixing_alpha(&self, base_round_id: u64, current_round_id: u64) -> Result<f32> {
        self.compute_hinge_mixing_alpha(base_round_id, current_round_id, 0)
    }
}

/// Audit and telemetry receipt emitted by [`apply_fedasync_stale_delta_v1`] (`ADR-0023` §4.1).
#[derive(Debug, Clone, PartialEq)]
pub struct FedAsyncStaleMixOutcomeV1 {
    /// Updated server model parameters $w_{\text{new}}$.
    pub updated_weights: Vec<f32>,
    /// Effective staleness-decayed mixing weight $\alpha(\tau) \in (0, 1]$.
    pub effective_alpha: f32,
    /// Staleness lag $\tau = r_{\text{current}} - r_{\text{base}}$.
    pub staleness_rounds: u64,
    /// Pre-clipping $\ell_2$ norm $\|\Delta_{\text{stale}}\|_2$ in `f64`.
    pub raw_delta_l2_norm: f64,
    /// Post-clipping $\ell_2$ norm $\|\Delta_{\text{clipped}}\|_2$ in `f64`.
    pub clipped_delta_l2_norm: f64,
    /// Server base-drift $\ell_2$ norm $\|w_{\text{base}} - w_{\text{current}}\|_2$ in `f64`.
    pub base_drift_l2_norm: f64,
    /// Whether the incoming stale delta exceeded `max_delta_norm_tau` and was $\ell_2$-clipped.
    pub was_clipped: bool,
}

/// Apply a bounded-staleness asynchronous client delta $\Delta_i = w_i^{\text{local}} - w_{t_{\text{base}}}$
/// to the current server model $w_{t_{\text{cur}}}$ with exact base-drift compensation (`ADR-0023` §4.1):
///
/// $$w_{\text{new}} = (1 - \alpha(\tau)) w_{t_{\text{cur}}} + \alpha(\tau) \left(w_{t_{\text{base}}} + \Delta_{\text{clipped}}\right) = w_{t_{\text{cur}}} + \alpha(\tau) \left(w_{t_{\text{base}}} - w_{t_{\text{cur}}} + \Delta_{\text{clipped}}\right)$$
///
/// ### Security & Numerical Invariants
/// 1. All three input vectors (`w_current`, `w_base`, `stale_delta`) must have identical non-zero length
///    and contain strictly finite `f32` elements (`NaN` $\to$ [`Error::NanDetected`], `Inf` $\to$ [`Error::InfDetected`]).
/// 2. Optional $\ell_2$ norm clipping (`max_delta_norm_tau`) is applied to `stale_delta` **before** adding
///    the server drift term $(w_{t_{\text{base}}} - w_{t_{\text{cur}}})$ so a Byzantine contributor cannot
///    exploit model drift to smuggle an unbounded update.
#[allow(clippy::too_many_arguments)]
pub fn apply_fedasync_stale_delta_v1<P: StalenessWeightingPolicy + ?Sized>(
    policy: &P,
    base_round_id: u64,
    current_round_id: u64,
    w_current: &[f32],
    w_base: &[f32],
    stale_delta: &[f32],
    max_delta_norm_tau: Option<f32>,
) -> Result<FedAsyncStaleMixOutcomeV1> {
    if w_current.is_empty()
        || w_current.len() != w_base.len()
        || w_current.len() != stale_delta.len()
    {
        return Err(Error::InvalidGradient);
    }

    let effective_alpha = policy.compute_mixing_alpha(base_round_id, current_round_id)?;
    if !effective_alpha.is_finite() || effective_alpha <= 0.0 || effective_alpha > 1.0 {
        return Err(Error::InvalidThreshold);
    }
    let staleness_rounds = current_round_id
        .checked_sub(base_round_id)
        .ok_or(Error::InvalidThreshold)?;

    let mut delta_sq_sum = 0.0f64;
    let mut drift_sq_sum = 0.0f64;

    for j in 0..w_current.len() {
        let wc = w_current[j];
        let wb = w_base[j];
        let d = stale_delta[j];
        if wc.is_nan() || wb.is_nan() || d.is_nan() {
            return Err(Error::NanDetected);
        }
        if wc.is_infinite() || wb.is_infinite() || d.is_infinite() {
            return Err(Error::InfDetected);
        }
        let d_f64 = d as f64;
        delta_sq_sum += d_f64 * d_f64;
        let drift_f64 = (wb as f64) - (wc as f64);
        drift_sq_sum += drift_f64 * drift_f64;
    }

    let raw_delta_l2_norm = crate::num::sqrt_f64_soft(delta_sq_sum);
    let base_drift_l2_norm = crate::num::sqrt_f64_soft(drift_sq_sum);
    if !raw_delta_l2_norm.is_finite() || !base_drift_l2_norm.is_finite() {
        return Err(Error::InfDetected);
    }

    let (clip_scale, clipped_delta_l2_norm, was_clipped) = match max_delta_norm_tau {
        Some(tau) => {
            if !tau.is_finite() || tau <= 0.0 {
                return Err(Error::InvalidThreshold);
            }
            let tau_f64 = tau as f64;
            if raw_delta_l2_norm > tau_f64 {
                (tau_f64 / raw_delta_l2_norm, tau_f64, true)
            } else {
                (1.0, raw_delta_l2_norm, false)
            }
        }
        None => (1.0, raw_delta_l2_norm, false),
    };

    let alpha_f64 = effective_alpha as f64;
    let mut updated_weights = Vec::with_capacity(w_current.len());
    for j in 0..w_current.len() {
        let wc = w_current[j] as f64;
        let wb = w_base[j] as f64;
        let d_clipped = (stale_delta[j] as f64) * clip_scale;
        let next_w = wc + alpha_f64 * ((wb - wc) + d_clipped);
        let next_f32 = next_w as f32;
        if !next_f32.is_finite() {
            return Err(Error::InfDetected);
        }
        updated_weights.push(next_f32);
    }

    Ok(FedAsyncStaleMixOutcomeV1 {
        updated_weights,
        effective_alpha,
        staleness_rounds,
        raw_delta_l2_norm,
        clipped_delta_l2_norm,
        base_drift_l2_norm,
        was_clipped,
    })
}

/// Configuration for robust aggregation
#[derive(Debug, Clone)]
pub enum RobustAggregation {
    /// Simple averaging (no protection)
    FedAvg,
    /// Coordinate-wise median
    Median,
    /// Trimmed mean with specified trim ratio
    TrimmedMean { trim_ratio: f32 },
    /// Krum algorithm
    #[cfg(feature = "krum")]
    Krum { num_byzantine: usize },
    /// One-pass Multi-Krum algorithm
    #[cfg(feature = "krum")]
    MultiKrum {
        num_byzantine: usize,
        num_selected: usize,
    },
    /// Bulyan algorithm with explicit versioned profile
    #[cfg(feature = "bulyan")]
    Bulyan {
        max_byzantine: usize,
        profile: BulyanProfile,
    },
    /// Stateful / iterative Centered Clipping (`CClip`, `ADR-0019` §4, `W11-01`).
    CenteredClipping { tau: f32, max_iterations: usize },
}

impl Default for RobustAggregation {
    fn default() -> Self {
        Self::TrimmedMean { trim_ratio: 0.2 }
    }
}

#[cfg(all(test, feature = "alloc"))]
#[path = "aggregation_tests.rs"]
mod tests;
