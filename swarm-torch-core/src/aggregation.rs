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
use crate::compression::{CompressedGradient, CompressionError, CompressionMethod};
#[cfg(feature = "alloc")]
use crate::dataops::TransformAuditV0;
use crate::traits::GradientUpdate;
#[cfg(feature = "alloc")]
use crate::traits::UpdateTransform;
use crate::{Error, Result};

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

/// Stage 2 of Bulyan (`ADR-0019`, `W9-01`): given `theta` selected vectors and `f = max_byzantine`,
/// computes each coordinate's median $m_j$ over the `theta` values and averages the
/// $\beta = \theta - 2f$ values minimizing $|v - m_j|$ in `f64`.
#[cfg(all(feature = "alloc", feature = "bulyan"))]
pub fn bulyan_stage2_nearest_to_median(
    selected_updates: &[&[f32]],
    max_byzantine: usize,
) -> Result<Vec<f32>> {
    if max_byzantine == 0 {
        return Err(Error::InvalidThreshold);
    }
    let theta = selected_updates.len();
    let min_theta = max_byzantine
        .checked_mul(2)
        .and_then(|two_f| two_f.checked_add(1))
        .ok_or(Error::ResourceExhausted)?;
    if theta < min_theta {
        return Err(Error::InsufficientPeers);
    }

    let dim = selected_updates[0].len();
    if dim == 0 || dim > MAX_GRADIENT_DIM {
        return Err(Error::InvalidGradient);
    }
    for slice in selected_updates {
        if slice.len() != dim {
            return Err(Error::InvalidGradient);
        }
        for &val in *slice {
            if val.is_nan() {
                return Err(Error::NanDetected);
            }
            if val.is_infinite() {
                return Err(Error::InfDetected);
            }
        }
    }

    let beta = theta - 2 * max_byzantine;
    let inv_beta = 1.0f64 / (beta as f64);
    let mut out = alloc::vec![0.0f32; dim];
    let mut sorted = alloc::vec![0.0f32; theta];
    let mut ranked: Vec<(usize, f32, f64)> = Vec::with_capacity(theta);

    for (coord, slot) in out.iter_mut().enumerate().take(dim) {
        for (idx, slice) in selected_updates.iter().enumerate() {
            sorted[idx] = slice[coord];
        }
        sorted.sort_by(f32::total_cmp);

        let median = if theta % 2 == 0 {
            ((sorted[theta / 2 - 1] as f64) + (sorted[theta / 2] as f64)) * 0.5
        } else {
            sorted[theta / 2] as f64
        };

        ranked.clear();
        for (idx, slice) in selected_updates.iter().enumerate() {
            let val = slice[coord];
            let dist = ((val as f64) - median).abs();
            ranked.push((idx, val, dist));
        }

        ranked.sort_by(|&(idx_a, val_a, dist_a), &(idx_b, val_b, dist_b)| {
            dist_a
                .total_cmp(&dist_b)
                .then_with(|| val_a.total_cmp(&val_b))
                .then_with(|| idx_a.cmp(&idx_b))
        });

        let sum: f64 = ranked
            .iter()
            .take(beta)
            .map(|&(_, val, _)| val as f64)
            .sum();
        *slot = (sum * inv_beta) as f32;
    }

    validate_finite_output(&out)?;
    Ok(out)
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
        let selected_slices: Vec<&[f32]> = selected_indices
            .iter()
            .map(|&idx| updates[idx].gradients.as_slice())
            .collect();
        bulyan_stage2_nearest_to_median(&selected_slices, self.max_byzantine)
    }

    fn byzantine_tolerance(&self) -> f32 {
        0.25
    }

    fn complexity(&self) -> AggregatorComplexity {
        AggregatorComplexity::Cubic
    }
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
}

impl Default for RobustAggregation {
    fn default() -> Self {
        Self::TrimmedMean { trim_ratio: 0.2 }
    }
}

#[cfg(all(test, feature = "alloc"))]
#[path = "aggregation_tests.rs"]
mod tests;
