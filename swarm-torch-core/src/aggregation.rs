//! Robust aggregation algorithms for Byzantine-resilient swarm learning
//!
//! This module provides various aggregation strategies that can tolerate
//! malicious or faulty participants in the swarm.

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
use crate::Result;

const MAX_GRADIENT_DIM: usize = 10_000_000;

fn validate_gradient_shapes(updates: &[GradientUpdate]) -> Result<usize> {
    if updates.is_empty() {
        return Err(crate::Error::InsufficientUpdates);
    }

    let dim = updates[0].gradients.len();
    if dim == 0 || dim > MAX_GRADIENT_DIM {
        return Err(crate::Error::InvalidGradient);
    }

    for update in updates.iter().skip(1) {
        if update.gradients.len() != dim {
            return Err(crate::Error::InvalidGradient);
        }
    }

    Ok(dim)
}

/// Trait for robust aggregation algorithms
pub trait RobustAggregator: Send + Sync {
    /// Aggregate multiple gradient updates into one
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
#[derive(Debug)]
pub enum AggregationPipelineError {
    Aggregation(crate::Error),
    Compression(CompressionError),
}

/// Composable aggregation pipeline over a concrete robust aggregator.
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

/// Simple averaging aggregator (no Byzantine protection)
#[derive(Debug, Clone, Default)]
pub struct FedAvg;

impl RobustAggregator for FedAvg {
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>> {
        #[cfg(feature = "alloc")]
        {
            let dim = validate_gradient_shapes(updates)?;
            let n = updates.len() as f32;
            let mut result = alloc::vec![0.0f32; dim];

            // Sum all gradients first to minimize floating-point rounding drift.
            for update in updates {
                for (slot, &gradient) in result.iter_mut().zip(update.gradients.iter()) {
                    *slot += gradient;
                }
            }
            // Single division pass after accumulation.
            for slot in result.iter_mut() {
                *slot /= n;
            }

            Ok(result)
        }

        #[cfg(not(feature = "alloc"))]
        Err(crate::Error::ResourceExhausted)
    }

    fn byzantine_tolerance(&self) -> f32 {
        0.0 // No Byzantine tolerance
    }

    fn complexity(&self) -> AggregatorComplexity {
        AggregatorComplexity::Linear
    }
}

/// Trimmed mean aggregator — discards top/bottom k% of values per coordinate.
///
/// # Rounding Behavior (M-10)
///
/// - `trim_ratio` is clamped to `[0.0, 0.49]` by [`TrimmedMean::new`].
///   A ratio of 0.49 trims nearly half from each end.
/// - The trim count is computed as `((n as f32) * trim_ratio) as usize`,
///   which truncates toward zero. For small peer counts this means fewer
///   values are actually trimmed than the ratio might suggest.
/// - If the trim count leaves no values (`n <= 2 * trim_count`), the guard
///   returns [`Error::InsufficientUpdates`](crate::Error::InsufficientUpdates).
/// - The guard is only reachable via direct struct construction (bypassing
///   `new()`'s clamp) when using a `trim_ratio` above 0.49.
#[derive(Debug, Clone)]
pub struct TrimmedMean {
    /// Fraction of values to trim from each end (e.g., 0.2 for 20%).
    ///
    /// Clamped to `[0.0, 0.49]` by [`TrimmedMean::new`].
    pub trim_ratio: f32,
}

impl TrimmedMean {
    /// Create a new TrimmedMean aggregator.
    ///
    /// `trim_ratio` is clamped to `[0.0, 0.49]`. A value of 0.0 produces
    /// a simple mean (no trimming). Values above 0.49 are capped to prevent
    /// all peers from being trimmed.
    pub fn new(trim_ratio: f32) -> Self {
        Self {
            trim_ratio: trim_ratio.clamp(0.0, 0.49),
        }
    }
}

impl Default for TrimmedMean {
    fn default() -> Self {
        Self::new(0.2)
    }
}

impl RobustAggregator for TrimmedMean {
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>> {
        #[cfg(feature = "alloc")]
        {
            let dim = validate_gradient_shapes(updates)?;
            let n = updates.len();
            let trim_count = ((n as f32) * self.trim_ratio) as usize;

            if n <= 2 * trim_count {
                return Err(crate::Error::InsufficientUpdates);
            }

            let mut result = alloc::vec![0.0f32; dim];

            // For each coordinate, sort values and compute trimmed mean
            for (i, slot) in result.iter_mut().enumerate().take(dim) {
                let mut values: Vec<f32> = updates.iter().map(|u| u.gradients[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

                // Trim and average
                let trimmed = &values[trim_count..n - trim_count];
                let sum: f32 = trimmed.iter().sum();
                *slot = sum / (trimmed.len() as f32);
            }

            Ok(result)
        }

        #[cfg(not(feature = "alloc"))]
        Err(crate::Error::ResourceExhausted)
    }

    fn byzantine_tolerance(&self) -> f32 {
        self.trim_ratio
    }

    fn complexity(&self) -> AggregatorComplexity {
        AggregatorComplexity::Linear
    }
}

/// Coordinate-wise median aggregator
#[derive(Debug, Clone, Default)]
pub struct CoordinateMedian;

impl RobustAggregator for CoordinateMedian {
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>> {
        #[cfg(feature = "alloc")]
        {
            let dim = validate_gradient_shapes(updates)?;
            let n = updates.len();
            let mut result = alloc::vec![0.0f32; dim];

            for (i, slot) in result.iter_mut().enumerate().take(dim) {
                let mut values: Vec<f32> = updates.iter().map(|u| u.gradients[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

                // Compute median
                *slot = if n % 2 == 0 {
                    (values[n / 2 - 1] + values[n / 2]) / 2.0
                } else {
                    values[n / 2]
                };
            }

            Ok(result)
        }

        #[cfg(not(feature = "alloc"))]
        Err(crate::Error::ResourceExhausted)
    }

    fn byzantine_tolerance(&self) -> f32 {
        0.5 // Can tolerate up to 50% malicious (theoretically)
    }

    fn complexity(&self) -> AggregatorComplexity {
        AggregatorComplexity::Linear
    }
}

/// Krum aggregator - selects the update closest to others
#[cfg(feature = "krum")]
#[derive(Debug, Clone)]
pub struct Krum {
    /// Number of updates to select
    pub num_selected: usize,
    /// Expected number of Byzantine nodes
    pub num_byzantine: usize,
}

#[cfg(feature = "krum")]
impl Krum {
    /// Create a new Krum aggregator
    pub fn new(num_byzantine: usize) -> Self {
        Self {
            num_selected: 1,
            num_byzantine,
        }
    }
}

#[cfg(feature = "krum")]
impl RobustAggregator for Krum {
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>> {
        #[cfg(feature = "alloc")]
        {
            const MAX_KRUM_PEERS: usize = 50;

            let _ = validate_gradient_shapes(updates)?;
            let n = updates.len();
            let f = self.num_byzantine;

            if n < 2 * f + 3 {
                return Err(crate::Error::InsufficientUpdates);
            }
            if n > MAX_KRUM_PEERS {
                return Err(crate::Error::ResourceExhausted);
            }

            // Compute pairwise distances
            let mut distances: Vec<Vec<f32>> = alloc::vec![alloc::vec![0.0; n]; n];
            for i in 0..n {
                for j in (i + 1)..n {
                    let dist: f32 = updates[i]
                        .gradients
                        .iter()
                        .zip(&updates[j].gradients)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    distances[i][j] = dist;
                    distances[j][i] = dist;
                }
            }

            // For each update, compute sum of distances to n-f-2 closest neighbors
            let k = n - f - 2;
            let mut scores: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let mut dists: Vec<f32> = distances[i].clone();
                    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                    let score: f32 = dists[1..=k].iter().sum(); // Skip self (distance 0)
                    (i, score)
                })
                .collect();

            scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

            // Return the update with lowest Krum score
            Ok(updates[scores[0].0].gradients.clone())
        }

        #[cfg(not(feature = "alloc"))]
        Err(crate::Error::ResourceExhausted)
    }

    fn byzantine_tolerance(&self) -> f32 {
        // Krum tolerates f Byzantine nodes out of n >= 2f+3
        0.33
    }

    fn complexity(&self) -> AggregatorComplexity {
        AggregatorComplexity::Quadratic
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
}

impl Default for RobustAggregation {
    fn default() -> Self {
        Self::TrimmedMean { trim_ratio: 0.2 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionMethod;
    use crate::traits::UpdateTransform;

    fn update(gradients: Vec<f32>) -> GradientUpdate {
        GradientUpdate {
            sender: [0u8; 32],
            sequence: 0,
            gradients,
            round_id: 0,
        }
    }

    fn update_with_meta(
        sender_byte: u8,
        sequence: u64,
        round_id: u64,
        gradients: Vec<f32>,
    ) -> GradientUpdate {
        GradientUpdate {
            sender: [sender_byte; 32],
            sequence,
            gradients,
            round_id,
        }
    }

    struct AddOneTransform;

    impl UpdateTransform for AddOneTransform {
        fn transform(&self, mut update: GradientUpdate) -> GradientUpdate {
            for g in &mut update.gradients {
                *g += 1.0;
            }
            update
        }

        fn name(&self) -> &str {
            "add_one"
        }

        fn is_core_trusted(&self) -> bool {
            true
        }
    }

    struct UntrustedClipTransform;

    impl UpdateTransform for UntrustedClipTransform {
        fn transform(&self, mut update: GradientUpdate) -> GradientUpdate {
            for g in &mut update.gradients {
                *g = g.clamp(-1.0, 1.0);
            }
            update
        }

        fn name(&self) -> &str {
            "clip"
        }
    }

    #[test]
    fn fedavg_rejects_mismatched_gradient_dimensions() {
        let updates = vec![update(vec![1.0, 2.0]), update(vec![3.0])];
        let result = FedAvg.aggregate(&updates);
        assert!(matches!(result, Err(crate::Error::InvalidGradient)));
    }

    #[test]
    fn trimmed_mean_rejects_mismatched_gradient_dimensions() {
        let updates = vec![update(vec![1.0, 2.0]), update(vec![3.0])];
        let result = TrimmedMean::new(0.2).aggregate(&updates);
        assert!(matches!(result, Err(crate::Error::InvalidGradient)));
    }

    #[test]
    fn coordinate_median_rejects_empty_gradient_vectors() {
        let updates = vec![update(vec![]), update(vec![])];
        let result = CoordinateMedian.aggregate(&updates);
        assert!(matches!(result, Err(crate::Error::InvalidGradient)));
    }

    #[test]
    fn aggregation_rejects_oversized_gradient_dim() {
        let oversized = [0.0; 10_000_001].to_vec(); // Just barely over MAX_GRADIENT_DIM
        let updates = vec![update(oversized.clone())];
        let result = FedAvg.aggregate(&updates);
        assert!(matches!(result, Err(crate::Error::InvalidGradient)));
    }

    #[test]
    #[cfg(feature = "krum")]
    fn krum_rejects_oversized_peer_set() {
        let mut updates = Vec::new();
        // MAX_KRUM_PEERS is 50. Provide 51 peers.
        for _ in 0..51 {
            updates.push(update(vec![1.0; 2]));
        }
        let krum = Krum::new(0); // 0 byzantine to keep `n < 2*f + 3` check satisfied
        let result = krum.aggregate(&updates);
        assert!(matches!(result, Err(crate::Error::ResourceExhausted)));
    }

    /// NEW-04: FedAvg sum-then-divide produces results matching f64 oracle
    /// within 1e-6 relative tolerance.
    #[test]
    fn fedavg_sum_then_divide_precision() {
        // Use many small values where per-element division would drift.
        let n = 100;
        let dim = 4;
        let updates: Vec<GradientUpdate> = (0..n)
            .map(|i| {
                let base = (i as f32) * 0.001 + 1e-7;
                update(vec![base; dim])
            })
            .collect();

        let result = FedAvg.aggregate(&updates).unwrap();

        // f64 oracle: compute exact mean in double precision.
        for (d, &actual_f32) in result.iter().enumerate() {
            let oracle: f64 =
                updates.iter().map(|u| u.gradients[d] as f64).sum::<f64>() / (n as f64);
            let actual = actual_f32 as f64;
            let rel_err = if oracle.abs() > 1e-15 {
                (actual - oracle).abs() / oracle.abs()
            } else {
                (actual - oracle).abs()
            };
            assert!(
                rel_err < 1e-6,
                "FedAvg precision drift: dim={d}, oracle={oracle}, actual={actual}, rel_err={rel_err}"
            );
        }
    }

    /// M-10: TrimmedMean::new() clamp prevents all-trimmed via normal construction.
    #[test]
    fn trimmed_mean_new_clamps_ratio() {
        let tm = TrimmedMean::new(0.6);
        assert!(
            (tm.trim_ratio - 0.49).abs() < f32::EPSILON,
            "ratio above 0.49 should clamp"
        );
    }

    /// M-10: InsufficientUpdates guard reachable via direct struct construction.
    #[test]
    fn trimmed_mean_direct_construction_insufficient_updates() {
        // Direct construction bypasses new()'s clamp.
        let tm = TrimmedMean { trim_ratio: 0.6 };
        // 2 peers, 0.6 ratio: trim_count = (2 * 0.6) as usize = 1, n=2, 2<=2*1.
        let updates = vec![update(vec![1.0, 2.0]), update(vec![3.0, 4.0])];
        let result = tm.aggregate(&updates);
        assert!(
            matches!(result, Err(crate::Error::InsufficientUpdates)),
            "guard should return InsufficientUpdates, got {result:?}"
        );
    }

    #[test]
    fn update_transform_preserves_sender_sequence_round_id() {
        let updates = vec![
            update_with_meta(1, 11, 101, vec![0.5, 1.5]),
            update_with_meta(2, 12, 101, vec![2.5, 3.5]),
        ];
        let (transformed, _) = apply_update_transforms(updates.clone(), &AddOneTransform);

        for (input, output) in updates.iter().zip(transformed.iter()) {
            assert_eq!(input.sender, output.sender);
            assert_eq!(input.sequence, output.sequence);
            assert_eq!(input.round_id, output.round_id);
        }
        assert_eq!(transformed[0].gradients, vec![1.5, 2.5]);
    }

    #[test]
    fn untrusted_transform_audit_has_core_trusted_false() {
        let updates = vec![update_with_meta(7, 1, 9, vec![2.0, -4.0])];
        let (_, audit) = apply_update_transforms(updates, &UntrustedClipTransform);
        assert_eq!(audit.transform_name, "clip");
        assert!(!audit.core_trusted);
        assert_eq!(audit.round_id, 9);
    }

    #[test]
    fn core_trusted_transform_audit_has_core_trusted_true() {
        let updates = vec![update_with_meta(7, 1, 3, vec![1.0])];
        let (_, audit) = apply_update_transforms(updates, &AddOneTransform);
        assert_eq!(audit.transform_name, "add_one");
        assert!(audit.core_trusted);
        assert_eq!(audit.round_id, 3);
    }

    #[test]
    fn apply_transforms_then_aggregate_trimmed_mean_end_to_end() {
        let updates = vec![
            update_with_meta(1, 1, 4, vec![1.0, 2.0]),
            update_with_meta(2, 2, 4, vec![3.0, 4.0]),
        ];
        let (transformed, _) = apply_update_transforms(updates, &AddOneTransform);
        let aggregated = TrimmedMean::new(0.0).aggregate(&transformed).unwrap();
        assert_eq!(aggregated, vec![3.0, 4.0]);
    }

    #[test]
    fn pipeline_trimmed_mean_no_compression_matches_direct() {
        let updates = vec![update(vec![1.0, 2.0]), update(vec![3.0, 4.0])];
        let direct = TrimmedMean::new(0.0).aggregate(&updates).unwrap();
        let piped = AggregationPipeline::new(TrimmedMean::new(0.0))
            .run(&updates)
            .unwrap();
        assert_eq!(piped, direct);
    }

    #[test]
    fn pipeline_fedavg_topk_compression_roundtrip() {
        let updates = vec![update(vec![1.0, 2.0, 3.0]), update(vec![5.0, 6.0, 7.0])];
        let direct = FedAvg.aggregate(&updates).unwrap();
        let piped = AggregationPipeline::new(FedAvg)
            .with_compression(CompressionMethod::TopK { k_ratio: 1.0 })
            .run(&updates)
            .unwrap();
        assert_eq!(piped, direct);
    }

    #[test]
    fn pipeline_returns_error_on_empty_updates() {
        let updates: Vec<GradientUpdate> = Vec::new();
        let err = AggregationPipeline::new(FedAvg).run(&updates).unwrap_err();
        assert!(matches!(
            err,
            AggregationPipelineError::Aggregation(crate::Error::InsufficientUpdates)
        ));
    }

    #[test]
    fn pipeline_trace_nonzero_estimates_for_valid_inputs() {
        let trace = AggregationPipeline::new(FedAvg).trace(1024, 10);
        assert!(trace.estimated_memory_bytes > 0);
        assert!(trace.estimated_flops > 0);
    }

    #[test]
    fn pipeline_trace_topk_compressed_less_than_raw() {
        let trace = AggregationPipeline::new(FedAvg)
            .with_compression(CompressionMethod::TopK { k_ratio: 0.1 })
            .trace(1000, 8);
        let compressed = trace.estimated_compressed_bytes.unwrap();
        assert!(compressed < trace.estimated_memory_bytes);
    }

    #[test]
    fn pipeline_trace_memory_lower_bound_holds() {
        let gradient_len = 257usize;
        let num_peers = 19usize;
        let trace = AggregationPipeline::new(FedAvg).trace(gradient_len, num_peers);
        let lower_bound = (gradient_len as u64) * (num_peers as u64) * 4;
        assert!(trace.estimated_memory_bytes >= lower_bound);
    }

    #[test]
    fn pipeline_trace_zero_peers_zero_estimates() {
        let trace = AggregationPipeline::new(FedAvg)
            .with_compression(CompressionMethod::TopK { k_ratio: 0.2 })
            .trace(0, 0);
        assert_eq!(trace.estimated_memory_bytes, 0);
        assert_eq!(trace.estimated_flops, 0);
        assert_eq!(trace.estimated_compressed_bytes, Some(0));
    }
}
