//! Robust aggregation algorithms for Byzantine-resilient swarm learning
//!
//! This module provides various aggregation strategies that can tolerate
//! malicious or faulty participants in the swarm.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::traits::GradientUpdate;
use crate::Result;

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

/// Simple averaging aggregator (no Byzantine protection)
#[derive(Debug, Clone, Default)]
pub struct FedAvg;

impl RobustAggregator for FedAvg {
    fn aggregate(&self, updates: &[GradientUpdate]) -> Result<Vec<f32>> {
        if updates.is_empty() {
            return Err(crate::Error::InsufficientUpdates);
        }

        #[cfg(feature = "alloc")]
        {
            let n = updates.len() as f32;
            let dim = updates[0].gradients.len();
            let mut result = alloc::vec![0.0f32; dim];

            for update in updates {
                for (i, &g) in update.gradients.iter().enumerate() {
                    result[i] += g / n;
                }
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

/// Trimmed mean aggregator - discards top/bottom k% of values per coordinate
#[derive(Debug, Clone)]
pub struct TrimmedMean {
    /// Fraction of values to trim from each end (e.g., 0.2 for 20%)
    pub trim_ratio: f32,
}

impl TrimmedMean {
    /// Create a new TrimmedMean aggregator
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
        if updates.is_empty() {
            return Err(crate::Error::InsufficientUpdates);
        }

        #[cfg(feature = "alloc")]
        {
            let n = updates.len();
            let trim_count = ((n as f32) * self.trim_ratio) as usize;

            if n <= 2 * trim_count {
                return Err(crate::Error::InsufficientUpdates);
            }

            let dim = updates[0].gradients.len();
            let mut result = alloc::vec![0.0f32; dim];

            // For each coordinate, sort values and compute trimmed mean
            for i in 0..dim {
                let mut values: Vec<f32> = updates.iter().map(|u| u.gradients[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

                // Trim and average
                let trimmed = &values[trim_count..n - trim_count];
                let sum: f32 = trimmed.iter().sum();
                result[i] = sum / (trimmed.len() as f32);
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
        if updates.is_empty() {
            return Err(crate::Error::InsufficientUpdates);
        }

        #[cfg(feature = "alloc")]
        {
            let n = updates.len();
            let dim = updates[0].gradients.len();
            let mut result = alloc::vec![0.0f32; dim];

            for i in 0..dim {
                let mut values: Vec<f32> = updates.iter().map(|u| u.gradients[i]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

                // Compute median
                result[i] = if n % 2 == 0 {
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
        if updates.is_empty() {
            return Err(crate::Error::InsufficientUpdates);
        }

        #[cfg(feature = "alloc")]
        {
            let n = updates.len();
            let f = self.num_byzantine;

            if n < 2 * f + 3 {
                return Err(crate::Error::InsufficientUpdates);
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
