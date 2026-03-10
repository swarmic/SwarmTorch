//! # SwarmTorch
//!
//! **Mission-grade swarm learning for heterogeneous fleets—Rust-native, Byzantine-resilient, with a portable `swarm-torch-core` base.**
//!
//! SwarmTorch is a distributed machine learning framework designed for real-world edge deployments
//! where devices are resource-constrained, connections are unreliable, and trust cannot be assumed.
//! The top-level `swarm-torch` crate is currently `std`-only; `no_std` portability is exposed via `swarm-torch-core`.
//!
//! ## Quick Start
//!
//! The snippet below is an illustrative design-target example; current high-level training/transport APIs are still evolving.
//!
//! ```rust,ignore
//! use swarm_torch::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Define swarm cluster with 3 nodes
//!     let swarm = SwarmCluster::builder()
//!         .topology(Topology::gossip(fanout: 2))
//!         .consensus(RobustAggregation::trimmed_mean(trim_ratio: 0.2))
//!         .transport(TcpTransport::local_cluster(num_nodes: 3))
//!         .build()
//!         .await?;
//!
//!     // Train across swarm
//!     let trained_model = swarm
//!         .train(model, optimizer, data)
//!         .max_rounds(100)
//!         .await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Feature Flags
//!
//! - `std` (default): Standard library support
//! - `tokio-runtime` (default): Use Tokio for async
//! - `embassy-runtime`: Enable Embassy placeholder adapter (experimental; not in Rust 1.75 conformance gate)
//! - `burn-backend` (default): Burn ML framework integration
//! - `robust-aggregation` (default): Byzantine-resilient aggregators
//!
//! ## Crate Structure
//!
//! - [`swarm_torch_core`]: Core algorithms and traits (no_std compatible)
//! - [`swarm_torch_net`]: Network transport abstractions
//! - [`swarm_torch_runtime`]: Async runtime glue (Tokio/Embassy)
//! - [`swarm_torch_models`]: Model utilities and backend integrations

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[cfg(not(feature = "std"))]
compile_error!(
    "`swarm-torch` requires the `std` feature. Use `swarm-torch-core` for no_std targets."
);

// Re-export sub-crates
pub use swarm_torch_core as core;
pub use swarm_torch_models as models;
pub use swarm_torch_net as net;
pub use swarm_torch_runtime as runtime;

// Re-export commonly used items at the top level
pub use swarm_torch_core::{
    aggregation::{self, RobustAggregation, RobustAggregator},
    algorithms::Topology,
    traits::{GradientUpdate, PeerId, SwarmModel},
    Error, Result,
};

pub use swarm_torch_net::{
    protocol::{MessageEnvelope, MessageType},
    traits::{SwarmTransport, TransportCapabilities},
};

/// Artifact bundle writing/validation (std-only).
#[cfg(feature = "std")]
pub mod artifacts;

/// Standalone report generator (std-only).
#[cfg(feature = "std")]
pub mod report;

/// Minimal native OpRunner (std-only).
#[cfg(feature = "std")]
pub mod native_runner;

/// Sequential graph scheduler (std-only).
#[cfg(feature = "std")]
pub mod scheduler;

/// Prelude module for convenient imports
///
/// ```rust,ignore
/// use swarm_torch::prelude::*;
/// ```
pub mod prelude {
    pub use crate::core::prelude::*;
    pub use crate::models::prelude::*;
    pub use crate::net::prelude::*;

    pub use crate::{SwarmCluster, SwarmConfig};
}

/// Configuration for a SwarmTorch cluster
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// Network topology
    pub topology: Topology,
    /// Aggregation strategy
    pub aggregation: RobustAggregation,
    /// Maximum training rounds
    pub max_rounds: u64,
    /// Convergence threshold for early stopping
    pub convergence_threshold: f32,
}

/// SwarmConfig validation error (M-13).
#[derive(Debug, Clone, PartialEq)]
pub enum SwarmConfigError {
    /// `max_rounds` must be non-zero.
    MaxRoundsZero,
    /// `convergence_threshold` must be a finite number.
    ConvergenceThresholdNotFinite,
    /// `convergence_threshold` must be non-negative.
    ConvergenceThresholdNegative,
}

impl std::fmt::Display for SwarmConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MaxRoundsZero => write!(f, "max_rounds must be non-zero"),
            Self::ConvergenceThresholdNotFinite => {
                write!(f, "convergence_threshold must be finite")
            }
            Self::ConvergenceThresholdNegative => {
                write!(f, "convergence_threshold must be non-negative")
            }
        }
    }
}

impl std::error::Error for SwarmConfigError {}

impl SwarmConfig {
    /// Validate configuration.
    pub fn validate(&self) -> std::result::Result<(), SwarmConfigError> {
        if self.max_rounds == 0 {
            return Err(SwarmConfigError::MaxRoundsZero);
        }
        if !self.convergence_threshold.is_finite() {
            return Err(SwarmConfigError::ConvergenceThresholdNotFinite);
        }
        if self.convergence_threshold < 0.0 {
            return Err(SwarmConfigError::ConvergenceThresholdNegative);
        }
        Ok(())
    }
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            topology: Topology::default(),
            aggregation: RobustAggregation::default(),
            max_rounds: 100,
            convergence_threshold: 0.01,
        }
    }
}

/// Builder for SwarmConfig
#[derive(Debug, Default)]
pub struct SwarmConfigBuilder {
    config: SwarmConfig,
}

impl SwarmConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the network topology
    pub fn topology(mut self, topology: Topology) -> Self {
        self.config.topology = topology;
        self
    }

    /// Set the aggregation strategy
    pub fn consensus(mut self, aggregation: RobustAggregation) -> Self {
        self.config.aggregation = aggregation;
        self
    }

    /// Set the maximum number of training rounds
    pub fn max_rounds(mut self, rounds: u64) -> Self {
        self.config.max_rounds = rounds;
        self
    }

    /// Set the convergence threshold
    pub fn convergence_threshold(mut self, threshold: f32) -> Self {
        self.config.convergence_threshold = threshold;
        self
    }

    /// Build the configuration
    pub fn build(self) -> SwarmConfig {
        self.config
    }

    /// Build the configuration with validation (M-13).
    ///
    /// Returns `Err` if configuration values are invalid.
    pub fn try_build(self) -> std::result::Result<SwarmConfig, SwarmConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// A SwarmTorch cluster for distributed training
#[derive(Debug)]
pub struct SwarmCluster {
    /// Cluster configuration
    pub config: SwarmConfig,
    /// Local peer ID
    pub local_peer: PeerId,
}

impl SwarmCluster {
    /// Create a new cluster builder
    pub fn builder() -> SwarmConfigBuilder {
        SwarmConfigBuilder::new()
    }

    /// Create a cluster with the given configuration
    pub fn new(config: SwarmConfig, local_peer: PeerId) -> Self {
        Self { config, local_peer }
    }

    /// Get the cluster configuration
    pub fn config(&self) -> &SwarmConfig {
        &self.config
    }

    /// Get the local peer ID
    pub fn local_peer(&self) -> &PeerId {
        &self.local_peer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = SwarmCluster::builder()
            .topology(Topology::gossip(4))
            .max_rounds(50)
            .convergence_threshold(0.001)
            .build();

        assert_eq!(config.max_rounds, 50);
        assert!((config.convergence_threshold - 0.001).abs() < f32::EPSILON);
    }

    #[test]
    fn test_peer_id() {
        let bytes = [1u8; 32];
        let peer = PeerId::new(bytes);
        assert_eq!(peer.as_bytes(), &bytes);
    }

    // ── M-13: SwarmConfig validation tests ──

    #[test]
    fn try_build_rejects_zero_max_rounds() {
        let result = SwarmCluster::builder().max_rounds(0).try_build();
        assert_eq!(result.unwrap_err(), SwarmConfigError::MaxRoundsZero);
    }

    #[test]
    fn try_build_rejects_nan_threshold() {
        let result = SwarmCluster::builder()
            .convergence_threshold(f32::NAN)
            .try_build();
        assert_eq!(
            result.unwrap_err(),
            SwarmConfigError::ConvergenceThresholdNotFinite
        );
    }

    #[test]
    fn try_build_rejects_negative_threshold() {
        let result = SwarmCluster::builder()
            .convergence_threshold(-0.1)
            .try_build();
        assert_eq!(
            result.unwrap_err(),
            SwarmConfigError::ConvergenceThresholdNegative
        );
    }

    #[test]
    fn try_build_accepts_valid_config() {
        let result = SwarmCluster::builder()
            .max_rounds(50)
            .convergence_threshold(0.001)
            .try_build();
        assert!(result.is_ok());
    }
}
