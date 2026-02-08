//! Core traits for SwarmTorch
//!
//! These traits define the fundamental abstractions used throughout SwarmTorch.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::Result;

/// A model that can participate in swarm learning
pub trait SwarmModel: Send + Sync {
    /// The input type for the model
    type Input;
    /// The output type for the model
    type Output;

    /// Perform forward pass
    fn forward(&self, input: Self::Input) -> Self::Output;

    /// Get model parameters as a flat vector
    fn parameters(&self) -> &[f32];

    /// Get mutable model parameters
    fn parameters_mut(&mut self) -> &mut [f32];

    /// Load parameters from a flat vector
    fn load_parameters(&mut self, params: &[f32]) -> Result<()>;

    /// Get the number of parameters
    fn num_parameters(&self) -> usize {
        self.parameters().len()
    }
}

/// A gradient update from a participant
#[derive(Debug, Clone)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct GradientUpdate {
    /// Sender's peer ID
    pub sender: [u8; 32],
    /// Sequence number for ordering
    pub sequence: u64,
    /// The gradient values
    #[cfg(feature = "alloc")]
    pub gradients: Vec<f32>,
    /// Round this update belongs to
    pub round_id: u64,
}

/// An optimizer that can be used in swarm learning
pub trait SwarmOptimizer: Send + Sync {
    /// Apply an optimization step given gradients
    fn step(&mut self, model: &mut dyn SwarmModel<Input = (), Output = ()>, gradients: &[f32]);

    /// Get the current learning rate
    fn learning_rate(&self) -> f32;

    /// Set the learning rate
    fn set_learning_rate(&mut self, lr: f32);
}

/// A fitness function for swarm optimization
pub trait FitnessFunction: Send + Sync {
    /// Evaluate the fitness of a model
    fn evaluate(&self, model: &dyn SwarmModel<Input = (), Output = ()>) -> f32;
}

/// Peer identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct PeerId(pub [u8; 32]);

impl PeerId {
    /// Create a new PeerId from bytes
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Create a PeerId from a public key (SHA-256 hash)
    #[cfg(feature = "std")]
    pub fn from_public_key(public_key: &[u8]) -> Self {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(public_key);
        let result = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&result);
        Self(bytes)
    }

    /// Get the raw bytes
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl AsRef<[u8]> for PeerId {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
