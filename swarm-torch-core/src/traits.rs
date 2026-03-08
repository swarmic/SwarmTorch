//! Core traits for SwarmTorch
//!
//! These traits define the fundamental abstractions used throughout SwarmTorch.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::crypto::CryptoError;
use crate::Result;

/// A model that can participate in swarm learning
pub trait SwarmModel: Send + Sync {
    /// The input type for the model
    type Input;
    /// The output type for the model
    type Output;
    /// Model forward error type
    type Error: core::fmt::Debug;

    /// Perform forward pass
    fn forward(&self, input: &Self::Input) -> core::result::Result<Self::Output, Self::Error>;

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
    /// Model type this optimizer updates.
    type Model: SwarmModel<Input = (), Output = ()>;
    /// Optimizer error type.
    type Error: core::fmt::Debug;

    /// Apply an optimization step given gradients
    fn step(
        &mut self,
        model: &mut Self::Model,
        gradients: &[f32],
    ) -> core::result::Result<(), Self::Error>;

    /// Get the current learning rate
    fn learning_rate(&self) -> f32;

    /// Set the learning rate
    fn set_learning_rate(&mut self, lr: f32);
}

/// A fitness function for swarm optimization
pub trait FitnessFunction: Send + Sync {
    /// Model type this evaluator accepts.
    type Model: SwarmModel<Input = (), Output = ()>;
    /// Evaluation error type.
    type Error: core::fmt::Debug;

    /// Evaluate the fitness of a model
    fn evaluate(&self, model: &Self::Model) -> core::result::Result<f32, Self::Error>;
}

/// Peer identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct PeerId(pub [u8; 32]);

impl PeerId {
    /// Create a new PeerId from bytes.
    ///
    /// **Unchecked** — does not validate the bytes. Prefer
    /// [`try_from_public_key_bytes`] at network entry points.
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Create a PeerId from a public key (SHA-256 hash)
    ///
    /// This is the canonical derivation rule for all profiles (std and no_std).
    pub fn from_public_key(public_key: &[u8]) -> Self {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(public_key);
        let result = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&result);
        Self(bytes)
    }

    /// Validated construction from raw public key bytes.
    ///
    /// Rejects inputs that are not exactly 32 bytes or are all-zeros.
    /// Use at network entry points to reject degenerate peer identities.
    pub fn try_from_public_key_bytes(public_key: &[u8]) -> core::result::Result<Self, CryptoError> {
        if public_key.len() != 32 {
            return Err(CryptoError::InvalidPublicKey);
        }
        if public_key.iter().all(|b| *b == 0) {
            return Err(CryptoError::InvalidPublicKey);
        }
        Ok(Self::from_public_key(public_key))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peer_id_try_from_rejects_all_zero() {
        let result = PeerId::try_from_public_key_bytes(&[0u8; 32]);
        assert_eq!(result, Err(CryptoError::InvalidPublicKey));
    }

    #[test]
    fn peer_id_try_from_rejects_wrong_length() {
        let short = [1u8; 16];
        assert_eq!(
            PeerId::try_from_public_key_bytes(&short),
            Err(CryptoError::InvalidPublicKey)
        );
        let long = [1u8; 64];
        assert_eq!(
            PeerId::try_from_public_key_bytes(&long),
            Err(CryptoError::InvalidPublicKey)
        );
    }

    #[test]
    fn peer_id_try_from_valid_key_succeeds() {
        let key = [42u8; 32];
        let result = PeerId::try_from_public_key_bytes(&key);
        assert!(result.is_ok());
        // Must match canonical derivation
        assert_eq!(result.unwrap(), PeerId::from_public_key(&key));
    }

    #[test]
    fn peer_id_ord_is_byte_lexicographic() {
        let a = PeerId::new([0u8; 32]);
        let mut b_bytes = [0u8; 32];
        b_bytes[31] = 1;
        let b = PeerId::new(b_bytes);
        assert!(a < b);
    }

    #[derive(Default)]
    struct DummyModel {
        params: [f32; 1],
    }

    impl SwarmModel for DummyModel {
        type Input = ();
        type Output = ();
        type Error = core::convert::Infallible;

        fn forward(&self, _input: &Self::Input) -> core::result::Result<Self::Output, Self::Error> {
            Ok(())
        }

        fn parameters(&self) -> &[f32] {
            &self.params
        }

        fn parameters_mut(&mut self) -> &mut [f32] {
            &mut self.params
        }

        fn load_parameters(&mut self, params: &[f32]) -> Result<()> {
            self.params.copy_from_slice(params);
            Ok(())
        }
    }

    struct DummyOptimizer;

    impl SwarmOptimizer for DummyOptimizer {
        type Model = DummyModel;
        type Error = core::convert::Infallible;

        fn step(
            &mut self,
            model: &mut Self::Model,
            gradients: &[f32],
        ) -> core::result::Result<(), Self::Error> {
            model.load_parameters(gradients).unwrap();
            Ok(())
        }

        fn learning_rate(&self) -> f32 {
            0.1
        }

        fn set_learning_rate(&mut self, _lr: f32) {}
    }

    struct DummyFitness;

    impl FitnessFunction for DummyFitness {
        type Model = DummyModel;
        type Error = core::convert::Infallible;

        fn evaluate(&self, _model: &Self::Model) -> core::result::Result<f32, Self::Error> {
            Ok(1.0)
        }
    }

    #[test]
    fn swarm_optimizer_associated_model_type_compiles() {
        let mut model = DummyModel::default();
        let mut opt = DummyOptimizer;
        opt.step(&mut model, &[0.5]).unwrap();
        assert_eq!(model.parameters(), &[0.5]);
    }

    #[test]
    fn fitness_function_associated_model_type_compiles() {
        let model = DummyModel::default();
        let fitness = DummyFitness;
        assert_eq!(fitness.evaluate(&model).unwrap(), 1.0);
    }
}
