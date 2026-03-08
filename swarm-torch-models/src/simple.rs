//! Simple reference model implementations
//!
//! These models are useful for testing and examples.

use swarm_torch_core::traits::SwarmModel;
use swarm_torch_core::Result;

/// Error type for model creation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// Model dimensions exceed supported size for fixed storage
    InvalidDimensions,
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimensions => write!(
                f,
                "model dimensions exceed supported size for fixed storage"
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Parameter buffer:
    /// `[0..weight_count)` = weights, `[weight_count..weight_count+output_dim)` = bias.
    params: [f32; 144],
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
}

impl Default for LinearModel {
    fn default() -> Self {
        Self::new(8, 16).expect("default dimensions are valid")
    }
}

impl LinearModel {
    /// Create a new linear model
    pub fn new(input_dim: usize, output_dim: usize) -> core::result::Result<Self, ModelError> {
        if input_dim.checked_mul(output_dim).map_or(true, |p| p > 128) || output_dim > 16 {
            return Err(ModelError::InvalidDimensions);
        }

        Ok(Self {
            params: [0.0; 144],
            input_dim,
            output_dim,
        })
    }

    fn weight_count(&self) -> usize {
        self.input_dim * self.output_dim
    }

    /// Input dimension.
    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Output dimension.
    pub const fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Initialize with random weights
    pub fn with_random_init(mut self, seed: u64) -> Self {
        // Simple LCG for reproducible random initialization
        let mut state = seed;
        let weight_count = self.weight_count();
        for w in self.params[..weight_count].iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
        }
        let bias_start = weight_count;
        let bias_end = bias_start + self.output_dim;
        for b in self.params[bias_start..bias_end].iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = ((state >> 33) as f32 / (1u64 << 31) as f32) * 0.1;
        }
        self
    }
}

impl SwarmModel for LinearModel {
    type Input = ();
    type Output = ();
    type Error = core::convert::Infallible;

    fn forward(&self, _input: &Self::Input) -> core::result::Result<Self::Output, Self::Error> {
        Ok(())
    }

    fn parameters(&self) -> &[f32] {
        let len = self.weight_count() + self.output_dim;
        &self.params[..len]
    }

    fn parameters_mut(&mut self) -> &mut [f32] {
        let len = self.weight_count() + self.output_dim;
        &mut self.params[..len]
    }

    fn load_parameters(&mut self, params: &[f32]) -> Result<()> {
        let expected = self.weight_count() + self.output_dim;
        if params.len() != expected {
            return Err(swarm_torch_core::Error::InvalidGradient);
        }
        self.params[..expected].copy_from_slice(params);
        Ok(())
    }
}

/// A simple MLP (Multi-Layer Perceptron) for testing
#[derive(Debug, Clone)]
pub struct SimpleMLP {
    /// First layer weights
    pub layer1: LinearModel,
    /// Second layer weights  
    pub layer2: LinearModel,
}

impl SimpleMLP {
    /// Create a new MLP with the given dimensions
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> core::result::Result<Self, ModelError> {
        Ok(Self {
            layer1: LinearModel::new(input_dim, hidden_dim)?,
            layer2: LinearModel::new(hidden_dim, output_dim)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_model_new_rejects_oversized_dims() {
        assert!(matches!(
            LinearModel::new(100, 100),
            Err(ModelError::InvalidDimensions)
        ));
        assert!(matches!(
            LinearModel::new(8, 20),
            Err(ModelError::InvalidDimensions)
        ));
        assert!(LinearModel::new(8, 16).is_ok());
    }

    #[test]
    fn linear_model_parameters_include_bias() {
        let model = LinearModel::new(8, 16).unwrap();
        assert_eq!(model.parameters().len(), 8 * 16 + 16);
    }

    #[test]
    fn linear_model_load_parameters_roundtrip_preserves_bias() {
        let mut model = LinearModel::new(4, 3).unwrap();
        let expected_len = 4 * 3 + 3;
        let source: Vec<f32> = (0..expected_len).map(|i| i as f32 + 0.5).collect();
        model.load_parameters(&source).unwrap();
        assert_eq!(model.parameters(), source.as_slice());
    }

    #[test]
    fn linear_model_forward_returns_ok_with_ref_input() {
        let model = LinearModel::new(8, 16).unwrap();
        let out = model.forward(&());
        assert!(out.is_ok());
    }
}
