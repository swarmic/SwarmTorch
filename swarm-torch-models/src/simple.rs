//! Simple reference model implementations
//!
//! These models are useful for testing and examples.

use swarm_torch_core::traits::SwarmModel;
use swarm_torch_core::Result;

/// A simple linear model for testing
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Weight matrix (flattened, row-major)
    pub weights: [f32; 128],
    /// Bias vector
    pub bias: [f32; 16],
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
}

impl Default for LinearModel {
    fn default() -> Self {
        Self::new(8, 16)
    }
}

impl LinearModel {
    /// Create a new linear model
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        assert!(
            input_dim * output_dim <= 128,
            "Model too large for fixed storage"
        );
        assert!(output_dim <= 16, "Output dim too large for fixed storage");

        Self {
            weights: [0.0; 128],
            bias: [0.0; 16],
            input_dim,
            output_dim,
        }
    }

    /// Initialize with random weights
    pub fn with_random_init(mut self, seed: u64) -> Self {
        // Simple LCG for reproducible random initialization
        let mut state = seed;
        for w in self.weights.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
        }
        for b in self.bias.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = ((state >> 33) as f32 / (1u64 << 31) as f32) * 0.1;
        }
        self
    }
}

impl SwarmModel for LinearModel {
    type Input = ();
    type Output = ();

    fn forward(&self, _input: Self::Input) -> Self::Output {
        // Placeholder - real implementation would do matrix multiply
        ()
    }

    fn parameters(&self) -> &[f32] {
        // Return weights as slice (bias handled separately for simplicity)
        &self.weights[..self.input_dim * self.output_dim]
    }

    fn parameters_mut(&mut self) -> &mut [f32] {
        &mut self.weights[..self.input_dim * self.output_dim]
    }

    fn load_parameters(&mut self, params: &[f32]) -> Result<()> {
        let expected = self.input_dim * self.output_dim;
        if params.len() != expected {
            return Err(swarm_torch_core::Error::InvalidGradient);
        }
        self.weights[..expected].copy_from_slice(params);
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
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            layer1: LinearModel::new(input_dim, hidden_dim),
            layer2: LinearModel::new(hidden_dim, output_dim),
        }
    }
}
