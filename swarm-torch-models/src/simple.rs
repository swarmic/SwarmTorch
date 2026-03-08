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

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_unit_f32(&mut self) -> f32 {
        // 53-bit precision path for stable [0, 1) generation.
        let u = self.next_u64() >> 11;
        (u as f64 * (1.0 / ((1u64 << 53) as f64))) as f32
    }
}

#[derive(Debug, Clone)]
pub struct LinearModelWithCapacity<const MAX_PARAMS: usize = 144, const MAX_OUTPUT: usize = 16> {
    /// Parameter buffer:
    /// `[0..weight_count)` = weights, `[weight_count..weight_count+output_dim)` = bias.
    params: [f32; MAX_PARAMS],
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
}

/// Backward-compatible default linear model type.
pub type LinearModel = LinearModelWithCapacity<144, 16>;

impl<const MAX_PARAMS: usize, const MAX_OUTPUT: usize> Default
    for LinearModelWithCapacity<MAX_PARAMS, MAX_OUTPUT>
{
    fn default() -> Self {
        let output_dim = MAX_OUTPUT.min(16).min(MAX_PARAMS);
        let max_weight_capacity = MAX_PARAMS.saturating_sub(MAX_OUTPUT);
        let input_dim = if output_dim == 0 {
            0
        } else {
            (max_weight_capacity / output_dim).min(8)
        };
        Self::new(input_dim, output_dim).expect("default dimensions are valid for model capacity")
    }
}

impl<const MAX_PARAMS: usize, const MAX_OUTPUT: usize>
    LinearModelWithCapacity<MAX_PARAMS, MAX_OUTPUT>
{
    /// Create a new linear model
    pub fn new(input_dim: usize, output_dim: usize) -> core::result::Result<Self, ModelError> {
        let weight_count = input_dim
            .checked_mul(output_dim)
            .ok_or(ModelError::InvalidDimensions)?;
        let max_weight_capacity = MAX_PARAMS.saturating_sub(MAX_OUTPUT);
        let total = weight_count
            .checked_add(output_dim)
            .ok_or(ModelError::InvalidDimensions)?;
        if weight_count > max_weight_capacity
            || output_dim > MAX_OUTPUT
            || output_dim > MAX_PARAMS
            || total > MAX_PARAMS
        {
            return Err(ModelError::InvalidDimensions);
        }

        Ok(Self {
            params: [0.0; MAX_PARAMS],
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
        // E-02: SplitMix64 gives stronger deterministic seeded initialization than LCG.
        let mut rng = SplitMix64::new(seed);
        let weight_count = self.weight_count();
        for w in self.params[..weight_count].iter_mut() {
            *w = rng.next_unit_f32() - 0.5;
        }
        let bias_start = weight_count;
        let bias_end = bias_start + self.output_dim;
        for b in self.params[bias_start..bias_end].iter_mut() {
            *b = (rng.next_unit_f32() - 0.5) * 0.1;
        }
        self
    }
}

impl<const MAX_PARAMS: usize, const MAX_OUTPUT: usize> SwarmModel
    for LinearModelWithCapacity<MAX_PARAMS, MAX_OUTPUT>
{
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

    #[test]
    fn linear_model_init_is_seed_deterministic() {
        let a = LinearModel::new(8, 16).unwrap().with_random_init(42);
        let b = LinearModel::new(8, 16).unwrap().with_random_init(42);
        let c = LinearModel::new(8, 16).unwrap().with_random_init(43);

        assert_eq!(a.parameters(), b.parameters());
        assert_ne!(a.parameters(), c.parameters());
    }

    #[test]
    fn linear_model_init_distribution_sanity() {
        let model = LinearModel::new(8, 16).unwrap().with_random_init(1337);
        let params = model.parameters();
        let n = params.len() as f32;
        let mean = params.iter().copied().sum::<f32>() / n;
        let var = params
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f32>()
            / n;

        assert!(mean.abs() < 0.1, "mean drift too large: {mean}");
        assert!(var > 1.0e-4, "variance too small: {var}");
        assert!(var < 0.2, "variance too large: {var}");
    }

    #[test]
    fn linear_model_const_generic_capacity_bounds() {
        type TinyLinear = LinearModelWithCapacity<32, 4>;

        assert!(TinyLinear::new(7, 4).is_ok(), "7*4=28 fits");
        assert!(
            matches!(TinyLinear::new(8, 4), Err(ModelError::InvalidDimensions)),
            "8*4=32 exceeds MAX_PARAMS-MAX_OUTPUT (28)"
        );
        assert!(
            matches!(TinyLinear::new(7, 5), Err(ModelError::InvalidDimensions)),
            "output_dim exceeds MAX_OUTPUT"
        );
    }

    #[test]
    fn linear_model_rejects_output_dim_exceeding_param_capacity() {
        type WeirdLinear = LinearModelWithCapacity<8, 32>;
        assert!(
            matches!(WeirdLinear::new(0, 16), Err(ModelError::InvalidDimensions)),
            "output region must fit inside MAX_PARAMS even when MAX_OUTPUT is larger"
        );
    }
}
