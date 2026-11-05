//! Simple reference model implementations (`ADR-0024`, `W9-04`, `W9-06`).
//!
//! Provides stack-allocated, `no_std`-compatible reference models with contiguous
//! parameter buffers, widened `f64` forward/backward evaluation, and strict
//! finiteness/dimension validation.

pub use crate::mlp::{MlpModel, MlpModelWithCapacity, SimpleMLP};
use swarm_torch_core::traits::SwarmModel;
use swarm_torch_core::{Error, Result};

/// Error type for model creation, forward inference, and backward gradient passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    /// Model dimensions exceed supported size for fixed storage.
    InvalidDimensions,
    /// Input slice length does not match `input_dim`.
    InputDimensionMismatch { expected: usize, got: usize },
    /// Upstream output gradient length does not match `output_dim`.
    GradOutputDimensionMismatch { expected: usize, got: usize },
    /// Destination parameter gradient buffer length does not match `num_parameters()`.
    GradParamsDimensionMismatch { expected: usize, got: usize },
    /// Input contains `NaN` or `±Inf`.
    NonFiniteInput,
    /// Model parameters or upstream gradients contain `NaN` or `±Inf`.
    NonFiniteParameter,
    /// Computed activation or output overflowed to `NaN` or `±Inf`.
    NonFiniteOutput,
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimensions => write!(
                f,
                "model dimensions exceed supported size for fixed storage"
            ),
            Self::InputDimensionMismatch { expected, got } => write!(
                f,
                "model input dimension mismatch: expected {expected}, got {got}"
            ),
            Self::GradOutputDimensionMismatch { expected, got } => write!(
                f,
                "model grad_output dimension mismatch: expected {expected}, got {got}"
            ),
            Self::GradParamsDimensionMismatch { expected, got } => write!(
                f,
                "model grad_params dimension mismatch: expected {expected}, got {got}"
            ),
            Self::NonFiniteInput => write!(f, "non-finite value in model input"),
            Self::NonFiniteParameter => {
                write!(
                    f,
                    "non-finite value in model parameters or upstream gradient"
                )
            }
            Self::NonFiniteOutput => write!(f, "non-finite value produced during model evaluation"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ModelError {}

/// Stack-allocated output buffer returned by [`LinearModelWithCapacity`] and [`MlpModelWithCapacity`].
#[derive(Debug, Clone, PartialEq)]
pub struct LinearOutput<const MAX_OUTPUT: usize = 16> {
    values: [f32; MAX_OUTPUT],
    len: usize,
}

impl<const MAX_OUTPUT: usize> LinearOutput<MAX_OUTPUT> {
    pub(crate) const fn new(values: [f32; MAX_OUTPUT], len: usize) -> Self {
        Self { values, len }
    }

    /// Active output slice of length `output_dim`.
    pub fn as_slice(&self) -> &[f32] {
        &self.values[..self.len]
    }

    /// Number of active output elements.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the active output is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<const MAX_OUTPUT: usize> core::ops::Deref for LinearOutput<MAX_OUTPUT> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub(crate) fn next_unit_f32(&mut self) -> f32 {
        // 53-bit precision path for stable [0, 1) generation.
        let u = self.next_u64() >> 11;
        (u as f64 * (1.0 / ((1u64 << 53) as f64))) as f32
    }
}

pub(crate) fn validate_finite_params(params: &[f32]) -> Result<()> {
    for &p in params {
        if p.is_nan() {
            return Err(Error::NanDetected);
        }
        if p.is_infinite() {
            return Err(Error::InfDetected);
        }
    }
    Ok(())
}

/// Stack-allocated affine linear model $y = W x + b$.
#[derive(Debug, Clone)]
pub struct LinearModelWithCapacity<const MAX_PARAMS: usize = 144, const MAX_OUTPUT: usize = 16> {
    /// Contiguous parameter buffer:
    /// `[0..weight_count)` = row-major weights ($output\_dim \times input\_dim$),
    /// `[weight_count..weight_count+output_dim)` = bias.
    params: [f32; MAX_PARAMS],
    /// Input dimension.
    input_dim: usize,
    /// Output dimension.
    output_dim: usize,
}

/// Default linear model type (`MAX_PARAMS = 144`, `MAX_OUTPUT = 16`).
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
    /// Create a new linear model.
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

    /// Initialize with deterministic seeded weights and biases.
    pub fn with_random_init(mut self, seed: u64) -> Self {
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

    /// Compute analytical parameter gradients $\partial L / \partial W$ and $\partial L / \partial b$
    /// into `grad_params_out` given `input` and upstream `grad_output` ($\partial L / \partial y$).
    pub fn backward_into(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_params_out: &mut [f32],
    ) -> core::result::Result<(), ModelError> {
        if input.len() != self.input_dim {
            return Err(ModelError::InputDimensionMismatch {
                expected: self.input_dim,
                got: input.len(),
            });
        }
        if grad_output.len() != self.output_dim {
            return Err(ModelError::GradOutputDimensionMismatch {
                expected: self.output_dim,
                got: grad_output.len(),
            });
        }
        let total_params = self.weight_count() + self.output_dim;
        if grad_params_out.len() != total_params {
            return Err(ModelError::GradParamsDimensionMismatch {
                expected: total_params,
                got: grad_params_out.len(),
            });
        }
        if !input.iter().all(|v| v.is_finite()) {
            return Err(ModelError::NonFiniteInput);
        }
        if !grad_output.iter().all(|v| v.is_finite()) {
            return Err(ModelError::NonFiniteParameter);
        }

        let weight_count = self.weight_count();
        for (o, &dy_f32) in grad_output.iter().enumerate().take(self.output_dim) {
            let dy = dy_f32 as f64;
            let row_offset = o * self.input_dim;
            for (i, &x) in input.iter().enumerate().take(self.input_dim) {
                let dw = (dy * (x as f64)) as f32;
                if !dw.is_finite() {
                    return Err(ModelError::NonFiniteOutput);
                }
                grad_params_out[row_offset + i] = dw;
            }
            grad_params_out[weight_count + o] = dy_f32;
        }
        Ok(())
    }
}

impl<const MAX_PARAMS: usize, const MAX_OUTPUT: usize> SwarmModel
    for LinearModelWithCapacity<MAX_PARAMS, MAX_OUTPUT>
{
    type Input = [f32];
    type Output = LinearOutput<MAX_OUTPUT>;
    type Error = ModelError;

    fn forward(&self, input: &Self::Input) -> core::result::Result<Self::Output, Self::Error> {
        if input.len() != self.input_dim {
            return Err(ModelError::InputDimensionMismatch {
                expected: self.input_dim,
                got: input.len(),
            });
        }
        if !input.iter().all(|v| v.is_finite()) {
            return Err(ModelError::NonFiniteInput);
        }
        let active_params = self.parameters();
        if !active_params.iter().all(|v| v.is_finite()) {
            return Err(ModelError::NonFiniteParameter);
        }

        let mut values = [0.0f32; MAX_OUTPUT];
        let weight_count = self.weight_count();
        for (o, out_slot) in values.iter_mut().enumerate().take(self.output_dim) {
            let row_offset = o * self.input_dim;
            let mut acc = self.params[weight_count + o] as f64;
            for (i, &x) in input.iter().enumerate().take(self.input_dim) {
                acc += (self.params[row_offset + i] as f64) * (x as f64);
            }
            let out_f32 = acc as f32;
            if !out_f32.is_finite() {
                return Err(ModelError::NonFiniteOutput);
            }
            *out_slot = out_f32;
        }

        Ok(LinearOutput::new(values, self.output_dim))
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
            return Err(Error::InvalidGradient);
        }
        validate_finite_params(params)?;
        self.params[..expected].copy_from_slice(params);
        Ok(())
    }
}

#[cfg(test)]
#[path = "simple_tests.rs"]
mod tests;
