//! Contiguous two-layer ReLU Multi-Layer Perceptron (`ADR-0024`, `W9-04`, `W9-06`).

use crate::simple::{validate_finite_params, LinearOutput, ModelError, SplitMix64};
use swarm_torch_core::traits::SwarmModel;
use swarm_torch_core::{Error, Result};

/// Two-layer ReLU Multi-Layer Perceptron with a single contiguous parameter buffer (`ADR-0024`).
///
/// Parameter layout inside `params[..total_params]`:
/// 1. `[0 .. w1_count)`: Layer 1 weights ($hidden\_dim \times input\_dim$)
/// 2. `[w1_count .. layer1_end)`: Layer 1 bias ($hidden\_dim$)
/// 3. `[layer1_end .. layer1_end + w2_count)`: Layer 2 weights ($output\_dim \times hidden\_dim$)
/// 4. `[layer1_end + w2_count .. total_params)`: Layer 2 bias ($output\_dim$)
#[derive(Debug, Clone)]
pub struct MlpModelWithCapacity<
    const MAX_PARAMS: usize = 512,
    const MAX_HIDDEN: usize = 16,
    const MAX_OUTPUT: usize = 16,
> {
    params: [f32; MAX_PARAMS],
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    layer1_end: usize,
    total_params: usize,
}

/// Default contiguous MLP model type (`MAX_PARAMS = 512`, `MAX_HIDDEN = 16`, `MAX_OUTPUT = 16`).
pub type MlpModel = MlpModelWithCapacity<512, 16, 16>;
/// Backward-compatible alias for [`MlpModel`].
pub type SimpleMLP = MlpModel;

impl<const MAX_PARAMS: usize, const MAX_HIDDEN: usize, const MAX_OUTPUT: usize>
    MlpModelWithCapacity<MAX_PARAMS, MAX_HIDDEN, MAX_OUTPUT>
{
    /// Create a new two-layer ReLU MLP with checked contiguous parameter bounds.
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> core::result::Result<Self, ModelError> {
        if hidden_dim > MAX_HIDDEN || output_dim > MAX_OUTPUT {
            return Err(ModelError::InvalidDimensions);
        }
        let w1_count = input_dim
            .checked_mul(hidden_dim)
            .ok_or(ModelError::InvalidDimensions)?;
        let layer1_end = w1_count
            .checked_add(hidden_dim)
            .ok_or(ModelError::InvalidDimensions)?;
        let w2_count = hidden_dim
            .checked_mul(output_dim)
            .ok_or(ModelError::InvalidDimensions)?;
        let layer2_count = w2_count
            .checked_add(output_dim)
            .ok_or(ModelError::InvalidDimensions)?;
        let total_params = layer1_end
            .checked_add(layer2_count)
            .ok_or(ModelError::InvalidDimensions)?;
        if total_params > MAX_PARAMS {
            return Err(ModelError::InvalidDimensions);
        }

        Ok(Self {
            params: [0.0; MAX_PARAMS],
            input_dim,
            hidden_dim,
            output_dim,
            layer1_end,
            total_params,
        })
    }

    /// Input dimension.
    pub const fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Hidden layer dimension.
    pub const fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Output dimension.
    pub const fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Number of parameters belonging to Layer 1 (`weights + bias`).
    pub const fn layer1_param_count(&self) -> usize {
        self.layer1_end
    }

    /// Number of parameters belonging to Layer 2 (`weights + bias`).
    pub const fn layer2_param_count(&self) -> usize {
        self.total_params - self.layer1_end
    }

    /// Layer 1 parameter slice (`[weights | bias]`).
    pub fn layer1_parameters(&self) -> &[f32] {
        &self.params[..self.layer1_end]
    }

    /// Layer 2 parameter slice (`[weights | bias]`).
    pub fn layer2_parameters(&self) -> &[f32] {
        &self.params[self.layer1_end..self.total_params]
    }

    /// Initialize both layers with deterministic seeded weights and biases.
    pub fn with_random_init(mut self, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let w1_count = self.input_dim * self.hidden_dim;
        for w in self.params[..w1_count].iter_mut() {
            *w = rng.next_unit_f32() - 0.5;
        }
        for b in self.params[w1_count..self.layer1_end].iter_mut() {
            *b = (rng.next_unit_f32() - 0.5) * 0.1;
        }
        let w2_start = self.layer1_end;
        let w2_end = w2_start + self.hidden_dim * self.output_dim;
        for w in self.params[w2_start..w2_end].iter_mut() {
            *w = rng.next_unit_f32() - 0.5;
        }
        for b in self.params[w2_end..self.total_params].iter_mut() {
            *b = (rng.next_unit_f32() - 0.5) * 0.1;
        }
        self
    }

    /// Compute analytical parameter gradients across the entire contiguous MLP buffer.
    ///
    /// Uses the pinned `ADR-0024` ReLU subgradient convention:
    /// $\operatorname{ReLU}'(z) = 1.0$ for $z > 0.0$ and $0.0$ for $z \le 0.0$.
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
        if grad_params_out.len() != self.total_params {
            return Err(ModelError::GradParamsDimensionMismatch {
                expected: self.total_params,
                got: grad_params_out.len(),
            });
        }
        if !input.iter().all(|v| v.is_finite()) {
            return Err(ModelError::NonFiniteInput);
        }
        if !grad_output.iter().all(|v| v.is_finite())
            || !self.parameters().iter().all(|v| v.is_finite())
        {
            return Err(ModelError::NonFiniteParameter);
        }

        let w1_count = self.input_dim * self.hidden_dim;
        let mut pre_hidden = [0.0f32; MAX_HIDDEN];
        let mut post_hidden = [0.0f32; MAX_HIDDEN];
        for h in 0..self.hidden_dim {
            let row_offset = h * self.input_dim;
            let mut acc = self.params[w1_count + h] as f64;
            for (i, &x) in input.iter().enumerate().take(self.input_dim) {
                acc += (self.params[row_offset + i] as f64) * (x as f64);
            }
            let z = acc as f32;
            if !z.is_finite() {
                return Err(ModelError::NonFiniteOutput);
            }
            pre_hidden[h] = z;
            post_hidden[h] = if z > 0.0 { z } else { 0.0 };
        }

        let w2_start = self.layer1_end;
        let b2_start = w2_start + self.hidden_dim * self.output_dim;
        for o in 0..self.output_dim {
            let dy = grad_output[o] as f64;
            let row_offset = w2_start + o * self.hidden_dim;
            for (h, &act) in post_hidden.iter().enumerate().take(self.hidden_dim) {
                let dw2 = (dy * (act as f64)) as f32;
                if !dw2.is_finite() {
                    return Err(ModelError::NonFiniteOutput);
                }
                grad_params_out[row_offset + h] = dw2;
            }
            grad_params_out[b2_start + o] = grad_output[o];
        }

        for h in 0..self.hidden_dim {
            let dz = if pre_hidden[h] > 0.0 {
                let mut sum_dh = 0.0f64;
                for (o, &dy) in grad_output.iter().enumerate().take(self.output_dim) {
                    let w2_oh = self.params[w2_start + o * self.hidden_dim + h] as f64;
                    sum_dh += (dy as f64) * w2_oh;
                }
                let dz_f32 = sum_dh as f32;
                if !dz_f32.is_finite() {
                    return Err(ModelError::NonFiniteOutput);
                }
                dz_f32
            } else {
                0.0
            };

            let row_offset = h * self.input_dim;
            for (i, &x) in input.iter().enumerate().take(self.input_dim) {
                let dw1 = ((dz as f64) * (x as f64)) as f32;
                if !dw1.is_finite() {
                    return Err(ModelError::NonFiniteOutput);
                }
                grad_params_out[row_offset + i] = dw1;
            }
            grad_params_out[w1_count + h] = dz;
        }

        Ok(())
    }
}

impl<const MAX_PARAMS: usize, const MAX_HIDDEN: usize, const MAX_OUTPUT: usize> SwarmModel
    for MlpModelWithCapacity<MAX_PARAMS, MAX_HIDDEN, MAX_OUTPUT>
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
        if !self.parameters().iter().all(|v| v.is_finite()) {
            return Err(ModelError::NonFiniteParameter);
        }

        let w1_count = self.input_dim * self.hidden_dim;
        let mut hidden = [0.0f32; MAX_HIDDEN];
        for (h, hidden_slot) in hidden.iter_mut().enumerate().take(self.hidden_dim) {
            let row_offset = h * self.input_dim;
            let mut acc = self.params[w1_count + h] as f64;
            for (i, &x) in input.iter().enumerate().take(self.input_dim) {
                acc += (self.params[row_offset + i] as f64) * (x as f64);
            }
            let z = acc as f32;
            if !z.is_finite() {
                return Err(ModelError::NonFiniteOutput);
            }
            *hidden_slot = if z > 0.0 { z } else { 0.0 };
        }

        let w2_start = self.layer1_end;
        let b2_start = w2_start + self.hidden_dim * self.output_dim;
        let mut values = [0.0f32; MAX_OUTPUT];
        for (o, out_slot) in values.iter_mut().enumerate().take(self.output_dim) {
            let row_offset = w2_start + o * self.hidden_dim;
            let mut acc = self.params[b2_start + o] as f64;
            for (h, &act) in hidden.iter().enumerate().take(self.hidden_dim) {
                acc += (self.params[row_offset + h] as f64) * (act as f64);
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
        &self.params[..self.total_params]
    }

    fn parameters_mut(&mut self) -> &mut [f32] {
        &mut self.params[..self.total_params]
    }

    fn load_parameters(&mut self, params: &[f32]) -> Result<()> {
        if params.len() != self.total_params {
            return Err(Error::InvalidGradient);
        }
        validate_finite_params(params)?;
        self.params[..self.total_params].copy_from_slice(params);
        Ok(())
    }
}
