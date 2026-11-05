//! Reference Stochastic Gradient Descent (`SgdOptimizer`) implementation (`ADR-0024`, `W9-06`).
//!
//! Enforces finite positive learning-rate bounds, optional momentum and weight-decay
//! regularization, and transactional parameter updates over any [`SwarmModel`].

use core::marker::PhantomData;
use swarm_torch_core::traits::{SwarmModel, SwarmOptimizer};

/// Error returned by [`SgdOptimizer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimError {
    /// Learning rate must be finite and strictly positive ($\eta > 0$).
    InvalidLearningRate,
    /// Momentum must be finite and in $[0.0, 1.0)$.
    InvalidMomentum,
    /// Weight decay must be finite and non-negative ($\lambda \ge 0$).
    InvalidWeightDecay,
    /// Gradient slice length does not match `model.num_parameters()` or exceeds `MAX_PARAMS`.
    DimensionMismatch { expected: usize, got: usize },
    /// Gradient slice contains `NaN` or `±Inf`.
    NonFiniteGradient,
    /// Model parameters or updated parameters contain `NaN` or `±Inf`.
    NonFiniteParameter,
}

impl core::fmt::Display for OptimError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidLearningRate => {
                write!(f, "learning rate must be finite and strictly positive")
            }
            Self::InvalidMomentum => write!(f, "momentum must be finite and in [0.0, 1.0)"),
            Self::InvalidWeightDecay => write!(f, "weight decay must be finite and non-negative"),
            Self::DimensionMismatch { expected, got } => write!(
                f,
                "optimizer parameter dimension mismatch: expected {expected}, got {got}"
            ),
            Self::NonFiniteGradient => write!(f, "non-finite value in optimizer gradient input"),
            Self::NonFiniteParameter => write!(f, "non-finite value in model parameters"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for OptimError {}

/// Stack-allocated Stochastic Gradient Descent optimizer with optional momentum and weight decay.
///
/// Update convention (`ADR-0024` §2.2):
/// $$g_t^{(i)} = \nabla_i L(w_t) + \lambda w_t^{(i)}, \qquad v_{t+1}^{(i)} = \beta v_t^{(i)} + g_t^{(i)}, \qquad w_{t+1}^{(i)} = w_t^{(i)} - \eta v_{t+1}^{(i)}$$
#[derive(Debug, Clone)]
pub struct SgdOptimizer<M: SwarmModel, const MAX_PARAMS: usize = 512> {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    velocity: [f32; MAX_PARAMS],
    velocity_len: usize,
    _marker: PhantomData<fn(&mut M)>,
}

impl<M: SwarmModel, const MAX_PARAMS: usize> SgdOptimizer<M, MAX_PARAMS> {
    /// Create a new `SgdOptimizer` with validated finite learning rate $\eta > 0$.
    pub fn new(learning_rate: f32) -> core::result::Result<Self, OptimError> {
        Self::with_hyperparams(learning_rate, 0.0, 0.0)
    }

    /// Create a new `SgdOptimizer` with validated learning rate, momentum $\beta \in [0, 1)$, and weight decay $\lambda \ge 0$.
    pub fn with_hyperparams(
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
    ) -> core::result::Result<Self, OptimError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(OptimError::InvalidLearningRate);
        }
        if !momentum.is_finite() || !(0.0..1.0).contains(&momentum) {
            return Err(OptimError::InvalidMomentum);
        }
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            return Err(OptimError::InvalidWeightDecay);
        }
        Ok(Self {
            learning_rate,
            momentum,
            weight_decay,
            velocity: [0.0; MAX_PARAMS],
            velocity_len: 0,
            _marker: PhantomData,
        })
    }

    /// Fallible setter for learning rate $\eta > 0$.
    pub fn try_set_learning_rate(&mut self, lr: f32) -> core::result::Result<(), OptimError> {
        if !lr.is_finite() || lr <= 0.0 {
            return Err(OptimError::InvalidLearningRate);
        }
        self.learning_rate = lr;
        Ok(())
    }

    /// Reset accumulated momentum velocity to zero.
    pub fn reset_velocity(&mut self) {
        for v in self.velocity[..self.velocity_len].iter_mut() {
            *v = 0.0;
        }
        self.velocity_len = 0;
    }

    /// Active velocity buffer slice.
    pub fn velocity(&self) -> &[f32] {
        &self.velocity[..self.velocity_len]
    }
}

impl<M: SwarmModel, const MAX_PARAMS: usize> SwarmOptimizer for SgdOptimizer<M, MAX_PARAMS> {
    type Model = M;
    type Error = OptimError;

    fn step(
        &mut self,
        model: &mut Self::Model,
        gradients: &[f32],
    ) -> core::result::Result<(), Self::Error> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(OptimError::InvalidLearningRate);
        }
        let param_len = model.parameters().len();
        if gradients.len() != param_len || param_len > MAX_PARAMS {
            return Err(OptimError::DimensionMismatch {
                expected: param_len,
                got: gradients.len(),
            });
        }
        if !gradients.iter().all(|g| g.is_finite()) {
            return Err(OptimError::NonFiniteGradient);
        }
        if !model.parameters().iter().all(|p| p.is_finite()) {
            return Err(OptimError::NonFiniteParameter);
        }
        if self.velocity_len != 0 && self.velocity_len != param_len {
            return Err(OptimError::DimensionMismatch {
                expected: self.velocity_len,
                got: param_len,
            });
        }

        let lr = self.learning_rate as f64;
        let beta = self.momentum as f64;
        let wd = self.weight_decay as f64;

        let mut next_params = [0.0f32; MAX_PARAMS];
        let mut next_vel = [0.0f32; MAX_PARAMS];
        let current_params = model.parameters();

        for i in 0..param_len {
            let w_i = current_params[i] as f64;
            let g_i = (gradients[i] as f64) + wd * w_i;
            let v_prev = if self.velocity_len == param_len {
                self.velocity[i] as f64
            } else {
                0.0
            };
            let v_next = beta * v_prev + g_i;
            let w_next = w_i - lr * v_next;

            let v_f32 = v_next as f32;
            let w_f32 = w_next as f32;
            if !v_f32.is_finite() || !w_f32.is_finite() {
                return Err(OptimError::NonFiniteParameter);
            }
            next_vel[i] = v_f32;
            next_params[i] = w_f32;
        }

        self.velocity[..param_len].copy_from_slice(&next_vel[..param_len]);
        self.velocity_len = param_len;
        model
            .parameters_mut()
            .copy_from_slice(&next_params[..param_len]);
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        if lr.is_finite() && lr > 0.0 {
            self.learning_rate = lr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::MseLoss;
    use crate::simple::LinearModel;

    #[test]
    fn sgd_optimizer_validates_hyperparameters_and_reduces_mse_loss() {
        assert_eq!(
            SgdOptimizer::<LinearModel>::new(0.0).unwrap_err(),
            OptimError::InvalidLearningRate
        );
        assert_eq!(
            SgdOptimizer::<LinearModel>::with_hyperparams(0.1, 1.0, 0.0).unwrap_err(),
            OptimError::InvalidMomentum
        );
        assert_eq!(
            SgdOptimizer::<LinearModel>::with_hyperparams(0.1, 0.0, -0.1).unwrap_err(),
            OptimError::InvalidWeightDecay
        );

        let mut model = LinearModel::new(2, 1).unwrap().with_random_init(99);
        let mut opt = SgdOptimizer::<LinearModel>::with_hyperparams(0.05, 0.9, 1e-4).unwrap();
        let x = [1.5f32, -2.0];
        let y = [3.0f32];

        let out0 = model.forward(&x).unwrap();
        let mut grad_y = [0.0f32; 1];
        let loss0 = MseLoss::compute_and_grad(&out0, &y, &mut grad_y).unwrap();

        let mut grad_params = [0.0f32; 3];
        model.backward_into(&x, &grad_y, &mut grad_params).unwrap();
        opt.step(&mut model, &grad_params).unwrap();

        let out1 = model.forward(&x).unwrap();
        let loss1 = MseLoss::compute_and_grad(&out1, &y, &mut grad_y).unwrap();
        assert!(
            loss1 < loss0,
            "one SGD step should reduce loss: before={loss0} after={loss1}"
        );
    }

    #[test]
    fn sgd_optimizer_rejects_non_finite_gradient_without_mutating_model() {
        let mut model = LinearModel::new(2, 1).unwrap().with_random_init(7);
        let snapshot: [f32; 3] = [
            model.parameters()[0],
            model.parameters()[1],
            model.parameters()[2],
        ];
        let mut opt = SgdOptimizer::<LinearModel>::new(0.1).unwrap();
        assert_eq!(
            opt.step(&mut model, &[1.0, f32::NAN, 0.0]),
            Err(OptimError::NonFiniteGradient)
        );
        assert_eq!(model.parameters(), &snapshot[..]);
    }
}
