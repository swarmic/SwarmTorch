//! Reference Stochastic Gradient Descent (`SgdOptimizer`) implementation (`ADR-0024`, `W9-06`).
//!
//! Enforces finite positive learning-rate bounds, optional momentum and weight-decay
//! regularization, and transactional parameter updates over any [`SwarmModel`].

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::marker::PhantomData;
#[cfg(feature = "alloc")]
use swarm_torch_core::traits::{OptimizerStateSnapshotV1, OPTIMIZER_STATE_SNAPSHOT_SCHEMA_V1};
use swarm_torch_core::traits::{SwarmModel, SwarmOptimizer};
#[cfg(feature = "alloc")]
use swarm_torch_core::Error as CoreError;

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
    /// Underlying model or device backend rejected the updated parameter vector.
    ModelUpdateFailed,
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
            Self::ModelUpdateFailed => write!(f, "model backend rejected updated parameters"),
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
    fn v_prev(&self, param_len: usize, idx: usize) -> f32 {
        if self.velocity_len == param_len {
            self.velocity[idx]
        } else {
            0.0
        }
    }
}

#[inline]
fn compute_sgd_coord(
    w_i: f32,
    g_i: f32,
    v_prev: f32,
    lr: f64,
    beta: f64,
    wd: f64,
) -> core::result::Result<(f32, f32), OptimError> {
    let w_f64 = w_i as f64;
    let g_reg = (g_i as f64) + wd * w_f64;
    let v_next = beta * (v_prev as f64) + g_reg;
    let w_next = w_f64 - lr * v_next;

    let v_f32 = v_next as f32;
    let w_f32 = w_next as f32;
    if !v_f32.is_finite() || !w_f32.is_finite() {
        return Err(OptimError::NonFiniteParameter);
    }
    Ok((v_f32, w_f32))
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

        #[cfg(feature = "alloc")]
        {
            let current_params = model.parameters();
            let mut next_params = Vec::with_capacity(param_len);
            let mut next_vel = Vec::with_capacity(param_len);
            for i in 0..param_len {
                let (v_f32, w_f32) = compute_sgd_coord(
                    current_params[i],
                    gradients[i],
                    self.v_prev(param_len, i),
                    lr,
                    beta,
                    wd,
                )?;
                next_vel.push(v_f32);
                next_params.push(w_f32);
            }

            // Commit to the model/device backend BEFORE updating momentum velocity so a backend
            // failure leaves both model weights and optimizer state completely untouched.
            model
                .load_parameters(&next_params)
                .map_err(|err| match err {
                    CoreError::NanDetected | CoreError::InfDetected => {
                        OptimError::NonFiniteParameter
                    }
                    _ => OptimError::ModelUpdateFailed,
                })?;
            self.velocity[..param_len].copy_from_slice(&next_vel);
            self.velocity_len = param_len;
        }

        #[cfg(not(feature = "alloc"))]
        {
            // Fixed O(1) stack footprint (512 bytes) even for large `MAX_PARAMS` on RTOS targets:
            // Pass 1 proves every coordinate update stays finite before Pass 2 mutates anything.
            const STACK_CHUNK: usize = 64;
            {
                let current_params = model.parameters();
                for i in 0..param_len {
                    compute_sgd_coord(
                        current_params[i],
                        gradients[i],
                        self.v_prev(param_len, i),
                        lr,
                        beta,
                        wd,
                    )?;
                }
            }
            let mut offset = 0usize;
            while offset < param_len {
                let chunk_len = (param_len - offset).min(STACK_CHUNK);
                let mut w_chunk = [0.0f32; STACK_CHUNK];
                let mut v_chunk = [0.0f32; STACK_CHUNK];
                {
                    let current_params = model.parameters();
                    for j in 0..chunk_len {
                        let idx = offset + j;
                        let (v_f32, w_f32) = compute_sgd_coord(
                            current_params[idx],
                            gradients[idx],
                            self.v_prev(param_len, idx),
                            lr,
                            beta,
                            wd,
                        )?;
                        v_chunk[j] = v_f32;
                        w_chunk[j] = w_f32;
                    }
                }
                self.velocity[offset..offset + chunk_len].copy_from_slice(&v_chunk[..chunk_len]);
                model.parameters_mut()[offset..offset + chunk_len]
                    .copy_from_slice(&w_chunk[..chunk_len]);
                offset += chunk_len;
            }
            self.velocity_len = param_len;
        }

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

    #[cfg(feature = "alloc")]
    fn snapshot_state(&self) -> swarm_torch_core::Result<Option<OptimizerStateSnapshotV1>> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(swarm_torch_core::Error::InvalidThreshold);
        }
        if !self.momentum.is_finite() || !(0.0..1.0).contains(&self.momentum) {
            return Err(swarm_torch_core::Error::InvalidThreshold);
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err(swarm_torch_core::Error::InvalidThreshold);
        }
        if self.velocity_len > MAX_PARAMS {
            return Err(swarm_torch_core::Error::InvalidGradient);
        }
        let mut velocity = alloc::vec::Vec::with_capacity(self.velocity_len);
        for &v in &self.velocity[..self.velocity_len] {
            if !v.is_finite() {
                return Err(swarm_torch_core::Error::NanDetected);
            }
            velocity.push(v);
        }
        Ok(Some(OptimizerStateSnapshotV1 {
            schema_version: OPTIMIZER_STATE_SNAPSHOT_SCHEMA_V1,
            optimizer_kind: alloc::string::String::from("swarmtorch.sgd.v1"),
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            weight_decay: self.weight_decay,
            velocity,
            custom_state: alloc::vec::Vec::new(),
        }))
    }

    #[cfg(feature = "alloc")]
    fn restore_state(
        &mut self,
        snapshot: &OptimizerStateSnapshotV1,
    ) -> swarm_torch_core::Result<()> {
        if snapshot.schema_version != OPTIMIZER_STATE_SNAPSHOT_SCHEMA_V1 {
            return Err(swarm_torch_core::Error::Serialization);
        }
        if snapshot.optimizer_kind != "swarmtorch.sgd.v1" || !snapshot.custom_state.is_empty() {
            return Err(swarm_torch_core::Error::VerificationFailed);
        }
        if !snapshot.learning_rate.is_finite()
            || snapshot.learning_rate <= 0.0
            || !snapshot.momentum.is_finite()
            || !(0.0..1.0).contains(&snapshot.momentum)
            || !snapshot.weight_decay.is_finite()
            || snapshot.weight_decay < 0.0
        {
            return Err(swarm_torch_core::Error::InvalidThreshold);
        }
        if snapshot.velocity.len() > MAX_PARAMS {
            return Err(swarm_torch_core::Error::ResourceExhausted);
        }
        for &v in &snapshot.velocity {
            if !v.is_finite() {
                return Err(swarm_torch_core::Error::NanDetected);
            }
        }
        // All validation passed; atomically apply restored optimizer state.
        self.learning_rate = snapshot.learning_rate;
        self.momentum = snapshot.momentum;
        self.weight_decay = snapshot.weight_decay;
        for slot in self.velocity.iter_mut() {
            *slot = 0.0;
        }
        self.velocity[..snapshot.velocity.len()].copy_from_slice(&snapshot.velocity);
        self.velocity_len = snapshot.velocity.len();
        Ok(())
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

    #[cfg(feature = "alloc")]
    #[test]
    fn sgd_optimizer_snapshot_restore_preserves_momentum_and_rejects_corrupt_state() {
        let mut model_live = LinearModel::new(2, 1).unwrap().with_random_init(17);
        let mut opt_live =
            SgdOptimizer::<LinearModel, 8>::with_hyperparams(0.05, 0.9, 1e-4).unwrap();
        opt_live.step(&mut model_live, &[0.4, -0.3, 0.2]).unwrap();
        let snap = opt_live.snapshot_state().unwrap().expect("sgd snapshot");
        assert_eq!(snap.optimizer_kind, "swarmtorch.sgd.v1");
        assert_eq!(snap.velocity.len(), 3);

        // Restoring into a fresh optimizer produces bit-for-bit identical next step.
        let mut model_restored = model_live.clone();
        let mut opt_restored = SgdOptimizer::<LinearModel, 8>::new(0.9).unwrap();
        opt_restored.restore_state(&snap).unwrap();
        opt_live
            .step(&mut model_live, &[-0.1, 0.25, -0.05])
            .unwrap();
        opt_restored
            .step(&mut model_restored, &[-0.1, 0.25, -0.05])
            .unwrap();
        assert_eq!(model_restored.parameters(), model_live.parameters());
        assert_eq!(opt_restored.velocity(), opt_live.velocity());

        // Fail closed without mutating existing optimizer state when kind/NaN/bounds are invalid.
        let vel_before = opt_restored.velocity().to_vec();
        let mut bad_kind = snap.clone();
        bad_kind.optimizer_kind = alloc::string::String::from("wrong.kind");
        assert!(opt_restored.restore_state(&bad_kind).is_err());
        assert_eq!(opt_restored.velocity(), &vel_before[..]);

        let mut nan_vel = snap.clone();
        nan_vel.velocity[0] = f32::NAN;
        assert!(opt_restored.restore_state(&nan_vel).is_err());
        assert_eq!(opt_restored.velocity(), &vel_before[..]);
    }

    #[test]
    fn sgd_optimizer_multi_chunk_step_and_overflow_atomic_rollback_work_without_alloc() {
        use crate::simple::LinearModelWithCapacity;

        // 150 params (29 * 5 + 5) crosses STACK_CHUNK = 64 boundaries (`no_std` and `alloc`).
        type LargeLinear = LinearModelWithCapacity<256, 16>;
        let mut model = LargeLinear::new(29, 5).unwrap().with_random_init(123);
        let baseline = [1.0f32; 150];
        let mut grads = [0.25f32; 150];
        model.parameters_mut()[..150].copy_from_slice(&baseline);

        let mut opt = SgdOptimizer::<LargeLinear, 256>::with_hyperparams(0.1, 0.5, 0.2).unwrap();
        opt.step(&mut model, &grads).unwrap();

        // Expected first step: v_1 = 0.25 + 0.2 * 1.0 = 0.45; w_1 = 1.0 - 0.1 * 0.45 = 0.955.
        for &w in &model.parameters()[..150] {
            assert!((w - 0.955).abs() < 1e-6, "got {w}");
        }
        for &v in opt.velocity() {
            assert!((v - 0.45).abs() < 1e-6, "got {v}");
        }

        // If coordinate 149 (final chunk!) overflows to +inf/-inf, ZERO coordinates [0..150]
        // and ZERO velocity elements may mutate.
        let snapshot_params = [0.955f32; 150];
        grads[149] = f32::MAX;
        let mut overflow_opt =
            SgdOptimizer::<LargeLinear, 256>::with_hyperparams(1e10, 0.5, 0.0).unwrap();
        assert_eq!(
            overflow_opt.step(&mut model, &grads),
            Err(OptimError::NonFiniteParameter)
        );
        for &w in &model.parameters()[..150] {
            assert!((w - snapshot_params[0]).abs() < 1e-6);
        }
        assert!(overflow_opt.velocity().is_empty());
    }

    #[test]
    fn sgd_optimizer_chunk_boundaries_and_mlp_multi_chunk_work_in_no_std_and_alloc() {
        use crate::mlp::MlpModelWithCapacity;
        use crate::simple::LinearModelWithCapacity;

        // Exact STACK_CHUNK = 64 boundary: Linear(15, 4) -> 15*4 + 4 = 64 params.
        type Exact64Linear = LinearModelWithCapacity<64, 4>;
        let mut m64 = Exact64Linear::new(15, 4).unwrap();
        m64.parameters_mut().fill(1.0);
        let g64 = [0.5f32; 64];
        let mut opt64 = SgdOptimizer::<Exact64Linear, 64>::with_hyperparams(0.1, 0.9, 0.2).unwrap();
        opt64.step(&mut m64, &g64).unwrap();
        // v1 = 0.5 + 0.2*1.0 = 0.7; w1 = 1.0 - 0.1*0.7 = 0.93
        opt64.step(&mut m64, &g64).unwrap();
        // v2 = 0.9*0.7 + (0.5 + 0.2*0.93) = 0.63 + 0.686 = 1.316; w2 = 0.93 - 0.1*1.316 = 0.7984
        for &w in m64.parameters() {
            assert!((w - 0.7984).abs() < 1e-5, "got {w}");
        }
        for &v in opt64.velocity() {
            assert!((v - 1.316).abs() < 1e-5, "got {v}");
        }

        // STACK_CHUNK + 1 = 65 boundary: Linear(64, 1) -> 64 + 1 = 65 params.
        type Edge65Linear = LinearModelWithCapacity<65, 1>;
        let mut m65 = Edge65Linear::new(64, 1).unwrap();
        m65.parameters_mut().fill(2.0);
        let g65 = [-0.25f32; 65];
        let mut opt65 = SgdOptimizer::<Edge65Linear, 65>::with_hyperparams(0.2, 0.5, 0.1).unwrap();
        opt65.step(&mut m65, &g65).unwrap();
        opt65.step(&mut m65, &g65).unwrap();
        // step1: v1 = -0.25 + 0.1*2.0 = -0.05; w1 = 2.0 - 0.2*(-0.05) = 2.01
        // step2: v2 = 0.5*(-0.05) + (-0.25 + 0.1*2.01) = -0.074; w2 = 2.01 - 0.2*(-0.074) = 2.0248
        for &w in m65.parameters() {
            assert!((w - 2.0248).abs() < 1e-5, "got {w}");
        }

        // Zero-param Linear(0, 0) works cleanly when gradients == &[], and rejects stale velocity.
        type ZeroLinear = LinearModelWithCapacity<4, 1>;
        let mut m0 = ZeroLinear::new(0, 0).unwrap();
        let mut opt0 = SgdOptimizer::<ZeroLinear, 4>::new(0.1).unwrap();
        opt0.step(&mut m0, &[]).unwrap();
        assert!(opt0.velocity().is_empty());

        // MlpModel multi-chunk (>64 params): 8 -> 10 -> 2 => 8*10 + 10 + 10*2 + 2 = 112 params.
        type LargeMlp = MlpModelWithCapacity<128, 16, 4>;
        let mut mlp = LargeMlp::new(8, 10, 2).unwrap().with_random_init(777);
        assert_eq!(mlp.parameters().len(), 112);
        let grads_mlp = [0.01f32; 112];
        let mut opt_mlp = SgdOptimizer::<LargeMlp, 128>::with_hyperparams(0.05, 0.9, 0.01).unwrap();
        opt_mlp.step(&mut mlp, &grads_mlp).unwrap();
        opt_mlp.step(&mut mlp, &grads_mlp).unwrap();
        assert_eq!(opt_mlp.velocity().len(), 112);
        assert!(mlp.forward(&[0.5f32; 8]).is_ok());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn sgd_optimizer_restore_state_enforces_bounds_empty_velocity_and_dimension_mismatch() {
        let mut model = LinearModel::new(2, 1).unwrap().with_random_init(33);
        let mut opt = SgdOptimizer::<LinearModel, 4>::with_hyperparams(0.1, 0.9, 0.05).unwrap();
        opt.step(&mut model, &[0.5, -0.5, 0.25]).unwrap();
        let valid_snap = opt.snapshot_state().unwrap().unwrap();

        // Reject invalid schema_version, non-empty custom_state, out-of-range momentum/wd/lr,
        // and velocity exceeding MAX_PARAMS without mutating prior optimizer state.
        let prior_vel = opt.velocity().to_vec();

        let mut bad_ver = valid_snap.clone();
        bad_ver.schema_version = OPTIMIZER_STATE_SNAPSHOT_SCHEMA_V1 + 1;
        assert!(opt.restore_state(&bad_ver).is_err());

        let mut bad_custom = valid_snap.clone();
        bad_custom.custom_state = alloc::vec![1u8];
        assert!(opt.restore_state(&bad_custom).is_err());

        let mut bad_mom = valid_snap.clone();
        bad_mom.momentum = 1.0;
        assert!(opt.restore_state(&bad_mom).is_err());

        let mut bad_wd = valid_snap.clone();
        bad_wd.weight_decay = -1e-6;
        assert!(opt.restore_state(&bad_wd).is_err());

        let mut bad_len = valid_snap.clone();
        bad_len.velocity = alloc::vec![0.0; 5]; // > MAX_PARAMS = 4
        assert!(opt.restore_state(&bad_len).is_err());
        assert_eq!(opt.velocity(), prior_vel.as_slice());

        // Restoring a snapshot with mismatched velocity length fails cleanly on subsequent step().
        let mut small_model = LinearModel::new(1, 1).unwrap(); // 2 params != 3 velocity
        assert_eq!(
            opt.step(&mut small_model, &[0.1, 0.2]),
            Err(OptimError::DimensionMismatch {
                expected: 3,
                got: 2,
            })
        );
        assert_eq!(small_model.parameters(), &[0.0, 0.0]);
        assert_eq!(opt.velocity(), prior_vel.as_slice());

        // Restoring an empty-velocity snapshot clears prior velocity_len so a fresh step initializes cleanly.
        let mut empty_vel_snap = valid_snap;
        empty_vel_snap.velocity.clear();
        opt.restore_state(&empty_vel_snap).unwrap();
        assert!(opt.velocity().is_empty());
        opt.step(&mut small_model, &[0.1, 0.2]).unwrap();
        assert_eq!(opt.velocity().len(), 2);
    }
}
