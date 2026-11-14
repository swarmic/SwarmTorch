//! Bounded analytical loss package and local `FedProx` objective regularizer (`ADR-0023`, `ADR-0024`, `W9-06`).
//!
//! All losses validate inputs for finiteness and matching dimensions, accumulate in
//! widened `f64`, and compute analytical gradients into caller-supplied slices so
//! `no_std` targets require zero heap allocations.

/// Error returned by analytical loss or regularizer evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LossError {
    /// Input prediction or target slice is empty.
    EmptyInput,
    /// Prediction, target, or gradient slice lengths do not match.
    DimensionMismatch { expected: usize, got: usize },
    /// Prediction slice contains `NaN` or `±Inf`.
    NonFinitePrediction,
    /// Target slice contains `NaN` or `±Inf`.
    NonFiniteTarget,
    /// Target class index is out of bounds (`target >= num_classes`).
    InvalidTargetClass { target: usize, num_classes: usize },
    /// Hyperparameter (e.g., Huber $\delta$ or FedProx $\mu$) is out of domain.
    InvalidHyperparameter,
    /// Loss or analytical gradient overflowed `f32`.
    NonFiniteResult,
}

impl core::fmt::Display for LossError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "loss input slice must not be empty"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "loss dimension mismatch: expected {expected}, got {got}")
            }
            Self::NonFinitePrediction => write!(f, "non-finite value in model predictions"),
            Self::NonFiniteTarget => write!(f, "non-finite value in target values"),
            Self::InvalidTargetClass {
                target,
                num_classes,
            } => write!(
                f,
                "invalid target class {target} for {num_classes} output classes"
            ),
            Self::InvalidHyperparameter => write!(f, "invalid loss hyperparameter"),
            Self::NonFiniteResult => write!(f, "loss or gradient overflowed finite f32"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LossError {}

/// Portable `f64` exponential (`std` intrinsics when available; IEEE-754 range-reduced polynomial in `no_std`).
#[inline]
pub(crate) fn exp_f64(x: f64) -> f64 {
    swarm_torch_core::num::exp_f64(x)
}

/// Portable `f64` natural logarithm (`std` intrinsics when available; IEEE-754 mantissa reduction in `no_std`).
#[inline]
pub(crate) fn ln_f64(x: f64) -> f64 {
    swarm_torch_core::num::ln_f64(x)
}

/// Mean Squared Error (MSE) loss (`ADR-0024` §2.2):
/// $$L(p, y) = \frac{1}{N} \sum_{i=0}^{N-1} (p_i - y_i)^2, \qquad \frac{\partial L}{\partial p_i} = \frac{2}{N}(p_i - y_i)$$
#[derive(Debug, Clone, Copy, Default)]
pub struct MseLoss;

impl MseLoss {
    /// Compute MSE loss and analytical gradient $\partial L / \partial p$ into `grad_out`.
    pub fn compute_and_grad(
        predictions: &[f32],
        targets: &[f32],
        grad_out: &mut [f32],
    ) -> core::result::Result<f32, LossError> {
        if predictions.is_empty() {
            return Err(LossError::EmptyInput);
        }
        let n = predictions.len();
        if targets.len() != n {
            return Err(LossError::DimensionMismatch {
                expected: n,
                got: targets.len(),
            });
        }
        if grad_out.len() != n {
            return Err(LossError::DimensionMismatch {
                expected: n,
                got: grad_out.len(),
            });
        }
        if !predictions.iter().all(|v| v.is_finite()) {
            return Err(LossError::NonFinitePrediction);
        }
        if !targets.iter().all(|v| v.is_finite()) {
            return Err(LossError::NonFiniteTarget);
        }

        let inv_n = 1.0 / (n as f64);
        let scale_grad = 2.0 * inv_n;
        let mut sum_sq = 0.0f64;

        // Pass 1: validate all scalar gradients and accumulated loss before mutating `grad_out`.
        for i in 0..n {
            let diff = (predictions[i] as f64) - (targets[i] as f64);
            sum_sq += diff * diff;
            let g = (scale_grad * diff) as f32;
            if !g.is_finite() {
                return Err(LossError::NonFiniteResult);
            }
        }

        let loss = (sum_sq * inv_n) as f32;
        if !loss.is_finite() {
            return Err(LossError::NonFiniteResult);
        }

        // Pass 2: commit validated gradients atomically.
        for i in 0..n {
            let diff = (predictions[i] as f64) - (targets[i] as f64);
            grad_out[i] = (scale_grad * diff) as f32;
        }
        Ok(loss)
    }
}

/// Numerically stable Cross-Entropy loss over raw unnormalized logits (`ADR-0024` §2.2):
/// $$L(z, y) = \log\!\left(\sum_{c=0}^{C-1} \exp(z_c - z_{\max})\right) + z_{\max} - z_y, \qquad \frac{\partial L}{\partial z_c} = \operatorname{softmax}(z)_c - \mathbb{I}[c = y]$$
#[derive(Debug, Clone, Copy, Default)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Compute cross-entropy loss and analytical logit gradient $\partial L / \partial z$ into `grad_out`.
    pub fn compute_and_grad(
        logits: &[f32],
        target_class: usize,
        grad_out: &mut [f32],
    ) -> core::result::Result<f32, LossError> {
        if logits.is_empty() {
            return Err(LossError::EmptyInput);
        }
        let num_classes = logits.len();
        if target_class >= num_classes {
            return Err(LossError::InvalidTargetClass {
                target: target_class,
                num_classes,
            });
        }
        if grad_out.len() != num_classes {
            return Err(LossError::DimensionMismatch {
                expected: num_classes,
                got: grad_out.len(),
            });
        }
        if !logits.iter().all(|v| v.is_finite()) {
            return Err(LossError::NonFinitePrediction);
        }

        let mut max_logit = f64::NEG_INFINITY;
        for &z in logits {
            let z64 = z as f64;
            if z64 > max_logit {
                max_logit = z64;
            }
        }

        let mut sum_exp = 0.0f64;
        for &z in logits {
            sum_exp += exp_f64((z as f64) - max_logit);
        }
        if !sum_exp.is_finite() || sum_exp <= 0.0 {
            return Err(LossError::NonFiniteResult);
        }

        let log_z = max_logit + ln_f64(sum_exp);
        let loss = (log_z - (logits[target_class] as f64)) as f32;
        if !loss.is_finite() {
            return Err(LossError::NonFiniteResult);
        }

        // Pass 1: validate every gradient element before mutating `grad_out`.
        for (c, &z) in logits.iter().enumerate() {
            let prob = exp_f64((z as f64) - max_logit) / sum_exp;
            let indicator = if c == target_class { 1.0 } else { 0.0 };
            let g = (prob - indicator) as f32;
            if !g.is_finite() {
                return Err(LossError::NonFiniteResult);
            }
        }

        // Pass 2: commit validated gradients.
        for (c, (&z, g_slot)) in logits.iter().zip(grad_out.iter_mut()).enumerate() {
            let prob = exp_f64((z as f64) - max_logit) / sum_exp;
            let indicator = if c == target_class { 1.0 } else { 0.0 };
            *g_slot = (prob - indicator) as f32;
        }

        Ok(loss)
    }
}

/// Huber loss with validated positive threshold $\delta > 0$ (`ADR-0024` §2.2):
/// $$\ell_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \le \delta \\ \delta\left(|r| - \frac{1}{2}\delta\right) & \text{if } |r| > \delta \end{cases}, \qquad L(p, y) = \frac{1}{N} \sum_{i=0}^{N-1} \ell_\delta(p_i - y_i)$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HuberLoss {
    delta: f32,
}

impl HuberLoss {
    /// Construct a Huber loss with validated finite $\delta > 0$.
    pub fn new(delta: f32) -> core::result::Result<Self, LossError> {
        if !delta.is_finite() || delta <= 0.0 {
            return Err(LossError::InvalidHyperparameter);
        }
        Ok(Self { delta })
    }

    /// Configured transition threshold $\delta$.
    pub const fn delta(&self) -> f32 {
        self.delta
    }

    /// Compute mean Huber loss and analytical gradient $\partial L / \partial p$ into `grad_out`.
    pub fn compute_and_grad(
        &self,
        predictions: &[f32],
        targets: &[f32],
        grad_out: &mut [f32],
    ) -> core::result::Result<f32, LossError> {
        if !self.delta.is_finite() || self.delta <= 0.0 {
            return Err(LossError::InvalidHyperparameter);
        }
        if predictions.is_empty() {
            return Err(LossError::EmptyInput);
        }
        let n = predictions.len();
        if targets.len() != n {
            return Err(LossError::DimensionMismatch {
                expected: n,
                got: targets.len(),
            });
        }
        if grad_out.len() != n {
            return Err(LossError::DimensionMismatch {
                expected: n,
                got: grad_out.len(),
            });
        }
        if !predictions.iter().all(|v| v.is_finite()) {
            return Err(LossError::NonFinitePrediction);
        }
        if !targets.iter().all(|v| v.is_finite()) {
            return Err(LossError::NonFiniteTarget);
        }

        let delta64 = self.delta as f64;
        let inv_n = 1.0 / (n as f64);
        let mut total_loss = 0.0f64;

        // Pass 1: validate every item gradient and final mean loss before mutating `grad_out`.
        for i in 0..n {
            let r = (predictions[i] as f64) - (targets[i] as f64);
            let abs_r = swarm_torch_core::num::abs_f64(r);
            let (item_loss, item_grad) = if abs_r <= delta64 {
                (0.5 * r * r, r * inv_n)
            } else {
                let sign = if r >= 0.0 { 1.0 } else { -1.0 };
                (delta64 * (abs_r - 0.5 * delta64), delta64 * sign * inv_n)
            };
            total_loss += item_loss;
            let g = item_grad as f32;
            if !g.is_finite() {
                return Err(LossError::NonFiniteResult);
            }
        }

        let loss = (total_loss * inv_n) as f32;
        if !loss.is_finite() {
            return Err(LossError::NonFiniteResult);
        }

        // Pass 2: commit validated gradients atomically.
        for i in 0..n {
            let r = (predictions[i] as f64) - (targets[i] as f64);
            let abs_r = swarm_torch_core::num::abs_f64(r);
            let item_grad = if abs_r <= delta64 {
                r * inv_n
            } else {
                let sign = if r >= 0.0 { 1.0 } else { -1.0 };
                delta64 * sign * inv_n
            };
            grad_out[i] = item_grad as f32;
        }
        Ok(loss)
    }
}

/// Local `FedProx` proximal objective regularizer (`ADR-0023` §1.1, `ADR-0024` §2.2):
/// $$R_\mu(w, w_{\mathrm{parent}}) = \frac{\mu}{2} \|w - w_{\mathrm{parent}}\|_2^2, \qquad \nabla_w R_\mu = \mu (w - w_{\mathrm{parent}})$$
///
/// Per `ADR-0023`, `FedProx` proximal regularization is owned **exclusively** by the local
/// training objective layer (never duplicated inside `SgdOptimizer::step`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FedProxRegularizer {
    mu: f32,
}

impl FedProxRegularizer {
    /// Construct a `FedProx` regularizer with validated finite $\mu \ge 0$.
    pub fn new(mu: f32) -> core::result::Result<Self, LossError> {
        if !mu.is_finite() || mu < 0.0 {
            return Err(LossError::InvalidHyperparameter);
        }
        Ok(Self { mu })
    }

    /// Configured proximal coefficient $\mu$.
    pub const fn mu(&self) -> f32 {
        self.mu
    }

    /// Add $\mu (w_i - w_{\mathrm{parent}, i})$ in-place to `param_grads` and return
    /// the proximal penalty $\frac{\mu}{2}\|w - w_{\mathrm{parent}}\|_2^2$.
    pub fn apply_to_gradients(
        &self,
        params: &[f32],
        parent_params: &[f32],
        param_grads: &mut [f32],
    ) -> core::result::Result<f32, LossError> {
        if !self.mu.is_finite() || self.mu < 0.0 {
            return Err(LossError::InvalidHyperparameter);
        }
        if params.is_empty() {
            return Err(LossError::EmptyInput);
        }
        let d = params.len();
        if parent_params.len() != d {
            return Err(LossError::DimensionMismatch {
                expected: d,
                got: parent_params.len(),
            });
        }
        if param_grads.len() != d {
            return Err(LossError::DimensionMismatch {
                expected: d,
                got: param_grads.len(),
            });
        }
        if !params.iter().all(|v| v.is_finite()) {
            return Err(LossError::NonFinitePrediction);
        }
        if !parent_params.iter().all(|v| v.is_finite())
            || !param_grads.iter().all(|v| v.is_finite())
        {
            return Err(LossError::NonFiniteTarget);
        }

        let mu64 = self.mu as f64;
        let mut sq_norm = 0.0f64;

        // Pass 1: validate every updated gradient and proximal penalty before mutating `param_grads`.
        for i in 0..d {
            let diff = (params[i] as f64) - (parent_params[i] as f64);
            sq_norm += diff * diff;
            let updated_g = ((param_grads[i] as f64) + mu64 * diff) as f32;
            if !updated_g.is_finite() {
                return Err(LossError::NonFiniteResult);
            }
        }

        let penalty = (0.5 * mu64 * sq_norm) as f32;
        if !penalty.is_finite() {
            return Err(LossError::NonFiniteResult);
        }

        // Pass 2: commit validated gradient updates atomically.
        for i in 0..d {
            let diff = (params[i] as f64) - (parent_params[i] as f64);
            param_grads[i] = ((param_grads[i] as f64) + mu64 * diff) as f32;
        }
        Ok(penalty)
    }
}

#[cfg(test)]
#[path = "loss_tests.rs"]
mod tests;
