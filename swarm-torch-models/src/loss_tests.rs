use super::*;

#[test]
fn mse_loss_matches_finite_difference_oracle() {
    let preds = [1.5f32, -0.5, 2.25, 0.0];
    let targets = [1.0f32, 0.25, 1.5, -0.75];
    let mut grad = [0.0f32; 4];
    let loss = MseLoss::compute_and_grad(&preds, &targets, &mut grad).unwrap();
    assert!(loss > 0.0 && loss.is_finite());

    let eps = 1e-3f32;
    for (i, &g_analytical) in grad.iter().enumerate() {
        let mut p_plus = preds;
        let mut p_minus = preds;
        p_plus[i] += eps;
        p_minus[i] -= eps;
        let mut dummy = [0.0f32; 4];
        let l_plus = MseLoss::compute_and_grad(&p_plus, &targets, &mut dummy).unwrap();
        let l_minus = MseLoss::compute_and_grad(&p_minus, &targets, &mut dummy).unwrap();
        let fd = (l_plus - l_minus) / (2.0 * eps);
        assert!(
            (g_analytical - fd).abs() < 1e-3,
            "MSE grad mismatch at {i}: analytical={g_analytical} fd={fd}"
        );
    }
}

#[test]
fn cross_entropy_loss_matches_finite_difference_and_handles_extreme_logits() {
    let logits = [0.7f32, -1.2, 2.1, 0.1];
    let target = 2usize;
    let mut grad = [0.0f32; 4];
    let loss = CrossEntropyLoss::compute_and_grad(&logits, target, &mut grad).unwrap();
    assert!(loss > 0.0 && loss.is_finite());

    let eps = 1e-3f32;
    for (i, &g_analytical) in grad.iter().enumerate() {
        let mut z_plus = logits;
        let mut z_minus = logits;
        z_plus[i] += eps;
        z_minus[i] -= eps;
        let mut dummy = [0.0f32; 4];
        let l_plus = CrossEntropyLoss::compute_and_grad(&z_plus, target, &mut dummy).unwrap();
        let l_minus = CrossEntropyLoss::compute_and_grad(&z_minus, target, &mut dummy).unwrap();
        let fd = (l_plus - l_minus) / (2.0 * eps);
        assert!(
            (g_analytical - fd).abs() < 1e-3,
            "CE grad mismatch at {i}: analytical={g_analytical} fd={fd}"
        );
    }

    // Extreme logits (+500 / -500) must not overflow exp() due to log-sum-exp shift.
    let extreme = [-500.0f32, 500.0, -500.0];
    let mut extreme_grad = [0.0f32; 3];
    let extreme_loss = CrossEntropyLoss::compute_and_grad(&extreme, 1, &mut extreme_grad).unwrap();
    assert!(extreme_loss.abs() < 1e-5);
    assert!(extreme_grad.iter().all(|g| g.is_finite()));
}

#[test]
fn huber_loss_validates_delta_and_matches_piecewise_oracle() {
    assert_eq!(HuberLoss::new(0.0), Err(LossError::InvalidHyperparameter));
    assert_eq!(HuberLoss::new(-1.0), Err(LossError::InvalidHyperparameter));
    assert_eq!(
        HuberLoss::new(f32::NAN),
        Err(LossError::InvalidHyperparameter)
    );

    let huber = HuberLoss::new(1.0).unwrap();
    // Residuals: r = [0.5 (quadratic), -2.0 (linear)]
    let preds = [1.5f32, -1.0];
    let targets = [1.0f32, 1.0];
    let mut grad = [0.0f32; 2];
    let loss = huber.compute_and_grad(&preds, &targets, &mut grad).unwrap();
    // Item 0: 0.5 * 0.25 = 0.125; Item 1: 1.0 * (2.0 - 0.5) = 1.5; mean = 0.8125
    assert!((loss - 0.8125).abs() < 1e-6);
    // Item 0 grad: 0.5 / 2 = 0.25; Item 1 grad: -1.0 / 2 = -0.5
    assert!((grad[0] - 0.25).abs() < 1e-6);
    assert!((grad[1] - (-0.5)).abs() < 1e-6);
}

#[test]
fn fedprox_regularizer_adds_single_owner_proximal_term() {
    let reg = FedProxRegularizer::new(0.1).unwrap();
    let params = [2.0f32, -1.0];
    let parent = [1.0f32, 1.0];
    let mut grads = [0.5f32, -0.25];
    let penalty = reg
        .apply_to_gradients(&params, &parent, &mut grads)
        .unwrap();
    // ||w - w_parent||^2 = 1^2 + (-2)^2 = 5 => 0.5 * 0.1 * 5 = 0.25
    assert!((penalty - 0.25).abs() < 1e-6);
    assert!((grads[0] - 0.6).abs() < 1e-6);
    assert!((grads[1] - (-0.45)).abs() < 1e-6);
}
