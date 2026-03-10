#[cfg(feature = "krum")]
use swarm_torch_core::aggregation::Krum;
use swarm_torch_core::aggregation::{CoordinateMedian, FedAvg, RobustAggregator, TrimmedMean};
use swarm_torch_core::traits::GradientUpdate;

#[derive(Clone, Copy)]
enum AttackFamily {
    Outlier,
    SignFlip,
    NaNInf,
    CollusionCluster,
}

fn next_rand(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let bits = ((*seed >> 32) as u32) as f32;
    bits / (u32::MAX as f32)
}

fn target_gradient(dim: usize) -> Vec<f32> {
    vec![1.0; dim]
}

fn l2_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn benign_update(dim: usize, seed: &mut u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(dim);
    for _ in 0..dim {
        let noise = (next_rand(seed) - 0.5) * 0.04;
        out.push(1.0 + noise);
    }
    out
}

fn malicious_update(dim: usize, attack: AttackFamily) -> Vec<f32> {
    match attack {
        AttackFamily::Outlier => vec![50.0; dim],
        AttackFamily::SignFlip => vec![-3.0; dim],
        AttackFamily::NaNInf => {
            let mut v = vec![f32::NAN; dim];
            if dim > 0 {
                v[0] = f32::INFINITY;
            }
            v
        }
        AttackFamily::CollusionCluster => vec![7.5; dim],
    }
}

fn make_round_updates(
    n: usize,
    f: usize,
    dim: usize,
    attack: AttackFamily,
    seed: u64,
) -> (Vec<GradientUpdate>, Vec<f32>) {
    assert!(n > f, "n must be greater than f");
    let mut updates = Vec::with_capacity(n);
    let target = target_gradient(dim);
    let mut prng = seed;

    for i in 0..(n - f) {
        updates.push(GradientUpdate {
            sender: [i as u8; 32],
            sequence: i as u64,
            gradients: benign_update(dim, &mut prng),
            round_id: 1,
        });
    }
    for i in (n - f)..n {
        updates.push(GradientUpdate {
            sender: [i as u8; 32],
            sequence: i as u64,
            gradients: malicious_update(dim, attack),
            round_id: 1,
        });
    }

    (updates, target)
}

#[test]
fn robustness_harness_trimmed_mean_outperforms_fedavg_under_outlier_attack() {
    let (updates, target) = make_round_updates(13, 3, 64, AttackFamily::Outlier, 0x5EED_CAFE);

    let fedavg = FedAvg.aggregate(&updates).expect("fedavg should aggregate");
    let trimmed = TrimmedMean::new(0.25)
        .aggregate(&updates)
        .expect("trimmed mean should aggregate");
    let median = CoordinateMedian
        .aggregate(&updates)
        .expect("median should aggregate");

    let fedavg_error = l2_error(&fedavg, &target);
    let trimmed_error = l2_error(&trimmed, &target);
    let median_error = l2_error(&median, &target);

    assert!(
        trimmed_error < fedavg_error,
        "trimmed_mean should improve over fedavg under outlier attack: trimmed={trimmed_error}, fedavg={fedavg_error}"
    );
    assert!(
        median_error < fedavg_error,
        "median should improve over fedavg under outlier attack: median={median_error}, fedavg={fedavg_error}"
    );
}

#[test]
fn robustness_harness_collusion_cluster_degrades_fedavg_more_than_median() {
    let (updates, target) =
        make_round_updates(15, 4, 64, AttackFamily::CollusionCluster, 0x1234_5678);

    let fedavg = FedAvg.aggregate(&updates).expect("fedavg should aggregate");
    let median = CoordinateMedian
        .aggregate(&updates)
        .expect("median should aggregate");

    let fedavg_error = l2_error(&fedavg, &target);
    let median_error = l2_error(&median, &target);
    assert!(
        median_error < fedavg_error,
        "median should remain closer to baseline under collusion cluster attack: median={median_error}, fedavg={fedavg_error}"
    );
}

#[test]
fn robustness_harness_sign_flip_attack_penalizes_fedavg() {
    let (updates, target) =
        make_round_updates(11, 2, 32, AttackFamily::SignFlip, 0x000A_11CE_5EED);
    let fedavg = FedAvg.aggregate(&updates).expect("fedavg should aggregate");
    let trimmed = TrimmedMean::new(0.2)
        .aggregate(&updates)
        .expect("trimmed mean should aggregate");
    let fedavg_error = l2_error(&fedavg, &target);
    let trimmed_error = l2_error(&trimmed, &target);
    assert!(
        trimmed_error < fedavg_error,
        "trimmed mean should reduce sign-flip distortion: trimmed={trimmed_error}, fedavg={fedavg_error}"
    );
}

#[test]
fn robustness_harness_nan_inf_attack_surfaces_non_finite_outputs() {
    let (updates, _) = make_round_updates(9, 2, 16, AttackFamily::NaNInf, 0xBAD5EED);
    let fedavg = FedAvg.aggregate(&updates).expect("fedavg should aggregate");
    assert!(
        fedavg.iter().any(|v| !v.is_finite()),
        "non-finite attack inputs should be observable in fedavg output to enable downstream guards"
    );
}

#[cfg(feature = "krum")]
#[test]
fn robustness_harness_krum_tolerates_f_byzantine_within_bound() {
    let n = 13usize;
    let f = 3usize;
    let (updates, target) = make_round_updates(n, f, 64, AttackFamily::SignFlip, 0xC001_CAFE);
    let krum = Krum::new(f);
    let result = krum
        .aggregate(&updates)
        .expect("krum should aggregate when n >= 2f+3");
    let error = l2_error(&result, &target);

    assert!(
        error < 2.0,
        "krum output should stay close to benign baseline within bound: error={error}"
    );
}
