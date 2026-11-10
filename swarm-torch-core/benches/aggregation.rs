//! Benchmarks for Byzantine-resilient aggregation and verifiable sparse compression
//! (`ADR-0019`, `ADR-0021`, `W9-01`, `W10-03`, `W11-01`, `W11-02`).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(feature = "bulyan")]
use swarm_torch_core::aggregation::Bulyan;
#[cfg(feature = "krum")]
use swarm_torch_core::aggregation::Krum;
use swarm_torch_core::aggregation::{
    aggregate_with_bucketing_v1, AggregationBudget, BucketRemainderPolicy, BucketingConfig,
    CenteredClipping, CoordinateMedian, FedAvg, PostBucketingRule, RobustAggregator, TrimmedMean,
};
use swarm_torch_core::compression::{
    sample_shared_indices_v1, verify_shared_seed_openings_v1, SharedSeedResidualCompressor,
};
use swarm_torch_core::traits::GradientUpdate;

fn make_bench_updates(n: usize, dim: usize) -> Vec<GradientUpdate> {
    (0..n)
        .map(|i| GradientUpdate {
            sender: [i as u8; 32],
            sequence: i as u64,
            gradients: (0..dim)
                .map(|j| ((i * 13 + j * 7) % 97) as f32 * 0.01)
                .collect(),
            round_id: 1,
        })
        .collect()
}

fn aggregation_benchmark(c: &mut Criterion) {
    let updates = make_bench_updates(15, 256);

    c.bench_function("fedavg_15x256", |b| {
        b.iter(|| FedAvg.aggregate(black_box(&updates)).unwrap())
    });

    c.bench_function("coordinate_median_15x256", |b| {
        b.iter(|| CoordinateMedian.aggregate(black_box(&updates)).unwrap())
    });

    c.bench_function("trimmed_mean_15x256", |b| {
        let agg = TrimmedMean::new(0.2);
        b.iter(|| agg.aggregate(black_box(&updates)).unwrap())
    });

    c.bench_function("cclip_15x256_tau1_l3", |b| {
        let agg = CenteredClipping::try_new(1.0, 3, vec![0.25f32; 256]).unwrap();
        b.iter(|| agg.aggregate(black_box(&updates)).unwrap())
    });

    #[cfg(feature = "krum")]
    c.bench_function("krum_15x256_f2", |b| {
        let agg = Krum::new(2);
        b.iter(|| agg.aggregate(black_box(&updates)).unwrap())
    });

    #[cfg(feature = "krum")]
    c.bench_function("bucketing_krum_15x256_s2_f1", |b| {
        let cfg = BucketingConfig {
            bucket_size: 2,
            byzantine_count: 1,
            round_seed: [0x42; 32],
            remainder_policy: BucketRemainderPolicy::DropRemainderDeterministic,
        };
        let budget = AggregationBudget::default();
        b.iter(|| {
            aggregate_with_bucketing_v1(
                black_box(&updates),
                black_box(&cfg),
                PostBucketingRule::Krum,
                None,
                &budget,
            )
            .unwrap()
        })
    });

    #[cfg(feature = "krum")]
    c.bench_function("multi_krum_15x256_f2_m5", |b| {
        let agg = Krum::multi(2, 5);
        b.iter(|| agg.aggregate(black_box(&updates)).unwrap())
    });

    #[cfg(feature = "bulyan")]
    c.bench_function("bulyan_15x256_f2", |b| {
        let agg = Bulyan::new(2);
        b.iter(|| agg.aggregate(black_box(&updates)).unwrap())
    });

    c.bench_function("floyd_shared_seed_sample_1024_k64", |b| {
        let seed = [0xAB; 32];
        b.iter(|| sample_shared_indices_v1(1024, 64, black_box(&seed)).unwrap())
    });

    c.bench_function("merkle_commit_and_open_256_k32", |b| {
        let grad = &updates[0].gradients;
        let seed = [0xCD; 32];
        b.iter(|| {
            let mut compressor = SharedSeedResidualCompressor::new(256, 32).unwrap();
            let (u, commitment) = compressor.prepare_pre_seed_vector(black_box(grad)).unwrap();
            let (payload, batch) = compressor
                .finalize_with_seed(&u, &commitment, black_box(seed))
                .unwrap();
            verify_shared_seed_openings_v1(&commitment, &payload, &batch).unwrap()
        })
    });
}

criterion_group!(benches, aggregation_benchmark);
criterion_main!(benches);
