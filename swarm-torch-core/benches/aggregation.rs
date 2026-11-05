//! Benchmarks for Byzantine-resilient aggregation algorithms (`ADR-0019`, `W9-01`).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(feature = "bulyan")]
use swarm_torch_core::aggregation::Bulyan;
#[cfg(feature = "krum")]
use swarm_torch_core::aggregation::Krum;
use swarm_torch_core::aggregation::{CoordinateMedian, FedAvg, RobustAggregator, TrimmedMean};
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

    #[cfg(feature = "krum")]
    c.bench_function("krum_15x256_f2", |b| {
        let agg = Krum::new(2);
        b.iter(|| agg.aggregate(black_box(&updates)).unwrap())
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
}

criterion_group!(benches, aggregation_benchmark);
criterion_main!(benches);
