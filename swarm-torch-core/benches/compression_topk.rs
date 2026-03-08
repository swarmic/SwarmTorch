//! TopK compression throughput benchmarks (E-01).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use swarm_torch_core::compression::{CompressedGradient, CompressionMethod};

fn make_gradients(n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut state = 0x1234_5678_9abc_def0u64;
    for i in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = ((state >> 16) as u32) as f32 / (u32::MAX as f32);
        let sign = if (i & 1) == 0 { 1.0 } else { -1.0 };
        out.push(sign * (x - 0.5) * 20.0);
    }
    out
}

fn legacy_topk_reference(gradients: &[f32], k_ratio: f32) -> (Vec<u32>, Vec<u8>) {
    let raw = (gradients.len() as f32) * k_ratio;
    let mut k = raw as usize;
    if (k as f32) < raw {
        k = k.saturating_add(1);
    }
    let k = k.max(1).min(gradients.len());

    let mut indexed: Vec<(usize, f32)> =
        gradients.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(k);
    indexed.sort_unstable_by_key(|(idx, _)| *idx);

    (
        indexed.iter().map(|(idx, _)| *idx as u32).collect(),
        indexed.iter().flat_map(|(_, v)| v.to_le_bytes()).collect(),
    )
}

fn bench_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_topk");

    for n in [1_024usize, 16_384, 131_072] {
        let gradients = make_gradients(n);
        let k_ratio = 0.01f32;

        group.bench_with_input(BenchmarkId::new("legacy_full_sort_ref", n), &n, |b, _| {
            b.iter(|| {
                let (i, v) = legacy_topk_reference(black_box(&gradients), black_box(k_ratio));
                black_box((i, v));
            })
        });

        group.bench_with_input(BenchmarkId::new("selection_impl", n), &n, |b, _| {
            b.iter(|| {
                let compressed = CompressedGradient::compress(
                    black_box(&gradients),
                    CompressionMethod::TopK { k_ratio },
                )
                .expect("topk compression should succeed");
                black_box(compressed);
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_topk);
criterion_main!(benches);
