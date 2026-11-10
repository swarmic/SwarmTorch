//! End-to-end benchmarks for SwarmTorch transport framing, SafeTensors checkpointing,
//! and reference model training (`ADR-0022`, `ADR-0024`, `W10-03`, `W11-03`, `W11-04`).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use swarm_torch::core::traits::{SwarmModel, SwarmOptimizer};
use swarm_torch::models::optim::SgdOptimizer;
use swarm_torch::models::safetensors_v1::{
    export_safetensors_checkpoint_v1, import_safetensors_checkpoint_v1, ModelTensorSchemaV1,
    SafeTensorsLimitsV1,
};
use swarm_torch::models::simple::LinearModelWithCapacity;
use swarm_torch::net::chunked::{ChunkedFrameCodec, ObservedSourceReassembler, ReassemblerLimits};

fn swarm_benchmark(c: &mut Criterion) {
    let payload = vec![0xA5u8; 8192];
    c.bench_function("chunked_reassembly_8kb_mtu1200", |b| {
        let limits = ReassemblerLimits::default();
        b.iter(|| {
            let frames = ChunkedFrameCodec::encode_message(
                0x0102_0304,
                limits.mtu,
                limits.max_message_bytes,
                limits.max_chunks_per_transfer,
                black_box(&payload),
            )
            .unwrap();
            let mut reassembler = ObservedSourceReassembler::<u32>::new(limits).unwrap();
            let mut now_ms = 1_000u64;
            let mut completed_len = 0usize;
            for frame in &frames {
                if let Some(msg) = reassembler.ingest_frame(1, 42u32, frame, now_ms).unwrap() {
                    completed_len = msg.payload.len();
                }
                now_ms += 1;
            }
            assert_eq!(completed_len, payload.len());
        })
    });

    type BenchLinear = LinearModelWithCapacity<1040, 16>;
    let model = BenchLinear::new(128, 8).unwrap().with_random_init(77);
    let schema = ModelTensorSchemaV1::for_linear(128, 8).unwrap();
    let limits = SafeTensorsLimitsV1::default();
    c.bench_function("safetensors_checkpoint_v1_roundtrip_128x8", |b| {
        b.iter(|| {
            let bytes =
                export_safetensors_checkpoint_v1(black_box(&model), &schema, &limits).unwrap();
            let mut restored = BenchLinear::new(128, 8).unwrap();
            import_safetensors_checkpoint_v1(&mut restored, &schema, black_box(&bytes), &limits)
                .unwrap()
        })
    });

    let grads = vec![0.01f32; 1032];
    c.bench_function("linear_sgd_step_128x8", |b| {
        let mut train_model = BenchLinear::new(128, 8).unwrap().with_random_init(77);
        let mut opt: SgdOptimizer<BenchLinear, 1040> =
            SgdOptimizer::with_hyperparams(0.01, 0.9, 1e-4).unwrap();
        b.iter(|| {
            opt.step(&mut train_model, black_box(&grads)).unwrap();
            black_box(train_model.parameters()[0])
        })
    });
}

criterion_group!(benches, swarm_benchmark);
criterion_main!(benches);
