# SwarmTorch (`swarm-torch`)

**Mission-grade swarm learning for heterogeneous fleets — Rust-native, with a `no_std` core and Byzantine-resilient security posture.**

`swarm-torch` is the top-level orchestration, round coordination, and artifact verification crate for the [SwarmTorch](https://github.com/swarmic/SwarmTorch) workspace:

- **`swarm-torch-core`**: `no_std`-compatible Byzantine-resilient aggregators (`FedAvg`, `CoordinateMedian`, `TrimmedMean`, `Krum`/`Multi-Krum`, `Bulyan IterativeKrumV1`, stateful `CenteredClippingAggregator`), Non-IID deterministic `SBucketingTransformer`, error-feedback & verifiable sparse codecs (`TopK`, `RandomK`, `Quantized`, `ClassicResidual`, `EF21`, `SharedSeedSparse`), Ed25519 domain-separated signing, and safe `BTreeMap` replay protection.
- **`swarm-torch-models`**: `no_std`-compatible reference models (`LinearModel`, `MlpModel`), analytical loss functions (`MseLoss`, `CrossEntropyLoss`, `HuberLoss`, `FedProxRegularizer`), `SgdOptimizer`, and zero-dependency `SwarmTorchCheckpointV1` SafeTensors v1 persistence.
- **`swarm-torch-net`**: `PacketTransport` & GAT-based `SwarmTransport` traits, `MessageEnvelope` v1 6-stage atomic receive pipeline, originator-authenticated gossip wrappers, 16-byte `ChunkedFrameCodec` (`CRC-16/IBM-3740`) low-MTU reassembly, and optional `TcpPacketTransport` (`tcp-transport`) / `UdpPacketTransport` (`udp-transport`).
- **`swarm-torch-runtime`**: `SwarmClock` and `SwarmRuntime` abstractions across Tokio (`std`), deterministic `MockRuntime`, and `no_std` spin/Embassy surfaces.
- **`swarm-torch`**: Unified `SwarmRoundCoordinator`, write-once content-addressed checkpoint store (`blobs/sha256/<digest>`), Ed25519-signed `RunArtifactBundle` v1 with cross-process `.bundle.lock` and Draft 2020-12 JSON Schema validation (`TrustPolicy::Strict`), sequential `RunGraph` scheduler, and HTML/Markdown report rendering (`swarm_torch_report`).

## Quick Start

```toml
[dependencies]
swarm-torch = { version = "0.1", features = ["std", "tokio-runtime"] }
```

Run the included end-to-end examples:

```bash
cargo run -p swarm-torch --example hello_swarm
cargo run -p swarm-torch --example artifact_pipeline
cargo run -p swarm-torch --example byzantine_federated_round
```

## Canonical Documentation

- [Workspace README](https://github.com/swarmic/SwarmTorch/blob/main/README.md)
- [Technical White Paper (`v0.1`)](https://github.com/swarmic/SwarmTorch/blob/main/SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.md)
- [Architecture Decision Records (`ADRs.md`)](https://github.com/swarmic/SwarmTorch/blob/main/ADRs.md)
- [Changelog (`CHANGELOG.md`)](https://github.com/swarmic/SwarmTorch/blob/main/CHANGELOG.md)
- [Security Policy (`SECURITY.md`)](https://github.com/swarmic/SwarmTorch/blob/main/SECURITY.md)
- [API Documentation (`docs.rs`)](https://docs.rs/swarm-torch)

## License

Licensed under the [Mozilla Public License 2.0 (MPL-2.0)](https://github.com/swarmic/SwarmTorch/blob/main/LICENSE).
