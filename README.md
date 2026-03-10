<p align="center">
  <img src="SwarmTorch.png" alt="SwarmTorch logo" width="160" />
</p>

# SwarmTorch

[![Crates.io](https://img.shields.io/crates/v/swarm-torch.svg)](https://crates.io/crates/swarm-torch)
[![Documentation](https://docs.rs/swarm-torch/badge.svg)](https://docs.rs/swarm-torch)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](LICENSE)
[![CI](https://github.com/swarmic/SwarmTorch/actions/workflows/rust.yml/badge.svg)](https://github.com/swarmic/SwarmTorch/actions/workflows/rust.yml)

**Mission-grade swarm learning for heterogeneous fleets—Rust-native, with a `no_std` core and Byzantine-resilient security posture.**

SwarmTorch is a distributed machine learning framework designed for real-world edge deployments where devices are resource-constrained, connections are unreliable, and trust cannot be assumed. Built in Rust from the ground up, it brings swarm intelligence and federated learning primitives to environments where PyTorch and TensorFlow cannot operate: embedded microcontrollers, intermittent networks, and adversarial conditions.

## Canonical Docs

- [`SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.md`](SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.md) (system model + conformance)
- [`ADRs.md`](ADRs.md) (architecture decisions)
- [`CONTEXT_SOURCES.md`](CONTEXT_SOURCES.md) (document hierarchy; drift controls)
- [`SECURITY.md`](SECURITY.md) (security policy + supply chain gates)

## Implementation Reality (2026-02-13)

- `swarm-torch` (top-level crate) is currently **std-only**.
- `swarm-torch-core` provides the portable baseline and builds under:
  - `no_std + alloc` (supported)
  - minimal `no_std` build (compiles, limited utility)
- Replay/auth enforcement is implemented (`M4-02`, `M4-02.5`).
- Concrete TCP/UDP/BLE/LoRa/WiFi transport implementations are still planned (current net path is trait + mock transport).

-----

## Why SwarmTorch?

The Rust ML ecosystem has excellent tools for specific problems:

- **[Burn](https://github.com/tracel-ai/burn)** and **[Candle](https://github.com/huggingface/candle)** for training and inference
- **[tch-rs](https://github.com/LaurentMazare/tch-rs)** for LibTorch interop
- **[tract](https://github.com/sonos/tract)** for efficient inference

**SwarmTorch adds the missing pieces:** distributed swarm learning primitives that work across heterogeneous fleets—from ESP32 microcontrollers to cloud servers—with built-in Byzantine fault tolerance, supply chain integrity, useful data-wrangling automations and visualizations, and real-world network assumptions.

### What Makes SwarmTorch Different

|Feature                        |SwarmTorch               |PyTorch Distributed |TensorFlow Federated    |
|-------------------------------|-------------------------|--------------------|------------------------|
|**Embedded targets** (`no_std`)|⚠️ Partial (`no_std + alloc`); `embedded_min` planned|❌ Not supported     |❌ Not supported         |
|**Asynchronous participation** |✅ Core design            |⚠️ Limited           |⚠️ Experimental          |
|**Byzantine robustness**       |✅ Pluggable aggregators  |❌ No defense        |⚠️ Research only         |
|**Heterogeneous networks**     |⚠️ Trait + mock transport (concrete transports planned)|❌ Assumes datacenter|❌ Assumes reliable links|
|**Memory footprint**           |✅ <256KB for participants|❌ GBs required      |❌ GBs required          |
|**Zero Python runtime**        |✅ Pure Rust              |❌ Python required   |❌ Python required       |

-----

## Quick Start

### Prerequisites

```bash
# Install Rust (1.81+ for `swarm-torch`; core path supports lower MSRV)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# For embedded targets (optional)
cargo install probe-rs-tools
rustup target add thumbv7em-none-eabihf
```

### Hello Swarm (Tier 1: Desktop/Server)

Train a simple model across 3 local nodes with gossip topology:

The snippets below are **design-target pseudocode** for planned high-level APIs.
Current public APIs are more limited than these examples.

```rust,ignore
use swarm_torch::prelude::*;
use tokio;

#[tokio::main]
async fn main() -> Result<()> {
    // Define swarm cluster with 3 nodes
    let swarm = SwarmCluster::builder()
        .topology(Topology::gossip(fanout: 2))
        .consensus(RobustAggregation::trimmed_mean(trim_ratio: 0.2))
        .transport(TcpTransport::local_cluster(num_nodes: 3))
        .build()
        .await?;

    // Define model (using Burn backend)
    let model = SimpleNet::new(input_dim: 784, hidden_dim: 128, output_dim: 10);

    // Particle Swarm Optimizer for distributed training
    let optimizer = ParticleSwarm::new()
        .particles(50)
        .inertia(0.7)
        .fitness(|model, data| cross_entropy_loss(model, data));

    // Train across swarm
    let trained_model = swarm
        .train(model, optimizer, local_mnist_data())
        .max_rounds(100)
        .convergence_threshold(0.01)
        .on_round(|round, metrics| {
            println!("Round {}: loss={:.4}, accuracy={:.2}%", 
                round, metrics.loss, metrics.accuracy * 100.0);
        })
        .await?;

    Ok(())
}
```

### Hello Swarm (Tier 2: Edge Gateway)

Cross-compile for ARM Linux (e.g., Raspberry Pi):

```bash
# Add ARM target
rustup target add aarch64-unknown-linux-gnu

# Build with edge-optimized features
cargo build --release --target aarch64-unknown-linux-gnu --features edge

# Deploy to device
scp target/aarch64-unknown-linux-gnu/release/swarm-node pi@192.168.1.100:~/
```

```rust,ignore
use swarm_torch::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Edge gateway coordinating IoT devices
    let swarm = SwarmCluster::builder()
        .topology(Topology::hierarchical(layers: 2))
        .transport(MultiTransport::new()
            .add_backend(WiFiTransport::new("192.168.1.0/24"))
            .add_backend(BleTransport::new(scan_window: Duration::from_secs(5))))
        .role(NodeRole::Gateway)
        .build()
        .await?;

    // Lightweight gradient aggregation for IoT nodes
    swarm.serve_aggregator().await?;
    Ok(())
}
```

### Hello Swarm (Tier 3: Embedded/`no_std`)

Minimal example for ESP32 or STM32 microcontrollers:

```rust,ignore
#![no_std]
#![no_main]

use embassy_executor::Spawner;
use swarm_torch_core::*; // no_std-compatible core

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let participant = SwarmParticipant::new()
        .role(ParticipantRole::Contributor)
        .memory_limit(64.kb())
        .transport(LoRaTransport::new(frequency: 915_000_000));

    // Compute compressed gradient sketch on device
    loop {
        let local_update = participant.compute_update(sensor_data()).await;
        participant.send_update(local_update).await;
        embassy_time::Timer::after_secs(60).await;
    }
}
```

**Note:** Full training on microcontrollers requires `alloc` feature. The `no_std` core provides gradient computation and communication primitives only.

-----

## Supported Targets & Embedded Profiles

SwarmTorch defines three explicit **embedded profiles** as portability targets (see [ADR-0002](ADRs.md#adr-0002-crate-topology-feature-flag-policy-and-embedded-profiles)). Current status: `embedded_alloc` is experimental; `embedded_min` remains planned.

|Profile                        |Feature Flags            |Allocator |Types Available                      |Examples                 |
|-------------------------------|-------------------------|----------|-------------------------------------|-------------------------|
|**`edge_std`**                 |`std`                    |Standard  |Full Rust std library                |Raspberry Pi, Jetson Nano|
|**`embedded_alloc`**           |`no_std` + `alloc`       |Required  |`Vec<u8>`, `Box<dyn Trait>`, `Arc`   |ESP32, STM32H7 (≥256KB)  |
|**`embedded_min`**             |`no_std`, no `alloc`     |None      |`heapless::Vec<T,N>`, fixed arrays   |Cortex-M0+, nRF52 (≤64KB)|

**Target compatibility matrix:**

|Target Class                   |Examples                 |Status                 |Profile           |
|-------------------------------|-------------------------|-----------------------|------------------|
|**Server/Desktop**             |`x86_64-*`, `aarch64-*`  |✅ Full support         |`edge_std`        |
|**Edge Gateways**              |Raspberry Pi, Jetson Nano|✅ Full support         |`edge_std`        |
|**Embedded (std)**             |OpenWRT routers          |✅ Full support         |`edge_std`        |
|**Embedded (no_std + alloc)**  |ESP32, STM32H7           |⚠️ Experimental         |`embedded_alloc`  |
|**Embedded (no_std, no alloc)**|Cortex-M0+               |🧭 Planned (no-alloc profile)|`embedded_min`    |

### Cross-Compilation Setup

SwarmTorch uses **[probe-rs](https://probe.rs/)** as the golden path for embedded development:

```bash
# Install probe-rs toolchain
cargo install probe-rs-tools cargo-embed

# Flash and debug embedded target
cd examples/embedded
cargo embed --chip STM32H743ZITx --release
```

See the [Supported Targets](#supported-targets--embedded-profiles) section above for detailed target-specific setup.

-----

## Feature Flags

SwarmTorch uses Cargo features to control compilation for different environments:

```toml
[dependencies]
swarm-torch = { version = "0.1", features = ["std", "tokio-runtime"] }
```

|Feature             |Description                              |Default     |
|--------------------|-----------------------------------------|------------|
|`std`               |Enable standard library support          |✅ Yes       |
|`alloc`             |Enable allocator for dynamic memory      |✅ (with std)|
|`tokio-runtime`     |Use Tokio runtime adapter                |✅ Yes       |
|`embassy-runtime`   |Enable Embassy placeholder adapter (experimental; not in Rust 1.75 conformance gate)|❌ No|
|`burn-backend`      |Use Burn backend integration scaffold    |✅ Yes       |
|`tch-backend`       |Use LibTorch (`tch-rs`) integration scaffold|❌ No    |
|`robust-aggregation`|Enable robust aggregation strategies     |✅ Yes       |
|`tcp-transport`     |Transport feature flag surface (backend implementation planned)|❌ No|
|`udp-transport`     |Transport feature flag surface (backend implementation planned)|❌ No|
|`ble-transport`     |Transport feature flag surface (backend implementation planned)|❌ No|
|`lora-transport`    |Transport feature flag surface (backend implementation planned)|❌ No|
|`telemetry`         |Core telemetry module re-exports         |❌ No        |
|`python`            |Reserved marker for future Python boundary work|❌ No   |

Roadmap-only items from ADRs (for example WGPU/CUDA backend wiring and PyO3 bindings) are not yet exposed as active Cargo features in `swarm-torch`.

**Embedded profile presets:**

```toml
# For embedded_min (no_std, no alloc)
swarm-torch-core = { version = "0.1", default-features = false }

# For embedded_alloc (no_std + alloc)  
swarm-torch-core = { version = "0.1", default-features = false, features = ["alloc"] }

# For edge_std (full std)
swarm-torch = { version = "0.1", features = ["std", "tokio-runtime"] }
```

-----

## Architecture Overview

SwarmTorch is structured in layers to maximize portability and composability.  
The diagram mixes implemented components and roadmap targets.

```
┌─────────────────────────────────────────────────────────┐
│            Application Layer (Your Code)                 │
├─────────────────────────────────────────────────────────┤
│   Swarm Learning API (design target surface)            │
│   • ParticleSwarm, AntColony, Firefly optimizers        │
│   • Robust aggregators (Krum, Trimmed Mean, Median)     │
├─────────────────────────────────────────────────────────┤
│   Model Abstraction Layer                               │
│   • Burn wrapper (placeholder)                          │
│   • tch-rs bridge (LibTorch interop)                    │
│   • Custom model traits                                 │
├─────────────────────────────────────────────────────────┤
│   Identity & Coordination Layer                         │
│   • Gossip-based eventual consistency (NOT consensus)   │
│   • Ed25519 identity with pluggable attestation         │
│   • Byzantine-robust aggregation                        │
├─────────────────────────────────────────────────────────┤
│   Transport Abstraction (swarm-torch-net)               │
│   • Trait + mock transport (implemented)                │
│   • TCP/UDP/BLE/LoRa backends (planned)                 │
├─────────────────────────────────────────────────────────┤
│   Runtime Abstraction                                   │
│   • Tokio (implemented)  │  Embassy (placeholder)       │
└─────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Portability First:** Same API compiles to server and embedded targets
1. **Zero-Cost Abstractions:** Trait-based design with compile-time specialization
1. **Reality-First Networking:** Assumes intermittent connections, not datacenter reliability
1. **Explicit Trust Model:** Byzantine robustness is opt-in with clear guarantees
1. **Incremental Adoption:** Works alongside existing Rust ML tools (Burn, tch-rs, tract)

### GPU Acceleration Strategy (Roadmap)

GPU backend policy is defined in [ADR-0015](ADRs.md#adr-0015-gpu-acceleration-strategy-wgpu-first-cuda-optional), but current crate wiring remains CPU-only.

|Backend |Current State       |Planned Direction |
|--------|--------------------|------------------|
|**CPU** |✅ Implemented       |Baseline path     |
|**WGPU**|🧭 Planned           |Portable GPU path |
|**CUDA**|🧭 Planned           |Optional NVIDIA path |

```rust,ignore
// Design-target pseudocode (not current API surface)
let config = SwarmConfig::builder()
    .gpu_backend(GpuBackend::Wgpu)
    .build();
```

**Policy remains unchanged:** GPU acceleration must stay out of `swarm-torch-core` and behind explicit opt-in features once wired.

-----

## Byzantine Robustness & Robust Aggregation

SwarmTorch treats adversarial participants as a first-class concern. When devices can fail, lie, or be compromised, classical federated averaging breaks down.

### Supported Robust Aggregators

|Aggregator                |Attack Resistance|Computational Cost|Best For                     |
|--------------------------|-----------------|------------------|-----------------------------|
|**Coordinate-wise Median**|✅ High           |Low               |Low-dimensional models       |
|**Trimmed Mean**          |✅ Medium-High    |Low               |Balanced performance         |
|**Krum**                  |✅ High           |Medium            |Small-medium fleets          |

`Bulyan` and `Multi-Krum` are roadmap items, not implemented in the current crate surface.

**Important caveat:** Research shows that all robust aggregators can degrade under specific attack strategies, especially in high-dimensional settings. SwarmTorch provides:

- **Transparent telemetry:** See why updates are rejected
- **Attack harnesses:** Test your aggregator against known attacks
- **Topology-aware weighting:** Use network structure to improve robustness

```rust,ignore
use swarm_torch::RobustAggregation;

let strategy = RobustAggregation::TrimmedMean { trim_ratio: 0.2 };
let _ = strategy;
```

See [SECURITY.md](SECURITY.md) for detailed threat model and **[ADR-0007: Robust Aggregation Strategy](ADRs.md#adr-0007-robust-aggregation-strategy)** for design rationale.

-----

## Network Transports

SwarmTorch provides a unified `SwarmTransport` trait.  
Current implementation includes mock transport/network for integration tests; concrete TCP/UDP/BLE/LoRa/WiFi backends are roadmap work.

```rust,ignore
pub trait SwarmTransport {
    async fn send(&self, peer: PeerId, msg: &[u8]) -> Result<()>;
    async fn recv(&self) -> Result<(PeerId, Vec<u8>)>;
    async fn broadcast(&self, msg: &[u8]) -> Result<()>;
    fn reliability_class(&self) -> ReliabilityClass; // BestEffort | AtLeastOnce | Reliable
}
```

### Transport Classes (Roadmap Backends)

|Transport|Reliability|Bandwidth     |Range  |Use Case                      |
|---------|-----------|--------------|-------|------------------------------|
|**TCP**  |Reliable   |High (Gbps)   |LAN/WAN|Datacenter, edge gateways     |
|**UDP**  |Best-effort|High (Gbps)   |LAN/WAN|High-throughput, loss-tolerant|
|**WiFi** |Reliable   |Medium (Mbps) |100m   |IoT, robotics                 |
|**BLE**  |Best-effort|Low (Kbps)    |10m    |Wearables, sensors            |
|**LoRa** |Best-effort|Very low (bps)|10km+  |Remote sensors, agriculture   |

**Multi-transport design-target example:**

```rust,ignore
let transport = MultiTransport::new()
    .add(WiFiTransport::new("192.168.1.0/24"), priority: 1)
    .add(LoRaTransport::new(freq: 915_000_000), priority: 2)
    .fallback_policy(FallbackPolicy::PreferLowLatency);
```

Envelope serialization is provided by the protocol module (`postcard`).  
Concrete transport-specific framing/compression policies remain planned.  
See **[ADR-0004: Network Transport Layer](ADRs.md#adr-0004-network-transport-layer)**.

-----

## Examples & Demos

Roadmap example themes are listed below for planned scenario coverage.
Only a subset is currently checked into this repository.

### Phase 1 Themes: Robotics & ROS2 (planned)

- Multi-robot policy learning
- Topology comparison (ring vs mesh vs hierarchical)
- Byzantine robot injection scenario

### Phase 2 Themes: Edge & Embedded ML (planned)

- ESP32 contributor node
- Multi-tier quantization
- On-device continual learning

### Phase 3 Themes: Privacy & Compliance (planned)

- Audit trail demo
- Cross-org swarm training
- Differential privacy scenario

### Python Interoperability (planned)

> **⚠️ Model Zoo Constraint:** Python interop is limited to a **supported model zoo** with explicit portability contracts—NOT arbitrary PyTorch model conversion. See [ADR-0009](ADRs.md#adr-0009-python-interoperability-pyo3-and-model-portability-contract).

- Jupyter notebook workflow
- Model zoo reference

Currently available examples in-repo:

```bash
cargo run -p swarm-torch --example hello_swarm
cargo run -p swarm-torch --example artifact_pipeline
```

-----

## Roadmap

### v0.1.0 - Swarm Robotics MVP (Current)

- ⚠️ Core algorithm configuration/types (full execution engine is in progress)
- ❌ TCP/UDP concrete transports (planned; trait + mock currently implemented)
- ✅ Basic robust aggregators (Median, Trimmed Mean, Krum)
- ❌ Burn backend integration (placeholder wrapper currently)
- ⏳ ROS2 bridge (beta)
- ⏳ 5 reference robotics examples

### v0.2.0 - Edge ML Toolkit

- ✅ Wave 4 remediation sign-off gate (L-* + E-* semver batch) required before v0.2 feature rollout
- ⏳ Embassy runtime support for `no_std` targets
- ⏳ BLE and WiFi transport implementations
- ⏳ Compressed gradient protocols (TopK, randomized sparsification)
- ⏳ Memory-bounded participant roles (<256KB)
- ⏳ ESP32 and STM32 reference implementations
- ✅ F1: run-graph execution hints (`ExecutionHint`) for profile/device-aware planning
- ✅ F2a/F2/F3/F4 batch: transform audit plumbing, `UpdateTransform`, composable aggregation pipeline, lightweight tracing

### v0.3.0 - Production Hardening

- ⏳ LoRa transport with duty-cycle management
- ⏳ Advanced Byzantine defense (Bulyan, Multi-Krum)
- ⏳ Attack simulation harness
- ⏳ tch-rs backend for PyTorch model compatibility (model zoo only)
- ⏳ ONNX export for inference interchange (import for inference, NOT training)

### v1.0.0 - Enterprise Ready

- ⏳ Audit logging and reproducibility tooling
- ⏳ Compliance mode (HIPAA, GDPR)
- ⏳ Python bindings (PyO3) with zero-copy data exchange
- ⏳ Formal verification of core protocols (TLA+)
- ⏳ Commercial support and SLA

See [ROADMAP.md](ROADMAP.md) for detailed milestones.

-----

## Limitations & Non-Goals (v1.0)

To maintain focus and prevent scope creep, SwarmTorch explicitly **does not aim to**:

❌ **Import arbitrary PyTorch models** - We support a **model zoo** with explicit portability contracts, not arbitrary conversion ([ADR-0009](ADRs.md#adr-0009-python-interoperability-pyo3-and-model-portability-contract))  
❌ **Use ONNX for training interchange** - ONNX is **inference-only**; training happens in SwarmTorch-native models ([ADR-0010](ADRs.md#adr-0010-model-portability-and-onnx-integration-inference-interchange))  
❌ **Provide Byzantine consensus** - We use gossip-based eventual consistency, NOT PBFT/Tendermint-style consensus ([ADR-0006A](ADRs.md#adr-0006a-coordination-protocol-gossip-based-eventual-consistency))  
❌ **Solve Sybil resistance internally** - Identity/attestation is delegated to external layer (Swarmic Network) ([ADR-0008A](ADRs.md#adr-0008a-identity-and-sybil-resistance-boundary))  
❌ **Support all models on all targets** - Embedded `embedded_min` profile runs participant roles only  
❌ **Replace existing Rust ML frameworks** - SwarmTorch complements Burn/Candle/tch-rs

### Known Limitations (v0.1)

- **Embedded support is experimental** - `no_std` core works, but tooling rough edges remain
- **No automatic hyperparameter tuning** - You must configure swarm algorithms manually
- **Limited model zoo** - Reference models only; bring your own architectures
- **Byzantine defense degrades in high dimensions** - Known issue in literature; we provide telemetry, not miracles

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to help address these limitations.

-----

## Benchmarks (Pre-Release Status)

Benchmark rigor is a release gate for SwarmTorch. Cross-framework benchmark claims are temporarily withheld until a reproducible suite is published with pinned environments, scripts, and raw artifacts.

Current status:

- `cargo bench` covers internal crate-level performance checks.
- Cross-framework comparisons (for example PyTorch DDP, TensorFlow Federated) are in progress and not yet published as canonical results.
- Until the benchmark suite is published, treat performance numbers as target goals, not authoritative claims.

-----

## Contributing

SwarmTorch is open-source (MPL-2.0 license) and welcomes contributions. We especially need help with:

- 🦀 **Embedded drivers** - BLE, LoRa, ESP-NOW implementations
- 🤖 **Robotics integrations** - ROS2 bridge improvements, real robot testing
- 🔒 **Security research** - Novel attack/defense strategies for swarm learning
- 📚 **Documentation** - Tutorials, case studies, API improvements
- 🧪 **Testing** - Cross-platform CI, fuzzing, property-based tests

**Before contributing:**

1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
1. Check [open issues](https://github.com/swarmic/SwarmTorch/issues)
1. Review [ADRs](ADRs.md) to understand design decisions

**Code of Conduct:** We follow the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).

-----

## Governance & Safety

### API Stability

- **v0.x:** Breaking changes allowed, but deprecated features get 2-version warning
- **v1.0+:** Semantic versioning strictly followed

### Unsafe Code Policy

- Unsafe code is permitted only in:

1. FFI boundaries (PyO3, C interop)
1. Performance-critical paths with extensive testing
1. Platform-specific optimizations (SIMD, etc.)

- All `unsafe` blocks require safety documentation and fuzzing

### Security Disclosure

Report vulnerabilities to **security@swarmtorch.dev** (see [SECURITY.md](SECURITY.md) for full disclosure policy)

**Do not** open public issues for security bugs. We follow a 90-day disclosure policy.

-----

## Citation

If you use SwarmTorch in research, please cite:

```bibtex
@software{swarmtorch2025,
  title = {SwarmTorch: Mission-Grade Swarm Learning for Heterogeneous Fleets},
  author = {SwarmTorch Contributors},
  year = {2025},
  url = {https://github.com/swarmic/SwarmTorch},
  version = {0.1.0}
}
```

**Key academic foundations:**

- Byzantine-resilient aggregation: [Blanchard et al., 2017](https://arxiv.org/abs/1703.02757)
- Gossip learning: [Hegedűs et al., 2019](https://arxiv.org/abs/1901.08769)
- Decentralized federated learning: [Lalitha et al., 2019](https://arxiv.org/abs/1805.09063)

See [ADRs.md](ADRs.md) for full technical references.

-----

## License

SwarmTorch is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**. See [LICENSE](LICENSE) for details.

**Why MPL-2.0?** File-level copyleft balances openness with protection:

- Modifications to MPL-covered files must be shared
- Can be combined with proprietary code in a "Larger Work"
- Widely adopted in Rust ecosystem (e.g., Servo, many Mozilla projects)
- Compatible with Apache-2.0 and MIT for integration

-----

## Community & Support

Public community channels are being prepared and will be linked here once live.

- 🐞 **Issues:** [github.com/swarmic/SwarmTorch/issues](https://github.com/swarmic/SwarmTorch/issues)
- 🔐 **Security Reports:** [security@swarmtorch.dev](mailto:security@swarmtorch.dev)

**Commercial Support:** Enterprise licensing, training, and consulting available via [support@swarmtorch.dev](mailto:support@swarmtorch.dev)

-----

## Acknowledgments

SwarmTorch builds on the shoulders of giants:

- **[Burn](https://github.com/tracel-ai/burn)** - Rust-native deep learning framework
- **[Embassy](https://embassy.dev/)** - Async embedded framework
- **[probe-rs](https://probe.rs/)** - Embedded debugging and flashing
- **[Tokio](https://tokio.rs/)** - Async runtime for servers/edge
- **Byzantine FL research community** - Foundational security work

Special thanks to early adopters and the SwarmTorch contributors.

-----

**Built with 🦀 Rust · Designed for 🤖 Reality · Secured for 🔒 Production**
