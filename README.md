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
- [`SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md`](SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md) (public reference bibliography)
- [`ADRs.md`](ADRs.md) (architecture decisions)
- [`ROADMAP.md`](ROADMAP.md) (milestone history & Wave 13–16 milestones)
- [`CHANGELOG.md`](CHANGELOG.md) (release history & containment notes)
- [`SECURITY.md`](SECURITY.md) (security policy, boundaries & supply chain gates)

## Implementation Reality (Wave 12–16 `13-0` .. `16F` Complete)

- `swarm-torch` (top-level orchestration, round coordination, and artifact verification crate) is **std-only**.
- `swarm-torch-core`, `swarm-torch-models`, `swarm-torch-net`, and `swarm-torch-runtime` build under both `std` and `no_std + alloc` (plus minimal `no_std` for `swarm-torch-core`, `swarm-torch-models`, `swarm-torch-net` `PacketTransport` / `ChunkHeaderV1` / `lora-transport` / `ble-transport`, and `swarm-torch-runtime` `EmbassyRuntime`).
- **Certified `DpFedAvgV1` vs. Experimental DP Accounting (`ST-02`, Wave `16B`):** `RdpAccountantV1`, `dp/ledger.ndjson`, `SwarmRoundCoordinator`, and `McpInspectorServerV1` strictly segregate **`ExperimentalHypotheticalBudgetOnly`** (`guarantees_differential_privacy = false`) from **`VerifiedMechanismProtected`** (`guarantees_differential_privacy = true`, **`Adversarial conformance passed`**). Certified `DpFedAvgV1` implements exact integer-lattice $\ell_2$ clipping (`B = 65_536`), associative `i128` summation, floating-point-free Canonne, Kamath, & Steinke (2020, `arXiv:2004.00010`, Prop. 5 & Algorithms 1–3) Discrete Gaussian noise (`sample_dgauss_u128`) seeded via OS CSPRNG (`getrandom 0.2.17`), sealed `SampledDpCohortV1`/`DpNoiseEntropyV1`/`VerifiedDpMechanismRoundV1` proofs, one-shot anti-resampling Poisson cohort draws, zero-conditioning release on empty/below-quorum/dropout rounds, external bundle telemetry redaction under the trusted central coordinator model, and non-refundable pre-release ledger debits — scoped strictly to synchronous `FedAvg` + `EqualPeerVote` + `ClientLevel` under `AddRemoveOne` ($q \in (0, 1]$) or `ReplaceOne` at full participation $q = 1.0$ (`ReplaceOne` with $q < 1.0$, `FedAsync`, `LoRA`, `SharedSeedSparse`, `$s$-Bucketing`, and non-linear aggregators remain fail-closed).
- **Wasm Sandbox Containment (`ST-09`, `13D-1`, `16C`):** Default `NativeOpRunner::new()`, `WasmSandboxRunnerV1`, and `Scheduler` fail closed (`UnsupportedUntrustedWasmSandbox`) unless configured with a `WasmExtensionExecutor`; `adapters/swarm-torch-wasm` provides `WasmiExtensionExecutorV1` (`wasmi 2.0.0`, `EnforcedLimits::strict()`, `RawMem0V0` + structured `SwarmTorchAllocV1` guest ABI with pre-`(start)` validation, 16 MiB pre-copy output cap, strict 4-case ownership/aliasing rules, cumulative fuel + wall-clock timeout slicing, zero imports, SHA-256 pinning, and end-to-end physical CAS DAG execution), while the legacy bespoke interpreter is gated under `#[cfg(test)]`.
- **Authenticated Admission, Replay V2, Transactional Commits, Evented I/O, Collective Sparse, `no_std` Radio Transports, & Live Adapters (`13A`–`16E`):** `SwarmRoundCoordinator` enforces 3-stage Ed25519/replay/context admission failing closed on legacy `PeerRoundSubmissionV1`; `ReplayProtection` defaults to `RequireSessionHandshake` with receiver-nonce `session_id` v2 recovery + monotonic timestamp floor (`with_timestamp_floor`); `SwarmTransport` reports `ObservedSourceRef` with bounded `WaiterQueue` evented I/O; `swarm-torch-net` adds pure `no_std` `lora-transport` (`LoraPacketTransportV1`, Semtech AN1200.13 integer-$\mu\text{s}$ ToA, 61-slot `prepare -> tx -> commit` duty limiter) and `ble-transport` (`BlePacketTransportV1`, Bluetooth Core Spec 512B ATT attribute ceiling, pre-TX credit gating) at `Simulation verified`; `SwarmRuntime::try_spawn` is primary fallible alongside `EmbassyRuntime` & ARM/RISC-V CI checks; `RunArtifactBundle` uses persistent `fs4` locks, `SyncEachAppend` durability, and atomic `CommittedHeadV1` rollbacks (`PhysicalStrict` default); `Bulyan` enforces Q9 lower observed median + `PeerId` ties; two-phase collective `SharedSeedSparse` freezes `commitments_root` before beacon seed derivation; and standalone adapters run live `wasmi 2.0.0`, `rmcp 3.11`, `iroh 1.4.1`, `zenoh 1.10.2`, and `burn 0.21.0` (CPU + `real-burn-wgpu` GPU) / `candle 0.10.2` (CPU) integrations with `[W | b]` / `[W_1 | b_1 | W_2 | b_2]` SafeTensors layout mapping (`ModelFlatTensorLayoutV1`).

### Adapter & Capability Maturity Matrix

| Surface | Maturity Tier | Notes |
|---------|---------------|-------|
| Certified `DpFedAvgV1` (`swarm-torch-core::dp` + `coordinator` + `dp_ledger` + `mcp_inspector`) | `Adversarial conformance passed` | Exact integer-lattice $\ell_2$ clipping + Canonne et al. (2020) Discrete Gaussian (`getrandom 0.2.17`), sealed proofs, zero-conditioning Poisson release, telemetry redaction, `AddRemoveOne` ($q \in (0, 1]$) & `ReplaceOne` ($q = 1.0$) (`ST-02`, `16B-1`..`16B-5`) |
| `adapters/swarm-torch-wasm` | `Adversarial conformance passed` | Isolated `wasmi 2.0.0` executor (`WasmiExtensionExecutorV1`, MSRV `1.86`), legacy `RawMem0V0` + structured `SwarmTorchAllocV1` (`swarmtorch_alloc`/`dealloc`, pre-`(start)` checks, 16 MiB pre-copy cap, 4-case ownership/aliasing rules), cumulative fuel/timeout slicing, physical CAS E2E (`ST-09`, `15D-1`, `16C`) |
| `adapters/swarm-torch-transport-iroh` | `Integration verified` | Live `iroh 1.4.1` 1-RTT QUIC (`IrohEndpointBridgeV1`, MSRV `1.91`), bounded accept/stream timeouts, pre-body quota gating, CSPRNG nonces, ALPN + `remote_id` attestation (`ST-08`, `15D-2`) |
| `adapters/swarm-torch-transport-zenoh` | `Integration verified` | Live `zenoh 1.10.2` TCP peer pub/sub (`ZenohSessionBridgeV1`, MSRV `1.88`, `time` patched for `RUSTSEC-2026-0009`), isolated-peer rejection, bounded `ZBytes` reads, strict topic parity (`ST-08`, `15D-3`) |
| `adapters/swarm-torch-backends` | `Integration verified` | Live `burn 0.21.0` CPU (`real-burn`) + WGPU GPU (`real-burn-wgpu`: `BurnWgpuLinearBridgeV1`/`BurnWgpuMlpBridgeV1`, serialized backend-scoped `enumerate_wgpu_devices()`, `WgpuDeviceSelectorV1`, verified live on NVIDIA RTX 5080 + synthetic mixed-GPU topologies) & `candle 0.10.2` CPU (`real-candle`), MSRV `1.92`; `DevicePlacementV1::Cuda` remains `Contract defined` (`ST-08`, `15D-4`, `16D`) |
| `adapters/swarm-torch-mcp` | `Integration verified` | Zero-port read-only strict MCP inspector on the official `rmcp 3.11` SDK (`SwarmTorchMcpAdapterV1`, MSRV `1.88`, Decision `A3`) over `stdio_pipes`, bounded 1 MiB line cap, semaphore + cancellation-safe dispatch, truthful certified vs. experimental DP notice (`ADR-0016`, `15D-5`, `16B-4`) |
| `lora-transport` / `ble-transport` (`swarm-torch-net`) | `Simulation verified` | Pure `no_std` hardware-agnostic HAL packet transports (`LoraPacketTransportV1`, `BlePacketTransportV1`) with AN1200.13 integer-$\mu\text{s}$ ToA, 61-slot duty limiter, 512B BLE Core Spec ATT cap, `ChunkedFrameCodec`, and `thumbv6m`/`thumbv7em`/`riscv32imc` + Rust `1.75` builds (`16E-1`, `16E-2`; physical SPI/HCI board drivers deferred; `wifi-transport` placeholder only) |

> **6-Tier Scale:** `Contract defined` < `Simulation verified` < `SDK integrated` < `Integration verified` < `Adversarial conformance passed` < `Deployment supported`.

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
|**Heterogeneous networks**     |✅ Low-MTU chunking + TCP/UDP/Gossip|❌ Assumes datacenter|❌ Assumes reliable links|
|**Memory footprint**           |🎯 <256KB target for participants (`no_std + alloc` today; `embedded_min` planned)|❌ GBs required      |❌ GBs required          |
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

# Verify no_std + alloc core build for embedded targets
cargo build -p swarm-torch-core --no-default-features --features alloc
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
|`embassy-runtime`   |Enable Embassy runtime adapter (`no_std`)|❌ No        |
|`robust-aggregation`|Enable robust aggregation strategies (`krum`, `bulyan`)|✅ Yes|
|`tcp-transport`     |Enable Tokio TCP length-prefixed transport (`TcpTransport`)|❌ No|
|`udp-transport`     |Enable Tokio UDP chunk-multiplexed gossip/packet transport (`UdpGossipTransport` / `StdUdpPacketTransport`)|❌ No|
|`ble-transport`     |Enable pure `no_std` BLE GATT HAL packet transport (`BlePacketTransportV1`, 512B Core Spec ATT ceiling, `Simulation verified`)|❌ No|
|`lora-transport`    |Enable pure `no_std` LoRa HAL packet transport (`LoraPacketTransportV1`, AN1200.13 integer-$\mu\text{s}$ ToA & 61-slot duty limiter, `Simulation verified`)|❌ No|
|`telemetry`         |Core telemetry module re-exports         |❌ No        |

External framework bridges (such as Burn and Candle in `swarm-torch-backends`, WGPU/CUDA acceleration, and PyO3 bindings) are isolated to optional adapter crates and roadmap waves (`ADR-0015`, `ADR-0024`).

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
│   Swarm Learning & Coordination Layer                   │
│   • SwarmRoundCoordinator & CAS/SafeTensors Checkpoints │
│   • Robust aggregators (Median, Trimmed Mean,           │
│     Krum / Multi-Krum, Bulyan, CClip, s-Bucketing)      │
│   • Compression (ClassicResidual, EF21, SharedSeed)     │
├─────────────────────────────────────────────────────────┤
│   Model & Optimization Layer (swarm-torch-models)       │
│   • Reference LinearModel & contiguous MlpModel         │
│   • Analytical MSE, Cross-Entropy, Huber, FedProx       │
│   • SgdOptimizer & SwarmTorchCheckpointV1 SafeTensors   │
├─────────────────────────────────────────────────────────┤
│   Identity & Security Layer                             │
│   • Gossip-based eventual consistency (NOT consensus)   │
│   • Ed25519 EnvelopeV2 + receiver-bound SessionId       │
│   • Signed RunArtifactBundle v1 + TrustPolicy::Strict   │
├─────────────────────────────────────────────────────────┤
│   Transport Abstraction (swarm-torch-net)               │
│   • PacketTransport & GAT SwarmTransport traits         │
│   • ChunkedFrameCodec (16B CRC-16) + TCP/UDP/LoRa/BLE   │
├─────────────────────────────────────────────────────────┤
│   Runtime Abstraction (swarm-torch-runtime)             │
│   • Tokio & PollLoop (std)  │  Embassy & Spin (no_std)  │
└─────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Portability First:** Same API compiles to server and embedded targets
1. **Zero-Cost Abstractions:** Trait-based design with compile-time specialization
1. **Reality-First Networking:** Assumes intermittent connections, not datacenter reliability
1. **Explicit Trust Model:** Byzantine robustness is opt-in with clear guarantees
1. **Incremental Adoption:** Works alongside existing Rust ML tools (Burn, tch-rs, tract)

### GPU Acceleration Strategy (`ADR-0015`, `Wave 16D`)

GPU backend policy is defined in [ADR-0015](ADRs.md#adr-0015-gpu-acceleration-strategy-wgpu-first-cuda-optional). Portable WGPU execution is implemented in the isolated `adapters/swarm-torch-backends` crate behind `--features real-burn-wgpu` (`BurnWgpuLinearBridgeV1`, `BurnWgpuMlpBridgeV1`, `WgpuDeviceSelectorV1`, `enumerate_wgpu_devices()`).

|Backend |Current State       |Planned Direction |
|--------|--------------------|------------------|
|**CPU** |✅ Implemented (`real-burn`, `real-candle`, `swarm-torch-models`) |Baseline path     |
|**WGPU**|✅ `Integration verified` (`adapters/swarm-torch-backends --features real-burn-wgpu`, verified live on NVIDIA RTX 5080 + synthetic mixed-GPU topologies) |Portable GPU path |
|**CUDA**|🧭 `Contract defined` (`DevicePlacementV1::Cuda`) |Optional NVIDIA path |

```rust,ignore
// Design-target pseudocode (not current API surface)
let config = SwarmConfig::builder()
    .gpu_backend(GpuBackend::Wgpu)
    .build();
```

**Policy remains unchanged:** GPU acceleration stays out of `swarm-torch-core` and behind explicit opt-in adapter features (`adapters/swarm-torch-backends --features real-burn-wgpu`).

-----

## Byzantine Robustness & Robust Aggregation

SwarmTorch treats adversarial participants as a first-class concern. When devices can fail, lie, or be compromised, classical federated averaging breaks down.

### Supported Robust Aggregators

|Aggregator                |Attack Resistance|Computational Cost|Best For                     |
|--------------------------|-----------------|------------------|-----------------------------|
|**Coordinate-wise Median**|✅ High           |Low               |Low-dimensional models       |
|**Trimmed Mean**          |✅ Medium-High    |Low               |Balanced performance         |
|**Krum / Multi-Krum**     |✅ High           |Medium ($\mathcal{O}(n^2 d)$)|Small-medium fleets ($n \ge 2f + 3$)|
|**Bulyan (`IterativeKrumV1`)**|✅ Very High  |Medium-High ($\mathcal{O}(n^2 d)$)|High-resilience cohorts ($n \ge 4f + 3$)|
|**Centered Clipping (`CClip`)**|✅ High (Non-IID)|Low ($\mathcal{O}(L n d)$)|Stateful momentum/history clipping (`W11-01`)|
|**`BucketingConfig` / `aggregate_with_bucketing_v1`**|✅ Variance Reduction|Low ($\mathcal{O}(n d)$)|Pre-aggregation Non-IID $s$-bucketing (`W11-01`)|

Stateful `CenteredClipping` (`CClip`) and deterministic `Sha256CounterDrbgV1` Non-IID $s$-bucketing (`BucketingConfig` / `aggregate_with_bucketing_v1`) are implemented in `swarm-torch-core::aggregation` per [ADR-0019](ADRs.md#adr-0019-byzantine-resilient-aggregation-hardening-multi-krum-bulyan-centered-clipping-cclip-and-non-iid-s-bucketing) (`W11-01`).

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

SwarmTorch provides synchronous `PacketTransport` and async GAT-based `SwarmTransport` traits (`swarm-torch-net`), along with `ChunkedFrameCodec` (16-byte `CRC-16/IBM-3740` header with bounded out-of-order reassembly), `TcpTransport` (`tcp-transport`), `UdpGossipTransport` / `StdUdpPacketTransport` (`udp-transport`), pure `no_std` hardware-agnostic `LoraPacketTransportV1` (`lora-transport`) and `BlePacketTransportV1` (`ble-transport`, both `Simulation verified`), and `MockTransport`. Physical SPI/HCI radio board drivers and `wifi-transport` remain follow-on work.

```rust,ignore
pub trait SwarmTransport: Send + Sync {
    type SendFuture<'a>: Future<Output = Result<()>> + Send + 'a where Self: 'a;
    type RecvFuture<'a>: Future<Output = Result<ObservedTransportMessage>> + Send + 'a where Self: 'a;
    type BroadcastFuture<'a>: Future<Output = Result<BroadcastStats>> + Send + 'a where Self: 'a;
    type DiscoverFuture<'a>: Future<Output = Result<Vec<PeerId>>> + Send + 'a where Self: 'a;

    fn send<'a>(&'a self, peer: PeerId, msg: &'a [u8]) -> Self::SendFuture<'a>;
    fn recv(&self) -> Self::RecvFuture<'_>;
    fn broadcast<'a>(&'a self, msg: &'a [u8]) -> Self::BroadcastFuture<'a>;
    fn discover(&self) -> Self::DiscoverFuture<'_>;
    fn capabilities(&self) -> TransportCapabilities; // includes ReliabilityClass
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

Envelope serialization (`postcard` + Ed25519 `MessageEnvelope` v2, `VERSION_V2` `(0, 3)` default) and 16-byte `ChunkedFrameCodec` low-MTU fragmentation/reassembly (`W11-03`) are implemented in `swarm-torch-net`.  
See **[ADR-0004: Network Transport Layer](ADRs.md#adr-0004-network-transport-layer)** and **[ADR-0022: Additive Transport Traits, Observed-Source Pre-Auth Reassembly, Originator-Authenticated Gossip Wrappers, and Zenoh/Iroh Adapter Crates](ADRs.md#adr-0022-additive-transport-traits-observed-source-pre-auth-reassembly-originator-authenticated-gossip-wrappers-and-zenohiroh-adapter-crates)**.

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
cargo run -p swarm-torch --example byzantine_federated_round
```

-----

## Roadmap

### v0.1.0–v0.2.0 - Swarm Learning Core, Coordinator, Certified DP, Wasm Alloc ABI, WGPU, & `no_std` Radio Transports (Waves 6–16 Complete)

- ✅ Sequential execution engine MVP & artifact/report bundle pipeline (Waves 6–8)
- ✅ Wave 9 (`v0.1.0-alpha.9`): Byzantine aggregation hardening (`Multi-Krum`, `Bulyan IterativeKrumV1`), receiver-bound `SessionId` replay/auth (`VERSION_V1`), bounded compression & `ClassicResidual`/`EF21` error feedback, additive `PacketTransport`/`SwarmTransport` traits + gossip recipient wrappers, contiguous `LinearModel`/`MlpModel` + analytical losses (`MSE`, `CrossEntropy`, `Huber`) + `SgdOptimizer`, and `lru`/`burn` dependency isolation
- ✅ Wave 10 (`v0.1.0-beta.1`): Unified `SwarmRoundCoordinator`, physical CAS checkpoints (`blobs/sha256/<digest>`), `TrustPolicy::Strict`, cross-process bundle lock (`BundleLockGuard`), Draft 2020-12 JSON Schema conformance (`schemas/v1/*.schema.json`), and deterministic multi-node simulation
- ✅ Wave 11 (`v0.1.0` RC): Stateful `CenteredClipping` (`CClip`), Non-IID $s$-bucketing (`BucketingConfig` / `aggregate_with_bucketing_v1`), verifiable `SharedSeedSparse` projection (`SharedSeedSparsePayloadV1` / `SharedSeedResidualCompressor` + `SharedSeedCoordinateVerifier`), 16-byte chunked packet reassembly (`ChunkedFrameCodec` `CRC-16/IBM-3740`) + `TcpTransport` / `UdpGossipTransport` / `StdUdpPacketTransport`, and `SwarmTorchCheckpointV1` SafeTensors v1 profile
- ✅ Wave 12–15 (`v0.2.0` + `13-0`–`15E`): Isolated `adapters/*` SDKs (`wasmi 2.0.0`, `rmcp 3.11`, `iroh 1.4.1`, `zenoh 1.10.2`, CPU-only `burn 0.21.0` + `candle 0.10.2` — all `Integration verified` with committed MSRV-checked lockfiles), LoRA/FLoRA updates, `FedAsync` + experimental accounting ledger (`ST-02`), 3-stage Ed25519/v2-replay admission (`MessageEnvelope::VERSION_V2` `(0, 3)` default), atomic `CommittedHeadV1` storage + `fs4` locks, `PhysicalStrict` columnar runner, evented transport/runtime (`EmbassyRuntime`, `portable-atomic` on 32-bit ARM/RISC-V), Q9 `Bulyan` Stage 2, two-phase collective `SharedSeedSparse`, and `[W | b]` / `[W_1 | b_1 | W_2 | b_2]` SafeTensors layout mapping (`ModelFlatTensorLayoutV1`)
- ✅ Wave 16 (`16A`–`16F`): Exact 32-bit Lemire DRBG unification + local Windows CI-pinned supply-chain parity + Wave 16 exemption review dispositions; certified exact Discrete-Gaussian `DpFedAvgV1` (`ST-02`, integer-lattice clipping + Canonne et al. 2020 sampler + OS CSPRNG `getrandom 0.2.17` + sealed proofs + one-shot Poisson lock + telemetry redaction + fail-closed `ReplaceOne` with $q < 1.0$, `Adversarial conformance passed`); structured Wasm `SwarmTorchAllocV1` guest ABI (`ADR-0025` Addendum, `Adversarial conformance passed`); portable WGPU GPU backend (`adapters/swarm-torch-backends --features real-burn-wgpu`, `Integration verified` on NVIDIA RTX 5080); and pure `no_std` hardware-agnostic `lora-transport` & `ble-transport` HAL packet transports (`Simulation verified`, Rust `1.75`, `thumbv6m`/`thumbv7em`/`riscv32imc`)

### v0.3.0 - Hardware Board Drivers, CUDA, & Local Viewer Tooling

- ⏳ Out-of-tree physical RF SPI/HCI drivers (`SX126x`/`SX127x`, `trouble-host`/`nrf-softdevice`) implementing `LoraRadioHal` / `BleGattHal`, plus `wifi-transport`
- ⏳ Optional NVIDIA CUDA backend (`DevicePlacementV1::Cuda`) and memory-bounded ESP32 / STM32 reference deployments
- ⏳ Local-only read-only artifact HTTP server & desktop visualization surfaces (`ADR-0016`, `P2-16`, `P2-17`)
- ⏳ ONNX export for inference interchange (import for inference, NOT training; `ADR-0010`)

### v1.0.0 - Enterprise Ready

- ⏳ Audit logging and reproducibility tooling
- ⏳ Compliance mode (HIPAA, GDPR)
- ⏳ Python bindings (PyO3) with zero-copy data exchange (`ADR-0009`)
- ⏳ Formal verification of core protocols (TLA+)

See [ROADMAP.md](ROADMAP.md) for the canonical wave breakdown.

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

Report vulnerabilities privately via **[GitHub Security Advisories](https://github.com/swarmic/SwarmTorch/security/advisories/new)** or email **[swarmiclabs@gmail.com](mailto:swarmiclabs@gmail.com)** (see [SECURITY.md](SECURITY.md) for full disclosure policy).

**Do not** open public issues for security bugs. We follow a 90-day coordinated disclosure policy.

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

- Byzantine-resilient aggregation: [Blanchard et al., 2017](https://arxiv.org/abs/1703.02757), [El Mhamdi et al., 2018](https://arxiv.org/abs/1802.07927)
- Error-feedback compression: [Richtárik et al., 2021 (EF21)](https://arxiv.org/abs/2106.05203)
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

- 🐞 **Issues:** [github.com/swarmic/SwarmTorch/issues](https://github.com/swarmic/SwarmTorch/issues)
- 🔐 **Security Reports:** [GitHub Security Advisories](https://github.com/swarmic/SwarmTorch/security/advisories/new) or [swarmiclabs@gmail.com](mailto:swarmiclabs@gmail.com)
- ✉️ **Architecture & Roadmap Collaboration:** [swarmiclabs@gmail.com](mailto:swarmiclabs@gmail.com) (see [CONTRIBUTING.md](CONTRIBUTING.md))

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
