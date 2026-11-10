# Roadmap

For a high-level project overview and usage guide, see [`README.md`](README.md). For contribution scope rules and formal collaboration (`swarmiclabs@gmail.com`), see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quick Overview

| Version | Focus | Status |
|---------|-------|--------|
| `v0.1.0-alpha.9` | Core Correctness, Security Hardening, & Dependency Isolation (Wave 9) | Completed |
| `v0.1.0-beta.1` | Coordinator Learning Spine, Physical CAS Checkpoints, & Strict Artifact Trust (Wave 10) | Completed |
| `v0.1.0` RC | Non-IID Resilience, Verifiable Sparse Co-Design, Low-MTU Framing, & SafeTensors Profile (Wave 11) | Current (Implemented & Verified) |
| `v0.2.0` | Isolated Framework/Transport Adapters, `FedAsync` + Durable RDP, & `wasmi` Sandbox (Wave 12) | Planned |
| `v0.3.0+` | Production Hardening, Local Artifact Reader (`P2-16`), & Optional Desktop Wrapper (`P2-17`) | Future |

## Release Gates

All releases must pass the conformance and benchmark gates defined in [ADR-0014](ADRs.md#adr-0014-benchmark-dataset-and-release-gates). Every architectural slice progresses through four explicit states: `Contract Ratified` -> `Implementation Landed` -> `Conformance Passed` -> `Profile Released`.

---

## Completed Milestones (Waves 4–11)

### Waves 4–6: Remediation & Artifact/Report Canon Lock
- Completed `E-01..E-05` semver batch, `F1` execution hints (`ExecutionHint`), `F2a/F2/F3/F4` (`TransformAuditV0`, `UpdateTransform`, composable aggregation pipeline, lightweight tracing), and behavior-preserving `A6X-06` decomposition of artifact/report modules.
- Evidence: `docs/release_notes/v0.1.0-alpha.6x-wave4.md`, `docs/release_notes/v0.1.0-alpha.6x-wave5.md`, `docs/release_notes/v0.1.0-alpha.6x-wave6.md`.

### Waves 7–8.1: Sequential Scheduler, Path Confinement, & Supply-Chain Baseline
- Shipped sequential graph execution engine MVP, typed `NodeV1` policy/resource fields, initial Byzantine robustness harness, Rust 1.75 `no_std + alloc` conformance in `run_graph`, artifact manifest path confinement (`validate_manifest_*`), `swarmtorch.*` telemetry namespace sync, and blocking `cargo deny` / `cargo audit` CI lanes.
- Evidence: `docs/release_notes/v0.1.0-alpha.7x.md`, `docs/release_notes/v0.1.0-alpha.8x-wave0.md`, `docs/release_notes/v0.1.0-alpha.8x-wave1.md`.

### Wave 9 (`v0.1.0-alpha.9` — Landed & Conformance Verified)
1. **Byzantine Aggregator Hardening, One-Pass `Multi-Krum`, & `BulyanProfile::IterativeKrumV1` (`W9-01`, `ADR-0019`):**
   - Enforces pre-aggregation finiteness/shape validation (`validate_gradient_batch`, `MAX_COHORT_SIZE = 1024`), pre-operation `f64` widening across `FedAvg`, `CoordinateMedian`, `TrimmedMean`, `Krum`, and `Bulyan`, `total_cmp` total ordering, one-pass `Multi-Krum` ($1 \le m \le n - f - 2$), and `Bulyan` with `BulyanProfile::IterativeKrumV1` ($f \ge 1, n \ge 4f + 3, \theta = n - 2f - 2, \beta = n - 4f - 2$) and `bulyan_stage2_nearest_to_median`.
2. **Expiry-Indexed Safe `BTreeMap` Replay Cache, `CohortAllowlist`, & Envelope v1 (`W9-02`, `ADR-0020` / `ADR-0021`):**
   - Replaces `lru` with a 100% safe dual-tree `BTreeMap` replay store (`entries_by_peer` + `expiry_index`), adds `CohortAllowlist` admission, `RecoveryRequired` startup/restart state with `SessionHandshakeChallenge`, `KeyPair::sign_with_domain` / `verify_with_domain`, and `MessageEnvelope` `ProtocolVersion::V1_0` (`1.0`) with the 6-stage `decode_and_verify_payload` pipeline (`MAX_FRAME_BYTES = 65_536`).
3. **Source-Accurate Compression Validation, `ClassicResidualCompressor`, & `EF21` (`W9-03`, `ADR-0020` / `ADR-0021`):**
   - Validates compression parameters and coordinate finiteness before allocation, ranks Top-$k$ via `select_nth_unstable_by` with deterministic index tie-breaking, implements seeded xorshift64* Reservoir Sampling in `RandomK`, and adds `ClassicResidualCompressor` and `Ef21Compressor` + `Ef21ServerAccumulator`.
4. **Trait Seams, Reference Models/Losses, Runtime Split, & Dependency Diet (`W9-04`, `W9-05`, `W9-06`, `W9-API-01`, `ADR-0002`, `ADR-0003`, `ADR-0022`, `ADR-0024`):**
   - Removes unused `burn = "0.14"`, `burn-ndarray = "0.14"`, `bincode 2.0.0-rc.3`, `async-trait`, and `lru` from the workspace (`RUSTSEC-2025-0141` and `EX-2026-04-05-02`/`03` closed).
   - Relaxes `SwarmModel::Input: ?Sized`, upgrades `LinearModelWithCapacity` and contiguous-buffer `MlpModelWithCapacity`, adds analytical `MseLoss`, `CrossEntropyLoss` (with zero-dependency `no_std` IEEE-754 `exp_f64`/`ln_f64`), `HuberLoss`, `FedProxRegularizer`, and `SgdOptimizer`, splits `PacketTransport` and GAT-based `SwarmTransport`, and splits `SwarmClock` / `SwarmRuntime` with 32-bit portable `MockRuntime`.

### Wave 10 (`v0.1.0-beta.1` — Landed & Conformance Verified)
1. **Strict Artifact Trust, Cross-Process Bundle Locking, & JSON Schema 2020-12 Contracts (`W10-02`, `ADR-0016`, `ADR-0023` §3, `P2-02`, `P2-05`, `P2-10`, `P2-20a`):**
   - Adds cross-process exclusive `.bundle.lock` guard (`BundleLockGuard`), unique `.staging.*` atomic writes with checked `sync_all()?` and POSIX parent-directory fsync, `ManifestSignatureV1` over raw 16-byte `RunId` preimages, single-pass zero-TOCTOU `VerifiedBundleV1` and `load_report_strict` (`TrustPolicy::Strict`), 9 canonical JSON Schema Draft 2020-12 files in `schemas/v1/*.schema.json` with lossless `u64` conformance tests, and promotes `cargo vet` to blocking in CI.
2. **Physical CAS Checkpoints, Explicit `DataOpsSession` Lifecycle, & `SwarmRoundCoordinator` (`W10-01`, `ADR-0023` §1–2, `P2-07`..`P2-09`, `P2-12`):**
   - Implements write-once CAS model checkpoints (`blobs/sha256/<digest>`) separating `CheckpointRecipeKeyV1`, `CheckpointContentDigestV1`, and `RoundResumeStateV1`, adds atomic generation/seal fencing on `RunArtifactSink`, explicit `DataOpsSession::{create, open, resume, commit_and_close}`, physical CAS cache-hit verification and runner output `Vec<AssetInstanceV1>` preservation in `scheduler.rs`, and unified `SwarmRoundCoordinator` (`ModelDeltaV1` / `GradientV1`, `WeightingPolicyV1::EqualPeerVote` / `RegistryCertified`, future-round rejection before unsigned subtraction, 1-per-peer idempotent duplicate ack vs. equivocation revocation, frozen input buffers, and authoritative CAS commit).
3. **Deterministic `MockRuntime` Scheduler, Originator-Authenticated Gossip Wrappers, & Multi-Node Simulation (`W10-03`, `ADR-0003`, `ADR-0014`, `ADR-0022` §3):**
   - Adds bounded FIFO task scheduling and deadline-ordered timer wakers (`std::task::Wake`) to `MockRuntime`, `InnerUpdateV1` + `RecipientWrapperV1` + `BundledGossipDeliveryV1` and topology-aware `MockNetworkHub` in `swarm-torch-net`, and the 6-scenario multi-node honest & Byzantine training simulation suite (`swarm-torch/tests/federated_round_simulation.rs`).

### Wave 11 (`v0.1.0` RC — Landed & Conformance Verified)
1. **Stateful `CenteredClipping` (`CClip`), Conservative `$s$-Bucketing`, & Exact `no_std` `sqrt_f64` (`W11-01`, `ADR-0019` §4):**
   - Replaces the legacy fixed-3-iteration Newton-Raphson `sqrt_f32` in `GradientValidator` with a shared IEEE-754 exponent-halving + 4-iteration Newton-Raphson `sqrt_f64_soft` (`swarm-torch-core::num`), implements `CenteredClipping` (`CClip`, Karimireddy et al., ICML 2021) with stateful across-round center tracking in `SwarmRoundCoordinator`, implements conservative `$s$-Bucketing` (`BucketingConfig`, Karimireddy et al., ICLR 2022) with canonical `PeerId`-ordered shuffling, uniform $1/s$ intra-bucket weights, and $f_B = \min(f, B)$ post-bucketing quorum re-validation, and establishes Wave 12 extension hooks (`StalenessWeightingPolicy`, `PrivacyAccountantHook`).
2. **Verifiable `SharedSeedSparse` Co-Design & Compressed-Subspace Aggregation (`W11-02`, `ADR-0021` §3):**
   - Implements Robert Floyd's $\mathcal{O}(k)$-space unbiased subset sampler (`sample_shared_indices_v1`), `SharedSeedSparsePayloadV1` (`UnscaledProjectionV1` & `ScaledUnbiasedV1`), pre-seed binary SHA-256 Merkle vector commitments (`PreSeedVectorCommitmentV1`) + coordinate opening proofs (`MerkleOpeningBatchV1`), `derive_verifiable_round_seed_v1`, `SharedSeedResidualCompressor`, and $\mathcal{O}(n^2 k)$ / $\mathcal{O}(Lnk)$ compressed-subspace robust aggregation (`aggregate_shared_seed_sparse_v1`, `aggregate_verifiable_shared_seed_sparse_v1`).
3. **Frozen 16-Byte `ChunkedFrameCodec`, Pre-Auth `ObservedSourceReassembler`, & Concrete `TcpTransport`/`UdpGossipTransport` (`W11-03`, `ADR-0022` §2):**
   - Implements the frozen 16-byte `ChunkedFrameCodec` (`[0x53, 0x54]` magic, `CRC-16/IBM-3740`, pinned 19-byte golden vector `53 54 01 03 04 03 02 01 00 00 01 00 03 00 7D A5 AA BB CC`, fixed stride $P = M - 16$), pre-authentication `ObservedSourceReassembler<E>` keyed by `(adapter_generation, observed_source, transfer_id)` with per-source/global budgets and conflict quarantine tombstones, cancellation-safe `TcpTransport` (`TcpFrameReaderState`), and `UdpGossipTransport`.
4. **`SwarmTorchCheckpointV1` SafeTensors Profile, CAS Integration, & Criterion Benchmarks (`W11-04`, `W10-03`, `ADR-0024` §3):**
   - Implements the `SwarmTorchCheckpointV1` SafeTensors v1 binary codec (`swarm-torch-models::safetensors_v1`) with strict duplicate-key-rejecting JSON header validation, contiguous offset verification, canonical `model_schema_hash`, transactional `SwarmModel` import/export, CAS helpers (`put_safetensors_checkpoint_v1`, `get_verified_safetensors_checkpoint_v1`), Wave 12 (`W12-03` `LoRaAdapterDescriptorV1`, `W12-04` `RunnerIsolationProfile`) stubs, Criterion benchmarks (`aggregation`, `swarm_bench`), and runnable examples (`hello_swarm`, `byzantine_federated_round`).

---

## Upcoming Ratified Waves (Wave 12)

1. **Wave 12 (`v0.2.0` — Isolated Framework/Transport Adapters, `FedAsync` + Durable RDP, & Installed `wasmi` Sandbox):**
   - `W12-01` (`ADR-0024`): Optional `swarm-torch-backends` (Burn/Candle adapters) and explicit LoRA / FLoRA delta aggregation.
   - `W12-02` (`ADR-0022`): Separate optional `swarm-torch-transport-zenoh` and `swarm-torch-transport-iroh` crates (1-RTT QUIC streams, distinct link `EndpointId` vs. originator `PeerId`).
   - `W12-03` (`ADR-0023`): Stale-delta `FedAsync` mixing and durable pre-release Rényi DP accounting.
   - `W12-04` (`ADR-0025`, `P2-20b`): `wasmi` sandbox host with installed `store.limiter(...)` + `trap_on_grow_failure(true)` + fuel, physical NDJSON/Arrow/Parquet execution in `NativeOpRunner`, and Parquet schema contracts (`P2-20b`).
   - `W12-05` (`ADR-0016`): Optional isolated `swarm-torch-mcp` read-only artifact inspection adapter over `stdio` backed by `TrustPolicy::Strict`.

---

## Conformance Lock (A6X-05 / Wave 10)

| Area | Current State | Evidence |
|------|---------------|----------|
| MSRV policy | Tiered by crate: `swarm-torch-core`, `swarm-torch-models`, and base `swarm-torch-net` preserve `no_std` portability; top-level/dev-dependency graphs use newer compiler floor | `Cargo.toml`, `swarm-torch-core/Cargo.toml`, `swarm-torch-models/Cargo.toml`, `swarm-torch-net/Cargo.toml` |
| Top-level crate portability | `swarm-torch` is std-only (fail-fast without `std`) | `swarm-torch/src/lib.rs` |
| Core & model portability | `swarm-torch-core`, `swarm-torch-models`, `swarm-torch-net`, and `swarm-torch-runtime` pass unit tests under `--no-default-features` | `.github/workflows/rust.yml`, crate `src/lib.rs` |
| Runtime abstraction | `SwarmClock` and `SwarmRuntime` split; `TokioRuntime` and 32-bit portable `MockRuntime` with deterministic task/timer scheduler implemented; `EmbassyRuntime` gated on `embassy` + `embassy-time-driver` | `swarm-torch-runtime/src/lib.rs` |
| Transport layer | Allocation-free `PacketTransport`, GAT-based `SwarmTransport`, originator-signed `BundledGossipDeliveryV1`, and `MockNetworkHub` implemented; concrete TCP/UDP/BLE/LoRa backends planned | `swarm-torch-net/src/traits.rs`, `swarm-torch-net/src/gossip.rs`, `swarm-torch-net/src/mock.rs` |
| Replay/auth security | `MessageEnvelope` v1.0 domain-separated Ed25519 signatures, 6-stage `decode_and_verify_payload`, safe `BTreeMap` replay store, `CohortAllowlist`, and `RecoveryRequired` challenge handshake implemented | `swarm-torch-core/src/replay.rs`, `swarm-torch-net/src/protocol.rs`, `swarm-torch-net/tests/replay_integration.rs` |
| Coordinator, CAS, & strict artifacts | `SwarmRoundCoordinator`, physical CAS checkpoints (`blobs/sha256/<digest>`), `.bundle.lock`, `ManifestSignatureV1`, `TrustPolicy::Strict`, and 9 Draft 2020-12 JSON Schemas implemented | `swarm-torch/src/coordinator.rs`, `swarm-torch/src/artifacts/`, `schemas/v1/`, `swarm-torch/tests/federated_round_simulation.rs` |

### Baseline Command Set (must remain green)

```bash
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --no-default-features -p swarm-torch-core --lib --locked
cargo test --no-default-features -p swarm-torch-models --lib --locked
cargo test --no-default-features -p swarm-torch-net --lib --locked
cargo test --no-default-features -p swarm-torch-runtime --lib --locked
cargo run -p swarm-torch --example artifact_pipeline --locked
```
