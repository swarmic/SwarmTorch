# Roadmap

For a high-level project overview and usage guide, see [`README.md`](README.md). For contribution scope rules and formal collaboration (`swarmiclabs@gmail.com`), see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Quick Overview

| Version | Focus | Status |
|---------|-------|--------|
| `v0.1.0-alpha.9` | Core Correctness, Security Hardening, & Dependency Isolation (Wave 9) | Current (Implemented & Verified) |
| `v0.1.0-beta.1` | Coordinator Learning Spine, Physical CAS Checkpoints, & Strict Artifact Trust (Wave 10) | Planned |
| `v0.1.0` RC | Non-IID Resilience, Verifiable Sparse Co-Design, Low-MTU Framing, & SafeTensors Profile (Wave 11) | Planned |
| `v0.2.0` | Isolated Framework/Transport Adapters, `FedAsync` + Durable RDP, & `wasmi` Sandbox (Wave 12) | Planned |
| `v0.3.0+` | Production Hardening, Local Artifact Reader (`P2-16`), & Optional Desktop Wrapper (`P2-17`) | Future |

## Release Gates

All releases must pass the conformance and benchmark gates defined in [ADR-0014](ADRs.md#adr-0014-benchmark-dataset-and-release-gates). Every architectural slice progresses through four explicit states: `Contract Ratified` -> `Implementation Landed` -> `Conformance Passed` -> `Profile Released`.

---

## Completed Milestones (Waves 4–9)

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

---

## Upcoming Ratified Waves (Waves 10–12)

1. **Wave 10 (`v0.1.0-beta.1` — Coordinator Learning Spine, Physical CAS Checkpoints, & Strict Artifact Trust):**
   - `W10-01` (`ADR-0023`): `SwarmRoundCoordinator` (`ModelDeltaV1` default, single-owner local `FedProx`, one contribution per `(run, round, verified_originator)`), content-addressed checkpoints (`blobs/sha256/<digest>`) with checked `sync_all`, validated cache hits in `scheduler.rs`, and explicit fallible `DataOpsSession` lifecycle.
   - `W10-02` (`ADR-0023`, `ADR-0016`): `ManifestSignatureV1` (raw 16-byte `RunId`), `TrustPolicy::Strict` verified-consumer default, OS-backed exclusive bundle lockfile (`P2-02`), versioned JSON Schema 2020-12 contracts (`P2-20a`), and promotion of `cargo vet` to blocking (`P2-05`).
   - `W10-03` (`ADR-0022`, `ADR-0003`, `ADR-0014`): Deterministic task/timer scheduler in `MockRuntime`, multi-peer `MockNetworkHub` with originator-signed recipient wrappers, and reproducible benchmark harness.
2. **Wave 11 (`v0.1.0` RC — Non-IID Resilience, Verifiable Sparse Co-Design, Low-MTU Framing, & SafeTensors Profile):**
   - `W11-01` (`ADR-0019`): `CenteredClipping` (`CClip`) and conservative `$s$-Bucketing` ($n \bmod s = 0, f_B = \min(f, B)$) with post-bucketing quorum re-checks.
   - `W11-02` (`ADR-0021`): `SharedSeedSparse` with $\mathcal{O}(k)$ sparse sampling and verifiable pre-seed Merkle/vector commitments.
   - `W11-03` (`ADR-0022`): Frozen 16-byte `ChunkedFrameCodec` (`[0x53, 0x54]`, `CRC-16/IBM-3740`, fixed stride $P$, observed-source pre-auth budgets) and concrete `TcpTransport`/`UdpGossipTransport`.
   - `W11-04` (`ADR-0024`): `SwarmTorchCheckpointV1` profile over additive `safetensors` (`0.7+`/`0.8+`) in an optional codec boundary.
3. **Wave 12 (`v0.2.0` — Isolated Framework/Transport Adapters, `FedAsync` + Durable RDP, & Installed `wasmi` Sandbox):**
   - `W12-01` (`ADR-0024`): Optional `swarm-torch-backends` (Burn/Candle adapters) and explicit LoRA / FLoRA delta aggregation.
   - `W12-02` (`ADR-0022`): Separate optional `swarm-torch-transport-zenoh` and `swarm-torch-transport-iroh` crates (1-RTT QUIC streams, distinct link `EndpointId` vs. originator `PeerId`).
   - `W12-03` (`ADR-0023`): Stale-delta `FedAsync` mixing and durable pre-release Rényi DP accounting.
   - `W12-04` (`ADR-0025`, `P2-20b`): `wasmi` sandbox host with installed `store.limiter(...)` + `trap_on_grow_failure(true)` + fuel, physical NDJSON/Arrow/Parquet execution in `NativeOpRunner`, and Parquet schema contracts (`P2-20b`).
   - `W12-05` (`ADR-0016`): Optional isolated `swarm-torch-mcp` read-only artifact inspection adapter over `stdio` backed by `TrustPolicy::Strict`.

---

## Conformance Lock (A6X-05 / Wave 9)

| Area | Current State | Evidence |
|------|---------------|----------|
| MSRV policy | Tiered by crate: `swarm-torch-core`, `swarm-torch-models`, and base `swarm-torch-net` preserve `no_std` portability; top-level/dev-dependency graphs use newer compiler floor | `Cargo.toml`, `swarm-torch-core/Cargo.toml`, `swarm-torch-models/Cargo.toml`, `swarm-torch-net/Cargo.toml` |
| Top-level crate portability | `swarm-torch` is std-only (fail-fast without `std`) | `swarm-torch/src/lib.rs` |
| Core & model portability | `swarm-torch-core`, `swarm-torch-models`, `swarm-torch-net`, and `swarm-torch-runtime` pass unit tests under `--no-default-features` | `.github/workflows/rust.yml`, crate `src/lib.rs` |
| Runtime abstraction | `SwarmClock` and `SwarmRuntime` split; `TokioRuntime` and 32-bit portable `MockRuntime` implemented; `EmbassyRuntime` gated on `embassy` + `embassy-time-driver` | `swarm-torch-runtime/src/lib.rs` |
| Transport layer | Allocation-free `PacketTransport` and GAT-based `SwarmTransport` + `MockTransport` implemented; concrete TCP/UDP/BLE/LoRa backends planned | `swarm-torch-net/src/traits.rs`, `swarm-torch-net/src/mock.rs` |
| Replay/auth security | `MessageEnvelope` v1.0 domain-separated Ed25519 signatures, 6-stage `decode_and_verify_payload`, safe `BTreeMap` replay store, `CohortAllowlist`, and `RecoveryRequired` challenge handshake implemented | `swarm-torch-core/src/replay.rs`, `swarm-torch-net/src/protocol.rs`, `swarm-torch-net/tests/replay_integration.rs` |

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
