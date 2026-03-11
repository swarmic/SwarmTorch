# Roadmap

For detailed milestones and current status, see the [**Roadmap section in README**](README.md#roadmap).

## Quick Overview

| Version | Focus | Status |
|---------|-------|--------|
| v0.1.0 | Swarm Robotics MVP | Current |
| v0.2.0 | Edge ML Toolkit | Planned |
| v0.3.0 | Production Hardening | Planned |
| v1.0.0 | Enterprise Ready | Future |

## Release Gates

All releases must pass the gates defined in [ADR-0014](ADRs.md#adr-0014-benchmark-dataset-and-release-gates).

## Wave 4 Convergence Gate

Wave 4 is the remediation convergence point for `E-01..E-05` and semver-batched API tightening.
Wave 4 is signed off; the v0.2 feature-track kickoff has landed in Wave 5:

1. `F1` execution hints (`ExecutionHint`) additive schema slice.
2. `F2a + F2 + F3 + F4` grouped feature batch (`TransformAuditV0`, `UpdateTransform`, aggregation pipeline, lightweight tracing).

See `docs/release_notes/v0.1.0-alpha.6x-wave5.md` for implementation details and gate evidence.

Wave 6 completes backlog convergence and canon lock:

1. Behavior-preserving `A6X-06` decomposition of artifact/report modules.
2. ADR/whitepaper/roadmap truth-sync for Wave 5 (`F1..F4`) implementation reality.
3. A6X canon stabilization for artifact/report contracts and release-gate evidence.

See `docs/release_notes/v0.1.0-alpha.6x-wave6.md` for decomposition scope and compatibility evidence.

Wave 7 extends reliability and execution scope:

1. Reliability hardening on artifact/report load and manifest/write-path consistency.
2. Sequential graph execution engine MVP (deterministic topo scheduling, cycle detection, policy gating, fail-closed node status recording).
3. Typed `NodeV1` optional fields shipped (`op_hash`, `resources`, `cache_policy`, `materialization_policy`).
4. Initial robustness harness coverage for Byzantine attack evaluation.

See `docs/release_notes/v0.1.0-alpha.7x.md` for full Wave 7 implementation and gate evidence.

## Conformance Lock (A6X-05)

This section is the repo-local contract lock for current implementation reality.

| Area | Current State | Evidence |
|------|---------------|----------|
| MSRV policy | Tiered by crate: `swarm-torch-core` security path validated on Rust 1.75; top-level/model crates require newer compiler floor | `Cargo.toml`, `swarm-torch/Cargo.toml`, `swarm-torch-models/Cargo.toml` |
| Top-level crate portability | `swarm-torch` is std-only (fail-fast without `std`) | `swarm-torch/src/lib.rs` |
| Core portability | `swarm-torch-core` builds for minimal `no_std` and `no_std + alloc` | `.github/workflows/rust.yml`, `swarm-torch-core/src/lib.rs` |
| Runtime abstraction | Tokio implemented and validated on Rust 1.75; Embassy remains placeholder/experimental and is outside the Rust 1.75 conformance gate | `swarm-torch-runtime/src/lib.rs` |
| Transport layer | Trait + mock transport implemented; concrete TCP/UDP/BLE/LoRa/WiFi backends planned | `swarm-torch-net/src/lib.rs`, `swarm-torch-net/src/mock.rs` |
| Replay/auth security | Ed25519 message auth + replay enforcement implemented and hardened | `docs/release_notes/M4-01_ed25519_signatures.md`, `docs/release_notes/M4-02_replay_protection_enforcement.md`, `docs/release_notes/M4-02.5_replay_protection_hardening.md` |

### Baseline Command Set (must remain green)

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo +1.75 build -p swarm-torch-core --no-default-features
cargo +1.75 build -p swarm-torch-core --no-default-features --features alloc
cargo run -p swarm-torch --example artifact_pipeline
```

### Contract-Probe Commands (expected behavior checks)

```bash
# Expected to fail with explicit std-only boundary message:
cargo build -p swarm-torch --no-default-features

# Expected to fail on 1.75 due tiered MSRV policy for top-level/model crates:
cargo +1.75 build -p swarm-torch

# Expected to fail on 1.75 for the experimental embassy feature path:
cargo +1.75 build -p swarm-torch-runtime --no-default-features --features embassy
```
