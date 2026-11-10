# Changelog

All notable changes to SwarmTorch are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-10 (Release Candidate — Wave 11)

### Added
- **Non-IID Byzantine Resilience (`W11-01`, `ADR-0019`):**
  - Stateful `CenteredClippingAggregator` (`CClip`) with explicit reference-iterate contract (`step_from_reference`), $\tau$-scaled empty-round stability, and `f64` norm/update accumulation.
  - Deterministic `SBucketingTransformer` using `ChaCha8Rng` Fisher-Yates shuffling (`domain: "swarmtorch.sbucket.v1"`) with `f64` bucket averaging.
- **Verifiable `SharedSeedSparse` Co-Design (`W11-02`, `ADR-0021`):**
  - `SharedSeedSparseCompressor` (`ChaCha8Rng` partial Fisher-Yates index derivation from `(round_id, peer_id, session_nonce)` under `"swarmtorch.shared_seed_sparse.v1"`), eliminating per-index wire overhead (`4k` bytes vs `8k` bytes).
  - `SharedSeedCoordinateVerifier` (`UpdateTransform`) verifying canonical strictly-ascending PRNG coordinate indices prior to Byzantine distance evaluation.
- **Low-MTU Chunked Framing & Concrete Transports (`W11-03`, `ADR-0004`, `ADR-0022`):**
  - 16-byte `ChunkHeaderV1` (`CRC-16/IBM-3740` CCITT-FALSE) and `ChunkedFrameCodec` reassembly pool (`MAX_CONCURRENT_REASSEMBLIES = 32`, `MAX_CHUNKS_PER_MESSAGE = 256`, bitmap deduplication, conflict eviction, and lazy TTL sweep) for LoRa/BLE/UDP links.
  - Feature-gated `TcpPacketTransport` (`tcp-transport`) with 4-byte length prefixing and `MAX_FRAME_BYTES` pre-allocation gating.
  - Feature-gated `UdpPacketTransport` (`udp-transport`) multiplexing single-datagram and `ChunkedFrameCodec`-fragmented frames.
- **SafeTensors v1 Checkpoint Persistence (`W11-04`, `ADR-0023`, `ADR-0024`):**
  - Zero-dependency `SwarmTorchCheckpointV1` reader/writer in `swarm-torch-models::safetensors_v1` with `HEADER_SIZE_LIMIT = 65_536`, strict `F32` contiguous interval validation, 8-byte alignment padding, and `SwarmModel` round-trip helpers.
  - Coordinator checkpoint export/import hooks (`save_safetensors_checkpoint_v1`, `load_safetensors_checkpoint_v1`).
- **Examples & Benchmarks (`W10-03` / `W11`):**
  - `byzantine_federated_round` end-to-end example and `swarm_bench` Criterion benchmark suite.

## [0.1.0-beta.1] - 2026-11-06 (Wave 10)

### Added
- **Unified Swarm Round Coordinator & Physical CAS Checkpoints (`W10-01`, `ADR-0023`):**
  - `SwarmRoundCoordinator` supporting `ModelDeltaV1` and `GradientV1` modes, `WeightingPolicyV1` (`EqualPeerVote` and `RegistryCertified`), future-round rejection, equivocation revocation, and authoritative CAS commits (`blobs/sha256/<digest>`).
  - Explicit `DataOpsSession` lifecycle (`create`, `open`, `resume`, `commit_and_close`) with atomic generation/seal fencing on `RunArtifactSink`.
- **Strict Artifact Trust & Draft 2020-12 JSON Schemas (`W10-02`, `ADR-0016`, `ADR-0023`):**
  - Cross-process exclusive `.bundle.lock` guard (`BundleLockGuard`), unique `.staging.*` atomic writes, `ManifestSignatureV1` over raw 16-byte `RunId` preimages, single-pass zero-TOCTOU `VerifiedBundleV1` (`TrustPolicy::Strict`), and 9 canonical JSON Schema Draft 2020-12 contracts in `schemas/v1/*.schema.json`.
  - Deterministic `MockRuntime` task/timer scheduler, originator-authenticated gossip wrappers (`InnerUpdateV1`, `RecipientWrapperV1`, `BundledGossipDeliveryV1`), and 6-scenario multi-node Byzantine simulation suite (`federated_round_simulation.rs`).
  - Promoted `cargo vet` to blocking in CI alongside `cargo deny` and `cargo audit`.

## [0.1.0-alpha.9] - 2026-11-05 (Wave 9)

### Added
- **Byzantine Aggregation & Compression Hardening (`W9-01`, `W9-03`, `ADR-0019`, `ADR-0021`):**
  - Pre-aggregation finiteness/shape validation (`validate_gradient_batch`, `MAX_COHORT_SIZE = 1024`), `f64` accumulation across `FedAvg`, `CoordinateMedian`, `TrimmedMean`, one-pass `Multi-Krum`, and `Bulyan` (`BulyanProfile::IterativeKrumV1`).
  - `ClassicResidualCompressor` and `Ef21Compressor` + `Ef21ServerAccumulator` error-feedback codecs.
- **Safe `BTreeMap` Replay Protection & Envelope v1 (`W9-02`, `ADR-0020`, `ADR-0021`):**
  - Replaced `lru` with a 100% safe dual-tree `BTreeMap` replay store (`entries_by_peer` + `expiry_index`), added `CohortAllowlist`, `RecoveryRequired` startup challenge (`SessionHandshakeChallenge`), and 6-stage `MessageEnvelope::decode_and_verify_payload` (`ProtocolVersion::V1_0`, `MAX_FRAME_BYTES = 65_536`).
- **Models, Losses, & Dependency Diet (`W9-04`..`W9-06`, `ADR-0002`, `ADR-0022`, `ADR-0024`):**
  - Added `LinearModelWithCapacity`, contiguous `MlpModelWithCapacity`, analytical `MseLoss`, `CrossEntropyLoss` (`no_std` IEEE-754 `exp_f64`/`ln_f64`), `HuberLoss`, `FedProxRegularizer`, and `SgdOptimizer`.
  - Removed unused `burn`, `burn-ndarray`, `bincode`, `async-trait`, and `lru` dependencies from the core workspace.

## [0.1.0-alpha.1 .. 0.1.0-alpha.8] - 2026-03 .. 2026-04 (Waves 0–8)

### Added
- Initial 5-crate workspace scaffold (`swarm-torch-core`, `swarm-torch-models`, `swarm-torch-net`, `swarm-torch-runtime`, `swarm-torch`), `RunArtifactBundle` v1 observability and lineage pipeline, sequential `RunGraph` scheduler MVP, Ed25519 message signing baseline, Windows/Unix artifact path-confinement checks, and supply-chain CI baseline (`cargo deny`, `cargo audit`, `cargo vet`).
