# Architecture Decision Records (ADRs)

This file records the concise normative architecture contracts for public contributors. Full deliberation histories, threat-model trade-offs, and wave closeout logs are maintained in `swarm-torch-planning` (`ADRs-full-deliberations.md`).

## Index

### Foundation (Phase 0)
- [ADR-0000: ADR Process and Template](#adr-0000-adr-process-and-template)
- [ADR-0001: Autodiff and Tensor Backend Strategy](#adr-0001-autodiff-and-tensor-backend-strategy)
- [ADR-0002: Crate Topology, Feature Flag Policy, and Embedded Profiles](#adr-0002-crate-topology-feature-flag-policy-and-embedded-profiles)
- [ADR-0003: Runtime Abstraction (Embassy ↔ Tokio)](#adr-0003-runtime-abstraction-embassy--tokio)

### Networking & Distribution (Phase 1)
- [ADR-0004: Network Transport Layer](#adr-0004-network-transport-layer)
- [ADR-0004A: Topology Policy as First-Class Primitive](#adr-0004a-topology-policy-as-first-class-primitive)
- [ADR-0004B: Asynchrony and Staleness Model](#adr-0004b-asynchrony-and-staleness-model)
- [ADR-0005: State Serialization and Message Formats](#adr-0005-state-serialization-and-message-formats)
- [ADR-0006: Identity and Membership Model](#adr-0006-identity-and-membership-model)
- [ADR-0006A: Coordination Protocol (Gossip-Based Eventual Consistency)](#adr-0006a-coordination-protocol-gossip-based-eventual-consistency)

### Security & Robustness (Phase 2)
- [ADR-0007: Robust Aggregation Strategy](#adr-0007-robust-aggregation-strategy)
- [ADR-0007A: Aggregation Validation and Attack Harness](#adr-0007a-aggregation-validation-and-attack-harness)
- [ADR-0008: Threat Model and Trust Boundaries](#adr-0008-threat-model-and-trust-boundaries)
- [ADR-0008A: Identity and Sybil Resistance Boundary](#adr-0008a-identity-and-sybil-resistance-boundary)
- [ADR-0008B: Message Envelope Sender Identity Contract](#adr-0008b-message-envelope-sender-identity-contract)

### Interoperability (Phase 3)
- [ADR-0009: Python Interoperability (PyO3) and Model Portability Contract](#adr-0009-python-interoperability-pyo3-and-model-portability-contract)
- [ADR-0010: Model Portability and ONNX Integration (Inference Interchange)](#adr-0010-model-portability-and-onnx-integration-inference-interchange)

### Operations & Acceleration (Phases 4–5)
- [ADR-0011: CI/CD Build Matrix and Artifact Strategy](#adr-0011-cicd-build-matrix-and-artifact-strategy)
- [ADR-0012: Observability and Telemetry](#adr-0012-observability-and-telemetry)
- [ADR-0013: Reproducibility and Determinism](#adr-0013-reproducibility-and-determinism)
- [ADR-0014: Benchmark, Dataset, and Release Gates](#adr-0014-benchmark-dataset-and-release-gates)
- [ADR-0015: GPU Acceleration Strategy (WGPU-first, CUDA Optional)](#adr-0015-gpu-acceleration-strategy-wgpu-first-cuda-optional)
- [ADR-0016: Run Artifacts and Visualization Surface](#adr-0016-run-artifacts-and-visualization-surface)
- [ADR-0017: Data Pipeline DSL and Asset Model](#adr-0017-data-pipeline-dsl-and-asset-model)
- [ADR-0018: Extension and Execution Policy](#adr-0018-extension-and-execution-policy)

### Hardening, Convergence, & Adapter Architecture (Waves 9–12)
- [ADR-0019: Byzantine-Resilient Aggregation Hardening, Multi-Krum, Bulyan, Centered Clipping (CClip), and Non-IID s-Bucketing](#adr-0019-byzantine-resilient-aggregation-hardening-multi-krum-bulyan-centered-clipping-cclip-and-non-iid-s-bucketing)
- [ADR-0020: Bounded-State Replay Protection, Receiver-Bound Context Recovery, and Cohort Admission Policy](#adr-0020-bounded-state-replay-protection-receiver-bound-context-recovery-and-cohort-admission-policy)
- [ADR-0021: Source-Accurate Compression Validation, Stateful Error Feedback (`ClassicResidual` & `EF21`), and Verifiable Shared-Seed Sparse Co-Design](#adr-0021-source-accurate-compression-validation-stateful-error-feedback-classicresidual--ef21-and-verifiable-shared-seed-sparse-co-design)
- [ADR-0022: Additive Transport Traits, Observed-Source Pre-Auth Reassembly, Originator-Authenticated Gossip Wrappers, and Zenoh/Iroh Adapter Crates](#adr-0022-additive-transport-traits-observed-source-pre-auth-reassembly-originator-authenticated-gossip-wrappers-and-zenohiroh-adapter-crates)
- [ADR-0023: Unified Swarm Round Coordinator, Physical CAS Checkpoints, Strict Manifest Trust Policy, Cross-Process Bundle Locking, `FedAsync`, and Durable RDP Accounting](#adr-0023-unified-swarm-round-coordinator-physical-cas-checkpoints-strict-manifest-trust-policy-cross-process-bundle-locking-fedasync-and-durable-rdp-accounting)
- [ADR-0024: Reference Models, Analytical Losses, SafeTensors Checkpoint Profile, and Isolated Framework Backend Adapters](#adr-0024-reference-models-analytical-losses-safetensors-checkpoint-profile-and-isolated-framework-backend-adapters)
- [ADR-0025: Deterministic WebAssembly Extension Sandbox (`wasmi`) and Columnar Data Runner](#adr-0025-deterministic-webassembly-extension-sandbox-wasmi-and-columnar-data-runner)

---

## ADR-0000: ADR Process and Template
**Status:** Accepted | **Date:** 2025-01-08  
Architectural decisions are numbered `ADR-NNNN`, progress through `Contract Ratified -> Implementation Landed -> Conformance Passed -> Profile Released`, and are superseded rather than silently mutated.

## ADR-0001: Autodiff and Tensor Backend Strategy
**Status:** Accepted | **Date:** 2025-01-08  
`swarm-torch-core` defines dependency-light tensor/model traits (`SwarmModel`, `SwarmOptimizer`), keeping `swarm-torch-models` portable to `no_std + alloc` while heavy DL backends (`Burn` / `Candle`) live strictly in isolated adapter crates (`adapters/swarm-torch-backends`).

## ADR-0002: Crate Topology, Feature Flag Policy, and Embedded Profiles
**Status:** Accepted (Amended 2026-11-02) | **Date:** 2025-01-08  
Workspace separates `swarm-torch-core` (`no_std`), `swarm-torch-models` (`no_std`), `swarm-torch-net` (`no_std`), `swarm-torch-runtime` (`no_std`), and `swarm-torch` (`std`-only facade). External framework/transport/MCP bridges are isolated under `adapters/*` outside the root `Cargo.lock` so core MSRV (`1.75` for core crates, `1.81` for `swarm-torch`) and supply-chain audits remain compact. No-op feature flags are prohibited prior to `0.1.0` publication.

## ADR-0003: Runtime Abstraction (Embassy ↔ Tokio)
**Status:** Conformance Passed (`W9-04`, `W10-03`) | **Date:** 2025-01-08  
Splits `SwarmClock` and `SwarmRuntime`; provides `TokioRuntime` (`tokio-runtime`), portable 32-bit `MockRuntime` with deterministic task queues and deadline-ordered `std::task::Wake` timers, and feature-gated `EmbassyRuntime`.

## ADR-0004: Network Transport Layer
**Status:** Accepted | **Date:** 2025-01-08  
Transport traits decouple framing/wire formats from concrete links (`TcpTransport`, `UdpGossipTransport`, chunked LoRa/BLE links, and isolated `iroh`/`zenoh` adapters) under strict pre-allocation `MAX_FRAME_BYTES` bounds (`65_536` wire default).

## ADR-0004A: Topology Policy as First-Class Primitive
**Status:** Accepted | **Date:** 2025-01-13  
Communication topologies (mesh, ring, tree, hierarchical) are explicit first-class contracts governing routing, neighbor validation, and convergence assumptions without ambient global broadcast assumptions.

## ADR-0004B: Asynchrony and Staleness Model
**Status:** Accepted (Extended by `ADR-0023` §4) | **Date:** 2025-01-13  
Asynchronous aggregation explicitly tracks round provenance, staleness distance $\tau = t - t_0$, bounded lag $\tau_{\max}$, and staleness decay policies (`Constant`, `Polynomial`, `Exponential`).

## ADR-0005: State Serialization and Message Formats
**Status:** Accepted | **Date:** 2025-01-08  
Internal wire framing uses canonical `postcard` encoding paired with versioned `ProtocolVersion::V1_0` (`MessageEnvelope`), rejecting trailing bytes and oversized frames before allocation.

## ADR-0006: Identity and Membership Model
**Status:** Accepted | **Date:** 2025-01-08  
Nodes authenticate via Ed25519 keys (`PeerId` derived from verifying key); membership admission is governed explicitly by `CohortAllowlist` rather than implicit open-peer discovery.

## ADR-0006A: Coordination Protocol (Gossip-Based Eventual Consistency)
**Status:** Accepted | **Date:** 2025-01-13  
Round state propagation uses gossip-based eventual consistency with originator-authenticated wrappers (`BundledGossipDeliveryV1`) rather than synchronous BFT consensus.

## ADR-0007: Robust Aggregation Strategy
**Status:** Accepted (Hardened by `ADR-0019`) | **Date:** 2025-01-08  
SwarmTorch treats Byzantine robustness as a default requirement when aggregating untrusted peer updates (`FedAvg`, `CoordinateMedian`, `TrimmedMean`, `Krum`/`Multi-Krum`, `Bulyan`, `CenteredClipping`).

## ADR-0007A: Aggregation Validation and Attack Harness
**Status:** Accepted (`W10-03` simulation suite) | **Date:** 2025-01-13  
Every aggregator must validate finite inputs, matching parameter shapes, and Byzantine quorum hypotheses, verified against deterministic multi-node label-flip, sign-flip, ALIE, and equivocation attack tests.

## ADR-0008: Threat Model and Trust Boundaries
**Status:** Accepted | **Date:** 2025-01-08  
Untrusted peers, untrusted datasets (`TrustClass::Untrusted`), and third-party extensions (`ExecutionTrust::SandboxedExtension` / `UnsafeExtension`) are isolated by explicit cryptographic, validation, and sandbox boundaries.

## ADR-0008A: Identity and Sybil Resistance Boundary
**Status:** Accepted (Amended 2026-11-02) | **Date:** 2025-01-13  
SwarmTorch authenticates `PeerId` signatures and cohort membership locally; Sybil admission control / stake / hardware attestation belongs to the external mission-plane boundary.

## ADR-0008B: Message Envelope Sender Identity Contract
**Status:** Accepted | **Date:** 2025-01-13  
`sender_id` in `MessageEnvelope` MUST bind cryptographically to the Ed25519 verifying key under domain-separated signatures (`KeyPair::sign_with_domain` / `verify_with_domain`).

## ADR-0009: Python Interoperability (PyO3) and Model Portability Contract
**Status:** Accepted | **Date:** 2025-01-08  
Python interop is scoped to explicit supported model schemas and SafeTensors (`SwarmTorchCheckpointV1`) / contract adapters, never arbitrary serialized pickle/PyTorch execution.

## ADR-0010: Model Portability and ONNX Integration (Inference Interchange)
**Status:** Accepted | **Date:** 2025-01-08  
ONNX serves inference interchange only; federated training state, optimizer accumulators, and checkpoint digests remain native to SwarmTorch.

## ADR-0011: CI/CD Build Matrix and Artifact Strategy
**Status:** Accepted | **Date:** 2025-01-08  
Blocking CI gates verify `rustfmt`, `clippy -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc`, `no_std`/`alloc`/`std` matrix tests, MSRV compilation, `cargo deny`, `cargo audit`, and `cargo vet`.

## ADR-0012: Observability and Telemetry
**Status:** Accepted | **Date:** 2025-01-08  
Spans, events, and metrics use the `swarmtorch.*` namespace (`SpanRecord`, `EventRecord`, `MetricRecord`) serialized deterministically to NDJSON artifacts without ambient secret leakage.

## ADR-0013: Reproducibility and Determinism
**Status:** Accepted | **Date:** 2025-01-08  
Node definitions (`node_def_hash_v1`, `op_hash_v0`), graph normalization (`GraphV1::normalize`), PRNG seeds (`ChaCha8Rng` with domain tags), and canonical JSON/binary schemas enforce cross-platform bit-deterministic provenance.

## ADR-0014: Benchmark, Dataset, and Release Gates
**Status:** Conformance Passed (`W10-03` / `W11`) | **Date:** 2025-01-08  
Releases require green CI supply-chain checks, schema conformance, Criterion microbenchmarks (`aggregation`, `swarm_bench`), and multi-node federated round simulations.

## ADR-0015: GPU Acceleration Strategy (WGPU-first, CUDA Optional)
**Status:** Proposed | **Date:** 2025-01-08  
Future hardware acceleration targets WGPU/Burn/Candle adapters outside the `no_std` core workspace.

## ADR-0016: Run Artifacts and Visualization Surface
**Status:** Implementation Landed & Conformance Verified (`W10-02`, `W12-04`, `W12-05`) | **Date:** 2026-01-19  
- Versioned Draft 2020-12 JSON Schemas in `schemas/v1/` (10 schemas, including `columnar_table.schema.json` closing `P2-20b`) mirrored byte-identically inside `swarm-torch/schemas/v1/`.
- Mandatory `TrustPolicy::Strict` single-pass bundle verification (`VerifiedBundleV1`, `load_report_strict`).
- Zero-port `stdio` JSON-RPC 2.0 read-only MCP inspector (`McpInspectorServerV1` / `swarm_torch_mcp` + `adapters/swarm-torch-mcp`) failing closed on `LegacyPermissive`, tampered manifests, path traversal, and write attempts.

## ADR-0017: Data Pipeline DSL and Asset Model
**Status:** Accepted | **Date:** 2026-01-19  
`GraphV1`, `DatasetRegistryV1`, `DatasetLineageV1`, and `MaterializationRecordV2` track asset fingerprints, physical row/byte measurements, `TrustClass`, and `TransformAuditV0` across CAS blobs.

## ADR-0018: Extension and Execution Policy
**Status:** Accepted (Implemented by `ADR-0025` in `W12-04`) | **Date:** 2026-01-19  
`Scheduler` enforces `ExecutionPolicy` and `ExecutionTrust` (`Core`, `SandboxedExtension`, `UnsafeExtension`) *before* invoking `NativeOpRunner`.

## ADR-0019: Byzantine-Resilient Aggregation Hardening, Multi-Krum, Bulyan, Centered Clipping (CClip), and Non-IID s-Bucketing
**Status:** Conformance Passed (`W9-01`, `W11-01`) | **Date:** 2026-11-01  
Enforces finite/shape pre-checks (`MAX_COHORT_SIZE = 1024`), `f64` widening, `total_cmp` ordering, one-pass `Multi-Krum`, `BulyanProfile::IterativeKrumV1`, stateful `CenteredClipping` (`CClip`) with explicit reference-iterate contracts, conservative `$s$-Bucketing` with uniform $1/s$ weights and $f_B = \min(f, B)$ quorum re-validation, and IEEE-754 `sqrt_f64_soft` in `swarm-torch-core::num`.

## ADR-0020: Bounded-State Replay Protection, Receiver-Bound Context Recovery, and Cohort Admission Policy
**Status:** Conformance Passed (`W9-02`) | **Date:** 2026-11-01  
Uses a 100% safe dual-tree `BTreeMap` replay store (`entries_by_peer` + `expiry_index`), explicit `CohortAllowlist` admission, `RecoveryRequired` restart handshake (`SessionHandshakeChallenge`), and 6-stage `decode_and_verify_payload` verification.

## ADR-0021: Source-Accurate Compression Validation, Stateful Error Feedback (`ClassicResidual` & `EF21`), and Verifiable Shared-Seed Sparse Co-Design
**Status:** Conformance Passed (`W9-03`, `W11-02`) | **Date:** 2026-11-01  
Provides Top-$k$, seeded reservoir `RandomK`, `ClassicResidualCompressor`, `Ef21Compressor`/`Ef21ServerAccumulator`, Robert Floyd $\mathcal{O}(k)$ `sample_shared_indices_v1`, binary SHA-256 Merkle vector commitments (`PreSeedVectorCommitmentV1` + `MerkleOpeningBatchV1`), `SharedSeedResidualCompressor`, and compressed-subspace Byzantine aggregation.

## ADR-0022: Additive Transport Traits, Observed-Source Pre-Auth Reassembly, Originator-Authenticated Gossip Wrappers, and Zenoh/Iroh Adapter Crates
**Status:** Core & Wrapper Contracts Landed; Adapters at `Simulation verified` (`W10-03`, `W11-03`, `W12-02`, `ST-08`) | **Date:** 2026-11-01 (Amended 2026-11-14)  
- Splits `PacketTransport`, GAT `SwarmTransport`, originator-signed `InnerUpdateV1` + `RecipientWrapperV1` + `BundledGossipDeliveryV1`, and the frozen 16-byte `ChunkedFrameCodec` with pre-auth `ObservedSourceReassembler<E>`, `TcpTransport`, and `UdpGossipTransport`.
- Wave 12 (`W12-02`) & Wave 13 Phase `13-0` (`ST-08`): `swarm-torch-net::carrier` defines `QuicStreamHandshakePolicyV1`, `QuicLengthPrefixedCodecV1`, and `ZenohTopicContractV1`; `adapters/swarm-torch-transport-iroh` and `adapters/swarm-torch-transport-zenoh` remain isolated contract adapters at `Simulation verified` (`iroh` enforces pre-replay `bind_originator` and bounded sliding-window `LinkFrameQuotaTrackerV1`, while live SDK wiring ships in Phases `13D-2`/`13D-3`). Cross-crate sealed link attestations and async socket I/O land in Phases `13A-1`/`13C-1`.

## ADR-0023: Unified Swarm Round Coordinator, Physical CAS Checkpoints, Strict Manifest Trust Policy, Cross-Process Bundle Locking, `FedAsync`, and Experimental RDP Accounting
**Status:** Core Spine Landed; DP Accounting Reclassified & Transactional Hardening Scheduled (`W10-01`, `W10-02`, `W12-03`, `ST-02`) | **Date:** 2026-11-01 (Amended 2026-11-14)  
- Establishes `.bundle.lock` (upgraded to `fs4` kernel locking in `13B-2`), atomic `.staging.*` writes with `sync_all()`, `ManifestSignatureV1` over raw 16-byte `RunId`, single-pass `VerifiedBundleV1`, dual-key CAS checkpoints, and `SwarmRoundCoordinator`.
- Wave 12 (`W12-03`) & Wave 13 Phase `13-0` (`ST-02`): Implements `apply_fedasync_stale_delta_v1` in `swarm-torch-core::aggregation` with `no_std` IEEE-754 `exp_f64_soft`/`ln_f64_soft` (`num.rs`), `historical_anchors`, and `RdpAccountantV1` (`swarm-torch-core::dp`) + `dp/ledger.ndjson` (`dp_ledger.rs`). Until certified `DpFedAvgV1` clipping + Gaussian noise + audited subsampling ship post-Wave-13, RDP accounting is strictly classified as **`ExperimentalHypotheticalBudgetOnly`** (`guarantees_differential_privacy = false`), and `VerifiedMechanismProtected` fails closed. Full staged `CandidateRoundStateV1` / `committed_head.json` atomic commits land in `13B-1`.

## ADR-0024: Reference Models, Analytical Losses, SafeTensors Checkpoint Profile, and Isolated Framework Backend Adapters
**Status:** Core Models Landed; Burn/Candle Adapters at `Simulation verified` (`W9-05`, `W11-04`, `W12-01`, `ST-08`) | **Date:** 2026-11-02 (Amended 2026-11-14)  
- Ships `LinearModelWithCapacity`, contiguous `MlpModelWithCapacity`, analytical losses, `SgdOptimizer`, `FedProxRegularizer`, and zero-dependency `SwarmTorchCheckpointV1` SafeTensors v1 codec.
- Wave 12 (`W12-01`): Adds `LoraAdapterPairV1`, `FloraStackedDeltaV1`, `LoraAggregationModeV1` (`FreezeASingleFactorV1`, `FloraStackingV1`, `BiasedFactorFedAvgV1`), `aggregate_lora_adapters_v1`, and `BackendTensorBridge`/`BackendModelHandleV1` in `swarm-torch-models`, coordinated atomically with staged base weights in `SwarmRoundCoordinator::aggregate_lora_cohort_round`, plus **`Simulation verified`** contract bridges in `adapters/swarm-torch-backends` (live `burn`/`candle` SDKs & canonical `[W | b]` offset mapping ship in Phase `13D-4`).

## ADR-0025: Fail-Closed Wasm Extension Containment (Transitioning to `wasmi`) and Columnar Data Runner
**Status:** Containment Fail-Closed Pending `wasmi` Integration (`W12-04`, `P2-20b`, `ST-09`, Phase `13D-1`) | **Date:** 2026-11-02 (Amended 2026-11-14)  
- Untrusted Wasm extension execution via `RunnerIsolationProfile::WasmSandboxedV1`, `WasmSandboxRunnerV1::{execute_byte_transform, execute_scalar_i32}`, and `NativeOpRunner` `"wasm_transform"` **fails closed** (`UnsupportedUntrustedWasmSandbox`) until an audited `wasmi` engine ships in `adapters/swarm-torch-wasm` (Phase `13D-1`).
- The legacy subset VM remains gated behind `#[doc(hidden)] run_experimental_trusted_test_fixture_only` with `F64` rejection, `4096` cumulative local bounds, and decode fuel metering for internal test fixtures only. `NativeOpRunner` validates physical `ColumnarTableV1` / NDJSON payloads against `schemas/v1/columnar_table.schema.json` (`P2-20b`) before writing write-once CAS blobs (`PhysicalStrict` defaults land in `13B-3`).
