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
- [ADR-0023: Unified Swarm Round Coordinator, Physical CAS Checkpoints, Strict Manifest Trust Policy, Cross-Process Bundle Locking, `FedAsync`, and Certified Discrete-Gaussian `DpFedAvgV1` (`ST-02`)](#adr-0023-unified-swarm-round-coordinator-physical-cas-checkpoints-strict-manifest-trust-policy-cross-process-bundle-locking-fedasync-and-certified-discrete-gaussian-dpfedavgv1-st-02)
- [ADR-0024: Reference Models, Analytical Losses, SafeTensors Checkpoint Profile, and Isolated Framework Backend Adapters](#adr-0024-reference-models-analytical-losses-safetensors-checkpoint-profile-and-isolated-framework-backend-adapters)
- [ADR-0025: Fail-Closed Wasm Extension Containment (`wasmi` Adapter, `RawMem0V0` & `SwarmTorchAllocV1` Guest ABIs) and `PhysicalStrict` Columnar Data Runner](#adr-0025-fail-closed-wasm-extension-containment-wasmi-adapter-rawmem0v0--swarmtorchallocv1-guest-abis-and-physicalstrict-columnar-data-runner)

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
**Status:** Conformance Passed (`W9-04`, `W10-03`, Wave 13 `13C-1`) | **Date:** 2025-01-08 (Amended 2026-11-14)  
Splits `SwarmClock` and `SwarmRuntime` (`try_spawn -> Result<Self::TaskHandle, SpawnError>` primary fallible admission with zero silent task drops); provides `TokioRuntime` (`tokio-runtime`), portable 32-bit `MockRuntime` with deterministic task queues and deadline-ordered `std::task::Wake` timers, and feature-gated `EmbassyRuntime` backed by real `embassy_time::{Instant, Timer}` plus static `#[embassy_executor::task]` admission via `EmbassyRuntime::try_spawn_task` (`embassy_executor::SendSpawner`) verified across `thumbv6m-none-eabi` and `thumbv7em-none-eabihf`.

## ADR-0004: Network Transport Layer
**Status:** Accepted | **Date:** 2025-01-08  
Transport traits decouple framing/wire formats from concrete links (`TcpTransport`, `UdpGossipTransport`, chunked LoRa/BLE links, and isolated `iroh`/`zenoh` adapters) under strict pre-allocation frame/envelope bounds (`DEFAULT_MAX_ENVELOPE_BYTES = 1_048_576` wire default, alongside per-transport link frame caps).

## ADR-0004A: Topology Policy as First-Class Primitive
**Status:** Accepted | **Date:** 2025-01-13  
Communication topologies (mesh, ring, tree, hierarchical) are explicit first-class contracts governing routing, neighbor validation, and convergence assumptions without ambient global broadcast assumptions.

## ADR-0004B: Asynchrony and Staleness Model
**Status:** Accepted (Extended by `ADR-0023` §4) | **Date:** 2025-01-13  
Asynchronous aggregation explicitly tracks round provenance, staleness distance $\tau = t - t_0$, bounded lag $\tau_{\max}$, and staleness decay policies (`Constant`, `Polynomial`, `Exponential`).

## ADR-0005: State Serialization and Message Formats
**Status:** Accepted (Amended Wave 15 `15C-3`) | **Date:** 2025-01-08  
Internal wire framing uses canonical `postcard` encoding paired with versioned `MessageEnvelope` (`VERSION_V0` `(0, 1)`, `VERSION_V1` `(0, 2)`, and default `CURRENT_VERSION = VERSION_V2` `(0, 3)` with gated legacy compatibility), rejecting trailing bytes and oversized frames before allocation.

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
`sender_id` in `MessageEnvelope` MUST bind cryptographically to the Ed25519 verifying key under domain-separated signatures (`MessageAuth::{sign_v1, sign_v2, verify, sign_prehashed_domain, verify_prehashed_domain}`).

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
Node definitions (`node_def_hash_v1`, `op_hash_v0`), graph normalization (`GraphV1::normalize`), domain-separated PRNG streams (`Sha256CounterDrbgV1`), and canonical JSON/binary schemas enforce cross-platform bit-deterministic provenance.

## ADR-0014: Benchmark, Dataset, and Release Gates
**Status:** Conformance Passed (`W10-03` / `W11`) | **Date:** 2025-01-08  
Releases require green CI supply-chain checks, schema conformance, Criterion microbenchmarks (`aggregation`, `swarm_bench`), and multi-node federated round simulations.

## ADR-0015: GPU Acceleration Strategy (WGPU-first, CUDA Optional)
**Status:** WGPU Backend Landed (`Wave 16D`, `Integration verified` on named host GPU); CUDA `Contract defined` | **Date:** 2025-01-08 (Amended 2026-11-21)  
Portable GPU acceleration stays isolated outside the `no_std` workspace inside `adapters/swarm-torch-backends` (`real-burn-wgpu` feature over `burn-wgpu 0.21.0` / `cubecl 0.10.0` / `wgpu 29.0.4`, MSRV `1.92`). `enumerate_wgpu_devices()` uses a mutex-serialized backend-scoped `wgpu::Instance` restricted to `AutoGraphicsApi::backend()` with explicit `WgpuDeviceSelectorV1` per-type ordinal selection, fail-closed `DeviceUnavailable` handling, and `BurnWgpuLinearBridgeV1` / `BurnWgpuMlpBridgeV1` SafeTensors forward & MSE-gradient parity (`DevicePlacementV1::Cuda` remains `Contract defined`).

## ADR-0016: Run Artifacts and Visualization Surface
**Status:** Implementation Landed & Conformance Verified (`W10-02`, `W12-04`, `W12-05`, Wave 15 `15D-5`); MCP Adapter at `Integration verified` | **Date:** 2026-01-19 (Amended 2026-11-19)  
- Versioned Draft 2020-12 JSON Schemas in `schemas/v1/` (10 schemas, including `columnar_table.schema.json` closing `P2-20b`) mirrored byte-identically inside `swarm-torch/schemas/v1/`.
- Mandatory `TrustPolicy::Strict` single-pass bundle verification (`VerifiedBundleV1`, `load_report_strict`).
- Zero-port `stdio` JSON-RPC 2.0 read-only MCP inspector (`McpInspectorServerV1` / `swarm_torch_mcp` + standalone `adapters/swarm-torch-mcp` integrating the official `rmcp 3.11` SDK over `stdio_pipes` at MSRV `1.88`, Wave 15 Decision `A3`) failing closed on `LegacyPermissive`, tampered manifests (`-32001`), path traversal/undeclared arguments/unsupported cursors (`-32602`), and write attempts.

## ADR-0017: Data Pipeline DSL and Asset Model
**Status:** Accepted | **Date:** 2026-01-19  
`GraphV1`, `DatasetRegistryV1`, `DatasetLineageV1`, and `MaterializationRecordV2` track asset fingerprints, physical row/byte measurements, `TrustClass`, and `TransformAuditV0` across CAS blobs.

## ADR-0018: Extension and Execution Policy
**Status:** Accepted (Implemented by `ADR-0025` in `W12-04`) | **Date:** 2026-01-19  
`Scheduler` enforces `ExecutionPolicy` and `ExecutionTrust` (`Core`, `SandboxedExtension`, `UnsafeExtension`) *before* invoking `NativeOpRunner`.

## ADR-0019: Byzantine-Resilient Aggregation Hardening, Multi-Krum, Bulyan, Centered Clipping (CClip), and Non-IID s-Bucketing
**Status:** Conformance Passed (`W9-01`, `W11-01`, Wave 13 `13C-2`) | **Date:** 2026-11-01 (Amended 2026-11-14)  
Enforces finite/shape pre-checks (`MAX_COHORT_SIZE = 1024`), `f64` widening, `total_cmp` ordering, one-pass `Multi-Krum`, `BulyanProfile::IterativeKrumV1` with Q9 Stage-2 lower observed median (`sorted[(theta - 1) / 2]`) and canonical `(|v - M|, PeerId)` tie-breaking (`bulyan_stage2_with_peers` alongside slice helper `bulyan_stage2_nearest_to_median`, explicitly documenting non-equivariance for even $\theta$), stateful `CenteredClipping` (`CClip`) with explicit reference-iterate contracts, conservative `$s$-Bucketing` with uniform $1/s$ weights and $f_B = \min(f, B)$ quorum re-validation, and IEEE-754 `sqrt_f64_soft` in `swarm-torch-core::num`.

## ADR-0020: Bounded-State Replay Protection, Receiver-Bound Context Recovery, and Cohort Admission Policy
**Status:** Conformance Passed (`W9-02`, Wave 13 `13A-2`, Wave 14 `14A` `AR-14-05`) | **Date:** 2026-11-01 (Amended 2026-11-17)  
Uses a 100% safe dual-tree `BTreeMap` replay store (`entries_by_peer` / `expiry_index` plus v2 tables), explicit `CohortAllowlist` admission, default `RequireSessionHandshake` startup/restart gating (`ReplayStartupPolicyV1`), monotonic timestamp floor bounds (`with_timestamp_floor`), and 6-stage `MessageEnvelope::verify_and_decode_payload` verification. Phase `13A-2` adds `SessionHandshakeChallengeV2` with 256-bit receiver nonce (`session_id = Hash(domain || swarm_id || run_id || receiver_peer || sender_peer || epoch || initial_seq || receiver_nonce)`), v2 replay keys scoped by `(SwarmInstanceId, RunId, PeerId, SessionEpoch, SessionId)`, atomic `complete_recovery_handshake_v2` (capacity validated prior to clearing pending challenges), and two-step `check_v2_admission_readonly` -> `commit_prepared_admission`. Wave 14 (`AR-14-05`) persists per-session `high_water_sequence` inside `peer_epochs` and `active_sessions_v2` so sequence numbers can never regress or replay across idle-window expiry (`> max_clock_skew_secs`) within an active session, prunes expired sessions at `current_time` rather than packet `timestamp`, and enforces `check_timestamp_only` in `validate_sequence_scoped_at`.

## ADR-0021: Source-Accurate Compression Validation, Stateful Error Feedback (`ClassicResidual` & `EF21`), and Verifiable Shared-Seed Sparse Co-Design
**Status:** Conformance Passed (`W9-03`, `W11-02`, Wave 13 `13C-3`) | **Date:** 2026-11-01 (Amended 2026-11-14)  
Provides Top-$k$, `Quantized`, `ClassicResidualCompressor`, `Ef21SenderState`/`Ef21ReceiverState`, Robert Floyd $\mathcal{O}(k)$ `sample_shared_indices_v1`, binary SHA-256 Merkle vector commitments (`PreSeedVectorCommitmentV1` + `MerkleOpeningBatchV1`), `SharedSeedResidualCompressor`, and compressed-subspace Byzantine aggregation. Phase `13C-3` (`ST-12`) wires end-to-end two-phase collective `SharedSeedSparse` in `swarm-torch-core`, `swarm-torch-net` (`PayloadKindV1::{SharedSeedCommitmentV1, SharedSeedOpeningV1}`), and `SwarmRoundCoordinator` (`SparseRoundContextV1`, `SharedSeedCommitmentMessageV1`, `SharedSeedOpeningMessageV1`, `compute_commitments_root_v1`, `derive_context_bound_round_seed_v1`, `SparseBeaconPolicyV1::{PrecommittedHashLock, Ed25519TranscriptBeacon}`, `SharedSeedCoordinateVerifier`, and `SharedSeedBandwidthReportV1` reporting honest per-regime proof + framing overhead).

## ADR-0022: Additive Transport Traits, Observed-Source Pre-Auth Reassembly, Originator-Authenticated Gossip Wrappers, and Zenoh/Iroh Adapter Crates
**Status:** Core, Auth Identity, Bounded Reassembly, Evented Async I/O, Live Adapters, & `no_std` LoRa/BLE HAL Transports Landed (`W10-03`, `W11-03`, `13A`, `13C-1`, `13D-2`, `13D-3`, `15C-3`, `15D-2`, `15D-3`, `16E-1`, `16E-2`); Live Adapters `Integration verified`, LoRa/BLE `Simulation verified` | **Date:** 2026-11-01 (Amended 2026-11-21)  
- Splits `PacketTransport`, GAT `SwarmTransport` (`ObservedTransportMessage` + `ObservedSourceRef` eliminating sentinel `PeerId([0xFF; 32])` attribution), originator-signed `InnerUpdateV2` (`MessageEnvelope::VERSION_V2` `(0, 3)`) + `RecipientWrapperV2` + `BundledGossipDeliveryV2` (binding `run_id`, `session_id`, and compression schema version alongside explicitly gated legacy `VERSION_V0` `(0, 1)` / `VERSION_V1` `(0, 2)` compatibility paths), and frozen 16-byte `ChunkedFrameCodec`.
- Pre-auth `ObservedSourceReassembler<E>` (`ST-13`) uses dimensioned `Vec<u64>` bitmaps supporting `MAX_CHUNKS_PER_MESSAGE > 256` without silent modular wrap, contiguous `Vec<u8>` buffer storage (zero-copy completion move), non-evicting quarantine tombstones with $O(\log S)$ source quarantine escalation, and full peak heap reservation accounting (`payload + bitmap + per-transfer + source/tombstone overhead`).
- Wave 12–16 (`W12-02`, `13A`–`13D`, `15C-3`, `15D-2`, `15D-3`, `16E-1`, `16E-2`): `swarm-torch-net::carrier` seals `VerifiedLinkPrincipalV1` (private fields constructible only via adapter minting / link validation hooks, never forged from external raw `[u8; 32]` bytes) alongside `QuicStreamHandshakePolicyV1`, `QuicLengthPrefixedCodecV1`, and `ZenohTopicContractV1`; `MockNetworkHub`, `TcpTransport`, and `UdpGossipTransport` replace polling `thread::sleep` loops with bounded `WaiterQueue` waker notifications, Slowloris idle deadlines, Windows `WSAECONNRESET` isolation, bounded security audit ring, and evented Tokio OS socket I/O; `adapters/swarm-torch-transport-iroh` (`IrohEndpointBridgeV1`, `iroh 1.4.1`, MSRV `1.91`, strict 1-RTT QUIC with zero `into_0rtt()`, bounded accept/stream timeouts, CSPRNG nonces, ALPN pinning, TLS `remote_id == attestation_key` challenge-response, and pre-body sliding-window quotas) and `adapters/swarm-torch-transport-zenoh` (`ZenohSessionBridgeV1` + `ZenohValidatedSubscriberV1`, `zenoh 1.10.2`, MSRV `1.88`, `time` patched for `RUSTSEC-2026-0009`, isolated-peer rejection, pre-copy payload caps, and canonical topic enforcement) sit at **`Integration verified`**; and pure `no_std` hardware-agnostic radio packet transports `swarm_torch_net::lora` (`lora-transport`: `LoraPacketTransportV1<R>`, `LoraRadioHal`, exact Semtech AN1200.13 integer-$\mu\text{s}$ Time-on-Air, 61-slot `prepare -> tx -> commit` `LoraDutyCycleLimiterV1`, `250`B MTU / `234`B chunk stride) and `swarm_torch_net::ble` (`ble-transport`: `BlePacketTransportV1<H>`, `BleGattHal`, Bluetooth Core Spec Vol. 3 Part F §3.2.8–§3.2.9 hard `512`B ATT attribute ceiling, `247`B DLE -> `244`B notify -> `228`B stride, conservative static floor vs. per-peer MTU, pre-TX credit gating, `send_chunked_message_v1`) sit at **`Simulation verified`** across Rust `1.75` and `thumbv6m`/`thumbv7em`/`riscv32imc` (physical RF SPI/HCI drivers out of tree; `wifi-transport` deferred).

## ADR-0023: Unified Swarm Round Coordinator, Physical CAS Checkpoints, Strict Manifest Trust Policy, Cross-Process Bundle Locking, `FedAsync`, and Certified Discrete-Gaussian `DpFedAvgV1` (`ST-02`)
**Status:** Auth Admission, Transactional Storage, & Certified `DpFedAvgV1` Conformance Passed (`13A`, `13B-1`, `13B-2`, `16B-1`..`16B-5`) | **Date:** 2026-11-01 (Amended 2026-11-21)  
- Establishes persistent `fs4`-locked `.bundle.lock` (`BundleLockGuard`, kernel `flock` / `LockFileEx` with zero unlink races), atomic `.staging.*` writes with `sync_all()` + `sync_parent_dir` (POSIX directory `fsync`; documented non-Unix no-op where regular-file `sync_all()` calls `FlushFileBuffers` without uniform OS directory-entry flush guarantees), `DurabilityPolicyV1` (`SyncEachAppend` default for DP ledgers and DataOps streams; `SyncAtCommit` for telemetry streams), `ManifestSignatureV1` over raw 16-byte `RunId`, single-pass `VerifiedBundleV1` with write-once CAS SHA-256 verification, and `RunArtifactSink` re-entrant append lock avoidance (`write_graph_unlocked`).
- Wave 13 Phase `13A-1` (`ST-01`, `ST-03`) & Wave 15 (`15A`): Replaces unauthenticated `PeerRoundSubmissionV1` ingress (`#[deprecated]`, failing closed with `UnverifiedSubmissionRejected` and zero state mutation) with a strict 3-stage pipeline (**Stage A** `prepare_verified_delivery` read-only Ed25519/v2-replay/compression check -> `PreparedSubmissionV1`; **Stage B** `ExpectedRoundContextV1::validate_{sync,fedasync,lora}_submission` read-only instance/policy-digest/round/digest/allowlist/epoch check -> **Stage C** atomic replay commit & coordinator admission yielding sealed private-field proofs `VerifiedRoundSubmissionV1`, `VerifiedFedAsyncSubmissionV1`, `VerifiedLoraSubmissionV1`). Legacy `SwarmRoundCoordinator::from_resume_state` unconditionally fails closed (`RoundError::ResumeStateMismatch`), while `resume_exact_training_state` and `restore_weights_only` enforce `RequireSessionHandshake` and a monotonic timestamp floor (`with_timestamp_floor`).
- Wave 13 Phase `13B-1` (`ST-04`): Enforces atomic round transitions on `SwarmRoundCoordinator::finalize_round`, `step_fedasync_*`, and `aggregate_lora_cohort_*` via staged artifacts before promoting `checkpoints/committed_head.json` (`CommittedHeadV1` signed inside the bundle manifest and verified via `RunArtifactBundle::verify_committed_head`), storing full `CoordinatorResumeStateV2`, `OptimizerStateSnapshotV1`, `CenteredClippingCenterBlobV1`, and `HistoricalAnchorsBlobV1` in write-once CAS so failed commits preserve prior committed state cleanly and restarts choose `RestoreModeV1::{ExactTrainingState, WeightsOnlyMigratePolicy}` (`resume_exact_training_state` / `restore_weights_only`).
- Wave 12 (`W12-03`), Wave 13 (`13-0`), & Wave 16 (`16B-1`..`16B-5`, `ST-02`): Implements `apply_fedasync_stale_delta_v1` (`swarm-torch-core::aggregation`) with `no_std` IEEE-754 `exp_f64_soft`/`ln_f64_soft` (`num.rs`), `historical_anchors`, and strictly segregated DP ledger accounting (`dp/ledger.ndjson`) supporting:
  - **`ExperimentalHypotheticalBudgetOnly`** (`guarantees_differential_privacy = false`): budget-only accounting with zero DP guarantee claims.
  - **`VerifiedMechanismProtected`** (`guarantees_differential_privacy = true`, **`Adversarial conformance passed`**): exact integer-lattice $\ell_2$ clipping (`DP_DISCRETE_LATTICE_RADIUS_V1 = 65_536`, `u128`-checked $\|\hat{\Delta}_i\|_2^2 \le B^2$), associative `i128` summation, floating-point-free Canonne, Kamath, & Steinke (2020, `arXiv:2004.00010`, Prop. 5 / JPC Prop. 2.4 & Algorithms 1–3) Discrete Gaussian perturbation (`sample_dgauss_u128`) seeded from OS CSPRNG (`getrandom 0.2.17`, first-party audited), deterministic `f32` post-processing via fixed expected denominator $m = q N \ge 1.0$, sealed `SampledDpCohortV1` + `DpNoiseEntropyV1` + `VerifiedDpMechanismRoundV1` proof binding, one-shot anti-resampling Poisson cohort lock (`begin_dp_poisson_round_os_rng_v1`), zero-conditioning release on empty/below-quorum/dropout rounds, external bundle telemetry redaction (`output_rows = None`, omitting `swarmtorch.admitted_peers` under the trusted central coordinator vs. external observer model), and non-refundable crash-consistent pre-release debits (`pre_release_debit_verified_dp_accountant`). Supported strictly for synchronous `FedAvg` + `EqualPeerVote` + `ClientLevel` under `AddRemoveOne` ($q \in (0, 1]$, Mironov, Talwar, & Zhang 2019 §3.3 & Theorem 5) or `ReplaceOne` at full participation $q = 1.0$ (`verified_dp_adjacency_subsampling_supported_v1` fails closed on `ReplaceOne` with $q < 1.0$ across all 6 sibling gates; `ExampleLevel`, `FedAsync`, `LoRA`, `SharedSeedSparse`, `$s$-Bucketing`, `RegistryCertified`, and non-linear aggregators remain fail-closed).

## ADR-0024: Reference Models, Analytical Losses, SafeTensors Checkpoint Profile, and Isolated Framework Backend Adapters
**Status:** Core Models & Live Burn (CPU + WGPU) / Candle (CPU) Bridges Landed (`W9-05`, `W11-04`, `W12-01`, `13D-4`, `14B`, `15D-4`, `16D`); Backends `Integration verified` | **Date:** 2026-11-02 (Amended 2026-11-21)  
- Ships `LinearModelWithCapacity`, contiguous `MlpModelWithCapacity`, analytical losses, `SgdOptimizer`, `FedProxRegularizer`, and zero-dependency `SwarmTorchCheckpointV1` SafeTensors v1 codec (`ModelTensorSchemaV1`, `ModelFlatTensorLayoutV1`, `SafeTensorsLimitsV1` defaulting to `max_header_bytes = 1_048_576`, `export_safetensors_checkpoint_with_layout_v1`, `import_safetensors_checkpoint_with_layout_v1` mapping declaration-ordered `[W | b]` / `[W1 | b1 | W2 | b2]` flat buffers against alphabetically sorted SafeTensors entries while preserving `model_schema_hash`).
- Wave 12–16 (`W12-01`, `13D-4`, `14B`, `15C-2`, `15D-4`, `16D`, `ST-08`): Adds `LoraAdapterPairV1`, `FloraStackedDeltaV1`, `LoraAggregationModeV1` (`FreezeASingleFactorV1`, `FloraStackingV1`, `BiasedFactorFedAvgV1`), `aggregate_lora_adapters_v1`, and `BackendTensorBridge`/`BackendModelHandleV1` in `swarm-torch-models`, coordinated atomically with staged base weights in `SwarmRoundCoordinator::aggregate_lora_cohort_verified_round`, plus **`Integration verified`** live framework bridges in `adapters/swarm-torch-backends` (MSRV `1.92`, Decision `A5`; `real-burn`: `BurnLinearBridgeV1` / `BurnMlpBridgeV1` over `burn 0.21.0` `Autodiff<NdArray<f32>>`; `real-burn-wgpu`: `BurnWgpuLinearBridgeV1` / `BurnWgpuMlpBridgeV1`, `WgpuDeviceSelectorV1`, serialized backend-scoped `enumerate_wgpu_devices()`, `resolve_wgpu_device` / `probe_wgpu_device` over `burn-wgpu 0.21.0` / `cubecl 0.10.0` / `wgpu 29.0.4`, verified live on NVIDIA GeForce RTX 5080 and synthetic mixed-GPU topologies; `real-candle`: `CandleLinearBridgeV1` / `CandleMlpBridgeV1` over `candle-core`/`candle-nn` `0.10.2` with pre-set `dtype`/`device`/`shape` validation), verified with cross-framework SafeTensors forward & MSE gradient parity tests; `DevicePlacementV1::Cuda` remains `Contract defined`.

## ADR-0025: Fail-Closed Wasm Extension Containment (`wasmi` Adapter, `RawMem0V0` & `SwarmTorchAllocV1` Guest ABIs) and `PhysicalStrict` Columnar Data Runner
**Status:** Conformance Passed (`W12-04`, `P2-20b`, `13B-3`, `13D-1`, `15D-1` `ST-09`, `16C` Addendum); Adapter `Adversarial conformance passed` (`RawMem0V0` + `SwarmTorchAllocV1`) | **Date:** 2026-11-02 (Amended 2026-11-21)  
- Default `NativeOpRunner::new()`, `WasmSandboxRunnerV1::{execute_byte_transform, execute_scalar_i32}`, and `Scheduler` (`RunnerIsolationProfile::WasmSandboxedV1` without a configured `WasmExtensionExecutor`) **fail closed** (`UnsupportedUntrustedWasmSandbox`); `adapters/swarm-torch-wasm` (MSRV `1.86`) implements `WasmiExtensionExecutorV1` over `wasmi 2.0.0` (`auto-dispatch`, `validate`, `deterministic`) with `EnforcedLimits::strict()`, `StoreLimits` covering both exported and un-exported tables/memories, bounded value/call stacks, `WasmStoreLimitsV1::max_start_fuel`, zero imports, SHA-256 module pinning, fuel + wall-clock timeout slicing via `TypedResumableCall`, and end-to-end `DataOpsSession` physical CAS integration (`NativeOpRunner::with_wasm_executor`).
- **Wave 16 Addendum (`16C`):** Extends `SandboxedExtensionPolicyV1` with `guest_abi: WasmGuestAbiV1` (`RawMem0V0` backward-compatible default or `SwarmTorchAllocV1` structured guest ABI) and `WasmStoreLimitsV1::max_output_bytes` (`DEFAULT_WASM_MAX_OUTPUT_BYTES = 16 MiB`), both hashed into `policy_fingerprint_hex()`. Under `SwarmTorchAllocV1`, `preflight_wasm_extension_v1` rejects reserved `swarmtorch_alloc`/`swarmtorch_dealloc` export targets, `prepare_instance` validates target export kind/signature, `alloc`/`dealloc` signatures, and exported linear memory BEFORE guest `(start)` executes, `max_output_bytes` is enforced before host `Vec<u8>` allocation, strict 4-case ownership/aliasing/single-free dealloc rules prevent double-free and reject interior/partial overlap (`OutputAliasingViolation`) and cleanup traps fail closed, and a single cumulative fuel + wall-clock budget meters `start + alloc + transform + dealloc`.
- The legacy bespoke subset interpreter is gated strictly under `#[cfg(test)]` inside `swarm-torch/src/wasm_sandbox.rs` and excluded from all production/library builds.
- Wave 13 Phase `13B-3` (`ST-10`, `ST-14`) & Wave 14–15 (`14A`–`15D`, `AR-14-01`..`AR-14-09`): `NativeOpRunner::new()` defaults to `RunnerExecutionModeV1::PhysicalStrict`, failing closed on legacy `run_with_context`, non-CAS or missing inputs, `{` byte sniffing (`ColumnarInputFormatV1`), partial `filter_rows` `column`/`value` pairs, unknown filter operators, or oversized NDJSON lines (`AR-14-08`). `WasmiExtensionExecutorV1` caps Wasm `(start)` instantiation fuel at `policy.limits.max_fuel.min(policy.limits.max_start_fuel)` (`AR-14-06`, `15D-1`). `CoordinatorResumeStateV2` persists `dp_profile` (and Wave 16 `dp_certified_subsampling_q`) so `resume_exact_training_state` automatically restores `dp_accountant` and verifies committed `.ndjson` prefixes across mid-round crash recovery (`AR-14-01`). `generate_report_with_policy` and `swarm_torch_report` enforce `TrustPolicy::Strict` (`--public-key-hex`) unless `--legacy-permissive-untrusted` is explicitly passed (`AR-14-02`). `BurnLinearBridgeV1`, `CandleLinearBridgeV1`, `BurnMlpBridgeV1`, and `CandleMlpBridgeV1` align in-memory parameter/gradient layouts to `[W | b]` and `[W_1 | b_1 | W_2 | b_2]` (`AR-14-07`, `14B`), `TcpTransport`/`UdpGossipTransport` use non-blocking async `broadcast` and bound concurrent `ActiveConnection` slots (`AR-14-09`), and `.github/workflows/rust.yml` runs standalone adapter conformance lanes (`AR-14-03`, `AR-14-03R`).
