# Changelog

All notable changes to SwarmTorch are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-11-14 (Wave 13 Phases `13-0` & `13A`)

### Security (`Phase 13A` — Auth Admission, Replay V2, & Bounded Reassembly)
- **3-Stage Authenticated Coordinator Admission (`ST-01`):**
  - Replaced unauthenticated `PeerRoundSubmissionV1` ingress (`#[deprecated]`, failing closed with `RoundError::UnverifiedSubmissionRejected` and zero state mutation) with a strict prepare -> validate -> commit pipeline in `swarm-torch::coordinator`: **Stage A** `prepare_verified_delivery` / `prepare_verified_delivery_struct` (`&self`, read-only Ed25519 + v2 replay preflight + `InnerUpdateV2` schema decode into `PreparedSubmissionV1`), **Stage B** `ExpectedRoundContextV1::validate_{sync,fedasync,lora}_submission` (read-only binding of `swarm_instance_id`, `context_policy_digest`, `run_id`, `round_id`, `base_checkpoint_digest`, update kind/dimension, allowlist, and active cohort epoch), and **Stage C** atomic replay commit & admission returning sealed private-field proof tokens (`VerifiedRoundSubmissionV1`, `VerifiedFedAsyncSubmissionV1`, `VerifiedLoraSubmissionV1`).
  - Added public canonical update payload encoders `encode_{dense,compressed,lora}_update_payload_v2` matching compression schema versions `0`, `1`, and `2`.
- **Receiver-Nonce `session_id` V2 Replay Recovery & Default Fail-Closed Restart (`ST-03`):**
  - Defaulted `ReplayProtection` startup/recovery policy to `ReplayStartupPolicyV1::RequireSessionHandshake` (`ReplayProtection::try_with_config`) and wired `SwarmRoundCoordinator::from_resume_state` to enforce `RequireSessionHandshake` plus monotonic `TimestampFloor` (`committed_unix_nanos / 1e9`).
  - Added `SessionHandshakeChallengeV2` with 256-bit `receiver_nonce` and deterministic domain-separated `session_id`, `ProtocolVersion::V0_3` (`(0, 3)`), `InnerUpdateV2`, `RecipientWrapperV2`, `BundledGossipDeliveryV2`, `BTreeMap` v2 replay tracking keyed by `(SwarmInstanceId, RunId, PeerId, SessionEpoch, SessionId)`, atomic `complete_recovery_handshake_v2` (capacity validated prior to mutating pending challenges), and read-only `check_v2_admission_readonly` + `commit_prepared_admission`.
- **Non-Forged Transport Attribution & Sealed Carrier Proofs (`ST-06` Identity, `ST-08`):**
  - Replaced `SwarmTransport::recv() -> (PeerId, Vec<u8>)` (and sentinel `PeerId([0xFF; 32])` fallbacks in `UdpGossipTransport` / `TcpTransport`) with `ObservedTransportMessage` carrying transport-observed `ObservedSourceRef` (`SocketAddr`, `MockNodeId`, `LinkLabel`, `Opaque`), preventing pre-auth sender forgery across all transports and multi-node simulations.
  - Sealed `VerifiedLinkPrincipalV1` in `swarm-torch-net::carrier` (private fields constructible only via adapter minting / link validation paths, rejecting external raw `[u8; 32]` construction).
- **Dimensioned Bitmaps, Non-Evicting Quarantines, & Full Heap Accounting in Reassembly (`ST-13`):**
  - Re-architected `ObservedSourceReassembler<E>` (`swarm-torch-net::chunked`) with dynamically dimensioned `Vec<u64>` bitmaps (`ceil(total_chunks / 64)` words supporting `MAX_CHUNKS_PER_MESSAGE > 256` without silent modular wrap), contiguous `Vec<u8>` reassembly buffers (zero-copy move upon completion), non-evicting conflict quarantine tombstones paired with $O(\log S)$ source-level quarantine escalation (`SourceQuarantined`), and complete peak heap reservation accounting (`payload + bitmap + per-transfer + source/tombstone overhead`).

### Security (`Phase 13-0` — Containment)
- **Differential Privacy Accounting Containment (`ST-02`):**
  - Reclassified `RdpAccountantV1` (`swarm-torch-core::dp`), `dp/ledger.ndjson` (`swarm-torch::artifacts::dp_ledger`), and `McpInspectorServerV1::tool_trace_aggregation_budget` (`swarm-torch::mcp_inspector`) as **`ExperimentalHypotheticalBudgetOnly`** (`guarantees_differential_privacy = false`).
  - Added fail-closed `VerifiedMechanismProtected` rejection across `RdpAccountantV1`, `dp_ledger.rs`, and `SwarmRoundCoordinator` (`supports_certified_differential_privacy() == false`, `RoundConfigError::UnsupportedCertifiedDifferentialPrivacy`), and staged `base_weights_to_merge` validation/application prior to pre-release DP debit in `SwarmRoundCoordinator::aggregate_lora_cohort_round`.
- **Wasm Sandbox Containment (`ST-09`):**
  - Made `WasmSandboxRunnerV1::{execute_byte_transform, execute_scalar_i32}`, `Scheduler` (`RunnerIsolationProfile::WasmSandboxedV1`), and `NativeOpRunner` `"wasm_transform"` fail closed (`UnsupportedUntrustedWasmSandbox`) until `wasmi` integration lands (`Phase 13D-1`).
  - Moved the bespoke interpreter strictly behind `#[doc(hidden)] WasmSandboxRunnerV1::run_experimental_trusted_test_fixture_only`, rejecting `F64` (`0x7c`), bounding cumulative locals per function (`4096`), and metering decode/pre-scan fuel.
- **Iroh Carrier Adapter Hardening (`ST-08`):**
  - Fixed `IrohCarrierAdapterV1::{accept_envelope_frame, accept_gossip_frame}` (`adapters/swarm-torch-transport-iroh`) to enforce `bind_originator` and wrapper/inner originator parity **before** mutating `ReplayProtection`, replaced unbounded `counts` with sliding-window bounded `LinkFrameQuotaTrackerV1` (`DEFAULT_MAX_TRACKED_LINK_PRINCIPALS = 1024`, `DEFAULT_MAX_WINDOW_TIMESTAMPS_PER_LINK = 1024`, lazy expiry + fail-closed `LinkQuotaTableFull`), and hid `simulated_link_principal_for_test` behind `#[doc(hidden)]`.

### Documentation
- **Public Bibliography & Adapter Maturity Transparency (`ST-15`):**
  - Restored `SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md` as a public file in `SwarmTorch`.
  - Updated `README.md`, `ROADMAP.md`, `SECURITY.md`, `ADRs.md`, adapter crate docs, and `Cargo.toml` radio-feature comments with explicit 6-tier adapter maturity labels and honest Wave 13 remediation scope.

## [0.2.0] - 2026-11-13 (Wave 12)

### Added
- **Isolated Framework Bridges & LoRA / FLoRA Aggregation (`W12-01`, `ADR-0024` §4):**
  - Added `BackendTensorBridge` and `BackendModelHandleV1` in `swarm-torch-models::backend_bridge`.
  - Added `LoraAdapterPairV1`, `FloraStackedDeltaV1`, `LoraAggregationModeV1` (`FreezeASingleFactorV1`, `FloraStackingV1`, `BiasedFactorFedAvgV1`), and `aggregate_lora_adapters_v1` in `swarm-torch-models::lora`, integrated with `SwarmRoundCoordinator::aggregate_lora_cohort_round`.
  - Added isolated `adapters/swarm-torch-backends` crate (`BurnModuleBridgeV1`, `CandleModuleBridgeV1`, excluded from root `Cargo.lock`).
- **Zenoh/Iroh Carrier Contracts & Isolated Adapters (`W12-02`, `ADR-0022` §4):**
  - Added `QuicStreamHandshakePolicyV1`, `QuicLengthPrefixedCodecV1`, and `ZenohTopicContractV1` in `swarm-torch-net::carrier`.
  - Added isolated `adapters/swarm-torch-transport-iroh` (`IrohCarrierAdapterV1`) and `adapters/swarm-torch-transport-zenoh` (`ZenohCarrierAdapterV1`) crates outside root `Cargo.lock`.
- **Stale-Delta `FedAsync` & Durable Pre-Release RDP Accounting (`W12-03`, `ADR-0023` §4):**
  - Added pure `no_std` IEEE-754 `abs_f64`/`abs_f32`, `exp_f64_soft`/`exp_f64`, `ln_f64_soft`/`ln_f64`, and `log_sum_exp_pair_f64` in `swarm-torch-core::num` (MSRV 1.75 compatible).
  - Added `FedAsyncStalenessDescriptorV1`, `StalenessWeightingPolicy`, `FedAsyncStaleMixOutcomeV1`, and `apply_fedasync_stale_delta_v1` in `swarm-torch-core::aggregation`.
  - Added `RdpAccountantV1` (`DpMechanismProfileV1`, `DpUnitV1`, `DpAdjacencyV1`, `DpConversionFormulaV1`, `restore_from_orders`) in `swarm-torch-core::dp` and crash-idempotent `dp/ledger.ndjson` debits in `swarm-torch::artifacts::dp_ledger`, integrated into `SwarmRoundCoordinator` (`configure_dp_accountant`, `enable_fedasync`, `step_fedasync_stale_submission`, pre-release debit before checkpoint publication in `finalize_round`).
- **Deterministic Wasm Sandbox, Columnar Data Runner, & Schema `P2-20b` (`W12-04`, `ADR-0025`):**
  - Published `schemas/v1/columnar_table.schema.json` (mirrored byte-identically in `swarm-torch/schemas/v1/columnar_table.schema.json`) and registered `COLUMNAR_TABLE_SCHEMA_JSON` in `swarm-torch::artifacts::schema_check`.
  - Added pure-Rust `#![forbid(unsafe_code)]` fuel/memory-limited `WasmSandboxRunnerV1` in `swarm-torch::wasm_sandbox` with pre-parse SHA-256 allowlists, WASI/host import rejection, `Start` section execution under limits, and checked `(ptr, len)` guest slice reads.
  - Added `RunnerIsolationProfile::WasmSandboxedV1` enforcement in `swarm-torch::scheduler` (`execute_graph_with_isolation`) and schema-validated physical columnar/NDJSON execution (`ColumnarTableV1`, `PhysicalExecOutcomeV1`) in `NativeOpRunner::run_in_session`.
- **Zero-Port Read-Only Strict MCP Inspector (`W12-05`, `ADR-0016`):**
  - Added `McpInspectorServerV1` and the `swarm_torch_mcp` `stdio` JSON-RPC 2.0 binary (`swarm-torch::mcp_inspector`), requiring `TrustPolicy::Strict` verification on every request and serving 4 read-only `swarmtorch://runs/{run_id}/*` resources and 4 inspection/audit/prediction/DP-trace tools.
  - Added isolated `adapters/swarm-torch-mcp` crate (`SwarmTorchMcpAdapterV1`).

### Changed
- Compacted public `ADRs.md` to concise normative contracts and moved whitepaper source bibliography (`SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md`) and unabridged deliberations to `swarm-torch-planning`.

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
