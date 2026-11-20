# Changelog

All notable changes to SwarmTorch are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-11-19 (Wave 15 `15A`–`15E` — Full-Spectrum Audit Remediation & Honesty Reset)

### CLI Auth, Bundle Integrity, Coordinator Admission, & Cache Trust (`15A`)
- Fixed `swarm_torch_report` CLI trust-flag XNOR inversion (`parse_report_cli_args` strict mutual exclusion) and added end-to-end `swarm_torch_report` / `swarm_torch_mcp` subprocess CLI tests (`swarm-torch/tests/cli_binaries.rs`).
- Hardened `RunArtifactBundle` (`bundle.rs`, `cas.rs`, `io.rs`, `dp_ledger.rs`, `report/{load,render}.rs`) with OS-independent lexical `validate_manifest_path`, full entry SHA-256 verification on `verify_committed_head`, symlink and lowercase-hex guards, atomic report writes, and manifest-bound DP ledger reconstruction (failing closed in `restore_weights_only` when a DP ledger/profile exists).
- Closed Stage B/C ingress asymmetry across `FedAsync`, `LoRA`, and sparse rounds (`coordinator.rs`), failed closed on deprecated unauthenticated `from_resume_state`, persisted `equivocated_peers` across rounds and crash recovery (`Decision A9`), and replaced `saturating_add` on generation counters with `checked_add`.

### DataOps Scheduler, Native Runner, & JSON Schema Conformance (`15B`)
- Hardened `scheduler.rs` / `session.rs` pre-cache isolation, trust-laundering prevention, atomic CAS bindings, and failure invalidation (`15B-1`), `native_runner.rs` columnar row/duplicate/digest/`i64` predicate bounds (`15B-2`), and `schema_check.rs` Unicode char lengths + `regex 1.12` pattern checks (`Decision A7`, `15B-3`) alongside strict `artifact_pipeline`.

### Core, Models, Net, & Runtime Hardening (`15C`)
- Unified `TrimmedMean`/Top-$k$ precision, expired challenge purge, finite `MetricRecord`, and `DatasetEntryV1` bounds (`15C-1`); hardened LoRA duplicate-peer / FLoRA bounds, `SgdOptimizer` stack/backend sync, and SafeTensors custom limits + RFC 8259 escapes (`15C-2`).
- Promoted `MessageEnvelope::CURRENT_VERSION` to `VERSION_V2` (`(0, 3)`) with gated legacy fallback (`Decision A10`), hardened `TcpTransport` against Slowloris stalls, isolated `UdpGossipTransport` from Windows `WSAECONNRESET` noise, and added bounded security audit ring emission (`Decision A8`, `15C-3`).
- Adopted `portable-atomic 1.15.0` (`fallback` + `critical-section`, audited in `supply-chain/audits.toml`, `Decision A6`) for 32-bit targets (`thumbv6m`, `thumbv7em`, `riscv32imc`) and hardened `MockRuntime` timer `Drop` / waker lifecycles (`15C-4`).

### Standalone Adapter Remediation & Official `rmcp 3.11` Integration (`15D`)
- `adapters/swarm-torch-wasm` (`15D-1`, MSRV `1.86`, `Integration verified`, scoped to `raw_mem0_v0` per `Decision A2`): hardened `wasmi` stack depth/size, `WasmStoreLimitsV1::max_start_fuel`, cached `Engine`, and un-exported internal table/memory checks.
- `adapters/swarm-torch-transport-iroh` (`15D-2`, MSRV `1.91`, `Integration verified`): bounded every accept/handshake/stream/ack stage with `IrohAcceptTimeoutsV1`, gated pre-auth quotas before body reads/allocation, and added CSPRNG nonces with reuse rejection.
- `adapters/swarm-torch-transport-zenoh` (`15D-3`, MSRV `1.88`, `Integration verified`): rejected isolated peers without explicit endpoints, disabled hidden default wildcard listeners, and bumped MSRV to `1.88` (founder decision superseding `A5`) so `time` stays patched for `RUSTSEC-2026-0009`.
- `adapters/swarm-torch-backends` (`15D-4`, MSRV `1.92`, `Integration verified`, `Decision A5`): achieved `ReferenceLinearModule` finiteness/batching/overflow parity, pre-validated Candle `VarMap` dtype/device/contiguity before mutation, documented Burn mutex requirement, and scoped docs to CPU-only.
- `adapters/swarm-torch-mcp` (`15D-5`, MSRV `1.88`, `Integration verified`, `Decision A3`): integrated the official `rmcp 3.11` SDK (`ServerHandler`, `AsyncRwTransport::with_max_line_length`, `stdio_pipes`) with bounded semaphore concurrency, cancellation awareness, canonical `additionalProperties: false` & pagination validation in `McpInspectorServerV1`, `resources/templates/list` parity, real OS pipe subprocess conformance (`rmcp_stdio_subprocess`), and MSRV-aware `1.88` `Cargo.lock`.

### Lockfiles, CI Matrix, Supply-Chain Honesty, & Docs Sync (`15E`)
- Committed all 5 standalone `adapters/*/Cargo.lock` files + `.github/scripts/check-locked-lockfiles.sh` + `.gitattributes` + `adapters/deny.toml`, expanded `.github/workflows/rust.yml` (`--all-features`, `--doc`, `RUSTDOCFLAGS="-D warnings"`, workspace `1.75`/`1.81` + all 5 adapter MSRV lanes, `riscv32imc`, and adapter `cargo deny`), replaced `rust,ignore` doctests in `swarm-torch/src/lib.rs`, corrected local vs CI supply-chain status in `supply-chain/exemption-rationale.md`, and aligned public docs (`README.md`, `ROADMAP.md`, `SECURITY.md`, `ADRs.md`, `CHANGELOG.md`, whitepaper) to verified live Rust symbols and capability tiers.

## [0.3.0-alpha.14] - 2026-11-17 (Wave 14 Phases `14A`, `14B`, `14C`, & `14D` — Full-Spectrum Wave 13 Audit Remediations `AR-14-01`..`AR-14-09`)

### Replay Hardening, Cryptographic DP Resume, & Verified Reports (`Phase 14A` — `AR-14-05`, `AR-14-01`, `AR-14-02`)
- **Active-Session Monotonicity Across Idle Windows (`AR-14-05`):**
  - Updated `ReplayProtection` (`swarm-torch-core::replay`) to persist per-session `high_water_sequence: u64` inside `peer_epochs` and `active_sessions_v2` so sequence numbers can never regress (`TooOld`) or replay (`Replay`) after sliding-window expiry (`> max_clock_skew_secs`) and record reclamation within the same `(session_epoch, session_id)`.
  - Fixed `commit_v1_unchecked` and `commit_v2_unchecked` to prune expired sessions at `current_time` rather than packet `timestamp`, enforced session-map capacity during read-only preflight (`check_v*_admission_readonly`) and recovery handshakes, and enforced `check_timestamp_only` in `validate_sequence_scoped_at`.
- **Cryptographic DP Accountant Resume & Crash-Consistent NDJSON Prefix Verification (`AR-14-01`):**
  - Added `dp_profile: Option<DpMechanismProfileV1>` to `CoordinatorResumeStateV2` (`swarm-torch::artifacts::cas`), `parse_and_validate_dp_ledger_slice` + `reconstruct_dp_accountant_from_verified_bundle` (`dp_ledger.rs`), and automatic `dp_accountant` reconstruction in `SwarmRoundCoordinator::resume_exact_training_state` (`coordinator.rs`), failing closed if `dp/ledger.ndjson` exists on disk while `dp_accountant` is unconfigured.
  - Unified `RunArtifactBundle::verify_committed_head` (`bundle.rs`) to verify committed `.ndjson` prefixes (`&bytes[..entry.bytes]`) against the signed checkpoint manifest across mid-round `AfterCasBeforeManifest` crash recovery while preserving non-refundable pre-release DP debits.
- **Verified Report Generation & CLI Trust Enforcement (`AR-14-02`):**
  - Added `verification_status: VerificationStatus` and `dp_accounting_classification: Option<String>` to `Report`, added `generate_report_with_policy` / `generate_report_html_with_policy` (`swarm-torch::report`), and updated `swarm_torch_report` CLI (`src/bin/swarm_torch_report.rs`) to require explicit `--public-key-hex <64-hex>` (`TrustPolicy::Strict`) or `--legacy-permissive-untrusted` (`TrustPolicy::LegacyPermissive`).

### `[W | b]` Backend Layout Alignment & Live 2-Layer ReLU MLP Bridges (`Phase 14B` — `AR-14-07`, `14B-1`, `14B-2`)
- **In-Memory `[W | b]` Parameter & Gradient Alignment (`AR-14-07`):**
  - Aligned `ReferenceLinearModule`, `BurnLinearBridgeV1`, and `CandleLinearBridgeV1` (`adapters/swarm-torch-backends`) in-memory `SwarmModel::parameters()`, `load_parameters()`, and `backward_mse()` gradients to canonical `[weight | bias]` (`[W | b]`) order matching `SwarmTorch::LinearModel` (`ModelFlatTensorLayoutV1::for_linear`) 1:1 with zero manual index permutation.
- **2-Layer ReLU MLP Bridges (`BurnMlpBridgeV1` & `CandleMlpBridgeV1`) & Federated Round Parity (`14B-1`, `14B-2`):**
  - Implemented `BurnMlpBridgeV1` (`burn 0.21.0` `Autodiff<NdArray<f32>>`) and `CandleMlpBridgeV1` (`candle-core`/`candle-nn` `0.10.2`) in canonical `[W_1 | b_1 | W_2 | b_2]` layout matching `SwarmTorch::MlpModel` (`ModelFlatTensorLayoutV1::for_mlp`), verified with cross-framework forward/backward numerical equivalence (`< 1e-5`), SafeTensors checkpoint handoff, and a multi-peer `TrimmedMean` + `RdpAccountantV1` federated training round.

### Non-Blocking Broadcast, TCP Connection Caps, Live Transport Replay Tests, & Adapter CI (`Phase 14C` — `AR-14-09`, `14C-1`, `AR-14-03`)
- **Async `SwarmTransport::broadcast` & Bounded TCP Active Connections (`AR-14-09`):**
  - Updated `TcpTransport::broadcast` (`swarm-torch-net::tcp`) and `UdpGossipTransport::broadcast` (`swarm-torch-net::udp`) to await `self.send(peer, msg).await` rather than blocking on `send_sync` or non-awaited `send_to_addr`.
  - Added `DEFAULT_MAX_ACTIVE_TCP_CONNECTIONS = 256` (`TcpTransport::bind_with_limits` and `TcpFrameReaderState::is_idle`) to evict oldest idle connections at capacity and prevent Slowloris connection growth.
- **Live `Iroh` & `Zenoh` Post-Idle Sequence Rollback Conformance & Adapter CI (`14C-1`, `AR-14-03`):**
  - Added post-idle-expiry sequence rollback rejection and monotonic continuation tests over live `Iroh` 1-RTT QUIC (`adapters/swarm-torch-transport-iroh`) and live `Zenoh` TCP pub/sub (`adapters/swarm-torch-transport-zenoh`).
  - Wired the initial `standalone_adapters` CI job in `.github/workflows/rust.yml` (reopened as `AR-14-03R` and completed in Wave 15 with committed per-adapter `Cargo.lock` files, MSRV lanes, and corrected CI-verified supply-chain rationale in `supply-chain/exemption-rationale.md`).

### Wasm `(start)` Fuel Slice Cap, Bounded NDJSON & `filter_rows` Validation, & MCP Subprocess Conformance (`Phase 14D` — `AR-14-06`, `AR-14-04`, `AR-14-08`)
- **Wasm `(start)` Instantiation Fuel Slicing (`AR-14-06`):**
  - Capped initial fuel during `linker.instantiate_and_start` in `adapters/swarm-torch-wasm` to `policy.limits.max_fuel.min(FUEL_SLICE_UNITS)` (`10,000` units) before checking `timeout_ms` and refilling the remaining budget for resumable export execution.
- **Bounded NDJSON Parsing, Strict `filter_rows` Validation, & MCP Child-Process Test (`AR-14-04`, `AR-14-08`):**
  - Wired `McpInspectorServerV1::tool_trace_aggregation_budget` (`swarm-torch::mcp_inspector`) to `parse_and_validate_dp_ledger_slice` and added an OS child-process (`std::process::Command` stdio pipe) JSON-RPC conformance test in `adapters/swarm-torch-mcp` (official `rmcp 3.11` SDK integration and `Integration verified` promotion landed in Wave 15 `15D-5`).
  - Enforced `MAX_NDJSON_LINE_BYTES` / `MAX_NDJSON_LINES` in `NativeOpRunner::parse_ndjson_rows` and fail-closed validation on partial `column`/`value` pairs or unknown `op` strings in `NativeOpRunner::physical_filter_rows` (`swarm-torch::native_runner`).

## [0.3.0-alpha.13] - 2026-11-16 (Wave 13 Phases `13-0`, `13A`, `13B`, `13C`, & `13D`)

### External Integrations, `wasmi` Sandbox, & Cross-Framework Parity (`Phase 13D` — `ST-08`, `ST-09`)
- **Isolated `wasmi 2.0.0` Extension Sandbox & Bespoke VM Retirement (`ST-09`, `13D-1`):**
  - Added `WasmExtensionExecutor`, `WasmSandboxError::ExecutionTimeout`, `parse_sha256_hex_v1`, and `preflight_wasm_extension_v1` in `swarm-torch::wasm_sandbox`, gated the legacy bespoke VM behind `#[cfg(test)]` (zero bespoke VM code compiled in production `--lib` builds), and wired optional `NativeOpRunner::with_wasm_executor` + `Scheduler` `"wasm_transform"` execution over physical CAS assets (keeping unconfigured runners strictly fail-closed `UnsupportedUntrustedWasmSandbox`).
  - Added standalone workspace crate `adapters/swarm-torch-wasm` (`WasmiExtensionExecutorV1` over `wasmi 2.0.0` with `auto-dispatch`, `validate`, `deterministic`, `EnforcedLimits::strict()`, `StoreLimits`, zero imports, SHA-256 pinning, and sliced `TypedResumableCall` enforcing both fuel and wall-clock `timeout_ms` across infinite loops, memory/table bombs, OOB reads, and physical `DataOpsSession` DAG execution).
- **Live `iroh 1.4.1` 1-RTT QUIC Carrier Bridge (`ST-08`, `13D-2`):**
  - Added `IrohEndpointBridgeV1` and `IrohCarrierError::Transport` in `adapters/swarm-torch-transport-iroh`, enforcing `presets::Minimal` strictly 1-RTT QUIC connect/accept (no 0-RTT fast path), SwarmTorch ALPN pinning, challenge-response `MessageAuth` Ed25519 binding to `conn.remote_id()`, bounded reads, pre-auth sliding-window `LinkFrameQuotaTrackerV1`, and live localhost QUIC loopback verification.
- **Live `zenoh 1.10.2` TCP Peer Pub/Sub Session Bridge (`ST-08`, `13D-3`):**
  - Added `ZenohSessionBridgeV1`, `ZenohValidatedSubscriberV1`, `ZenohCarrierAdapterV1::topic_for_gossip_v2`, and `ZenohCarrierError::Transport` in `adapters/swarm-torch-transport-zenoh`, enforcing pre-copy `ZBytes::len()` caps and strict `swarmtorch/v1/...` topic + signature + v2 session replay verification over live peer TCP sessions before mutating `ReplayProtection`.
- **SafeTensors `[W | b]` / `[W1 | b1 | W2 | b2]` Layout Mapping & Live `Burn` + `Candle` Bridges (`ST-08`, `13D-4`):**
  - Added `ModelTensorSchemaV1::for_mlp`, `ModelTensorFlatRangeV1`, `ModelFlatTensorLayoutV1` (`canonical`, `for_linear`, `for_mlp`, disjoint/gap-free `try_new`), `export_safetensors_checkpoint_with_layout_v1`, and `import_safetensors_checkpoint_with_layout_v1` in `swarm-torch-models::safetensors_v1` (plus `put_safetensors_checkpoint_with_layout_v1` / `get_verified_safetensors_checkpoint_with_layout_v1` in `swarm-torch::artifacts::cas`, updated `examples/hello_swarm.rs`, and `benches/swarm_bench.rs`) so declaration-ordered flat parameters never swap weights and biases under alphabetical SafeTensors sorting while preserving `model_schema_hash`.
  - Added optional `real-burn` (`BurnLinearBridgeV1` over `burn 0.21.0` `Autodiff<NdArray<f32>>` with explicit `[in, out] <-> [out, in]` transpose and leaf `.require_grad()`) and `real-candle` (`CandleLinearBridgeV1` over `candle-core`/`candle-nn` `0.10.2` with atomic multi-tensor `VarMap` updates) in `adapters/swarm-torch-backends`, proving bit-exact SafeTensors roundtrips and forward/MSE-gradient parity (`< 1e-5`) across `SwarmTorch LinearModel <-> Burn <-> Candle`.

### Security, Async Runtime, & Aggregation (`Phase 13C` — Evented I/O, Embassy Runtime, Bulyan Stage 2, & Two-Phase `SharedSeedSparse`)
- **Evented Transport Waiters, Fallible `SwarmRuntime::try_spawn`, & Real `EmbassyRuntime` (`ST-06`, `ST-07`, `13C-1`):**
  - Replaced polling `std::thread::sleep` loops across `swarm-torch-net` (`MockNetworkHub`, `TcpTransport`, `UdpGossipTransport`) with bounded deduplicated `WaiterQueue` waker notifications (`waiters.rs`) and evented Tokio OS socket I/O, verified by concurrent receive-before-send integration tests.
  - Replaced fire-and-forget task spawning with primary fallible `SwarmRuntime::try_spawn -> Result<Self::TaskHandle, SpawnError>` across `TokioRuntime`, `MockRuntime`, and `EmbassyRuntime`, wired `EmbassyRuntime` (`SwarmClock`) to `embassy_time::{Instant, Timer}` plus `EmbassyRuntime::try_spawn_task` (`embassy_executor::SendSpawner` with `embassy-std-test` integration), and added embedded ARM CI lanes (`thumbv6m-none-eabi`, `thumbv7em-none-eabihf`) in `.github/workflows/rust.yml`.
- **Q9 Bulyan Stage 2 Canonical Peer Tie-Breaking (`ST-11`, `13C-2`):**
  - Updated `swarm-torch-core::aggregation::bulyan_stage2_core` (`Bulyan::aggregate` passing `PeerId::new(updates[idx].sender)`, plus `bulyan_stage2_with_peers` and `bulyan_stage2_nearest_to_median`) to select the lower observed median `sorted[(theta - 1) / 2]` per coordinate and rank $\beta = \theta - 2f$ nearest survivors by `(|val - M|, PeerId)`, documenting even-$\theta$ non-equivariance (`Q9`).
- **End-to-End Two-Phase Collective `SharedSeedSparse` Protocol (`ST-12`, `13C-3`):**
  - Added `SparseRoundContextV1`, `SharedSeedCommitmentMessageV1`, `SharedSeedOpeningMessageV1`, `compute_commitments_root_v1`, `derive_context_bound_round_seed_v1`, `SharedSeedCoordinateVerifier`, and honest `SharedSeedBandwidthReportV1` (tracking projected values, Merkle opening proof overhead, and Phase-1/Phase-2 wire framing) in `swarm-torch-core::compression`.
  - Extended `PayloadKindV1` with `SharedSeedCommitmentV1 = 4` and `SharedSeedOpeningV1 = 5` (`swarm-torch-net::gossip`), implemented `SparseBeaconPolicyV1::{PrecommittedHashLock, Ed25519TranscriptBeacon}` + `ExpectedRoundContextV1::{validate_sparse_commitment_submission, validate_sparse_opening_submission}` + `SwarmRoundCoordinator::{begin_sparse_round_v1, commit_sparse_commitment_submission, submit_sparse_commitment_delivery, finalize_sparse_commitments_with_beacon, commit_sparse_opening_submission, submit_sparse_opening_delivery, finalize_sparse_round}` with active-sparse mutual exclusion checked before replay commitment (`ST-01` prepare -> validate -> commit), and updated `examples/byzantine_federated_round.rs`.

### Security & Reliability (`Phase 13B` — Transactional Learning, Durable Storage, & `PhysicalStrict` Execution)
- **Transactional Coordinator Commits & Resumable Training State (`ST-04`, `13B-1`):**
  - Added signed head promotion via `checkpoints/committed_head.json` (`CommittedHeadV1`, verified by `RunArtifactBundle::verify_committed_head`), staging `CoordinatorResumeStateV2`, `OptimizerStateSnapshotV1`, `CenteredClippingCenterBlobV1`, and `HistoricalAnchorsBlobV1` in write-once CAS prior to head swap so mid-commit failures roll back without advancing model weights, round IDs, or replay/epoch state.
  - Added `SwarmRoundCoordinator::{resume_exact_training_state, restore_weights_only, restore_with_mode}` (`RestoreModeV1`), persisting optimizer momentum/velocity, Centered Clipping centers, FedAsync historical anchors, and a monotonic replay timestamp floor (`with_timestamp_floor`) across restarts.
- **Persistent `fs4` Kernel Bundle Locks, Write-Once CAS Verification, & Durability Policies (`ST-05`, `13B-2`):**
  - Replaced `create_new`/unlink `.bundle.lock` handling in `swarm-torch::artifacts::lock` with persistent inode-safe `fs4` kernel locks (`flock` on Unix, `LockFileEx` on Windows, zero unlink races and automatic OS release on crash) and added supply-chain exemption `EX-2026-11-14-01` for `fs4`'s transitive graph.
  - Added `DurabilityPolicyV1::{SyncEachAppend, SyncAtCommit}` defaulting `dp/ledger.ndjson` and DataOps state streams to `SyncEachAppend` (`write + flush + sync_all` + POSIX `sync_parent_dir`), verified write-once CAS SHA-256 digests in `VerifiedBundleV1` / `verify_committed_head`, and added `RunArtifactSink::write_graph_unlocked` to eliminate self-lock re-entrancy.
- **`PhysicalStrict` Execution, Fail-Closed Cache Integrity, & Numerical Atomicity (`ST-10`, `ST-14`, `13B-3`):**
  - Defaulted `NativeOpRunner::new()` to `RunnerExecutionModeV1::PhysicalStrict` (`with_mode` opt-in for `MetadataOnlyForTests`), removing `{` byte sniffing in favor of explicit `CONTENT_TYPE_COLUMNAR_JSON_V1` / `CONTENT_TYPE_NDJSON` dispatch (`ColumnarInputFormatV1`) and failing closed on legacy `run_with_context` or non-CAS/missing inputs.
  - Propagated `io::Error` from `DataOpsSession::verify_cache_hit` on missing/corrupt CAS blobs (turning cache corruption into explicit node failures `cache_integrity_error:` in `scheduler.rs` with zero silent re-execution), propagated `output_content_types` across DAG edges, and preserved cached `rows`, `bytes`, `schema`, and `source` metrics across session reopen cache hits.
  - Fixed `exp_f64_soft` overflow at large positive binary exponents (`k >= 1024`) in `swarm-torch-core::num`, and refactored `MseLoss`, `CrossEntropyLoss`, `HuberLoss`, and `FedProxRegularizer` (`swarm-torch-models::loss`) to two-pass atomic gradient validation before mutating caller buffers.

### Security (`Phase 13A` — Auth Admission, Replay V2, & Bounded Reassembly)
- **3-Stage Authenticated Coordinator Admission (`ST-01`):**
  - Replaced unauthenticated `PeerRoundSubmissionV1` ingress (`#[deprecated]`, failing closed with `RoundError::UnverifiedSubmissionRejected` and zero state mutation) with a strict prepare -> validate -> commit pipeline in `swarm-torch::coordinator`: **Stage A** `prepare_verified_delivery` / `prepare_verified_delivery_struct` (`&self`, read-only Ed25519 + v2 replay preflight + `InnerUpdateV2` schema decode into `PreparedSubmissionV1`), **Stage B** `ExpectedRoundContextV1::validate_{sync,fedasync,lora}_submission` (read-only binding of `swarm_instance_id`, `context_policy_digest`, `run_id`, `round_id`, `base_checkpoint_digest`, update kind/dimension, allowlist, and active cohort epoch), and **Stage C** atomic replay commit & admission returning sealed private-field proof tokens (`VerifiedRoundSubmissionV1`, `VerifiedFedAsyncSubmissionV1`, `VerifiedLoraSubmissionV1`).
  - Added public canonical update payload encoders `encode_{dense,compressed,lora}_update_payload_v2` matching compression schema versions `0`, `1`, and `2`.
- **Receiver-Nonce `session_id` V2 Replay Recovery & Default Fail-Closed Restart (`ST-03`):**
  - Defaulted `ReplayProtection` startup/recovery policy to `ReplayStartupPolicyV1::RequireSessionHandshake` (`ReplayProtection::try_with_config`) and wired coordinator resume paths to enforce `RequireSessionHandshake` plus a monotonic timestamp floor (`with_timestamp_floor`, `committed_unix_nanos / 1e9`).
  - Added `SessionHandshakeChallengeV2` with 256-bit `receiver_nonce` and deterministic domain-separated `session_id`, `MessageEnvelope::VERSION_V2` (`(0, 3)`), `InnerUpdateV2`, `RecipientWrapperV2`, `BundledGossipDeliveryV2`, `BTreeMap` v2 replay tracking keyed by `(SwarmInstanceId, RunId, PeerId, SessionEpoch, SessionId)`, atomic `complete_recovery_handshake_v2` (capacity validated prior to mutating pending challenges), and read-only `check_v2_admission_readonly` + `commit_prepared_admission`.
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
  - Stateful `CenteredClipping` (`CClip`) with explicit reference-iterate contract (`aggregate_with_center` / `aggregate_with_center_bounded`), $\tau$-scaled empty-round stability, and `f64` norm/update accumulation.
  - Deterministic `BucketingConfig` / `aggregate_with_bucketing_v1` using `Sha256CounterDrbgV1` Fisher-Yates shuffling (`domain: "swarmtorch.sbucketing.perm.v1"`) with `f64` bucket averaging.
- **Verifiable `SharedSeedSparse` Co-Design (`W11-02`, `ADR-0021`):**
  - `SharedSeedSparsePayloadV1` / `SharedSeedResidualCompressor` (`Sha256CounterDrbgV1` + Robert Floyd $\mathcal{O}(k)$ index selection `sample_shared_indices_v1` under `"swarmtorch.shared_sparse.seed.v1"`), eliminating per-index wire overhead (`4k` bytes vs `8k` bytes).
  - `SharedSeedCoordinateVerifier` (`UpdateTransform`) verifying canonical strictly-ascending PRNG coordinate indices prior to Byzantine distance evaluation.
- **Low-MTU Chunked Framing & Concrete Transports (`W11-03`, `ADR-0004`, `ADR-0022`):**
  - 16-byte `ChunkHeaderV1` (`CRC-16/IBM-3740` CCITT-FALSE) and `ChunkedFrameCodec` reassembly pool (`MAX_CONCURRENT_REASSEMBLIES = 32`, `MAX_CHUNKS_PER_MESSAGE = 256`, bitmap deduplication, conflict eviction, and lazy TTL sweep) for LoRa/BLE/UDP links.
  - Feature-gated `TcpTransport` (`tcp-transport`) with 4-byte length prefixing and pre-allocation frame-size gating (`DEFAULT_MAX_ENVELOPE_BYTES` / per-transport cap).
  - Feature-gated `UdpGossipTransport` / `StdUdpPacketTransport` (`udp-transport`) multiplexing single-datagram and `ChunkedFrameCodec`-fragmented frames.
- **SafeTensors v1 Checkpoint Persistence (`W11-04`, `ADR-0023`, `ADR-0024`):**
  - Zero-dependency `SwarmTorchCheckpointV1` reader/writer in `swarm-torch-models::safetensors_v1` with `SafeTensorsLimitsV1::default().max_header_bytes = 1_048_576`, strict `F32` contiguous interval validation, 8-byte alignment padding, and `SwarmModel` round-trip helpers.
  - CAS checkpoint export/import helpers (`put_safetensors_checkpoint_v1`, `get_verified_safetensors_checkpoint_v1`).
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
  - Pre-aggregation finiteness/shape validation (`MAX_COHORT_SIZE = 1024`), `f64` accumulation across `FedAvg`, `CoordinateMedian`, `TrimmedMean`, one-pass `Multi-Krum`, and `Bulyan` (`BulyanProfile::IterativeKrumV1`).
  - `ClassicResidualCompressor` and `Ef21SenderState` + `Ef21ReceiverState` error-feedback codecs.
- **Safe `BTreeMap` Replay Protection & Envelope v1 (`W9-02`, `ADR-0020`, `ADR-0021`):**
  - Replaced `lru` with a 100% safe dual-tree `BTreeMap` replay store (`entries_by_peer` + `expiry_index`), added `CohortAllowlist`, `RecoveryRequired` startup challenge (`SessionHandshakeChallenge`), and 6-stage `MessageEnvelope::verify_and_decode_payload` (`MessageEnvelope::VERSION_V1` `(0, 2)`, `DEFAULT_MAX_ENVELOPE_BYTES = 1_048_576`).
- **Models, Losses, & Dependency Diet (`W9-04`..`W9-06`, `ADR-0002`, `ADR-0022`, `ADR-0024`):**
  - Added `LinearModelWithCapacity`, contiguous `MlpModelWithCapacity`, analytical `MseLoss`, `CrossEntropyLoss` (`no_std` IEEE-754 `exp_f64`/`ln_f64`), `HuberLoss`, `FedProxRegularizer`, and `SgdOptimizer`.
  - Removed unused `burn`, `burn-ndarray`, `bincode`, `async-trait`, and `lru` dependencies from the core workspace.

## [0.1.0-alpha.1 .. 0.1.0-alpha.8] - 2026-03 .. 2026-04 (Waves 0–8)

### Added
- Initial 5-crate workspace scaffold (`swarm-torch-core`, `swarm-torch-models`, `swarm-torch-net`, `swarm-torch-runtime`, `swarm-torch`), `RunArtifactBundle` v1 observability and lineage pipeline, sequential `RunGraph` scheduler MVP, Ed25519 message signing baseline, Windows/Unix artifact path-confinement checks, and supply-chain CI baseline (`cargo deny`, `cargo audit`, `cargo vet`).
