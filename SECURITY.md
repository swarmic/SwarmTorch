# Security Policy

## Vulnerability Disclosure

**Preferred:** [GitHub Private Vulnerability Reporting](https://github.com/swarmic/SwarmTorch/security/advisories/new)  
**Direct Contact:** `swarmiclabs@gmail.com`  
**Response Time:** Initial acknowledgment within 48 hours  
**Disclosure Policy:** 90-day coordinated disclosure

### Reporting Process

1. **Do NOT open public issues or pull requests** for security vulnerabilities.
2. Submit a private report via [GitHub Security Advisories](https://github.com/swarmic/SwarmTorch/security/advisories/new) (preferred), or email **`swarmiclabs@gmail.com`** with:
   - Description of the vulnerability and affected component/crate
   - Steps or minimal test case to reproduce
   - Affected versions / commit SHA and feature profile
   - Suggested remediation (if any)

### Scope

- `swarm-torch-*` crates (`swarm-torch-core`, `swarm-torch-models`, `swarm-torch-net`, `swarm-torch-runtime`, `swarm-torch`) and their locked dependency graph
- Cryptographic authentication, replay protection, cohort admission, and Byzantine aggregation logic
- Artifact manifest verification and path-confinement logic
- Build/CI configuration (`.github/workflows/`, `deny.toml`, `supply-chain/`)

---

## Current Security Posture (Wave 13–16 Hardening)

SwarmTorch is pre-1.0 and enforces fail-closed security contracts across its implemented layers:

- **Domain-Separated Message Authentication & V2 Session Binding (`ADR-0008B`, `ADR-0020`, `ADR-0021`, `W9-02`, `13A-2`, `15C-3`):**
  - `MessageEnvelope` defaults to `MessageEnvelope::CURRENT_VERSION = MessageEnvelope::VERSION_V2` (`(0, 3)`) signed under domain tag `"swarmtorch.envelope.v2"` (with explicit gated compatibility for `VERSION_V1` `(0, 2)` under `"swarmtorch.envelope.v1"` and `VERSION_V0` `(0, 1)`), alongside `InnerUpdateV2` / `RecipientWrapperV2` / `BundledGossipDeliveryV2` binding `swarm_id`, `run_id`, `round_id`, `session_id` (derived from 256-bit receiver nonce), compression schema version, sender Ed25519 verifying key (`[u8; 32]`), sequence, timestamp, and `SHA-256(payload)` (`MessageAuth::{sign_v1, sign_v2, verify, sign_prehashed_domain, verify_prehashed_domain}`).
- **3-Stage Coordinator Admission, Non-Forged Transport Sources, & Bounded Reassembly (`ADR-0022`, `ADR-0023`, `13A-1`, `13A-3`):**
  - `SwarmRoundCoordinator` fails closed on legacy unverified `PeerRoundSubmissionV1` ingress without state mutation and enforces 3-stage authentication (`PreparedSubmissionV1` -> `ExpectedRoundContextV1` -> sealed `VerifiedRoundSubmissionV1` / `VerifiedFedAsyncSubmissionV1` / `VerifiedLoraSubmissionV1`) across synchronous, FedAsync, and LoRA paths.
  - `SwarmTransport::recv` returns `ObservedTransportMessage { observed_source: ObservedSourceRef, payload }` (no sentinel `PeerId([0xFF; 32])` attribution), and `carrier::VerifiedLinkPrincipalV1` seals link-layer peer principals behind private fields.
  - `ObservedSourceReassembler<E>` (`CRC-16/IBM-3740`) enforces pre-allocation bounds across payload, dimensioned `Vec<u64>` bitmaps, and per-transfer/source/tombstone overhead (`max_total_bytes`), contiguous `Vec<u8>` buffer storage, non-evicting quarantine tombstones, and $O(\log S)$ source quarantine escalation.
- **Safe `BTreeMap` Replay Protection, Default `RequireSessionHandshake`, & Monotonic Timestamp Floor (`ADR-0020`, `W9-02`, `13A-2`):**
  - Replay state uses a 100% safe, deterministic dual-tree `BTreeMap` store (`entries_by_peer` + `expiry_index` plus v2 session-scoped tables) with explicit `CohortAllowlist` admission and default `ReplayStartupPolicyV1::RequireSessionHandshake` gating cleared atomically via receiver-issued `SessionHandshakeChallengeV2` (`complete_recovery_handshake_v2` validates capacity before clearing pending challenges), plus monotonic timestamp floor enforcement (`with_timestamp_floor`) on coordinator resume.
- **Evented Async Transport & Runtime Starvation Hardening (`ADR-0022`, `ST-06`, `ST-07`, `13C-1`):**
  - `swarm-torch-net` (`WaiterQueue`, `MockTransport`, `TcpTransport`, `UdpGossipTransport`) and `swarm-torch-runtime` (`TokioRuntime`, `EmbassyRuntime`, `MockRuntime`) eliminate busy-yield polling loops in favor of waker-driven readiness notifications (`poll_recv_os_async` / `recv_os_async`, `tokio::time::sleep`, `embassy_time::Timer::after`, `EmbassyRuntime::try_spawn_task`), preventing single-task executor starvation.
- **Byzantine Aggregation, Non-IID Bucketing, & Collective Verifiable Sparse Co-Design (`ADR-0019`, `ADR-0020`, `ADR-0021`, `W9-01`, `W9-03`, `W11-01`, `W11-02`, `13C-2`, `13C-3`):**
  - All aggregators (`FedAvg`, `CoordinateMedian`, `TrimmedMean`, one-pass `Multi-Krum`, `Bulyan` `BulyanProfile::IterativeKrumV1`, and stateful `CenteredClipping`) validate every coordinate of every update for `.is_finite()`, enforce `MAX_COHORT_SIZE = 1024`, sort via `total_cmp`, and widen sums/midpoints/distances to `f64` before narrowing to finite `f32`. In `Bulyan` Stage 2 (`bulyan_stage2_with_peers`, `bulyan_stage2_nearest_to_median`), median selection chooses the lower observed coordinate (`sorted[(theta - 1) / 2]`, tie-broken deterministically by `PeerId`) rather than arithmetic means of synthetic quantiles (`ST-11` closed).
  - `BucketingConfig` / `aggregate_with_bucketing_v1` deterministically shuffles client updates via `Sha256CounterDrbgV1` (`"swarmtorch.sbucketing.perm.v1"`) before robust aggregation.
  - Collective `SharedSeedSparse` (`ST-12` closed) enforces two-phase commitment-then-beacon execution (`commit_sparse_commitment_submission`, `submit_sparse_commitment_delivery`, `finalize_sparse_commitments_with_beacon`), binding `commitments_root` into `derive_context_bound_round_seed_v1` prior to mask reveal and rejecting concurrent standard/FedAsync submissions in Stage B/C before replay-state commit (`SharedSeedCoordinateVerifier` validates unmasked coordinates prior to Byzantine distance calculation).
  - Compression codecs (`compression.rs`) validate hyperparameters and finiteness prior to buffer mutation, and `dataops.rs` enforces `MAX_RAW_SOURCE_URI_LEN = 4096` before URI redaction allocation.
- **Signed Artifact Manifests, Persistent Kernel Locking, `PhysicalStrict` DataOps, & SafeTensors Bounds (`ADR-0016`, `ADR-0023`, `ADR-0025`, `W10-02`, `W11-04`, `13B-1..3`, `15A`, `15B`):**
  - `RunArtifactBundle` enforces persistent inode-safe `fs4` kernel locking on `.bundle.lock` (`BundleLockGuard`), unique `.staging.*` atomic writes with `sync_all()` (+ POSIX parent-dir `sync_parent_dir`), `DurabilityPolicyV1::SyncEachAppend` on DP ledgers and DataOps state streams, Ed25519 `ManifestSignatureV1` over raw 16-byte `RunId` preimages, single-pass zero-TOCTOU `VerifiedBundleV1` (`TrustPolicy::Strict`) with write-once CAS SHA-256 verification, and signed `CommittedHeadV1` (`CoordinatorResumeStateV2` + `OptimizerStateSnapshotV1` + `CenteredClippingCenterBlobV1` + `HistoricalAnchorsBlobV1`) rollback safety across `SwarmRoundCoordinator` round commits.
  - `NativeOpRunner` defaults to `RunnerExecutionModeV1::PhysicalStrict` with explicit MIME dispatch (`ColumnarInputFormatV1`, zero `{` sniffing) and fails closed on missing/corrupt CAS cache blobs (`verify_cache_hit` -> `cache_integrity_error:`). `SwarmTorchCheckpointV1` enforces `SafeTensorsLimitsV1::default().max_header_bytes = 1_048_576` (1 MiB, configurable via custom `SafeTensorsLimitsV1`) and strict contiguous `F32` byte-interval validation before parameter loading.

### Known Current Boundaries & Active Remediation (`v0.2.0` -> Wave 16)

- **Windows / Non-Unix Directory Metadata Sync Boundary (`ST-05`):** On Unix (`#[cfg(unix)]`), `sync_parent_dir` opens parent directories and invokes `sync_all()` after `SyncEachAppend` appends, CAS hardlinks, and staging renames; on non-Unix platforms (including Windows), `sync_parent_dir` is a documented no-op because regular-file `File::sync_all()` invokes `FlushFileBuffers`, whereas Windows directory handles do not guarantee uniform directory-entry flush semantics across all drivers and filesystems.

- **Certified vs. Experimental Differential Privacy (`ST-02`, `ADR-0023` §4.2, Wave `16B`):**
  - `RdpAccountantV1`, `dp/ledger.ndjson` (`dp_ledger.rs`), `SwarmRoundCoordinator`, and `McpInspectorServerV1` (`trace_aggregation_budget` tool) strictly segregate **`ExperimentalHypotheticalBudgetOnly`** (`guarantees_differential_privacy = false`, accounting-only) from **`VerifiedMechanismProtected`** (`guarantees_differential_privacy = true`, **`Adversarial conformance passed`**).
  - Certified `DpFedAvgV1` uses exact integer-lattice $\ell_2$ clipping (`B = 65_536`), associative `i128` summation, exact floating-point-free Canonne, Kamath, & Steinke (2020, `arXiv:2004.00010`, Prop. 5 & Algorithms 1–3) Discrete Gaussian noise seeded from OS CSPRNG (`getrandom 0.2.17`, first-party audited), sealed `SampledDpCohortV1` + `DpNoiseEntropyV1` + `VerifiedDpMechanismRoundV1` proofs, one-shot anti-resampling Poisson cohort draws, zero-conditioning release on empty/below-quorum/dropout rounds, non-refundable pre-release ledger debits, and external bundle telemetry redaction (`output_rows = None`, omitting `swarmtorch.admitted_peers`).
  - **Explicit Threat & Scope Boundary:** Operates under the **trusted central coordinator model** (protecting published checkpoints, `dp/ledger.ndjson`, `spans.ndjson`, `events.ndjson`, `datasets/registry.json`, and `McpInspectorServerV1` outputs against external observers). Supported strictly for synchronous `FedAvg` + `EqualPeerVote` + `ClientLevel` under `AddRemoveOne` ($q \in (0, 1]$) or `ReplaceOne` strictly at full participation $q = 1.0$; `ReplaceOne` with $q < 1.0$, `ExampleLevel`, `FedAsync`, `LoRA`, `SharedSeedSparse`, `$s$-Bucketing`, `RegistryCertified`, and non-linear aggregators remain fail-closed.
- **Wasm Extension Containment & Isolated `wasmi` Sandbox (`ST-09`, `13D-1`, `15D-1`, `16C`):** Default `NativeOpRunner::new()`, `WasmSandboxRunnerV1`, and `Scheduler` fail closed (`UnsupportedUntrustedWasmSandbox`) unless configured with a `WasmExtensionExecutor`; `adapters/swarm-torch-wasm` (`WasmiExtensionExecutorV1`, `wasmi 2.0.0`, MSRV `1.86`, **`Adversarial conformance passed`**) supports both legacy `WasmGuestAbiV1::RawMem0V0` and structured `WasmGuestAbiV1::SwarmTorchAllocV1` (`swarmtorch_alloc`/`swarmtorch_dealloc`, pre-`(start)` export/memory validation, `max_output_bytes = 16 MiB` pre-copy cap, strict 4-case ownership/aliasing/single-free dealloc rules, fail-closed cleanup traps, and cumulative fuel + wall-clock timeout slicing via `TypedResumableCall`). The legacy bespoke subset interpreter is compiled strictly under `#[cfg(test)]`.
- **Optional Transport, Framework, Wasm, & MCP Adapter Crates (`ST-08`, `13D`, `15D`, `16C`, `16D`):** Standalone crates under `adapters/` remain isolated outside root `Cargo.lock` with committed per-adapter `Cargo.lock` files and MSRV-verified CI lanes so core `no_std`/MSRV guarantees are unaffected: `swarm-torch-wasm` (`wasmi 2.0.0`, MSRV `1.86`, `RawMem0V0` + `SwarmTorchAllocV1`, `Adversarial conformance passed`), `swarm-torch-transport-iroh` (`iroh 1.4.1`, MSRV `1.91`, strict 1-RTT QUIC + `remote_id` attestation), `swarm-torch-transport-zenoh` (`zenoh 1.10.2`, MSRV `1.88`, `time` patched for `RUSTSEC-2026-0009`), `swarm-torch-backends` (`burn 0.21.0` CPU + `real-burn-wgpu` GPU via `burn-wgpu 0.21.0`/`cubecl 0.10.0`/`wgpu 29.0.4`, plus `candle 0.10.2` CPU, MSRV `1.92`, `Integration verified`; `Cuda` remains `Contract defined`), and `swarm-torch-mcp` (official `rmcp 3.11` SDK over `stdio_pipes`, MSRV `1.88`, `Integration verified`).
- **Hardware-Agnostic `no_std` Radio Transports (`16E-1`, `16E-2`):** `lora-transport` (`swarm_torch_net::lora`, `LoraPacketTransportV1`, Semtech AN1200.13 integer-$\mu\text{s}$ ToA, 61-slot `prepare -> tx -> commit` duty-cycle limiter) and `ble-transport` (`swarm_torch_net::ble`, `BlePacketTransportV1`, Bluetooth Core Spec Vol. 3 Part F §3.2.8–§3.2.9 hard `512`B ATT value cap, pre-TX credit gating) sit at **`Simulation verified`** (pure `no_std`, Rust `1.75`, `thumbv6m`/`thumbv7em`/`riscv32imc`); physical board SPI/HCI drivers remain out-of-tree follow-on, and `wifi-transport` remains a reserved placeholder flag.

---

## Supply-Chain Security (`P2-05`, `P2-11`)

### Dependency Auditing & Wave 9–11 Enforcement

CI enforces supply-chain checks on every push and pull request:

| Tool | Purpose | Config | Mode |
|------|---------|--------|------|
| `cargo deny` (`0.19.0`) | License, ban, and advisory enforcement (workspace + standalone adapters) | [`deny.toml`](deny.toml), [`adapters/deny.toml`](adapters/deny.toml) | **Blocking** |
| `cargo audit` (`0.22.1`) | RustSec advisory database | Built-in | **Blocking** |
| `cargo vet` (`0.10.2`) | Human audit & exemption tracking | [`supply-chain/`](supply-chain/) | **Blocking** |

In Wave 9 (`W9-02`, `W9-05`), SwarmTorch removed `lru 0.12.5`, `burn 0.14`, `burn-ndarray 0.14`, `bincode 2.0.0-rc.3`, and `async-trait` from the workspace dependency graph, retiring `RUSTSEC-2025-0141`, `RUSTSEC-2026-0002`, and `RUSTSEC-2026-0253`, and closing exemptions `EX-2026-04-05-02` and `EX-2026-04-05-03` in [`supply-chain/exemption-rationale.md`](supply-chain/exemption-rationale.md).

Maintainers and contributors can run the supply-chain checks locally:

```bash
cargo deny check
cargo audit
cargo vet
```

### Exemption Governance

- Any PR that adds a new `[[exemptions.*]]` entry in [`supply-chain/config.toml`](supply-chain/config.toml) **must** include an explicit rationale entry, owner, and removal target in [`supply-chain/exemption-rationale.md`](supply-chain/exemption-rationale.md) (enforced automatically by `.github/workflows/rust.yml`).
- Project license is **MPL-2.0**; incompatible strong-copyleft dependencies (GPL/AGPL) are rejected by [`deny.toml`](deny.toml).

---

## CI/CD, Review Isolation, & Unsafe Code Policy

1. **Workflow Least Privilege & Action Pinning:**
   - `.github/workflows/rust.yml` runs under `permissions: contents: read`, uses `pull_request` (never `pull_request_target` on untrusted code), and pins all third-party GitHub Actions to full 40-character commit SHAs.
2. **External Contribution Isolation:**
   - External pull requests are reviewed and executed in isolated environments without maintainer credentials or release authority. Changes to `Cargo.toml`, `Cargo.lock`, `.cargo/config.toml`, `build.rs`, `.github/**`, `supply-chain/**`, `CODEOWNERS`, or agent instruction files require explicit maintainer review (`@swarmic`).
3. **Zero Unsafe Code (`#![forbid(unsafe_code)]`):**
   - All workspace crates enforce `#![forbid(unsafe_code)]`.
