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

## Current Security Posture (Wave 13–15 Hardening)

SwarmTorch is pre-1.0 and enforces fail-closed security contracts across its implemented layers:

- **Domain-Separated Message Authentication & V2 Session Binding (`ADR-0008B`, `ADR-0020`, `ADR-0021`, `W9-02`, `13A-2`, `15C-3`):**
  - `MessageEnvelope` defaults to `MessageEnvelope::CURRENT_VERSION = MessageEnvelope::VERSION_V2` (`(0, 3)`) with explicit gated compatibility for `VERSION_V0` (`(0, 1)`) and `VERSION_V1` (`(0, 2)`), alongside `InnerUpdateV2` / `RecipientWrapperV2` / `BundledGossipDeliveryV2` binding `swarm_id`, `run_id`, `round_id`, `session_id` (derived from 256-bit receiver nonce), compression schema version, sender Ed25519 verifying key (`[u8; 32]`), sequence, timestamp, and `SHA-256(payload)` under domain tag `"swarmtorch.envelope.v1"` (`MessageAuth::{sign_v1, sign_v2, verify, sign_prehashed_domain, verify_prehashed_domain}`).
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
  - `BucketingConfig` / `aggregate_with_bucketing_v1` deterministically shuffles client updates via `ChaCha8Rng` (`"swarmtorch.sbucket.v1"`) before robust aggregation.
  - Collective `SharedSeedSparse` (`ST-12` closed) enforces two-phase commitment-then-beacon execution (`commit_sparse_commitment_submission`, `submit_sparse_commitment_delivery`, `finalize_sparse_commitments_with_beacon`), binding `commitments_root` into `derive_context_bound_round_seed_v1` prior to mask reveal and rejecting concurrent standard/FedAsync submissions in Stage B/C before replay-state commit (`SharedSeedCoordinateVerifier` validates unmasked coordinates prior to Byzantine distance calculation).
  - Compression codecs (`compression.rs`) validate hyperparameters and finiteness prior to buffer mutation, and `dataops.rs` enforces `MAX_RAW_SOURCE_URI_LEN = 4096` before URI redaction allocation.
- **Signed Artifact Manifests, Persistent Kernel Locking, `PhysicalStrict` DataOps, & SafeTensors Bounds (`ADR-0016`, `ADR-0023`, `ADR-0025`, `W10-02`, `W11-04`, `13B-1..3`, `15A`, `15B`):**
  - `RunArtifactBundle` enforces persistent inode-safe `fs4` kernel locking on `.bundle.lock` (`BundleLockGuard`), unique `.staging.*` atomic writes with `sync_all()` (+ POSIX parent-dir `sync_parent_dir`), `DurabilityPolicyV1::SyncEachAppend` on DP ledgers and DataOps state streams, Ed25519 `ManifestSignatureV1` over raw 16-byte `RunId` preimages, single-pass zero-TOCTOU `VerifiedBundleV1` (`TrustPolicy::Strict`) with write-once CAS SHA-256 verification, and signed `CommittedHeadV1` (`CoordinatorResumeStateV2` + `OptimizerStateSnapshotV1` + `CenteredClippingCenterBlobV1` + `HistoricalAnchorsBlobV1`) rollback safety across `SwarmRoundCoordinator` round commits.
  - `NativeOpRunner` defaults to `RunnerExecutionModeV1::PhysicalStrict` with explicit MIME dispatch (`ColumnarInputFormatV1`, zero `{` sniffing) and fails closed on missing/corrupt CAS cache blobs (`verify_cache_hit` -> `cache_integrity_error:`). `SwarmTorchCheckpointV1` enforces `SafeTensorsLimitsV1::default().max_header_bytes = 1_048_576` (1 MiB, configurable via custom `SafeTensorsLimitsV1`) and strict contiguous `F32` byte-interval validation before parameter loading.

### Known Current Boundaries & Active Remediation (`v0.2.0` -> Wave 15)

- **Windows / Non-Unix Directory Metadata Sync Boundary (`ST-05`):** On Unix (`#[cfg(unix)]`), `sync_parent_dir` opens parent directories and invokes `sync_all()` after `SyncEachAppend` appends, CAS hardlinks, and staging renames; on non-Unix platforms (including Windows), `sync_parent_dir` is a documented no-op because regular-file `File::sync_all()` invokes `FlushFileBuffers`, whereas Windows directory handles do not guarantee uniform directory-entry flush semantics across all drivers and filesystems.

- **Experimental Differential Privacy Accounting (`ST-02`, Phase `13-0`, `15A`):** `RdpAccountantV1` (`swarm-torch-core::dp`), `dp/ledger.ndjson` (`dp_ledger.rs`), and `trace_aggregation_budget` (`mcp_inspector.rs`) currently perform **experimental accounting-only tracking** (`guarantees_differential_privacy = false`). `SwarmRoundCoordinator::supports_certified_differential_privacy()` returns `false`, and any `VerifiedMechanismProtected` profile fails closed until a certified `DpFedAvgV1` mechanism profile ships in a dedicated ratified ADR.
- **Wasm Extension Containment & Isolated `wasmi` Sandbox (`ST-09`, `13D-1`, `15D-1`):** Default `NativeOpRunner::new()`, `WasmSandboxRunnerV1`, and `Scheduler` fail closed (`UnsupportedUntrustedWasmSandbox`) unless configured with a `WasmExtensionExecutor`; `adapters/swarm-torch-wasm` (`WasmiExtensionExecutorV1`, `wasmi 2.0.0`, scoped to `raw_mem0_v0` per Wave 15 Decision `A2`) enforces zero imports, `EnforcedLimits::strict()`, `StoreLimits` (including un-exported internal tables/memories), bounded stack depth/size, `WasmStoreLimitsV1::max_start_fuel`, SHA-256 pinning, and fuel + wall-clock timeout slicing via `TypedResumableCall`. The legacy bespoke subset interpreter is compiled strictly under `#[cfg(test)]`.
- **Optional Transport, Framework, Wasm, & MCP Adapter Crates (`ST-08`, `13D`, `15D`):** Standalone crates under `adapters/` remain isolated outside root `Cargo.lock` with committed per-adapter `Cargo.lock` files and MSRV-verified CI lanes so core `no_std`/MSRV guarantees are unaffected: `swarm-torch-wasm` (`wasmi 2.0.0`, MSRV `1.86`, `raw_mem0_v0`), `swarm-torch-transport-iroh` (`iroh 1.4.1`, MSRV `1.91`, bounded timeouts, pre-body quotas, CSPRNG nonces, strict 1-RTT QUIC + `remote_id` attestation), `swarm-torch-transport-zenoh` (`zenoh 1.10.2`, MSRV `1.88`, `time` patched for `RUSTSEC-2026-0009`, isolated-peer rejection, bounded reads + strict topic/replay verification), `swarm-torch-backends` (CPU-only `burn 0.21.0` + `candle 0.10.2`, MSRV `1.92`, linear & 2-layer ReLU MLP SafeTensors parity), and `swarm-torch-mcp` (official `rmcp 3.11` SDK over `stdio_pipes`, MSRV `1.88`, 1 MiB line cap, bounded concurrency, strict Ed25519 verification) all sit at **`Integration verified`**.
- **Placeholder Radio Feature Flags:** `ble-transport` and `lora-transport` in `swarm-torch` / `swarm-torch-net` are empty feature-flag surface reservations with no hardware radio drivers implemented yet.

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
