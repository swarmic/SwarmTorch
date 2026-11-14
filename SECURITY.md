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

## Current Security Posture (Wave 13 Phases `13-0`, `13A`, & `13B` Hardening)

SwarmTorch is pre-1.0 and enforces fail-closed security contracts across its implemented layers:

- **Domain-Separated Message Authentication & V2 Session Binding (`ADR-0008B`, `ADR-0020`, `ADR-0021`, `W9-02`, `13A-2`):**
  - `MessageEnvelope` supports `ProtocolVersion::V1_0` and `ProtocolVersion::V0_3` (`(0, 3)`) with `InnerUpdateV2` / `RecipientWrapperV2` / `BundledGossipDeliveryV2` binding `swarm_id`, `run_id`, `round_id`, `session_id` (derived from 256-bit receiver nonce), compression schema version, sender Ed25519 verifying key (`[u8; 32]`), sequence, timestamp, and `SHA-256(payload)` under domain tag `"swarmtorch.envelope.v1"` (`KeyPair::sign_with_domain` / `verify_with_domain`).
- **3-Stage Coordinator Admission, Non-Forged Transport Sources, & Bounded Reassembly (`ADR-0022`, `ADR-0023`, `13A-1`, `13A-3`):**
  - `SwarmRoundCoordinator` fails closed on legacy unverified `PeerRoundSubmissionV1` ingress without state mutation and enforces 3-stage authentication (`PreparedSubmissionV1` -> `ExpectedRoundContextV1` -> sealed `VerifiedRoundSubmissionV1` / `VerifiedFedAsyncSubmissionV1` / `VerifiedLoraSubmissionV1`) across synchronous, FedAsync, and LoRA paths.
  - `SwarmTransport::recv` returns `ObservedTransportMessage { observed_source: ObservedSourceRef, payload }` (no sentinel `PeerId([0xFF; 32])` attribution), and `carrier::VerifiedLinkPrincipalV1` seals link-layer peer principals behind private fields.
  - `ObservedSourceReassembler<E>` (`CRC-16/IBM-3740`) enforces pre-allocation bounds across payload, dimensioned `Vec<u64>` bitmaps, and per-transfer/source/tombstone overhead (`max_total_bytes`), contiguous `Vec<u8>` buffer storage, non-evicting quarantine tombstones, and $O(\log S)$ source quarantine escalation.
- **Safe `BTreeMap` Replay Protection, Default `RequireSessionHandshake`, & `TimestampFloor` (`ADR-0020`, `W9-02`, `13A-2`):**
  - Replay state uses a 100% safe, deterministic dual-tree `BTreeMap` store (`entries_by_peer` + `expiry_index` plus v2 session-scoped tables) with explicit `CohortAllowlist` admission and default `ReplayStartupPolicyV1::RequireSessionHandshake` gating cleared atomically via receiver-issued `SessionHandshakeChallengeV2` (`complete_recovery_handshake_v2` validates capacity before clearing pending challenges), plus monotonic `TimestampFloor` enforcement on coordinator resume.
- **Byzantine Aggregation, Non-IID Bucketing, & Verifiable Sparse Co-Design (`ADR-0019`, `ADR-0020`, `ADR-0021`, `W9-01`, `W9-03`, `W11-01`, `W11-02`):**
  - All aggregators (`FedAvg`, `CoordinateMedian`, `TrimmedMean`, one-pass `Multi-Krum`, `Bulyan` `BulyanProfile::IterativeKrumV1`, and stateful `CenteredClippingAggregator`) validate every coordinate of every update for `.is_finite()`, enforce `MAX_COHORT_SIZE = 1024`, sort via `total_cmp`, and widen sums/midpoints/distances to `f64` before narrowing to finite `f32`.
  - `SBucketingTransformer` deterministically shuffles client updates via `ChaCha8Rng` (`"swarmtorch.sbucket.v1"`) before robust aggregation, and `SharedSeedCoordinateVerifier` rejects out-of-policy sparse coordinate indices (`"swarmtorch.shared_seed_sparse.v1"`) before Byzantine distance calculation.
  - Compression codecs (`compression.rs`) validate hyperparameters and finiteness prior to buffer mutation, and `dataops.rs` enforces `MAX_RAW_SOURCE_URI_LEN = 4096` before URI redaction allocation.
- **Signed Artifact Manifests, Persistent Kernel Locking, `PhysicalStrict` DataOps, & SafeTensors Bounds (`ADR-0016`, `ADR-0023`, `ADR-0025`, `W10-02`, `W11-04`, `13B-1..3`):**
  - `RunArtifactBundle` enforces persistent inode-safe `fs4` kernel locking on `.bundle.lock` (`BundleLockGuard`), unique `.staging.*` atomic writes with `sync_all()` (+ POSIX parent-dir `sync_parent_dir`), `DurabilityPolicyV1::SyncEachAppend` on DP ledgers and DataOps state streams, Ed25519 `ManifestSignatureV1` over raw 16-byte `RunId` preimages, single-pass zero-TOCTOU `VerifiedBundleV1` (`TrustPolicy::Strict`) with write-once CAS SHA-256 verification, and signed `CommittedHeadV1` (`CoordinatorResumeStateV2` + `OptimizerStateSnapshotV1` + `CenteredClippingCenterBlobV1` + `HistoricalAnchorsBlobV1`) rollback safety across `SwarmRoundCoordinator` round commits.
  - `NativeOpRunner` defaults to `RunnerExecutionModeV1::PhysicalStrict` with explicit MIME dispatch (`ColumnarInputFormatV1`, zero `{` sniffing) and fails closed on missing/corrupt CAS cache blobs (`verify_cache_hit` -> `cache_integrity_error:`). `SwarmTorchCheckpointV1` enforces `HEADER_SIZE_LIMIT = 65_536` and strict contiguous `F32` byte-interval validation before parameter loading.

### Known Current Boundaries & Active Wave 13 Remediation (`v0.2.0` -> Wave 13)

- **Windows / Non-Unix Directory Metadata Sync Boundary (`ST-05`):** On Unix (`#[cfg(unix)]`), `sync_parent_dir` opens parent directories and invokes `sync_all()` after `SyncEachAppend` appends, CAS hardlinks, and staging renames; on non-Unix platforms (including Windows), `sync_parent_dir` is a documented no-op because regular-file `File::sync_all()` invokes `FlushFileBuffers`, whereas Windows directory handles do not guarantee uniform directory-entry flush semantics across all drivers and filesystems.

- **Experimental Differential Privacy Accounting (`ST-02`, Phase `13-0`):** `RdpAccountantV1` (`swarm-torch-core::dp`), `dp/ledger.ndjson` (`dp_ledger.rs`), and `trace_aggregation_budget` (`mcp_inspector.rs`) currently perform **experimental accounting-only tracking** (`guarantees_differential_privacy = false`). `SwarmRoundCoordinator::supports_certified_differential_privacy()` returns `false`, and any `VerifiedMechanismProtected` profile fails closed until a certified `DpFedAvgV1` mechanism profile ships post-Wave-13.
- **Untrusted Wasm Extension Containment (`ST-09`, Phase `13-0` -> `13D-1`):** `RunnerIsolationProfile::WasmSandboxedV1`, `WasmSandboxRunnerV1::{execute_byte_transform, execute_scalar_i32}`, and `NativeOpRunner` `"wasm_transform"` fail closed (`UnsupportedUntrustedWasmSandbox`) pending isolated `wasmi` integration (`adapters/swarm-torch-wasm`, Phase `13D-1`).
- **Optional Transport & Framework Adapter Crates (`ST-08`, Phase `13-0` -> `13D-2..4`):** Standalone crates under `adapters/` (`swarm-torch-transport-iroh`, `swarm-torch-transport-zenoh`, `swarm-torch-backends`, `swarm-torch-mcp`) are isolated outside root `Cargo.lock` and currently sit at **`Simulation verified`** (contract/policy/framing tested with bounded sliding-window rate limits and pre-replay binding checks; live external SDK wiring lands in Phase `13D`).
- **Placeholder Radio Feature Flags:** `ble-transport` and `lora-transport` in `swarm-torch` / `swarm-torch-net` are empty feature-flag surface reservations with no hardware radio drivers implemented yet.

---

## Supply-Chain Security (`P2-05`, `P2-11`)

### Dependency Auditing & Wave 9–11 Enforcement

CI enforces supply-chain checks on every push and pull request:

| Tool | Purpose | Config | Mode |
|------|---------|--------|------|
| `cargo deny` (`0.19.0`) | License, ban, and advisory enforcement | [`deny.toml`](deny.toml) | **Blocking** |
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
