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

## Current Security Posture (Wave 9 / `v0.1.0-alpha.9` Reality Check)

SwarmTorch is pre-1.0 and enforces fail-closed security contracts across its implemented layers:

- **Domain-Separated Message Authentication (`ADR-0008B`, `ADR-0021`, `W9-02`):**
  - `MessageEnvelope` (`ProtocolVersion::V1_0`) binds protocol version (`1.0`), `message_type`, sender Ed25519 public key (`[u8; 32]`), `sequence`, `timestamp`, `round_id`, `swarm_id`, `session_nonce`, and `SHA-256(payload)` under domain tag `"swarmtorch.envelope.v1"` (`KeyPair::sign_with_domain` / `verify_with_domain`).
- **Six-Stage Atomic Receive Pipeline (`ADR-0021`, `W9-02`):**
  - `MessageEnvelope::decode_and_verify_payload` enforces `MAX_FRAME_BYTES = 65_536` *before* deserialization, validates version/key/timestamp bounds, checks cohort allowlist and sequence window read-only before Ed25519 signature verification, commits replay state only after signature verification succeeds, and decodes the inner payload last. Unauthenticated frames cannot mutate replay or peer state.
- **Safe `BTreeMap` Replay Protection, `CohortAllowlist`, & Restart Recovery (`ADR-0020` / `ADR-0021`, `W9-02`):**
  - Replay state uses a 100% safe, deterministic dual-tree `BTreeMap` store (`entries_by_peer` + `expiry_index`) with bounded per-peer and total capacity (`max_tracked_peers`, `max_entries_per_peer`), eliminating `lru` from the dependency tree.
  - Supports pinned cohort admission (`CohortAllowlist`) and explicit startup/restart recovery gating (`RecoveryRequired`) cleared via a fresh receiver-issued challenge handshake (`SessionHandshakeChallenge`, domain `"swarmtorch.handshake.v1"`).
- **Byzantine Aggregation & Compression Validation (`ADR-0019`, `ADR-0020`, `W9-01`, `W9-03`):**
  - All aggregators (`FedAvg`, `CoordinateMedian`, `TrimmedMean`, one-pass `Multi-Krum`, and `Bulyan` `BulyanProfile::IterativeKrumV1`) validate every coordinate of every update for `.is_finite()`, enforce `MAX_COHORT_SIZE = 1024`, sort via `total_cmp`, and widen sums/midpoints/distances to `f64` before narrowing to finite `f32`.
  - Compression codecs (`compression.rs`) validate hyperparameters and finiteness prior to buffer mutation, and `dataops.rs` enforces `MAX_RAW_SOURCE_URI_LEN = 4096` before URI redaction allocation.
- **Artifact Bundle Path Confinement (`ADR-0016`, Wave 8.0/8.1):**
  - Manifest verification rejects relative path traversal (`..`), absolute paths, and symlink escapes on both Unix and Windows CI lanes.

### Known Current Boundaries & Planned Hardening (Waves 10–12)

- **Concrete Network Transports & Low-MTU Framing:** Current `swarm-torch-net` provides `PacketTransport`, `SwarmTransport`, `MockTransport`, and `MessageEnvelope` v1 verification. Low-MTU chunk reassembly (`ChunkedFrameCodec`, `W11-03`) and optional QUIC/Zenoh adapters (`W12-02`) are planned.
- **Signed Artifact Manifests & Cross-Process Bundle Locking:** Cryptographic manifest signatures (`ManifestSignatureV1`), `TrustPolicy::Strict` default report/checkpoint loading, and OS-backed exclusive bundle lockfiles (`P2-02`, `P2-10`) are scheduled for Wave 10 (`W10-02`).
- **Sandboxed Extension Host (`wasmi`):** Pipeline extension sandboxing with installed memory/table limiters and engine fuel (`ADR-0018`, `ADR-0025`) is scheduled for Wave 12 (`W12-04`).

---

## Supply-Chain Security (`P2-05`, `P2-11`)

### Dependency Auditing & Wave 9 Diet

CI enforces supply-chain checks on every push and pull request:

| Tool | Purpose | Config | Mode |
|------|---------|--------|------|
| `cargo deny` (`0.19.0`) | License, ban, and advisory enforcement | [`deny.toml`](deny.toml) | **Blocking** |
| `cargo audit` (`0.22.1`) | RustSec advisory database | Built-in | **Blocking** |
| `cargo vet` (`0.10.2`) | Human audit & exemption tracking | [`supply-chain/`](supply-chain/) | Advisory (promoted to blocking in `W10-02`) |

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
