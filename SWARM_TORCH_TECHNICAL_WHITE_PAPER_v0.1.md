# SwarmTorch Technical White Paper (ST-TWP)

**Version:** v0.1 (Draft)  
**Date:** 2026-03-11  
**Scope:** SwarmTorch-only (this is not a SwarmicOS/GridSwarm/Swarmflow spec)  

## Reader's Guide

### Normative Language

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this document are to be interpreted as described in RFC 2119 and RFC 8174.

Evidence: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#RFC-2119, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#RFC-8174

### Canon and Drift Controls

This document is intended to be a **canonical SwarmTorch reference**, co-equal with:

1. `ADRs.md` (architecture decisions)
2. The `swarm-torch-*` crate source code (implementation truth)

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`ADRs.md`](ADRs.md) (`ADR-0008A`) for the public document hierarchy and external mission-plane boundary (`P2-18`).

### Claim -> Evidence Convention (Mandatory)

Every normative statement (**MUST/SHOULD/MAY**) must include a short evidence pointer.

**Evidence pointer format:**

- `Evidence: ADR-XXXX`
- `Evidence: <crate>/src/<path>.rs`
- `Evidence: tests/<path>.rs` or `Evidence: <command>`

If a statement has no enforcement hook, downgrade it to **SHOULD**/**MAY** or move it to ΓÇ£Non-Normative NotesΓÇ¥.

### Change Control

- Changes to **normative interfaces** (artifact schemas, protocol envelopes, public APIs) require either:
  - a new ADR, or
  - a new white paper version (and a migration note).
- Non-normative clarifications may be edited directly (but should remain consistent with ADRs and code).

## Implementation Status Matrix (Grounded)

This matrix prevents aspirational scope creep. It is grounded against `ADRs.md` and the current crate code.

| Area | Claim / Component | Evidence | Status (Implemented / Partial / Planned / Non-goal) | Notes |
|------|-------------------|----------|-----------------------------------------------------|-------|
| Artifact spine | Run Artifact Bundle (v1) writer + signed manifest + Draft 2020-12 schema conformance | Evidence: ADR-0016, ADR-0023, swarm-torch/src/artifacts/, schemas/v1/ | Implemented | Cross-process `.bundle.lock`, `ManifestSignatureV1`, `VerifiedBundleV1` (`TrustPolicy::Strict`), and CAS checkpoints implemented |
| DataOps | Executable `graph.json` schema + sequential scheduler + `DataOpsSession` lifecycle | Evidence: ADR-0017, ADR-0023, swarm-torch-core/src/run_graph.rs, swarm-torch/src/scheduler.rs | Implemented | Sequential graph runner, physical CAS cache-hit verification, and lineage/materialization emitters implemented |
| Extensibility | Sandboxed-by-default extensions | Evidence: ADR-0018, ADR-0025 | Planned | WASM-first `wasmi` host (`W12-04`) |
| Telemetry | Canonical IDs + span/event/metric record types + emitter trait | Evidence: ADR-0012, ADR-0016, swarm-torch-core/src/observe.rs | Partial | `swarmtorch.*` canonical namespace enforced; external OTLP export planned |
| Threat model | Threat model + trust boundaries defined | Evidence: ADR-0008, ADR-0008A | Implemented | Documented; enforced across protocol, coordinator, and artifact boundaries |
| Robustness | Robust aggregation + Non-IID bucketing | Evidence: ADR-0007, ADR-0019, swarm-torch-core/src/aggregation.rs | Implemented | `FedAvg`, `CoordinateMedian`, `TrimmedMean`, `Krum`/`Multi-Krum`, `Bulyan IterativeKrumV1`, `CenteredClippingAggregator`, and `SBucketingTransformer` implemented |
| Compression | Update compression + error feedback + verifiable sparse projection | Evidence: ADR-0021, swarm-torch-core/src/compression.rs | Implemented | `TopK`, `RandomK`, `Quantized`, `ClassicResidual`, `EF21`, and `SharedSeedSparse` (`SharedSeedCoordinateVerifier`) implemented |
| Networking | Message envelope + transport traits + low-MTU chunked framing + TCP/UDP | Evidence: ADR-0004, ADR-0022, swarm-torch-net/src/ | Implemented | `PacketTransport`, `SwarmTransport`, `ChunkedFrameCodec` (16B `CRC-16/IBM-3740`), `TcpPacketTransport`, `UdpPacketTransport`, and gossip wrappers implemented |
| Replay protection | Sequence/timestamp validation enforced (memory-only cache) | Evidence: swarm-torch-core/src/replay.rs, swarm-torch-net/src/protocol.rs, ADR-0008B | Implemented | LRU cache, ┬▒60s skew window, 16-message out-of-order tolerance |
| Message auth | Ed25519 signing/verification with domain-separated preimage | Evidence: ADR-0008, ADR-0008B, ADR-0021, swarm-torch-core/src/crypto.rs | Implemented | 6-stage `MessageEnvelope::decode_and_verify_payload` under `"swarmtorch.envelope.v1"` |
| Gradient validation | Bounds checking (NaN/Inf/L2/coordinate) | Evidence: swarm-torch-core/src/crypto.rs, swarm-torch-core/src/aggregation.rs | Implemented | Pre-aggregation finiteness/shape validation (`MAX_COHORT_SIZE = 1024`) |
| Identity boundary | External identity provider boundary (Sybil resistance out-of-scope) | Evidence: ADR-0008A, swarm-torch-core/src/replay.rs | Partial | Static `CohortAllowlist` implemented; external RA provider profile planned |
| Coordination | Gossip coordination + unified `SwarmRoundCoordinator` | Evidence: ADR-0006A, ADR-0023, swarm-torch/src/coordinator.rs | Implemented | `SwarmRoundCoordinator` (`ModelDeltaV1`/`GradientV1`, equivocation revocation, CAS/SafeTensors commit) and authenticated gossip wrappers implemented |
| Topology | Topology as a first-class config | Evidence: swarm-torch-core/src/algorithms.rs, ADR-0004A | Partial | Enum and `MockNetworkHub` topology simulation implemented; adaptive rewiring planned |
| Staleness | Bounded async + staleness-aware policy | Evidence: ADR-0004B, ADR-0025 | Planned | `FedAsync` stale-delta mixing scheduled for Wave 12 (`W12-03`) |
| Runtime | Tokio/Embassy/`MockRuntime` runtime abstraction | Evidence: ADR-0003, swarm-torch-runtime/src/lib.rs | Implemented | `SwarmClock` / `SwarmRuntime` split with Tokio, deterministic `MockRuntime` scheduler, and `no_std` spin/Embassy surfaces |
| Models | Reference models + losses + SafeTensors v1 persistence | Evidence: ADR-0023, ADR-0024, swarm-torch-models/src/ | Implemented | `LinearModel`, `MlpModel`, `MseLoss`, `CrossEntropyLoss`, `HuberLoss`, `FedProxRegularizer`, `SgdOptimizer`, and `SwarmTorchCheckpointV1` implemented |
| Embedded | `no_std` portability posture | Evidence: ADR-0002, swarm-torch-core/src/lib.rs | Implemented | Core and models `no_std + alloc` and minimal `no_std` gates verified; hardware radio drivers planned |
| GPU | WGPU-first; CUDA optional backend | Evidence: ADR-0015, ADR-0024 | Planned | Isolated to optional `swarm-torch-backends` crate (`W12-01`) |
| Python | Optional, bounded Python interop | Evidence: ADR-0009 | Planned | Keep separate crate boundary |
| Supply chain | Dependency/license/source policy configs | Evidence: SECURITY.md, deny.toml, supply-chain/, .github/workflows/rust.yml | Implemented | `cargo deny`, `cargo audit`, and `cargo vet` are all CI-blocking |

MSRV policy is currently tiered by crate:
- `swarm-torch-core` security-critical path is validated on Rust 1.75.
- top-level/model crates have a higher compiler floor due backend dependency constraints.
- `swarm-torch-runtime` Tokio path is validated on Rust 1.75; Embassy feature path is experimental and not in the Rust 1.75 gate.

## Abstract

SwarmTorch is a Rust-native distributed machine learning and swarm learning framework designed for heterogeneous fleets operating under partial participation, unreliable links, and adversarial conditions. This white paper defines the SwarmTorch system model, architecture boundaries, protocol/algorithm choices, production-readiness requirements, and a conformance story that prevents context drift.

## Scope and Non-Goals

### In-Scope

- Federated/swarm learning primitives, update aggregation, and coordination suitable for hostile networks
- A portable run graph and artifact contract that covers **data wrangling + training**
- Robustness and security posture appropriate for untrusted participants and supply chain risk
- Rust-first developer experience (with optional, explicitly bounded Python integration)

### Out of Scope (Explicit)

- Mission-plane traffic classes, RA/capability gating, PoSw/receipt systems
- SwarmicOS runtime policy enforcement
- Any system that requires ΓÇ£always-onlineΓÇ¥ assumptions as a correctness dependency

## System Model (SwarmTorch-Only)

### Roles and Identities (Current)

SwarmTorch models a fleet as a set of nodes identified by a `PeerId` with an associated `NodeRole`:

- `Coordinator`: may coordinate/aggregate
- `Contributor`: trains locally and sends updates
- `Observer`: receives models but does not contribute
- `Gateway`: bridges networks (e.g., LoRa <-> WiFi) and may aggregate

Evidence: swarm-torch-core/src/identity.rs, swarm-torch-core/src/traits.rs

### Network and Failure Assumptions (Design Target)

SwarmTorch is designed for:

- partial participation
- intermittent connectivity
- heterogeneous link budgets and transport capabilities
- asynchronous delivery and bounded staleness

Evidence: ADR-0004B, ADR-0006A

Tracked follow-up: `P2-12` (`Status: Planned`, Wave 8.2) defines normative vs profiled assumptions (timeouts, quorum ratios, staleness thresholds) in an enforceable matrix.

### Coordination Model (Design Target vs Current)

**Design target:** gossip-based eventual consistency coordination with quorum-based rounds (not PBFT/Tendermint-class consensus).

Evidence: ADR-0006A

**Current implementation:** `SwarmRoundCoordinator` (`swarm-torch/src/coordinator.rs`) implements round-gated quorum aggregation, equivocation revocation, and authoritative CAS/SafeTensors commits, paired with originator-authenticated gossip wrappers (`swarm-torch-net/src/gossip.rs`).

Evidence: ADR-0022, ADR-0023, swarm-torch/src/coordinator.rs, swarm-torch-net/src/gossip.rs

### Transport Model (Current)

SwarmTorch defines `PacketTransport` and GAT-based `SwarmTransport` abstractions with explicit capabilities (reliability class, bandwidth class, message size, multicast support).

Evidence: ADR-0022, swarm-torch-net/src/traits.rs

Current status:

- `ChunkedFrameCodec` (16-byte `CRC-16/IBM-3740` header with bounded out-of-order reassembly), `TcpPacketTransport` (`tcp-transport`), `UdpPacketTransport` (`udp-transport`), `LoopbackPacketTransport`, and `MockTransport`/`MockNetworkHub` are implemented.
- Hardware radio drivers (`BLE`, `LoRa`, `WiFi`) and optional `Zenoh`/`Iroh` adapter crates (`W12-02`) are planned for `v0.2.0`.

Evidence: swarm-torch-net/src/chunked.rs, swarm-torch-net/src/tcp.rs, swarm-torch-net/src/udp.rs, swarm-torch-net/src/mock.rs

### Topology and Staleness (Design Target vs Current)

**Design target:** topology is a first-class, measurable policy; bounded asynchrony uses explicit staleness policies.

Evidence: ADR-0004A, ADR-0004B

**Current implementation:** a basic `Topology` enum exists, but adaptive policies and staleness policy implementation are not present.

Evidence: swarm-torch-core/src/algorithms.rs

## Threat Model and Trust Boundaries (SwarmTorch-Only)

### Threats (Design Target)

SwarmTorchΓÇÖs baseline threat model includes:

| Threat | In Scope | Notes |
|--------|----------|-------|
| Byzantine updates | Yes | Robust aggregation + validation hooks |
| Message tampering | Yes | Signatures (Ed25519) |
| Replay attacks | Yes | Sequence numbers + timestamps |
| Eavesdropping | Yes | Transport security (TLS/DTLS) |
| Sybil attacks | Partial | Requires external identity/attestation |
| Physical device compromise | No | Out of scope (future TEE work only) |
| Side-channels | No | Out of scope |
| Denial of service | Partial | Rate limiting, not exhaustive |

Evidence: ADR-0008, ADR-0008A

### Trust Boundaries (Design Target)

SwarmTorch uses a three-zone model:

- **Trusted zone:** local compute, local storage, key material
- **Verified zone:** signed messages from authenticated peers, validated gradients, robust aggregation outputs
- **Untrusted zone:** network traffic, peer claims, raw updates

Evidence: ADR-0008

### Current Security Posture (Reality Check)

SwarmTorch currently contains:

- Enforced authenticated envelope verification (version gate, signature verification, replay checks).
  Evidence: swarm-torch-net/src/protocol.rs, swarm-torch-core/src/replay.rs
- Production Ed25519 signing/verification with domain separation.
  Evidence: swarm-torch-core/src/crypto.rs
- Gradient bounds validation primitives (`GradientValidator`).
  Evidence: swarm-torch-core/src/crypto.rs

Known limits remain (for example memory-only replay cache persistence), but core message
auth + replay enforcement is implemented in current code.

## Run Graph + Artifact Spine (Normative)

This chapter defines the SwarmTorch ΓÇ£spineΓÇ¥: a unified run DAG plus persisted artifacts that power notebook views and future UIs.

### Unified Run Graph (`graph.json`)

`graph.json` MUST be a semantics-first DAG spec for both data wrangling and training.

Invariant: `graph.json` is the source of truth for run structure; any visualization is derived from it.

Evidence: ADR-0016, ADR-0017.

Hashing invariant: `node_def_hash` MUST be computed over a canonical binary encoding (postcard), not over JSON bytes.

Evidence: ADR-0017, swarm-torch-core/src/run_graph.rs

### Unified Signal Model

SwarmTorch uses one observability model:

- Span: node execution
- Event: discrete fact
- Attributes: structured tags

Artifacts are a persisted view of spans/events/metrics.

Evidence: ADR-0016.

### Canonical IDs and Namespaces (OTel-Compatible, Not OTel-Dependent)

Rule: SwarmTorch defines its own on-disk signal schema. It is OTel-compatible, not OTel-dependent.

- ID sizes match W3C Trace Context / OpenTelemetry conventions:
  - `trace_id`: 16 bytes (artifacts encode as 32 lowercase hex chars; all-zero invalid)
  - `span_id`: 8 bytes (artifacts encode as 16 lowercase hex chars; all-zero invalid)
  - `run_id`: 16 bytes (by default `run_id == trace_id` for the run root trace)
- SwarmTorch-defined attributes MUST use the `swarmtorch.*` namespace.
- Trace propagation SHOULD apply across coordinator <-> gateway <-> worker boundaries.
  - For constrained `embedded_min` nodes, propagation is OPTIONAL; if omitted, the coordinator/gateway must create linking spans and correlation attributes.

Evidence: ADR-0016.
Grounding: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#W3C-Trace-Context, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#OTel-Trace-API

### Artifact Bundle Layout (v1)

Minimum bundle layout:

```
runs/<run_id>/
  manifest.json
  run.json
  graph.json
  spans.ndjson
  events.ndjson
  metrics.ndjson
  datasets/
    registry.json
    lineage.json
    materializations.ndjson
  artifacts/
```

For `edge_std` coordinators/gateways, Parquet/Arrow is the preferred ΓÇ£facts at restΓÇ¥ format:

- `metrics.parquet`, `events.parquet`, `spans.parquet`, `datasets/materializations.parquet`

Evidence: ADR-0016.
Grounding: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#Arrow, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#Parquet

### Manifest Hashing Strategy (Normative)

- Hash algorithm: SHA-256 (raw file bytes)
- Addressing: path-addressed within the bundle (relative paths)
- Partial bundles: allowed; missing entries must be represented explicitly (expected hash/size and optional external URI for re-hydration)

Evidence: ADR-0016.

### Privacy/Security Requirements for Artifacts

Artifacts MUST NOT contain raw training data or raw dataset rows by default. Artifacts MUST support redaction of sensitive fields, and SHOULD default to pointers (URIs) + fingerprints/hashes for dataset payloads.

Evidence: ADR-0016, `SECURITY.md`.

## Architecture

### Workspace Crate Boundaries (Current)

SwarmTorch is a Rust workspace with these crates:

- `swarm-torch-core`: core traits + algorithms (`no_std`-compatible).  
  Evidence: swarm-torch-core/src/lib.rs
- `swarm-torch-net`: transport traits + message envelope (currently includes a mock transport; real transports are placeholders).  
  Evidence: swarm-torch-net/src/lib.rs
- `swarm-torch-runtime`: runtime abstraction (`tokio` and `embassy` feature-gated implementations).  
  Evidence: swarm-torch-runtime/src/lib.rs
- `swarm-torch-models`: reference models + backend integration scaffolding.  
  Evidence: swarm-torch-models/src/lib.rs
- `swarm-torch`: user-facing crate; re-exports and config surface.  
  Evidence: swarm-torch/src/lib.rs

### Feature Flag Contract (Current)

The `swarm-torch` crate defines feature gates for:

- Portability: `std`, `alloc`  
  Evidence: swarm-torch/Cargo.toml
- Runtime: `tokio-runtime`, `embassy-runtime`  
  Evidence: swarm-torch/Cargo.toml
- Models: Self-contained `no_std` reference models and analytical losses in `swarm-torch-models` (external ML framework adapters isolated per `ADR-0024`)  
  Evidence: swarm-torch-models/Cargo.toml, ADR-0024
- Transports: `tcp-transport`, `udp-transport`, `ble-transport`, `lora-transport`  
  Evidence: swarm-torch/Cargo.toml
- Robust aggregation: `robust-aggregation`  
  Evidence: swarm-torch/Cargo.toml, swarm-torch-core/Cargo.toml

**Policy (normative):** feature flags MUST be treated as additive and MUST NOT silently introduce SemVer-breaking behavior.

Evidence: ADR-0002 (policy).
Grounding: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#Cargo-Features, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#SemVer-2.0.0

### Runtime Abstraction (Current)

SwarmTorch defines a `SwarmRuntime` trait and feature-gated implementations:

- `tokio` runtime wrapper (usable today)
- `embassy` runtime wrapper (placeholder; not yet complete and not part of Rust 1.75 conformance)

Evidence: swarm-torch-runtime/src/lib.rs

## Protocol

### Message Envelope (Current)

SwarmTorch defines a single message envelope for swarm communications with:

- protocol version `(major, minor)`
- `message_type` discriminator
- `sender` (raw Ed25519 public key bytes, 32-byte array)
- monotonic `sequence` (replay/ordering primitive)
- `timestamp` (expiry primitive)
- `payload` bytes
- optional `signature`

Evidence: swarm-torch-net/src/protocol.rs

### Serialization (Current)

Envelope serialization is currently implemented via `postcard` for compact binary encoding.

Evidence: swarm-torch-net/src/protocol.rs

Authentication and replay semantics are enforced in the receive path:
- signatures MUST verify against sender raw public key bytes
- timestamps are evaluated in Unix seconds with bounded skew
- per-sender sequence windows reject duplicates and stale sequences

Evidence: swarm-torch-core/src/crypto.rs, swarm-torch-core/src/replay.rs, swarm-torch-net/src/protocol.rs, ADR-0008B.

## Algorithms

### Aggregation (Current)

Implemented robust aggregation configuration includes:

- FedAvg (no Byzantine protection)
- Coordinate-wise median
- Trimmed mean (`trim_ratio`)
- Krum (`krum` feature gate; implemented)

Evidence: swarm-torch-core/src/aggregation.rs

Planned (not yet implemented in code): Bulyan and rejection/telemetry integration.  
Implemented but not yet release-gated: initial robustness harness scenarios (outlier/sign-flip/NaN-Inf/collusion + optional Krum gate).

Evidence: ADR-0007, ADR-0007A

### Compression (Current)

Compression primitives include `TopK`, `RandomSparse`, and `Quantized` modes.

Evidence: swarm-torch-core/src/compression.rs

Tracked follow-up: `P2-15` (`Status: Planned`, Wave 8.3) specifies default compression profiles and convergence/accuracy expectations.

Evidence: ADR-0007, ADR-0007A.

## Security Engineering (Current + Planned)

### Identity and Signing (Current)

SwarmTorch enforces Ed25519 signing/verification for authenticated envelopes using a
domain-separated canonical preimage that binds protocol version, sender key, sequence,
timestamp, message type, and payload hash.

`MessageEnvelope.sender` is the sender's raw Ed25519 public key bytes; hashed peer IDs
are not a valid sender encoding for envelope verification.

Evidence: swarm-torch-core/src/crypto.rs, swarm-torch-net/src/protocol.rs, ADR-0008B.

### Gradient Validation (Current)

SwarmTorch includes a `GradientValidator` for basic bounds checking (NaN/Inf/coordinate bounds/L2 norm).

Evidence: swarm-torch-core/src/crypto.rs

Tracked follow-up: `P2-13` (`Status: Planned`, Wave 8.2) maps threat model requirements to concrete enforcement hooks (protocol parsing fuzzing, signature verification, byzantine harness, supply-chain gates).

Evidence: ADR-0008, ADR-0008A, `SECURITY.md`, `deny.toml`, `supply-chain/`.

## DataOps: Pipeline DSL + Assets

SwarmTorch treats datasets and intermediate products as named **assets**.

Minimum DataOps artifacts (always emitted for an artifact-backed run):

- `datasets/registry.json`: dataset IDs, fingerprints, schemas, logical locations, license flags, PII classification tags
- `datasets/lineage.json`: inputs -> transforms -> outputs edges
- `datasets/materializations.*`: one row per stage output (rows/bytes/timing/cache/quality)

Grounding: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#OpenLineage

Data trust boundary:

- Dataset sources MUST be classified (trusted vs untrusted) in `datasets/registry.json`.
- Nodes that read from untrusted locations MUST be marked as an unsafe surface in run artifacts.

Evidence: ADR-0017.

Fingerprint v0 (metadata-first) is defined so it can be computed without raw rows:

- `source_fingerprint_v0` = hash of normalized source descriptor (URI + content type + auth marker + optional etag/version)
- `schema_hash_v0` = hash of normalized schema descriptor (canonical form when available)
- `recipe_hash_v0` = hash of transform definition (node definition + upstream fingerprints)
- `dataset_fingerprint_v0` = hash of `{source_fingerprint_v0, schema_hash_v0, recipe_hash_v0}`

Evidence: ADR-0017, swarm-torch-core/src/dataops.rs

Tracked follow-up: `P2-14` (`Status: Planned`, Wave 8.2) publishes standalone `graph.json` schema docs with explicit compatibility/migration guidance for typed optional policy/resource fields.

Evidence: ADR-0017.

Current status: `graph.json` schema types and canonical hashing are implemented (with `node_id` derived from `node_key`, and `node_def_hash = sha256(postcard(NodeDefCanonicalV1))`). Materialization emitters are implemented (`DataOpsSession`/artifact sink path). A sequential execution engine MVP is implemented (`swarm-torch/src/scheduler.rs`) with deterministic topological execution and fail-closed status recording; distributed orchestration remains planned. `execution_hint` is planner metadata and excluded from `node_def_hash`. Typed optional `NodeV1` fields (`op_hash`, `resources`, `cache_policy`, `materialization_policy`) are implemented; `op_hash` is computed metadata.

Evidence: swarm-torch-core/src/run_graph.rs

## Developer Experience (DX)

### Notebook-Native Views (Phase 1)

Implemented (initial): standalone artifact reader that generates a self-contained `report.html` from a run bundle. Notebooks can render the HTML as a pure reader workflow.

Evidence: ADR-0016.
Evidence: swarm-torch/src/report/load.rs, swarm-torch/src/bin/swarm_torch_report.rs

### Local Dashboard (Phase 2)

Tracked follow-up: `P2-16` (`Status: Planned`, Wave 8.3) defines a local server profile that reads artifacts without requiring a tracking DB by default.

Evidence: ADR-0016.

### Desktop Wrapper (Phase 3)

Tracked follow-up: `P2-17` (`Status: Planned`, Wave 8.3) scopes the Tauri/Iced wrapper once UX stabilizes and treats the UI surface as part of the threat model.

Evidence: ADR-0016.

## Operations

### MSRV (Current)

Workspace `rust-version` is currently set to **1.75**.

Evidence: Cargo.toml

### docs.rs (Operational Constraints)

docs.rs builds are sandboxed and do not allow network access. Configure docs.rs builds using `package.metadata.docs.rs` (features/targets/rustdoc args) and ensure docs build without fetching at build time.

Evidence: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#Docs-RS, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#Docs-RS-Metadata

### Release Gates (Planned/Partial)

- Supply chain gates: `cargo audit`, `cargo deny`, `cargo vet`  
  Evidence: SECURITY.md, deny.toml, supply-chain/
- Reproducibility and benchmark gates (planned)  
  Evidence: ADR-0013, ADR-0014

Status: `Implemented` (Wave 8.1). Appendix B now maps CI workflow hooks to conformance gates and security policy.

Evidence: ADR-0011, ADR-0013, ADR-0014, `SECURITY.md`.

## Integration Profiles

### Swarmic Network Provider Profile (Secondary)

Tracked follow-up: `P2-18` (`Status: Planned`, Wave 8.3) defines integration-only mapping without importing mission-plane semantics into SwarmTorch core.

Evidence: `ADR-0008A`, `CONTRIBUTING.md`.

## Roadmap and Open Problems

Tracked follow-up: `P2-19` (`Status: Planned`, Wave 8.3) records open research gaps, heterogeneity mitigation, compression tradeoffs, privacy roadmap, and maturity milestones.

Evidence: `ROADMAP.md`.

## Appendices

### Appendix A: Normative Invariants vs Profiled Parameters (Template)

**Normative invariants (MUST):**

- `graph.json` is the source of truth for run structure; visualization is derived.  
  Evidence: ADR-0016, ADR-0017
- Artifact bundle uses path-addressed `manifest.json` with SHA-256 hashes of raw file bytes.  
  Evidence: ADR-0016
- IDs and namespaces:
  - `trace_id` = 16 bytes; `span_id` = 8 bytes; all-zero invalid
  - SwarmTorch attributes use `swarmtorch.*`
  - SwarmTorch schema is OTel-compatible but not OTel-dependent  
  Evidence: ADR-0016
- Artifacts must not include raw dataset rows by default; prefer pointers + hashes.  
  Evidence: ADR-0016
- Extensions are sandboxed-by-default; unsafe extensions are explicit and recorded in artifacts.  
  Evidence: ADR-0018

**Profiled parameters (SHOULD live in this document, with defaults):**

- telemetry sampling rates, retention, and sinks
- cache/materialization policy defaults
- dataset sample/preview policy (explicit opt-in; size limits)
- aggregation and compression defaults per deployment profile
- staleness/quorum/timeout defaults per topology profile

### Appendix B: Conformance Hooks and Release Gates (Template)

This appendix lists conformance hooks that keep the white paperΓÇÖs normative claims honest.

**Legend:** `Implemented` means the hook exists and can be run today. `Planned` means the hook is specified but not yet present as a script/test/CI gate in this repo.

#### Build and Basic Quality Gates

| Gate | Hook | Status | Evidence |
|------|------|--------|----------|
| Workspace builds | `cargo check --workspace` | Implemented | Evidence: Cargo.toml |
| Workspace tests | `cargo test --workspace` | Implemented | Evidence: Cargo.toml |
| Docs build | `cargo doc --workspace --no-deps` | Implemented | Evidence: Cargo.toml |
| embedded_min core build | `cargo build -p swarm-torch-core --no-default-features` | Implemented | Evidence: .github/workflows/rust.yml |
| no_std + alloc core build | `cargo build -p swarm-torch-core --no-default-features --features alloc` | Implemented | Evidence: swarm-torch-core/Cargo.toml |
| Embedded target build | `cargo build -p swarm-torch-core --no-default-features --features alloc --target thumbv7em-none-eabihf` | Planned | Evidence: ADR-0002 |

#### Supply-Chain and Policy Gates

| Gate | Hook | Status | Evidence |
|------|------|--------|----------|
| RustSec advisories | `cargo audit` | Implemented | Evidence: .github/workflows/rust.yml, SECURITY.md |
| License/source policy | `cargo deny check` | Implemented | Evidence: .github/workflows/rust.yml, deny.toml, SECURITY.md |
| Human audit tracking | `cargo vet` | Implemented | Evidence: .github/workflows/rust.yml, supply-chain/, SECURITY.md (blocking in CI) |
| Public API SemVer regression | `cargo semver-checks check-release` | Planned | Evidence: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#cargo-semver-checks |
| OSV lockfile scan | CI: `google/osv-scanner-action` (Cargo.lock) | Planned | Evidence: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#OSV-Scanner-Action |

#### CI/CD Hardening (Security)

| Gate | Hook (intended) | Status | Evidence |
|------|------------------|--------|----------|
| Least-privilege workflows | CI: explicit `permissions:` and minimal `GITHUB_TOKEN` scope | Implemented | Evidence: .github/workflows/rust.yml, SECURITY.md |
| Action pinning | CI: pin third-party `uses:` to full commit SHAs | Implemented | Evidence: .github/workflows/rust.yml, SECURITY.md |
| Artifact attestations | CI: generate/publish build provenance attestations | Planned | Evidence: SECURITY.md, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#GitHub-Attestations, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#SLSA |

#### Protocol Robustness (Fuzzing)

| Gate | Hook (intended) | Status | Evidence |
|------|------------------|--------|----------|
| Protocol parsing fuzzing | `cargo fuzz run <fuzz_target>` (envelope parse/serialize/roundtrip) | Planned | Evidence: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#cargo-fuzz, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#Rust-Fuzz-Book |

#### Release Gates (From ADR-0014)

These are blocking gates for releases once implemented in CI:

| Gate | Hook (intended) | Status | Evidence |
|------|------------------|--------|----------|
| Convergence benchmark | `cargo bench -p swarm-torch --bench swarm_bench` (or dedicated benchmark harness) | Planned | Evidence: ADR-0014 |
| Topology A/B test | Dedicated benchmark runner outputs comms + convergence curves for >=3 topologies | Planned | Evidence: ADR-0004A, ADR-0014 |
| Staleness + dropout | Simulation test (>=30% dropout) must not diverge under bounded staleness | Planned | Evidence: ADR-0004B, ADR-0014 |
| Robustness harness | Attack harness scenarios must pass; publish robustness report | Planned | Evidence: ADR-0007A, ADR-0014 |
| Embedded build | Embedded target build must pass | Planned | Evidence: ADR-0002, ADR-0014 |

#### Artifact Contract Conformance (ADR-0016/0017/0018)

| Gate | Hook (intended) | Status | Evidence |
|------|------------------|--------|----------|
| Artifact bundle schema | Validate bundle layout + required files + schema versions | Planned | Evidence: ADR-0016 |
| Manifest integrity | `cargo test -p swarm-torch artifacts::tests::bundle_manifest_roundtrip` (recompute SHA-256 for manifest entries) | Implemented | Evidence: swarm-torch/src/artifacts/bundle.rs |
| Graph hash normalization | `cargo test -p swarm-torch artifacts::tests::graph_write_normalizes_ids_and_hashes` | Implemented | Evidence: swarm-torch/src/artifacts/bundle.rs |
| Dataset fingerprint determinism | `cargo test -p swarm-torch-core dataops::tests::dataset_fingerprint_is_deterministic` | Implemented | Evidence: swarm-torch-core/src/dataops.rs |
| No raw rows by default | Assert bundle does not embed raw dataset rows unless explicitly enabled + size-limited | Planned | Evidence: ADR-0016 |
| Untrusted input marking | If dataset source is `untrusted`, artifacts mark unsafe surface | Implemented | Evidence: swarm-torch/src/artifacts/session.rs, swarm-torch/src/report/model.rs |
| Unsafe extension marking | `unsafe_extension` nodes are explicitly flagged in graph/events/materializations/run summary | Implemented | Evidence: swarm-torch/src/artifacts/session.rs, swarm-torch/src/report/model.rs |

#### Drift / Doc Hygiene (Repo-Local)

| Gate | Hook | Status | Evidence |
|------|------|--------|----------|
| No legacy docs/ links | `rg -n \"\\bdocs/\" -g\"*.md\" .` (expect only external URLs or clearly marked historical text) | Implemented | Evidence: CONTRIBUTING.md |
| Evidence discipline | Manual review: any **MUST/SHOULD/MAY** has an `Evidence:` pointer nearby | Implemented | Evidence: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.md |

### Appendix C: Artifact Schema References

Tracked follow-up: `P2-20` (`Status: Planned`, Wave 8.3) publishes versioned JSON schema definitions and Parquet schema notes.

### Appendix D: Glossary

**Run:** A single execution of a SwarmTorch pipeline (data + training), producing a run artifact bundle.

**RunId:** Stable identifier for a run (default equal to the run root `trace_id`).  

**TraceId / SpanId:** Identifiers for the unified spans/events model (compatible sizes with W3C/OTel).  

**Span:** A timed execution interval for a node in the run graph (dataset stage, training round, aggregation, export).

**Event:** A discrete fact that occurred during a span (schema inferred, outlier rejected, retry, checkpoint saved).

**Asset:** A named data product (dataset or intermediate) referenced by an asset key (e.g., `dataset://<namespace>/<name>`).

**Fingerprint:** A stable identifier for an asset instance (content hash and/or recipe hash).

**Materialization:** The act of persisting an asset output, with recorded stats (rows/bytes/schema hash/timing/cache hit).

**Node definition hash (`node_def_hash`):** Hash of the operation definition (what should run). Must exclude runtime-only fields.

**Content hash:** Hash of produced output bytes (what did run and what it produced).

### Appendix E: Bibliography

See `SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md`.


---

## Post-Wave 8.1 Engineering Specifications & Ratified Canon (Waves 9–12 / ADR-0019..ADR-0025)

This chapter records the normative mathematical, protocol, hardware, security, and verification contracts ratified in `ADR-0019` through `ADR-0025` (`ADRs.md`). Each slice progresses through four explicit states: `Contract Ratified` $\rightarrow$ `Implementation Landed` $\rightarrow$ `Conformance Passed` $\rightarrow$ `Profile Released`.

### 1. Normative vs. Profiled Operational Matrix (Closes `P2-12`, `ADR-0002`, `ADR-0003`, `ADR-0022`, `ADR-0023`)

SwarmTorch distinguishes between **normative protocol invariants** (enforced across all profiles) and **profiled deployment parameters** (tuned and verified per network/hardware class):

| Profile | Target Hardware & Runtime Tier | Typical RAM & Compiler Profile | Codec-Visible Frame / Carrier | `round_timeout_ms` | `min_quorum_ratio` (subject to $n, f, m$ check) | `max_staleness_rounds` | `StalenessSchedule` (`FedAsync`) | Recommended Aggregator & Compressor |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- | :--- |
| **`EmbeddedLora`** (`no_std + alloc`, Experimental until `P2-06` 6-gate firmware conformance passes) | ARM Cortex-M4/M7, ESP32-C3/S3 (`EmbassyRuntime` + explicit wall-clock or session-lease provider) | 64 KiB – 512 KiB SRAM; verified Rust `1.75` core/models/net | LoRa (`250B` frame $\implies 234\text{B}$ stride), BLE ATT (`247B` MTU $\implies 244\text{B}$ notification / $228\text{B}$ stride), `max_chunks = 256` (~58–59 KiB cap) | `120_000` | `0.30` (re-verified against $2f+3$) | `16` | `Polynomial { a: 0.5 }` | `CenteredClipping` ($\mathcal{O}(Lnd)$ time, $\mathcal{O}(d)$ scratch) + `ClassicResidual(SharedSeedSparse)` |
| **`TacticalMesh`** (`std` Edge / Gateway) | Jetson Orin Nano, Raspberry Pi 5, ROS 2 Robot (`TokioRuntime` `rt,time`) | 512 MiB – 8 GiB; adapter-specific compiler floor (`1.75` base, `1.80+` adapters) | UDP Gossip (`1200B` frame $\implies 1184\text{B}$ stride), `swarm-torch-transport-zenoh`, `swarm-torch-transport-iroh` (1-RTT QUIC stream) | `15_000` | `0.50` (re-verified against $B, f_B$) | `5` | `Hinge { a: 0.5, b: 2 }` | Conservative `$s$-Bucketing` ($n \bmod s = 0, f_B = \min(f, B)$) + One-Pass `Multi-Krum` ($1 \le m \le B - 2f_B - 2$) + `ClassicResidual` |
| **`ClusterFast`** (`std` Server/Cloud) | x86_64 / ARM64 Host (`TokioRuntime` + `swarm-torch-backends`) | $\ge 16$ GiB | Length-prefixed TCP (`u32_le(len)`), Iroh 1-RTT QUIC, Zenoh | `2_000` | `0.80` (re-verified against $4f_B+3$) | `1` | `Constant` | Conservative `$s$-Bucketing` + `Bulyan` (`BulyanProfile::IterativeKrumV1`) + `ClassicResidual` / synchronized `EF21` |

Evidence: ADR-0002, ADR-0003, ADR-0019, ADR-0021, ADR-0022, ADR-0023.  
Grounding: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#FedAsync-2019, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#CenteredClipping-ICML21

### 2. Threat-Model-to-Enforcement Mapping (Closes `P2-13`, `ADR-0019..ADR-0025`)

| Threat Class | Concrete Enforcement Mechanism & Code Seam | Owner ADR / Slice | Verification Gate |
| :--- | :--- | :--- | :--- |
| **Byzantine `NaN`/`Inf` sort poisoning & finite-input `f32` overflow (`B2`)** | 1. Fail-closed `.is_finite()` validation on all coordinates (including unselected/trimmed) before sorting/arithmetic<br/>2. `f32::total_cmp` / `f64::total_cmp` + canonical peer-index tie-breaking<br/>3. Pre-operation `f64` widening for all sums, means, midpoints (`f32((f64(a)+f64(b))/2)`), and squared distances + checked byte/scratch/work budgets | `ADR-0019` (`W9-01`) | `cargo test -p swarm-torch-core --test robustness_harness` |
| **High-dimensional & Non-IID Byzantine attacks (`B1`, `B6`, `B7`)** | 1. One-pass `Multi-Krum` ($n \ge 2f+3, k = n-f-2, 1 \le m \le n-f-2$, index self-exclusion)<br/>2. `Bulyan` Stage 2 averaging $\beta = n-4f$ values nearest to lower observed median (Stage 1 terminal pool gated)<br/>3. `CenteredClipping` & conservative `$s$-Bucketing` ($f_B = \min(f, B)$) | `ADR-0019` (`W9-01`, `W11-01`) | `swarm-torch-core::aggregation::tests` |
| **Sybil replay eviction, same-second replay, & restart state loss (`B3`, `P2-04`)** | 1. `CohortAllowlist` production default (`P2-03`) + expiry-indexed safe `BTreeMap` cache (removing `lru`, retiring `RUSTSEC-2026-0002` & `RUSTSEC-2026-0253`)<br/>2. Full $A+F$ validity retention ($n > t_{\max} + A$) and fail-closed `CapacityExhausted`<br/>3. `RecoveryRequired` startup/restart state + directional handshake committing 256-bit receiver challenge into `session_id` + 6-stage atomic receive pipeline (`P2-01`, `P2-04`) | `ADR-0020` (`W9-02`) | `cargo test -p swarm-torch-net --test replay_integration` |
| **Hostile wire metadata & adaptive post-seed sparse fabrication (`B4`, `B5`, `B6`)** | 1. Pre-allocation deserialization/shape/index/dequantization bounds on `compression.rs` wire types<br/>2. Transactional `ClassicResidual` default & gated per-peer `EF21` state sync (`StateAck`)<br/>3. Verifiable pre-seed Merkle/vector commitments (ordered *after* DP sanitization) + $\mathcal{O}(k)$ sparse sampling for `SharedSeedSparse` | `ADR-0021` (`W9-03`, `W11-02`) | `swarm-torch-core::compression::tests` |
| **Pre-auth chunk fragmentation & cross-relay identity spoofing (`T1`)** | 1. Frozen 16-byte header (`[0x53, 0x54]`, `CRC-16/IBM-3740`, fixed stride $P$), observed-source pre-auth byte/count caps, and conflict quarantine up to hard monotonic deadline<br/>2. Originator-signed recipient wrappers binding `(inner_digest, recipient_session_id)` | `ADR-0022` (`W9-04`, `W10-03`, `W11-03`) | `swarm-torch-net::tests` |
| **Artifact tampering, TOCTOU reopen, unlisted files, & concurrent writers (`C2`, `C3`, `P2-02`)** | 1. Write-once CAS checkpoints (`blobs/sha256/<digest>`) with checked `sync_all` + directory sync<br/>2. Exclusive OS-backed bundle lockfile (`P2-02`)<br/>3. `ManifestSignatureV1` (raw 16-byte `RunId`) + `TrustPolicy::Strict` default consuming only manifest-listed verified bytes | `ADR-0023` (`W10-01`, `W10-02`, `P2-10`) | `cargo test -p swarm-torch artifacts::tests` |
| **Untrusted Wasm extension resource exhaustion (`T3`)** | Installed `wasmi` `store.limiter(|s| &mut s.limits)` + `trap_on_grow_failure(true)` + engine fuel active *before* instantiation/start functions + checked guest pointer bounds + zero WASI ambient authority | `ADR-0018`, `ADR-0025` (`W12-04`) | `swarm-torch::wasm_sandbox::tests` |
| **Supply-chain advisories & unused default framework bloat (`D1`–`D4`, `P2-05`, `P2-11`)** | Remove `lru` (`W9-02`), remove `burn`/`tcp-transport` from defaults and sever `net/std -> tokio` (`W9-05`), isolate adapters (`swarm-torch-backends`, `zenoh`, `iroh`), and promote `cargo vet` to blocking before `W10` beta (`P2-05`) | `ADR-0002`, `ADR-0020`, `ADR-0024` | `cargo deny check && cargo audit && cargo vet` |

Evidence: ADR-0002, ADR-0003, ADR-0019..ADR-0025, `SECURITY.md`.

### 3. Standalone `graph.json` Schema Compatibility & JSON/Parquet Split (Closes `P2-14`, `P2-20a`/`P2-20b`, `ADR-0016`, `ADR-0017`, `ADR-0023`)

- **Canonical Hashed Fields (`NodeDefCanonicalV1`, `postcard`-serialized for `node_def_hash_v1`):**
  - `node_key` (`1..=256` UTF-8 bytes), `op_kind` (`Data | Train | Eval | Export`), `op_type` (`1..=128` bytes), `inputs` (`Vec<AssetRefV1>`), `outputs` (`Vec<AssetRefV1>`), `params` (`CanonParams` — `BTreeMap<String, CanonValue>`, rejecting non-finite `f64`), `code_ref`, `unsafe_surface`, and `execution_trust` (`Core | SandboxedExtension | UnsafeExtension`).
- **Additive Operational Metadata Fields (Excluded from `node_def_hash_v1`):**
  - `node_id` (`Option<NodeId>`, derived via `node_id_from_key`), `node_def_hash` (`Option<[u8; 32]>`), `execution_hint` (`Option<ExecutionHint>`), `cache_policy` (`Option<CachePolicyV1>`), `materialization_policy` (`Option<MaterializationPolicyV1>`), `resources` (`Option<ResourceRequestV1>`), and `op_hash` (`Option<[u8; 32]>`).
  - **Compatibility & Cache Rule (`C2`):** Modifying operational metadata fields does not mutate `node_def_hash_v1` or `recipe_hash_v0`. However, a valid training/checkpoint cache hit in `scheduler.rs` requires both recipe-key equality AND verified physical CAS payload presence, digest match, `TrustPolicy::Strict` satisfaction, and state restoration (`CachePolicy::Bypass` forces execution).
- **Schema Deliverables (`P2-20a` / `P2-20b`):** Versioned JSON Schema 2020-12 contracts (with streaming size bounds and lossless `u64` encodings) ship in `W10-02` (`P2-20a`), while Arrow/Parquet columnar schemas ship in `W12-04` (`P2-20b`).

Evidence: ADR-0016, ADR-0017, ADR-0023, ADR-0025, `swarm-torch-core/src/run_graph.rs`.

### 4. Compression Profiles, Wire Representations, & Qualified Convergence Contracts (Closes `P2-15`, `ADR-0021`)

| Compression Profile (`compression.rs`) | Actual Wire Representation ($d$ floats) | Error Feedback Contract | Robust Aggregator Compatibility & Complexity | Qualified Convergence & Theorem Assumptions |
| :--- | :--- | :--- | :--- | :--- |
| `CompressionMethod::None` + `CompressedData::Dense(Vec<u8>)` | $4d$ LE `f32` bytes (all finite) | None | All aggregators in $\mathbb{R}^d$ | Baseline uncompressed distributed SGD / `ModelDeltaV1` |
| `CompressionMethod::Quantized { scale }` + `CompressedData::Dense(Vec<u8>)` | $d$ signed `i8` bytes + 4B finite positive `scale` (widened dequantization check) | `ClassicResidual` recommended (truncating fixed-scale INT8 is not uniformly contractive near 0) | All aggregators in $\mathbb{R}^d$ | Bounded quantization residual under `ClassicResidual` |
| `CompressionMethod::TopK { k_ratio }` + `CompressedData::Sparse { indices: Vec<u32>, values: Vec<u8> }` | $8k$ bytes ($k = \lceil d \cdot k_{\text{ratio}} \rceil$, strictly increasing `u32` indices + 4B LE `f32` values) | `ClassicResidual` (`W9-03` default, one-shot sender commit) or synchronized `EF21` ($\mathcal{O}(nd)$ receiver state + `StateAck`) | `FedAvg` / `CenteredClipping` (disjoint support causes zero-vector bias under `Krum`/`Median` unless `EF21` reconstructs dense $s_{t+1}^i$) | Contractive compressor assumptions per Richtárik et al. (NeurIPS 2021) / *Byz-EF21*; no blanket $\mathcal{O}(1/T)$ claimed for arbitrary nonlinear compositions |
| `SharedSeedSparse { k_ratio, seed }` (`W11-02`) | $4k + 32$ bytes (equal to `TopK` at $k=8$, $\approx 2\times$ smaller asymptotically before wrapper/commitment proof bytes) | `ClassicResidual` over sanitized $z_t$ + pre-seed Merkle/vector commitments | Compatible in projected $\mathbb{R}^k$ ($\mathcal{O}(n^2 k)$ pairwise work, $\mathcal{O}(nk)$ stored vectors, $\mathcal{O}(k)$ sparse sampler) | Requires verifiable pre-seed coordinate commitments (or declared `TrustedCompressor`) and stated sparse-domain robustness assumptions (*RoSDHB*, AISTATS 2026) |

Evidence: ADR-0021, `swarm-torch-core/src/compression.rs`.  
Grounding: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#EF21-NeurIPS21, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#RoSDHB-AISTATS26

### 5. Mission-Plane Integration Boundary Specification (Closes `P2-18`, `ADR-0008A` Addendum)

SwarmTorch enforces a strict architectural firewall between its swarm learning/DataOps plane and external mission, autonomy, or fleet-control planes:
1. **Enrollment & Identity $\rightarrow$ `PeerAdmissionPolicy`:** External PKI, hardware attestation, or mission capability grants populate `PeerAdmissionPolicy` (`CohortAllowlist` default), Verified by the host integration adapter. External enrollment **never** bypasses originator Ed25519 signature verification, `ADR-0020` receiver-bound `session_id` establishment, replay protection, or `ADR-0023` one-contribution-per-round deduplication.
2. **Learning Rounds & Checkpoints $\rightarrow$ Mission Actuation:** Committed SwarmTorch checkpoints establish only that the declared training/aggregation recipe executed over the authenticated contributions. They **do not** certify physical actuation safety or mission authorization; external control loops own safety interlocks and actuation decisions.
3. **Transport Capabilities $\rightarrow$ Radio/Network QoS:** External traffic-class and link-key policies configure `PacketCapabilities` and stride $P$, but routing topics, radio addresses, or relay link authentication **never** grant contributor authority or override pre-authentication memory budgets.
4. **Signed Artifacts $\rightarrow$ External Attestations:** `TrustPolicy::Strict` verifies the integrity and provenance of manifest-listed bundle files under external trust anchors; exported receipts **do not** attest to unlisted files, raw external sensor truth, or mission execution outcomes.

Evidence: ADR-0008A Ratification Addendum, ADR-0002, ADR-0020, ADR-0022, ADR-0023.
