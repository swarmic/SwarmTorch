# SwarmTorch Technical White Paper (ST-TWP)

**Version:** v0.1 (Draft)  
**Date:** 2026-02-08  
**Scope:** SwarmTorch-only (this is not a SwarmicOS/GridSwarm/Swarmflow spec)  

## Reader's Guide

### Normative Language

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this document are to be interpreted as described in RFC 2119 and RFC 8174.

Evidence: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#RFC-2119, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#RFC-8174

### Canon and Drift Controls

This document is intended to be a **canonical SwarmTorch reference**, co-equal with:

1. `ADRs.md` (architecture decisions)
2. The `swarm-torch-*` crate source code (implementation truth)

The following documents are **secondary** and must not be treated as SwarmTorch requirements:

- `SWARMIC_NETWORK_WHITE_PAPER_v12.*.md` (integration context only; do not import mission-plane semantics into core)
- `adr-with-updates.md` (historical notes; superseded by `ADRs.md`)

### Claim -> Evidence Convention (Mandatory)

Every normative statement (**MUST/SHOULD/MAY**) must include a short evidence pointer.

**Evidence pointer format:**

- `Evidence: ADR-XXXX`
- `Evidence: <crate>/src/<path>.rs`
- `Evidence: tests/<path>.rs` or `Evidence: <command>`

If a statement has no enforcement hook, downgrade it to **SHOULD**/**MAY** or move it to “Non-Normative Notes”.

### Change Control

- Changes to **normative interfaces** (artifact schemas, protocol envelopes, public APIs) require either:
  - a new ADR, or
  - a new white paper version (and a migration note).
- Non-normative clarifications may be edited directly (but should remain consistent with ADRs and code).

## Implementation Status Matrix (Grounded)

This matrix prevents aspirational scope creep. It is grounded against `ADRs.md` and the current crate code.

| Area | Claim / Component | Evidence | Status (Implemented / Partial / Planned / Non-goal) | Notes |
|------|-------------------|----------|-----------------------------------------------------|-------|
| Artifact spine | Run Artifact Bundle (v1) writer + manifest hashing exists | Evidence: ADR-0016, swarm-torch/src/artifacts.rs | Partial | Atomic writes for JSON/manifest; thread-safe sink; schema validator + profiled defaults pending |
| DataOps | Executable `graph.json` schema + deterministic hashing + fingerprint metadata schemas | Evidence: ADR-0017, swarm-torch-core/src/run_graph.rs, swarm-torch-core/src/dataops.rs | Partial | Execution engine is pending; registry/lineage/materialization emitters are metadata-first |
| Extensibility | Sandboxed-by-default extensions | Evidence: ADR-0018 | Planned | WASM-first host (future) |
| Telemetry | Canonical IDs + span/event/metric record types + emitter trait | Evidence: ADR-0012, ADR-0016, swarm-torch-core/src/observe.rs | Partial | Tracing integration + OTLP export is not implemented yet |
| Threat model | Threat model + trust boundaries defined | Evidence: ADR-0008, ADR-0008A | Implemented | Documented; enforcement varies by feature |
| Robustness | Robust aggregation algorithms | Evidence: swarm-torch-core/src/aggregation.rs | Partial | FedAvg/TrimmedMean/Median implemented; harness pending |
| Compression | Update compression primitives | Evidence: swarm-torch-core/src/compression.rs | Partial | TopK/Quantized implemented; convergence guarantees TBD |
| Networking | Message envelope + transport trait | Evidence: swarm-torch-net/src/protocol.rs, swarm-torch-net/src/traits.rs | Partial | Real transports are placeholders |
| Replay protection | Sequence/timestamp validation enforced (memory-only cache) | Evidence: swarm-torch-core/src/replay.rs, swarm-torch-net/src/protocol.rs, ADR-0008B | Implemented | LRU cache, ±60s skew window, 16-message out-of-order tolerance |
| Message auth | Ed25519 signing/verification with domain-separated preimage | Evidence: ADR-0008, ADR-0008B, swarm-torch-core/src/crypto.rs | Implemented | Production Ed25519 path; sender field uses raw public key bytes |
| Gradient validation | Bounds checking (NaN/Inf/L2/coordinate) | Evidence: swarm-torch-core/src/crypto.rs | Implemented | Used as a building block for verified zone |
| Identity boundary | External identity provider boundary (Sybil resistance out-of-scope) | Evidence: ADR-0008A | Planned | IdentityProvider trait is not implemented in code yet |
| Coordination | Gossip coordination + quorum rounds | Evidence: ADR-0006A, swarm-torch-core/src/consensus.rs | Planned | Current crate has structs only; protocol implementation pending |
| Topology | Topology as a first-class config | Evidence: swarm-torch-core/src/algorithms.rs, ADR-0004A | Partial | Enum exists; adaptive policies/rewiring not implemented |
| Staleness | Bounded async + staleness-aware policy | Evidence: ADR-0004B | Planned | `staleness.rs` not implemented yet |
| Runtime | Tokio/Embassy runtime abstraction | Evidence: swarm-torch-runtime/src/lib.rs | Partial | Tokio wrapper works; Embassy is placeholder |
| Models | Reference models + backend integration | Evidence: swarm-torch-models/src/simple.rs | Partial | Burn integration is placeholder |
| Embedded | `no_std` portability posture | Evidence: ADR-0002, swarm-torch-core/src/lib.rs, swarm-torch-core/Cargo.toml | Partial | `no_std + alloc` builds (`--no-default-features --features alloc`); `embedded_min` (no alloc) is planned; end-to-end embedded example pending |
| GPU | WGPU-first; CUDA optional backend | Evidence: ADR-0015 | Planned | Backend wiring pending |
| Python | Optional, bounded Python interop | Evidence: ADR-0009 | Planned | Keep separate crate boundary |
| Supply chain | Dependency/license/source policy configs | Evidence: SECURITY.md, deny.toml, supply-chain/ | Partial | Configs exist; CI enforcement planned |

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
- Any system that requires “always-online” assumptions as a correctness dependency

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

TODO: define which assumptions are normative vs profiled (timeouts, quorum ratios, staleness thresholds).

### Coordination Model (Design Target vs Current)

**Design target:** gossip-based eventual consistency coordination with quorum-based rounds (not PBFT/Tendermint-class consensus).

Evidence: ADR-0006A

**Current implementation:** data structures exist (membership view, votes), but no end-to-end coordinator exists yet.

Evidence: swarm-torch-core/src/consensus.rs

### Transport Model (Current)

SwarmTorch defines a transport abstraction with explicit capabilities (reliability class, bandwidth class, message size, multicast support).

Evidence: swarm-torch-net/src/traits.rs

Current status:

- `MockTransport`/`MockNetwork` exist for testing/scaffolding.
- Real TCP/UDP/BLE/LoRa transports are placeholders (feature flags exist, implementations do not).

Evidence: swarm-torch-net/src/mock.rs, swarm-torch-net/Cargo.toml

### Topology and Staleness (Design Target vs Current)

**Design target:** topology is a first-class, measurable policy; bounded asynchrony uses explicit staleness policies.

Evidence: ADR-0004A, ADR-0004B

**Current implementation:** a basic `Topology` enum exists, but adaptive policies and staleness policy implementation are not present.

Evidence: swarm-torch-core/src/algorithms.rs

## Threat Model and Trust Boundaries (SwarmTorch-Only)

### Threats (Design Target)

SwarmTorch’s baseline threat model includes:

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

This chapter defines the SwarmTorch “spine”: a unified run DAG plus persisted artifacts that power notebook views and future UIs.

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

For `edge_std` coordinators/gateways, Parquet/Arrow is the preferred “facts at rest” format:

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
- Backends: `burn-backend`, `tch-backend`  
  Evidence: swarm-torch/Cargo.toml
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
- `embassy` runtime wrapper (placeholder; not yet complete)

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

Planned (not yet implemented in code): Bulyan, robust aggregation harness, and rejection/telemetry integration.

Evidence: ADR-0007, ADR-0007A

### Compression (Current)

Compression primitives include `TopK`, `RandomSparse`, and `Quantized` modes.

Evidence: swarm-torch-core/src/compression.rs

TODO: specify default compression profiles and convergence/accuracy expectations.

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

TODO: map threat model requirements to concrete enforcement hooks (protocol parsing fuzzing, signature verification, byzantine harness, supply-chain gates).

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

TODO: finalize cache/materialization semantics (cache keys, invalidation, execution profiles) and publish JSON schema docs.

Evidence: ADR-0017.

Current status: minimal `graph.json` schema types exist (with `node_id` derived from `node_key`, and `node_def_hash = sha256(postcard(NodeDefCanonicalV1))`). Execution and materialization emitters are planned.

Evidence: swarm-torch-core/src/run_graph.rs

## Developer Experience (DX)

### Notebook-Native Views (Phase 1)

Implemented (initial): standalone artifact reader that generates a self-contained `report.html` from a run bundle. Notebooks can render the HTML as a pure reader workflow.

Evidence: ADR-0016.
Evidence: swarm-torch/src/report.rs, swarm-torch/src/bin/swarm_torch_report.rs

### Local Dashboard (Phase 2)

TODO: local server that reads artifacts (no tracking DB by default).

Evidence: ADR-0016.

### Desktop Wrapper (Phase 3)

TODO: Tauri/Iced wrapper once UX stabilizes; treat UI surface as part of the threat model.

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

TODO: publish the exact CI workflow and conformance mapping in Appendix B.

Evidence: ADR-0011, ADR-0013, ADR-0014, `SECURITY.md`.

## Integration Profiles

### Swarmic Network Provider Profile (Secondary)

TODO: define integration-only mapping without importing mission-plane semantics into SwarmTorch core.

Evidence: `CONTEXT_SOURCES.md`.

## Roadmap and Open Problems

TODO: open research gaps, heterogeneity mitigation, compression tradeoffs, privacy roadmap, and maturity milestones.

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

This appendix lists conformance hooks that keep the white paper’s normative claims honest.

**Legend:** `Implemented` means the hook exists and can be run today. `Planned` means the hook is specified but not yet present as a script/test/CI gate in this repo.

#### Build and Basic Quality Gates

| Gate | Hook | Status | Evidence |
|------|------|--------|----------|
| Workspace builds | `cargo check --workspace` | Implemented | Evidence: Cargo.toml |
| Workspace tests | `cargo test --workspace` | Implemented | Evidence: Cargo.toml |
| Docs build | `cargo doc --workspace --no-deps` | Implemented | Evidence: Cargo.toml |
| embedded_min core build | `cargo build -p swarm-torch-core --no-default-features` | Planned | Evidence: ADR-0002 |
| no_std + alloc core build | `cargo build -p swarm-torch-core --no-default-features --features alloc` | Implemented | Evidence: swarm-torch-core/Cargo.toml |
| Embedded target build | `cargo build -p swarm-torch-core --no-default-features --features alloc --target thumbv7em-none-eabihf` | Planned | Evidence: ADR-0002 |

#### Supply-Chain and Policy Gates

| Gate | Hook | Status | Evidence |
|------|------|--------|----------|
| RustSec advisories | `cargo audit` | Planned | Evidence: SECURITY.md |
| License/source policy | `cargo deny check` | Planned | Evidence: deny.toml, SECURITY.md |
| Human audit tracking | `cargo vet` | Planned | Evidence: supply-chain/, SECURITY.md |
| Public API SemVer regression | `cargo semver-checks check-release` | Planned | Evidence: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#cargo-semver-checks |
| OSV lockfile scan | CI: `google/osv-scanner-action` (Cargo.lock) | Planned | Evidence: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#OSV-Scanner-Action |

#### CI/CD Hardening (Security)

| Gate | Hook (intended) | Status | Evidence |
|------|------------------|--------|----------|
| Least-privilege workflows | CI: explicit `permissions:` and minimal `GITHUB_TOKEN` scope | Planned | Evidence: SECURITY.md, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#GitHub-Actions-Hardening |
| Action pinning | CI: pin third-party `uses:` to full commit SHAs | Planned | Evidence: SECURITY.md, SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.sources.md#GitHub-Actions-Hardening |
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
| Manifest integrity | `cargo test -p swarm-torch artifacts::tests::bundle_manifest_roundtrip` (recompute SHA-256 for manifest entries) | Implemented | Evidence: swarm-torch/src/artifacts.rs |
| Graph hash normalization | `cargo test -p swarm-torch artifacts::tests::graph_write_normalizes_ids_and_hashes` | Implemented | Evidence: swarm-torch/src/artifacts.rs |
| Dataset fingerprint determinism | `cargo test -p swarm-torch-core dataops::tests::dataset_fingerprint_is_deterministic` | Implemented | Evidence: swarm-torch-core/src/dataops.rs |
| No raw rows by default | Assert bundle does not embed raw dataset rows unless explicitly enabled + size-limited | Planned | Evidence: ADR-0016 |
| Untrusted input marking | If dataset source is `untrusted`, artifacts mark unsafe surface | Planned | Evidence: ADR-0017 |
| Unsafe extension marking | `unsafe_extension` nodes are explicitly flagged in graph/events/materializations/run summary | Planned | Evidence: ADR-0018 |

#### Drift / Doc Hygiene (Repo-Local)

| Gate | Hook | Status | Evidence |
|------|------|--------|----------|
| No legacy docs/ links | `rg -n \"\\bdocs/\" -g\"*.md\" .` (expect only external URLs or clearly marked historical text) | Implemented | Evidence: CONTEXT_SOURCES.md |
| Evidence discipline | Manual review: any **MUST/SHOULD/MAY** has an `Evidence:` pointer nearby | Implemented | Evidence: SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.md |

### Appendix C: Artifact Schema References

TODO: publish JSON schema definitions and Parquet schema notes; keep versioned.

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
