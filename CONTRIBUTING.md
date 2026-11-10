# Contributing to SwarmTorch

Thank you for your interest in contributing to SwarmTorch!

## Governance & Contribution Disposition

SwarmTorch maintains **public, versioned contracts for its public implementation and a concise public roadmap**. Detailed strategic planning and confidential collaboration material live in a separately versioned private repository. Substantial contributions require prior scope agreement, with project email ([swarmiclabs@gmail.com](mailto:swarmiclabs@gmail.com)) available for formal collaboration. Discussion access, planning access, source write access, CI administration, and release permissions are granted independently. External contributions remain subject to isolated execution, review, and enforced merge requirements.

## Quick Links

- **Code of Conduct:** [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct)
- **Security Issues:** See [SECURITY.md](SECURITY.md) — **do not** open public issues; report via [GitHub Security Advisories](https://github.com/swarmic/SwarmTorch/security/advisories/new) or [swarmiclabs@gmail.com](mailto:swarmiclabs@gmail.com)
- **Coordinated Roadmap & Architecture Contact:** [swarmiclabs@gmail.com](mailto:swarmiclabs@gmail.com)

---

## Contribution Routes

To keep engineering publicly inspectable while preventing speculative drift, all contributions follow one of three explicit routes:

### Route 1: Small Fixes & Scoped Maintenance (Direct PR via Issue)

Use this route for:
- Bug fixes with a reproducible test case
- Documentation, typo, or broken-link fixes
- Self-contained unit/property/adversarial tests for existing public contracts
- Dependency hygiene or warning cleanup that does not alter public API or wire contracts

**Process:**
1. Open or claim an existing issue on [GitHub Issues](https://github.com/swarmic/SwarmTorch/issues) confirming the scope.
2. Submit a focused Pull Request referencing the issue (`Fixes #<id>`) and completing the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).

### Route 2: Architectural, Protocol, or New-Feature Proposals (Prior Scope Agreement Required)

Substantial work—including new public traits/types, wire format changes, cryptographic/replay changes, new aggregators/compressors, new crates/backends, or `unsafe` code—**requires maintainer scope approval before a large PR is opened**. Unapproved speculative feature PRs will be closed politely to protect maintainer review bandwidth.

**Process:**
1. Open an **Architecture Proposal** issue (or email [swarmiclabs@gmail.com](mailto:swarmiclabs@gmail.com)) answering the **5-Question Alignment Brief**:
   1. **Problem & Seam:** Which crate, module, and existing ADR contract (`ADR-0001`..`ADR-0025`) does this touch?
   2. **Invariants & Bounds:** What numerical, memory (`no_std` / `alloc`), and adversarial/fail-closed invariants must hold?
   3. **Producer $\rightarrow$ Consumer Trace:** How does data flow across admission, authentication, persistence/state transition, and consumer verification?
   4. **Target Profile Impact:** How does this affect `embedded_min`, `embedded_alloc`, and `edge_std` profiles and MSRV (`1.75` core/models/net floor)?
   5. **Verification Plan:** What deterministic unit, property, and adversarial regression tests will prove conformance?
2. Wait for a maintainer to ratify the scope and assign an ADR or roadmap slice before submitting implementation PRs.

### Route 3: Formal & Long-Horizon Collaboration

For multi-wave roadmap work, hardware/fleet integration, academic research collaboration, or commercial/enterprise inquiries, contact the maintainers at **[swarmiclabs@gmail.com](mailto:swarmiclabs@gmail.com)** with:
- Your background or organization context
- The specific SwarmTorch subsystem or roadmap wave (`ROADMAP.md`) you want to collaborate on
- A brief summary of the proposed technical contribution or integration profile

---

## Privilege Separation & Maintainer Governance

In alignment with OpenSSF supply-chain and anti-social-engineering best practices, SwarmTorch separates project capabilities into independent tiers:

1. **Public Discussion & Issue Triage:** Commenting on issues and submitting scoped proposals.
2. **Planning Access:** Granted separately for coordinated long-horizon design work; does not grant source repository write access.
3. **Source Write Access:** Protected by branch protection, mandatory `@swarmic` `CODEOWNERS` review, and passing CI gates.
4. **CI / Workflow Administration:** Changes to `.github/workflows/`, `Cargo.toml`, `Cargo.lock`, `deny.toml`, `supply-chain/`, `build.rs`, or proc-macros are treated as **privileged execution changes** requiring explicit maintainer security review.
5. **Release & Publishing Authority (`crates.io` / Signing Keys):** Held strictly by core release custodians; never delegated via issue/email requests.

> **Security Note:** Maintainer permissions are never granted in response to unsolicited urgency claims or private requests to bypass public review. All security disclosures follow [SECURITY.md](SECURITY.md).

---

## Canonical Docs & Mission-Plane Boundary (No Drift)

SwarmTorch maintains a strict public document hierarchy to prevent context drift:

- [`SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.md`](SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.md) (system model + conformance)
- [`ADRs.md`](ADRs.md) (ratified architecture decision contracts)
- [`ROADMAP.md`](ROADMAP.md) & [`CHANGELOG.md`](CHANGELOG.md) (public milestone, wave status, and release history)
- [`SECURITY.md`](SECURITY.md) (threat model, vulnerability reporting, and supply-chain gates)
- Crate source (`swarm-torch-*`) (implementation truth)

Per `ADR-0008A` (`P2-18`), SwarmTorch implements portable swarm learning and DataOps primitives (`no_std`-compatible core). External mission-plane control, remote attestation (RA) capability issuance, radio link key management, physical robot actuation interlocks, and external receipt ledgers remain outside SwarmTorch core.

If you add or change a **normative** requirement (`MUST`/`SHOULD`/`MAY`), you must update the corresponding ADR or white paper section and include an `Evidence:` pointer to the enforcing code and tests.

---

## Local Development & Verification Gates

```bash
# Clone and build
git clone https://github.com/swarmic/SwarmTorch.git
cd SwarmTorch
cargo build --workspace

# Format and lint gates (must pass with zero warnings)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings

# Full workspace test suite
cargo test --workspace

# Embedded portability checks (MSRV 1.75 compatible)
cargo check -p swarm-torch-core --no-default-features
cargo check -p swarm-torch-core --no-default-features --features alloc
cargo check -p swarm-torch-models --no-default-features
cargo check -p swarm-torch-models --no-default-features --features alloc
cargo check -p swarm-torch-net --no-default-features
cargo check -p swarm-torch-net --no-default-features --features alloc

# Supply-chain gates
cargo deny check
cargo audit
cargo vet
```

## Code Style & Engineering Standards

- **Formatting & Lints:** Follow `rustfmt` defaults and `cargo clippy --workspace --all-targets -- -D warnings`.
- **Safety:** `swarm-torch-core`, `swarm-torch-models`, and `swarm-torch-net` enforce `unsafe_code = "forbid"`.
- **Failure Meaning:** Never turn corruption, denied admission, non-finite values (`NaN`/`±Inf`), or resource exhaustion into silent truncation or empty success.
- **Hardware-Aware Bounds:** Bound queues, wire frame sizes, decompression buffers, and peer state explicitly; keep modules focused and lightweight.

## Commit Messages

Use conventional commits:
```text
feat(core): add IterativeKrumV1 Bulyan aggregation profile
fix(net): enforce receiver-bound SessionId challenge verification
docs(adr): update ADR-0020 replay retention contract
```

## License

By contributing to SwarmTorch, you agree that your contributions will be licensed under the **Mozilla Public License 2.0 (`MPL-2.0`)** (see [LICENSE](LICENSE)).
