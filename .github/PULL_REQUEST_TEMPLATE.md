## Contribution Route

- [ ] **Route 1 (Small Fix / Scoped Maintenance):** Links an approved issue (`Fixes #<issue-number>`).
- [ ] **Route 2 (Architectural / New Feature / Contract Change):** Links a maintainer-approved Architecture Proposal or ADR (`Closes #<issue-number>`).

**Linked Issue / Proposal:** `#`

## Architectural Contract & Invariants

- **Affected ADR(s):** `ADR-XXXX` (or `None — documentation/typo/test-only`)
- **Primary Invariant Preserved or Repaired:**
  - *State the exact safety, numerical, resource-bound, or protocol invariant this change upholds.*

## Privileged & Security-Sensitive Surface Checklist

Check any surface touched by this PR (requires explicit `@swarmic` security and supply-chain review):

- [ ] `.github/workflows/`, `Cargo.toml`, `Cargo.lock`, `deny.toml`, `supply-chain/`, `build.rs`, or proc-macros
- [ ] Cryptographic verification, `SessionId` handshake, or replay protection (`crypto.rs`, `replay.rs`, `protocol.rs`, `traits.rs`)
- [ ] Byzantine-robust aggregation or bounded wire decompression (`aggregation.rs`, `compression.rs`)
- [ ] `no_std` / `alloc` portability or `unsafe` blocks (note: `unsafe_code = "forbid"` in core/models/net)
- [ ] Artifact manifest signing, trust policy, or filesystem persistence (`swarm-torch/src/artifacts/`, `swarm-torch/src/report/`)

## Verification Evidence

- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] Unit and adversarial regression tests added/updated and passing (`cargo test`)
- [ ] `no_std` / `alloc` compatibility verified (`cargo check -p swarm-torch-core --no-default-features` and `--features alloc`)

## Contributor Attestation

- [ ] I have read [`CONTRIBUTING.md`](../CONTRIBUTING.md) and confirm this submission is my original work (or properly attributed under compatible open-source terms) and is licensed under `MPL-2.0`.
