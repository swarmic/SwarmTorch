# SwarmTorch Release Notes

This directory contains detailed release notes for SwarmTorch development.

## Purpose

These notes serve as an **audit trail** for contributors, providing:
- **What** was implemented and why
- **How** architectural decisions were realized in code
- **Context** for understanding the codebase from an engineering perspective

## Reading Order

New contributors should read these in order:

1. [`v0.1.0-alpha.1.md`](v0.1.0-alpha.1.md) — Initial scaffold and architecture foundation
2. [`v0.1.0-alpha.2.md`](v0.1.0-alpha.2.md) — Observability primitives and artifact bundle
3. [`v0.1.0-alpha.3.md`](v0.1.0-alpha.3.md) — Documentation framework, security guardrails, license migration
4. [`v0.1.0-alpha.4.md`](v0.1.0-alpha.4.md) — Technical whitepaper, release notes system, current state
5. [`v0.1.0-alpha.5.md`](v0.1.0-alpha.5.md) — DataOps wiring, ExecutionPolicy, trust propagation
6. [`v0.1.0-alpha.6.md`](v0.1.0-alpha.6.md) — Contract hardening, cache-hit detection, OpRunner, report derivations
7. [`v0.1.0-alpha.6x.md`](v0.1.0-alpha.6x.md) — Descriptor sanitization, unsafe-reason taxonomy, timeline/report reason rendering
8. [`M4-01_ed25519_signatures.md`](M4-01_ed25519_signatures.md) — Ed25519 envelope authentication baseline
9. [`M4-02_replay_protection_enforcement.md`](M4-02_replay_protection_enforcement.md) — Replay enforcement baseline
10. [`M4-02.5_replay_protection_hardening.md`](M4-02.5_replay_protection_hardening.md) — Replay/auth hardening follow-up
11. [`v0.1.0-alpha.6x-wave0.md`](v0.1.0-alpha.6x-wave0.md) — Wave 0 remediation: replay boundary, identity canonicalization, fail-closed hashing, clock/telemetry integrity
12. [`v0.1.0-alpha.6x-wave1.md`](v0.1.0-alpha.6x-wave1.md) — Wave 1 remediation: aggregation/compression/model safety guards, URI stripping, monotonic clock
13. [`v0.1.0-alpha.6x-wave2.md`](v0.1.0-alpha.6x-wave2.md) — Wave 2 remediation: replay configurability, compression fail-closed, node validation, descriptor bounds, SwarmConfig/Topology constructors
14. [`v0.1.0-alpha.6x-wave2.1.md`](v0.1.0-alpha.6x-wave2.1.md) — Archival bridge note restoring historical alpha.6x evidence references
15. [`v0.1.0-alpha.6x-wave3.md`](v0.1.0-alpha.6x-wave3.md) — Wave 3 remediation: identity/crypto hardening, observe bounds enforcement, trait/model API batch, mock backpressure realism
16. [`v0.1.0-alpha.6x-wave4.md`](v0.1.0-alpha.6x-wave4.md) — Wave 4 remediation convergence: TopK throughput, model RNG/capacity evolution, OpRunner error contract, error chaining, v0.2 handoff gate
17. [`v0.1.0-alpha.6x-wave5.md`](v0.1.0-alpha.6x-wave5.md) — v0.2 F-track rollout: ExecutionHint schema, transform-audit plumbing, UpdateTransform + aggregation pipeline + tracing
18. [`v0.1.0-alpha.6x-wave6.md`](v0.1.0-alpha.6x-wave6.md) — Backlog convergence: canon lock + behavior-preserving artifacts/report module decomposition (`A6X-06`)
19. [`v0.1.0-alpha.7x.md`](v0.1.0-alpha.7x.md) — Wave 7 delivery: reliability hardening, sequential scheduler MVP, typed NodeV1 deferred fields, robustness harness
20. [`v0.1.0-alpha.7x-wave1.md`](v0.1.0-alpha.7x-wave1.md) — Wave 7.1 convergence: scheduler fail-closed hardening, graph invariants, canon/release-trace reconciliation

## Versioning

We use semantic versioning with pre-release tags:
- `vX.Y.Z-alpha.N` — Early development, APIs unstable
- `vX.Y.Z-beta.N` — Feature-complete, APIs stabilizing
- `vX.Y.Z` — Stable release

## crates.io Packaging

These release notes are **repo-only** documentation (not intended to be included in the crates.io package).
