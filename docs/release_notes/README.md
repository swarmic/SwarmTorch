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

## Versioning

We use semantic versioning with pre-release tags:
- `vX.Y.Z-alpha.N` — Early development, APIs unstable
- `vX.Y.Z-beta.N` — Feature-complete, APIs stabilizing
- `vX.Y.Z` — Stable release

## crates.io Packaging

These release notes are **repo-only** documentation (not intended to be included in the crates.io package).
