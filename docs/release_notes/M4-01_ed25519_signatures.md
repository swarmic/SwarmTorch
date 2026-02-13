# SwarmTorch M4-01: Ed25519 Signature Authentication

**Date:** Historical (implemented prior to M4-02; this note was reconstructed on 2026-02-13)  
**Type:** Protocol Security Hardening (Message Authentication)

## Summary

M4-01 established cryptographic message authentication for SwarmTorch envelopes using Ed25519 signatures and domain-separated signing preimages.

This milestone delivered signature generation/verification primitives and integrated them into protocol envelope validation, creating the baseline that M4-02 later extended with replay enforcement.

## Key Changes

### 1. Ed25519 Authentication Primitives

**Files:** `swarm-torch-core/src/crypto.rs`

- Added deterministic Ed25519 signing and verification path (`MessageAuth`).
- Added explicit domain separation for envelope signing preimages.
- Added verification error discrimination for malformed signatures and invalid keys.

### 2. Envelope Signature Surface

**Files:** `swarm-torch-net/src/protocol.rs`

- Added envelope signature field and signature-aware validation path.
- Integrated signature verification with sender key material carried in envelope metadata.

### 3. Validation Hardening Tests

**Files:** `swarm-torch-core/src/crypto.rs` tests, `swarm-torch-net/tests/replay_integration.rs` (later expanded)

- Added tamper-detection and wrong-key rejection tests.
- Added deterministic signature behavior checks for fixed inputs.

## Security Posture at M4-01

What M4-01 established:

- Authenticity checks for envelope sender/payload binding.
- Clear cryptographic failure path (`Result`-based verification errors).

What was still pending at M4-01 time:

- Stateful replay enforcement for sequence/timestamp fields.

That replay gap was closed in M4-02 (`M4-02_replay_protection_enforcement.md`) and hardened in M4-02.5 (`M4-02.5_replay_protection_hardening.md`).

## Compatibility Notes

- Envelope signature storage remained `Option<Vec<u8>>` (no wire break introduced in this milestone).
- Sender identity semantics were later clarified in ADR-0008B and M4-02.5.

## Validation Commands

```bash
cargo test -p swarm-torch-core crypto
cargo test -p swarm-torch-net replay_integration
```

## Traceability

- Follow-up enforcement: `docs/release_notes/M4-02_replay_protection_enforcement.md`
- Hardening follow-up: `docs/release_notes/M4-02.5_replay_protection_hardening.md`
- Canonical contract: `ADRs.md` (ADR-0008, ADR-0008B)
