# Security Policy

## Vulnerability Disclosure

**Preferred:** GitHub private vulnerability reporting  
**Response Time:** Initial acknowledgment within 48 hours  
**Disclosure Policy:** 90-day coordinated disclosure

### Reporting Process

1. **Do NOT open public issues** for security vulnerabilities
2. Use [GitHub Security Advisories](https://github.com/swarmic/SwarmTorch/security/advisories/new) for private reporting
3. Optionally email security@swarmtorch.dev (if you prefer) with:
   - Description of vulnerability
   - Steps to reproduce
   - Affected versions
   - Suggested fix (if any)

### Scope

- `swarm-torch-*` crates and their dependencies
- Build/CI configuration
- Cryptographic implementations

---

## Current Security Posture (Reality Check)

SwarmTorch is pre-1.0 and actively building out its security posture. Core security primitives are implemented:

- **Message Authentication:** ✅ Ed25519 signatures (M4-01)
- **Sender Identity Contract:** ✅ `MessageEnvelope.sender` is raw Ed25519 public key bytes (ADR-0008B)
- **Replay Protection:** ✅ Sequence monotonicity and timestamp validation (M4-02)
- **Gradient Validation:** ✅ Bounds checking and NaN/Inf detection

**Known Limitations:**

- Replay cache is **memory-only** (non-persistent). Node restart resets the cache, creating a brief replay window (≤ max_clock_skew_secs, default 60s).
- Transport security (TLS/DTLS) is planned but not yet implemented.
- Sybil resistance requires external identity providers (see ADR-0008A).

See `SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.md` for the complete threat model and implementation status.

## Supply-Chain Security

### Dependency Auditing

**Planned CI gates:** once CI is added, we intend to enforce the following on every PR:

| Tool | Purpose | Config |
|------|---------|--------|
| `cargo audit` | RustSec advisory database | Built-in |
| `cargo deny` | License/ban/advisory enforcement | [`deny.toml`](deny.toml) |
| `cargo vet` | Human audit tracking | [`supply-chain/`](supply-chain/) |

**Current (today):** configs exist, but CI enforcement is not yet implemented. Maintainers should run the gates locally before merging:

```bash
cargo deny check
cargo audit
cargo vet
```

### Policy

- **No unvetted dependencies** may be merged without explicit sign-off
- **License policy violations** are blocked per [`deny.toml`](deny.toml) (project license is MPL-2.0; strong copyleft dependencies like GPL/AGPL remain disallowed)

---

## CI/CD Hardening

### Workflow Permissions

**Planned CI default:**

```yaml
permissions:
  contents: read        # Default least-privilege
  id-token: write       # Required for attestations
  attestations: write   # Required for artifact signing
```

### Action Pinning

All third-party actions pinned to full-length commit SHAs, not tags.

---

## Release Integrity

### Artifact Attestations

**Planned release posture:** releases should include Sigstore-backed attestations:
- Build provenance (SLSA Level 2+)
- Dependency manifest snapshot
- Reproducible build verification

### Verification

```bash
gh attestation verify swarm-torch-*.tar.gz --owner swarmic
```

---

## Docs.rs Constraints

Docs.rs builds are sandboxed:
- **Network access blocked** — no build-time fetching
- **Constrained environment** (CPU/memory/time), so docs must build reliably
- Build scripts and proc-macros run under these constraints and must not assume network access or arbitrary filesystem writes

Our crates must build documentation without network dependencies.

---

## Unsafe Code Policy

All SwarmTorch crates currently set `#![forbid(unsafe_code)]`.

If/when `unsafe` is introduced (FFI, SIMD, sandboxing hosts), it must be:

- isolated behind a narrow crate boundary or module
- documented with a safety rationale
- accompanied by tests and (where relevant) fuzzing

All `unsafe` blocks require safety documentation.
