# Security Policy

## Vulnerability Disclosure

**Contact:** security@swarmtorch.dev  
**Response Time:** Initial acknowledgment within 48 hours  
**Disclosure Policy:** 90-day coordinated disclosure

### Reporting Process

1. **Do NOT open public issues** for security vulnerabilities
2. Use [GitHub Security Advisories](https://github.com/swarmic/swarm-torch/security/advisories/new) for private reporting
3. Or email security@swarmtorch.dev with:
   - Description of vulnerability
   - Steps to reproduce
   - Affected versions
   - Suggested fix (if any)

### Scope

- `swarm-torch-*` crates and their dependencies
- Build/CI configuration
- Cryptographic implementations

---

## Supply-Chain Security

### Dependency Auditing

We enforce the following on every PR:

| Tool | Purpose | Config |
|------|---------|--------|
| `cargo audit` | RustSec advisory database | Built-in |
| `cargo deny` | License/ban/advisory enforcement | [`deny.toml`](deny.toml) |
| `cargo vet` | Human audit tracking | [`supply-chain/`](supply-chain/) |

### Policy

- **No unvetted dependencies** may be merged without explicit sign-off
- **Yanked crates** fail CI immediately
- **License violations** (copyleft in MIT project) are blocked

---

## CI/CD Hardening

### Workflow Permissions

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

Releases include Sigstore-backed attestations:
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

| Location | Permitted | Requirements |
|----------|-----------|--------------|
| Core crates | ❌ `#![forbid(unsafe_code)]` | No exceptions |
| FFI boundaries (PyO3, C) | ✅ Isolated modules | Safety docs, fuzzing |
| SIMD optimizations | ✅ Feature-gated | Extensive testing |

All `unsafe` blocks require safety documentation.
