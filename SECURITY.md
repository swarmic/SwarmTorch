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
- Replay cache uses bounded LRU capacity. Under extreme peer churn, an evicted peer can be re-admitted as a fresh replay state entry.
- Transport security (TLS/DTLS) is planned but not yet implemented.
- Sybil resistance requires external identity providers (see ADR-0008A).

**Operational controls for replay limitations (current wave policy):**

- Treat process restart as a replay-boundary event and avoid rejoining an untrusted mesh until `max_clock_skew_secs` has elapsed.
- Keep sender sequence generation monotonic per peer key and avoid sequence resets across rolling restarts.
- Use capacity sizing (`ReplayProtection::try_with_config`) appropriate for expected active-peer cardinality to reduce LRU-eviction replay exposure.

**SecurityConfig enforcement scope (current reality):**

- `swarm_torch_core::crypto::SecurityConfig` is currently a configuration surface, not a runtime enforcement hook in `swarm-torch-net` verifier flow.
- Do not treat `SecurityConfig` fields as active policy toggles for transport auth/replay decisions in this release.
- Enforcement wiring is tracked as near-term follow-on work in `docs/plans/v0.1.0-alpha.8x-phase2-remediation-roadmap.md`.

See `SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.md` for the complete threat model and implementation status.

## Supply-Chain Security

### Dependency Auditing

CI now includes a dedicated supply-chain job. Wave 8.1 policy:

| Tool | Purpose | Config |
|------|---------|--------|
| `cargo audit` | RustSec advisory database | Built-in |
| `cargo deny` | License/ban/advisory enforcement | [`deny.toml`](deny.toml) |
| `cargo vet` | Human audit tracking | [`supply-chain/`](supply-chain/) |

Enforcement mode:
- `cargo deny`: blocking in CI
- `cargo audit`: blocking in CI
- `cargo vet`: advisory for one wave (Wave 8.1), then promoted to blocking once audit baseline is stabilized

Pinned tool baseline for Wave 8.1 CI:
- `cargo-deny` `0.19.0`
- `cargo-audit` `0.22.1`
- `cargo-vet` `0.10.2`

Current `cargo audit` state at Wave 8.1:
- Vulnerability failures: none (blocking gate passes).
- Non-failing warnings remain on transitive dependencies (`bincode 2.0.0-rc.3` unmaintained via `burn`, `lru 0.12.5` unsound warning).
- `cargo deny` tracks `RUSTSEC-2025-0141` as an explicit temporary ignore in `deny.toml` with Wave 8.2 closure target (`P2-11`).
- `cargo vet` baseline now includes first-party audits for critical crates (`ed25519-dalek`, `sha2`, `postcard`) and reduced exemption count from the initial bootstrap.

Maintainers should still run the following locally before merging:

```bash
cargo deny check
cargo audit
cargo vet
```

Exemption governance:
- New `cargo vet` exemptions must include explicit rationale in [`supply-chain/exemption-rationale.md`](supply-chain/exemption-rationale.md).
- PRs that add `[[exemptions.*]]` without rationale updates are rejected by CI policy.
- Exemptions must include an owner and removal target.

### Policy

- **No unvetted dependencies** may be merged without explicit sign-off
- **License policy violations** are blocked by [`deny.toml`](deny.toml) checks in CI (project license is MPL-2.0; strong copyleft dependencies like GPL/AGPL remain disallowed).

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

Third-party actions in `.github/workflows/rust.yml` are pinned to full-length commit SHAs, not tags.

### Cross-Platform Trust-Boundary Checks

CI now includes a dedicated Windows lane that runs artifact manifest trust-boundary tests (`validate_manifest_*`) and report-loader trust checks (`load_report_*`) to reduce platform-specific path-validation blind spots.

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
