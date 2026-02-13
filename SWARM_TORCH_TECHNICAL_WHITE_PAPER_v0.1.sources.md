# SwarmTorch Technical White Paper v0.1 - Sources Ledger

**Purpose:** Keep citations stable across multi-shot drafting. This file is the long-lived bibliography and “why we cite” ledger for `SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.md`.

## How To Use This Ledger

- Prefer primary sources (RFCs, official docs, peer-reviewed papers).
- For each entry, include what it supports in the WP (chapter/claims).
- Record the last-checked date (web research pass will update).

### Entry Template

```text
## <ID>
- Title:
- URL:
- Type: RFC / spec / official docs / paper / survey
- Used for:
- Notes (what this source does and does not claim):
- Last checked: YYYY-MM-DD
```

## Normative Keywords and Versioning

## RFC-2119
- Title: Key words for use in RFCs to Indicate Requirement Levels
- URL: https://www.rfc-editor.org/rfc/rfc2119
- Type: RFC
- Used for: WP "Normative Language"
- Notes: Defines MUST/SHOULD/MAY semantics.
- Last checked: 2026-02-08

## RFC-8174
- Title: Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words
- URL: https://www.rfc-editor.org/rfc/rfc8174
- Type: RFC
- Used for: WP "Normative Language"
- Notes: Clarifies uppercase interpretation rules.
- Last checked: 2026-02-08

## SemVer-2.0.0
- Title: Semantic Versioning 2.0.0
- URL: https://semver.org/
- Type: Spec
- Used for: WP SemVer policy, release gates
- Notes: Versioning contract for public APIs.
- Last checked: 2026-02-08

## Rust Ecosystem Production Norms

## Rust-API-Guidelines
- Title: Rust API Guidelines
- URL: https://rust-lang.github.io/api-guidelines/
- Type: Official docs
- Used for: WP API/docs contract, examples/errors/panics conventions
- Notes: Style and stability expectations in Rust ecosystem.
- Last checked: 2026-02-08

## Cargo-Features
- Title: The Cargo Book (Features / Feature Unification)
- URL: https://doc.rust-lang.org/cargo/reference/features.html
- Type: Official docs
- Used for: WP feature-flag rules ("features must be additive")
- Notes: Downstream builds use the union of enabled features; avoid breaking changes via features.
- Last checked: 2026-02-08

## Cargo-Rust-Version
- Title: The Cargo Book (`rust-version`)
- URL: https://doc.rust-lang.org/stable/cargo/reference/rust-version.html
- Type: Official docs
- Used for: WP MSRV policy
- Notes: How MSRV is communicated/enforced for crates.
- Last checked: 2026-02-08

## Docs-RS
- Title: docs.rs documentation (build environment constraints)
- URL: https://docs.rs/about/builds
- Type: Official docs
- Used for: WP docs.rs constraints and metadata guidance
- Notes: Builds run in a sandbox with explicit resource limits; network access is blocked and not enabled for crates.
- Last checked: 2026-02-08

## Docs-RS-Metadata
- Title: docs.rs documentation (package.metadata.docs.rs)
- URL: https://docs.rs/about/metadata
- Type: Official docs
- Used for: WP docs.rs metadata configuration guidance
- Notes: How to configure docs.rs builds (features/targets/rustdoc args).
- Last checked: 2026-02-08

## Security, Supply Chain, and Provenance

## RustSec-Advisory-DB
- Title: RustSec Advisory Database
- URL: https://github.com/rustsec/advisory-db
- Type: Official database repo
- Used for: WP dependency audit posture
- Notes: Vulnerability advisories for Rust crates.
- Last checked: 2026-02-08

## cargo-audit
- Title: cargo-audit
- URL: https://github.com/rustsec/rustsec/tree/main/cargo-audit
- Type: Tool docs
- Used for: WP CI gates (RustSec scanning)
- Notes: RustSec advisory checks.
- Last checked: 2026-02-08

## cargo-deny
- Title: cargo-deny
- URL: https://embarkstudios.github.io/cargo-deny/
- Type: Tool docs
- Used for: WP license/source policy and advisories gate
- Notes: Licenses/bans/advisories/sources enforcement.
- Last checked: 2026-02-08

## cargo-vet
- Title: cargo-vet
- URL: https://github.com/mozilla/cargo-vet
- Type: Tool docs
- Used for: WP human audit tracking gate
- Notes: Tracks review/audit for dependencies and diffs.
- Last checked: 2026-02-08

## OSV-Scanner-Action
- Title: OSV-Scanner GitHub Action
- URL: https://google.github.io/osv-scanner/github-action/
- Type: Official docs
- Used for: WP complementary OSV-based lockfile scanning
- Notes: OSV is broader than RustSec and can be used as an additional signal.
- Last checked: 2026-02-08

## GitHub-Actions-Hardening
- Title: Security hardening for GitHub Actions
- URL: https://docs.github.com/en/actions/how-tos/security-for-github-actions/security-guides/security-hardening-for-github-actions
- Type: Official docs
- Used for: WP CI hardening requirements (least privilege, pin actions to SHAs)
- Notes: Pinning third-party actions to a full-length commit SHA is the only immutable reference; recommends least-privilege `GITHUB_TOKEN` permissions and careful handling of untrusted inputs.
- Last checked: 2026-02-08

## OpenSSF-Scorecard
- Title: OpenSSF Scorecard
- URL: https://scorecard.dev/
- Type: Official docs
- Used for: WP repo posture targets
- Notes: Measures security best practices; not a proof of security.
- Last checked: 2026-02-08

## SLSA
- Title: SLSA (Supply-chain Levels for Software Artifacts)
- URL: https://slsa.dev/
- Type: Spec
- Used for: WP release provenance expectations
- Notes: Provenance levels for build pipelines.
- Last checked: 2026-02-08

## Sigstore
- Title: Sigstore
- URL: https://www.sigstore.dev/
- Type: Official docs
- Used for: WP signing/verification posture
- Notes: Keyless signing and transparency log concepts.
- Last checked: 2026-02-08

## GitHub-Attestations
- Title: GitHub Artifact Attestations
- URL: https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations
- Type: Official docs
- Used for: WP provenance implementation guidance + required workflow permissions
- Notes: Documents generating build provenance/SBOM attestations and the workflow permission requirements (notably `id-token: write` and `attestations: write`).
- Last checked: 2026-02-08

## cargo-semver-checks
- Title: cargo-semver-checks
- URL: https://github.com/obi1kenobi/cargo-semver-checks
- Type: Tool docs
- Used for: WP release gates (public API compatibility)
- Notes: Detects SemVer-breaking changes in Rust public APIs.
- Last checked: 2026-02-08

## cargo-fuzz
- Title: cargo-fuzz
- URL: https://github.com/rust-fuzz/cargo-fuzz
- Type: Tool docs
- Used for: WP fuzzing gates (protocol parsing, serialization)
- Notes: libFuzzer integration for Rust.
- Last checked: 2026-02-08

## Rust-Fuzz-Book
- Title: The Rust Fuzz Book
- URL: https://rust-fuzz.github.io/book/
- Type: Official docs
- Used for: WP fuzzing guidance and operational approach
- Notes: Practical guidance for adopting fuzzing in Rust projects.
- Last checked: 2026-02-08

## Observability and Data Lineage

## OpenTelemetry
- Title: OpenTelemetry Specification (overview)
- URL: https://opentelemetry.io/docs/specs/otel/
- Type: Spec / official docs
- Used for: WP signal model concepts (traces/metrics/logs) and exporter separation
- Notes: SwarmTorch is OTel-compatible, not OTel-dependent; we adopt only select concepts.
- Last checked: 2026-02-08

## OTel-SemConv
- Title: OpenTelemetry Semantic Conventions
- URL: https://opentelemetry.io/docs/specs/semconv/
- Type: Spec / official docs
- Used for: WP naming alignment guidance (optional mapping)
- Notes: Conventions for attribute names and meanings; SwarmTorch exporters may map to these.
- Last checked: 2026-02-08

## OTel-Trace-API
- Title: OpenTelemetry Trace API
- URL: https://opentelemetry.io/docs/specs/otel/trace/api/
- Type: Spec / official docs
- Used for: WP ID sizing/format grounding (trace_id/span_id conventions)
- Notes: Defines trace/span concepts and ID requirements in OTel.
- Last checked: 2026-02-08

## W3C-Trace-Context
- Title: W3C Trace Context
- URL: https://www.w3.org/TR/trace-context/
- Type: Spec
- Used for: WP correlation IDs for distributed runs
- Notes: traceparent/tracestate header model.
- Last checked: 2026-02-08

## OpenLineage
- Title: OpenLineage
- URL: https://openlineage.io/
- Type: Spec / official docs
- Used for: WP lineage artifact conventions
- Notes: Standard lineage concepts; SwarmTorch may adopt concepts without adopting full stack.
- Last checked: 2026-02-08

## Arrow
- Title: Apache Arrow
- URL: https://arrow.apache.org/
- Type: Spec / official docs
- Used for: WP "facts at rest" columnar formats
- Notes: In-memory columnar format and ecosystem.
- Last checked: 2026-02-08

## Parquet
- Title: Apache Parquet
- URL: https://parquet.apache.org/
- Type: Spec / official docs
- Used for: WP metrics/events/materializations storage format
- Notes: Columnar on-disk format optimized for analytics reads.
- Last checked: 2026-02-08

## Federated Learning and Robust Aggregation (Primary Research)

## FedAvg
- Title: Communication-Efficient Learning of Deep Networks from Decentralized Data
- URL: https://proceedings.mlr.press/v54/mcmahan17a.html
- Type: Paper
- Used for: WP baseline FL framing and assumptions
- Notes: Canonical FedAvg reference.
- Last checked: 2026-02-08

## Krum
- Title: Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
- URL: https://proceedings.neurips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
- Type: Paper
- Used for: WP robust aggregation reference point
- Notes: Krum selection under Byzantine gradients.
- Last checked: 2026-02-08

## Bulyan
- Title: The Hidden Vulnerability of Distributed Learning in Byzantium
- URL: https://proceedings.mlr.press/v80/mhamdi18a.html
- Type: Paper
- Used for: WP robust aggregation + high-dimensional limitations discussion
- Notes: Bulyan and limitations of convergence-only claims.
- Last checked: 2026-02-08

## SecureAggregation
- Title: Practical Secure Aggregation for Privacy-Preserving Machine Learning
- URL: https://eprint.iacr.org/2017/281
- Type: Paper
- Used for: WP privacy roadmap (secure aggregation)
- Notes: Dropout-robust secure sum for FL.
- Last checked: 2026-02-08

## FL-Survey-Kairouz
- Title: Advances and Open Problems in Federated Learning
- URL: https://arxiv.org/abs/1912.04977
- Type: Survey / monograph preprint
- Used for: WP "why this is hard" framing, system model and threat landscape grounding
- Notes: Comprehensive survey/monograph; later published in Foundations and Trends in Machine Learning (doi:10.1561/2200000083).
- Last checked: 2026-02-08

## FedProx
- Title: FedProx: Federated Optimization in Heterogeneous Networks
- URL: https://arxiv.org/abs/1812.06127
- Type: Paper
- Used for: WP heterogeneity mitigation roadmap
- Notes: Addresses statistical/system heterogeneity via proximal term.
- Last checked: 2026-02-08

## SCAFFOLD
- Title: SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
- URL: https://proceedings.mlr.press/v119/karimireddy20a.html
- Type: Paper
- Used for: WP client drift mitigation roadmap
- Notes: Control variates to reduce client drift.
- Last checked: 2026-02-08

## Compression-QSGD
- Title: QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding
- URL: https://arxiv.org/abs/1610.02132
- Type: Paper
- Used for: WP update compression families
- Notes: Quantization with convergence guarantees.
- Last checked: 2026-02-08

## Compression-DGC
- Title: Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training
- URL: https://arxiv.org/abs/1712.01887
- Type: Paper
- Used for: WP sparsification + correction families
- Notes: Momentum correction and thresholding for high compression.
- Last checked: 2026-02-08
