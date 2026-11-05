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


## CenteredClipping-ICML21
- Title: Learning from History for Byzantine Robust Optimization
- URL: https://proceedings.mlr.press/v139/karimireddy21a.html
- Type: Paper (ICML 2021)
- Used for: ADR-0019 / WP `CenteredClipping` (`CClip`) algorithm and worker momentum defense against time-coupled Byzantine attacks in O(n * d) time and O(d) memory
- Notes: Proves memoryless aggregators fail against time-coupled attacks; introduces iterative L2 centered clipping relative to previous aggregate state.
- Last checked: 2026-11-01

## Bucketing-ICLR22
- Title: Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing
- URL: https://openreview.net/forum?id=jXKKDEi5vJt
- Type: Paper (ICLR 2022)
- Used for: ADR-0019 / WP pre-aggregation `s-Bucketing` (`BucketingConfig`) to mitigate Non-IID variance before Krum/Bulyan/CenteredClipping
- Notes: Demonstrates standard Byzantine aggregators fail on heterogeneous data and proves random s-bucketing reduces cross-client variance by 1/s.
- Last checked: 2026-11-01

## EF21-NeurIPS21
- Title: EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback
- URL: https://proceedings.neurips.cc/paper/2021/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html
- Type: Paper (NeurIPS 2021)
- Used for: ADR-0021 / WP `ErrorFeedbackCompressor` stateful compression convergence guarantees without bounded-gradient assumptions
- Notes: Compresses local state differences c_t^i = C(g_t^i - s_t^i), guaranteeing O(1/T) smooth non-convex convergence for contractive compressors like TopK.
- Last checked: 2026-11-01

## GAS-ICML23
- Title: Byzantine-Robust Learning on Heterogeneous Data via Gradient Splitting
- URL: https://proceedings.mlr.press/v202/liu23am.html
- Type: Paper (ICML 2023)
- Used for: ADR-0021 / WP compressed-domain & high-dimensional heterogeneous Byzantine defense (`SharedSeedSparse`)
- Notes: Addresses the curse of dimensionality and sparse support mismatch when combining gradient sparsification with Byzantine-robust aggregation.
- Last checked: 2026-11-01

## FedAsync-2019
- Title: Asynchronous Federated Optimization
- URL: https://arxiv.org/abs/1903.03934
- Type: Paper
- Used for: ADR-0023 / WP `StalenessPolicy` and `StalenessSchedule` (Constant, Polynomial, Hinge decay functions)
- Notes: Formalizes staleness-decayed mixing alpha_t = alpha * s(t - tau) for asynchronous edge updates.
- Last checked: 2026-11-01

## LoRA-ICLR22
- Title: LoRA: Low-Rank Adaptation of Large Language Models
- URL: https://openreview.net/forum?id=nZeVKeeFYf9
- Type: Paper (ICLR 2022)
- Used for: ADR-0024 / WP `TrainableParamFilter::PrefixAllowlist` adapter-only delta synchronization
- Notes: Freezes base model weights and synchronizes low-rank update matrices A, B to reduce communication payload by 10x-100x.
- Last checked: 2026-11-01

## Safetensors
- Title: Safetensors Rust Crate & Format Specification (0.5.3 vs 0.7.0 / 0.8.0)
- URL: https://github.com/safetensors/safetensors/blob/v0.8.0/safetensors/Cargo.toml
- Type: Crate / Format Spec
- Used for: ADR-0024 / WP `SwarmTorchCheckpointV1` profile over additive `safetensors` (`0.7+` / `0.8+`)
- Notes: In 0.5.3, enabling both `std` and `alloc` triggers a `compile_error!`; in 0.7.0/0.8.0, `alloc` is additive (always available, `alloc` retained as no-op compatibility feature) with declared MSRV 1.80. Upstream format permits NaN/Inf, so `SwarmTorchCheckpointV1` enforces explicit finiteness and length-delimited schema checks before transactional model application.
- Last checked: 2026-11-02

## Wasmi
- Title: Wasmi Deterministic WebAssembly Interpreter (`wasmi` 2.0.0 Store & StoreLimitsBuilder)
- URL: https://docs.rs/wasmi/2.0.0/wasmi/struct.Store.html
- Type: Crate / Official Docs
- Used for: ADR-0025 / WP `ExecutionTrust::SandboxedExtension` deterministic execution with `Config::consume_fuel(true)`, `StoreLimitsBuilder::trap_on_grow_failure(true)`, and `store.limiter(|s| &mut s.limits)` installed before instantiation
- Notes: `wasmi 2.0.0` declares MSRV 1.86 and separates `validate` and `deterministic` features; constructing `StoreLimits` without calling `store.limiter(...)` does not enforce memory/table caps.
- Last checked: 2026-11-02

## Zenoh
- Title: Eclipse Zenoh: Zero Overhead Pub/Sub, Store/Query, and Compute Protocol (`zenoh` 1.10.2 & `rmw_zenoh`)
- URL: https://github.com/ros2/rmw_zenoh/blob/rolling/docs/design.md
- Type: Protocol Spec / Rust Crate (`zenoh`)
- Used for: ADR-0022 / WP `swarm-torch-transport-zenoh` optional adapter crate
- Notes: Routing topics do not establish contributor authentication; native ROS 2 `rmw_zenoh` uses CDR serialization and type-hash key expressions distinct from SwarmTorch's signed wrapper carrier topics.
- Last checked: 2026-11-02

## Iroh
- Title: Iroh: Peer-to-Peer QUIC Networking (`iroh` 1.4.1 `EndpointId` & `Accepting` 0-RTT/0.5-RTT Security Notes)
- URL: https://docs.rs/iroh/latest/iroh/endpoint/struct.Accepting.html
- Type: Rust Crate (`iroh`)
- Used for: ADR-0022 / WP `swarm-torch-transport-iroh` optional adapter crate
- Notes: `iroh 1.4.1` declares Rust 1.91 (edition 2024), exposes verified TLS peer identity as `EndpointId` (kept distinct from SwarmTorch originator `PeerId`), and requires completed 1-RTT handshake enforcement so 0-RTT/0.5-RTT early data cannot mutate replay, EF, model, or DP state.
- Last checked: 2026-11-02

## Rust-TotalCmp
- Title: Rust `core::primitive::f32::total_cmp` (IEEE 754-2008 `totalOrder` predicate)
- URL: https://doc.rust-lang.org/core/primitive.f32.html#method.total_cmp
- Type: Rust Standard Library Specification
- Used for: ADR-0019 / WP deterministic total ordering in coordinate/distance sorting
- Notes: Stabilized in Rust 1.62; establishes canonical total ordering across all float bit patterns (paired with mandatory pre-sort `.is_finite()` validation).
- Last checked: 2026-11-02

## RoSDHB-AISTATS26
- Title: Reconciling Communication Compression and Byzantine-Robustness in Distributed Learning (RoSDHB)
- URL: https://proceedings.mlr.press/v300/gupta26a.html
- Type: Paper (AISTATS 2026)
- Used for: ADR-0021 / WP benchmark baseline for coordinated sparse compression + momentum + Byzantine aggregation (`W11-02`)
- Notes: Together with Li et al. (AISTATS 2026, `https://proceedings.mlr.press/v300/li26e.html`, Byz-DM21 / Byz-VR-DM21), establishes exact theoretical assumptions required when composing compression and Byzantine-robust aggregation.
- Last checked: 2026-11-02

## Bareilles-MultiKrum26
- Title: Byzantine Machine Learning: MultiKrum and an Optimal Notion of Robustness
- URL: https://arxiv.org/html/2602.03899v1
- Type: Paper (2026)
- Used for: ADR-0019 / WP comparison baseline for Multi-Krum neighborhood definitions vs. Blanchard et al. (NeurIPS 2017) one-pass `k = n - f - 2` scoring
- Notes: Highlights why the exact Multi-Krum neighborhood and `1 <= m <= n - f - 2` domain must be pinned explicitly.
- Last checked: 2026-11-02

## FLoRA-NeurIPS24
- Title: FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations
- URL: https://papers.nips.cc/paper_files/paper/2024/hash/28312c9491d60ed0c77f7fff4ad86dd1-Abstract-Conference.html
- Type: Paper (NeurIPS 2024)
- Used for: ADR-0024 / WP exact LoRA factor aggregation semantics (`W12-01`)
- Notes: Proves `mean(B) * mean(A) != mean(B * A)` cross-term interference in naive Federated LoRA averaging and introduces stacking-based exact aggregation.
- Last checked: 2026-11-02

## RFC-8613-9175-4303-8085
- Title: IETF RFC 8613 §7.5 (OSCORE Replay Recovery), RFC 9175 (Echo/Freshness Across Reboot), RFC 4303 (IPsec ESP Anti-Replay Atomic Commit), and RFC 8085 (UDP Usage Guidelines)
- URL: https://www.rfc-editor.org/rfc/rfc8613.html
- Type: IETF Standards Track RFCs
- Used for: ADR-0020 / ADR-0022 restart recovery via fresh receiver-bound challenge context, clock continuity across reboot, pre-commit signature verification, and UDP congestion/MTU guidelines
- Notes: Grounds the 6-stage atomic receive pipeline and `RecoveryRequired` fresh handshake state after replay window loss.
- Last checked: 2026-11-02

## RUSTSEC-2026-0253
- Title: RustSec Advisory `RUSTSEC-2026-0253` (`lru` panic safety in `pop`) & `RUSTSEC-2026-0002` (`lru::IterMut` aliasing unsoundness)
- URL: https://rustsec.org/advisories/RUSTSEC-2026-0253.html
- Type: Security Advisory
- Used for: ADR-0020 / `P2-11a` retirement of `lru` crate in favor of safe expiry-indexed `BTreeMap`
- Notes: Both advisories are retired when `lru` is removed from `Cargo.toml` and `Cargo.lock` in `W9-02`.
- Last checked: 2026-11-02
