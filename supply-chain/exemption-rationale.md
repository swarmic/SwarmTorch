# Cargo Vet Exemption Rationale Register

This file is the required rationale register for `[[exemptions.*]]` entries in
`supply-chain/config.toml`.

Policy:

1. Any PR that adds a new `[[exemptions.<crate>]]` entry must update this file.
2. Every exemption entry must include:
   - crate + version
   - criteria
   - reason for temporary exemption
   - owner
   - planned removal phase or review date
3. Exemptions are temporary controls, not permanent approval.
4. Never record `cargo-deny`, `cargo-audit`, or `cargo-vet` as passed on a host where those binaries were not executed; record CI-only verification status honestly.

## Verification Posture (Wave 16)

- **Blocking CI enforcement:** `.github/workflows/rust.yml` (`supply_chain` job) runs `cargo deny check`, `cargo audit`, and `cargo vet` against the root workspace `Cargo.lock` (plus `cargo deny --manifest-path adapters/<crate>/Cargo.toml --all-features --locked check` against `adapters/deny.toml` for all 5 standalone adapters). Hosted GitHub Actions Run `44491251480` on `SwarmTorch@3bdf06b04e4001b4459e605231b7f49149e8e5e4` passed **7/7 jobs** (`supply_chain`, `standalone_adapters`, `embedded_targets`, `test (stable / 1.81 / 1.75)`, `windows_artifact_trust`).
- **Local Windows build host status (Wave 16 `16A` / `16A-2`):** Exact CI-pinned binaries are installed and verified on the local Windows build host using CI-symmetric commands:
  - `cargo-deny 0.19.0`: `cargo deny check` (root) and all 5 `cargo deny --manifest-path adapters/<crate>/Cargo.toml --all-features --locked check` lanes exit `0` (`advisories ok, bans ok, licenses ok, sources ok`).
  - `cargo-audit 0.22.1` (`cargo-audit-audit 0.22.1`): `cargo audit --db "$env:TEMP\swarmtorch-rustsec-db"` (matching CI's isolated advisory DB path in `.github/workflows/rust.yml`) against live `rustsec/advisory-db` exits `0` (`0` vulnerabilities across 158 locked root dependencies; 12 allowed informational warnings: `pin-utils 0.1.0` `RUSTSEC-2026-0441`, `proc-macro-error2 2.0.1` `RUSTSEC-2026-0173`, `tokio 1.49.0` `RUSTSEC-2026-0386` unused `AsyncFd`, and 9 transitive `heapless 0.8.0` unsound warnings `RUSTSEC-2026-0574..0582` via optional `embassy-time`).
  - `cargo-vet 0.10.2`: `cargo vet` exits `0` (`Vetting Succeeded (6 fully audited, 147 exempted)`).
- **Wave 15–16 root supply-chain updates (no new exemptions added):**
  - `portable-atomic 1.15.0` (`Decision A6`, 32-bit `not(target_has_atomic = "64")` dependency in `swarm-torch-runtime`): covered by first-party `safe-to-deploy` audit in `supply-chain/audits.toml` (not an exemption).
  - `regex 1.12.2` (`Decision A7`, promoted to `swarm-torch` runtime dependency for JSON Schema `pattern` validation): already locked in `Cargo.lock` and covered at `safe-to-deploy` in `supply-chain/config.toml` alongside `regex-automata 0.4.13`, `regex-syntax 0.8.8`, `aho-corasick 1.1.4`, and `memchr 2.7.6`.
  - `getrandom 0.2.17` (`Wave 16A-2` / `ST-02`, `std`-only OS CSPRNG dependency in `swarm-torch-core` for certified `DpFedAvgV1` noise & Poisson cohort entropy): covered by first-party `safe-to-deploy` audit in `supply-chain/audits.toml` (not an exemption; zero new transitive dependencies beyond already-locked `cfg-if 1.0.4`, `libc 0.2.191`, and `wasi 0.11.1`).

## Exemption Entries

| ID | Crate | Version | Criteria | Reason & Wave 16 Disposition | Owner | Next Review / Removal Target |
|---|---|---|---|---|---|---|
| EX-2026-04-05-01 | `bytes` | `1.11.1` | `safe-to-deploy` | Added as immediate security lockfile remediation (`RUSTSEC-2026-0007` on `1.11.0`). **Wave 16 Review (2026-11-20) — Justified Renewal:** Re-verified `0` open RustSec advisories against `bytes 1.11.1` via local `cargo-deny 0.19.0` + `cargo-audit 0.22.1 --db` and hosted CI Run `44491251480`; `supply-chain/imports.lock` has no upstream publisher registry configured yet, so temporary exemption is renewed rather than replaced by an unverified rubber-stamp audit. | SwarmTorch maintainers (`swarmic`) | Wave 17 supply-chain upstream import configuration (`mozilla`/`google` `cargo-vet` imports) or full first-party `bytes` unsafe review |
| EX-2026-04-05-02 | `lru` | `0.12.5` | `safe-to-deploy` | **CLOSED (`W9-02` / `P2-04` / `P2-11`)**: Replaced `lru::LruCache` in `swarm-torch-core::replay` with a 100% safe, deterministic expiry-indexed `BTreeMap` replay store, eliminating `lru 0.12.5` (`RUSTSEC-2026-0002`) and 4 transitive dependencies (`hashbrown`, `foldhash`, `equivalent`, `allocator-api2`) from `Cargo.lock` and `supply-chain/config.toml`. | SwarmTorch maintainers (`swarmic`) | Closed in Wave 9 (`W9-02`) |
| EX-2026-04-05-03 | `bincode` | `2.0.0-rc.3` | `safe-to-deploy` | **CLOSED (`W9-05` / `P2-11`)**: Removed unused `burn = "0.14"` and `burn-ndarray = "0.14"` from workspace and `swarm-torch-models`, eliminating `bincode 2.0.0-rc.3` and 57 other transitive crates from `Cargo.lock`, `supply-chain/config.toml`, and `deny.toml` (`RUSTSEC-2025-0141` retired). | SwarmTorch maintainers (`swarmic`) | Closed in Wave 9 (`W9-05`) |
| EX-2026-11-05-01 | `crossbeam-epoch` | `0.9.21` | `safe-to-deploy` | Updated from `0.9.18` to `0.9.21` to remediate `RUSTSEC-2026-0204` (`fmt::Pointer` null-pointer dereference). **Wave 16 Review (2026-11-20) — Justified Renewal:** Confirmed `0` open RustSec advisories and dev/benchmark-only reachability (`criterion` -> `rayon` -> `crossbeam-deque`, absent from all runtime crates) via `cargo tree` / `cargo-audit --db` / `cargo-vet`; renewed pending Wave 17 upstream `cargo-vet` import setup. | SwarmTorch maintainers (`swarmic`) | Wave 17 upstream `cargo-vet` import configuration (or `safe-to-run` scope narrowing) |
| EX-2026-11-14-01 | `rustix` (`1.1.6`), `linux-raw-sys` (`0.12.1`), `errno` (`0.3.15`), `bitflags` (`2.13.2`), `libc` (`0.2.191`) | `1.1.6` / `0.12.1` / `0.3.15` / `2.13.2` / `0.2.191` | `safe-to-deploy` | Added transitive dependencies (and minor `libc` bump) required by audited `fs4 1.1.0` synchronous kernel file locking (`ST-05`, Phase `13B-2`). **Wave 16 Review (2026-11-20) — Justified Renewal:** Re-verified `0` open RustSec advisories across all 5 crates via local `cargo-deny 0.19.0` + `cargo-audit 0.22.1 --db` + `cargo-vet 0.10.2` and hosted CI Run `44491251480`; full syscall-layer FFI bindings are too large for superficial first-party review and await upstream `cargo-vet` publisher imports (`bytecode-alliance` / `mozilla` / `google`). | SwarmTorch maintainers (`swarmic`) | Wave 17 upstream `cargo-vet` import configuration (`bytecode-alliance` / `mozilla` / `google`) |
