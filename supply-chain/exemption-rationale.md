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

## Verification Posture (Wave 15)

- **Blocking CI enforcement:** `.github/workflows/rust.yml` (`supply_chain` job) runs `cargo deny check`, `cargo audit`, and `cargo vet` against the root workspace `Cargo.lock` (plus `cargo deny --manifest-path adapters/<crate>/Cargo.toml --all-features --locked check` against `adapters/deny.toml` for all 5 standalone adapters).
- **Local Windows build host status:** `cargo-deny`, `cargo-audit`, and `cargo-vet` are **NOT INSTALLED — UNVERIFIED** on the local Windows build host. Prior Wave 14 table rows that stated those binaries passed locally were corrected in Wave 15 (`15E`); authoritative pass status for the Wave 15 tip commit requires the hosted GitHub Actions `supply_chain` job after push.
- **Wave 15 root supply-chain updates (no new exemptions added):**
  - `portable-atomic 1.15.0` (`Decision A6`, 32-bit `not(target_has_atomic = "64")` dependency in `swarm-torch-runtime`): covered by first-party `safe-to-deploy` audit in `supply-chain/audits.toml` (not an exemption).
  - `regex 1.12.2` (`Decision A7`, promoted to `swarm-torch` runtime dependency for JSON Schema `pattern` validation): already locked in `Cargo.lock` and covered at `safe-to-deploy` in `supply-chain/config.toml` alongside `regex-automata 0.4.13`, `regex-syntax 0.8.8`, `aho-corasick 1.1.4`, and `memchr 2.7.6`.

## Exemption Entries

| ID | Crate | Version | Criteria | Reason | Owner | Removal Target |
|---|---|---|---|---|---|---|
| EX-2026-04-05-01 | `bytes` | `1.11.1` | `safe-to-deploy` | Added as immediate security lockfile remediation (`RUSTSEC-2026-0007` on `1.11.0`) while preserving Wave 8.1 delivery. Renewed in Wave 15 pending upstream audit import; blocking advisory/vet verification runs in CI (`supply_chain` job; local binaries not installed on Windows build host). | SwarmTorch maintainers | Wave 16 supply-chain audit sweep |
| EX-2026-04-05-02 | `lru` | `0.12.5` | `safe-to-deploy` | **CLOSED (`W9-02` / `P2-04` / `P2-11`)**: Replaced `lru::LruCache` in `swarm-torch-core::replay` with a 100% safe, deterministic expiry-indexed `BTreeMap` replay store, eliminating `lru 0.12.5` (`RUSTSEC-2026-0002`) and 4 transitive dependencies (`hashbrown`, `foldhash`, `equivalent`, `allocator-api2`) from `Cargo.lock` and `supply-chain/config.toml`. | SwarmTorch maintainers | Closed in Wave 9 (`W9-02`) |
| EX-2026-04-05-03 | `bincode` | `2.0.0-rc.3` | `safe-to-deploy` | **CLOSED (`W9-05` / `P2-11`)**: Removed unused `burn = "0.14"` and `burn-ndarray = "0.14"` from workspace and `swarm-torch-models`, eliminating `bincode 2.0.0-rc.3` and 57 other transitive crates from `Cargo.lock`, `supply-chain/config.toml`, and `deny.toml` (`RUSTSEC-2025-0141` retired). | SwarmTorch maintainers | Closed in Wave 9 (`W9-05`) |
| EX-2026-11-05-01 | `crossbeam-epoch` | `0.9.21` | `safe-to-deploy` | Updated from `0.9.18` to `0.9.21` to remediate `RUSTSEC-2026-0204` (`fmt::Pointer` null-pointer dereference in dev-dependency graph via `criterion` -> `rayon` -> `crossbeam-deque`). Dev/benchmark-only scope verified via `cargo metadata`; blocking advisory/vet verification runs in CI (`supply_chain` job; local binaries not installed on Windows build host). | SwarmTorch maintainers | Wave 16 supply-chain audit sweep |
| EX-2026-11-14-01 | `rustix` (`1.1.6`), `linux-raw-sys` (`0.12.1`), `errno` (`0.3.15`), `bitflags` (`2.13.2`), `libc` (`0.2.191`) | `1.1.6` / `0.12.1` / `0.3.15` / `2.13.2` / `0.2.191` | `safe-to-deploy` | Added transitive dependencies (and minor `libc` bump) required by audited `fs4 1.1.0` synchronous kernel file locking (`ST-05`, Phase `13B-2`). Renewed in Wave 15 pending upstream audit import; blocking advisory/vet verification runs in CI (`supply_chain` job; local binaries not installed on Windows build host). | SwarmTorch maintainers | Wave 16 supply-chain audit sweep |
