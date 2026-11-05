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

## Exemption Entries

| ID | Crate | Version | Criteria | Reason | Owner | Removal Target |
|---|---|---|---|---|---|---|
| EX-2026-04-05-01 | `bytes` | `1.11.1` | `safe-to-deploy` | Added as immediate security lockfile remediation (`RUSTSEC-2026-0007` on `1.11.0`) while preserving Wave 8.1 delivery. First-party vet baseline is now started (critical crates audited) but not complete for this dependency path. | SwarmTorch maintainers | Wave 8.2 (`P2-11`) |
| EX-2026-04-05-02 | `lru` | `0.12.5` | `safe-to-deploy` | **CLOSED (`W9-02` / `P2-04` / `P2-11`)**: Replaced `lru::LruCache` in `swarm-torch-core::replay` with a 100% safe, deterministic expiry-indexed `BTreeMap` replay store, eliminating `lru 0.12.5` (`RUSTSEC-2026-0002`) and 4 transitive dependencies (`hashbrown`, `foldhash`, `equivalent`, `allocator-api2`) from `Cargo.lock` and `supply-chain/config.toml`. | SwarmTorch maintainers | Closed in Wave 9 (`W9-02`) |
| EX-2026-04-05-03 | `bincode` | `2.0.0-rc.3` | `safe-to-deploy` | **CLOSED (`W9-05` / `P2-11`)**: Removed unused `burn = "0.14"` and `burn-ndarray = "0.14"` from workspace and `swarm-torch-models`, eliminating `bincode 2.0.0-rc.3` and 57 other transitive crates from `Cargo.lock`, `supply-chain/config.toml`, and `deny.toml` (`RUSTSEC-2025-0141` retired). | SwarmTorch maintainers | Closed in Wave 9 (`W9-05`) |
