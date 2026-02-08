# SwarmTorch Context Sources

> **Purpose:** Define document hierarchy to prevent context drift from cross-system documents.

## Primary Sources (Authoritative for SwarmTorch)

| Source | Purpose |
|--------|---------|
| [`ADRs.md`](ADRs.md) | All architectural decisions |
| `swarm-torch-*` crates | Implementation source of truth |
| [`readme.md`](readme.md) | Project overview, roadmap, usage |

## Secondary Sources (Reference Only)

| Source | Scope | Notes |
|--------|-------|-------|
| `SWARMIC_NETWORK_WHITE_PAPER_v12.*.md` | SwarmicOS/GridSwarm/Swarmflow baseline | **NOT** a SwarmTorch spec. Use only for `SwarmicNetworkProvider` integration profile (ADR-0008A). |
| `adr-with-updates.md` | Historical planning notes | Superseded by `ADRs.md`. Do not follow `/docs/adr/` instructions. |

## What SwarmTorch Does NOT Own

The Swarmic Network whitepaper defines requirements for:
- RA/capability gating
- Mission-plane traffic classes
- Hardware tier enforcement
- PoSt/PoSw receipt logic
- SwarmicOS/Swarmflow intent semantics

**These are NOT SwarmTorch responsibilities.** SwarmTorch implements swarm learning primitives; the control plane is external.

## Build Agent Instructions

1. **Start with `ADRs.md`** for any architectural question
2. **Check crate source** for implementation details
3. **Only consult whitepaper** when implementing `SwarmicNetworkProvider` integration
4. **Never import** mission-plane/RA/receipt requirements into SwarmTorch core
