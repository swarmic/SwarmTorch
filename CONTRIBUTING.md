# Contributing to SwarmTorch

Thank you for your interest in contributing to SwarmTorch!

## Quick Links

- **Code of Conduct:** [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct)
- **Security Issues:** See [SECURITY.md](SECURITY.md) — do not open public issues

## Canonical Docs (No Drift)

SwarmTorch uses a strict document hierarchy to prevent context drift:

- `SWARM_TORCH_TECHNICAL_WHITE_PAPER_v0.1.md` (system model + conformance)
- `ADRs.md` (architecture decisions)
- crate source (`swarm-torch-*`) (implementation truth)

If you add or change a **normative** requirement (MUST/SHOULD/MAY), you must either:

1. Update the relevant ADR, or
2. Update the white paper and include an `Evidence:` pointer to ADR/code/tests.

## Development Setup

```bash
# Clone and build
git clone https://github.com/swarmic/swarm-torch
cd swarm-torch
cargo build

# Run tests
cargo test --workspace

# Run lints
cargo clippy --workspace -- -D warnings
cargo fmt --check
```

## Pull Request Process

1. **Fork and branch** from `main`
2. **Write tests** for new functionality
3. **Run CI checks locally:**
   ```bash
   cargo test --workspace
   cargo clippy --workspace -- -D warnings
   cargo deny check
   cargo audit
   ```
4. **Update documentation** if adding public APIs
5. **Open PR** with clear description

## What We Need Help With

- 🦀 **Embedded drivers** — BLE, LoRa, ESP-NOW
- 🤖 **Robotics integrations** — ROS2 bridge
- 🔒 **Security research** — Attack/defense strategies
- 📚 **Documentation** — Tutorials, examples
- 🧪 **Testing** — Fuzzing, property-based tests

## Code Style

- Follow `rustfmt` defaults
- Use `clippy` with `-D warnings`
- Document all public items
- Core crates: `#![forbid(unsafe_code)]`

## Commit Messages

Use conventional commits:
```
feat(core): add staleness policy support
fix(net): handle transport reconnection
docs: update ADR-0002 embedded profiles
```

## License

By contributing, you agree that your contributions will be licensed under MPL-2.0.
