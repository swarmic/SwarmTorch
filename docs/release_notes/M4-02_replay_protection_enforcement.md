# SwarmTorch M4-02: Replay Protection Enforcement

**Date:** 2026-02-13
**Type:** Security Hardening (Replay Attack Defense)
**Builds on:** M4-01 (Protocol Security Hardening)

## Summary

This release implements replay protection enforcement for message envelopes, defending against replay and reorder attacks as specified in ADR-0008. The implementation provides sequence monotonicity tracking and timestamp freshness validation on a per-peer basis with bounded memory usage.

**Key Achievement:** Closes the gap identified in the White Paper security posture audit — replay protection fields existed in M4-01, but enforcement logic was not implemented. M4-02 delivers the complete enforcement layer.

---

## Key Changes

### 1. Core Replay Protection Module

**[NEW]** `swarm-torch-core/src/replay.rs`

**Components:**

- **`ReplayProtection`**: Stateful replay guard with LRU-based peer state cache
  - Per-peer sequence tracking (isolated state prevents cross-peer attacks)
  - Timestamp validation with configurable clock skew tolerance (default ±60s)
  - LRU eviction (default capacity: 1000 peers)
  - Deterministic `BTreeSet` for sequence window (out-of-order tolerance: 16 messages)

- **`PeerReplayState`**: Per-peer sequence tracking state
  - `last_sequence`: Highest sequence number observed
  - `recent_sequences`: Small sliding window for out-of-order detection (16 messages)
  - Automatic pruning to bound memory

- **`ReplayError`**: Three-tier error discrimination
  - `Expired`: Timestamp outside acceptable window
  - `Replay`: Duplicate sequence number
  - `TooOld`: Sequence number beyond tolerance window

**Design Decisions:**

| Decision | Rationale |
|----------|-----------|
| LRU cache (1000 peers default) | Bounded memory; graceful eviction under load |
| Clock skew ±60s | Balances NTP drift tolerance vs replay window |
| Sequence window (16 messages) | Tolerates minor network reordering without full buffering |
| BTreeSet (not HashMap) | Deterministic iteration order; repo pattern compliance |

**Evidence:** ADR-0008 (lines 1895-1907), White Paper § Replay Protection (line 60)

### 2. Integration with Message Protocol

**[MODIFY]** `swarm-torch-net/src/protocol.rs`

**Added:**

```rust
pub fn verify_authenticated(
    &self,
    replay_guard: &mut ReplayProtection,
    current_time: u32,
) -> Result<(), VerifyError>
```

**Optimized Validation Flow:**

```
1. CHEAP:    Timestamp expiry check (no state mutation, fail-fast)
2. EXPENSIVE: Signature verification (CPU-intensive Ed25519 crypto)
3. STATEFUL: Replay check (LRU cache mutation)
```

**Rationale:** Fail-fast on expired messages before paying crypto cost; mutate state only after crypto succeeds.

**Error Type:**

```rust
pub enum VerifyError {
    Crypto(swarm_torch_core::crypto::VerifyError),
    Replay(swarm_torch_core::replay::ReplayError),
    MissingSignature,
    InvalidSignatureLength { expected: usize, found: usize },
}
```

### 3. Dependency Addition

**[MODIFY]** `swarm-torch-core/Cargo.toml`

```toml
lru = { version = "0.12", default-features = false }
```

**Verification:** LRU 0.12 supports `no_std` with `hashbrown` backend (alloc-only).

**[MODIFY]** `swarm-torch-net/Cargo.toml`

```toml
std = ["alloc", "swarm-torch-core/std", "tokio"]  # Added "alloc" dependency
```

**Fix:** Ensures `verify_authenticated` method is available when `std` feature is enabled.

---

## Tests Added

### Core Unit Tests (15 tests in `swarm-torch-core/src/replay.rs`)

| Test | Invariant |
|------|-----------|
| `validate_accepts_first_message_from_peer` | First message always accepted |
| `validate_accepts_monotonic_sequence` | Increasing sequences accepted |
| `validate_rejects_duplicate_sequence` | Same seq twice → `Replay` error |
| `validate_rejects_retrograde_sequence` | seq far in past → `TooOld` error |
| `validate_accepts_out_of_order_within_window` | seq ±8 tolerance |
| `validate_rejects_timestamp_too_old` | ts < (now - 60s) → `Expired` error |
| `validate_rejects_timestamp_too_new` | ts > (now + 60s) → `Expired` error |
| `validate_accepts_timestamp_within_skew_window` | now ± 60s accepted |
| `lru_eviction_respects_capacity` | Cache capacity enforced |
| `peer_state_isolation` | Peer A and Peer B have independent sequence spaces |
| `sequence_window_prunes_correctly` | Old sequences removed to bound memory |
| `deterministic_btreeset_ordering` | BTreeSet ensures reproducible iteration |
| `check_timestamp_only_does_not_mutate_state` | Read-only timestamp check |
| `validate_sequence_mutates_state` | Sequence validation updates cache |
| `display_error_formats_correctly` | Error messages include diagnostic info |

### Integration Tests (10 tests in `swarm-torch-net/tests/replay_integration.rs`)

| Test | Invariant |
|------|-----------|
| `envelope_verify_authenticated_golden_path` | Valid sig + fresh ts + new seq → Ok |
| `envelope_verify_authenticated_rejects_replay` | Duplicate seq → Err |
| `envelope_verify_authenticated_rejects_expired` | Old timestamp → Err |
| `envelope_verify_authenticated_signature_before_replay` | Bad sig fails before replay check |
| `envelope_verify_authenticated_rejects_missing_signature` | No signature → Err |
| `envelope_verify_authenticated_rejects_wrong_signature_length` | Signature length ≠ 64 → Err |
| `envelope_verify_authenticated_rejects_tampered_payload` | Payload mismatch → Err |
| `envelope_verify_authenticated_monotonic_sequences` | 10 sequential messages accepted |
| `envelope_verify_authenticated_out_of_order_within_window` | Out-of-order ≤16 → Ok; >16 → Err |
| `envelope_verify_authenticated_multi_peer_isolation` | Peer A seq=5 ≠ Peer B seq=5 |

---

## Known Limitations

### Memory-Only Replay Cache (Acceptable for Alpha)

**Issue:** The replay cache is **non-persistent** across node restarts.

**Attack Window:**
- If a node restarts, the replay cache resets to empty.
- Messages within the timestamp window (≤60s) that were previously rejected can be replayed during the grace period after restart.
- Window = `min(time_since_restart, max_clock_skew_secs)`.

**Mitigation (M4-02):**
- Timestamp expiry (60s) bounds the replay window.
- Byzantine aggregation provides defense-in-depth (robust aggregators reject outliers).
- Acceptable tradeoff for embedded/stateless nodes in alpha releases.

**Future Work (post-M4-02):**
- Persistent replay cache (requires storage backend — ADR TBD).
- Coordinator sync for distributed replay state.
- Startup grace period with elevated validation.

**Documented:** SECURITY.md, agent MEMORY.md

### Clock Synchronization Assumption

**Assumption:** Nodes have loosely synchronized clocks via NTP (or equivalent).

**Tolerance:** ±60s window accommodates:
- Typical NTP drift (±1s steady-state)
- Transient skew during sync
- Network latency variance

**Failure Modes:**
- Clock skew >60s → false rejections of valid messages
- Backward clock adjustment → temporary acceptance of expired messages

**Operational Guidance:**
- Monitor clock drift metrics (`chrony tracking` or equivalent)
- Alert on skew >30s (half of tolerance window)
- Consider larger window (120s) for networks with poor NTP access

---

## Verification

```bash
# Core replay tests (15 tests)
cargo test -p swarm-torch-core replay
# ✅ 15 passed; 0 failed

# Integration tests (10 tests)
cargo test -p swarm-torch-net
# ✅ 10 passed; 0 failed

# Full workspace (98 tests total: 42 core + 46 swarm-torch + 10 net)
cargo test --workspace
# ✅ 98 passed; 0 failed; 2 ignored

# MSRV 1.75 compatibility
cargo +1.75 build -p swarm-torch-core --no-default-features --features alloc
# ✅ Finished dev [optimized + debuginfo] target(s) in 14.43s

# Clippy (strict mode)
cargo clippy --workspace --all-targets --all-features -- -D warnings
# ✅ Finished dev [optimized + debuginfo] target(s) in 2.06s
```

---

## ADR/White Paper Compliance

| Requirement | Source | Status |
|-------------|--------|--------|
| Ed25519 signatures | ADR-0008 line 1876-1893 | ✅ M4-01 |
| Replay protection | ADR-0008 line 1895-1907 | ✅ M4-02 |
| Sequence monotonicity | White Paper line 333 | ✅ M4-02 |
| Timestamp validation | White Paper line 334, ADR-0008 line 1945 | ✅ M4-02 |
| Clock skew tolerance (60s) | ADR-0008 line 1945 | ✅ M4-02 |
| Per-peer state isolation | ADR-0008 threat model | ✅ M4-02 |
| no_std compatibility | ADR-0002, White Paper line 69 | ✅ M4-02 |
| LRU bounded memory | ADR-0008 (implicit) | ✅ M4-02 |

---

## Migration Notes

### API Additions (No Breaking Changes)

| API | Location | Feature Gate |
|-----|----------|--------------|
| `ReplayProtection` | `swarm-torch-core/src/replay.rs` | `alloc` |
| `ReplayError` | `swarm-torch-core/src/replay.rs` | `alloc` |
| `MessageEnvelope::verify_authenticated()` | `swarm-torch-net/src/protocol.rs` | `alloc` |
| `VerifyError` | `swarm-torch-net/src/protocol.rs` | `alloc` |

### Behavioral Changes

**None.** This is a pure addition — no existing APIs modified.

### Integration Example

```rust
use swarm_torch_core::crypto::{KeyPair, MessageAuth};
use swarm_torch_core::replay::ReplayProtection;
use swarm_torch_net::protocol::{MessageEnvelope, MessageType};

// Setup
let seed = [1u8; 32];
let keypair = KeyPair::from_seed(seed);
let auth = MessageAuth::new(keypair.clone());
let mut replay_guard = ReplayProtection::new();

// Create signed envelope
let payload = b"test payload".to_vec();
let now = 1000;

let mut envelope = MessageEnvelope::new_with_public_key(
    *keypair.public_key(),
    MessageType::Heartbeat,
    payload,
)
    .with_sequence(1)
    .with_timestamp(now);

let sig = auth.sign(
    envelope.version,
    envelope.message_type as u8,
    envelope.sequence,
    envelope.timestamp,
    &envelope.payload,
);
envelope = envelope.with_signature(sig.as_bytes().to_vec());

// Verify: signature + replay protection
envelope.verify_authenticated(&mut replay_guard, now)?;
```

---

## Contributors

- Protocol Security Hardening (M4-01, M4-02): SwarmTorch team + Claude Sonnet 4.5

---

## Next Steps

**M4-03 Candidates:**

1. **Persistent Replay Cache:** Durable sequence tracking across restarts (requires storage backend decision — ADR TBD).
2. **Transport Security:** TLS/DTLS integration for transports (TCP/UDP).
3. **Identity Provider Integration:** Sybil resistance hooks (ADR-0008A implementation).
4. **Byzantine Attack Harness:** Simulation testing under adversarial conditions (ADR-0007A).

**White Paper Update Status (completed):**

- [x] Update line 60: "Replay protection: Partial" → "Replay protection: Implemented"
- [x] Add caveat: "Memory-only cache (non-persistent across restarts)"

---

## References

- **ADR-0008:** Threat Model and Trust Boundaries (lines 1818-1985)
- **ADR-0008A:** Identity and Sybil Resistance Boundary (lines 3155-3212)
- **White Paper:** § Security Engineering (lines 377-392), § Protocol (lines 323-348)
- **SECURITY.md:** Updated with current security posture (lines 27-38)
- **RFC 8032:** Edwards-Curve Digital Signature Algorithm (Ed25519)
- **LRU crate:** https://docs.rs/lru/0.12.5/lru/
