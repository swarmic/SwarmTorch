//! Replay protection for message envelopes
//!
//! This module enforces sequence monotonicity and timestamp freshness to defend
//! against replay and reorder attacks, as specified in ADR-0008.
//!
//! ## Design
//!
//! - **Per-peer state**: Isolated sequence tracking prevents cross-peer attacks
//! - **LRU cache**: Bounded memory (default 1000 peers)
//! - **Timestamp validation**: Clock skew tolerance (default ±60s)
//! - **Sequence window**: Small out-of-order tolerance (16 messages) for network reordering
//!
//! ## Known Limitations
//!
//! **Memory-only cache (non-persistent):**
//! - Node restart resets the replay cache
//! - Brief replay vulnerability window after restart (≤ max_clock_skew_secs)
//! - Acceptable for alpha; persistent cache requires storage backend (future work)
//!
//! ## Assumptions
//!
//! - Loosely synchronized clocks via NTP (±60s default tolerance)
//! - Monotonic sequence numbers per peer
//! - Timestamp in Unix seconds (u32)

use crate::traits::PeerId;

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::collections::BTreeSet;
#[cfg(feature = "alloc")]
use lru::LruCache;

/// Sequence tolerance window size (messages)
///
/// Allows minor out-of-order delivery while bounding replay risk.
/// Set to 0 for strict monotonic enforcement.
const SEQUENCE_TOLERANCE_WINDOW: usize = 16;
const DEFAULT_CACHE_CAPACITY: usize = 1000;
const DEFAULT_MAX_CLOCK_SKEW_SECS: u32 = 60;

/// Replay protection configuration errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayConfigError {
    /// Replay cache capacity must be non-zero.
    ZeroCapacity,
}

impl core::fmt::Display for ReplayConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReplayConfigError::ZeroCapacity => write!(f, "capacity must be non-zero"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReplayConfigError {}

/// Replay protection state
///
/// Tracks per-peer sequence numbers and validates timestamp freshness.
#[cfg(feature = "alloc")]
pub struct ReplayProtection {
    /// Per-peer replay state (LRU eviction)
    peer_state: LruCache<PeerId, PeerReplayState>,
    /// Maximum clock skew tolerance (seconds)
    max_clock_skew_secs: u32,
}

#[cfg(feature = "alloc")]
impl ReplayProtection {
    /// Create with default configuration
    ///
    /// - Capacity: 1000 peers
    /// - Clock skew: 60 seconds
    pub fn new() -> Self {
        match core::num::NonZeroUsize::new(DEFAULT_CACHE_CAPACITY) {
            Some(capacity) => Self {
                peer_state: LruCache::new(capacity),
                max_clock_skew_secs: DEFAULT_MAX_CLOCK_SKEW_SECS,
            },
            None => Self {
                peer_state: LruCache::new(core::num::NonZeroUsize::MIN),
                max_clock_skew_secs: DEFAULT_MAX_CLOCK_SKEW_SECS,
            },
        }
    }

    /// Create with custom configuration (fallible).
    pub fn try_with_config(
        capacity: usize,
        max_clock_skew_secs: u32,
    ) -> Result<Self, ReplayConfigError> {
        let non_zero_capacity =
            core::num::NonZeroUsize::new(capacity).ok_or(ReplayConfigError::ZeroCapacity)?;
        Ok(Self {
            peer_state: LruCache::new(non_zero_capacity),
            max_clock_skew_secs,
        })
    }

    /// Create with custom configuration
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of peers to track (LRU eviction)
    /// * `max_clock_skew_secs` - Maximum allowed clock skew in seconds
    pub fn with_config(capacity: usize, max_clock_skew_secs: u32) -> Self {
        match Self::try_with_config(capacity, max_clock_skew_secs) {
            Ok(protection) => protection,
            Err(ReplayConfigError::ZeroCapacity) => panic!("capacity must be non-zero"),
        }
    }

    /// Check timestamp validity without mutating state (fail-fast optimization)
    ///
    /// Validates that the message timestamp is within the acceptable clock skew window.
    pub fn check_timestamp_only(&self, ts: u32, now: u32) -> Result<(), ReplayError> {
        if !self.is_timestamp_valid(ts, now) {
            return Err(ReplayError::Expired {
                ts,
                now,
                window: self.max_clock_skew_secs,
            });
        }
        Ok(())
    }

    /// Validate sequence number and update state
    ///
    /// Checks for duplicate or retrograde sequences and updates peer state.
    pub fn validate_sequence(&mut self, peer: &PeerId, seq: u64) -> Result<(), ReplayError> {
        match self.peer_state.get_mut(peer) {
            Some(state) => state.validate_and_update(seq, *peer),
            None => {
                // First message from this peer
                self.peer_state.put(*peer, PeerReplayState::new(seq));
                Ok(())
            }
        }
    }

    /// Full validation: timestamp + sequence
    ///
    /// This is a convenience method that combines timestamp and sequence checks.
    pub fn validate(
        &mut self,
        peer: &PeerId,
        sequence: u64,
        timestamp: u32,
        current_time: u32,
    ) -> Result<(), ReplayError> {
        // 1. Check timestamp (cheap, no state mutation)
        self.check_timestamp_only(timestamp, current_time)?;

        // 2. Validate sequence (stateful)
        self.validate_sequence(peer, sequence)?;

        Ok(())
    }

    /// Check if timestamp is within acceptable window
    fn is_timestamp_valid(&self, ts: u32, now: u32) -> bool {
        let diff = now.abs_diff(ts);
        diff <= self.max_clock_skew_secs
    }

    /// Get current cache size (for testing/monitoring)
    pub fn cache_size(&self) -> usize {
        self.peer_state.len()
    }
}

#[cfg(feature = "alloc")]
impl Default for ReplayProtection {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-peer replay state
#[cfg(feature = "alloc")]
struct PeerReplayState {
    /// Highest sequence number seen
    last_sequence: u64,
    /// Recent sequences within tolerance window (for out-of-order detection)
    recent_sequences: BTreeSet<u64>,
}

#[cfg(feature = "alloc")]
impl PeerReplayState {
    fn new(initial_seq: u64) -> Self {
        let mut recent_sequences = BTreeSet::new();
        recent_sequences.insert(initial_seq);
        Self {
            last_sequence: initial_seq,
            recent_sequences,
        }
    }

    fn validate_and_update(&mut self, seq: u64, peer: PeerId) -> Result<(), ReplayError> {
        // Check if already seen (duplicate)
        if self.recent_sequences.contains(&seq) {
            return Err(ReplayError::Replay { peer, seq });
        }

        // Check if too old (beyond tolerance window)
        let max_acceptable_oldness = seq.saturating_add(SEQUENCE_TOLERANCE_WINDOW as u64);
        if max_acceptable_oldness < self.last_sequence {
            return Err(ReplayError::TooOld {
                peer,
                seq,
                last_seen: self.last_sequence,
            });
        }

        // Valid - update state
        self.recent_sequences.insert(seq);
        if seq > self.last_sequence {
            self.last_sequence = seq;
        }

        // Prune against the highest seen sequence so window is anchored to current frontier.
        self.prune_old_sequences(
            self.last_sequence
                .saturating_sub(SEQUENCE_TOLERANCE_WINDOW as u64),
        );

        Ok(())
    }

    fn prune_old_sequences(&mut self, threshold: u64) {
        // Remove sequences below threshold to bound memory
        self.recent_sequences.retain(|&s| s > threshold);
    }
}

/// Replay protection errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayError {
    /// Timestamp outside acceptable window
    Expired {
        /// Received timestamp
        ts: u32,
        /// Current time
        now: u32,
        /// Maximum allowed skew
        window: u32,
    },
    /// Sequence number already seen (duplicate)
    Replay {
        /// Peer ID
        peer: PeerId,
        /// Duplicate sequence
        seq: u64,
    },
    /// Sequence number too far in past
    TooOld {
        /// Peer ID
        peer: PeerId,
        /// Old sequence
        seq: u64,
        /// Last seen sequence
        last_seen: u64,
    },
}

impl core::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReplayError::Expired { ts, now, window } => {
                write!(
                    f,
                    "timestamp expired: ts={}, now={}, window={}s",
                    ts, now, window
                )
            }
            ReplayError::Replay { peer, seq } => {
                write!(f, "replay detected: peer={:?}, seq={}", peer, seq)
            }
            ReplayError::TooOld {
                peer,
                seq,
                last_seen,
            } => {
                write!(
                    f,
                    "sequence too old: peer={:?}, seq={}, last_seen={}",
                    peer, seq, last_seen
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReplayError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_peer(id: u8) -> PeerId {
        PeerId::new([id; 32])
    }

    #[test]
    fn validate_accepts_first_message_from_peer() {
        let mut guard = ReplayProtection::new();
        let peer = make_peer(1);
        let now = 1000;

        assert!(guard.validate(&peer, 1, now, now).is_ok());
    }

    #[test]
    fn try_with_config_rejects_zero_capacity() {
        let result = ReplayProtection::try_with_config(0, 60);
        assert!(matches!(result, Err(ReplayConfigError::ZeroCapacity)));
    }

    #[test]
    fn try_with_config_accepts_valid_capacity() {
        let result = ReplayProtection::try_with_config(32, 60);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_accepts_monotonic_sequence() {
        let mut guard = ReplayProtection::new();
        let peer = make_peer(1);
        let now = 1000;

        assert!(guard.validate(&peer, 1, now, now).is_ok());
        assert!(guard.validate(&peer, 2, now, now).is_ok());
        assert!(guard.validate(&peer, 3, now, now).is_ok());
        assert!(guard.validate(&peer, 100, now, now).is_ok());
    }

    #[test]
    fn validate_rejects_duplicate_sequence() {
        let mut guard = ReplayProtection::new();
        let peer = make_peer(1);
        let now = 1000;

        assert!(guard.validate(&peer, 1, now, now).is_ok());

        let result = guard.validate(&peer, 1, now, now);
        assert_eq!(result, Err(ReplayError::Replay { peer, seq: 1 }));
    }

    #[test]
    fn validate_rejects_retrograde_sequence() {
        let mut guard = ReplayProtection::new();
        let peer = make_peer(1);
        let now = 1000;

        assert!(guard.validate(&peer, 100, now, now).is_ok());

        // Sequence too far in past (beyond window)
        let result = guard.validate(&peer, 50, now, now);
        assert_eq!(
            result,
            Err(ReplayError::TooOld {
                peer,
                seq: 50,
                last_seen: 100,
            })
        );
    }

    #[test]
    fn validate_accepts_out_of_order_within_window() {
        let mut guard = ReplayProtection::new();
        let peer = make_peer(1);
        let now = 1000;

        assert!(guard.validate(&peer, 20, now, now).is_ok());
        // Within tolerance window (20 - 16 = 4, so seq >= 5 is ok)
        assert!(guard.validate(&peer, 10, now, now).is_ok());
        assert!(guard.validate(&peer, 5, now, now).is_ok());
    }

    #[test]
    fn sequence_window_check_no_overflow_at_u64_max() {
        let mut guard = ReplayProtection::new();
        let peer = make_peer(42);
        let now = 1000;

        let near_max = u64::MAX - 5;
        assert!(guard.validate(&peer, near_max, now, now).is_ok());
        assert!(guard.validate(&peer, u64::MAX - 3, now, now).is_ok());
        assert!(guard.validate(&peer, u64::MAX, now, now).is_ok());

        let old_seq = near_max.saturating_sub(20);
        let result = guard.validate(&peer, old_seq, now, now);
        assert!(matches!(result, Err(ReplayError::TooOld { .. })));
    }

    #[test]
    fn validate_rejects_timestamp_too_old() {
        let guard = ReplayProtection::with_config(100, 60);
        let now = 1000;
        let old_ts = now - 100; // Beyond 60s window

        let result = guard.check_timestamp_only(old_ts, now);
        assert_eq!(
            result,
            Err(ReplayError::Expired {
                ts: old_ts,
                now,
                window: 60,
            })
        );
    }

    #[test]
    fn validate_rejects_timestamp_too_new() {
        let guard = ReplayProtection::with_config(100, 60);
        let now = 1000;
        let future_ts = now + 100; // Beyond 60s window

        let result = guard.check_timestamp_only(future_ts, now);
        assert_eq!(
            result,
            Err(ReplayError::Expired {
                ts: future_ts,
                now,
                window: 60,
            })
        );
    }

    #[test]
    fn validate_accepts_timestamp_within_skew_window() {
        let guard = ReplayProtection::with_config(100, 60);
        let now = 1000;

        assert!(guard.check_timestamp_only(now - 50, now).is_ok());
        assert!(guard.check_timestamp_only(now, now).is_ok());
        assert!(guard.check_timestamp_only(now + 50, now).is_ok());
        assert!(guard.check_timestamp_only(now - 60, now).is_ok());
        assert!(guard.check_timestamp_only(now + 60, now).is_ok());
    }

    #[test]
    fn lru_eviction_respects_capacity() {
        let mut guard = ReplayProtection::with_config(10, 60);
        let now = 1000;

        // Fill cache with 10 peers
        for i in 0..10 {
            let peer = make_peer(i);
            assert!(guard.validate(&peer, 1, now, now).is_ok());
        }
        assert_eq!(guard.cache_size(), 10);

        // 11th peer should trigger eviction
        let peer_11 = make_peer(11);
        assert!(guard.validate(&peer_11, 1, now, now).is_ok());
        assert_eq!(guard.cache_size(), 10); // Still 10 due to LRU eviction

        // Peer 0 should be evicted (LRU), so replaying seq=1 should succeed as new peer
        let peer_0 = make_peer(0);
        assert!(guard.validate(&peer_0, 1, now, now).is_ok());
    }

    #[test]
    fn peer_state_isolation() {
        let mut guard = ReplayProtection::new();
        let peer_a = make_peer(1);
        let peer_b = make_peer(2);
        let now = 1000;

        // Peer A and B have independent sequence tracking
        assert!(guard.validate(&peer_a, 10, now, now).is_ok());
        assert!(guard.validate(&peer_b, 5, now, now).is_ok());

        // Peer A cannot replay seq=10, but Peer B can use seq=10
        assert!(guard.validate(&peer_a, 10, now, now).is_err());
        assert!(guard.validate(&peer_b, 10, now, now).is_ok());
    }

    #[test]
    fn sequence_window_prunes_correctly() {
        let mut guard = ReplayProtection::new();
        let peer = make_peer(1);
        let now = 1000;

        // Add sequences within window
        for seq in 1..=20 {
            assert!(guard.validate(&peer, seq, now, now).is_ok());
        }

        // Advance sequence far ahead to trigger pruning
        assert!(guard.validate(&peer, 100, now, now).is_ok());

        // Old sequences (< 100 - 16 = 84) should be pruned
        // Attempting to replay should fail as TooOld, not Replay
        let result = guard.validate(&peer, 50, now, now);
        assert_eq!(
            result,
            Err(ReplayError::TooOld {
                peer,
                seq: 50,
                last_seen: 100,
            })
        );
    }

    #[test]
    fn deterministic_btreeset_ordering() {
        // BTreeSet ensures deterministic iteration order
        let mut state = PeerReplayState::new(1);
        let peer = make_peer(1);

        // Insert out of order
        assert!(state.validate_and_update(5, peer).is_ok());
        assert!(state.validate_and_update(3, peer).is_ok());
        assert!(state.validate_and_update(4, peer).is_ok());
        assert!(state.validate_and_update(2, peer).is_ok());

        // BTreeSet maintains sorted order
        let sequences: Vec<u64> = state.recent_sequences.iter().copied().collect();
        assert_eq!(sequences, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn check_timestamp_only_does_not_mutate_state() {
        let guard = ReplayProtection::new();
        let now = 1000;

        // Multiple calls should all succeed (no state mutation)
        assert!(guard.check_timestamp_only(now, now).is_ok());
        assert!(guard.check_timestamp_only(now, now).is_ok());
        assert!(guard.check_timestamp_only(now - 30, now).is_ok());
        assert_eq!(guard.cache_size(), 0); // No peers added
    }

    #[test]
    fn validate_sequence_mutates_state() {
        let mut guard = ReplayProtection::new();
        let peer = make_peer(1);

        assert!(guard.validate_sequence(&peer, 1).is_ok());
        assert_eq!(guard.cache_size(), 1);

        // Duplicate should be rejected
        assert!(guard.validate_sequence(&peer, 1).is_err());
    }

    #[test]
    fn display_error_formats_correctly() {
        let peer = make_peer(1);

        let expired = ReplayError::Expired {
            ts: 900,
            now: 1000,
            window: 60,
        };
        assert_eq!(
            format!("{}", expired),
            "timestamp expired: ts=900, now=1000, window=60s"
        );

        let replay = ReplayError::Replay { peer, seq: 42 };
        assert!(format!("{}", replay).contains("replay detected"));
        assert!(format!("{}", replay).contains("seq=42"));

        let too_old = ReplayError::TooOld {
            peer,
            seq: 10,
            last_seen: 100,
        };
        assert!(format!("{}", too_old).contains("sequence too old"));
        assert!(format!("{}", too_old).contains("seq=10"));
        assert!(format!("{}", too_old).contains("last_seen=100"));
    }
}
