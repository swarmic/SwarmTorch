//! Replay protection, cohort allowlists, and session-epoch recovery (`ADR-0008`, `ADR-0021`, `W9-02`).
//!
//! Replaces `lru::LruCache` with a 100% safe, deterministic `BTreeMap` replay store
//! indexed by `(SwarmInstanceId, PeerId, SessionEpoch)` and a secondary expiry index
//! `BTreeMap<(ExpiresAt, ReplayRecordKey), ()>`.
//!
//! ## Security Invariants (`ADR-0021`)
//!
//! 1. **No In-Window Eviction**: Active records whose timestamp window has not expired
//!    ($\text{now} \le t_{\max} + A$) are **never** evicted when capacity is reached;
//!    admission fails closed with [`ReplayError::CacheFull`].
//! 2. **Expiry-Indexed Reclamation**: Once $\text{now} > t_{\max} + A$, any replay of a
//!    prior message with $t \le t_{\max}$ is already rejected by the stateless timestamp
//!    check $|\text{now} - t| \le A$, so the record can be safely reclaimed in $O(\log N)$ time.
//! 3. **Cohort Allowlist Gate**: When a [`CohortAllowlist`] is configured, unauthorized peers
//!    are rejected before any state allocation with [`ReplayError::UnauthorizedPeer`].
//! 4. **Receiver-Challenged Epoch Recovery**: A peer that restarts with a higher `session_epoch`
//!    must complete a signed 256-bit receiver challenge handshake (`swarmtorch.session.handshake.v1`);
//!    unauthenticated epoch jumps are rejected with [`ReplayError::RecoveryRequired`].

#[cfg(feature = "alloc")]
use crate::crypto::{MessageAuth, SessionHandshakeChallenge, Signature};
use crate::traits::PeerId;

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::collections::{BTreeMap, BTreeSet};

/// Default sequence tolerance window size (messages).
pub const DEFAULT_SEQUENCE_TOLERANCE_WINDOW: usize = 16;
/// Upper bound for configurable tolerance window.
pub const MAX_SEQUENCE_TOLERANCE_WINDOW: usize = 256;
/// Default replay cache capacity (records).
pub const DEFAULT_CACHE_CAPACITY: usize = 1000;
/// Default clock skew tolerance (seconds).
pub const DEFAULT_MAX_CLOCK_SKEW_SECS: u32 = 60;

/// 16-byte swarm deployment / cohort instance identifier (`ADR-0021`).
pub type SwarmInstanceId = [u8; 16];

/// Monotonically increasing sender session epoch (`ADR-0021`).
pub type SessionEpoch = u64;

/// Composite replay record key `(SwarmInstanceId, PeerId, SessionEpoch)` (`ADR-0008`, `ADR-0021`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReplayRecordKey {
    /// Swarm deployment / cohort identifier.
    pub swarm_instance_id: SwarmInstanceId,
    /// Canonical sender `PeerId`.
    pub peer: PeerId,
    /// Sender session epoch.
    pub session_epoch: SessionEpoch,
}

impl ReplayRecordKey {
    /// Create a scoped replay record key.
    pub const fn new(
        swarm_instance_id: SwarmInstanceId,
        peer: PeerId,
        session_epoch: SessionEpoch,
    ) -> Self {
        Self {
            swarm_instance_id,
            peer,
            session_epoch,
        }
    }

    /// Default single-swarm key (`swarm_instance_id = [0; 16]`, `session_epoch = 0`).
    pub const fn default_for_peer(peer: PeerId) -> Self {
        Self {
            swarm_instance_id: [0u8; 16],
            peer,
            session_epoch: 0,
        }
    }
}

/// Authorized cohort peer allowlist (`ADR-0008`, `ADR-0021`, `W9-02`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CohortAllowlist {
    allowed_peers: BTreeSet<PeerId>,
}

#[cfg(feature = "alloc")]
impl CohortAllowlist {
    /// Create an empty cohort allowlist.
    pub const fn new() -> Self {
        Self {
            allowed_peers: BTreeSet::new(),
        }
    }

    /// Construct a cohort allowlist from an iterator of authorized `PeerId`s.
    pub fn from_peers<I: IntoIterator<Item = PeerId>>(peers: I) -> Self {
        Self {
            allowed_peers: peers.into_iter().collect(),
        }
    }

    /// Admit a `PeerId` to the cohort allowlist.
    pub fn insert(&mut self, peer: PeerId) -> bool {
        self.allowed_peers.insert(peer)
    }

    /// Remove a `PeerId` from the cohort allowlist.
    pub fn remove(&mut self, peer: &PeerId) -> bool {
        self.allowed_peers.remove(peer)
    }

    /// Check whether `peer` is a member of the cohort allowlist.
    pub fn contains(&self, peer: &PeerId) -> bool {
        self.allowed_peers.contains(peer)
    }

    /// Number of authorized peers in the cohort.
    pub fn len(&self) -> usize {
        self.allowed_peers.len()
    }

    /// Returns `true` if no peers are registered in the allowlist.
    pub fn is_empty(&self) -> bool {
        self.allowed_peers.is_empty()
    }
}

/// Replay protection configuration errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayConfigError {
    /// Replay cache capacity must be non-zero.
    ZeroCapacity,
    /// Tolerance window exceeds maximum.
    ToleranceWindowTooLarge {
        /// Requested window size.
        window: usize,
        /// Maximum allowed window size.
        max: usize,
    },
}

impl core::fmt::Display for ReplayConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReplayConfigError::ZeroCapacity => write!(f, "capacity must be non-zero"),
            ReplayConfigError::ToleranceWindowTooLarge { window, max } => {
                write!(f, "tolerance window {window} exceeds maximum {max}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReplayConfigError {}

/// Safe expiry-indexed replay protection state (`ADR-0008`, `ADR-0021`, `W9-02`).
#[derive(Debug)]
#[cfg(feature = "alloc")]
pub struct ReplayProtection {
    /// Scoped per-peer replay state (`BTreeMap`, zero `unsafe`).
    records: BTreeMap<ReplayRecordKey, PeerReplayState>,
    /// Active session epoch per `(SwarmInstanceId, PeerId)`.
    peer_epochs: BTreeMap<(SwarmInstanceId, PeerId), SessionEpoch>,
    /// Secondary expiry index `(expires_at_secs, key)` for $O(\log N)$ post-window reclamation.
    expiry_index: BTreeMap<(u64, ReplayRecordKey), ()>,
    /// Outstanding receiver-issued session recovery challenges keyed by `(SwarmInstanceId, PeerId)`.
    pending_challenges: BTreeMap<(SwarmInstanceId, PeerId), SessionHandshakeChallenge>,
    /// Optional cohort membership allowlist.
    cohort_allowlist: Option<CohortAllowlist>,
    /// Maximum number of active replay records.
    capacity: usize,
    /// Maximum clock skew tolerance (seconds).
    max_clock_skew_secs: u32,
    /// Sequence tolerance window (messages).
    tolerance_window: usize,
}

#[cfg(feature = "alloc")]
impl ReplayProtection {
    /// Create with default configuration (`capacity = 1000`, `max_clock_skew_secs = 60`, `window = 16`).
    pub fn new() -> Self {
        Self {
            records: BTreeMap::new(),
            peer_epochs: BTreeMap::new(),
            expiry_index: BTreeMap::new(),
            pending_challenges: BTreeMap::new(),
            cohort_allowlist: None,
            capacity: DEFAULT_CACHE_CAPACITY,
            max_clock_skew_secs: DEFAULT_MAX_CLOCK_SKEW_SECS,
            tolerance_window: DEFAULT_SEQUENCE_TOLERANCE_WINDOW,
        }
    }

    /// Create with custom configuration (fallible).
    pub fn try_with_config(
        capacity: usize,
        max_clock_skew_secs: u32,
    ) -> Result<Self, ReplayConfigError> {
        Self::try_with_tolerance_window(
            capacity,
            max_clock_skew_secs,
            DEFAULT_SEQUENCE_TOLERANCE_WINDOW,
        )
    }

    /// Create with custom configuration including tolerance window (fallible).
    pub fn try_with_tolerance_window(
        capacity: usize,
        max_clock_skew_secs: u32,
        tolerance_window: usize,
    ) -> Result<Self, ReplayConfigError> {
        if capacity == 0 {
            return Err(ReplayConfigError::ZeroCapacity);
        }
        if tolerance_window > MAX_SEQUENCE_TOLERANCE_WINDOW {
            return Err(ReplayConfigError::ToleranceWindowTooLarge {
                window: tolerance_window,
                max: MAX_SEQUENCE_TOLERANCE_WINDOW,
            });
        }
        Ok(Self {
            records: BTreeMap::new(),
            peer_epochs: BTreeMap::new(),
            expiry_index: BTreeMap::new(),
            pending_challenges: BTreeMap::new(),
            cohort_allowlist: None,
            capacity,
            max_clock_skew_secs,
            tolerance_window,
        })
    }

    /// Attach a [`CohortAllowlist`] so unauthorized peers are rejected before state mutation.
    pub fn with_cohort_allowlist(mut self, allowlist: CohortAllowlist) -> Self {
        self.cohort_allowlist = Some(allowlist);
        self
    }

    /// Set or replace the [`CohortAllowlist`].
    pub fn set_cohort_allowlist(&mut self, allowlist: Option<CohortAllowlist>) {
        self.cohort_allowlist = allowlist;
    }

    /// Create with custom configuration (deprecated panicking wrapper).
    #[deprecated(
        since = "0.1.0-alpha.6x",
        note = "Use `try_with_config` instead. This constructor panics on invalid input."
    )]
    pub fn with_config(capacity: usize, max_clock_skew_secs: u32) -> Self {
        match Self::try_with_config(capacity, max_clock_skew_secs) {
            Ok(protection) => protection,
            Err(ReplayConfigError::ZeroCapacity) => panic!("capacity must be non-zero"),
            Err(ReplayConfigError::ToleranceWindowTooLarge { .. }) => {
                unreachable!("try_with_config uses default tolerance window")
            }
        }
    }

    /// Check timestamp validity without mutating state (fail-fast optimization).
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

    /// Check cohort allowlist authorization without mutating state.
    pub fn check_peer_authorized(&self, peer: &PeerId) -> Result<(), ReplayError> {
        if let Some(ref allowlist) = self.cohort_allowlist {
            if !allowlist.contains(peer) {
                return Err(ReplayError::UnauthorizedPeer { peer: *peer });
            }
        }
        Ok(())
    }

    /// Check session-epoch continuity for `key` without mutating state.
    pub fn check_epoch_continuity(&self, key: &ReplayRecordKey) -> Result<(), ReplayError> {
        if let Some(&active_epoch) = self.peer_epochs.get(&(key.swarm_instance_id, key.peer)) {
            if key.session_epoch < active_epoch {
                return Err(ReplayError::StaleSessionEpoch {
                    peer: key.peer,
                    current_epoch: active_epoch,
                    received_epoch: key.session_epoch,
                });
            }
            if key.session_epoch > active_epoch {
                return Err(ReplayError::RecoveryRequired {
                    peer: key.peer,
                    current_epoch: active_epoch,
                    requested_epoch: key.session_epoch,
                });
            }
        }
        Ok(())
    }

    /// Reclaim all replay records whose maximum observed timestamp window has expired
    /// ($\text{current\_time} > t_{\max} + A$).
    pub fn reclaim_expired(&mut self, current_time: u32) -> usize {
        let now_u64 = current_time as u64;
        let mut reclaimed = 0usize;

        while let Some((&(expires_at, key), _)) = self.expiry_index.iter().next() {
            if expires_at >= now_u64 {
                break;
            }
            self.expiry_index.remove(&(expires_at, key));
            self.records.remove(&key);
            if self.peer_epochs.get(&(key.swarm_instance_id, key.peer)) == Some(&key.session_epoch)
            {
                self.peer_epochs.remove(&(key.swarm_instance_id, key.peer));
            }
            reclaimed += 1;
        }
        reclaimed
    }

    /// Validate sequence number and update scoped state at `(timestamp, current_time)`.
    pub fn validate_sequence_scoped_at(
        &mut self,
        key: &ReplayRecordKey,
        seq: u64,
        timestamp: u32,
        current_time: u32,
    ) -> Result<(), ReplayError> {
        self.check_peer_authorized(&key.peer)?;
        self.check_epoch_continuity(key)?;
        self.reclaim_expired(current_time);

        let tw = self.tolerance_window;
        let skew = self.max_clock_skew_secs;

        if let Some(state) = self.records.get_mut(key) {
            let old_expires = state.expires_at;
            state.validate_and_update(seq, key.peer, tw)?;
            if timestamp > state.max_timestamp_seen {
                state.max_timestamp_seen = timestamp;
                state.expires_at = (timestamp as u64).saturating_add(skew as u64);
                self.expiry_index.remove(&(old_expires, *key));
                self.expiry_index.insert((state.expires_at, *key), ());
            }
            return Ok(());
        }

        if self.records.len() >= self.capacity {
            return Err(ReplayError::CacheFull {
                capacity: self.capacity,
            });
        }

        let state = PeerReplayState::new(seq, timestamp, skew);
        let expires_at = state.expires_at;
        self.records.insert(*key, state);
        self.peer_epochs
            .insert((key.swarm_instance_id, key.peer), key.session_epoch);
        self.expiry_index.insert((expires_at, *key), ());
        Ok(())
    }

    /// Validate sequence number and update state (default single-swarm scope).
    pub fn validate_sequence(&mut self, peer: &PeerId, seq: u64) -> Result<(), ReplayError> {
        let key = ReplayRecordKey::default_for_peer(*peer);
        self.validate_sequence_scoped_at(&key, seq, 0, 0)
    }

    /// Full scoped validation: timestamp + allowlist + epoch continuity + sequence.
    pub fn validate_scoped(
        &mut self,
        key: &ReplayRecordKey,
        sequence: u64,
        timestamp: u32,
        current_time: u32,
    ) -> Result<(), ReplayError> {
        self.check_timestamp_only(timestamp, current_time)?;
        self.validate_sequence_scoped_at(key, sequence, timestamp, current_time)
    }

    /// Full validation: timestamp + sequence (default single-swarm scope).
    pub fn validate(
        &mut self,
        peer: &PeerId,
        sequence: u64,
        timestamp: u32,
        current_time: u32,
    ) -> Result<(), ReplayError> {
        let key = ReplayRecordKey::default_for_peer(*peer);
        self.validate_scoped(&key, sequence, timestamp, current_time)
    }

    /// Register a receiver-generated 256-bit challenge for a restarting peer (`ADR-0021`, `W9-02`).
    pub fn register_recovery_challenge(
        &mut self,
        challenge: SessionHandshakeChallenge,
    ) -> Result<(), ReplayError> {
        self.check_peer_authorized(&challenge.sender_peer)?;
        if challenge.challenge_nonce == [0u8; 32] {
            return Err(ReplayError::HandshakeVerificationFailed);
        }
        if let Some(&active_epoch) = self
            .peer_epochs
            .get(&(challenge.swarm_instance_id, challenge.sender_peer))
        {
            if challenge.requested_epoch <= active_epoch {
                return Err(ReplayError::StaleSessionEpoch {
                    peer: challenge.sender_peer,
                    current_epoch: active_epoch,
                    received_epoch: challenge.requested_epoch,
                });
            }
        }
        self.pending_challenges.insert(
            (challenge.swarm_instance_id, challenge.sender_peer),
            challenge,
        );
        Ok(())
    }

    /// Complete a receiver-challenged session recovery handshake (`swarmtorch.session.handshake.v1`).
    pub fn complete_recovery_handshake(
        &mut self,
        swarm_instance_id: SwarmInstanceId,
        sender_public_key: &[u8; 32],
        signature: &Signature,
        current_time: u32,
    ) -> Result<ReplayRecordKey, ReplayError> {
        let sender_peer = PeerId::try_from_public_key_bytes(sender_public_key)
            .map_err(|_| ReplayError::HandshakeVerificationFailed)?;
        self.check_peer_authorized(&sender_peer)?;

        let challenge = self
            .pending_challenges
            .get(&(swarm_instance_id, sender_peer))
            .copied()
            .ok_or(ReplayError::HandshakeVerificationFailed)?;

        self.check_timestamp_only(challenge.issued_at_secs, current_time)?;
        MessageAuth::verify_session_handshake(sender_public_key, &challenge, signature)
            .map_err(|_| ReplayError::HandshakeVerificationFailed)?;

        self.pending_challenges
            .remove(&(swarm_instance_id, sender_peer));

        if let Some(old_epoch) = self.peer_epochs.remove(&(swarm_instance_id, sender_peer)) {
            let old_key = ReplayRecordKey::new(swarm_instance_id, sender_peer, old_epoch);
            if let Some(old_state) = self.records.remove(&old_key) {
                self.expiry_index.remove(&(old_state.expires_at, old_key));
            }
        }

        self.reclaim_expired(current_time);
        let new_key =
            ReplayRecordKey::new(swarm_instance_id, sender_peer, challenge.requested_epoch);
        if self.records.len() >= self.capacity {
            return Err(ReplayError::CacheFull {
                capacity: self.capacity,
            });
        }

        let state = PeerReplayState::new(
            challenge.initial_sequence,
            current_time,
            self.max_clock_skew_secs,
        );
        let expires_at = state.expires_at;
        self.records.insert(new_key, state);
        self.peer_epochs
            .insert((swarm_instance_id, sender_peer), challenge.requested_epoch);
        self.expiry_index.insert((expires_at, new_key), ());
        Ok(new_key)
    }

    fn is_timestamp_valid(&self, ts: u32, now: u32) -> bool {
        now.abs_diff(ts) <= self.max_clock_skew_secs
    }

    /// Get current active record count (for testing/monitoring).
    pub fn cache_size(&self) -> usize {
        self.records.len()
    }
}

#[cfg(feature = "alloc")]
impl Default for ReplayProtection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "alloc")]
#[derive(Debug)]
struct PeerReplayState {
    last_sequence: u64,
    recent_sequences: BTreeSet<u64>,
    max_timestamp_seen: u32,
    expires_at: u64,
}

#[cfg(feature = "alloc")]
impl PeerReplayState {
    fn new(initial_seq: u64, timestamp: u32, max_clock_skew_secs: u32) -> Self {
        let mut recent_sequences = BTreeSet::new();
        recent_sequences.insert(initial_seq);
        Self {
            last_sequence: initial_seq,
            recent_sequences,
            max_timestamp_seen: timestamp,
            expires_at: (timestamp as u64).saturating_add(max_clock_skew_secs as u64),
        }
    }

    fn validate_and_update(
        &mut self,
        seq: u64,
        peer: PeerId,
        tolerance_window: usize,
    ) -> Result<(), ReplayError> {
        if self.recent_sequences.contains(&seq) {
            return Err(ReplayError::Replay { peer, seq });
        }

        let max_acceptable_oldness = seq.saturating_add(tolerance_window as u64);
        if max_acceptable_oldness <= self.last_sequence {
            return Err(ReplayError::TooOld {
                peer,
                seq,
                last_seen: self.last_sequence,
            });
        }

        self.recent_sequences.insert(seq);
        if seq > self.last_sequence {
            self.last_sequence = seq;
        }

        self.prune_old_sequences(self.last_sequence.saturating_sub(tolerance_window as u64));
        Ok(())
    }

    fn prune_old_sequences(&mut self, threshold: u64) {
        self.recent_sequences.retain(|&s| s > threshold);
    }
}

/// Replay protection errors (`ADR-0008`, `ADR-0021`, `W9-02`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayError {
    /// Timestamp outside acceptable window.
    Expired {
        /// Received timestamp.
        ts: u32,
        /// Current time.
        now: u32,
        /// Maximum allowed skew.
        window: u32,
    },
    /// Sequence number already seen (duplicate).
    Replay {
        /// Peer ID.
        peer: PeerId,
        /// Duplicate sequence.
        seq: u64,
    },
    /// Sequence number too far in past.
    TooOld {
        /// Peer ID.
        peer: PeerId,
        /// Old sequence.
        seq: u64,
        /// Last seen sequence.
        last_seen: u64,
    },
    /// Active replay cache is full and no entries have expired past `t_max + A`.
    CacheFull {
        /// Configured maximum active record capacity.
        capacity: usize,
    },
    /// Sender peer is not present in the configured [`CohortAllowlist`].
    UnauthorizedPeer {
        /// Rejected peer ID.
        peer: PeerId,
    },
    /// Sender attempted to advance `session_epoch` without completing a recovery handshake.
    RecoveryRequired {
        /// Peer ID.
        peer: PeerId,
        /// Currently active session epoch.
        current_epoch: SessionEpoch,
        /// Requested session epoch.
        requested_epoch: SessionEpoch,
    },
    /// Sender presented a stale `session_epoch < current_epoch`.
    StaleSessionEpoch {
        /// Peer ID.
        peer: PeerId,
        /// Currently active session epoch.
        current_epoch: SessionEpoch,
        /// Received stale session epoch.
        received_epoch: SessionEpoch,
    },
    /// Session recovery handshake challenge or signature verification failed.
    HandshakeVerificationFailed,
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
            ReplayError::CacheFull { capacity } => {
                write!(
                    f,
                    "replay cache full at capacity={} (no expired entries)",
                    capacity
                )
            }
            ReplayError::UnauthorizedPeer { peer } => {
                write!(f, "unauthorized peer not in cohort allowlist: {:?}", peer)
            }
            ReplayError::RecoveryRequired {
                peer,
                current_epoch,
                requested_epoch,
            } => {
                write!(
                    f,
                    "session recovery handshake required: peer={:?}, current_epoch={}, requested_epoch={}",
                    peer, current_epoch, requested_epoch
                )
            }
            ReplayError::StaleSessionEpoch {
                peer,
                current_epoch,
                received_epoch,
            } => {
                write!(
                    f,
                    "stale session epoch: peer={:?}, current_epoch={}, received_epoch={}",
                    peer, current_epoch, received_epoch
                )
            }
            ReplayError::HandshakeVerificationFailed => {
                write!(f, "session recovery handshake verification failed")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReplayError {}

#[cfg(all(test, feature = "alloc"))]
#[path = "replay_tests.rs"]
mod tests;
