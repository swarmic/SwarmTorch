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
use crate::crypto::{
    MessageAuth, SessionHandshakeChallenge, SessionHandshakeChallengeV2, Signature,
};
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
/// Default cap on simultaneously outstanding uncompleted recovery challenges (`ST-03`).
pub const DEFAULT_MAX_PENDING_CHALLENGES: usize = 1024;

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

/// Composite v2 replay record key bound to `run_id` and receiver-derived `session_id` (`ST-03`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReplayRecordKeyV2 {
    /// Swarm deployment / cohort identifier.
    pub swarm_instance_id: SwarmInstanceId,
    /// Bound training/coordination run identifier.
    pub run_id: [u8; 16],
    /// Canonical sender `PeerId`.
    pub peer: PeerId,
    /// Sender session epoch.
    pub session_epoch: SessionEpoch,
    /// Receiver-nonce-derived session identifier ([`crate::crypto::derive_session_id_v1`]).
    pub session_id: [u8; 32],
}

impl ReplayRecordKeyV2 {
    /// Create a session-id-bound v2 replay record key.
    pub const fn new(
        swarm_instance_id: SwarmInstanceId,
        run_id: [u8; 16],
        peer: PeerId,
        session_epoch: SessionEpoch,
        session_id: [u8; 32],
    ) -> Self {
        Self {
            swarm_instance_id,
            run_id,
            peer,
            session_epoch,
            session_id,
        }
    }
}

/// Startup and restart admission policy for un-established peers (`ST-03`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReplayStartupPolicyV1 {
    /// Fail closed: un-established peers must complete a receiver-challenged recovery handshake
    /// before any data delivery is admitted (default for both cold start and crash restart).
    #[default]
    RequireSessionHandshake,
    /// Opt-in policy ONLY for explicit genesis initialization where peers do not yet hold prior
    /// session state. Still enforces `timestamp_floor_secs`, cohort allowlist, and `session_id`
    /// consistency once established.
    GenesisRosterBootstrap,
}

/// Move-only proof token produced by read-only replay preflight (`ST-01`, `ST-03`).
///
/// Consumed atomically in [`ReplayProtection::commit_prepared_admission`] only after all
/// downstream semantic and round-context validations have succeeded.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub struct PreparedReplayAdmission {
    key: PreparedReplayKey,
    sequence: u64,
    timestamp: u32,
    current_time: u32,
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy)]
enum PreparedReplayKey {
    V1(ReplayRecordKey),
    V2(ReplayRecordKeyV2),
}

#[cfg(feature = "alloc")]
impl PreparedReplayAdmission {
    /// Authenticated sender `PeerId` bound to this prepared replay admission.
    pub fn peer(&self) -> PeerId {
        match self.key {
            PreparedReplayKey::V1(ref key) => key.peer,
            PreparedReplayKey::V2(ref key) => key.peer,
        }
    }

    /// Replay sequence number checked in read-only preflight.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Sender message timestamp checked in read-only preflight.
    pub fn timestamp(&self) -> u32 {
        self.timestamp
    }

    /// Returns `true` if this admission token is bound to a v2 `session_id`.
    pub fn is_v2(&self) -> bool {
        matches!(self.key, PreparedReplayKey::V2(_))
    }

    /// Receiver-derived v2 `session_id`, if this admission came from a v2 check.
    pub fn session_id(&self) -> Option<[u8; 32]> {
        match self.key {
            PreparedReplayKey::V1(_) => None,
            PreparedReplayKey::V2(ref key) => Some(key.session_id),
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
    /// Pending challenge map capacity must be non-zero.
    ZeroPendingChallengesCapacity,
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
            ReplayConfigError::ZeroPendingChallengesCapacity => {
                write!(f, "pending challenges capacity must be non-zero")
            }
            ReplayConfigError::ToleranceWindowTooLarge { window, max } => {
                write!(f, "tolerance window {window} exceeds maximum {max}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReplayConfigError {}

#[cfg(feature = "alloc")]
type SessionScopeV2Key = (SwarmInstanceId, [u8; 16], PeerId);
#[cfg(feature = "alloc")]
type ActiveSessionV2Entry = (SessionEpoch, [u8; 32], u64);

/// Safe expiry-indexed replay protection state (`ADR-0008`, `ADR-0021`, `W9-02`, `ST-03`).
#[derive(Debug)]
#[cfg(feature = "alloc")]
pub struct ReplayProtection {
    /// Scoped per-peer replay state (`BTreeMap`, zero `unsafe`).
    records: BTreeMap<ReplayRecordKey, PeerReplayState>,
    /// Scoped v2 replay state keyed by `(SwarmInstanceId, RunId, PeerId, SessionEpoch, SessionId)`.
    records_v2: BTreeMap<ReplayRecordKeyV2, PeerReplayState>,
    /// Active v1 session epoch and latest expiry deadline per `(SwarmInstanceId, PeerId)`.
    peer_epochs: BTreeMap<(SwarmInstanceId, PeerId), (SessionEpoch, u64)>,
    /// Active receiver-bound v2 session `(SessionEpoch, SessionId, expires_at)` per `(SwarmInstanceId, RunId, PeerId)`.
    active_sessions_v2: BTreeMap<SessionScopeV2Key, ActiveSessionV2Entry>,
    /// Secondary expiry index `(expires_at_secs, key)` for $O(\log N)$ post-window reclamation.
    expiry_index: BTreeMap<(u64, ReplayRecordKey), ()>,
    /// Secondary expiry index `(expires_at_secs, key_v2)` for v2 records.
    expiry_index_v2: BTreeMap<(u64, ReplayRecordKeyV2), ()>,
    /// Outstanding receiver-issued v1 session recovery challenges keyed by `(SwarmInstanceId, PeerId)`.
    pending_challenges: BTreeMap<(SwarmInstanceId, PeerId), SessionHandshakeChallenge>,
    /// Outstanding receiver-issued v2 session recovery challenges keyed by `(SwarmInstanceId, RunId, PeerId)`.
    pending_challenges_v2: BTreeMap<SessionScopeV2Key, SessionHandshakeChallengeV2>,
    /// Optional cohort membership allowlist.
    cohort_allowlist: Option<CohortAllowlist>,
    /// Startup/restart admission policy (`RequireSessionHandshake` by default).
    startup_policy: ReplayStartupPolicyV1,
    /// Optional monotonic Unix-second restart timestamp floor.
    timestamp_floor_secs: Option<u32>,
    /// Maximum number of active replay records across v1 + v2 tables.
    capacity: usize,
    /// Maximum number of pending recovery challenges across v1 + v2 tables.
    max_pending_challenges: usize,
    /// Maximum clock skew tolerance (seconds).
    max_clock_skew_secs: u32,
    /// Sequence tolerance window (messages).
    tolerance_window: usize,
}

#[cfg(feature = "alloc")]
impl ReplayProtection {
    /// Create with default configuration (`capacity = 1000`, `max_clock_skew_secs = 60`, `window = 16`,
    /// `startup_policy = RequireSessionHandshake`).
    pub fn new() -> Self {
        Self {
            records: BTreeMap::new(),
            records_v2: BTreeMap::new(),
            peer_epochs: BTreeMap::new(),
            active_sessions_v2: BTreeMap::new(),
            expiry_index: BTreeMap::new(),
            expiry_index_v2: BTreeMap::new(),
            pending_challenges: BTreeMap::new(),
            pending_challenges_v2: BTreeMap::new(),
            cohort_allowlist: None,
            startup_policy: ReplayStartupPolicyV1::RequireSessionHandshake,
            timestamp_floor_secs: None,
            capacity: DEFAULT_CACHE_CAPACITY,
            max_pending_challenges: DEFAULT_MAX_PENDING_CHALLENGES,
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
            records_v2: BTreeMap::new(),
            peer_epochs: BTreeMap::new(),
            active_sessions_v2: BTreeMap::new(),
            expiry_index: BTreeMap::new(),
            expiry_index_v2: BTreeMap::new(),
            pending_challenges: BTreeMap::new(),
            pending_challenges_v2: BTreeMap::new(),
            cohort_allowlist: None,
            startup_policy: ReplayStartupPolicyV1::RequireSessionHandshake,
            timestamp_floor_secs: None,
            capacity,
            max_pending_challenges: DEFAULT_MAX_PENDING_CHALLENGES,
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

    /// Configure the [`ReplayStartupPolicyV1`] (`RequireSessionHandshake` by default).
    pub fn with_startup_policy(mut self, policy: ReplayStartupPolicyV1) -> Self {
        self.startup_policy = policy;
        self
    }

    /// Update the [`ReplayStartupPolicyV1`] in place.
    pub fn set_startup_policy(&mut self, policy: ReplayStartupPolicyV1) {
        self.startup_policy = policy;
    }

    /// Inspect the configured [`ReplayStartupPolicyV1`].
    pub fn startup_policy(&self) -> ReplayStartupPolicyV1 {
        self.startup_policy
    }

    /// Configure a monotonic Unix-second restart timestamp floor.
    pub fn with_timestamp_floor(mut self, floor_secs: u32) -> Self {
        self.timestamp_floor_secs = Some(floor_secs);
        self
    }

    /// Set or clear the monotonic Unix-second restart timestamp floor.
    pub fn set_timestamp_floor(&mut self, floor_secs: Option<u32>) {
        self.timestamp_floor_secs = floor_secs;
    }

    /// Inspect the configured restart timestamp floor, if any.
    pub fn timestamp_floor_secs(&self) -> Option<u32> {
        self.timestamp_floor_secs
    }

    /// Configure a custom upper bound on simultaneously pending recovery challenges.
    pub fn try_with_max_pending_challenges(
        mut self,
        max_pending_challenges: usize,
    ) -> Result<Self, ReplayConfigError> {
        if max_pending_challenges == 0 {
            return Err(ReplayConfigError::ZeroPendingChallengesCapacity);
        }
        self.max_pending_challenges = max_pending_challenges;
        Ok(self)
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
            Err(ReplayConfigError::ToleranceWindowTooLarge { .. })
            | Err(ReplayConfigError::ZeroPendingChallengesCapacity) => {
                unreachable!(
                    "try_with_config uses default tolerance window and pending challenge cap"
                )
            }
        }
    }

    fn check_timestamp_floor(&self, ts: u32) -> Result<(), ReplayError> {
        if let Some(floor) = self.timestamp_floor_secs {
            if ts < floor {
                return Err(ReplayError::TimestampBelowFloor { ts, floor });
            }
        }
        Ok(())
    }

    /// Check timestamp validity without mutating state (fail-fast optimization).
    pub fn check_timestamp_only(&self, ts: u32, now: u32) -> Result<(), ReplayError> {
        self.check_timestamp_floor(ts)?;
        self.check_timestamp_floor(now)?;
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

    /// Check session-epoch continuity for a v1 `key` without mutating state.
    ///
    /// Enforces [`ReplayStartupPolicyV1::RequireSessionHandshake`] when no session is
    /// currently established for `(swarm_instance_id, peer)`.
    pub fn check_epoch_continuity(&self, key: &ReplayRecordKey) -> Result<(), ReplayError> {
        match self.peer_epochs.get(&(key.swarm_instance_id, key.peer)) {
            Some(&(active_epoch, _)) => {
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
                Ok(())
            }
            None => match self.startup_policy {
                ReplayStartupPolicyV1::RequireSessionHandshake => {
                    Err(ReplayError::RecoveryRequired {
                        peer: key.peer,
                        current_epoch: 0,
                        requested_epoch: key.session_epoch,
                    })
                }
                ReplayStartupPolicyV1::GenesisRosterBootstrap => Ok(()),
            },
        }
    }

    /// Check session-id and epoch continuity for a v2 `key` without mutating state (`ST-03`).
    pub fn check_v2_session_continuity(&self, key: &ReplayRecordKeyV2) -> Result<(), ReplayError> {
        if key.session_id == [0u8; 32] {
            return Err(ReplayError::SessionIdMismatch { peer: key.peer });
        }
        match self
            .active_sessions_v2
            .get(&(key.swarm_instance_id, key.run_id, key.peer))
        {
            Some(&(active_epoch, active_session_id, _)) => {
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
                if key.session_id != active_session_id {
                    return Err(ReplayError::SessionIdMismatch { peer: key.peer });
                }
                Ok(())
            }
            None => match self.startup_policy {
                ReplayStartupPolicyV1::RequireSessionHandshake => {
                    Err(ReplayError::RecoveryRequired {
                        peer: key.peer,
                        current_epoch: 0,
                        requested_epoch: key.session_epoch,
                    })
                }
                ReplayStartupPolicyV1::GenesisRosterBootstrap => Ok(()),
            },
        }
    }

    /// Inspect active v2 session `(session_epoch, session_id)` if established.
    pub fn active_v2_session(
        &self,
        swarm_instance_id: SwarmInstanceId,
        run_id: [u8; 16],
        peer: &PeerId,
    ) -> Option<(SessionEpoch, [u8; 32])> {
        self.active_sessions_v2
            .get(&(swarm_instance_id, run_id, *peer))
            .map(|&(epoch, sid, _)| (epoch, sid))
    }

    fn total_records(&self) -> usize {
        self.records.len().saturating_add(self.records_v2.len())
    }

    fn count_expired(&self, current_time: u32) -> usize {
        let now_u64 = current_time as u64;
        let v1 = self
            .expiry_index
            .keys()
            .take_while(|(expires_at, _)| *expires_at < now_u64)
            .count();
        let v2 = self
            .expiry_index_v2
            .keys()
            .take_while(|(expires_at, _)| *expires_at < now_u64)
            .count();
        v1.saturating_add(v2)
    }

    /// Read-only preflight for v1 records (`ST-01`, `ST-03`).
    ///
    /// Performs ALL authorization, timestamp, floor, epoch/policy, sequence window, and
    /// capacity checks without mutating any state.
    pub fn check_v1_admission_readonly(
        &self,
        key: &ReplayRecordKey,
        seq: u64,
        timestamp: u32,
        current_time: u32,
    ) -> Result<PreparedReplayAdmission, ReplayError> {
        self.check_timestamp_only(timestamp, current_time)?;
        self.check_v1_sequence_readonly_inner(key, seq, timestamp, current_time)
    }

    fn check_v1_sequence_readonly_inner(
        &self,
        key: &ReplayRecordKey,
        seq: u64,
        timestamp: u32,
        current_time: u32,
    ) -> Result<PreparedReplayAdmission, ReplayError> {
        self.check_timestamp_floor(timestamp)?;
        self.check_peer_authorized(&key.peer)?;
        self.check_epoch_continuity(key)?;

        let now_u64 = current_time as u64;
        if let Some(state) = self.records.get(key) {
            if state.expires_at >= now_u64 {
                state.check_can_accept(seq, key.peer, self.tolerance_window)?;
                return Ok(PreparedReplayAdmission {
                    key: PreparedReplayKey::V1(*key),
                    sequence: seq,
                    timestamp,
                    current_time,
                });
            }
        }

        let projected_used = self
            .total_records()
            .saturating_sub(self.count_expired(current_time));
        if projected_used >= self.capacity {
            return Err(ReplayError::CacheFull {
                capacity: self.capacity,
            });
        }

        Ok(PreparedReplayAdmission {
            key: PreparedReplayKey::V1(*key),
            sequence: seq,
            timestamp,
            current_time,
        })
    }

    /// Read-only preflight for session-id-bound v2 records (`ST-01`, `ST-03`).
    ///
    /// Performs ALL authorization, timestamp, floor, session continuity, `session_id`, sequence
    /// window, and capacity checks without mutating any state.
    pub fn check_v2_admission_readonly(
        &self,
        key: &ReplayRecordKeyV2,
        seq: u64,
        timestamp: u32,
        current_time: u32,
    ) -> Result<PreparedReplayAdmission, ReplayError> {
        self.check_timestamp_only(timestamp, current_time)?;
        self.check_peer_authorized(&key.peer)?;
        self.check_v2_session_continuity(key)?;

        let now_u64 = current_time as u64;
        if let Some(state) = self.records_v2.get(key) {
            if state.expires_at >= now_u64 {
                state.check_can_accept(seq, key.peer, self.tolerance_window)?;
                return Ok(PreparedReplayAdmission {
                    key: PreparedReplayKey::V2(*key),
                    sequence: seq,
                    timestamp,
                    current_time,
                });
            }
        }

        let projected_used = self
            .total_records()
            .saturating_sub(self.count_expired(current_time));
        if projected_used >= self.capacity {
            return Err(ReplayError::CacheFull {
                capacity: self.capacity,
            });
        }

        Ok(PreparedReplayAdmission {
            key: PreparedReplayKey::V2(*key),
            sequence: seq,
            timestamp,
            current_time,
        })
    }

    /// Atomically re-verify and commit a [`PreparedReplayAdmission`] token (`ST-01`, `ST-03`).
    pub fn commit_prepared_admission(
        &mut self,
        prepared: PreparedReplayAdmission,
    ) -> Result<(), ReplayError> {
        let tw = self.tolerance_window;
        let skew = self.max_clock_skew_secs;
        match prepared.key {
            PreparedReplayKey::V1(key) => {
                // Re-check all preconditions atomically at commit time (TOCTOU-safe).
                self.check_timestamp_only(prepared.timestamp, prepared.current_time)?;
                self.check_peer_authorized(&key.peer)?;
                self.check_epoch_continuity(&key)?;
                self.reclaim_expired(prepared.current_time);
                self.commit_v1_unchecked(&key, prepared.sequence, prepared.timestamp, tw, skew)
            }
            PreparedReplayKey::V2(key) => {
                self.check_timestamp_only(prepared.timestamp, prepared.current_time)?;
                self.check_peer_authorized(&key.peer)?;
                self.check_v2_session_continuity(&key)?;
                self.reclaim_expired(prepared.current_time);
                self.commit_v2_unchecked(&key, prepared.sequence, prepared.timestamp, tw, skew)
            }
        }
    }

    fn commit_v1_unchecked(
        &mut self,
        key: &ReplayRecordKey,
        seq: u64,
        timestamp: u32,
        tw: usize,
        skew: u32,
    ) -> Result<(), ReplayError> {
        if let Some(state) = self.records.get_mut(key) {
            let old_expires = state.expires_at;
            state.validate_and_update(seq, key.peer, tw)?;
            if timestamp > state.max_timestamp_seen {
                state.max_timestamp_seen = timestamp;
                state.expires_at = (timestamp as u64).saturating_add(skew as u64);
                self.expiry_index.remove(&(old_expires, *key));
                self.expiry_index.insert((state.expires_at, *key), ());
            }
            self.peer_epochs.insert(
                (key.swarm_instance_id, key.peer),
                (key.session_epoch, state.expires_at),
            );
            return Ok(());
        }

        if self.total_records() >= self.capacity {
            return Err(ReplayError::CacheFull {
                capacity: self.capacity,
            });
        }

        let epoch_key = (key.swarm_instance_id, key.peer);
        if self.peer_epochs.len() >= self.capacity && !self.peer_epochs.contains_key(&epoch_key) {
            self.prune_oldest_expired_v1_session(timestamp as u64);
        }

        let state = PeerReplayState::new(seq, timestamp, skew);
        let expires_at = state.expires_at;
        self.records.insert(*key, state);
        self.peer_epochs
            .insert(epoch_key, (key.session_epoch, expires_at));
        self.expiry_index.insert((expires_at, *key), ());
        Ok(())
    }

    fn commit_v2_unchecked(
        &mut self,
        key: &ReplayRecordKeyV2,
        seq: u64,
        timestamp: u32,
        tw: usize,
        skew: u32,
    ) -> Result<(), ReplayError> {
        let session_map_key = (key.swarm_instance_id, key.run_id, key.peer);
        if let Some(state) = self.records_v2.get_mut(key) {
            let old_expires = state.expires_at;
            state.validate_and_update(seq, key.peer, tw)?;
            if timestamp > state.max_timestamp_seen {
                state.max_timestamp_seen = timestamp;
                state.expires_at = (timestamp as u64).saturating_add(skew as u64);
                self.expiry_index_v2.remove(&(old_expires, *key));
                self.expiry_index_v2.insert((state.expires_at, *key), ());
            }
            self.active_sessions_v2.insert(
                session_map_key,
                (key.session_epoch, key.session_id, state.expires_at),
            );
            return Ok(());
        }

        if self.total_records() >= self.capacity {
            return Err(ReplayError::CacheFull {
                capacity: self.capacity,
            });
        }

        if self.active_sessions_v2.len() >= self.capacity
            && !self.active_sessions_v2.contains_key(&session_map_key)
        {
            self.prune_oldest_expired_v2_session(timestamp as u64);
        }

        let state = PeerReplayState::new(seq, timestamp, skew);
        let expires_at = state.expires_at;
        self.records_v2.insert(*key, state);
        self.active_sessions_v2.insert(
            session_map_key,
            (key.session_epoch, key.session_id, expires_at),
        );
        self.expiry_index_v2.insert((expires_at, *key), ());
        Ok(())
    }

    fn prune_oldest_expired_v1_session(&mut self, cutoff_u64: u64) {
        let victim = self
            .peer_epochs
            .iter()
            .filter(|(&(swarm_id, peer), &(epoch, exp))| {
                exp < cutoff_u64
                    && !self
                        .records
                        .contains_key(&ReplayRecordKey::new(swarm_id, peer, epoch))
            })
            .min_by_key(|(_, &(_, exp))| exp)
            .map(|(&k, _)| k);
        if let Some(k) = victim {
            self.peer_epochs.remove(&k);
        }
    }

    fn prune_oldest_expired_v2_session(&mut self, cutoff_u64: u64) {
        let victim = self
            .active_sessions_v2
            .iter()
            .filter(|(&(swarm_id, run_id, peer), &(epoch, sid, exp))| {
                exp < cutoff_u64
                    && !self
                        .records_v2
                        .contains_key(&ReplayRecordKeyV2::new(swarm_id, run_id, peer, epoch, sid))
            })
            .min_by_key(|(_, &(_, _, exp))| exp)
            .map(|(&k, _)| k);
        if let Some(k) = victim {
            self.active_sessions_v2.remove(&k);
        }
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
            reclaimed += 1;
        }

        while let Some((&(expires_at, key), _)) = self.expiry_index_v2.iter().next() {
            if expires_at >= now_u64 {
                break;
            }
            self.expiry_index_v2.remove(&(expires_at, key));
            self.records_v2.remove(&key);
            reclaimed += 1;
        }

        reclaimed
    }

    /// Validate sequence number and update scoped v1 state at `(timestamp, current_time)`.
    pub fn validate_sequence_scoped_at(
        &mut self,
        key: &ReplayRecordKey,
        seq: u64,
        timestamp: u32,
        current_time: u32,
    ) -> Result<(), ReplayError> {
        let prepared = self.check_v1_sequence_readonly_inner(key, seq, timestamp, current_time)?;
        self.reclaim_expired(prepared.current_time);
        let tw = self.tolerance_window;
        let skew = self.max_clock_skew_secs;
        self.commit_v1_unchecked(key, prepared.sequence, prepared.timestamp, tw, skew)
    }

    /// Validate sequence number and update state (default single-swarm scope).
    pub fn validate_sequence(&mut self, peer: &PeerId, seq: u64) -> Result<(), ReplayError> {
        let key = ReplayRecordKey::default_for_peer(*peer);
        self.validate_sequence_scoped_at(&key, seq, 0, 0)
    }

    /// Full scoped v1 validation: timestamp + floor + allowlist + epoch continuity + sequence.
    pub fn validate_scoped(
        &mut self,
        key: &ReplayRecordKey,
        sequence: u64,
        timestamp: u32,
        current_time: u32,
    ) -> Result<(), ReplayError> {
        let prepared = self.check_v1_admission_readonly(key, sequence, timestamp, current_time)?;
        self.commit_prepared_admission(prepared)
    }

    /// Full v1 validation: timestamp + sequence (default single-swarm scope).
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

    /// Full scoped v2 validation: timestamp + floor + allowlist + session_id + sequence (`ST-03`).
    pub fn validate_session_v2_at(
        &mut self,
        key: &ReplayRecordKeyV2,
        sequence: u64,
        timestamp: u32,
        current_time: u32,
    ) -> Result<(), ReplayError> {
        let prepared = self.check_v2_admission_readonly(key, sequence, timestamp, current_time)?;
        self.commit_prepared_admission(prepared)
    }

    /// Register a receiver-generated 256-bit v1 challenge for a restarting peer (`ADR-0021`, `W9-02`).
    pub fn register_recovery_challenge(
        &mut self,
        challenge: SessionHandshakeChallenge,
    ) -> Result<(), ReplayError> {
        self.check_peer_authorized(&challenge.sender_peer)?;
        if challenge.challenge_nonce == [0u8; 32] {
            return Err(ReplayError::HandshakeVerificationFailed);
        }
        if let Some(&(active_epoch, _)) = self
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
        let challenge_key = (challenge.swarm_instance_id, challenge.sender_peer);
        if !self.pending_challenges.contains_key(&challenge_key)
            && self.pending_challenge_count() >= self.max_pending_challenges
        {
            return Err(ReplayError::PendingChallengesFull {
                capacity: self.max_pending_challenges,
            });
        }
        self.pending_challenges.insert(challenge_key, challenge);
        Ok(())
    }

    /// Register a receiver-generated 256-bit v2 challenge binding `run_id` and `policy_digest` (`ST-03`).
    pub fn register_recovery_challenge_v2(
        &mut self,
        challenge: SessionHandshakeChallengeV2,
    ) -> Result<(), ReplayError> {
        self.check_peer_authorized(&challenge.sender_peer)?;
        if challenge.receiver_nonce == [0u8; 32] || challenge.requested_epoch == 0 {
            return Err(ReplayError::HandshakeVerificationFailed);
        }
        let session_key = (
            challenge.swarm_instance_id,
            challenge.run_id,
            challenge.sender_peer,
        );
        if let Some(&(active_epoch, _, _)) = self.active_sessions_v2.get(&session_key) {
            if challenge.requested_epoch <= active_epoch {
                return Err(ReplayError::StaleSessionEpoch {
                    peer: challenge.sender_peer,
                    current_epoch: active_epoch,
                    received_epoch: challenge.requested_epoch,
                });
            }
        }
        if !self.pending_challenges_v2.contains_key(&session_key)
            && self.pending_challenge_count() >= self.max_pending_challenges
        {
            return Err(ReplayError::PendingChallengesFull {
                capacity: self.max_pending_challenges,
            });
        }
        self.pending_challenges_v2.insert(session_key, challenge);
        Ok(())
    }

    /// Complete a receiver-challenged v1 session recovery handshake atomically (`swarmtorch.session.handshake.v1`).
    ///
    /// No pending challenge, epoch mapping, or replay record is mutated until all signature,
    /// expiry, epoch, and projected capacity checks succeed.
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

        let challenge_key = (swarm_instance_id, sender_peer);
        let challenge = self
            .pending_challenges
            .get(&challenge_key)
            .copied()
            .ok_or(ReplayError::HandshakeVerificationFailed)?;

        self.check_timestamp_only(challenge.issued_at_secs, current_time)?;
        self.check_timestamp_floor(current_time)?;

        let existing_epoch = self.peer_epochs.get(&challenge_key).copied();
        if let Some((active_epoch, _)) = existing_epoch {
            if challenge.requested_epoch <= active_epoch {
                return Err(ReplayError::StaleSessionEpoch {
                    peer: sender_peer,
                    current_epoch: active_epoch,
                    received_epoch: challenge.requested_epoch,
                });
            }
        }

        MessageAuth::verify_session_handshake(sender_public_key, &challenge, signature)
            .map_err(|_| ReplayError::HandshakeVerificationFailed)?;

        // Dry-run capacity BEFORE mutating pending_challenges, peer_epochs, or records.
        let now_u64 = current_time as u64;
        let expired_count = self.count_expired(current_time);
        let superseded_live_slot = if let Some((old_epoch, _)) = existing_epoch {
            let old_key = ReplayRecordKey::new(swarm_instance_id, sender_peer, old_epoch);
            match self.records.get(&old_key) {
                Some(old_state) if old_state.expires_at >= now_u64 => 1usize,
                _ => 0usize,
            }
        } else {
            0usize
        };
        let projected_used = self
            .total_records()
            .saturating_sub(expired_count)
            .saturating_sub(superseded_live_slot);
        if projected_used >= self.capacity {
            return Err(ReplayError::CacheFull {
                capacity: self.capacity,
            });
        }

        // All validations and capacity checks passed — commit atomically.
        self.pending_challenges.remove(&challenge_key);
        self.reclaim_expired(current_time);

        if let Some((old_epoch, _)) = existing_epoch {
            let old_key = ReplayRecordKey::new(swarm_instance_id, sender_peer, old_epoch);
            if let Some(old_state) = self.records.remove(&old_key) {
                self.expiry_index.remove(&(old_state.expires_at, old_key));
            }
        }

        if self.peer_epochs.len() >= self.capacity && !self.peer_epochs.contains_key(&challenge_key)
        {
            self.prune_oldest_expired_v1_session(now_u64);
        }

        let new_key =
            ReplayRecordKey::new(swarm_instance_id, sender_peer, challenge.requested_epoch);
        let state = PeerReplayState::new(
            challenge.initial_sequence,
            current_time,
            self.max_clock_skew_secs,
        );
        let expires_at = state.expires_at;
        self.records.insert(new_key, state);
        self.peer_epochs
            .insert(challenge_key, (challenge.requested_epoch, expires_at));
        self.expiry_index.insert((expires_at, new_key), ());
        Ok(new_key)
    }

    /// Complete a receiver-challenged v2 session recovery handshake atomically (`swarmtorch.session.handshake.v2`, `ST-03`).
    ///
    /// Derives and activates the receiver-bound `session_id` only after all signature, expiry,
    /// epoch, and projected capacity checks pass. Zero state is mutated on error.
    pub fn complete_recovery_handshake_v2(
        &mut self,
        swarm_instance_id: SwarmInstanceId,
        run_id: [u8; 16],
        sender_public_key: &[u8; 32],
        signature: &Signature,
        current_time: u32,
    ) -> Result<ReplayRecordKeyV2, ReplayError> {
        let sender_peer = PeerId::try_from_public_key_bytes(sender_public_key)
            .map_err(|_| ReplayError::HandshakeVerificationFailed)?;
        self.check_peer_authorized(&sender_peer)?;

        let session_key = (swarm_instance_id, run_id, sender_peer);
        let challenge = self
            .pending_challenges_v2
            .get(&session_key)
            .copied()
            .ok_or(ReplayError::HandshakeVerificationFailed)?;

        self.check_timestamp_only(challenge.issued_at_secs, current_time)?;
        self.check_timestamp_floor(current_time)?;

        let existing_session = self.active_sessions_v2.get(&session_key).copied();
        if let Some((active_epoch, _, _)) = existing_session {
            if challenge.requested_epoch <= active_epoch {
                return Err(ReplayError::StaleSessionEpoch {
                    peer: sender_peer,
                    current_epoch: active_epoch,
                    received_epoch: challenge.requested_epoch,
                });
            }
        }

        let session_id =
            MessageAuth::verify_session_handshake_v2(sender_public_key, &challenge, signature)
                .map_err(|_| ReplayError::HandshakeVerificationFailed)?;

        // Dry-run capacity BEFORE mutating pending_challenges_v2, active_sessions_v2, or records_v2.
        let now_u64 = current_time as u64;
        let expired_count = self.count_expired(current_time);
        let superseded_live_slot = if let Some((old_epoch, old_sid, _)) = existing_session {
            let old_key =
                ReplayRecordKeyV2::new(swarm_instance_id, run_id, sender_peer, old_epoch, old_sid);
            match self.records_v2.get(&old_key) {
                Some(old_state) if old_state.expires_at >= now_u64 => 1usize,
                _ => 0usize,
            }
        } else {
            0usize
        };
        let projected_used = self
            .total_records()
            .saturating_sub(expired_count)
            .saturating_sub(superseded_live_slot);
        if projected_used >= self.capacity {
            return Err(ReplayError::CacheFull {
                capacity: self.capacity,
            });
        }

        // All validations and capacity checks passed — commit atomically.
        self.pending_challenges_v2.remove(&session_key);
        self.reclaim_expired(current_time);

        if let Some((old_epoch, old_sid, _)) = existing_session {
            let old_key =
                ReplayRecordKeyV2::new(swarm_instance_id, run_id, sender_peer, old_epoch, old_sid);
            if let Some(old_state) = self.records_v2.remove(&old_key) {
                self.expiry_index_v2
                    .remove(&(old_state.expires_at, old_key));
            }
        }

        if self.active_sessions_v2.len() >= self.capacity
            && !self.active_sessions_v2.contains_key(&session_key)
        {
            self.prune_oldest_expired_v2_session(now_u64);
        }

        let new_key = ReplayRecordKeyV2::new(
            swarm_instance_id,
            run_id,
            sender_peer,
            challenge.requested_epoch,
            session_id,
        );
        let state = PeerReplayState::new(
            challenge.initial_sequence,
            current_time,
            self.max_clock_skew_secs,
        );
        let expires_at = state.expires_at;
        self.records_v2.insert(new_key, state);
        self.active_sessions_v2.insert(
            session_key,
            (challenge.requested_epoch, session_id, expires_at),
        );
        self.expiry_index_v2.insert((expires_at, new_key), ());
        Ok(new_key)
    }

    fn is_timestamp_valid(&self, ts: u32, now: u32) -> bool {
        now.abs_diff(ts) <= self.max_clock_skew_secs
    }

    /// Get current active record count across v1 + v2 tables (for testing/monitoring).
    pub fn cache_size(&self) -> usize {
        self.total_records()
    }

    /// Get count of currently pending v1 + v2 recovery challenges.
    pub fn pending_challenge_count(&self) -> usize {
        self.pending_challenges
            .len()
            .saturating_add(self.pending_challenges_v2.len())
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

    fn check_can_accept(
        &self,
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

        Ok(())
    }

    fn validate_and_update(
        &mut self,
        seq: u64,
        peer: PeerId,
        tolerance_window: usize,
    ) -> Result<(), ReplayError> {
        self.check_can_accept(seq, peer, tolerance_window)?;

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
    /// Pending uncompleted recovery challenges table is full.
    PendingChallengesFull {
        /// Configured maximum pending recovery challenges.
        capacity: usize,
    },
    /// Message `session_id` did not match the established receiver-bound session.
    SessionIdMismatch {
        /// Peer ID.
        peer: PeerId,
    },
    /// Timestamp is older than the monotonic restart timestamp floor.
    TimestampBelowFloor {
        /// Received timestamp.
        ts: u32,
        /// Enforced restart timestamp floor.
        floor: u32,
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
            ReplayError::PendingChallengesFull { capacity } => {
                write!(
                    f,
                    "pending recovery challenges table full at capacity={}",
                    capacity
                )
            }
            ReplayError::SessionIdMismatch { peer } => {
                write!(f, "session_id mismatch for peer={:?}", peer)
            }
            ReplayError::TimestampBelowFloor { ts, floor } => {
                write!(
                    f,
                    "timestamp {} is below monotonic restart floor {}",
                    ts, floor
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ReplayError {}

#[cfg(all(test, feature = "alloc"))]
#[path = "replay_tests.rs"]
mod tests;
