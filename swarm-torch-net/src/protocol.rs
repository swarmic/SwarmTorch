//! Message protocol, versioned envelope signing, and 6-stage atomic receive pipeline (`ADR-0021`, `W9-02`).

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
#[cfg(feature = "alloc")]
use swarm_torch_core::crypto::{EnvelopeContextV1, EnvelopeContextV2, MessageAuth};
#[cfg(feature = "alloc")]
use swarm_torch_core::replay::{
    PreparedReplayAdmission, ReplayProtection, ReplayRecordKey, ReplayRecordKeyV2,
};
use swarm_torch_core::traits::PeerId;

/// Default pre-deserialization wire frame bound (`1,048,576` bytes = 1 MiB).
pub const DEFAULT_MAX_ENVELOPE_BYTES: usize = 1_048_576;

/// Explicit wire-version admission policy for [`MessageEnvelope`] verification (`Decision A10`).
///
/// Defaults to [`Self::RequireV2`] (`VERSION_V2 = (0, 3)`) so legacy `VERSION_V0` and
/// `VERSION_V1` envelopes fail closed unless an integration caller explicitly opts into a legacy
/// transition profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnvelopeVersionPolicyV1 {
    /// Strict `VERSION_V2` (`(0, 3)`) only (fail-closed default).
    #[default]
    RequireV2,
    /// Permit `VERSION_V1` (`(0, 2)`) alongside `VERSION_V2` (`(0, 3)`); reject `VERSION_V0`.
    AllowLegacyV1AndV2,
    /// Permit legacy `VERSION_V0` (`(0, 1)`), `VERSION_V1` (`(0, 2)`), and `VERSION_V2` (`(0, 3)`).
    AllowLegacyAll,
}

impl EnvelopeVersionPolicyV1 {
    /// Return `true` iff `version` is permitted under this admission policy.
    pub const fn allows_version(self, version: (u8, u8)) -> bool {
        match self {
            Self::RequireV2 => matches!(version, MessageEnvelope::VERSION_V2),
            Self::AllowLegacyV1AndV2 => matches!(
                version,
                MessageEnvelope::VERSION_V1 | MessageEnvelope::VERSION_V2
            ),
            Self::AllowLegacyAll => matches!(
                version,
                MessageEnvelope::VERSION_V0
                    | MessageEnvelope::VERSION_V1
                    | MessageEnvelope::VERSION_V2
            ),
        }
    }
}

/// Message envelope for all swarm communications (`ADR-0008`, `ADR-0021`, `W9-02`, `ST-03`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEnvelope {
    /// Protocol version `(major, minor)`.
    pub version: (u8, u8),
    /// Message type discriminator.
    pub message_type: MessageType,
    /// Sender's Ed25519 public key bytes (raw 32 bytes).
    pub sender: [u8; 32],
    /// Monotonic sequence number (replay protection).
    pub sequence: u64,
    /// Unix timestamp (seconds, for expiry).
    pub timestamp: u32,
    /// Scoped swarm deployment / cohort instance identifier (`VERSION_V1`, `VERSION_V2`).
    #[serde(default)]
    pub swarm_instance_id: [u8; 16],
    /// Bound training/coordination run identifier (`VERSION_V2`).
    #[serde(default)]
    pub run_id: [u8; 16],
    /// Monotonically increasing sender session epoch (`VERSION_V1`, `VERSION_V2`).
    #[serde(default)]
    pub session_epoch: u64,
    /// Receiver-nonce-derived session identifier (`VERSION_V2`).
    #[serde(default)]
    pub session_id: [u8; 32],
    /// Training / coordination round identifier (`VERSION_V1`, `VERSION_V2`).
    #[serde(default)]
    pub round_id: u64,
    /// Payload bytes.
    #[cfg(feature = "alloc")]
    pub payload: Vec<u8>,
    /// Optional cryptographic signature (64 bytes for Ed25519).
    #[cfg(feature = "alloc")]
    pub signature: Option<alloc::vec::Vec<u8>>,
}

impl MessageEnvelope {
    /// Legacy v0 envelope version (`swarmtorch.envelope.v0`).
    pub const VERSION_V0: (u8, u8) = (0, 1);
    /// Scoped v1 envelope version (`swarmtorch.envelope.v1`).
    pub const VERSION_V1: (u8, u8) = (0, 2);
    /// Receiver-session-bound v2 envelope version (`swarmtorch.envelope.v2`, `ST-03`).
    pub const VERSION_V2: (u8, u8) = (0, 3);
    /// Current default protocol version for [`MessageEnvelope`] (`VERSION_V2 = (0, 3)`).
    pub const CURRENT_VERSION: (u8, u8) = Self::VERSION_V2;
    /// Supported protocol versions across strict (`v2`) and explicit legacy compatibility policies.
    pub const SUPPORTED_VERSIONS: &'static [(u8, u8)] =
        &[Self::VERSION_V0, Self::VERSION_V1, Self::VERSION_V2];

    /// Create a new message envelope defaulting to [`Self::CURRENT_VERSION`] (`VERSION_V2 = (0, 3)`).
    ///
    /// Prefer [`Self::new_v2`] in production to bind `swarm_instance_id`, `run_id`, `session_epoch`,
    /// `session_id`, and `round_id` directly, or [`Self::new_legacy_v0`] / [`Self::new_v1`] when
    /// explicitly targeting legacy migration compatibility.
    #[cfg(feature = "alloc")]
    pub fn new_with_public_key(
        sender_public_key: [u8; 32],
        message_type: MessageType,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            message_type,
            sender: sender_public_key,
            sequence: 0,
            timestamp: 0,
            swarm_instance_id: [0u8; 16],
            run_id: [0u8; 16],
            session_epoch: 0,
            session_id: [0u8; 32],
            round_id: 0,
            payload,
            signature: None,
        }
    }

    /// Explicitly construct a legacy unscoped v0 message envelope (`VERSION_V0 = (0, 1)`).
    ///
    /// Legacy v0 envelopes are rejected by default under [`EnvelopeVersionPolicyV1::RequireV2`]
    /// unless the verifier explicitly opts into [`EnvelopeVersionPolicyV1::AllowLegacyAll`].
    #[cfg(feature = "alloc")]
    pub fn new_legacy_v0(
        sender_public_key: [u8; 32],
        message_type: MessageType,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            version: Self::VERSION_V0,
            message_type,
            sender: sender_public_key,
            sequence: 0,
            timestamp: 0,
            swarm_instance_id: [0u8; 16],
            run_id: [0u8; 16],
            session_epoch: 0,
            session_id: [0u8; 32],
            round_id: 0,
            payload,
            signature: None,
        }
    }

    /// Create a scoped v1 message envelope (`VERSION_V1 = (0, 2)`, `swarmtorch.envelope.v1`).
    #[cfg(feature = "alloc")]
    pub fn new_v1(
        sender_public_key: [u8; 32],
        swarm_instance_id: [u8; 16],
        session_epoch: u64,
        round_id: u64,
        message_type: MessageType,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            version: Self::VERSION_V1,
            message_type,
            sender: sender_public_key,
            sequence: 0,
            timestamp: 0,
            swarm_instance_id,
            run_id: [0u8; 16],
            session_epoch,
            session_id: [0u8; 32],
            round_id,
            payload,
            signature: None,
        }
    }

    /// Create a session-id-bound v2 message envelope (`VERSION_V2 = (0, 3)`, `swarmtorch.envelope.v2`, `ST-03`).
    #[cfg(feature = "alloc")]
    #[allow(clippy::too_many_arguments)]
    pub fn new_v2(
        sender_public_key: [u8; 32],
        swarm_instance_id: [u8; 16],
        run_id: [u8; 16],
        session_epoch: u64,
        session_id: [u8; 32],
        round_id: u64,
        message_type: MessageType,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            version: Self::VERSION_V2,
            message_type,
            sender: sender_public_key,
            sequence: 0,
            timestamp: 0,
            swarm_instance_id,
            run_id,
            session_epoch,
            session_id,
            round_id,
            payload,
            signature: None,
        }
    }

    /// Returns `true` when this envelope version is supported.
    pub fn is_version_supported(&self) -> bool {
        Self::SUPPORTED_VERSIONS.contains(&self.version)
    }

    /// Get sender public key bytes.
    pub const fn sender_public_key(&self) -> &[u8; 32] {
        &self.sender
    }

    /// Derive sender `PeerId` from sender public key bytes.
    pub fn sender_peer_id(
        &self,
    ) -> core::result::Result<PeerId, swarm_torch_core::crypto::CryptoError> {
        PeerId::try_from_public_key_bytes(&self.sender)
    }

    /// Get current Unix time in seconds.
    #[cfg(feature = "std")]
    pub fn current_unix_secs() -> Result<u32, TimeError> {
        use std::time::{SystemTime, UNIX_EPOCH};

        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| TimeError::BeforeEpoch)?;
        let seconds = duration.as_secs();
        u32::try_from(seconds).map_err(|_| TimeError::Overflow)
    }

    /// Set the sequence number.
    pub const fn with_sequence(mut self, seq: u64) -> Self {
        self.sequence = seq;
        self
    }

    /// Set the timestamp.
    pub const fn with_timestamp(mut self, ts: u32) -> Self {
        self.timestamp = ts;
        self
    }

    /// Set the swarm instance identifier (`VERSION_V1`).
    pub const fn with_swarm_instance_id(mut self, swarm_instance_id: [u8; 16]) -> Self {
        self.swarm_instance_id = swarm_instance_id;
        self
    }

    /// Set the sender session epoch (`VERSION_V1`).
    pub const fn with_session_epoch(mut self, session_epoch: u64) -> Self {
        self.session_epoch = session_epoch;
        self
    }

    /// Set the round identifier (`VERSION_V1`, `VERSION_V2`).
    pub const fn with_round_id(mut self, round_id: u64) -> Self {
        self.round_id = round_id;
        self
    }

    /// Set the bound training/coordination run identifier (`VERSION_V2`).
    pub const fn with_run_id(mut self, run_id: [u8; 16]) -> Self {
        self.run_id = run_id;
        self
    }

    /// Set the receiver-nonce-derived session identifier (`VERSION_V2`).
    pub const fn with_session_id(mut self, session_id: [u8; 32]) -> Self {
        self.session_id = session_id;
        self
    }

    /// Construct the [`EnvelopeContextV1`] bound by `swarmtorch.envelope.v1`.
    #[cfg(feature = "alloc")]
    pub const fn envelope_context_v1(&self) -> EnvelopeContextV1 {
        EnvelopeContextV1 {
            version: self.version,
            swarm_instance_id: self.swarm_instance_id,
            session_epoch: self.session_epoch,
            round_id: self.round_id,
            message_type: self.message_type as u8,
            sequence: self.sequence,
            timestamp: self.timestamp,
        }
    }

    /// Construct the [`EnvelopeContextV2`] bound by `swarmtorch.envelope.v2` (`ST-03`).
    #[cfg(feature = "alloc")]
    pub const fn envelope_context_v2(&self) -> EnvelopeContextV2 {
        EnvelopeContextV2 {
            version: self.version,
            swarm_instance_id: self.swarm_instance_id,
            run_id: self.run_id,
            session_epoch: self.session_epoch,
            session_id: self.session_id,
            round_id: self.round_id,
            message_type: self.message_type as u8,
            sequence: self.sequence,
            timestamp: self.timestamp,
        }
    }

    /// Sign this envelope using the appropriate domain-separated tag (`v0`, `v1`, or `v2`).
    #[cfg(feature = "alloc")]
    pub fn sign_with(mut self, auth: &MessageAuth) -> Self {
        let sig = if self.version == Self::VERSION_V2 {
            let ctx = self.envelope_context_v2();
            auth.sign_v2(&ctx, &self.payload)
        } else if self.version == Self::VERSION_V1 {
            let ctx = self.envelope_context_v1();
            auth.sign_v1(&ctx, &self.payload)
        } else {
            auth.sign(
                self.version,
                self.message_type as u8,
                self.sequence,
                self.timestamp,
                &self.payload,
            )
        };
        self.signature = Some(sig.as_bytes().to_vec());
        self
    }

    /// Set the signature.
    #[cfg(feature = "alloc")]
    pub fn with_signature(mut self, sig: alloc::vec::Vec<u8>) -> Self {
        self.signature = Some(sig);
        self
    }

    /// Serialize the envelope to bytes.
    #[cfg(feature = "alloc")]
    pub fn serialize(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Deserialize from bytes enforcing [`DEFAULT_MAX_ENVELOPE_BYTES`].
    pub fn deserialize(bytes: &[u8]) -> Result<Self, postcard::Error> {
        if bytes.len() > DEFAULT_MAX_ENVELOPE_BYTES {
            return Err(postcard::Error::DeserializeBadEncoding);
        }
        postcard::from_bytes(bytes)
    }

    /// Stage 1 bounded wire frame deserialization (`ADR-0021`, `W9-02`).
    #[cfg(feature = "alloc")]
    pub fn deserialize_bounded(bytes: &[u8], max_frame_bytes: usize) -> Result<Self, VerifyError> {
        if bytes.len() > max_frame_bytes {
            return Err(VerifyError::FrameTooLarge {
                len: bytes.len(),
                max: max_frame_bytes,
            });
        }
        postcard::from_bytes(bytes).map_err(|_| VerifyError::MalformedEnvelope)
    }

    /// Read-only Stages 2–5 preflight returning `(sender_peer, PreparedReplayAdmission)` under
    /// strict [`EnvelopeVersionPolicyV1::RequireV2`] with ZERO state mutation to `replay_guard`
    /// (`ST-01`, `ST-03`).
    #[cfg(feature = "alloc")]
    pub fn verify_authenticated_prepared(
        &self,
        replay_guard: &ReplayProtection,
        current_time: u32,
    ) -> Result<(PeerId, PreparedReplayAdmission), VerifyError> {
        self.verify_authenticated_prepared_with_policy(
            replay_guard,
            current_time,
            EnvelopeVersionPolicyV1::default(),
        )
    }

    /// Read-only Stages 2–5 preflight enforcing explicit [`EnvelopeVersionPolicyV1`] with ZERO
    /// state mutation to `replay_guard` (`ST-01`, `ST-03`, `A10`).
    #[cfg(feature = "alloc")]
    pub fn verify_authenticated_prepared_with_policy(
        &self,
        replay_guard: &ReplayProtection,
        current_time: u32,
        version_policy: EnvelopeVersionPolicyV1,
    ) -> Result<(PeerId, PreparedReplayAdmission), VerifyError> {
        // Stage 2a: Version & profile discipline check (before any state mutation).
        if !version_policy.allows_version(self.version) {
            return Err(VerifyError::UnsupportedVersion {
                major: self.version.0,
                minor: self.version.1,
            });
        }
        if self.version == Self::VERSION_V0
            && (self.swarm_instance_id != [0u8; 16]
                || self.run_id != [0u8; 16]
                || self.session_epoch != 0
                || self.session_id != [0u8; 32]
                || self.round_id != 0)
        {
            return Err(VerifyError::MalformedEnvelope);
        }
        if self.version == Self::VERSION_V1
            && (self.run_id != [0u8; 16] || self.session_id != [0u8; 32])
        {
            return Err(VerifyError::MalformedEnvelope);
        }

        // Stage 2b: Stateless timestamp expiry & restart floor check.
        replay_guard
            .check_timestamp_only(self.timestamp, current_time)
            .map_err(VerifyError::Replay)?;

        // Stage 2c: Canonical sender PeerId derivation.
        let sender_id = self
            .sender_peer_id()
            .map_err(|_| VerifyError::InvalidSenderKey)?;

        // Stage 3: Cohort allowlist and epoch/session continuity gate (before crypto).
        replay_guard
            .check_peer_authorized(&sender_id)
            .map_err(VerifyError::Replay)?;

        if self.version == Self::VERSION_V2 {
            let record_key_v2 = ReplayRecordKeyV2::new(
                self.swarm_instance_id,
                self.run_id,
                sender_id,
                self.session_epoch,
                self.session_id,
            );
            replay_guard
                .check_v2_session_continuity(&record_key_v2)
                .map_err(VerifyError::Replay)?;

            let signature = self.extract_signature()?;
            let ctx = self.envelope_context_v2();
            MessageAuth::verify_v2(&self.sender, &ctx, &self.payload, &signature)
                .map_err(VerifyError::Crypto)?;

            let prepared = replay_guard
                .check_v2_admission_readonly(
                    &record_key_v2,
                    self.sequence,
                    self.timestamp,
                    current_time,
                )
                .map_err(VerifyError::Replay)?;
            Ok((sender_id, prepared))
        } else {
            let record_key =
                ReplayRecordKey::new(self.swarm_instance_id, sender_id, self.session_epoch);
            replay_guard
                .check_epoch_continuity(&record_key)
                .map_err(VerifyError::Replay)?;

            let signature = self.extract_signature()?;
            if self.version == Self::VERSION_V1 {
                let ctx = self.envelope_context_v1();
                MessageAuth::verify_v1(&self.sender, &ctx, &self.payload, &signature)
                    .map_err(VerifyError::Crypto)?;
            } else {
                MessageAuth::verify(
                    &self.sender,
                    self.version,
                    self.message_type as u8,
                    self.sequence,
                    self.timestamp,
                    &self.payload,
                    &signature,
                )
                .map_err(VerifyError::Crypto)?;
            }

            let prepared = replay_guard
                .check_v1_admission_readonly(
                    &record_key,
                    self.sequence,
                    self.timestamp,
                    current_time,
                )
                .map_err(VerifyError::Replay)?;
            Ok((sender_id, prepared))
        }
    }

    #[cfg(feature = "alloc")]
    fn extract_signature(&self) -> Result<swarm_torch_core::crypto::Signature, VerifyError> {
        let sig_bytes = self
            .signature
            .as_ref()
            .ok_or(VerifyError::MissingSignature)?;

        if sig_bytes.len() != 64 {
            return Err(VerifyError::InvalidSignatureLength {
                expected: 64,
                found: sig_bytes.len(),
            });
        }

        let mut sig_array = [0u8; 64];
        sig_array.copy_from_slice(sig_bytes);
        Ok(swarm_torch_core::crypto::Signature::from_bytes(sig_array))
    }

    /// Execute Stages 2–5 of the atomic receive pipeline under [`EnvelopeVersionPolicyV1::RequireV2`]
    /// (`ADR-0021`, `W9-02`, `ST-03`).
    #[cfg(feature = "alloc")]
    pub fn verify_authenticated(
        &self,
        replay_guard: &mut ReplayProtection,
        current_time: u32,
    ) -> Result<(), VerifyError> {
        self.verify_authenticated_with_policy(
            replay_guard,
            current_time,
            EnvelopeVersionPolicyV1::default(),
        )
    }

    /// Execute Stages 2–5 of the atomic receive pipeline under an explicit
    /// [`EnvelopeVersionPolicyV1`] (`ADR-0021`, `ST-03`, `A10`).
    #[cfg(feature = "alloc")]
    pub fn verify_authenticated_with_policy(
        &self,
        replay_guard: &mut ReplayProtection,
        current_time: u32,
        version_policy: EnvelopeVersionPolicyV1,
    ) -> Result<(), VerifyError> {
        let (_sender_id, prepared) = self.verify_authenticated_prepared_with_policy(
            replay_guard,
            current_time,
            version_policy,
        )?;
        replay_guard
            .commit_prepared_admission(prepared)
            .map_err(VerifyError::Replay)
    }

    /// Execute the full 6-stage atomic receive pipeline under [`EnvelopeVersionPolicyV1::RequireV2`],
    /// decoding the inner payload `T` only after authentication and replay admission succeed
    /// (`ADR-0021`, `W9-02`).
    #[cfg(feature = "alloc")]
    pub fn verify_and_decode_payload<T: serde::de::DeserializeOwned>(
        &self,
        replay_guard: &mut ReplayProtection,
        current_time: u32,
    ) -> Result<T, VerifyError> {
        self.verify_and_decode_payload_with_policy(
            replay_guard,
            current_time,
            EnvelopeVersionPolicyV1::default(),
        )
    }

    /// Execute the full 6-stage atomic receive pipeline under an explicit
    /// [`EnvelopeVersionPolicyV1`].
    #[cfg(feature = "alloc")]
    pub fn verify_and_decode_payload_with_policy<T: serde::de::DeserializeOwned>(
        &self,
        replay_guard: &mut ReplayProtection,
        current_time: u32,
        version_policy: EnvelopeVersionPolicyV1,
    ) -> Result<T, VerifyError> {
        self.verify_authenticated_with_policy(replay_guard, current_time, version_policy)?;
        postcard::from_bytes(&self.payload).map_err(|_| VerifyError::MalformedPayload)
    }
}

/// Message type discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum MessageType {
    /// Gradient update from training.
    GradientUpdate = 0x01,
    /// Full model checkpoint.
    ModelCheckpoint = 0x02,
    /// Consensus vote.
    ConsensusVote = 0x03,
    /// Heartbeat for liveness.
    Heartbeat = 0x04,
    /// Peer discovery.
    PeerDiscovery = 0x05,
    /// Topology change notification.
    TopologyChange = 0x06,
    /// Aggregation result.
    AggregationResult = 0x07,
    /// Round start announcement.
    RoundStart = 0x08,
    /// Round complete announcement.
    RoundComplete = 0x09,
    /// Error/rejection notification.
    Error = 0xFF,
}

/// Heartbeat message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatMessage {
    /// Sender's current round.
    pub current_round: u64,
    /// Sender's role.
    pub role: u8,
    /// Number of known peers.
    pub known_peers: u16,
    /// Load indicator (0-255).
    pub load: u8,
}

/// Peer discovery request/response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PeerDiscoveryMessage {
    /// Whether this is a request or response.
    pub is_request: bool,
    /// Known peers to share.
    #[cfg(feature = "alloc")]
    pub peers: Vec<[u8; 32]>,
}

/// Round start announcement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundStartMessage {
    /// Round identifier.
    pub round_id: u64,
    /// Expected participants.
    pub expected_participants: u32,
    /// Round deadline (unix timestamp).
    pub deadline: u32,
    /// Aggregation method to use.
    pub aggregation_method: u8,
}

/// Verification errors for authenticated messages (`ADR-0021`, `W9-02`).
#[cfg(feature = "alloc")]
#[derive(Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// Cryptographic verification failed.
    Crypto(swarm_torch_core::crypto::VerifyError),
    /// Replay protection or cohort admission check failed.
    Replay(swarm_torch_core::replay::ReplayError),
    /// Signature field is missing.
    MissingSignature,
    /// Signature has invalid length.
    InvalidSignatureLength {
        /// Expected length.
        expected: usize,
        /// Found length.
        found: usize,
    },
    /// Unsupported protocol version.
    UnsupportedVersion {
        /// Unsupported major version.
        major: u8,
        /// Unsupported minor version.
        minor: u8,
    },
    /// Sender public key is invalid (all-zeros or wrong length).
    InvalidSenderKey,
    /// Wire frame exceeds pre-deserialization byte limit (Stage 1).
    FrameTooLarge {
        /// Actual byte length.
        len: usize,
        /// Maximum allowed byte length.
        max: usize,
    },
    /// Wire frame could not be decoded into a valid `MessageEnvelope`.
    MalformedEnvelope,
    /// Authenticated envelope payload could not be decoded into target message type (Stage 6).
    MalformedPayload,
    /// System time lookup failed.
    Time(TimeError),
}

#[cfg(feature = "alloc")]
impl core::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VerifyError::Crypto(e) => write!(f, "crypto error: {}", e),
            VerifyError::Replay(e) => write!(f, "replay error: {}", e),
            VerifyError::MissingSignature => write!(f, "missing signature"),
            VerifyError::InvalidSignatureLength { expected, found } => {
                write!(
                    f,
                    "invalid signature length: expected {}, found {}",
                    expected, found
                )
            }
            VerifyError::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported protocol version: {}.{}", major, minor)
            }
            VerifyError::InvalidSenderKey => write!(f, "invalid sender key"),
            VerifyError::FrameTooLarge { len, max } => {
                write!(f, "frame length {} exceeds maximum {}", len, max)
            }
            VerifyError::MalformedEnvelope => write!(f, "malformed message envelope"),
            VerifyError::MalformedPayload => write!(f, "malformed authenticated payload"),
            VerifyError::Time(e) => write!(f, "time error: {}", e),
        }
    }
}

#[cfg(all(feature = "alloc", feature = "std"))]
impl std::error::Error for VerifyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VerifyError::Crypto(e) => Some(e),
            VerifyError::Replay(e) => Some(e),
            VerifyError::Time(e) => Some(e),
            _ => None,
        }
    }
}

/// Time source errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeError {
    /// System clock is before Unix epoch.
    BeforeEpoch,
    /// System time exceeds `u32` second range.
    Overflow,
}

impl core::fmt::Display for TimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TimeError::BeforeEpoch => write!(f, "system clock is before Unix epoch"),
            TimeError::Overflow => write!(f, "unix timestamp overflows u32"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TimeError {}

/// Replay+signature enforcement wrapper for incoming envelopes.
#[cfg(feature = "alloc")]
pub struct AuthenticatedEnvelopeVerifier {
    replay_guard: ReplayProtection,
    version_policy: EnvelopeVersionPolicyV1,
}

#[cfg(feature = "alloc")]
impl AuthenticatedEnvelopeVerifier {
    /// Create a verifier with default replay protection and strict [`EnvelopeVersionPolicyV1::RequireV2`].
    pub fn new() -> Self {
        Self {
            replay_guard: ReplayProtection::new(),
            version_policy: EnvelopeVersionPolicyV1::default(),
        }
    }

    /// Create a verifier with caller-provided replay protection state and default `RequireV2` policy.
    pub fn with_replay_guard(replay_guard: ReplayProtection) -> Self {
        Self {
            replay_guard,
            version_policy: EnvelopeVersionPolicyV1::default(),
        }
    }

    /// Configure explicit wire version compatibility policy (opt-in legacy fallback).
    pub const fn with_version_policy(mut self, version_policy: EnvelopeVersionPolicyV1) -> Self {
        self.version_policy = version_policy;
        self
    }

    /// Return the configured [`EnvelopeVersionPolicyV1`].
    pub const fn version_policy(&self) -> EnvelopeVersionPolicyV1 {
        self.version_policy
    }

    /// Verify and return the envelope using current wall clock time.
    #[cfg(feature = "std")]
    pub fn verify_and_unwrap(
        &mut self,
        envelope: MessageEnvelope,
    ) -> Result<MessageEnvelope, VerifyError> {
        let now = MessageEnvelope::current_unix_secs().map_err(VerifyError::Time)?;
        envelope.verify_authenticated_with_policy(
            &mut self.replay_guard,
            now,
            self.version_policy,
        )?;
        Ok(envelope)
    }

    /// Verify and return the envelope with injected current time.
    pub fn verify_and_unwrap_with_time(
        &mut self,
        envelope: MessageEnvelope,
        current_time_secs: u32,
    ) -> Result<MessageEnvelope, VerifyError> {
        envelope.verify_authenticated_with_policy(
            &mut self.replay_guard,
            current_time_secs,
            self.version_policy,
        )?;
        Ok(envelope)
    }

    /// Get immutable access to replay guard state.
    pub fn replay_guard(&self) -> &ReplayProtection {
        &self.replay_guard
    }

    /// Get mutable access to replay guard state.
    pub fn replay_guard_mut(&mut self) -> &mut ReplayProtection {
        &mut self.replay_guard
    }
}

#[cfg(feature = "alloc")]
impl Default for AuthenticatedEnvelopeVerifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn current_version_defaults_to_v2_and_legacy_fallback_is_explicitly_gated() {
        assert_eq!(
            MessageEnvelope::CURRENT_VERSION,
            MessageEnvelope::VERSION_V2
        );
        let default_env =
            MessageEnvelope::new_with_public_key([0x11; 32], MessageType::Heartbeat, vec![1, 2]);
        assert_eq!(default_env.version, MessageEnvelope::VERSION_V2);
        let legacy_env =
            MessageEnvelope::new_legacy_v0([0x11; 32], MessageType::Heartbeat, vec![1, 2]);
        assert_eq!(legacy_env.version, MessageEnvelope::VERSION_V0);

        assert!(EnvelopeVersionPolicyV1::default().allows_version(MessageEnvelope::VERSION_V2));
        assert!(!EnvelopeVersionPolicyV1::RequireV2.allows_version(MessageEnvelope::VERSION_V1));
        assert!(!EnvelopeVersionPolicyV1::RequireV2.allows_version(MessageEnvelope::VERSION_V0));
        assert!(
            EnvelopeVersionPolicyV1::AllowLegacyV1AndV2.allows_version(MessageEnvelope::VERSION_V1)
        );
        assert!(
            EnvelopeVersionPolicyV1::AllowLegacyV1AndV2.allows_version(MessageEnvelope::VERSION_V2)
        );
        assert!(!EnvelopeVersionPolicyV1::AllowLegacyV1AndV2
            .allows_version(MessageEnvelope::VERSION_V0));
        assert!(EnvelopeVersionPolicyV1::AllowLegacyAll.allows_version(MessageEnvelope::VERSION_V0));
        assert!(EnvelopeVersionPolicyV1::AllowLegacyAll.allows_version(MessageEnvelope::VERSION_V1));
        assert!(EnvelopeVersionPolicyV1::AllowLegacyAll.allows_version(MessageEnvelope::VERSION_V2));
    }

    #[test]
    fn default_require_v2_rejects_v0_and_v1_without_mutating_replay_guard() {
        use swarm_torch_core::crypto::KeyPair;
        use swarm_torch_core::replay::{CohortAllowlist, ReplayStartupPolicyV1};

        let kp = KeyPair::from_seed([0x51; 32]).unwrap();
        let auth = MessageAuth::new(kp.clone());
        let now = 1_000u32;
        let hb = HeartbeatMessage {
            current_round: 3,
            role: 1,
            known_peers: 2,
            load: 10,
        };
        let payload = postcard::to_allocvec(&hb).unwrap();

        let env_v0 = MessageEnvelope::new_legacy_v0(
            *kp.public_key(),
            MessageType::Heartbeat,
            payload.clone(),
        )
        .with_sequence(1)
        .with_timestamp(now)
        .sign_with(&auth);
        let env_v1 = MessageEnvelope::new_v1(
            *kp.public_key(),
            [0x11; 16],
            1,
            3,
            MessageType::Heartbeat,
            payload,
        )
        .with_sequence(1)
        .with_timestamp(now)
        .sign_with(&auth);

        let mut guard = ReplayProtection::try_with_config(64, 60)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(CohortAllowlist::from_peers([kp.peer_id()]));

        assert_eq!(
            env_v0
                .verify_authenticated_prepared(&guard, now)
                .unwrap_err(),
            VerifyError::UnsupportedVersion { major: 0, minor: 1 }
        );
        assert_eq!(
            env_v1
                .verify_authenticated_prepared(&guard, now)
                .unwrap_err(),
            VerifyError::UnsupportedVersion { major: 0, minor: 2 }
        );
        assert_eq!(
            env_v0
                .verify_and_decode_payload::<HeartbeatMessage>(&mut guard, now)
                .unwrap_err(),
            VerifyError::UnsupportedVersion { major: 0, minor: 1 }
        );
        assert_eq!(
            env_v1
                .verify_and_decode_payload::<HeartbeatMessage>(&mut guard, now)
                .unwrap_err(),
            VerifyError::UnsupportedVersion { major: 0, minor: 2 }
        );
        assert_eq!(guard.cache_size(), 0);

        let mut strict_verifier = AuthenticatedEnvelopeVerifier::with_replay_guard(guard);
        assert_eq!(
            strict_verifier.version_policy(),
            EnvelopeVersionPolicyV1::RequireV2
        );
        assert_eq!(
            strict_verifier
                .verify_and_unwrap_with_time(env_v1.clone(), now)
                .unwrap_err(),
            VerifyError::UnsupportedVersion { major: 0, minor: 2 }
        );
        assert_eq!(strict_verifier.replay_guard().cache_size(), 0);

        let mut v1_verifier =
            strict_verifier.with_version_policy(EnvelopeVersionPolicyV1::AllowLegacyV1AndV2);
        assert_eq!(
            v1_verifier
                .verify_and_unwrap_with_time(env_v0, now)
                .unwrap_err(),
            VerifyError::UnsupportedVersion { major: 0, minor: 1 }
        );
        assert_eq!(v1_verifier.replay_guard().cache_size(), 0);

        let decoded: HeartbeatMessage = env_v1
            .verify_and_decode_payload_with_policy(
                v1_verifier.replay_guard_mut(),
                now,
                EnvelopeVersionPolicyV1::AllowLegacyV1AndV2,
            )
            .unwrap();
        assert_eq!(decoded, hb);
        assert_eq!(v1_verifier.replay_guard().cache_size(), 1);
    }
}
