//! Message protocol, versioned envelope signing, and 6-stage atomic receive pipeline (`ADR-0021`, `W9-02`).

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
#[cfg(feature = "alloc")]
use swarm_torch_core::crypto::{EnvelopeContextV1, MessageAuth};
#[cfg(feature = "alloc")]
use swarm_torch_core::replay::{ReplayProtection, ReplayRecordKey};
use swarm_torch_core::traits::PeerId;

/// Default pre-deserialization wire frame bound (`1,048,576` bytes = 1 MiB).
pub const DEFAULT_MAX_ENVELOPE_BYTES: usize = 1_048_576;

/// Message envelope for all swarm communications (`ADR-0008`, `ADR-0021`, `W9-02`).
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
    /// Scoped swarm deployment / cohort instance identifier (`VERSION_V1`).
    #[serde(default)]
    pub swarm_instance_id: [u8; 16],
    /// Monotonically increasing sender session epoch (`VERSION_V1`).
    #[serde(default)]
    pub session_epoch: u64,
    /// Training / coordination round identifier (`VERSION_V1`).
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
    /// Current default protocol version for `new_with_public_key`.
    pub const CURRENT_VERSION: (u8, u8) = Self::VERSION_V0;
    /// Supported protocol versions.
    pub const SUPPORTED_VERSIONS: &'static [(u8, u8)] = &[Self::VERSION_V0, Self::VERSION_V1];

    /// Create a new legacy v0 message envelope with explicit public key bytes.
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
            session_epoch: 0,
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
            session_epoch,
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
        if seconds > u32::MAX as u64 {
            return Err(TimeError::Overflow);
        }
        Ok(seconds as u32)
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

    /// Set the round identifier (`VERSION_V1`).
    pub const fn with_round_id(mut self, round_id: u64) -> Self {
        self.round_id = round_id;
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

    /// Sign this envelope using the appropriate domain-separated tag (`v0` or `v1`).
    #[cfg(feature = "alloc")]
    pub fn sign_with(mut self, auth: &MessageAuth) -> Self {
        let sig = if self.version == Self::VERSION_V1 {
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

    /// Execute Stages 2–5 of the atomic receive pipeline (`ADR-0021`, `W9-02`):
    /// - Stage 2: Stateless protocol version, timestamp window, and sender key validation.
    /// - Stage 3: Cohort allowlist and session-epoch continuity gate (before crypto or state mutation).
    /// - Stage 4: Domain-separated Ed25519 signature verification (`v0` or `v1`).
    /// - Stage 5: Stateful replay window and expiry-indexed cache update.
    #[cfg(feature = "alloc")]
    pub fn verify_authenticated(
        &self,
        replay_guard: &mut ReplayProtection,
        current_time: u32,
    ) -> Result<(), VerifyError> {
        // Stage 2a: Version check (before any state mutation).
        if !self.is_version_supported() {
            return Err(VerifyError::UnsupportedVersion {
                major: self.version.0,
                minor: self.version.1,
            });
        }
        // Reject unsigned v1 scope fields on legacy v0 envelopes.
        if self.version == Self::VERSION_V0
            && (self.swarm_instance_id != [0u8; 16]
                || self.session_epoch != 0
                || self.round_id != 0)
        {
            return Err(VerifyError::MalformedEnvelope);
        }

        // Stage 2b: Stateless timestamp expiry check.
        replay_guard
            .check_timestamp_only(self.timestamp, current_time)
            .map_err(VerifyError::Replay)?;

        // Stage 2c: Canonical sender PeerId derivation.
        let sender_id = self
            .sender_peer_id()
            .map_err(|_| VerifyError::InvalidSenderKey)?;

        // Stage 3: Cohort allowlist and epoch continuity gate (before expensive crypto or state mutation).
        replay_guard
            .check_peer_authorized(&sender_id)
            .map_err(VerifyError::Replay)?;
        let record_key =
            ReplayRecordKey::new(self.swarm_instance_id, sender_id, self.session_epoch);
        replay_guard
            .check_epoch_continuity(&record_key)
            .map_err(VerifyError::Replay)?;

        // Stage 4: Ed25519 signature verification.
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
        let signature = swarm_torch_core::crypto::Signature::from_bytes(sig_array);

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

        // Stage 5: Stateful replay check and expiry-indexed cache update.
        replay_guard
            .validate_sequence_scoped_at(&record_key, self.sequence, self.timestamp, current_time)
            .map_err(VerifyError::Replay)?;

        Ok(())
    }

    /// Execute the full 6-stage atomic receive pipeline, decoding the inner payload `T`
    /// only after authentication and replay admission succeed (`ADR-0021`, `W9-02`).
    #[cfg(feature = "alloc")]
    pub fn verify_and_decode_payload<T: serde::de::DeserializeOwned>(
        &self,
        replay_guard: &mut ReplayProtection,
        current_time: u32,
    ) -> Result<T, VerifyError> {
        self.verify_authenticated(replay_guard, current_time)?;
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
}

#[cfg(feature = "alloc")]
impl AuthenticatedEnvelopeVerifier {
    /// Create a verifier with default replay protection configuration.
    pub fn new() -> Self {
        Self {
            replay_guard: ReplayProtection::new(),
        }
    }

    /// Create a verifier with caller-provided replay protection state.
    pub fn with_replay_guard(replay_guard: ReplayProtection) -> Self {
        Self { replay_guard }
    }

    /// Verify and return the envelope using current wall clock time.
    #[cfg(feature = "std")]
    pub fn verify_and_unwrap(
        &mut self,
        envelope: MessageEnvelope,
    ) -> Result<MessageEnvelope, VerifyError> {
        let now = MessageEnvelope::current_unix_secs().map_err(VerifyError::Time)?;
        envelope.verify_authenticated(&mut self.replay_guard, now)?;
        Ok(envelope)
    }

    /// Verify and return the envelope with injected current time.
    pub fn verify_and_unwrap_with_time(
        &mut self,
        envelope: MessageEnvelope,
        current_time_secs: u32,
    ) -> Result<MessageEnvelope, VerifyError> {
        envelope.verify_authenticated(&mut self.replay_guard, current_time_secs)?;
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
