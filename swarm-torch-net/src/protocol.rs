//! Message protocol and framing
//!
//! This module defines the wire format for swarm messages.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
#[cfg(feature = "alloc")]
use swarm_torch_core::replay::ReplayProtection;
#[cfg(feature = "alloc")]
use swarm_torch_core::traits::PeerId;

/// Message envelope for all swarm communications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEnvelope {
    /// Protocol version (major, minor)
    pub version: (u8, u8),
    /// Message type discriminator
    pub message_type: MessageType,
    /// Sender's Ed25519 public key bytes (raw 32 bytes)
    pub sender: [u8; 32],
    /// Monotonic sequence number (replay protection)
    pub sequence: u64,
    /// Unix timestamp (seconds, for expiry)
    pub timestamp: u32,
    /// Payload bytes
    #[cfg(feature = "alloc")]
    pub payload: Vec<u8>,
    /// Optional cryptographic signature (64 bytes for Ed25519)
    #[cfg(feature = "alloc")]
    pub signature: Option<alloc::vec::Vec<u8>>,
}

impl MessageEnvelope {
    /// Current protocol version
    pub const CURRENT_VERSION: (u8, u8) = (0, 1);
    /// Supported protocol versions
    pub const SUPPORTED_VERSIONS: &'static [(u8, u8)] = &[Self::CURRENT_VERSION];

    /// Create a new message envelope with explicit public key bytes.
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
            payload,
            signature: None,
        }
    }

    /// Create a new message envelope from a `PeerId`.
    ///
    /// Deprecated because `PeerId::from_public_key()` hashes the public key,
    /// while `MessageEnvelope::sender` must contain raw Ed25519 public key bytes.
    #[cfg(feature = "alloc")]
    #[deprecated(
        since = "0.1.0-alpha.6",
        note = "Use new_with_public_key() with raw Ed25519 public key bytes."
    )]
    pub fn new(sender: PeerId, message_type: MessageType, payload: Vec<u8>) -> Self {
        Self::new_with_public_key(*sender.as_bytes(), message_type, payload)
    }

    /// Returns true when this envelope version is supported.
    pub fn is_version_supported(&self) -> bool {
        Self::SUPPORTED_VERSIONS.contains(&self.version)
    }

    /// Get sender public key bytes.
    pub const fn sender_public_key(&self) -> &[u8; 32] {
        &self.sender
    }

    /// Derive sender `PeerId` from sender public key bytes.
    #[cfg(feature = "std")]
    pub fn sender_peer_id(&self) -> PeerId {
        PeerId::from_public_key(&self.sender)
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

    /// Set the sequence number
    pub fn with_sequence(mut self, seq: u64) -> Self {
        self.sequence = seq;
        self
    }

    /// Set the timestamp
    pub fn with_timestamp(mut self, ts: u32) -> Self {
        self.timestamp = ts;
        self
    }

    /// Set the signature
    #[cfg(feature = "alloc")]
    pub fn with_signature(mut self, sig: alloc::vec::Vec<u8>) -> Self {
        self.signature = Some(sig);
        self
    }

    /// Serialize the envelope to bytes
    #[cfg(feature = "alloc")]
    pub fn serialize(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Deserialize from bytes
    pub fn deserialize(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }

    /// Verify signature and replay protection
    ///
    /// This method performs a three-stage validation:
    /// 1. Timestamp expiry check (cheap, fail-fast)
    /// 2. Cryptographic signature verification (expensive)
    /// 3. Replay protection (stateful)
    ///
    /// # Arguments
    ///
    /// * `replay_guard` - Replay protection state (mutated on success)
    /// * `current_time` - Current Unix timestamp in seconds
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Signature is missing or invalid
    /// - Timestamp is outside acceptable window
    /// - Sequence number is duplicate or retrograde
    #[cfg(feature = "alloc")]
    pub fn verify_authenticated(
        &self,
        replay_guard: &mut ReplayProtection,
        current_time: u32,
    ) -> Result<(), VerifyError> {
        use swarm_torch_core::crypto::MessageAuth;

        // 0. VERSION CHECK (before any state mutation)
        if !self.is_version_supported() {
            return Err(VerifyError::UnsupportedVersion {
                major: self.version.0,
                minor: self.version.1,
            });
        }

        // OPTIMIZATION: Fail-fast checks before expensive crypto

        // 1. CHEAP: Timestamp expiry (no state mutation)
        replay_guard
            .check_timestamp_only(self.timestamp, current_time)
            .map_err(VerifyError::Replay)?;

        // 2. EXPENSIVE: Signature verification (CPU-intensive)
        let sig_bytes = self
            .signature
            .as_ref()
            .ok_or(VerifyError::MissingSignature)?;

        // Validate signature length
        if sig_bytes.len() != 64 {
            return Err(VerifyError::InvalidSignatureLength {
                expected: 64,
                found: sig_bytes.len(),
            });
        }

        // Convert to [u8; 64]
        let mut sig_array = [0u8; 64];
        sig_array.copy_from_slice(sig_bytes);
        let signature = swarm_torch_core::crypto::Signature::from_bytes(sig_array);

        // Verify signature
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

        // 3. STATEFUL: Replay check (mutates cache)
        let sender_id = PeerId::new(self.sender);
        replay_guard
            .validate_sequence(&sender_id, self.sequence)
            .map_err(VerifyError::Replay)?;

        Ok(())
    }
}

/// Message type discriminator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum MessageType {
    /// Gradient update from training
    GradientUpdate = 0x01,
    /// Full model checkpoint
    ModelCheckpoint = 0x02,
    /// Consensus vote
    ConsensusVote = 0x03,
    /// Heartbeat for liveness
    Heartbeat = 0x04,
    /// Peer discovery
    PeerDiscovery = 0x05,
    /// Topology change notification
    TopologyChange = 0x06,
    /// Aggregation result
    AggregationResult = 0x07,
    /// Round start announcement
    RoundStart = 0x08,
    /// Round complete announcement
    RoundComplete = 0x09,
    /// Error/rejection notification
    Error = 0xFF,
}

/// Heartbeat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatMessage {
    /// Sender's current round
    pub current_round: u64,
    /// Sender's role
    pub role: u8,
    /// Number of known peers
    pub known_peers: u16,
    /// Load indicator (0-255)
    pub load: u8,
}

/// Peer discovery request/response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerDiscoveryMessage {
    /// Whether this is a request or response
    pub is_request: bool,
    /// Known peers to share
    #[cfg(feature = "alloc")]
    pub peers: Vec<[u8; 32]>,
}

/// Round start announcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundStartMessage {
    /// Round identifier
    pub round_id: u64,
    /// Expected participants
    pub expected_participants: u32,
    /// Round deadline (unix timestamp)
    pub deadline: u32,
    /// Aggregation method to use
    pub aggregation_method: u8,
}

/// Verification errors for authenticated messages
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub enum VerifyError {
    /// Cryptographic verification failed
    Crypto(swarm_torch_core::crypto::VerifyError),
    /// Replay protection check failed
    Replay(swarm_torch_core::replay::ReplayError),
    /// Signature field is missing
    MissingSignature,
    /// Signature has invalid length
    InvalidSignatureLength {
        /// Expected length
        expected: usize,
        /// Found length
        found: usize,
    },
    /// Unsupported protocol version
    UnsupportedVersion {
        /// Unsupported major version
        major: u8,
        /// Unsupported minor version
        minor: u8,
    },
    /// System time lookup failed
    Time(TimeError),
}

#[cfg(feature = "alloc")]
impl core::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VerifyError::Crypto(e) => write!(f, "crypto error: {:?}", e),
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
            VerifyError::Time(e) => write!(f, "time error: {}", e),
        }
    }
}

#[cfg(all(feature = "alloc", feature = "std"))]
impl std::error::Error for VerifyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
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
