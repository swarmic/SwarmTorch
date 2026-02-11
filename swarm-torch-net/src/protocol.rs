//! Message protocol and framing
//!
//! This module defines the wire format for swarm messages.

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
#[cfg(feature = "alloc")]
use swarm_torch_core::traits::PeerId;

/// Message envelope for all swarm communications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEnvelope {
    /// Protocol version (major, minor)
    pub version: (u8, u8),
    /// Message type discriminator
    pub message_type: MessageType,
    /// Sender's peer ID
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

    /// Create a new message envelope
    #[cfg(feature = "alloc")]
    pub fn new(sender: PeerId, message_type: MessageType, payload: Vec<u8>) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            message_type,
            sender: *sender.as_bytes(),
            sequence: 0,
            timestamp: 0,
            payload,
            signature: None,
        }
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
