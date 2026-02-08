//! Consensus mechanisms for swarm coordination
//!
//! This module provides gossip-based consensus for coordinating
//! training rounds across distributed nodes.

use crate::traits::PeerId;

/// Configuration for gossip-based consensus
#[derive(Debug, Clone)]
pub struct GossipConfig {
    /// Number of peers to gossip to per round
    pub fanout: usize,
    /// Probability of forwarding received messages
    pub forward_probability: f32,
    /// Time before message expires (in seconds)
    pub message_ttl_secs: u32,
    /// Heartbeat interval (in seconds)
    pub heartbeat_interval_secs: u32,
    /// Minimum quorum ratio for round completion
    pub quorum_ratio: f32,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout: 3,
            forward_probability: 0.7,
            message_ttl_secs: 60,
            heartbeat_interval_secs: 10,
            quorum_ratio: 0.67,
        }
    }
}

/// Round identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoundId(pub u64);

/// State of a training round
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundState {
    /// Waiting for round to start
    Pending,
    /// Training in progress
    Training,
    /// Collecting updates
    Collecting,
    /// Aggregating updates
    Aggregating,
    /// Round complete
    Complete,
    /// Round failed
    Failed,
}

/// Membership view of the cluster
#[derive(Debug, Clone, Default)]
pub struct MembershipView {
    /// Known active peers
    #[cfg(feature = "alloc")]
    pub active_peers: alloc::vec::Vec<PeerId>,
    /// Peers suspected to be offline
    #[cfg(feature = "alloc")]
    pub suspected_peers: alloc::vec::Vec<PeerId>,
    /// Last update timestamp (unix seconds)
    pub last_updated: u64,
}

impl MembershipView {
    /// Get the number of active peers
    #[cfg(feature = "alloc")]
    pub fn active_count(&self) -> usize {
        self.active_peers.len()
    }

    /// Check if a peer is active
    #[cfg(feature = "alloc")]
    pub fn is_active(&self, peer: &PeerId) -> bool {
        self.active_peers.contains(peer)
    }
}

/// Vote in consensus protocol
#[derive(Debug, Clone)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct Vote {
    /// Round this vote is for
    pub round_id: u64,
    /// Voter's peer ID
    pub voter: PeerId,
    /// Vote value (true = accept, false = reject)
    pub accept: bool,
    /// Optional reason for rejection
    #[cfg(feature = "alloc")]
    pub reason: Option<alloc::string::String>,
}
