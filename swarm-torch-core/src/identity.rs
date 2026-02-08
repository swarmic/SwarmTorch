//! Node identity and role management
//!
//! This module defines node identities and their roles in the swarm.

use crate::crypto::KeyPair;
use crate::traits::PeerId;

/// Role of a node in the swarm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub enum NodeRole {
    /// Full participant: trains, aggregates, coordinates
    Coordinator,
    /// Contributor: trains locally, sends updates
    Contributor,
    /// Observer: receives model, doesn't contribute
    Observer,
    /// Gateway: bridges networks (e.g., LoRa ↔ WiFi)
    Gateway,
}

impl Default for NodeRole {
    fn default() -> Self {
        Self::Contributor
    }
}

/// Node identity containing keys and role
pub struct NodeIdentity {
    /// Unique node ID (derived from public key)
    pub id: PeerId,
    /// Cryptographic key pair
    pub key_pair: KeyPair,
    /// Role in the swarm
    pub role: NodeRole,
    /// Human-readable name (optional)
    #[cfg(feature = "alloc")]
    pub name: Option<alloc::string::String>,
}

impl NodeIdentity {
    /// Create a new node identity from a key pair
    pub fn new(key_pair: KeyPair, role: NodeRole) -> Self {
        let id = key_pair.peer_id();
        Self {
            id,
            key_pair,
            role,
            #[cfg(feature = "alloc")]
            name: None,
        }
    }

    /// Create a new contributor identity
    pub fn contributor(key_pair: KeyPair) -> Self {
        Self::new(key_pair, NodeRole::Contributor)
    }

    /// Create a new coordinator identity
    pub fn coordinator(key_pair: KeyPair) -> Self {
        Self::new(key_pair, NodeRole::Coordinator)
    }

    /// Set the human-readable name
    #[cfg(feature = "alloc")]
    pub fn with_name(mut self, name: impl Into<alloc::string::String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Check if this node can aggregate updates
    pub fn can_aggregate(&self) -> bool {
        matches!(self.role, NodeRole::Coordinator | NodeRole::Gateway)
    }

    /// Check if this node can contribute updates
    pub fn can_contribute(&self) -> bool {
        matches!(self.role, NodeRole::Coordinator | NodeRole::Contributor)
    }
}

/// Participant configuration for swarm learning
#[derive(Debug, Clone)]
pub struct ParticipantConfig {
    /// Role in the swarm
    pub role: NodeRole,
    /// Memory limit in bytes (for embedded)
    pub memory_limit: Option<usize>,
    /// Maximum concurrent connections
    pub max_connections: usize,
}

impl Default for ParticipantConfig {
    fn default() -> Self {
        Self {
            role: NodeRole::Contributor,
            memory_limit: None,
            max_connections: 10,
        }
    }
}

impl ParticipantConfig {
    /// Create a minimal embedded participant configuration
    pub fn embedded(memory_limit_kb: usize) -> Self {
        Self {
            role: NodeRole::Contributor,
            memory_limit: Some(memory_limit_kb * 1024),
            max_connections: 3,
        }
    }

    /// Create an edge gateway configuration
    pub fn gateway() -> Self {
        Self {
            role: NodeRole::Gateway,
            memory_limit: None,
            max_connections: 50,
        }
    }

    /// Create a server/coordinator configuration
    pub fn coordinator() -> Self {
        Self {
            role: NodeRole::Coordinator,
            memory_limit: None,
            max_connections: 100,
        }
    }
}
