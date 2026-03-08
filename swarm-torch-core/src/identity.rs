//! Node identity and role management
//!
//! This module defines node identities and their roles in the swarm.

use crate::crypto::KeyPair;
use crate::traits::PeerId;

/// Role of a node in the swarm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub enum NodeRole {
    /// Full participant: trains, aggregates, coordinates
    Coordinator,
    /// Contributor: trains locally, sends updates
    #[default]
    Contributor,
    /// Observer: receives model, doesn't contribute
    Observer,
    /// Gateway: bridges networks (e.g., LoRa ↔ WiFi)
    Gateway,
}

/// Node identity containing keys and role.
///
/// `id` and `key_pair` are private to enforce the invariant `id == key_pair.peer_id()`.
/// Use [`NodeIdentity::new`] (or convenience constructors) to create instances,
/// and [`NodeIdentity::id()`] / [`NodeIdentity::key_pair()`] for read access.
pub struct NodeIdentity {
    /// Unique node ID (derived from public key) — private to enforce invariant.
    id: PeerId,
    /// Cryptographic key pair — private; mutation would break id invariant.
    key_pair: KeyPair,
    /// Role in the swarm
    pub role: NodeRole,
    /// Human-readable name (optional)
    #[cfg(feature = "alloc")]
    pub name: Option<alloc::string::String>,
}

impl NodeIdentity {
    /// Create a new node identity from a key pair.
    ///
    /// `id` is always derived from `key_pair.peer_id()` to enforce the identity invariant.
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

    /// Get the node's peer identity (immutable).
    ///
    /// Always equals `key_pair().peer_id()`.
    pub fn id(&self) -> &PeerId {
        &self.id
    }

    /// Get the node's cryptographic key pair (immutable).
    ///
    /// No mutable accessor is provided — changing the key pair would desync `id`.
    /// To change identity, construct a new `NodeIdentity`.
    pub fn key_pair(&self) -> &KeyPair {
        &self.key_pair
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::KeyPair;

    #[test]
    fn node_identity_id_always_equals_keypair_peer_id() {
        let kp = KeyPair::from_seed([42u8; 32]).expect("non-zero seed");
        let identity = NodeIdentity::new(kp.clone(), NodeRole::Contributor);
        assert_eq!(*identity.id(), kp.peer_id());
    }

    #[test]
    fn node_identity_getters_return_correct_values() {
        let kp = KeyPair::from_seed([7u8; 32]).expect("non-zero seed");
        let expected_id = kp.peer_id();
        let identity = NodeIdentity::coordinator(kp);
        assert_eq!(*identity.id(), expected_id);
        assert_eq!(identity.key_pair().peer_id(), expected_id);
        assert_eq!(identity.role, NodeRole::Coordinator);
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
