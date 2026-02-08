//! Transport traits and types
//!
//! This module defines the core transport abstraction.

use swarm_torch_core::traits::PeerId;
use crate::Result;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Reliability classification for transports
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReliabilityClass {
    /// No delivery guarantees (LoRa, BLE)
    BestEffort,
    /// At least once delivery with retries (UDP with ACK)
    AtLeastOnce,
    /// Reliable ordered delivery (TCP)
    Reliable,
}

/// Bandwidth classification for transports
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthClass {
    /// < 10 kbps (LoRa)
    UltraLow,
    /// 10 - 1000 kbps (BLE)
    Low,
    /// 1 - 100 Mbps (WiFi)
    Medium,
    /// > 100 Mbps (Ethernet, datacenter)
    High,
}

/// Transport capabilities
#[derive(Debug, Clone)]
pub struct TransportCapabilities {
    /// Reliability class of the transport
    pub reliability: ReliabilityClass,
    /// Bandwidth class of the transport
    pub bandwidth_class: BandwidthClass,
    /// Maximum message size in bytes
    pub max_message_size: usize,
    /// Whether the transport supports multicast/broadcast
    pub supports_multicast: bool,
}

/// Statistics from a broadcast operation
#[derive(Debug, Clone, Default)]
pub struct BroadcastStats {
    /// Number of peers the message was sent to
    pub peers_sent: usize,
    /// Number of confirmed deliveries (if applicable)
    pub confirmed: usize,
    /// Number of failed sends
    pub failed: usize,
}

/// Core transport trait for swarm communication
#[cfg(feature = "std")]
#[async_trait::async_trait]
pub trait SwarmTransport: Send + Sync {
    /// Send a message to a specific peer
    async fn send(&self, peer: PeerId, msg: &[u8]) -> Result<()>;

    /// Receive the next message (blocking until available)
    async fn recv(&self) -> Result<(PeerId, Vec<u8>)>;

    /// Broadcast a message to all known peers (best-effort)
    async fn broadcast(&self, msg: &[u8]) -> Result<BroadcastStats>;

    /// Discover peers on the network
    async fn discover(&self) -> Result<Vec<PeerId>>;

    /// Get transport capabilities
    fn capabilities(&self) -> TransportCapabilities;
}

/// Synchronous transport trait for no_std environments
#[cfg(not(feature = "std"))]
pub trait SwarmTransport: Send + Sync {
    /// Send a message to a specific peer
    fn send(&self, peer: PeerId, msg: &[u8]) -> Result<()>;

    /// Try to receive a message (non-blocking)
    fn try_recv(&self) -> Result<Option<(PeerId, [u8; 256])>>;

    /// Get transport capabilities
    fn capabilities(&self) -> TransportCapabilities;
}

/// Priority level for multi-transport routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(pub u8);

impl Priority {
    /// Highest priority (use first)
    pub const HIGH: Self = Self(0);
    /// Normal priority
    pub const NORMAL: Self = Self(128);
    /// Low priority (fallback)
    pub const LOW: Self = Self(255);
}

/// Fallback policy for multi-transport
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackPolicy {
    /// Prefer the transport with lowest latency
    PreferLowLatency,
    /// Prefer the transport with highest reliability
    PreferReliability,
    /// Try all transports in priority order
    PriorityOrder,
    /// Try the transport with best battery efficiency
    PreferPowerEfficient,
}

impl Default for FallbackPolicy {
    fn default() -> Self {
        Self::PriorityOrder
    }
}
