//! Mock transport for testing
//!
//! This module provides a mock transport implementation for unit testing.

#[cfg(feature = "alloc")]
use alloc::collections::VecDeque;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::traits::{BandwidthClass, ReliabilityClass, TransportCapabilities};
#[cfg(feature = "alloc")]
use swarm_torch_core::traits::PeerId;

/// Mock transport for testing without real networking
#[derive(Debug, Default)]
pub struct MockTransport {
    /// Simulated failure rate (0.0 to 1.0)
    pub failure_rate: f32,
    /// Simulated latency in milliseconds
    pub latency_ms: u32,
    /// Whether the transport is connected
    pub connected: bool,
}

impl MockTransport {
    /// Create a new mock transport
    pub fn new() -> Self {
        Self {
            failure_rate: 0.0,
            latency_ms: 0,
            connected: true,
        }
    }

    /// Set the simulated failure rate
    pub fn with_failure_rate(mut self, rate: f32) -> Self {
        self.failure_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set the simulated latency
    pub fn with_latency(mut self, latency_ms: u32) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Get transport capabilities
    pub fn capabilities(&self) -> TransportCapabilities {
        TransportCapabilities {
            reliability: ReliabilityClass::Reliable,
            bandwidth_class: BandwidthClass::High,
            max_message_size: 65536,
            supports_multicast: true,
        }
    }
}

/// A network of interconnected mock transports for testing
#[cfg(feature = "alloc")]
#[derive(Debug, Default)]
pub struct MockNetwork {
    /// Messages in transit
    pub messages: VecDeque<(PeerId, PeerId, Vec<u8>)>,
    /// Connected peers
    pub peers: Vec<PeerId>,
}

#[cfg(feature = "alloc")]
impl MockNetwork {
    /// Create a new mock network with the specified number of peers
    pub fn new(num_peers: usize) -> Self {
        let peers = (0..num_peers)
            .map(|i| {
                let mut bytes = [0u8; 32];
                bytes[0] = i as u8;
                PeerId::new(bytes)
            })
            .collect();

        Self {
            messages: VecDeque::new(),
            peers,
        }
    }

    /// Queue a message for delivery
    pub fn send(&mut self, from: PeerId, to: PeerId, msg: Vec<u8>) {
        self.messages.push_back((from, to, msg));
    }

    /// Deliver the next message for a peer
    pub fn receive(&mut self, peer: &PeerId) -> Option<(PeerId, Vec<u8>)> {
        let idx = self.messages.iter().position(|(_, to, _)| to == peer)?;
        let (from, _, msg) = self.messages.remove(idx)?;
        Some((from, msg))
    }

    /// Get all peers in the network
    pub fn peers(&self) -> &[PeerId] {
        &self.peers
    }
}
