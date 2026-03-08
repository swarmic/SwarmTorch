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
#[derive(Debug)]
pub struct MockNetwork {
    /// Messages in transit
    pub messages: VecDeque<(PeerId, PeerId, Vec<u8>)>,
    /// Connected peers
    pub peers: Vec<PeerId>,
    /// Maximum queued messages before backpressure error.
    pub max_queue_depth: usize,
}

#[cfg(feature = "alloc")]
impl Default for MockNetwork {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Mock network send error.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockTransportError {
    QueueFull,
}

#[cfg(feature = "alloc")]
impl MockNetwork {
    /// Create a new mock network with the specified number of peers
    pub fn new(num_peers: usize) -> Self {
        let peers = (0..num_peers)
            .map(|i| {
                let mut seed = [0u8; 40];
                seed[..13].copy_from_slice(b"mock-peer-v1:");
                seed[13..21].copy_from_slice(&(i as u64).to_le_bytes());
                PeerId::from_public_key(&seed)
            })
            .collect();

        Self {
            messages: VecDeque::new(),
            peers,
            max_queue_depth: 1024,
        }
    }

    /// Queue a message for delivery
    pub fn send(
        &mut self,
        from: PeerId,
        to: PeerId,
        msg: Vec<u8>,
    ) -> Result<(), MockTransportError> {
        if self.messages.len() >= self.max_queue_depth {
            return Err(MockTransportError::QueueFull);
        }
        self.messages.push_back((from, to, msg));
        Ok(())
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

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn mock_peer_ids_are_distinct_and_fully_populated() {
        let net = MockNetwork::new(16);
        let mut unique = std::collections::HashSet::new();
        for peer in &net.peers {
            unique.insert(*peer.as_bytes());
            assert!(
                peer.as_bytes().iter().any(|b| *b != 0),
                "peer id should not be all-zero"
            );
        }
        assert_eq!(unique.len(), 16, "peer IDs should be unique");
    }

    #[test]
    fn mock_network_send_rejects_when_queue_full() {
        let mut net = MockNetwork::new(2);
        net.max_queue_depth = 1;
        let a = net.peers[0];
        let b = net.peers[1];
        net.send(a, b, vec![1]).unwrap();
        let err = net.send(a, b, vec![2]).unwrap_err();
        assert_eq!(err, MockTransportError::QueueFull);
    }

    #[test]
    fn mock_network_send_accepts_under_capacity() {
        let mut net = MockNetwork::new(2);
        net.max_queue_depth = 2;
        let a = net.peers[0];
        let b = net.peers[1];
        assert!(net.send(a, b, vec![1]).is_ok());
        assert!(net.send(a, b, vec![2]).is_ok());
    }
}
