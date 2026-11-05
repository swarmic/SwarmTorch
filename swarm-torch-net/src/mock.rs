//! Mock transport for testing (`ADR-0022`).
//!
//! Provides both a low-level [`PacketTransport`] implementation on [`MockTransport`]
//! and an `alloc`-enabled [`MockNetwork`] for multi-peer queue testing.

#[cfg(feature = "alloc")]
use alloc::collections::VecDeque;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::traits::{
    BandwidthClass, PacketCapabilities, PacketRecvError, PacketSendError, PacketTransport,
    ReceivedFrame, ReliabilityClass, TransportCapabilities,
};
#[cfg(feature = "alloc")]
use swarm_torch_core::traits::PeerId;

/// Mock transport for testing without real networking.
#[derive(Debug)]
pub struct MockTransport {
    /// Simulated failure rate (0.0 to 1.0).
    pub failure_rate: f32,
    /// Simulated latency in milliseconds.
    pub latency_ms: u32,
    /// Whether the transport is connected.
    pub connected: bool,
    /// Maximum frame size in bytes for [`PacketTransport`].
    pub max_frame_bytes: usize,
    /// Single-slot loopback frame buffer for `no_std` [`PacketTransport`] testing.
    loopback_frame: [u8; 256],
    loopback_len: usize,
    loopback_source: Option<u32>,
}

impl Default for MockTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl MockTransport {
    /// Create a new mock transport.
    pub fn new() -> Self {
        Self {
            failure_rate: 0.0,
            latency_ms: 0,
            connected: true,
            max_frame_bytes: 250,
            loopback_frame: [0u8; 256],
            loopback_len: 0,
            loopback_source: None,
        }
    }

    /// Set the simulated failure rate.
    pub fn with_failure_rate(mut self, rate: f32) -> Self {
        self.failure_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set the simulated latency.
    pub fn with_latency(mut self, latency_ms: u32) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Get transport capabilities.
    pub fn capabilities(&self) -> TransportCapabilities {
        TransportCapabilities {
            reliability: ReliabilityClass::Reliable,
            bandwidth_class: BandwidthClass::High,
            max_message_size: 65536,
            supports_multicast: true,
        }
    }
}

impl PacketTransport for MockTransport {
    type Endpoint = u32;
    type Error = &'static str;

    fn try_send_frame(
        &mut self,
        destination: &Self::Endpoint,
        frame: &[u8],
    ) -> core::result::Result<(), PacketSendError<Self::Error>> {
        if !self.connected {
            return Err(PacketSendError::Transport("mock transport disconnected"));
        }
        let max = self.max_frame_bytes.min(self.loopback_frame.len());
        if frame.len() > max {
            return Err(PacketSendError::FrameTooLarge {
                len: frame.len(),
                max,
            });
        }
        if self.loopback_source.is_some() {
            return Err(PacketSendError::WouldBlock);
        }
        self.loopback_frame[..frame.len()].copy_from_slice(frame);
        self.loopback_len = frame.len();
        self.loopback_source = Some(*destination);
        Ok(())
    }

    fn try_recv_frame(
        &mut self,
        buffer: &mut [u8],
    ) -> core::result::Result<Option<ReceivedFrame<Self::Endpoint>>, PacketRecvError<Self::Error>>
    {
        if !self.connected {
            return Err(PacketRecvError::Transport("mock transport disconnected"));
        }
        let Some(source) = self.loopback_source else {
            return Ok(None);
        };
        if buffer.len() < self.loopback_len {
            return Err(PacketRecvError::BufferTooSmall {
                required: self.loopback_len,
                available: buffer.len(),
            });
        }
        buffer[..self.loopback_len].copy_from_slice(&self.loopback_frame[..self.loopback_len]);
        let len = self.loopback_len;
        self.loopback_source = None;
        self.loopback_len = 0;
        Ok(Some(ReceivedFrame { source, len }))
    }

    fn capabilities(&self) -> PacketCapabilities {
        PacketCapabilities {
            reliability: ReliabilityClass::Reliable,
            bandwidth_class: BandwidthClass::High,
            max_frame_bytes: self.max_frame_bytes.min(self.loopback_frame.len()),
            max_message_bytes: 65536,
            supports_multicast: true,
            has_verified_link_principal: false,
        }
    }
}

/// A network of interconnected mock transports for testing.
#[cfg(feature = "alloc")]
#[derive(Debug)]
pub struct MockNetwork {
    /// Messages in transit.
    pub messages: VecDeque<(PeerId, PeerId, Vec<u8>)>,
    /// Connected peers.
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
    /// Create a new mock network with the specified number of peers.
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

    /// Queue a message for delivery.
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

    /// Deliver the next message for a peer.
    pub fn receive(&mut self, peer: &PeerId) -> Option<(PeerId, Vec<u8>)> {
        let idx = self.messages.iter().position(|(_, to, _)| to == peer)?;
        let (from, _, msg) = self.messages.remove(idx)?;
        Some((from, msg))
    }

    /// Get all peers in the network.
    pub fn peers(&self) -> &[PeerId] {
        &self.peers
    }
}

#[cfg(test)]
mod packet_tests {
    use super::*;

    #[test]
    fn packet_transport_roundtrip_and_bounds_work_in_no_std() {
        let mut transport = MockTransport::new();
        transport.max_frame_bytes = 8;

        assert_eq!(
            PacketTransport::try_send_frame(&mut transport, &7u32, &[0u8; 9]),
            Err(PacketSendError::FrameTooLarge { len: 9, max: 8 })
        );

        PacketTransport::try_send_frame(&mut transport, &7u32, &[1, 2, 3, 4]).unwrap();
        assert_eq!(
            PacketTransport::try_send_frame(&mut transport, &7u32, &[5]),
            Err(PacketSendError::WouldBlock)
        );

        let mut small_buf = [0u8; 2];
        assert_eq!(
            PacketTransport::try_recv_frame(&mut transport, &mut small_buf),
            Err(PacketRecvError::BufferTooSmall {
                required: 4,
                available: 2
            })
        );

        let mut buf = [0u8; 8];
        let received = PacketTransport::try_recv_frame(&mut transport, &mut buf)
            .unwrap()
            .expect("frame present");
        assert_eq!(received, ReceivedFrame { source: 7, len: 4 });
        assert_eq!(&buf[..4], &[1, 2, 3, 4]);
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;
    use alloc::vec;

    #[test]
    fn mock_peer_ids_are_distinct_and_fully_populated() {
        let net = MockNetwork::new(16);
        let mut unique = BTreeSet::new();
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
