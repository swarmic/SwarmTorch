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

/// Deterministic network topology for [`MockNetworkHub`] (`ADR-0022` §3, `W10-03`).
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Topology {
    /// Every peer can send directly to every other peer.
    FullMesh,
    /// Bidirectional ring topology ($i \leftrightarrow (i \pm 1) \bmod N$).
    Ring,
    /// Star topology where all spokes communicate through `hub`.
    Star { hub: PeerId },
    /// Deterministic $k$-regular gossip ring ($i \to (i + 1 \dots i + \text{fanout}) \bmod N$).
    Gossip { fanout: usize },
}

#[cfg(feature = "std")]
struct HubInner {
    peers: Vec<PeerId>,
    topology: Topology,
    max_queue_depth: usize,
    blocked_links: std::collections::BTreeSet<(PeerId, PeerId)>,
    drop_budgets: std::collections::BTreeMap<(PeerId, PeerId), usize>,
    messages: VecDeque<(PeerId, PeerId, Vec<u8>)>,
    waiters: std::collections::BTreeMap<PeerId, crate::waiters::WaiterQueue>,
}

/// Multi-peer deterministic network hub with configurable topology and link fault injection
/// (`ADR-0022` §3, `W10-03`).
#[cfg(feature = "std")]
#[derive(Clone)]
pub struct MockNetworkHub {
    inner: std::sync::Arc<std::sync::Mutex<HubInner>>,
}

#[cfg(feature = "std")]
impl core::fmt::Debug for MockNetworkHub {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MockNetworkHub").finish()
    }
}

#[cfg(feature = "std")]
impl MockNetworkHub {
    /// Create a new [`MockNetworkHub`] over `peers` with `topology` and `max_queue_depth`.
    pub fn new(peers: Vec<PeerId>, topology: Topology, max_queue_depth: usize) -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::Mutex::new(HubInner {
                peers,
                topology,
                max_queue_depth,
                blocked_links: std::collections::BTreeSet::new(),
                drop_budgets: std::collections::BTreeMap::new(),
                messages: VecDeque::new(),
                waiters: std::collections::BTreeMap::new(),
            })),
        }
    }

    /// Create a [`MockPeerTransport`] handle bound to `local_peer`.
    pub fn peer_transport(&self, local_peer: PeerId) -> MockPeerTransport {
        MockPeerTransport {
            local_peer,
            hub: self.clone(),
        }
    }

    /// Block or unblock the directed link `(from, to)`.
    pub fn set_link_blocked(&self, from: PeerId, to: PeerId, blocked: bool) {
        if let Ok(mut guard) = self.inner.lock() {
            if blocked {
                guard.blocked_links.insert((from, to));
            } else {
                guard.blocked_links.remove(&(from, to));
            }
        }
    }

    /// Drop the next `count` frames transmitted over the directed link `(from, to)`.
    pub fn set_drop_next(&self, from: PeerId, to: PeerId, count: usize) {
        if let Ok(mut guard) = self.inner.lock() {
            if count == 0 {
                guard.drop_budgets.remove(&(from, to));
            } else {
                guard.drop_budgets.insert((from, to), count);
            }
        }
    }

    /// Return the adjacent outgoing neighbors of `peer` according to [`Topology`] and active link blocks.
    pub fn neighbors(&self, peer: PeerId) -> Vec<PeerId> {
        let Ok(guard) = self.inner.lock() else {
            return Vec::new();
        };
        Self::neighbors_locked(&guard, peer)
    }

    fn neighbors_locked(guard: &HubInner, peer: PeerId) -> Vec<PeerId> {
        let n = guard.peers.len();
        let Some(idx) = guard.peers.iter().position(|&p| p == peer) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        match guard.topology {
            Topology::FullMesh => {
                for &other in &guard.peers {
                    if other != peer && !guard.blocked_links.contains(&(peer, other)) {
                        out.push(other);
                    }
                }
            }
            Topology::Ring => {
                if n > 1 {
                    let prev = guard.peers[(idx + n - 1) % n];
                    let next = guard.peers[(idx + 1) % n];
                    for cand in [prev, next] {
                        if cand != peer
                            && !out.contains(&cand)
                            && !guard.blocked_links.contains(&(peer, cand))
                        {
                            out.push(cand);
                        }
                    }
                }
            }
            Topology::Star { hub } => {
                if peer == hub {
                    for &other in &guard.peers {
                        if other != hub && !guard.blocked_links.contains(&(peer, other)) {
                            out.push(other);
                        }
                    }
                } else if !guard.blocked_links.contains(&(peer, hub)) {
                    out.push(hub);
                }
            }
            Topology::Gossip { fanout } => {
                let k = fanout.min(n.saturating_sub(1));
                for step in 1..=k {
                    let cand = guard.peers[(idx + step) % n];
                    if cand != peer
                        && !out.contains(&cand)
                        && !guard.blocked_links.contains(&(peer, cand))
                    {
                        out.push(cand);
                    }
                }
            }
        }
        out
    }

    /// Send a message from `from` to `to` if permitted by topology and queue bounds.
    pub fn send_from(&self, from: PeerId, to: PeerId, payload: &[u8]) -> crate::Result<()> {
        let wakers = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| crate::Error::TransportUnavailable)?;
            let neighbors = Self::neighbors_locked(&guard, from);
            if !neighbors.contains(&to) {
                return Err(crate::Error::PeerNotFound);
            }
            if let Some(rem) = guard.drop_budgets.get_mut(&(from, to)) {
                if *rem > 0 {
                    *rem -= 1;
                    return Ok(());
                }
            }
            if guard.messages.len() >= guard.max_queue_depth {
                return Err(crate::Error::SendFailed);
            }
            guard.messages.push_back((from, to, payload.to_vec()));
            guard
                .waiters
                .get_mut(&to)
                .map(crate::waiters::WaiterQueue::take_all)
                .unwrap_or_default()
        };
        crate::waiters::wake_all(wakers);
        Ok(())
    }

    fn format_observed_message(
        from: PeerId,
        payload: Vec<u8>,
    ) -> crate::traits::ObservedTransportMessage {
        let bytes = from.as_bytes();
        let label = alloc::format!(
            "mock-link:{:02x}{:02x}{:02x}{:02x}",
            bytes[0],
            bytes[1],
            bytes[2],
            bytes[3]
        );
        crate::traits::ObservedTransportMessage {
            observed_source: crate::traits::ObservedSourceRef::LinkLabel(label),
            payload,
        }
    }

    fn take_message_for_locked(
        guard: &mut HubInner,
        recipient: PeerId,
    ) -> Option<crate::traits::ObservedTransportMessage> {
        let pos = guard
            .messages
            .iter()
            .position(|(_, to, _)| *to == recipient)?;
        let (from, _, payload) = guard.messages.remove(pos)?;
        Some(Self::format_observed_message(from, payload))
    }

    /// Pop the next queued message addressed to `recipient` with an unauthenticated observed
    /// link source hint (`ObservedSourceRef`, never a trusted `PeerId`).
    pub fn recv_for(
        &self,
        recipient: PeerId,
    ) -> crate::Result<crate::traits::ObservedTransportMessage> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| crate::Error::TransportUnavailable)?;
        Self::take_message_for_locked(&mut guard, recipient).ok_or(crate::Error::Timeout)
    }

    fn poll_recv_for(
        &self,
        recipient: PeerId,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<crate::Result<crate::traits::ObservedTransportMessage>> {
        let Ok(mut guard) = self.inner.lock() else {
            return core::task::Poll::Ready(Err(crate::Error::TransportUnavailable));
        };
        if let Some(msg) = Self::take_message_for_locked(&mut guard, recipient) {
            return core::task::Poll::Ready(Ok(msg));
        }
        guard
            .waiters
            .entry(recipient)
            .or_default()
            .register(cx.waker());
        core::task::Poll::Pending
    }
}

/// Evented async receive future for [`MockPeerTransport`] (`ST-06`).
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct MockRecvFuture<'a> {
    transport: &'a MockPeerTransport,
}

#[cfg(feature = "std")]
impl<'a> core::future::Future for MockRecvFuture<'a> {
    type Output = crate::Result<crate::traits::ObservedTransportMessage>;

    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        self.transport
            .hub
            .poll_recv_for(self.transport.local_peer, cx)
    }
}

/// Per-peer [` crate::traits::SwarmTransport`] handle backed by a shared [`MockNetworkHub`].
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct MockPeerTransport {
    local_peer: PeerId,
    hub: MockNetworkHub,
}

#[cfg(feature = "std")]
impl MockPeerTransport {
    /// Local peer identity of this transport endpoint.
    pub const fn local_peer(&self) -> PeerId {
        self.local_peer
    }
}

#[cfg(feature = "std")]
impl crate::traits::SwarmTransport for MockPeerTransport {
    type SendFuture<'a> = core::future::Ready<crate::Result<()>>;
    type RecvFuture<'a> = MockRecvFuture<'a>;
    type BroadcastFuture<'a> = core::future::Ready<crate::Result<crate::traits::BroadcastStats>>;
    type DiscoverFuture<'a> = core::future::Ready<crate::Result<Vec<PeerId>>>;

    fn send<'a>(&'a self, peer: PeerId, msg: &'a [u8]) -> Self::SendFuture<'a> {
        core::future::ready(self.hub.send_from(self.local_peer, peer, msg))
    }

    fn recv(&self) -> Self::RecvFuture<'_> {
        MockRecvFuture { transport: self }
    }

    fn broadcast<'a>(&'a self, msg: &'a [u8]) -> Self::BroadcastFuture<'a> {
        let neighbors = self.hub.neighbors(self.local_peer);
        let mut stats = crate::traits::BroadcastStats {
            peers_sent: neighbors.len(),
            confirmed: 0,
            failed: 0,
        };
        for peer in neighbors {
            match self.hub.send_from(self.local_peer, peer, msg) {
                Ok(()) => stats.confirmed += 1,
                Err(_) => stats.failed += 1,
            }
        }
        core::future::ready(Ok(stats))
    }

    fn discover(&self) -> Self::DiscoverFuture<'_> {
        core::future::ready(Ok(self.hub.neighbors(self.local_peer)))
    }

    fn capabilities(&self) -> TransportCapabilities {
        TransportCapabilities {
            reliability: ReliabilityClass::Reliable,
            bandwidth_class: BandwidthClass::High,
            max_message_size: 1_048_576,
            supports_multicast: true,
        }
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

    #[cfg(feature = "std")]
    #[tokio::test]
    async fn mock_peer_transport_recv_suspends_until_send_wakes_waiter() {
        use crate::traits::SwarmTransport;
        use core::time::Duration;

        let peer_a = PeerId::new([0xA1; 32]);
        let peer_b = PeerId::new([0xB2; 32]);
        let hub = MockNetworkHub::new(alloc::vec![peer_a, peer_b], Topology::FullMesh, 8);
        let t_a = hub.peer_transport(peer_a);
        let t_b = hub.peer_transport(peer_b);

        let rx_task = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(2), t_b.recv())
                .await
                .expect("evented recv must be woken cleanly by send")
                .unwrap()
        });

        tokio::task::yield_now().await;
        t_a.send(peer_b, b"wake-after-pending").await.unwrap();

        let observed = rx_task.await.unwrap();
        assert_eq!(observed.payload, b"wake-after-pending");
    }
}
