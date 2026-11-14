//! Chunked UDP packet and gossip transport (`ADR-0022` §1.2, §1.3, §2, `W11-03`).
//!
//! Layers [`ChunkedFrameCodec`] and [`ObservedSourceReassembler`] over `std::net::UdpSocket`
//! (or an [`InMemoryUdpDatagramHub`] in network-sandboxed environments) so datagrams are
//! fragmented with the frozen 16-byte `"ST"` header + `CRC-16/IBM-3740` and reassembled
//! pre-authentication by `(adapter_generation, observed_source: SocketAddr, transfer_id)` before
//! entering the `MessageEnvelope` / `BundledGossipDeliveryV1` verification pipeline.

use std::collections::{BTreeMap, VecDeque};
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use swarm_torch_core::replay::ReplayProtection;
use swarm_torch_core::traits::PeerId;

use crate::chunked::{ChunkedFrameCodec, ObservedSourceReassembler, ReassemblerLimits};
use crate::protocol::{MessageEnvelope, VerifyError};
use crate::traits::{
    BandwidthClass, BroadcastStats, PacketCapabilities, PacketRecvError, PacketSendError,
    PacketTransport, ReceivedFrame, ReliabilityClass, SwarmTransport, TransportCapabilities,
};
use crate::{Error, Result};

/// Low-level [`PacketTransport`] implementation over a non-blocking [`std::net::UdpSocket`]
/// (`ADR-0022` §1.2).
#[derive(Debug)]
pub struct StdUdpPacketTransport {
    socket: UdpSocket,
    max_frame_bytes: usize,
    max_message_bytes: usize,
}

impl StdUdpPacketTransport {
    /// Bind a non-blocking UDP socket to `bind_addr` with `max_frame_bytes` ($M$) and `max_message_bytes` ($L_{\max}$).
    pub fn bind(
        bind_addr: SocketAddr,
        max_frame_bytes: usize,
        max_message_bytes: usize,
    ) -> Result<Self> {
        let _ = ChunkedFrameCodec::stride_for_mtu(max_frame_bytes)
            .map_err(|_| Error::InvalidMessage)?;
        let socket = UdpSocket::bind(bind_addr).map_err(|_| Error::ConnectionFailed)?;
        socket
            .set_nonblocking(true)
            .map_err(|_| Error::ConnectionFailed)?;
        Ok(Self {
            socket,
            max_frame_bytes,
            max_message_bytes,
        })
    }

    /// Local bound socket address.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.socket
            .local_addr()
            .map_err(|_| Error::ConnectionFailed)
    }
}

impl PacketTransport for StdUdpPacketTransport {
    type Endpoint = SocketAddr;
    type Error = io::ErrorKind;

    fn try_send_frame(
        &mut self,
        destination: &Self::Endpoint,
        frame: &[u8],
    ) -> core::result::Result<(), PacketSendError<Self::Error>> {
        if frame.len() > self.max_frame_bytes {
            return Err(PacketSendError::FrameTooLarge {
                len: frame.len(),
                max: self.max_frame_bytes,
            });
        }
        match self.socket.send_to(frame, destination) {
            Ok(written) if written == frame.len() => Ok(()),
            Ok(_) => Err(PacketSendError::Transport(io::ErrorKind::WriteZero)),
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Err(PacketSendError::WouldBlock),
            Err(e) => Err(PacketSendError::Transport(e.kind())),
        }
    }

    fn try_recv_frame(
        &mut self,
        buffer: &mut [u8],
    ) -> core::result::Result<Option<ReceivedFrame<Self::Endpoint>>, PacketRecvError<Self::Error>>
    {
        if buffer.len() < self.max_frame_bytes {
            return Err(PacketRecvError::BufferTooSmall {
                required: self.max_frame_bytes,
                available: buffer.len(),
            });
        }
        match self.socket.recv_from(buffer) {
            Ok((len, source)) => {
                if len > self.max_frame_bytes {
                    return Err(PacketRecvError::TruncatedFrame);
                }
                Ok(Some(ReceivedFrame { source, len }))
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(PacketRecvError::Transport(e.kind())),
        }
    }

    fn capabilities(&self) -> PacketCapabilities {
        PacketCapabilities {
            reliability: ReliabilityClass::BestEffort,
            bandwidth_class: BandwidthClass::Medium,
            max_frame_bytes: self.max_frame_bytes,
            max_message_bytes: self.max_message_bytes,
            supports_multicast: false,
            has_verified_link_principal: false,
        }
    }
}

type DatagramQueueMap = BTreeMap<SocketAddr, VecDeque<(SocketAddr, Vec<u8>)>>;

/// Shared in-memory UDP datagram bus used when OS socket creation is restricted by sandbox policy.
#[derive(Default, Clone)]
pub struct InMemoryUdpDatagramHub {
    queues: Arc<Mutex<DatagramQueueMap>>,
}

impl InMemoryUdpDatagramHub {
    /// Create a new in-memory UDP datagram hub.
    pub fn new() -> Self {
        Self::default()
    }
}

enum UdpBackend {
    OsSocket(UdpSocket),
    InMemory(InMemoryUdpDatagramHub),
}

/// Chunked UDP [`SwarmTransport`] integrating [`ChunkedFrameCodec`] and
/// [`ObservedSourceReassembler<SocketAddr>`] (`ADR-0022` §2, `W11-03`).
pub struct UdpGossipTransport {
    local_peer: PeerId,
    local_addr: SocketAddr,
    adapter_generation: u64,
    limits: ReassemblerLimits,
    backend: UdpBackend,
    next_transfer_id: AtomicU32,
    start_instant: Instant,
    peers: RwLock<BTreeMap<PeerId, SocketAddr>>,
    reassembler: Mutex<ObservedSourceReassembler<SocketAddr>>,
    inbox: Mutex<VecDeque<(SocketAddr, Vec<u8>)>>,
}

impl core::fmt::Debug for UdpGossipTransport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UdpGossipTransport")
            .field("local_peer", &self.local_peer)
            .field("local_addr", &self.local_addr)
            .field("adapter_generation", &self.adapter_generation)
            .field("limits", &self.limits)
            .finish()
    }
}

impl UdpGossipTransport {
    /// Bind a new [`UdpGossipTransport`] to `bind_addr` with `adapter_generation` and `limits`.
    pub fn bind(
        local_peer: PeerId,
        bind_addr: SocketAddr,
        adapter_generation: u64,
        limits: ReassemblerLimits,
    ) -> Result<Arc<Self>> {
        let reassembler =
            ObservedSourceReassembler::new(limits).map_err(|_| Error::InvalidMessage)?;
        let socket = UdpSocket::bind(bind_addr).map_err(|_| Error::ConnectionFailed)?;
        socket
            .set_nonblocking(true)
            .map_err(|_| Error::ConnectionFailed)?;
        let local_addr = socket.local_addr().map_err(|_| Error::ConnectionFailed)?;

        Ok(Arc::new(Self {
            local_peer,
            local_addr,
            adapter_generation,
            limits,
            backend: UdpBackend::OsSocket(socket),
            next_transfer_id: AtomicU32::new(1),
            start_instant: Instant::now(),
            peers: RwLock::new(BTreeMap::new()),
            reassembler: Mutex::new(reassembler),
            inbox: Mutex::new(VecDeque::new()),
        }))
    }

    /// Create a [`UdpGossipTransport`] endpoint attached to an [`InMemoryUdpDatagramHub`] that
    /// executes the identical [`ChunkedFrameCodec`] fragmentation and [`ObservedSourceReassembler`]
    /// pipeline.
    pub fn bind_in_memory(
        local_peer: PeerId,
        virtual_addr: SocketAddr,
        adapter_generation: u64,
        limits: ReassemblerLimits,
        hub: InMemoryUdpDatagramHub,
    ) -> Result<Arc<Self>> {
        let reassembler =
            ObservedSourceReassembler::new(limits).map_err(|_| Error::InvalidMessage)?;
        if let Ok(mut map) = hub.queues.lock() {
            map.entry(virtual_addr).or_default();
        }
        Ok(Arc::new(Self {
            local_peer,
            local_addr: virtual_addr,
            adapter_generation,
            limits,
            backend: UdpBackend::InMemory(hub),
            next_transfer_id: AtomicU32::new(1),
            start_instant: Instant::now(),
            peers: RwLock::new(BTreeMap::new()),
            reassembler: Mutex::new(reassembler),
            inbox: Mutex::new(VecDeque::new()),
        }))
    }

    /// Local bound socket address.
    pub const fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Register a peer's outbound `SocketAddr` in the transport directory (`PeerId` lookup is
    /// strictly outbound; inbound identity always comes from authenticated envelope verification).
    pub fn register_peer(&self, peer: PeerId, addr: SocketAddr) -> Result<()> {
        self.peers
            .write()
            .map_err(|_| Error::TransportUnavailable)?
            .insert(peer, addr);
        Ok(())
    }

    fn elapsed_mono_ms(&self) -> u64 {
        self.start_instant.elapsed().as_millis() as u64
    }

    /// Fragment `msg` via [`ChunkedFrameCodec::encode_message`] and transmit all chunk frames to `dest`.
    pub fn send_to_addr(&self, dest: SocketAddr, msg: &[u8]) -> Result<u32> {
        let transfer_id = self.next_transfer_id.fetch_add(1, Ordering::Relaxed);
        let frames = ChunkedFrameCodec::encode_message(
            transfer_id,
            self.limits.mtu,
            self.limits.max_message_bytes,
            self.limits.max_chunks_per_transfer,
            msg,
        )
        .map_err(|_| Error::InvalidMessage)?;

        match &self.backend {
            UdpBackend::OsSocket(socket) => {
                for frame in &frames {
                    socket.send_to(frame, dest).map_err(|_| Error::SendFailed)?;
                }
            }
            UdpBackend::InMemory(hub) => {
                let mut guard = hub.queues.lock().map_err(|_| Error::TransportUnavailable)?;
                let q = guard.get_mut(&dest).ok_or(Error::PeerNotFound)?;
                for frame in frames {
                    q.push_back((self.local_addr, frame));
                }
            }
        }
        Ok(transfer_id)
    }

    /// Drain pending UDP datagrams into [`ObservedSourceReassembler`] and return the next
    /// completed `(observed_source_addr, reassembled_payload)` if available.
    pub fn poll_reassembled_once(&self) -> Result<Option<(SocketAddr, Vec<u8>)>> {
        if let Some(ready) = self
            .inbox
            .lock()
            .map_err(|_| Error::TransportUnavailable)?
            .pop_front()
        {
            return Ok(Some(ready));
        }

        match &self.backend {
            UdpBackend::OsSocket(socket) => {
                let mut frame_buf = vec![0u8; self.limits.mtu + 64];
                loop {
                    match socket.recv_from(&mut frame_buf) {
                        Ok((len, source_addr)) => {
                            let now_ms = self.elapsed_mono_ms();
                            let mut reasm = self
                                .reassembler
                                .lock()
                                .map_err(|_| Error::TransportUnavailable)?;
                            if let Ok(Some(done)) = reasm.ingest_frame(
                                self.adapter_generation,
                                source_addr,
                                &frame_buf[..len],
                                now_ms,
                            ) {
                                drop(reasm);
                                if let Ok(mut inbox) = self.inbox.lock() {
                                    inbox.push_back((done.key.observed_source, done.payload));
                                }
                            }
                        }
                        Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                        Err(_) => return Err(Error::ReceiveFailed),
                    }
                }
            }
            UdpBackend::InMemory(hub) => {
                let pending: Vec<(SocketAddr, Vec<u8>)> = {
                    let mut guard = hub.queues.lock().map_err(|_| Error::TransportUnavailable)?;
                    guard
                        .get_mut(&self.local_addr)
                        .map(|q| q.drain(..).collect())
                        .unwrap_or_default()
                };
                for (source_addr, frame) in pending {
                    let now_ms = self.elapsed_mono_ms();
                    let mut reasm = self
                        .reassembler
                        .lock()
                        .map_err(|_| Error::TransportUnavailable)?;
                    if let Ok(Some(done)) =
                        reasm.ingest_frame(self.adapter_generation, source_addr, &frame, now_ms)
                    {
                        drop(reasm);
                        if let Ok(mut inbox) = self.inbox.lock() {
                            inbox.push_back((done.key.observed_source, done.payload));
                        }
                    }
                }
            }
        }

        let next = self
            .inbox
            .lock()
            .map_err(|_| Error::TransportUnavailable)?
            .pop_front();
        Ok(next)
    }

    /// Poll for a reassembled datagram payload and immediately verify it through the
    /// [`MessageEnvelope::deserialize_bounded`] + [`MessageEnvelope::verify_authenticated`]
    /// pipeline (`ADR-0022` §2).
    pub fn poll_recv_verified_envelope_v1(
        &self,
        expected_swarm_instance_id: &[u8; 16],
        expected_round_id: u64,
        replay: &mut ReplayProtection,
        now_secs: u32,
    ) -> core::result::Result<Option<(SocketAddr, PeerId, MessageEnvelope)>, VerifyError> {
        let Ok(Some((source_addr, bytes))) = self.poll_reassembled_once() else {
            return Ok(None);
        };
        let env = MessageEnvelope::deserialize_bounded(&bytes, self.limits.max_message_bytes)?;
        if env.swarm_instance_id != *expected_swarm_instance_id || env.round_id != expected_round_id
        {
            return Err(VerifyError::MalformedEnvelope);
        }
        env.verify_authenticated(replay, now_secs)?;
        let sender_peer = env
            .sender_peer_id()
            .map_err(|_| VerifyError::InvalidSenderKey)?;
        Ok(Some((source_addr, sender_peer, env)))
    }
}

impl SwarmTransport for UdpGossipTransport {
    type SendFuture<'a> = Pin<Box<dyn core::future::Future<Output = Result<()>> + Send + 'a>>;
    type RecvFuture<'a> = Pin<
        Box<
            dyn core::future::Future<Output = Result<crate::traits::ObservedTransportMessage>>
                + Send
                + 'a,
        >,
    >;
    type BroadcastFuture<'a> =
        Pin<Box<dyn core::future::Future<Output = Result<BroadcastStats>> + Send + 'a>>;
    type DiscoverFuture<'a> =
        Pin<Box<dyn core::future::Future<Output = Result<Vec<PeerId>>> + Send + 'a>>;

    fn send<'a>(&'a self, peer: PeerId, msg: &'a [u8]) -> Self::SendFuture<'a> {
        Box::pin(async move {
            let dest = {
                let guard = self.peers.read().map_err(|_| Error::TransportUnavailable)?;
                *guard.get(&peer).ok_or(Error::PeerNotFound)?
            };
            let _ = self.send_to_addr(dest, msg)?;
            Ok(())
        })
    }

    fn recv(&self) -> Self::RecvFuture<'_> {
        Box::pin(async move {
            for _ in 0..200 {
                if let Some((source_addr, payload)) = self.poll_reassembled_once()? {
                    return Ok(crate::traits::ObservedTransportMessage {
                        observed_source: crate::traits::ObservedSourceRef::from(source_addr),
                        payload,
                    });
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(Error::Timeout)
        })
    }

    fn broadcast<'a>(&'a self, msg: &'a [u8]) -> Self::BroadcastFuture<'a> {
        Box::pin(async move {
            let targets: Vec<SocketAddr> = self
                .peers
                .read()
                .map_err(|_| Error::TransportUnavailable)?
                .values()
                .copied()
                .collect();
            let mut stats = BroadcastStats {
                peers_sent: targets.len(),
                confirmed: 0,
                failed: 0,
            };
            for dest in targets {
                match self.send_to_addr(dest, msg) {
                    Ok(_) => stats.confirmed += 1,
                    Err(_) => stats.failed += 1,
                }
            }
            Ok(stats)
        })
    }

    fn discover(&self) -> Self::DiscoverFuture<'_> {
        Box::pin(async move {
            let peers: Vec<PeerId> = self
                .peers
                .read()
                .map_err(|_| Error::TransportUnavailable)?
                .keys()
                .copied()
                .collect();
            Ok(peers)
        })
    }

    fn capabilities(&self) -> TransportCapabilities {
        TransportCapabilities {
            reliability: ReliabilityClass::BestEffort,
            bandwidth_class: BandwidthClass::Medium,
            max_message_size: self.limits.max_message_bytes,
            supports_multicast: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::MessageType;
    use swarm_torch_core::crypto::{KeyPair, MessageAuth};
    use swarm_torch_core::replay::{CohortAllowlist, ReplayStartupPolicyV1};

    #[tokio::test]
    async fn udp_gossip_transport_recv_returns_observed_source_without_sentinel_identity() {
        let kp_a = KeyPair::from_seed([0x21; 32]).unwrap();
        let kp_b = KeyPair::from_seed([0x22; 32]).unwrap();
        let limits = ReassemblerLimits::default();
        let hub = InMemoryUdpDatagramHub::new();
        let addr_a: SocketAddr = "127.0.0.1:42101".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:42102".parse().unwrap();

        let udp_a =
            UdpGossipTransport::bind_in_memory(kp_a.peer_id(), addr_a, 1, limits, hub.clone())
                .unwrap();
        let udp_b =
            UdpGossipTransport::bind_in_memory(kp_b.peer_id(), addr_b, 1, limits, hub).unwrap();

        // Notice: udp_b intentionally does NOT register peer_a's IP:port in any socket map.
        udp_a
            .send_to_addr(addr_b, b"raw-unverified-payload")
            .unwrap();
        let msg = udp_b.recv().await.unwrap();
        assert_eq!(
            msg.observed_source,
            crate::traits::ObservedSourceRef::from(addr_a)
        );
        assert_eq!(msg.payload, b"raw-unverified-payload");
    }

    #[tokio::test]
    async fn udp_gossip_transport_fragments_and_reassembles_multi_chunk_signed_envelope() {
        let kp_a = KeyPair::from_seed([0x31; 32]).unwrap();
        let kp_b = KeyPair::from_seed([0x32; 32]).unwrap();
        let peer_a = kp_a.peer_id();
        let peer_b = kp_b.peer_id();

        let limits = ReassemblerLimits {
            mtu: 128, // stride P = 112 bytes -> forces multi-chunk fragmentation
            max_message_bytes: 16_384,
            max_chunks_per_transfer: 64,
            ..ReassemblerLimits::default()
        };

        let hub = InMemoryUdpDatagramHub::new();
        let addr_a: SocketAddr = "127.0.0.1:42001".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:42002".parse().unwrap();
        let udp_a =
            UdpGossipTransport::bind_in_memory(peer_a, addr_a, 1, limits, hub.clone()).unwrap();
        let udp_b = UdpGossipTransport::bind_in_memory(peer_b, addr_b, 1, limits, hub).unwrap();

        udp_a.register_peer(peer_b, udp_b.local_addr()).unwrap();
        udp_b.register_peer(peer_a, udp_a.local_addr()).unwrap();

        // Build a signed v1 MessageEnvelope larger than MTU=128 (~1200 bytes)
        let sender_pk = *kp_a.public_key();
        let auth_a = MessageAuth::new(kp_a);
        let swarm_id = [0xAB; 16];
        let body = vec![0x5Au8; 1024];
        let env = MessageEnvelope::new_v1(
            sender_pk,
            swarm_id,
            1,
            5,
            MessageType::GradientUpdate,
            body.clone(),
        )
        .with_sequence(1)
        .with_timestamp(1_700_000_000)
        .sign_with(&auth_a);
        let wire_bytes = env.serialize().unwrap();
        assert!(wire_bytes.len() > limits.mtu * 8);

        // Send over UdpGossipTransport (fragments into ~10 ChunkedFrameCodec datagrams)
        udp_a.send(peer_b, &wire_bytes).await.unwrap();

        let allowlist = CohortAllowlist::from_peers([peer_a, peer_b]);
        let mut replay = ReplayProtection::try_with_config(64, 60)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(allowlist);

        // Poll and verify through the MessageEnvelope gate on receiver B
        let mut verified = None;
        for _ in 0..100 {
            if let Some(res) = udp_b
                .poll_recv_verified_envelope_v1(&swarm_id, 5, &mut replay, 1_700_000_000)
                .unwrap()
            {
                verified = Some(res);
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }

        let (obs_addr, verified_peer, verified_env) =
            verified.expect("multi-chunk UDP envelope must reassemble and verify");
        assert_eq!(obs_addr, udp_a.local_addr());
        assert_eq!(verified_peer, peer_a);
        assert_eq!(verified_env.payload, body);
    }
}
