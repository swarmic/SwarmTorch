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
use std::task::{Context, Poll};
use std::time::Instant;

use swarm_torch_core::observe::{EventRecord, RunEventEmitter};
use swarm_torch_core::replay::{ReplayProtection, ReplayStartupPolicyV1};
use swarm_torch_core::traits::PeerId;

use crate::audit::{AuditSeverity, TransportAuditBufferV1, TransportDiagnosticsSnapshotV1};
use crate::chunked::{ChunkError, ChunkedFrameCodec, ObservedSourceReassembler, ReassemblerLimits};
use crate::protocol::{EnvelopeVersionPolicyV1, MessageEnvelope, VerifyError};

/// Default maximum consecutive transient UDP socket errors (`WSAECONNRESET` / `Interrupted`)
/// absorbed per synchronous [`UdpGossipTransport::poll_reassembled_once`] drain pass.
pub const DEFAULT_MAX_UDP_TRANSIENT_ERRORS_PER_POLL: usize = 256;
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

#[derive(Default)]
struct InMemoryUdpHubState {
    queues: DatagramQueueMap,
    waiters: BTreeMap<SocketAddr, crate::waiters::WaiterQueue>,
}

/// Shared in-memory UDP datagram bus used when OS socket creation is restricted by sandbox policy.
#[derive(Default, Clone)]
pub struct InMemoryUdpDatagramHub {
    inner: Arc<Mutex<InMemoryUdpHubState>>,
}

impl InMemoryUdpDatagramHub {
    /// Create a new in-memory UDP datagram hub.
    pub fn new() -> Self {
        Self::default()
    }
}

enum OsUdpSocketSlot {
    Std(UdpSocket),
    #[cfg(feature = "udp-transport")]
    Tokio(Arc<tokio::net::UdpSocket>),
    #[cfg(feature = "udp-transport")]
    Empty,
}

impl OsUdpSocketSlot {
    #[cfg(feature = "udp-transport")]
    fn ensure_tokio(&mut self) -> Result<Arc<tokio::net::UdpSocket>> {
        if matches!(self, Self::Std(_)) {
            let Self::Std(std_socket) = core::mem::replace(self, Self::Empty) else {
                return Err(Error::TransportUnavailable);
            };
            let tokio_socket = tokio::net::UdpSocket::from_std(std_socket)
                .map_err(|_| Error::TransportUnavailable)?;
            *self = Self::Tokio(Arc::new(tokio_socket));
        }
        match self {
            Self::Tokio(ref socket) => Ok(Arc::clone(socket)),
            Self::Std(_) | Self::Empty => Err(Error::TransportUnavailable),
        }
    }
}

enum UdpBackend {
    OsSocket(Mutex<OsUdpSocketSlot>),
    InMemory(InMemoryUdpDatagramHub),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UdpRecvErrorDisposition {
    BreakWouldBlock,
    ContinueTransient,
    FatalReceiveFailed,
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
    audit: TransportAuditBufferV1,
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
            backend: UdpBackend::OsSocket(Mutex::new(OsUdpSocketSlot::Std(socket))),
            next_transfer_id: AtomicU32::new(1),
            start_instant: Instant::now(),
            peers: RwLock::new(BTreeMap::new()),
            reassembler: Mutex::new(reassembler),
            inbox: Mutex::new(VecDeque::new()),
            audit: TransportAuditBufferV1::new(local_peer),
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
        if let Ok(mut state) = hub.inner.lock() {
            state.queues.entry(virtual_addr).or_default();
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
            audit: TransportAuditBufferV1::new(local_peer),
        }))
    }

    /// Return a point-in-time snapshot of bounded UDP drop, quarantine, and ICMP-reset diagnostics.
    pub fn diagnostics(&self) -> TransportDiagnosticsSnapshotV1 {
        self.audit.snapshot()
    }

    /// Drain buffered validated security audit [`EventRecord`] entries (`Decision A8`).
    pub fn drain_audit_events(&self) -> Vec<EventRecord> {
        self.audit.drain_records()
    }

    /// Flush buffered security audit [`EventRecord`] entries into a [`RunEventEmitter`].
    pub fn flush_audit_events<E: RunEventEmitter>(
        &self,
        emitter: &E,
    ) -> core::result::Result<usize, E::Error> {
        self.audit.flush_to_emitter(emitter)
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
        u64::try_from(self.start_instant.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    fn lookup_peer_addr(&self, peer: PeerId) -> Result<SocketAddr> {
        let guard = self.peers.read().map_err(|_| Error::TransportUnavailable)?;
        guard.get(&peer).copied().ok_or(Error::PeerNotFound)
    }

    fn encode_chunk_frames(&self, msg: &[u8]) -> Result<(u32, Vec<Vec<u8>>)> {
        let transfer_id = self.next_transfer_id.fetch_add(1, Ordering::Relaxed);
        let frames = ChunkedFrameCodec::encode_message(
            transfer_id,
            self.limits.mtu,
            self.limits.max_message_bytes,
            self.limits.max_chunks_per_transfer,
            msg,
        )
        .map_err(|_| Error::InvalidMessage)?;
        Ok((transfer_id, frames))
    }

    fn send_in_memory(
        &self,
        hub: &InMemoryUdpDatagramHub,
        dest: SocketAddr,
        frames: Vec<Vec<u8>>,
    ) -> Result<()> {
        let wakers = {
            let mut state = hub.inner.lock().map_err(|_| Error::TransportUnavailable)?;
            let q = state.queues.get_mut(&dest).ok_or(Error::PeerNotFound)?;
            for frame in frames {
                q.push_back((self.local_addr, frame));
            }
            state
                .waiters
                .get_mut(&dest)
                .map(crate::waiters::WaiterQueue::take_all)
                .unwrap_or_default()
        };
        crate::waiters::wake_all(wakers);
        Ok(())
    }

    fn ingest_datagram(&self, source_addr: SocketAddr, datagram: &[u8]) -> Result<()> {
        let now_ms = self.elapsed_mono_ms();
        let completed = {
            let mut reasm = self
                .reassembler
                .lock()
                .map_err(|_| Error::TransportUnavailable)?;
            reasm.ingest_frame(self.adapter_generation, source_addr, datagram, now_ms)
        };
        match completed {
            Ok(Some(done)) => {
                let mut inbox = self.inbox.lock().map_err(|_| Error::TransportUnavailable)?;
                inbox.push_back((done.key.observed_source, done.payload));
            }
            Ok(None) => {}
            Err(chunk_err) => {
                let is_initial_conflict = matches!(
                    chunk_err,
                    ChunkError::ConflictingTotalChunks | ChunkError::ConflictingChunkPayload
                );
                let is_repeat_quarantine = matches!(
                    chunk_err,
                    ChunkError::TransferQuarantined | ChunkError::SourceQuarantined
                );
                let is_quarantine = is_initial_conflict || is_repeat_quarantine;
                let (severity, event_name) = if is_initial_conflict {
                    (
                        AuditSeverity::Critical,
                        "swarmtorch.net.udp.source_quarantined",
                    )
                } else if is_repeat_quarantine {
                    (AuditSeverity::Noisy, "swarmtorch.net.udp.quarantined_drop")
                } else {
                    (AuditSeverity::Noisy, "swarmtorch.net.udp.datagram_rejected")
                };
                self.audit.record_event(
                    severity,
                    event_name,
                    "udp",
                    Some(source_addr.to_string()),
                    &chunk_err.to_string(),
                    move |c| {
                        if is_quarantine {
                            c.quarantined_udp_events = c.quarantined_udp_events.saturating_add(1);
                        } else {
                            c.dropped_malformed_frames =
                                c.dropped_malformed_frames.saturating_add(1);
                        }
                    },
                );
            }
        }
        Ok(())
    }

    fn classify_udp_recv_error(&self, err: &io::Error) -> UdpRecvErrorDisposition {
        match err.kind() {
            io::ErrorKind::WouldBlock => UdpRecvErrorDisposition::BreakWouldBlock,
            io::ErrorKind::Interrupted
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionAborted => {
                self.audit.record_event(
                    AuditSeverity::Noisy,
                    "swarmtorch.net.udp.icmp_reset_ignored",
                    "udp",
                    None,
                    &err.to_string(),
                    |c| {
                        c.icmp_port_unreachable_resets =
                            c.icmp_port_unreachable_resets.saturating_add(1);
                    },
                );
                UdpRecvErrorDisposition::ContinueTransient
            }
            // Windows `WSAEMSGSIZE` (10040) on oversized UDP datagrams must drop & isolate rather
            // than tearing down the shared socket receive loop.
            _ if err.raw_os_error() == Some(10040) => {
                self.audit.record_event(
                    AuditSeverity::Noisy,
                    "swarmtorch.net.udp.datagram_rejected",
                    "udp",
                    None,
                    &err.to_string(),
                    |c| {
                        c.dropped_malformed_frames = c.dropped_malformed_frames.saturating_add(1);
                    },
                );
                UdpRecvErrorDisposition::ContinueTransient
            }
            _ => {
                self.audit.record_event(
                    AuditSeverity::Critical,
                    "swarmtorch.net.udp.recv_error",
                    "udp",
                    None,
                    &err.to_string(),
                    |c| {
                        c.dropped_connection_errors = c.dropped_connection_errors.saturating_add(1);
                    },
                );
                UdpRecvErrorDisposition::FatalReceiveFailed
            }
        }
    }

    fn pop_inbox_observed(&self) -> Result<Option<crate::traits::ObservedTransportMessage>> {
        let next = self
            .inbox
            .lock()
            .map_err(|_| Error::TransportUnavailable)?
            .pop_front();
        Ok(next.map(
            |(source_addr, payload)| crate::traits::ObservedTransportMessage {
                observed_source: crate::traits::ObservedSourceRef::from(source_addr),
                payload,
            },
        ))
    }

    fn poll_recv_in_memory(
        &self,
        hub: &InMemoryUdpDatagramHub,
        cx: &mut Context<'_>,
    ) -> Poll<Result<crate::traits::ObservedTransportMessage>> {
        loop {
            match self.pop_inbox_observed() {
                Ok(Some(msg)) => return Poll::Ready(Ok(msg)),
                Ok(None) => {}
                Err(err) => return Poll::Ready(Err(err)),
            }

            let pending: Vec<(SocketAddr, Vec<u8>)> = {
                let Ok(mut state) = hub.inner.lock() else {
                    return Poll::Ready(Err(Error::TransportUnavailable));
                };
                state
                    .queues
                    .get_mut(&self.local_addr)
                    .map(|q| q.drain(..).collect())
                    .unwrap_or_default()
            };
            for (source_addr, frame) in pending {
                if let Err(err) = self.ingest_datagram(source_addr, &frame) {
                    return Poll::Ready(Err(err));
                }
            }

            match self.pop_inbox_observed() {
                Ok(Some(msg)) => return Poll::Ready(Ok(msg)),
                Ok(None) => {}
                Err(err) => return Poll::Ready(Err(err)),
            }

            let Ok(mut state) = hub.inner.lock() else {
                return Poll::Ready(Err(Error::TransportUnavailable));
            };
            let is_empty = state
                .queues
                .get(&self.local_addr)
                .map_or(true, VecDeque::is_empty);
            if is_empty {
                state
                    .waiters
                    .entry(self.local_addr)
                    .or_default()
                    .register(cx.waker());
                return Poll::Pending;
            }
        }
    }

    #[cfg(feature = "udp-transport")]
    async fn send_os_async(
        &self,
        socket_slot: &Mutex<OsUdpSocketSlot>,
        dest: SocketAddr,
        frames: Vec<Vec<u8>>,
    ) -> Result<()> {
        let sock = {
            let mut slot = socket_slot
                .lock()
                .map_err(|_| Error::TransportUnavailable)?;
            slot.ensure_tokio()?
        };
        for frame in &frames {
            sock.send_to(frame, dest)
                .await
                .map_err(|_| Error::SendFailed)?;
        }
        Ok(())
    }

    #[cfg(feature = "udp-transport")]
    async fn recv_os_async(
        &self,
        socket_slot: &Mutex<OsUdpSocketSlot>,
    ) -> Result<crate::traits::ObservedTransportMessage> {
        if let Some(msg) = self.pop_inbox_observed()? {
            return Ok(msg);
        }
        let sock = {
            let mut slot = socket_slot
                .lock()
                .map_err(|_| Error::TransportUnavailable)?;
            slot.ensure_tokio()?
        };
        let mut frame_buf = vec![0u8; self.limits.mtu + 64];
        let mut consecutive_transient = 0usize;
        loop {
            if let Some(msg) = self.pop_inbox_observed()? {
                return Ok(msg);
            }
            match sock.recv_from(&mut frame_buf).await {
                Ok((len, source_addr)) => {
                    consecutive_transient = 0;
                    self.ingest_datagram(source_addr, &frame_buf[..len])?;
                }
                Err(ref err) => match self.classify_udp_recv_error(err) {
                    UdpRecvErrorDisposition::BreakWouldBlock
                    | UdpRecvErrorDisposition::ContinueTransient => {
                        consecutive_transient += 1;
                        if consecutive_transient >= 64 {
                            consecutive_transient = 0;
                            let mut yielded = false;
                            core::future::poll_fn(|cx| {
                                if !yielded {
                                    yielded = true;
                                    cx.waker().wake_by_ref();
                                    Poll::Pending
                                } else {
                                    Poll::Ready(())
                                }
                            })
                            .await;
                        }
                        continue;
                    }
                    UdpRecvErrorDisposition::FatalReceiveFailed => {
                        return Err(Error::ReceiveFailed);
                    }
                },
            }
        }
    }

    /// Fragment `msg` via [`ChunkedFrameCodec::encode_message`] and transmit all chunk frames to `dest`.
    pub fn send_to_addr(&self, dest: SocketAddr, msg: &[u8]) -> Result<u32> {
        let (transfer_id, frames) = self.encode_chunk_frames(msg)?;

        match &self.backend {
            UdpBackend::OsSocket(socket_slot) => {
                let slot = socket_slot
                    .lock()
                    .map_err(|_| Error::TransportUnavailable)?;
                match &*slot {
                    OsUdpSocketSlot::Std(socket) => {
                        for frame in &frames {
                            socket.send_to(frame, dest).map_err(|_| Error::SendFailed)?;
                        }
                    }
                    #[cfg(feature = "udp-transport")]
                    OsUdpSocketSlot::Tokio(socket) => {
                        for frame in &frames {
                            socket
                                .try_send_to(frame, dest)
                                .map_err(|_| Error::SendFailed)?;
                        }
                    }
                    #[cfg(feature = "udp-transport")]
                    OsUdpSocketSlot::Empty => return Err(Error::TransportUnavailable),
                }
            }
            UdpBackend::InMemory(hub) => {
                self.send_in_memory(hub, dest, frames)?;
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
            UdpBackend::OsSocket(socket_slot) => {
                let mut frame_buf = vec![0u8; self.limits.mtu + 64];
                let slot = socket_slot
                    .lock()
                    .map_err(|_| Error::TransportUnavailable)?;
                let mut transient_errors = 0usize;
                loop {
                    let recv_res = match &*slot {
                        OsUdpSocketSlot::Std(socket) => socket.recv_from(&mut frame_buf),
                        #[cfg(feature = "udp-transport")]
                        OsUdpSocketSlot::Tokio(socket) => socket.try_recv_from(&mut frame_buf),
                        #[cfg(feature = "udp-transport")]
                        OsUdpSocketSlot::Empty => return Err(Error::TransportUnavailable),
                    };
                    match recv_res {
                        Ok((len, source_addr)) => {
                            self.ingest_datagram(source_addr, &frame_buf[..len])?;
                        }
                        Err(ref err) => match self.classify_udp_recv_error(err) {
                            UdpRecvErrorDisposition::BreakWouldBlock => break,
                            UdpRecvErrorDisposition::ContinueTransient => {
                                transient_errors += 1;
                                if transient_errors >= DEFAULT_MAX_UDP_TRANSIENT_ERRORS_PER_POLL {
                                    break;
                                }
                                continue;
                            }
                            UdpRecvErrorDisposition::FatalReceiveFailed => {
                                return Err(Error::ReceiveFailed);
                            }
                        },
                    }
                }
            }
            UdpBackend::InMemory(hub) => {
                let pending: Vec<(SocketAddr, Vec<u8>)> = {
                    let mut state = hub.inner.lock().map_err(|_| Error::TransportUnavailable)?;
                    state
                        .queues
                        .get_mut(&self.local_addr)
                        .map(|q| q.drain(..).collect())
                        .unwrap_or_default()
                };
                for (source_addr, frame) in pending {
                    self.ingest_datagram(source_addr, &frame)?;
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

    /// Poll for a reassembled datagram payload and verify it under an explicit
    /// [`EnvelopeVersionPolicyV1`] (`ADR-0022` §2, `Decision A10`).
    pub fn poll_recv_verified_envelope_with_policy(
        &self,
        expected_swarm_instance_id: &[u8; 16],
        expected_round_id: u64,
        replay: &mut ReplayProtection,
        now_secs: u32,
        version_policy: EnvelopeVersionPolicyV1,
    ) -> core::result::Result<Option<(SocketAddr, PeerId, MessageEnvelope)>, VerifyError> {
        let Ok(Some((source_addr, bytes))) = self.poll_reassembled_once() else {
            return Ok(None);
        };
        let env = MessageEnvelope::deserialize_bounded(&bytes, self.limits.max_message_bytes)?;
        if env.swarm_instance_id != *expected_swarm_instance_id || env.round_id != expected_round_id
        {
            return Err(VerifyError::MalformedEnvelope);
        }
        env.verify_authenticated_with_policy(replay, now_secs, version_policy)?;
        let sender_peer = env
            .sender_peer_id()
            .map_err(|_| VerifyError::InvalidSenderKey)?;
        Ok(Some((source_addr, sender_peer, env)))
    }

    /// Poll for a reassembled datagram payload and strictly verify a `VERSION_V2` (`(0, 3)`)
    /// [`MessageEnvelope`] through [`MessageEnvelope::verify_authenticated`].
    pub fn poll_recv_verified_envelope_v2(
        &self,
        expected_swarm_instance_id: &[u8; 16],
        expected_round_id: u64,
        replay: &mut ReplayProtection,
        now_secs: u32,
    ) -> core::result::Result<Option<(SocketAddr, PeerId, MessageEnvelope)>, VerifyError> {
        self.poll_recv_verified_envelope_with_policy(
            expected_swarm_instance_id,
            expected_round_id,
            replay,
            now_secs,
            EnvelopeVersionPolicyV1::RequireV2,
        )
    }

    /// Legacy v1/v2 compatibility datagram receiver (`ADR-0022` §2).
    ///
    /// Permits `VERSION_V1` only when `replay.startup_policy()` is explicitly
    /// [`ReplayStartupPolicyV1::GenesisRosterBootstrap`]; under
    /// [`ReplayStartupPolicyV1::RequireSessionHandshake`], enforces strict
    /// [`EnvelopeVersionPolicyV1::RequireV2`].
    pub fn poll_recv_verified_envelope_v1(
        &self,
        expected_swarm_instance_id: &[u8; 16],
        expected_round_id: u64,
        replay: &mut ReplayProtection,
        now_secs: u32,
    ) -> core::result::Result<Option<(SocketAddr, PeerId, MessageEnvelope)>, VerifyError> {
        let policy = match replay.startup_policy() {
            ReplayStartupPolicyV1::GenesisRosterBootstrap => {
                EnvelopeVersionPolicyV1::AllowLegacyV1AndV2
            }
            ReplayStartupPolicyV1::RequireSessionHandshake => EnvelopeVersionPolicyV1::RequireV2,
        };
        self.poll_recv_verified_envelope_with_policy(
            expected_swarm_instance_id,
            expected_round_id,
            replay,
            now_secs,
            policy,
        )
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
            let dest = self.lookup_peer_addr(peer)?;
            let (_, frames) = self.encode_chunk_frames(msg)?;
            match &self.backend {
                UdpBackend::InMemory(hub) => self.send_in_memory(hub, dest, frames),
                UdpBackend::OsSocket(socket_slot) => {
                    #[cfg(feature = "udp-transport")]
                    {
                        self.send_os_async(socket_slot, dest, frames).await
                    }
                    #[cfg(not(feature = "udp-transport"))]
                    {
                        let _ = (socket_slot, dest, frames);
                        Err(Error::TransportUnavailable)
                    }
                }
            }
        })
    }

    fn recv(&self) -> Self::RecvFuture<'_> {
        Box::pin(async move {
            match &self.backend {
                UdpBackend::InMemory(hub) => {
                    core::future::poll_fn(|cx| self.poll_recv_in_memory(hub, cx)).await
                }
                UdpBackend::OsSocket(socket_slot) => {
                    #[cfg(feature = "udp-transport")]
                    {
                        self.recv_os_async(socket_slot).await
                    }
                    #[cfg(not(feature = "udp-transport"))]
                    {
                        let _ = socket_slot;
                        let (source_addr, payload) = self
                            .poll_reassembled_once()?
                            .ok_or(Error::TransportUnavailable)?;
                        Ok(crate::traits::ObservedTransportMessage {
                            observed_source: crate::traits::ObservedSourceRef::from(source_addr),
                            payload,
                        })
                    }
                }
            }
        })
    }

    fn broadcast<'a>(&'a self, msg: &'a [u8]) -> Self::BroadcastFuture<'a> {
        Box::pin(async move {
            let peers: Vec<PeerId> = self
                .peers
                .read()
                .map_err(|_| Error::TransportUnavailable)?
                .keys()
                .copied()
                .collect();
            let mut stats = BroadcastStats {
                peers_sent: peers.len(),
                confirmed: 0,
                failed: 0,
            };
            for peer in peers {
                match self.send(peer, msg).await {
                    Ok(()) => stats.confirmed += 1,
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
            .with_cohort_allowlist(allowlist.clone());

        // Poll and verify through the MessageEnvelope gate on receiver B without polling sleeps.
        // First confirm strict v2 rejects legacy v1 without touching replay state.
        udp_a.send(peer_b, &wire_bytes).await.unwrap();
        let mut strict_replay = ReplayProtection::try_with_config(64, 60)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(allowlist);
        assert!(matches!(
            udp_b.poll_recv_verified_envelope_v2(&swarm_id, 5, &mut strict_replay, 1_700_000_000),
            Err(VerifyError::UnsupportedVersion { major: 0, minor: 2 })
        ));
        assert_eq!(strict_replay.cache_size(), 0);
        // Under default RequireSessionHandshake, even `poll_recv_verified_envelope_v1` rejects v1.
        udp_a.send(peer_b, &wire_bytes).await.unwrap();
        let mut default_handshake_replay = ReplayProtection::try_with_config(64, 60).unwrap();
        assert!(matches!(
            udp_b.poll_recv_verified_envelope_v1(
                &swarm_id,
                5,
                &mut default_handshake_replay,
                1_700_000_000,
            ),
            Err(VerifyError::UnsupportedVersion { major: 0, minor: 2 })
        ));
        assert_eq!(default_handshake_replay.cache_size(), 0);

        let (obs_addr, verified_peer, verified_env) = udp_b
            .poll_recv_verified_envelope_v1(&swarm_id, 5, &mut replay, 1_700_000_000)
            .unwrap()
            .expect("multi-chunk UDP envelope must reassemble and verify");
        assert_eq!(obs_addr, udp_a.local_addr());
        assert_eq!(verified_peer, peer_a);
        assert_eq!(verified_env.payload, body);
    }

    #[tokio::test]
    async fn udp_in_memory_recv_suspends_across_multi_chunk_delivery() {
        use std::time::Duration;

        let peer_a = PeerId::new([0x71; 32]);
        let peer_b = PeerId::new([0x72; 32]);
        let hub = InMemoryUdpDatagramHub::new();
        let limits = ReassemblerLimits {
            mtu: 256,
            max_message_bytes: 8_192,
            ..ReassemblerLimits::default()
        };
        let udp_a = UdpGossipTransport::bind_in_memory(
            peer_a,
            "127.0.0.1:42101".parse().unwrap(),
            1,
            limits,
            hub.clone(),
        )
        .unwrap();
        let udp_b = UdpGossipTransport::bind_in_memory(
            peer_b,
            "127.0.0.1:42102".parse().unwrap(),
            1,
            limits,
            hub,
        )
        .unwrap();
        udp_a.register_peer(peer_b, udp_b.local_addr()).unwrap();

        let rx_task = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(2), udp_b.recv())
                .await
                .expect("in-memory UDP recv must wake after multi-chunk send")
                .unwrap()
        });

        tokio::task::yield_now().await;
        let payload = vec![0x5Au8; 1024];
        udp_a.send(peer_b, &payload).await.unwrap();
        let observed = rx_task.await.unwrap();
        assert_eq!(observed.payload, payload);
    }

    #[cfg(feature = "udp-transport")]
    #[tokio::test]
    async fn udp_os_socket_concurrent_recv_before_send_reassembles_chunks() {
        use std::time::Duration;

        let peer_a = PeerId::new([0x81; 32]);
        let peer_b = PeerId::new([0x82; 32]);
        let limits = ReassemblerLimits {
            mtu: 256,
            max_message_bytes: 8_192,
            ..ReassemblerLimits::default()
        };
        let Ok(udp_a) = UdpGossipTransport::bind(peer_a, "127.0.0.1:0".parse().unwrap(), 1, limits)
        else {
            return;
        };
        let Ok(udp_b) = UdpGossipTransport::bind(peer_b, "127.0.0.1:0".parse().unwrap(), 1, limits)
        else {
            return;
        };
        udp_a.register_peer(peer_b, udp_b.local_addr()).unwrap();

        let rx_task = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(2), udp_b.recv())
                .await
                .expect("OS UDP recv must complete asynchronously via reactor")
                .unwrap()
        });

        tokio::task::yield_now().await;
        let payload = vec![0x6Bu8; 1024];
        udp_a.send(peer_b, &payload).await.unwrap();
        let observed = rx_task.await.unwrap();
        assert_eq!(observed.payload, payload);
    }

    #[test]
    fn udp_recv_error_classifier_isolates_wsa_connreset_and_rejects_fatal() {
        let limits = ReassemblerLimits::default();
        let hub = InMemoryUdpDatagramHub::new();
        let udp = UdpGossipTransport::bind_in_memory(
            PeerId::new([0x90; 32]),
            "127.0.0.1:42900".parse().unwrap(),
            1,
            limits,
            hub,
        )
        .unwrap();

        assert_eq!(
            udp.classify_udp_recv_error(&io::Error::new(
                io::ErrorKind::ConnectionReset,
                "WSAECONNRESET"
            )),
            UdpRecvErrorDisposition::ContinueTransient
        );
        assert_eq!(
            udp.classify_udp_recv_error(&io::Error::new(io::ErrorKind::Interrupted, "EINTR")),
            UdpRecvErrorDisposition::ContinueTransient
        );
        assert_eq!(
            udp.classify_udp_recv_error(&io::Error::new(io::ErrorKind::WouldBlock, "EWOULDBLOCK")),
            UdpRecvErrorDisposition::BreakWouldBlock
        );
        assert_eq!(
            udp.classify_udp_recv_error(&io::Error::new(io::ErrorKind::InvalidInput, "EINVAL")),
            UdpRecvErrorDisposition::FatalReceiveFailed
        );

        let diag = udp.diagnostics();
        assert_eq!(diag.icmp_port_unreachable_resets, 2);
        assert_eq!(diag.dropped_connection_errors, 1);
        assert!(udp
            .drain_audit_events()
            .iter()
            .any(|e| e.name == "swarmtorch.net.udp.icmp_reset_ignored"));
    }

    #[tokio::test]
    async fn udp_malformed_and_conflicting_datagrams_isolated_and_audited() {
        let peer_a = PeerId::new([0x91; 32]);
        let peer_rogue = PeerId::new([0x92; 32]);
        let peer_b = PeerId::new([0x93; 32]);
        let hub = InMemoryUdpDatagramHub::new();
        let limits = ReassemblerLimits {
            mtu: 64,
            max_message_bytes: 1024,
            ..ReassemblerLimits::default()
        };
        let addr_a: SocketAddr = "127.0.0.1:42911".parse().unwrap();
        let addr_rogue: SocketAddr = "127.0.0.1:42912".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:42913".parse().unwrap();

        let udp_a =
            UdpGossipTransport::bind_in_memory(peer_a, addr_a, 1, limits, hub.clone()).unwrap();
        let udp_b =
            UdpGossipTransport::bind_in_memory(peer_b, addr_b, 1, limits, hub.clone()).unwrap();
        udp_a.register_peer(peer_b, addr_b).unwrap();

        // Inject malformed datagram + conflicting transfer_id=77 chunks from rogue into addr_b queue.
        // Non-last chunks must match exact stride P = mtu - 16 = 48 bytes.
        let mut conflict_a = [0u8; 64];
        let ca_len =
            ChunkedFrameCodec::encode_frame_into(77, 0, 2, &[0x11u8; 48], &mut conflict_a).unwrap();
        let mut conflict_b = [0u8; 64];
        let cb_len =
            ChunkedFrameCodec::encode_frame_into(77, 1, 3, &[0x22u8; 48], &mut conflict_b).unwrap();
        {
            let mut state = hub.inner.lock().unwrap();
            let q = state.queues.get_mut(&addr_b).unwrap();
            q.push_back((addr_rogue, b"NOT_ST_HEADER".to_vec()));
            q.push_back((addr_rogue, conflict_a[..ca_len].to_vec()));
            q.push_back((addr_rogue, conflict_b[..cb_len].to_vec()));
        }
        let _ = peer_rogue;

        // Honest peer sends valid multi-chunk message into the same receiver!
        let honest_payload = vec![0x7Cu8; 180];
        udp_a.send(peer_b, &honest_payload).await.unwrap();

        let delivered = udp_b
            .recv()
            .await
            .expect("malformed/quarantined datagrams must isolate and deliver honest message");
        assert_eq!(delivered.payload, honest_payload);

        let diag = udp_b.diagnostics();
        assert!(diag.dropped_malformed_frames >= 1);
        assert!(diag.quarantined_udp_events >= 1);
        let events = udp_b.drain_audit_events();
        assert!(events
            .iter()
            .any(|e| e.name == "swarmtorch.net.udp.datagram_rejected"));
        assert!(events
            .iter()
            .any(|e| e.name == "swarmtorch.net.udp.source_quarantined"));
    }

    #[cfg(feature = "udp-transport")]
    #[tokio::test]
    async fn udp_os_socket_recovers_after_closed_peer_icmp_reset_and_garbage_datagram() {
        use std::time::Duration;

        let peer_a = PeerId::new([0x94; 32]);
        let peer_b = PeerId::new([0x95; 32]);
        let limits = ReassemblerLimits {
            mtu: 256,
            max_message_bytes: 8_192,
            ..ReassemblerLimits::default()
        };
        let Ok(udp_a) = UdpGossipTransport::bind(peer_a, "127.0.0.1:0".parse().unwrap(), 1, limits)
        else {
            return;
        };
        let Ok(udp_b) = UdpGossipTransport::bind(peer_b, "127.0.0.1:0".parse().unwrap(), 1, limits)
        else {
            return;
        };

        // Trigger ICMP Port Unreachable (`WSAECONNRESET` on Windows) from udp_b -> closed socket.
        if let Ok(closed_sock) = UdpSocket::bind("127.0.0.1:0") {
            let dead_addr = closed_sock.local_addr().unwrap();
            drop(closed_sock);
            let _ = udp_b.send_to_addr(dead_addr, b"probe-closed-port");
        }
        // Also send raw garbage datagram directly at udp_b OS socket.
        if let Ok(raw) = UdpSocket::bind("127.0.0.1:0") {
            let _ = raw.send_to(b"CORRUPT_UDP_FRAME", udp_b.local_addr());
        }

        udp_a.register_peer(peer_b, udp_b.local_addr()).unwrap();
        let rx_task = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(2), udp_b.recv())
                .await
                .expect("OS UDP receiver must survive WSAECONNRESET and garbage frames")
                .unwrap()
        });

        tokio::task::yield_now().await;
        let payload = vec![0x55u8; 640];
        udp_a.send(peer_b, &payload).await.unwrap();
        let observed = rx_task.await.unwrap();
        assert_eq!(observed.payload, payload);
    }

    #[test]
    fn udp_recv_error_classifier_covers_icmp_variants_and_wsa_emsgsize() {
        let limits = ReassemblerLimits::default();
        let hub = InMemoryUdpDatagramHub::new();
        let udp = UdpGossipTransport::bind_in_memory(
            PeerId::new([0x96; 32]),
            "127.0.0.1:42950".parse().unwrap(),
            1,
            limits,
            hub,
        )
        .unwrap();

        assert_eq!(
            udp.classify_udp_recv_error(&io::Error::new(
                io::ErrorKind::ConnectionRefused,
                "ECONNREFUSED"
            )),
            UdpRecvErrorDisposition::ContinueTransient
        );
        assert_eq!(
            udp.classify_udp_recv_error(&io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "ECONNABORTED"
            )),
            UdpRecvErrorDisposition::ContinueTransient
        );
        assert_eq!(
            udp.classify_udp_recv_error(&io::Error::from_raw_os_error(10040)),
            UdpRecvErrorDisposition::ContinueTransient
        );
        let diag = udp.diagnostics();
        assert_eq!(diag.icmp_port_unreachable_resets, 2);
        assert_eq!(diag.dropped_malformed_frames, 1);
    }

    #[test]
    fn udp_poll_reassembled_once_isolates_malformed_conflicts_and_icmp_resets() {
        let peer_a = PeerId::new([0x97; 32]);
        let peer_b = PeerId::new([0x98; 32]);
        let limits = ReassemblerLimits {
            mtu: 128,
            max_message_bytes: 4096,
            ..ReassemblerLimits::default()
        };
        let Ok(udp_a) = UdpGossipTransport::bind(peer_a, "127.0.0.1:0".parse().unwrap(), 1, limits)
        else {
            return;
        };
        let Ok(udp_b) = UdpGossipTransport::bind(peer_b, "127.0.0.1:0".parse().unwrap(), 1, limits)
        else {
            return;
        };

        // Probe dead port to trigger OS ICMP Port Unreachable (`WSAECONNRESET` on Windows),
        // then send an oversized (> mtu + 64) raw datagram + garbage frame to udp_b.
        if let Ok(closed_sock) = UdpSocket::bind("127.0.0.1:0") {
            let dead = closed_sock.local_addr().unwrap();
            drop(closed_sock);
            let _ = udp_b.send_to_addr(dead, b"probe-dead");
        }
        if let Ok(raw) = UdpSocket::bind("127.0.0.1:0") {
            let _ = raw.send_to(&[0xAAu8; 512], udp_b.local_addr());
            let _ = raw.send_to(b"BAD_MAGIC", udp_b.local_addr());
        }

        let payload = vec![0x3Cu8; 300];
        let _ = udp_a.send_to_addr(udp_b.local_addr(), &payload).unwrap();

        let mut delivered = None;
        for _ in 0..30 {
            match udp_b.poll_reassembled_once() {
                Ok(Some((_src, bytes))) => {
                    delivered = Some(bytes);
                    break;
                }
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(5)),
                Err(err) => {
                    panic!("poll_reassembled_once must isolate transient/malformed: {err:?}")
                }
            }
        }
        assert_eq!(delivered, Some(payload));
        assert!(udp_b.diagnostics().dropped_malformed_frames >= 1);
    }

    #[tokio::test]
    async fn udp_poll_recv_verified_envelope_v2_accepts_v2_and_v1_rejects_v0_even_under_genesis() {
        use swarm_torch_core::crypto::SessionHandshakeChallengeV2;

        let kp_a = KeyPair::from_seed([0xA1; 32]).unwrap();
        let kp_b = KeyPair::from_seed([0xA2; 32]).unwrap();
        let peer_a = kp_a.peer_id();
        let peer_b = kp_b.peer_id();
        let auth_a = MessageAuth::new(kp_a.clone());

        let limits = ReassemblerLimits {
            mtu: 256,
            max_message_bytes: 8_192,
            ..ReassemblerLimits::default()
        };
        let hub = InMemoryUdpDatagramHub::new();
        let addr_a: SocketAddr = "127.0.0.1:42961".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:42962".parse().unwrap();
        let udp_a =
            UdpGossipTransport::bind_in_memory(peer_a, addr_a, 1, limits, hub.clone()).unwrap();
        let udp_b = UdpGossipTransport::bind_in_memory(peer_b, addr_b, 1, limits, hub).unwrap();
        udp_a.register_peer(peer_b, addr_b).unwrap();

        let swarm_id = [0xCD; 16];
        let run_id = [0xEF; 16];
        let now = 1_700_000_000u32;

        // 1. Even under GenesisRosterBootstrap, poll_recv_verified_envelope_v1 must reject v0!
        let mut genesis_replay = ReplayProtection::try_with_config(64, 60)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(CohortAllowlist::from_peers([peer_a]));
        let env_v0 =
            MessageEnvelope::new_legacy_v0(*kp_a.public_key(), MessageType::Heartbeat, vec![1, 2])
                .with_sequence(1)
                .with_timestamp(now)
                .sign_with(&auth_a);
        udp_a
            .send(peer_b, &env_v0.serialize().unwrap())
            .await
            .unwrap();
        assert_eq!(
            udp_b
                .poll_recv_verified_envelope_v1(&[0u8; 16], 0, &mut genesis_replay, now)
                .unwrap_err(),
            VerifyError::UnsupportedVersion { major: 0, minor: 1 }
        );
        assert_eq!(genesis_replay.cache_size(), 0);

        // 2. Complete v2 session handshake and verify poll_recv_verified_envelope_v2 admits v2!
        let mut replay_v2 = ReplayProtection::try_with_config(64, 60)
            .unwrap()
            .with_cohort_allowlist(CohortAllowlist::from_peers([peer_a]));
        let challenge = SessionHandshakeChallengeV2 {
            swarm_instance_id: swarm_id,
            run_id,
            policy_digest: [0x11; 32],
            receiver_peer: peer_b,
            sender_peer: peer_a,
            requested_epoch: 1,
            initial_sequence: 0,
            receiver_nonce: [0x22; 32],
            issued_at_secs: now,
        };
        let session_id = challenge.derive_session_id().unwrap();
        replay_v2.register_recovery_challenge_v2(challenge).unwrap();
        let hs_sig = auth_a.sign_session_handshake_v2(&challenge).unwrap();
        replay_v2
            .complete_recovery_handshake_v2(swarm_id, run_id, kp_a.public_key(), &hs_sig, now)
            .unwrap();

        let env_v2 = MessageEnvelope::new_v2(
            *kp_a.public_key(),
            swarm_id,
            run_id,
            1,
            session_id,
            7,
            MessageType::GradientUpdate,
            vec![9, 8, 7, 6],
        )
        .with_sequence(1)
        .with_timestamp(now)
        .sign_with(&auth_a);
        udp_a
            .send(peer_b, &env_v2.serialize().unwrap())
            .await
            .unwrap();
        let (obs_addr, verified_peer, accepted) = udp_b
            .poll_recv_verified_envelope_v2(&swarm_id, 7, &mut replay_v2, now)
            .unwrap()
            .expect("v2 envelope must admit cleanly under poll_recv_verified_envelope_v2");
        assert_eq!(obs_addr, addr_a);
        assert_eq!(verified_peer, peer_a);
        assert_eq!(accepted.payload, vec![9, 8, 7, 6]);
    }

    #[test]
    fn udp_quarantined_source_flood_does_not_evict_critical_audit_records() {
        let peer_b = PeerId::new([0xA3; 32]);
        let hub = InMemoryUdpDatagramHub::new();
        let limits = ReassemblerLimits {
            mtu: 64,
            max_message_bytes: 512,
            max_quarantines_per_source: 2,
            ..ReassemblerLimits::default()
        };
        let addr_b: SocketAddr = "127.0.0.1:42970".parse().unwrap();
        let addr_rogue: SocketAddr = "127.0.0.1:42971".parse().unwrap();
        let udp_b =
            UdpGossipTransport::bind_in_memory(peer_b, addr_b, 1, limits, hub.clone()).unwrap();

        // Record an earlier critical transport error before quarantined-peer flood.
        assert_eq!(
            udp_b.classify_udp_recv_error(&io::Error::new(io::ErrorKind::InvalidInput, "fatal")),
            UdpRecvErrorDisposition::FatalReceiveFailed
        );

        // Escalate `addr_rogue` into SourceQuarantined (`max_quarantines_per_source = 2`).
        for tid in 1..=2u32 {
            let mut f1 = [0u8; 64];
            let l1 = ChunkedFrameCodec::encode_frame_into(tid, 0, 2, &[0x11; 48], &mut f1).unwrap();
            let mut f2 = [0u8; 64];
            let l2 = ChunkedFrameCodec::encode_frame_into(tid, 0, 3, &[0x22; 48], &mut f2).unwrap();
            udp_b.ingest_datagram(addr_rogue, &f1[..l1]).unwrap();
            udp_b.ingest_datagram(addr_rogue, &f2[..l2]).unwrap();
        }

        // Now flood > DEFAULT_MAX_TRANSPORT_AUDIT_EVENTS junk packets from already-quarantined rogue.
        // Repeat SourceQuarantined rejections must be rate-limited rather than evicting earlier Critical events!
        for _ in 0..300 {
            udp_b
                .ingest_datagram(addr_rogue, b"quarantined-spam")
                .unwrap();
        }

        let events = udp_b.drain_audit_events();
        assert!(
            events
                .iter()
                .any(|e| e.name == "swarmtorch.net.udp.recv_error"),
            "already-quarantined UDP source flood must not evict prior Critical audit records"
        );
    }

    #[tokio::test]
    async fn udp_poll_recv_verified_envelope_v2_rejects_v0_even_with_zero_scope_and_round() {
        let kp_a = KeyPair::from_seed([0xA4; 32]).unwrap();
        let kp_b = KeyPair::from_seed([0xA5; 32]).unwrap();
        let auth_a = MessageAuth::new(kp_a.clone());

        let limits = ReassemblerLimits {
            mtu: 256,
            max_message_bytes: 8_192,
            ..ReassemblerLimits::default()
        };
        let hub = InMemoryUdpDatagramHub::new();
        let addr_a: SocketAddr = "127.0.0.1:42981".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:42982".parse().unwrap();
        let udp_a =
            UdpGossipTransport::bind_in_memory(kp_a.peer_id(), addr_a, 1, limits, hub.clone())
                .unwrap();
        let udp_b =
            UdpGossipTransport::bind_in_memory(kp_b.peer_id(), addr_b, 1, limits, hub).unwrap();
        udp_a.register_peer(kp_b.peer_id(), addr_b).unwrap();

        let now = 1_700_000_000u32;
        let mut replay = ReplayProtection::try_with_config(64, 60)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(CohortAllowlist::from_peers([kp_a.peer_id()]));

        let env_v0 =
            MessageEnvelope::new_legacy_v0(*kp_a.public_key(), MessageType::Heartbeat, vec![4, 2])
                .with_sequence(1)
                .with_timestamp(now)
                .sign_with(&auth_a);
        udp_a
            .send(kp_b.peer_id(), &env_v0.serialize().unwrap())
            .await
            .unwrap();

        assert_eq!(
            udp_b
                .poll_recv_verified_envelope_v2(&[0u8; 16], 0, &mut replay, now)
                .unwrap_err(),
            VerifyError::UnsupportedVersion { major: 0, minor: 1 }
        );
        assert_eq!(replay.cache_size(), 0);
    }
}
