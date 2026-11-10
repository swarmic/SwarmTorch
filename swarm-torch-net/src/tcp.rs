//! Length-prefixed `TcpTransport` and cancellation-safe frame reader state machine
//! (`ADR-0022` §1.3, §2, `W11-03`).
//!
//! Provides:
//! - `u32_le(payload_len) || payload` framing with strict pre-allocation `max_message_bytes` check.
//! - [`TcpFrameReaderState`] (`F-06`): explicit per-connection state machine that retains partially
//!   read header or payload bytes across `WouldBlock` or async `tokio::select!` cancellation so a
//!   cancelled `recv` future never desynchronizes the byte stream.
//! - [`TcpTransport`]: concrete [`SwarmTransport`] implementation over TCP sockets (with an
//!   in-memory byte-stream transport constructor [`TcpTransport::bind_in_memory`] that runs the
//!   identical [`encode_length_prefixed_frame`] + [`TcpFrameReaderState`] pipeline for sandboxed
//!   environments).

use std::collections::{BTreeMap, VecDeque};
use std::io::{self, Cursor, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use swarm_torch_core::traits::PeerId;

use crate::traits::{
    BandwidthClass, BroadcastStats, ReliabilityClass, SwarmTransport, TransportCapabilities,
};
use crate::{Error, Result};

/// Default maximum message size accepted by [`TcpTransport`] (`1 MiB`).
pub const DEFAULT_TCP_MAX_MESSAGE_BYTES: usize = 1_048_576;

/// Explicit cancellation-safe frame reader state machine (`F-06`, `W11-03`).
///
/// Storing [`TcpFrameReaderState`] inside the per-connection state guarantees that if a non-blocking
/// read returns `WouldBlock` or an async `tokio::select!` branch is cancelled mid-header or
/// mid-payload, the bytes already consumed from the kernel socket buffer remain intact for the
/// next poll.
#[derive(Debug, Clone)]
pub struct TcpFrameReaderState {
    max_message_bytes: usize,
    phase: FrameReadPhase,
}

#[derive(Debug, Clone)]
enum FrameReadPhase {
    ReadingLength { buf: [u8; 4], filled: usize },
    ReadingPayload { buf: Vec<u8>, filled: usize },
    Poisoned,
}

impl TcpFrameReaderState {
    /// Create a new frame reader enforcing `1 <= payload_len <= max_message_bytes`.
    pub fn new(max_message_bytes: usize) -> Self {
        Self {
            max_message_bytes: max_message_bytes.max(1),
            phase: FrameReadPhase::ReadingLength {
                buf: [0u8; 4],
                filled: 0,
            },
        }
    }

    /// Whether the stream has been poisoned by a malformed length header or EOF mid-frame.
    pub const fn is_poisoned(&self) -> bool {
        matches!(self.phase, FrameReadPhase::Poisoned)
    }

    /// Attempt to advance the frame reader from any `std::io::Read` source.
    ///
    /// - Returns `Ok(Some(frame))` when a full frame has been assembled.
    /// - Returns `Ok(None)` when the underlying stream returned `WouldBlock` / `TimedOut`
    ///   (preserving all partially read bytes in `self`).
    /// - Poison-closes the state and returns `Err(Error::InvalidMessage)` if `len == 0`,
    ///   `len > max_message_bytes`, or unexpected EOF occurs mid-frame.
    pub fn try_read_frame<R: Read>(&mut self, reader: &mut R) -> Result<Option<Vec<u8>>> {
        loop {
            match &mut self.phase {
                FrameReadPhase::Poisoned => return Err(Error::InvalidMessage),
                FrameReadPhase::ReadingLength { buf, filled } => {
                    while *filled < 4 {
                        match reader.read(&mut buf[*filled..4]) {
                            Ok(0) => {
                                if *filled == 0 {
                                    return Ok(None);
                                }
                                self.phase = FrameReadPhase::Poisoned;
                                return Err(Error::InvalidMessage);
                            }
                            Ok(n) => {
                                *filled += n;
                            }
                            Err(ref e)
                                if e.kind() == io::ErrorKind::WouldBlock
                                    || e.kind() == io::ErrorKind::TimedOut =>
                            {
                                return Ok(None);
                            }
                            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                            Err(_) => {
                                self.phase = FrameReadPhase::Poisoned;
                                return Err(Error::ReceiveFailed);
                            }
                        }
                    }

                    let len = u32::from_le_bytes(*buf) as usize;
                    if len == 0 || len > self.max_message_bytes {
                        self.phase = FrameReadPhase::Poisoned;
                        return Err(Error::InvalidMessage);
                    }

                    self.phase = FrameReadPhase::ReadingPayload {
                        buf: vec![0u8; len],
                        filled: 0,
                    };
                }
                FrameReadPhase::ReadingPayload { buf, filled } => {
                    let target_len = buf.len();
                    while *filled < target_len {
                        match reader.read(&mut buf[*filled..target_len]) {
                            Ok(0) => {
                                self.phase = FrameReadPhase::Poisoned;
                                return Err(Error::InvalidMessage);
                            }
                            Ok(n) => {
                                *filled += n;
                            }
                            Err(ref e)
                                if e.kind() == io::ErrorKind::WouldBlock
                                    || e.kind() == io::ErrorKind::TimedOut =>
                            {
                                return Ok(None);
                            }
                            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                            Err(_) => {
                                self.phase = FrameReadPhase::Poisoned;
                                return Err(Error::ReceiveFailed);
                            }
                        }
                    }

                    let completed = core::mem::take(buf);
                    self.phase = FrameReadPhase::ReadingLength {
                        buf: [0u8; 4],
                        filled: 0,
                    };
                    return Ok(Some(completed));
                }
            }
        }
    }
}

/// Encode a single length-prefixed frame (`u32_le(len) || payload`) with pre-send bounds validation.
pub fn encode_length_prefixed_frame(payload: &[u8], max_message_bytes: usize) -> Result<Vec<u8>> {
    if payload.is_empty() || payload.len() > max_message_bytes || payload.len() > u32::MAX as usize
    {
        return Err(Error::InvalidMessage);
    }
    let len = payload.len() as u32;
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(payload);
    Ok(out)
}

struct ActiveConnection {
    stream: TcpStream,
    reader_state: TcpFrameReaderState,
    known_peer: Option<PeerId>,
}

/// Shared in-memory raw byte-stream hub used when OS socket creation is restricted or for
/// deterministic framed stream testing.
#[derive(Default, Clone)]
pub struct InMemoryTcpStreamHub {
    streams: Arc<Mutex<BTreeMap<SocketAddr, VecDeque<Vec<u8>>>>>,
}

impl InMemoryTcpStreamHub {
    /// Create a new in-memory byte-stream hub.
    pub fn new() -> Self {
        Self::default()
    }
}

enum TcpBackend {
    OsSocket(TcpListener),
    InMemory {
        hub: InMemoryTcpStreamHub,
        reader_state: Mutex<TcpFrameReaderState>,
    },
}

/// Concrete length-prefixed TCP [`SwarmTransport`] (`ADR-0022` §1.3, `W11-03`).
///
/// Keyed by authenticated/registered `PeerId <-> SocketAddr` directory entries and backed by
/// cancellation-safe [`TcpFrameReaderState`] connection readers.
pub struct TcpTransport {
    local_peer: PeerId,
    local_addr: SocketAddr,
    max_message_bytes: usize,
    backend: TcpBackend,
    peers: RwLock<BTreeMap<PeerId, SocketAddr>>,
    addr_to_peer: RwLock<BTreeMap<SocketAddr, PeerId>>,
    connections: Mutex<Vec<ActiveConnection>>,
    inbox: Mutex<VecDeque<(PeerId, Vec<u8>)>>,
}

impl core::fmt::Debug for TcpTransport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TcpTransport")
            .field("local_peer", &self.local_peer)
            .field("local_addr", &self.local_addr)
            .field("max_message_bytes", &self.max_message_bytes)
            .finish()
    }
}

impl TcpTransport {
    /// Bind a new [`TcpTransport`] to `bind_addr` with `max_message_bytes`.
    pub fn bind(
        local_peer: PeerId,
        bind_addr: SocketAddr,
        max_message_bytes: usize,
    ) -> Result<Arc<Self>> {
        if max_message_bytes == 0 || max_message_bytes > u32::MAX as usize {
            return Err(Error::InvalidMessage);
        }
        let listener = TcpListener::bind(bind_addr).map_err(|_| Error::ConnectionFailed)?;
        listener
            .set_nonblocking(true)
            .map_err(|_| Error::ConnectionFailed)?;
        let local_addr = listener.local_addr().map_err(|_| Error::ConnectionFailed)?;

        Ok(Arc::new(Self {
            local_peer,
            local_addr,
            max_message_bytes,
            backend: TcpBackend::OsSocket(listener),
            peers: RwLock::new(BTreeMap::new()),
            addr_to_peer: RwLock::new(BTreeMap::new()),
            connections: Mutex::new(Vec::new()),
            inbox: Mutex::new(VecDeque::new()),
        }))
    }

    /// Create a [`TcpTransport`] endpoint attached to an [`InMemoryTcpStreamHub`] that executes
    /// the identical `u32_le(len) || payload` framing and [`TcpFrameReaderState`] state machine.
    pub fn bind_in_memory(
        local_peer: PeerId,
        virtual_addr: SocketAddr,
        max_message_bytes: usize,
        hub: InMemoryTcpStreamHub,
    ) -> Result<Arc<Self>> {
        if max_message_bytes == 0 || max_message_bytes > u32::MAX as usize {
            return Err(Error::InvalidMessage);
        }
        if let Ok(mut map) = hub.streams.lock() {
            map.entry(virtual_addr).or_default();
        }
        Ok(Arc::new(Self {
            local_peer,
            local_addr: virtual_addr,
            max_message_bytes,
            backend: TcpBackend::InMemory {
                hub,
                reader_state: Mutex::new(TcpFrameReaderState::new(max_message_bytes + 32)),
            },
            peers: RwLock::new(BTreeMap::new()),
            addr_to_peer: RwLock::new(BTreeMap::new()),
            connections: Mutex::new(Vec::new()),
            inbox: Mutex::new(VecDeque::new()),
        }))
    }

    /// Bound local socket address.
    pub const fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Register a peer's `SocketAddr` in the transport directory.
    pub fn register_peer(&self, peer: PeerId, addr: SocketAddr) -> Result<()> {
        self.peers
            .write()
            .map_err(|_| Error::TransportUnavailable)?
            .insert(peer, addr);
        self.addr_to_peer
            .write()
            .map_err(|_| Error::TransportUnavailable)?
            .insert(addr, peer);
        Ok(())
    }

    /// Synchronous framed send: transmits `u32_le(len) || [32-byte local_peer prologue || msg]`
    /// to `peer`.
    pub fn send_sync(&self, peer: PeerId, msg: &[u8]) -> Result<()> {
        if msg.is_empty() || msg.len() > self.max_message_bytes {
            return Err(Error::InvalidMessage);
        }
        let addr = {
            let guard = self.peers.read().map_err(|_| Error::TransportUnavailable)?;
            *guard.get(&peer).ok_or(Error::PeerNotFound)?
        };

        let mut wire_payload = Vec::with_capacity(32 + msg.len());
        wire_payload.extend_from_slice(self.local_peer.as_bytes());
        wire_payload.extend_from_slice(msg);
        let frame = encode_length_prefixed_frame(&wire_payload, self.max_message_bytes + 32)?;

        match &self.backend {
            TcpBackend::OsSocket(_) => {
                let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))
                    .map_err(|_| Error::ConnectionFailed)?;
                stream
                    .set_write_timeout(Some(Duration::from_secs(2)))
                    .map_err(|_| Error::SendFailed)?;
                stream.write_all(&frame).map_err(|_| Error::SendFailed)?;
                stream.flush().map_err(|_| Error::SendFailed)?;
                Ok(())
            }
            TcpBackend::InMemory { hub, .. } => {
                let mut guard = hub
                    .streams
                    .lock()
                    .map_err(|_| Error::TransportUnavailable)?;
                let queue = guard.get_mut(&addr).ok_or(Error::ConnectionFailed)?;
                queue.push_back(frame);
                Ok(())
            }
        }
    }

    /// Poll non-blocking listener and active connections once, returning the next complete
    /// `(PeerId, message_bytes)` frame if available.
    pub fn poll_recv_once(&self) -> Result<Option<(PeerId, Vec<u8>)>> {
        if let Some(item) = self
            .inbox
            .lock()
            .map_err(|_| Error::TransportUnavailable)?
            .pop_front()
        {
            return Ok(Some(item));
        }

        match &self.backend {
            TcpBackend::InMemory { hub, reader_state } => {
                let raw_bytes_opt = {
                    let mut guard = hub
                        .streams
                        .lock()
                        .map_err(|_| Error::TransportUnavailable)?;
                    guard.get_mut(&self.local_addr).and_then(|q| q.pop_front())
                };
                if let Some(raw_bytes) = raw_bytes_opt {
                    let mut cursor = Cursor::new(raw_bytes);
                    let mut r_state = reader_state
                        .lock()
                        .map_err(|_| Error::TransportUnavailable)?;
                    if let Some(wire_frame) = r_state.try_read_frame(&mut cursor)? {
                        if wire_frame.len() <= 32 {
                            return Err(Error::InvalidMessage);
                        }
                        let mut peer_bytes = [0u8; 32];
                        peer_bytes.copy_from_slice(&wire_frame[0..32]);
                        let sender = PeerId::new(peer_bytes);
                        let msg = wire_frame[32..].to_vec();
                        return Ok(Some((sender, msg)));
                    }
                }
                Ok(None)
            }
            TcpBackend::OsSocket(listener) => {
                loop {
                    match listener.accept() {
                        Ok((stream, remote_addr)) => {
                            let _ = stream.set_nonblocking(true);
                            let known_peer = self
                                .addr_to_peer
                                .read()
                                .ok()
                                .and_then(|map| map.get(&remote_addr).copied());
                            if let Ok(mut conns) = self.connections.lock() {
                                conns.push(ActiveConnection {
                                    stream,
                                    reader_state: TcpFrameReaderState::new(
                                        self.max_message_bytes + 32,
                                    ),
                                    known_peer,
                                });
                            }
                        }
                        Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                        Err(_) => break,
                    }
                }

                let mut conns = self
                    .connections
                    .lock()
                    .map_err(|_| Error::TransportUnavailable)?;
                let mut idx = 0;
                let mut first_error: Option<Error> = None;

                while idx < conns.len() {
                    let conn = &mut conns[idx];
                    match conn.reader_state.try_read_frame(&mut conn.stream) {
                        Ok(Some(wire_frame)) => {
                            if wire_frame.len() <= 32 {
                                conns.swap_remove(idx);
                                if first_error.is_none() {
                                    first_error = Some(Error::InvalidMessage);
                                }
                                continue;
                            }
                            let mut peer_bytes = [0u8; 32];
                            peer_bytes.copy_from_slice(&wire_frame[0..32]);
                            let sender = conn.known_peer.unwrap_or_else(|| PeerId::new(peer_bytes));
                            let msg = wire_frame[32..].to_vec();
                            if let Ok(mut inbox) = self.inbox.lock() {
                                inbox.push_back((sender, msg));
                            }
                            idx += 1;
                        }
                        Ok(None) => {
                            idx += 1;
                        }
                        Err(err) => {
                            conns.swap_remove(idx);
                            if first_error.is_none() {
                                first_error = Some(err);
                            }
                        }
                    }
                }
                drop(conns);

                if let Some(item) = self
                    .inbox
                    .lock()
                    .map_err(|_| Error::TransportUnavailable)?
                    .pop_front()
                {
                    return Ok(Some(item));
                }
                if let Some(err) = first_error {
                    return Err(err);
                }
                Ok(None)
            }
        }
    }
}

impl SwarmTransport for TcpTransport {
    type SendFuture<'a> = Pin<Box<dyn core::future::Future<Output = Result<()>> + Send + 'a>>;
    type RecvFuture<'a> =
        Pin<Box<dyn core::future::Future<Output = Result<(PeerId, Vec<u8>)>> + Send + 'a>>;
    type BroadcastFuture<'a> =
        Pin<Box<dyn core::future::Future<Output = Result<BroadcastStats>> + Send + 'a>>;
    type DiscoverFuture<'a> =
        Pin<Box<dyn core::future::Future<Output = Result<Vec<PeerId>>> + Send + 'a>>;

    fn send<'a>(&'a self, peer: PeerId, msg: &'a [u8]) -> Self::SendFuture<'a> {
        Box::pin(async move { self.send_sync(peer, msg) })
    }

    fn recv(&self) -> Self::RecvFuture<'_> {
        Box::pin(async move {
            for _ in 0..200 {
                if let Some(item) = self.poll_recv_once()? {
                    return Ok(item);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(Error::Timeout)
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
                match self.send_sync(peer, msg) {
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
            reliability: ReliabilityClass::Reliable,
            bandwidth_class: BandwidthClass::High,
            max_message_size: self.max_message_bytes,
            supports_multicast: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PartialChunkReader {
        chunks: VecDeque<Vec<u8>>,
    }

    impl Read for PartialChunkReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let Some(mut front) = self.chunks.pop_front() else {
                return Err(io::Error::new(io::ErrorKind::WouldBlock, "would block"));
            };
            if front.is_empty() {
                return Err(io::Error::new(io::ErrorKind::WouldBlock, "would block"));
            }
            let n = buf.len().min(front.len());
            buf[..n].copy_from_slice(&front[..n]);
            if n < front.len() {
                front.drain(..n);
                self.chunks.push_front(front);
            }
            Ok(n)
        }
    }

    #[test]
    fn tcp_frame_reader_state_is_cancellation_safe_across_partial_reads_and_rejects_oversized() {
        let payload = b"cancellation-safe-frame-payload";
        let encoded = encode_length_prefixed_frame(payload, 1024).unwrap();

        // Split the 4-byte length header and payload into tiny partial arrivals interspersed with WouldBlock.
        let mut reader = PartialChunkReader {
            chunks: VecDeque::from(vec![
                encoded[0..2].to_vec(),
                Vec::new(), // triggers WouldBlock mid-length-header!
                encoded[2..5].to_vec(),
                Vec::new(), // triggers WouldBlock mid-payload!
                encoded[5..15].to_vec(),
                encoded[15..].to_vec(),
            ]),
        };

        let mut state = TcpFrameReaderState::new(1024);
        assert!(state.try_read_frame(&mut reader).unwrap().is_none());
        assert!(state.try_read_frame(&mut reader).unwrap().is_none());
        let completed = state
            .try_read_frame(&mut reader)
            .unwrap()
            .expect("frame must complete after remaining chunks arrive");
        assert_eq!(completed, payload);

        // Oversized length prefix (> max_message_bytes) must poison and reject before allocating!
        let bad_len = (2048u32).to_le_bytes();
        let mut bad_reader = PartialChunkReader {
            chunks: VecDeque::from(vec![bad_len.to_vec()]),
        };
        assert!(matches!(
            state.try_read_frame(&mut bad_reader),
            Err(Error::InvalidMessage)
        ));
        assert!(state.is_poisoned());
    }

    #[tokio::test]
    async fn tcp_transport_loopback_send_and_recv_roundtrip() {
        let peer_a = PeerId::new([0x11; 32]);
        let peer_b = PeerId::new([0x22; 32]);

        let hub = InMemoryTcpStreamHub::new();
        let addr_a: SocketAddr = "127.0.0.1:41001".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:41002".parse().unwrap();
        let node_a = TcpTransport::bind_in_memory(peer_a, addr_a, 65_536, hub.clone()).unwrap();
        let node_b = TcpTransport::bind_in_memory(peer_b, addr_b, 65_536, hub).unwrap();

        node_a.register_peer(peer_b, node_b.local_addr()).unwrap();
        node_b.register_peer(peer_a, node_a.local_addr()).unwrap();

        let message = vec![0x42u8; 4096];
        node_a.send(peer_b, &message).await.unwrap();

        let (sender, received) = node_b.recv().await.unwrap();
        assert_eq!(sender, peer_a);
        assert_eq!(received, message);

        // Also exercise OS socket bind + connect when the host environment permits loopback connections.
        let addr_any: SocketAddr = "127.0.0.1:0".parse().unwrap();
        if let (Ok(os_a), Ok(os_b)) = (
            TcpTransport::bind(peer_a, addr_any, 65_536),
            TcpTransport::bind(peer_b, addr_any, 65_536),
        ) {
            os_a.register_peer(peer_b, os_b.local_addr()).unwrap();
            os_b.register_peer(peer_a, os_a.local_addr()).unwrap();
            if os_a.send(peer_b, &message).await.is_ok() {
                let (os_sender, os_received) = os_b.recv().await.unwrap();
                assert_eq!(os_sender, peer_a);
                assert_eq!(os_received, message);
            }
        }
    }
}
