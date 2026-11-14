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
use std::task::{Context, Poll};
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

struct EofTrackingStdRead<'a, R> {
    inner: &'a mut R,
    saw_eof: bool,
}

impl<R: Read> Read for EofTrackingStdRead<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n == 0 && !buf.is_empty() {
            self.saw_eof = true;
        }
        Ok(n)
    }
}

#[cfg(feature = "tcp-transport")]
struct TokioReadAdapter<'a> {
    stream: &'a tokio::net::TcpStream,
    hit_would_block: bool,
    saw_eof: bool,
}

#[cfg(feature = "tcp-transport")]
impl Read for TokioReadAdapter<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self.stream.try_read(buf) {
            Ok(0) => {
                if !buf.is_empty() {
                    self.saw_eof = true;
                }
                Ok(0)
            }
            Ok(n) => Ok(n),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                self.hit_would_block = true;
                Err(e)
            }
            Err(e) => Err(e),
        }
    }
}

enum OsTcpStreamSlot {
    Std(TcpStream),
    #[cfg(feature = "tcp-transport")]
    Tokio(tokio::net::TcpStream),
    #[cfg(feature = "tcp-transport")]
    Empty,
}

impl OsTcpStreamSlot {
    #[cfg(feature = "tcp-transport")]
    fn ensure_tokio(&mut self) -> Result<&tokio::net::TcpStream> {
        if matches!(self, Self::Std(_)) {
            let Self::Std(std_stream) = core::mem::replace(self, Self::Empty) else {
                return Err(Error::TransportUnavailable);
            };
            let tokio_stream = tokio::net::TcpStream::from_std(std_stream)
                .map_err(|_| Error::TransportUnavailable)?;
            *self = Self::Tokio(tokio_stream);
        }
        match self {
            Self::Tokio(ref stream) => Ok(stream),
            Self::Std(_) | Self::Empty => Err(Error::TransportUnavailable),
        }
    }
}

enum OsTcpListenerSlot {
    Std(TcpListener),
    #[cfg(feature = "tcp-transport")]
    Tokio(tokio::net::TcpListener),
    #[cfg(feature = "tcp-transport")]
    Empty,
}

impl OsTcpListenerSlot {
    #[cfg(feature = "tcp-transport")]
    fn ensure_tokio(&mut self) -> Result<&tokio::net::TcpListener> {
        if matches!(self, Self::Std(_)) {
            let Self::Std(std_listener) = core::mem::replace(self, Self::Empty) else {
                return Err(Error::TransportUnavailable);
            };
            let tokio_listener = tokio::net::TcpListener::from_std(std_listener)
                .map_err(|_| Error::TransportUnavailable)?;
            *self = Self::Tokio(tokio_listener);
        }
        match self {
            Self::Tokio(ref listener) => Ok(listener),
            Self::Std(_) | Self::Empty => Err(Error::TransportUnavailable),
        }
    }
}

struct ActiveConnection {
    stream: OsTcpStreamSlot,
    reader_state: TcpFrameReaderState,
    remote_addr: SocketAddr,
}

impl ActiveConnection {
    fn try_read_frame_sync(&mut self, saw_eof: &mut bool) -> Result<Option<Vec<u8>>> {
        *saw_eof = false;
        match &mut self.stream {
            OsTcpStreamSlot::Std(stream) => {
                let mut reader = EofTrackingStdRead {
                    inner: stream,
                    saw_eof: false,
                };
                let res = self.reader_state.try_read_frame(&mut reader);
                *saw_eof = reader.saw_eof;
                res
            }
            #[cfg(feature = "tcp-transport")]
            OsTcpStreamSlot::Tokio(stream) => {
                let mut reader = TokioReadAdapter {
                    stream,
                    hit_would_block: false,
                    saw_eof: false,
                };
                let res = self.reader_state.try_read_frame(&mut reader);
                *saw_eof = reader.saw_eof;
                res
            }
            #[cfg(feature = "tcp-transport")]
            OsTcpStreamSlot::Empty => Err(Error::TransportUnavailable),
        }
    }
}

type InMemoryStreamQueueMap = BTreeMap<SocketAddr, VecDeque<(SocketAddr, Vec<u8>)>>;

#[derive(Default)]
struct InMemoryTcpHubState {
    streams: InMemoryStreamQueueMap,
    waiters: BTreeMap<SocketAddr, crate::waiters::WaiterQueue>,
}

/// Shared in-memory raw byte-stream hub used when OS socket creation is restricted or for
/// deterministic framed stream testing.
#[derive(Default, Clone)]
pub struct InMemoryTcpStreamHub {
    inner: Arc<Mutex<InMemoryTcpHubState>>,
}

impl InMemoryTcpStreamHub {
    /// Create a new in-memory byte-stream hub.
    pub fn new() -> Self {
        Self::default()
    }
}

enum TcpBackend {
    OsSocket(Mutex<OsTcpListenerSlot>),
    InMemory {
        hub: InMemoryTcpStreamHub,
        reader_state: Mutex<TcpFrameReaderState>,
    },
}

/// Concrete length-prefixed TCP [`SwarmTransport`] (`ADR-0022` §1.3, `W11-03`, `ST-06`).
///
/// Keyed for outbound routing by `PeerId -> SocketAddr` and backed on receive by
/// cancellation-safe [`TcpFrameReaderState`] readers that surface only an
/// unauthenticated [`crate::traits::ObservedSourceRef`].
pub struct TcpTransport {
    local_peer: PeerId,
    local_addr: SocketAddr,
    max_message_bytes: usize,
    backend: TcpBackend,
    peers: RwLock<BTreeMap<PeerId, SocketAddr>>,
    connections: Mutex<Vec<ActiveConnection>>,
    inbox: Mutex<VecDeque<crate::traits::ObservedTransportMessage>>,
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
            backend: TcpBackend::OsSocket(Mutex::new(OsTcpListenerSlot::Std(listener))),
            peers: RwLock::new(BTreeMap::new()),
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
        if let Ok(mut state) = hub.inner.lock() {
            state.streams.entry(virtual_addr).or_default();
        }
        Ok(Arc::new(Self {
            local_peer,
            local_addr: virtual_addr,
            max_message_bytes,
            backend: TcpBackend::InMemory {
                hub,
                reader_state: Mutex::new(TcpFrameReaderState::new(max_message_bytes)),
            },
            peers: RwLock::new(BTreeMap::new()),
            connections: Mutex::new(Vec::new()),
            inbox: Mutex::new(VecDeque::new()),
        }))
    }

    /// Bound local socket address.
    pub const fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Register a peer's outbound `SocketAddr` in the transport directory (`PeerId` lookup is
    /// strictly outbound; inbound identity always comes from downstream cryptographic verification).
    pub fn register_peer(&self, peer: PeerId, addr: SocketAddr) -> Result<()> {
        self.peers
            .write()
            .map_err(|_| Error::TransportUnavailable)?
            .insert(peer, addr);
        Ok(())
    }

    /// Synchronous framed send: transmits `u32_le(len) || msg` to `peer` with no spoofable
    /// unauthenticated `PeerId` header prologue.
    pub fn send_sync(&self, peer: PeerId, msg: &[u8]) -> Result<()> {
        if msg.is_empty() || msg.len() > self.max_message_bytes {
            return Err(Error::InvalidMessage);
        }
        let addr = {
            let guard = self.peers.read().map_err(|_| Error::TransportUnavailable)?;
            *guard.get(&peer).ok_or(Error::PeerNotFound)?
        };

        let frame = encode_length_prefixed_frame(msg, self.max_message_bytes)?;

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
            TcpBackend::InMemory { hub, .. } => self.send_in_memory(hub, addr, frame),
        }
    }

    fn lookup_peer_addr(&self, peer: PeerId) -> Result<SocketAddr> {
        let guard = self.peers.read().map_err(|_| Error::TransportUnavailable)?;
        guard.get(&peer).copied().ok_or(Error::PeerNotFound)
    }

    fn send_in_memory(
        &self,
        hub: &InMemoryTcpStreamHub,
        addr: SocketAddr,
        frame: Vec<u8>,
    ) -> Result<()> {
        let wakers = {
            let mut state = hub.inner.lock().map_err(|_| Error::TransportUnavailable)?;
            let queue = state
                .streams
                .get_mut(&addr)
                .ok_or(Error::ConnectionFailed)?;
            queue.push_back((self.local_addr, frame));
            state
                .waiters
                .get_mut(&addr)
                .map(crate::waiters::WaiterQueue::take_all)
                .unwrap_or_default()
        };
        crate::waiters::wake_all(wakers);
        Ok(())
    }

    fn poll_recv_in_memory(
        &self,
        hub: &InMemoryTcpStreamHub,
        reader_state: &Mutex<TcpFrameReaderState>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<crate::traits::ObservedTransportMessage>> {
        if let Some(item) = match self.inbox.lock() {
            Ok(mut inbox) => inbox.pop_front(),
            Err(_) => return Poll::Ready(Err(Error::TransportUnavailable)),
        } {
            return Poll::Ready(Ok(item));
        }

        loop {
            let item_opt = {
                let Ok(mut state) = hub.inner.lock() else {
                    return Poll::Ready(Err(Error::TransportUnavailable));
                };
                state
                    .streams
                    .get_mut(&self.local_addr)
                    .and_then(|q| q.pop_front())
            };

            if let Some((source_addr, raw_bytes)) = item_opt {
                let mut cursor = Cursor::new(raw_bytes);
                let Ok(mut r_state) = reader_state.lock() else {
                    return Poll::Ready(Err(Error::TransportUnavailable));
                };
                match r_state.try_read_frame(&mut cursor) {
                    Ok(Some(payload)) => {
                        return Poll::Ready(Ok(crate::traits::ObservedTransportMessage {
                            observed_source: crate::traits::ObservedSourceRef::from(source_addr),
                            payload,
                        }));
                    }
                    Ok(None) => continue,
                    Err(err) => return Poll::Ready(Err(err)),
                }
            }

            let Ok(mut state) = hub.inner.lock() else {
                return Poll::Ready(Err(Error::TransportUnavailable));
            };
            let is_empty = state
                .streams
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

    #[cfg(feature = "tcp-transport")]
    fn poll_recv_os_async(
        &self,
        listener_slot: &Mutex<OsTcpListenerSlot>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<crate::traits::ObservedTransportMessage>> {
        if let Some(item) = match self.inbox.lock() {
            Ok(mut inbox) => inbox.pop_front(),
            Err(_) => return Poll::Ready(Err(Error::TransportUnavailable)),
        } {
            return Poll::Ready(Ok(item));
        }

        let mut accepted = Vec::new();
        let mut accept_err: Option<Error> = None;
        {
            let Ok(mut slot) = listener_slot.lock() else {
                return Poll::Ready(Err(Error::TransportUnavailable));
            };
            let listener = match slot.ensure_tokio() {
                Ok(l) => l,
                Err(err) => return Poll::Ready(Err(err)),
            };
            for _ in 0..256 {
                match listener.poll_accept(cx) {
                    Poll::Ready(Ok((stream, remote_addr))) => {
                        accepted.push((OsTcpStreamSlot::Tokio(stream), remote_addr));
                    }
                    Poll::Pending => break,
                    Poll::Ready(Err(_)) => {
                        accept_err = Some(Error::ReceiveFailed);
                        break;
                    }
                }
            }
        }

        let Ok(mut conns) = self.connections.lock() else {
            return Poll::Ready(Err(Error::TransportUnavailable));
        };
        for (stream, remote_addr) in accepted {
            conns.push(ActiveConnection {
                stream,
                reader_state: TcpFrameReaderState::new(self.max_message_bytes),
                remote_addr,
            });
        }

        let mut idx = 0;
        let mut first_error: Option<Error> = accept_err;
        while idx < conns.len() {
            if let Err(err) = conns[idx].stream.ensure_tokio() {
                conns.swap_remove(idx);
                if first_error.is_none() {
                    first_error = Some(err);
                }
                continue;
            }

            let mut removed = false;
            loop {
                let conn = &mut conns[idx];
                let OsTcpStreamSlot::Tokio(ref tokio_stream) = conn.stream else {
                    removed = true;
                    break;
                };
                match tokio_stream.poll_read_ready(cx) {
                    Poll::Ready(Ok(())) => {
                        let mut adapter = TokioReadAdapter {
                            stream: tokio_stream,
                            hit_would_block: false,
                            saw_eof: false,
                        };
                        match conn.reader_state.try_read_frame(&mut adapter) {
                            Ok(Some(payload)) => {
                                let msg = crate::traits::ObservedTransportMessage {
                                    observed_source: crate::traits::ObservedSourceRef::from(
                                        conn.remote_addr,
                                    ),
                                    payload,
                                };
                                if conns.len() > 1 {
                                    let served = conns.remove(idx);
                                    conns.push(served);
                                }
                                return Poll::Ready(Ok(msg));
                            }
                            Ok(None) if adapter.hit_would_block => {
                                continue;
                            }
                            Ok(None) if adapter.saw_eof => {
                                conns.swap_remove(idx);
                                removed = true;
                                break;
                            }
                            Ok(None) => {
                                continue;
                            }
                            Err(err) => {
                                conns.swap_remove(idx);
                                if first_error.is_none() {
                                    first_error = Some(err);
                                }
                                removed = true;
                                break;
                            }
                        }
                    }
                    Poll::Ready(Err(_)) => {
                        conns.swap_remove(idx);
                        if first_error.is_none() {
                            first_error = Some(Error::ReceiveFailed);
                        }
                        removed = true;
                        break;
                    }
                    Poll::Pending => break,
                }
            }

            if !removed {
                idx += 1;
            }
        }
        drop(conns);

        if let Some(err) = first_error {
            return Poll::Ready(Err(err));
        }
        Poll::Pending
    }

    /// Poll non-blocking listener and active connections once, returning the next complete
    /// [`crate::traits::ObservedTransportMessage`] if available.
    pub fn poll_recv_once(&self) -> Result<Option<crate::traits::ObservedTransportMessage>> {
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
                let item_opt = {
                    let mut state = hub.inner.lock().map_err(|_| Error::TransportUnavailable)?;
                    state
                        .streams
                        .get_mut(&self.local_addr)
                        .and_then(|q| q.pop_front())
                };
                if let Some((source_addr, raw_bytes)) = item_opt {
                    let mut cursor = Cursor::new(raw_bytes);
                    let mut r_state = reader_state
                        .lock()
                        .map_err(|_| Error::TransportUnavailable)?;
                    if let Some(payload) = r_state.try_read_frame(&mut cursor)? {
                        return Ok(Some(crate::traits::ObservedTransportMessage {
                            observed_source: crate::traits::ObservedSourceRef::from(source_addr),
                            payload,
                        }));
                    }
                }
                Ok(None)
            }
            TcpBackend::OsSocket(listener_slot) => {
                let mut accepted = Vec::new();
                {
                    let mut slot = listener_slot
                        .lock()
                        .map_err(|_| Error::TransportUnavailable)?;
                    match &mut *slot {
                        OsTcpListenerSlot::Std(listener) => loop {
                            match listener.accept() {
                                Ok((stream, remote_addr)) => {
                                    let _ = stream.set_nonblocking(true);
                                    accepted.push((OsTcpStreamSlot::Std(stream), remote_addr));
                                }
                                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                                Err(_) => break,
                            }
                        },
                        #[cfg(feature = "tcp-transport")]
                        OsTcpListenerSlot::Tokio(listener) => {
                            let waker = std::task::Waker::from(Arc::new(crate::waiters::NoopWake));
                            let mut cx = Context::from_waker(&waker);
                            while let Poll::Ready(Ok((stream, remote_addr))) =
                                listener.poll_accept(&mut cx)
                            {
                                accepted.push((OsTcpStreamSlot::Tokio(stream), remote_addr));
                            }
                        }
                        #[cfg(feature = "tcp-transport")]
                        OsTcpListenerSlot::Empty => return Err(Error::TransportUnavailable),
                    }
                }

                let mut conns = self
                    .connections
                    .lock()
                    .map_err(|_| Error::TransportUnavailable)?;
                for (stream, remote_addr) in accepted {
                    conns.push(ActiveConnection {
                        stream,
                        reader_state: TcpFrameReaderState::new(self.max_message_bytes),
                        remote_addr,
                    });
                }

                let mut idx = 0;
                let mut first_error: Option<Error> = None;
                while idx < conns.len() {
                    let mut saw_eof = false;
                    let res = conns[idx].try_read_frame_sync(&mut saw_eof);
                    match res {
                        Ok(Some(payload)) => {
                            let observed_source =
                                crate::traits::ObservedSourceRef::from(conns[idx].remote_addr);
                            if let Ok(mut inbox) = self.inbox.lock() {
                                inbox.push_back(crate::traits::ObservedTransportMessage {
                                    observed_source,
                                    payload,
                                });
                            }
                            idx += 1;
                        }
                        Ok(None) if saw_eof => {
                            conns.swap_remove(idx);
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
            if msg.is_empty() || msg.len() > self.max_message_bytes {
                return Err(Error::InvalidMessage);
            }
            let addr = self.lookup_peer_addr(peer)?;
            let frame = encode_length_prefixed_frame(msg, self.max_message_bytes)?;
            match &self.backend {
                TcpBackend::OsSocket(_) => {
                    #[cfg(feature = "tcp-transport")]
                    {
                        use tokio::io::AsyncWriteExt;
                        let mut stream = tokio::net::TcpStream::connect(addr)
                            .await
                            .map_err(|_| Error::ConnectionFailed)?;
                        stream
                            .write_all(&frame)
                            .await
                            .map_err(|_| Error::SendFailed)?;
                        stream.flush().await.map_err(|_| Error::SendFailed)?;
                        Ok(())
                    }
                    #[cfg(not(feature = "tcp-transport"))]
                    {
                        let _ = addr;
                        let _ = frame;
                        Err(Error::TransportUnavailable)
                    }
                }
                TcpBackend::InMemory { hub, .. } => self.send_in_memory(hub, addr, frame),
            }
        })
    }

    fn recv(&self) -> Self::RecvFuture<'_> {
        Box::pin(async move {
            match &self.backend {
                TcpBackend::InMemory { hub, reader_state } => {
                    core::future::poll_fn(|cx| self.poll_recv_in_memory(hub, reader_state, cx))
                        .await
                }
                TcpBackend::OsSocket(listener_slot) => {
                    #[cfg(feature = "tcp-transport")]
                    {
                        core::future::poll_fn(|cx| self.poll_recv_os_async(listener_slot, cx)).await
                    }
                    #[cfg(not(feature = "tcp-transport"))]
                    {
                        let _ = listener_slot;
                        self.poll_recv_once()?.ok_or(Error::TransportUnavailable)
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

        let msg_rx = node_b.recv().await.unwrap();
        assert_eq!(
            msg_rx.observed_source,
            crate::traits::ObservedSourceRef::from(addr_a)
        );
        assert_eq!(msg_rx.payload, message);

        // Short frames (< 32 bytes) pass untouched without bogus 32-byte PeerId prologue stripping.
        node_a.send(peer_b, b"short-raw").await.unwrap();
        let short_rx = node_b.recv().await.unwrap();
        assert_eq!(short_rx.payload, b"short-raw");

        // Also exercise OS socket bind + connect when `tcp-transport` and loopback sockets are enabled.
        #[cfg(feature = "tcp-transport")]
        {
            let addr_any: SocketAddr = "127.0.0.1:0".parse().unwrap();
            if let (Ok(os_a), Ok(os_b)) = (
                TcpTransport::bind(peer_a, addr_any, 65_536),
                TcpTransport::bind(peer_b, addr_any, 65_536),
            ) {
                os_a.register_peer(peer_b, os_b.local_addr()).unwrap();
                os_b.register_peer(peer_a, os_a.local_addr()).unwrap();
                if os_a.send(peer_b, &message).await.is_ok() {
                    let os_rx = os_b.recv().await.unwrap();
                    assert!(matches!(
                        os_rx.observed_source,
                        crate::traits::ObservedSourceRef::Ipv4 { .. }
                    ));
                    assert_eq!(os_rx.payload, message);
                }
            }
        }
    }

    #[tokio::test]
    async fn tcp_in_memory_recv_suspends_until_send_wakes_task() {
        let peer_a = PeerId::new([0x31; 32]);
        let peer_b = PeerId::new([0x32; 32]);
        let hub = InMemoryTcpStreamHub::new();
        let addr_a: SocketAddr = "127.0.0.1:41101".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:41102".parse().unwrap();
        let node_a = TcpTransport::bind_in_memory(peer_a, addr_a, 65_536, hub.clone()).unwrap();
        let node_b = TcpTransport::bind_in_memory(peer_b, addr_b, 65_536, hub).unwrap();
        node_a.register_peer(peer_b, node_b.local_addr()).unwrap();

        let rx_task = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(2), node_b.recv())
                .await
                .expect("in-memory tcp recv must wake on send")
                .unwrap()
        });

        tokio::task::yield_now().await;
        node_a.send(peer_b, b"evented-in-memory-tcp").await.unwrap();
        let msg = rx_task.await.unwrap();
        assert_eq!(msg.payload, b"evented-in-memory-tcp");
    }

    #[cfg(feature = "tcp-transport")]
    #[tokio::test]
    async fn tcp_os_socket_concurrent_recv_before_send_and_partial_frame_resumption() {
        use tokio::io::AsyncWriteExt;

        let peer_a = PeerId::new([0x41; 32]);
        let peer_b = PeerId::new([0x42; 32]);
        let addr_any: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let Ok(os_a) = TcpTransport::bind(peer_a, addr_any, 65_536) else {
            return;
        };
        let Ok(os_b) = TcpTransport::bind(peer_b, addr_any, 65_536) else {
            return;
        };
        os_a.register_peer(peer_b, os_b.local_addr()).unwrap();
        let b_addr = os_b.local_addr();

        let rx_task = tokio::spawn(async move {
            let first = tokio::time::timeout(Duration::from_secs(2), os_b.recv())
                .await
                .expect("first OS recv must wake without polling sleep")
                .unwrap();
            let second = tokio::time::timeout(Duration::from_secs(2), os_b.recv())
                .await
                .expect("second OS recv must reassemble split frame cleanly")
                .unwrap();
            (first, second)
        });

        tokio::task::yield_now().await;
        os_a.send(peer_b, b"first-async-tcp").await.unwrap();

        // Also deliver a second frame split across two writes (header + half payload, then rest).
        let frame = encode_length_prefixed_frame(b"split-tcp-payload", 65_536).unwrap();
        let mut raw_stream = tokio::net::TcpStream::connect(b_addr).await.unwrap();
        raw_stream.write_all(&frame[..8]).await.unwrap();
        raw_stream.flush().await.unwrap();
        tokio::task::yield_now().await;
        raw_stream.write_all(&frame[8..]).await.unwrap();
        raw_stream.flush().await.unwrap();
        drop(raw_stream);

        let (first, second) = rx_task.await.unwrap();
        assert_eq!(first.payload, b"first-async-tcp");
        assert_eq!(second.payload, b"split-tcp-payload");
    }
}
