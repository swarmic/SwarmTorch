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
use std::time::{Duration, Instant};

use swarm_torch_core::observe::{EventRecord, RunEventEmitter};
use swarm_torch_core::traits::PeerId;

use crate::audit::{AuditSeverity, TransportAuditBufferV1, TransportDiagnosticsSnapshotV1};
use crate::traits::{
    BandwidthClass, BroadcastStats, ReliabilityClass, SwarmTransport, TransportCapabilities,
};
use crate::{Error, Result};

/// Default maximum message size accepted by [`TcpTransport`] (`1 MiB`).
pub const DEFAULT_TCP_MAX_MESSAGE_BYTES: usize = 1_048_576;

/// Default maximum concurrent inbound TCP connections retained by [`TcpTransport`] (`AR-14-09`).
pub const DEFAULT_MAX_ACTIVE_TCP_CONNECTIONS: usize = 256;

/// Default maximum inbound sockets accepted per poll turn (`15C-3b`).
pub const DEFAULT_MAX_TCP_ACCEPT_BATCH: usize = 256;

/// Default unauthenticated partial-frame Slowloris deadline (`15C-3b`).
pub const DEFAULT_TCP_PARTIAL_FRAME_TIMEOUT: Duration = Duration::from_secs(5);

/// Default idle TCP connection timeout before reclamation (`15C-3b`).
pub const DEFAULT_TCP_IDLE_CONNECTION_TIMEOUT: Duration = Duration::from_secs(30);

/// Tunable bounds and anti-Slowloris deadlines for [`TcpTransport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcpTransportConfigV1 {
    /// Maximum length-prefixed payload size accepted before pre-allocation rejection.
    pub max_message_bytes: usize,
    /// Maximum concurrent active inbound sockets retained in memory.
    pub max_active_connections: usize,
    /// Maximum time a socket may stay mid-frame without completing a valid frame.
    pub partial_frame_timeout: Duration,
    /// Maximum time an idle socket may remain open between completed frames.
    pub idle_connection_timeout: Duration,
    /// Maximum inbound sockets accepted per poll iteration before yielding/re-waking.
    pub max_accept_batch: usize,
}

impl Default for TcpTransportConfigV1 {
    fn default() -> Self {
        Self {
            max_message_bytes: DEFAULT_TCP_MAX_MESSAGE_BYTES,
            max_active_connections: DEFAULT_MAX_ACTIVE_TCP_CONNECTIONS,
            partial_frame_timeout: DEFAULT_TCP_PARTIAL_FRAME_TIMEOUT,
            idle_connection_timeout: DEFAULT_TCP_IDLE_CONNECTION_TIMEOUT,
            max_accept_batch: DEFAULT_MAX_TCP_ACCEPT_BATCH,
        }
    }
}

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

    /// Whether the reader is currently between frames with zero buffered partial bytes (`AR-14-09`).
    pub const fn is_idle(&self) -> bool {
        matches!(self.phase, FrameReadPhase::ReadingLength { filled: 0, .. })
    }

    /// Attempt to advance the frame reader from any `std::io::Read` source.
    ///
    /// - Returns `Ok(Some(frame))` when a full frame has been assembled.
    /// - Returns `Ok(None)` when the underlying stream returned `WouldBlock` / `TimedOut`
    ///   (preserving all partially read bytes in `self`).
    /// - Poison-closes the state and returns `Err(Error::InvalidMessage)` if `len == 0`,
    ///   `len > max_message_bytes`, or unexpected EOF occurs mid-frame.
    pub fn try_read_frame<R: Read>(&mut self, reader: &mut R) -> Result<Option<Vec<u8>>> {
        self.try_read_frame_with_progress(reader).0
    }

    pub(crate) fn try_read_frame_with_progress<R: Read>(
        &mut self,
        reader: &mut R,
    ) -> (Result<Option<Vec<u8>>>, usize) {
        let mut bytes_read = 0usize;
        loop {
            match &mut self.phase {
                FrameReadPhase::Poisoned => return (Err(Error::InvalidMessage), bytes_read),
                FrameReadPhase::ReadingLength { buf, filled } => {
                    while *filled < 4 {
                        match reader.read(&mut buf[*filled..4]) {
                            Ok(0) => {
                                if *filled == 0 {
                                    return (Ok(None), bytes_read);
                                }
                                self.phase = FrameReadPhase::Poisoned;
                                return (Err(Error::InvalidMessage), bytes_read);
                            }
                            Ok(n) => {
                                *filled += n;
                                bytes_read = bytes_read.saturating_add(n);
                            }
                            Err(ref e)
                                if e.kind() == io::ErrorKind::WouldBlock
                                    || e.kind() == io::ErrorKind::TimedOut =>
                            {
                                return (Ok(None), bytes_read);
                            }
                            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                            Err(_) => {
                                self.phase = FrameReadPhase::Poisoned;
                                return (Err(Error::ReceiveFailed), bytes_read);
                            }
                        }
                    }

                    let len = u32::from_le_bytes(*buf) as usize;
                    if len == 0 || len > self.max_message_bytes {
                        self.phase = FrameReadPhase::Poisoned;
                        return (Err(Error::InvalidMessage), bytes_read);
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
                                return (Err(Error::InvalidMessage), bytes_read);
                            }
                            Ok(n) => {
                                *filled += n;
                                bytes_read = bytes_read.saturating_add(n);
                            }
                            Err(ref e)
                                if e.kind() == io::ErrorKind::WouldBlock
                                    || e.kind() == io::ErrorKind::TimedOut =>
                            {
                                return (Ok(None), bytes_read);
                            }
                            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                            Err(_) => {
                                self.phase = FrameReadPhase::Poisoned;
                                return (Err(Error::ReceiveFailed), bytes_read);
                            }
                        }
                    }

                    let completed = core::mem::take(buf);
                    self.phase = FrameReadPhase::ReadingLength {
                        buf: [0u8; 4],
                        filled: 0,
                    };
                    return (Ok(Some(completed)), bytes_read);
                }
            }
        }
    }
}

/// Encode a single length-prefixed frame (`u32_le(len) || payload`) with pre-send bounds validation.
pub fn encode_length_prefixed_frame(payload: &[u8], max_message_bytes: usize) -> Result<Vec<u8>> {
    if payload.is_empty() || payload.len() > max_message_bytes {
        return Err(Error::InvalidMessage);
    }
    let len = u32::try_from(payload.len()).map_err(|_| Error::InvalidMessage)?;
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
    accepted_at: Instant,
    partial_frame_started_at: Option<Instant>,
    last_progress_at: Instant,
    completed_frames: u64,
}

impl ActiveConnection {
    fn new(
        stream: OsTcpStreamSlot,
        remote_addr: SocketAddr,
        max_message_bytes: usize,
        now: Instant,
    ) -> Self {
        Self {
            stream,
            reader_state: TcpFrameReaderState::new(max_message_bytes),
            remote_addr,
            accepted_at: now,
            partial_frame_started_at: None,
            last_progress_at: now,
            completed_frames: 0,
        }
    }

    fn record_read_progress(&mut self, bytes_read: usize, completed: bool, now: Instant) {
        if bytes_read > 0 || completed {
            self.last_progress_at = now;
        }
        if completed {
            self.completed_frames = self.completed_frames.saturating_add(1);
            self.partial_frame_started_at = if self.reader_state.is_idle() {
                None
            } else {
                Some(now)
            };
        } else if !self.reader_state.is_idle() && self.partial_frame_started_at.is_none() {
            self.partial_frame_started_at = Some(now);
        }
    }

    fn try_read_frame_sync(&mut self, saw_eof: &mut bool, now: Instant) -> Result<Option<Vec<u8>>> {
        *saw_eof = false;
        let (res, bytes_read) = match &mut self.stream {
            OsTcpStreamSlot::Std(stream) => {
                let mut reader = EofTrackingStdRead {
                    inner: stream,
                    saw_eof: false,
                };
                let outcome = self.reader_state.try_read_frame_with_progress(&mut reader);
                *saw_eof = reader.saw_eof;
                outcome
            }
            #[cfg(feature = "tcp-transport")]
            OsTcpStreamSlot::Tokio(stream) => {
                let mut reader = TokioReadAdapter {
                    stream,
                    hit_would_block: false,
                    saw_eof: false,
                };
                let outcome = self.reader_state.try_read_frame_with_progress(&mut reader);
                *saw_eof = reader.saw_eof;
                outcome
            }
            #[cfg(feature = "tcp-transport")]
            OsTcpStreamSlot::Empty => (Err(Error::TransportUnavailable), 0),
        };
        self.record_read_progress(bytes_read, matches!(res, Ok(Some(_))), now);
        res
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
    config: TcpTransportConfigV1,
    backend: TcpBackend,
    peers: RwLock<BTreeMap<PeerId, SocketAddr>>,
    connections: Mutex<Vec<ActiveConnection>>,
    inbox: Mutex<VecDeque<crate::traits::ObservedTransportMessage>>,
    audit: TransportAuditBufferV1,
}

impl core::fmt::Debug for TcpTransport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TcpTransport")
            .field("local_peer", &self.local_peer)
            .field("local_addr", &self.local_addr)
            .field("max_message_bytes", &self.config.max_message_bytes)
            .field(
                "max_active_connections",
                &self.config.max_active_connections,
            )
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
        Self::bind_with_limits(
            local_peer,
            bind_addr,
            max_message_bytes,
            DEFAULT_MAX_ACTIVE_TCP_CONNECTIONS,
        )
    }

    /// Bind a new [`TcpTransport`] to `bind_addr` with explicit message and connection caps (`AR-14-09`).
    pub fn bind_with_limits(
        local_peer: PeerId,
        bind_addr: SocketAddr,
        max_message_bytes: usize,
        max_active_connections: usize,
    ) -> Result<Arc<Self>> {
        Self::bind_with_config(
            local_peer,
            bind_addr,
            TcpTransportConfigV1 {
                max_message_bytes,
                max_active_connections: max_active_connections.max(1),
                ..TcpTransportConfigV1::default()
            },
        )
    }

    /// Bind a new [`TcpTransport`] to `bind_addr` using explicit bounds and anti-Slowloris deadlines.
    pub fn bind_with_config(
        local_peer: PeerId,
        bind_addr: SocketAddr,
        mut config: TcpTransportConfigV1,
    ) -> Result<Arc<Self>> {
        if config.max_message_bytes == 0 || config.max_message_bytes > u32::MAX as usize {
            return Err(Error::InvalidMessage);
        }
        config.max_active_connections = config.max_active_connections.max(1);
        config.max_accept_batch = config.max_accept_batch.max(1);

        let listener = TcpListener::bind(bind_addr).map_err(|_| Error::ConnectionFailed)?;
        listener
            .set_nonblocking(true)
            .map_err(|_| Error::ConnectionFailed)?;
        let local_addr = listener.local_addr().map_err(|_| Error::ConnectionFailed)?;

        Ok(Arc::new(Self {
            local_peer,
            local_addr,
            config,
            backend: TcpBackend::OsSocket(Mutex::new(OsTcpListenerSlot::Std(listener))),
            peers: RwLock::new(BTreeMap::new()),
            connections: Mutex::new(Vec::new()),
            inbox: Mutex::new(VecDeque::new()),
            audit: TransportAuditBufferV1::new(local_peer),
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
        let config = TcpTransportConfigV1 {
            max_message_bytes,
            ..TcpTransportConfigV1::default()
        };
        Ok(Arc::new(Self {
            local_peer,
            local_addr: virtual_addr,
            config,
            backend: TcpBackend::InMemory {
                hub,
                reader_state: Mutex::new(TcpFrameReaderState::new(max_message_bytes)),
            },
            peers: RwLock::new(BTreeMap::new()),
            connections: Mutex::new(Vec::new()),
            inbox: Mutex::new(VecDeque::new()),
            audit: TransportAuditBufferV1::new(local_peer),
        }))
    }

    /// Configured maximum concurrent active inbound connections (`AR-14-09`).
    pub const fn max_active_connections(&self) -> usize {
        self.config.max_active_connections
    }

    /// Return a point-in-time snapshot of bounded security, Slowloris-eviction, and drop counters.
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

    fn sweep_expired_connections(
        conns: &mut Vec<ActiveConnection>,
        config: &TcpTransportConfigV1,
        now: Instant,
        audit: &TransportAuditBufferV1,
    ) {
        let mut idx = 0;
        while idx < conns.len() {
            let conn = &conns[idx];
            let partial_expired = !conn.reader_state.is_idle()
                && conn.partial_frame_started_at.is_some_and(|started| {
                    now.saturating_duration_since(started) >= config.partial_frame_timeout
                });
            if partial_expired {
                let evicted = conns.swap_remove(idx);
                audit.record_event(
                    AuditSeverity::Critical,
                    "swarmtorch.net.tcp.slowloris_evicted",
                    "tcp",
                    Some(evicted.remote_addr.to_string()),
                    "partial_frame_timeout_exceeded",
                    |c| {
                        c.evicted_stalled_partial_connections =
                            c.evicted_stalled_partial_connections.saturating_add(1);
                    },
                );
                continue;
            }
            let idle_expired = conn.reader_state.is_idle()
                && now.saturating_duration_since(conn.last_progress_at)
                    >= config.idle_connection_timeout;
            if idle_expired {
                let evicted = conns.swap_remove(idx);
                audit.record_event(
                    AuditSeverity::Noisy,
                    "swarmtorch.net.tcp.idle_evicted",
                    "tcp",
                    Some(evicted.remote_addr.to_string()),
                    "idle_connection_timeout_exceeded",
                    |c| {
                        c.evicted_idle_connections = c.evicted_idle_connections.saturating_add(1);
                    },
                );
                continue;
            }
            idx += 1;
        }
    }

    fn push_bounded_connection(
        conns: &mut Vec<ActiveConnection>,
        incoming: ActiveConnection,
        config: &TcpTransportConfigV1,
        now: Instant,
        audit: &TransportAuditBufferV1,
    ) {
        Self::sweep_expired_connections(conns, config, now, audit);
        let max_conns = config.max_active_connections.max(1);
        if conns.len() >= max_conns {
            // Priority 1: evict oldest unauthenticated stalled partial-frame connection (`Slowloris`)
            // before honest idle sockets or new incoming connections.
            let unauth_partial = conns
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.reader_state.is_idle() && c.completed_frames == 0)
                .min_by_key(|(_, c)| {
                    (
                        c.partial_frame_started_at.unwrap_or(c.accepted_at),
                        c.last_progress_at,
                        c.accepted_at,
                    )
                })
                .map(|(i, _)| (i, true));

            // Priority 2: any other oldest mid-frame socket before honest idle sockets.
            let any_partial = unauth_partial.or_else(|| {
                conns
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| !c.reader_state.is_idle())
                    .min_by_key(|(_, c)| {
                        (
                            c.partial_frame_started_at.unwrap_or(c.accepted_at),
                            c.last_progress_at,
                            c.accepted_at,
                        )
                    })
                    .map(|(i, _)| (i, true))
            });

            // Priority 3: oldest idle socket when no partial-frame Slowloris sockets remain.
            let victim = any_partial.or_else(|| {
                conns
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, c)| (c.last_progress_at, c.accepted_at))
                    .map(|(i, _)| (i, false))
            });

            if let Some((victim_idx, is_partial)) = victim {
                let evicted = conns.swap_remove(victim_idx);
                if is_partial {
                    audit.record_event(
                        AuditSeverity::Critical,
                        "swarmtorch.net.tcp.slowloris_evicted",
                        "tcp",
                        Some(evicted.remote_addr.to_string()),
                        "stalled_partial_frame_evicted_under_capacity",
                        |c| {
                            c.evicted_stalled_partial_connections =
                                c.evicted_stalled_partial_connections.saturating_add(1);
                        },
                    );
                } else {
                    audit.record_event(
                        AuditSeverity::Noisy,
                        "swarmtorch.net.tcp.idle_evicted",
                        "tcp",
                        Some(evicted.remote_addr.to_string()),
                        "oldest_idle_evicted_under_capacity",
                        |c| {
                            c.evicted_idle_connections =
                                c.evicted_idle_connections.saturating_add(1);
                        },
                    );
                }
            }
        }
        conns.push(incoming);
    }

    fn classify_tcp_accept_error(&self, err: &io::Error) -> bool {
        let is_transient = matches!(
            err.kind(),
            io::ErrorKind::Interrupted
                | io::ErrorKind::ConnectionReset
                | io::ErrorKind::ConnectionAborted
        );
        self.audit.record_event(
            if is_transient {
                AuditSeverity::Noisy
            } else {
                AuditSeverity::Critical
            },
            "swarmtorch.net.tcp.accept_error",
            "tcp",
            None,
            &err.to_string(),
            |c| {
                c.dropped_connection_errors = c.dropped_connection_errors.saturating_add(1);
            },
        );
        is_transient
    }

    fn record_conn_frame_error(audit: &TransportAuditBufferV1, source: SocketAddr, err: &Error) {
        let is_malformed = matches!(err, Error::InvalidMessage);
        audit.record_event(
            AuditSeverity::Noisy,
            if is_malformed {
                "swarmtorch.net.tcp.frame_rejected"
            } else {
                "swarmtorch.net.tcp.connection_error"
            },
            "tcp",
            Some(source.to_string()),
            &err.to_string(),
            move |c| {
                if is_malformed {
                    c.dropped_malformed_frames = c.dropped_malformed_frames.saturating_add(1);
                } else {
                    c.dropped_connection_errors = c.dropped_connection_errors.saturating_add(1);
                }
            },
        );
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
        if msg.is_empty() || msg.len() > self.config.max_message_bytes {
            return Err(Error::InvalidMessage);
        }
        let addr = {
            let guard = self.peers.read().map_err(|_| Error::TransportUnavailable)?;
            *guard.get(&peer).ok_or(Error::PeerNotFound)?
        };

        let frame = encode_length_prefixed_frame(msg, self.config.max_message_bytes)?;

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
                    Err(err) => {
                        Self::record_conn_frame_error(&self.audit, source_addr, &err);
                        return Poll::Ready(Err(err));
                    }
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

        let max_accept_batch = self.config.max_accept_batch.max(1);
        let mut accepted = Vec::new();
        let mut accept_exhausted = true;
        let mut fatal_accept_error = None;
        {
            let Ok(mut slot) = listener_slot.lock() else {
                return Poll::Ready(Err(Error::TransportUnavailable));
            };
            let listener = match slot.ensure_tokio() {
                Ok(l) => l,
                Err(err) => return Poll::Ready(Err(err)),
            };
            for _ in 0..max_accept_batch {
                match listener.poll_accept(cx) {
                    Poll::Ready(Ok((stream, remote_addr))) => {
                        accepted.push((OsTcpStreamSlot::Tokio(stream), remote_addr));
                    }
                    Poll::Pending => {
                        accept_exhausted = false;
                        break;
                    }
                    Poll::Ready(Err(err)) => {
                        // Transient handshake aborts (`ConnectionReset` / `ConnectionAborted` /
                        // `Interrupted`) must keep looping so `poll_accept` either drains healthy
                        // backlog peers or observes `Poll::Pending` to register `cx.waker()`.
                        if self.classify_tcp_accept_error(&err) {
                            continue;
                        }
                        accept_exhausted = false;
                        fatal_accept_error = Some(Error::ReceiveFailed);
                        break;
                    }
                }
            }
        }

        let now = Instant::now();
        let Ok(mut conns) = self.connections.lock() else {
            return Poll::Ready(Err(Error::TransportUnavailable));
        };
        Self::sweep_expired_connections(&mut conns, &self.config, now, &self.audit);
        for (stream, remote_addr) in accepted {
            Self::push_bounded_connection(
                &mut conns,
                ActiveConnection::new(stream, remote_addr, self.config.max_message_bytes, now),
                &self.config,
                now,
                &self.audit,
            );
        }

        let mut idx = 0;
        while idx < conns.len() {
            if let Err(err) = conns[idx].stream.ensure_tokio() {
                let bad = conns.swap_remove(idx);
                Self::record_conn_frame_error(&self.audit, bad.remote_addr, &err);
                continue;
            }

            let mut removed = false;
            loop {
                let poll_now = Instant::now();
                let conn = &mut conns[idx];
                let OsTcpStreamSlot::Tokio(ref tokio_stream) = conn.stream else {
                    removed = true;
                    break;
                };
                match tokio_stream.poll_read_ready(cx) {
                    Poll::Ready(Ok(())) => {
                        let ((res, bytes_read), hit_would_block, saw_eof) = {
                            let mut adapter = TokioReadAdapter {
                                stream: tokio_stream,
                                hit_would_block: false,
                                saw_eof: false,
                            };
                            let outcome =
                                conn.reader_state.try_read_frame_with_progress(&mut adapter);
                            (outcome, adapter.hit_would_block, adapter.saw_eof)
                        };
                        conn.record_read_progress(bytes_read, matches!(res, Ok(Some(_))), poll_now);
                        match res {
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
                            Ok(None) if hit_would_block => {
                                continue;
                            }
                            Ok(None) if saw_eof => {
                                conns.swap_remove(idx);
                                removed = true;
                                break;
                            }
                            Ok(None) => {
                                break;
                            }
                            Err(err) => {
                                let bad = conns.swap_remove(idx);
                                Self::record_conn_frame_error(&self.audit, bad.remote_addr, &err);
                                removed = true;
                                break;
                            }
                        }
                    }
                    Poll::Ready(Err(err)) => {
                        let bad = conns.swap_remove(idx);
                        self.audit.record_event(
                            AuditSeverity::Noisy,
                            "swarmtorch.net.tcp.connection_error",
                            "tcp",
                            Some(bad.remote_addr.to_string()),
                            &err.to_string(),
                            |c| {
                                c.dropped_connection_errors =
                                    c.dropped_connection_errors.saturating_add(1);
                            },
                        );
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
        Self::sweep_expired_connections(&mut conns, &self.config, Instant::now(), &self.audit);
        drop(conns);

        // `tokio::net::TcpListener::poll_accept` only re-arms its waker when it observes
        // `WouldBlock` (`Poll::Pending`). If our bounded `max_accept_batch` filled up, re-wake
        // before suspending so remaining queued sockets are drained on the next poll turn.
        if accept_exhausted {
            cx.waker().wake_by_ref();
        } else if let Some(err) = fatal_accept_error {
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
                    match r_state.try_read_frame(&mut cursor) {
                        Ok(Some(payload)) => {
                            return Ok(Some(crate::traits::ObservedTransportMessage {
                                observed_source: crate::traits::ObservedSourceRef::from(
                                    source_addr,
                                ),
                                payload,
                            }));
                        }
                        Ok(None) => {}
                        Err(err) => {
                            Self::record_conn_frame_error(&self.audit, source_addr, &err);
                            return Err(err);
                        }
                    }
                }
                Ok(None)
            }
            TcpBackend::OsSocket(listener_slot) => {
                let max_accept_batch = self.config.max_accept_batch.max(1);
                let mut accepted = Vec::new();
                {
                    let mut slot = listener_slot
                        .lock()
                        .map_err(|_| Error::TransportUnavailable)?;
                    match &mut *slot {
                        OsTcpListenerSlot::Std(listener) => {
                            for _ in 0..max_accept_batch {
                                match listener.accept() {
                                    Ok((stream, remote_addr)) => {
                                        let _ = stream.set_nonblocking(true);
                                        accepted.push((OsTcpStreamSlot::Std(stream), remote_addr));
                                    }
                                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => break,
                                    Err(err) => {
                                        if self.classify_tcp_accept_error(&err) {
                                            continue;
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        #[cfg(feature = "tcp-transport")]
                        OsTcpListenerSlot::Tokio(listener) => {
                            let waker = std::task::Waker::from(Arc::new(crate::waiters::NoopWake));
                            let mut cx = Context::from_waker(&waker);
                            for _ in 0..max_accept_batch {
                                match listener.poll_accept(&mut cx) {
                                    Poll::Ready(Ok((stream, remote_addr))) => {
                                        accepted
                                            .push((OsTcpStreamSlot::Tokio(stream), remote_addr));
                                    }
                                    Poll::Pending => break,
                                    Poll::Ready(Err(err)) => {
                                        if self.classify_tcp_accept_error(&err) {
                                            continue;
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        #[cfg(feature = "tcp-transport")]
                        OsTcpListenerSlot::Empty => return Err(Error::TransportUnavailable),
                    }
                }

                let now = Instant::now();
                let mut conns = self
                    .connections
                    .lock()
                    .map_err(|_| Error::TransportUnavailable)?;
                Self::sweep_expired_connections(&mut conns, &self.config, now, &self.audit);
                for (stream, remote_addr) in accepted {
                    Self::push_bounded_connection(
                        &mut conns,
                        ActiveConnection::new(
                            stream,
                            remote_addr,
                            self.config.max_message_bytes,
                            now,
                        ),
                        &self.config,
                        now,
                        &self.audit,
                    );
                }

                let mut idx = 0;
                while idx < conns.len() {
                    let mut saw_eof = false;
                    let res = conns[idx].try_read_frame_sync(&mut saw_eof, Instant::now());
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
                            let bad = conns.swap_remove(idx);
                            Self::record_conn_frame_error(&self.audit, bad.remote_addr, &err);
                        }
                    }
                }
                Self::sweep_expired_connections(
                    &mut conns,
                    &self.config,
                    Instant::now(),
                    &self.audit,
                );
                drop(conns);

                let next = self
                    .inbox
                    .lock()
                    .map_err(|_| Error::TransportUnavailable)?
                    .pop_front();
                Ok(next)
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
            if msg.is_empty() || msg.len() > self.config.max_message_bytes {
                return Err(Error::InvalidMessage);
            }
            let addr = self.lookup_peer_addr(peer)?;
            let frame = encode_length_prefixed_frame(msg, self.config.max_message_bytes)?;
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
            reliability: ReliabilityClass::Reliable,
            bandwidth_class: BandwidthClass::High,
            max_message_size: self.config.max_message_bytes,
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

    #[test]
    fn tcp_transport_enforces_max_active_connections_cap_against_idle_sockets() {
        let p_rx = PeerId::new([0x31u8; 32]);
        let Ok(rx) = TcpTransport::bind_with_limits(p_rx, "127.0.0.1:0".parse().unwrap(), 4096, 2)
        else {
            return;
        };
        assert_eq!(rx.max_active_connections(), 2);

        // Open 4 idle TCP sockets without sending any frame bytes, then send a valid frame on a 5th socket:
        let mut idle_clients = Vec::new();
        for _ in 0..4 {
            if let Ok(s) = TcpStream::connect(rx.local_addr()) {
                idle_clients.push(s);
            }
        }
        let _ = rx.poll_recv_once();
        assert!(rx.connections.lock().unwrap().len() <= 2);

        let p_tx = PeerId::new([0x32u8; 32]);
        let Ok(tx) = TcpTransport::bind(p_tx, "127.0.0.1:0".parse().unwrap(), 4096) else {
            return;
        };
        tx.register_peer(p_rx, rx.local_addr()).unwrap();
        tx.send_sync(p_rx, b"bounded-tcp-ok").unwrap();

        let mut received = None;
        for _ in 0..20 {
            if let Ok(Some(msg)) = rx.poll_recv_once() {
                received = Some(msg.payload);
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(received.as_deref(), Some(&b"bounded-tcp-ok"[..]));
        assert!(rx.connections.lock().unwrap().len() <= 2);
        drop(idle_clients);
    }

    #[test]
    fn tcp_slowloris_partial_frames_evicted_before_honest_idle_and_incoming_admitted() {
        let p_rx = PeerId::new([0x33u8; 32]);
        let Ok(rx) = TcpTransport::bind_with_limits(p_rx, "127.0.0.1:0".parse().unwrap(), 4096, 2)
        else {
            return;
        };

        // Slot 1: honest idle peer (never drips partial garbage).
        let Ok(mut honest_idle) = TcpStream::connect(rx.local_addr()) else {
            return;
        };
        // Slot 2: unauthenticated Slowloris peer drips 1 partial length-header byte and stalls.
        let Ok(mut slowloris) = TcpStream::connect(rx.local_addr()) else {
            return;
        };
        slowloris.write_all(&[0x40]).unwrap();
        slowloris.flush().unwrap();

        for _ in 0..20 {
            let _ = rx.poll_recv_once();
            if rx.connections.lock().unwrap().len() == 2 {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        // Now a 3rd incoming peer connects and sends a complete valid frame:
        // Slowloris slot MUST be evicted BEFORE honest_idle or incoming peer!
        let Ok(mut incoming) = TcpStream::connect(rx.local_addr()) else {
            return;
        };
        let frame_in = encode_length_prefixed_frame(b"incoming-wins-slot", 4096).unwrap();
        incoming.write_all(&frame_in).unwrap();
        incoming.flush().unwrap();

        let mut got_incoming = None;
        for _ in 0..30 {
            if let Ok(Some(msg)) = rx.poll_recv_once() {
                got_incoming = Some(msg.payload);
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(got_incoming.as_deref(), Some(&b"incoming-wins-slot"[..]));

        // Honest idle peer STILL holds its slot and can now deliver its valid frame!
        let frame_idle = encode_length_prefixed_frame(b"honest-idle-survived", 4096).unwrap();
        honest_idle.write_all(&frame_idle).unwrap();
        honest_idle.flush().unwrap();
        let mut got_idle = None;
        for _ in 0..30 {
            if let Ok(Some(msg)) = rx.poll_recv_once() {
                got_idle = Some(msg.payload);
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(got_idle.as_deref(), Some(&b"honest-idle-survived"[..]));

        let diag = rx.diagnostics();
        assert!(diag.evicted_stalled_partial_connections >= 1);
        let events = rx.drain_audit_events();
        assert!(events
            .iter()
            .any(|e| e.name == "swarmtorch.net.tcp.slowloris_evicted"));
        drop(slowloris);
    }

    #[test]
    fn tcp_partial_frame_timeout_evicts_stalled_connection_and_audits() {
        let p_rx = PeerId::new([0x34u8; 32]);
        let cfg = TcpTransportConfigV1 {
            max_message_bytes: 4096,
            max_active_connections: 4,
            partial_frame_timeout: Duration::from_millis(25),
            idle_connection_timeout: Duration::from_secs(30),
            max_accept_batch: 64,
        };
        let Ok(rx) = TcpTransport::bind_with_config(p_rx, "127.0.0.1:0".parse().unwrap(), cfg)
        else {
            return;
        };

        let Ok(mut stalled) = TcpStream::connect(rx.local_addr()) else {
            return;
        };
        stalled.write_all(&[0x10, 0x00]).unwrap();
        stalled.flush().unwrap();

        for _ in 0..20 {
            let _ = rx.poll_recv_once();
            if rx.connections.lock().unwrap().len() == 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(rx.connections.lock().unwrap().len(), 1);

        std::thread::sleep(Duration::from_millis(40));
        assert!(rx.poll_recv_once().unwrap().is_none());
        assert_eq!(rx.connections.lock().unwrap().len(), 0);
        assert_eq!(rx.diagnostics().evicted_stalled_partial_connections, 1);
        drop(stalled);
    }

    #[test]
    fn tcp_single_malformed_peer_isolated_from_healthy_peers_in_poll_recv_once() {
        let p_rx = PeerId::new([0x35u8; 32]);
        let Ok(rx) = TcpTransport::bind(p_rx, "127.0.0.1:0".parse().unwrap(), 512) else {
            return;
        };

        let Ok(mut bad_zero) = TcpStream::connect(rx.local_addr()) else {
            return;
        };
        bad_zero.write_all(&0u32.to_le_bytes()).unwrap();
        bad_zero.flush().unwrap();

        let Ok(mut bad_oversize) = TcpStream::connect(rx.local_addr()) else {
            return;
        };
        bad_oversize.write_all(&4096u32.to_le_bytes()).unwrap();
        bad_oversize.flush().unwrap();

        let Ok(mut good) = TcpStream::connect(rx.local_addr()) else {
            return;
        };
        let good_frame = encode_length_prefixed_frame(b"healthy-peer-survives", 512).unwrap();
        good.write_all(&good_frame).unwrap();
        good.flush().unwrap();

        let mut received = None;
        for _ in 0..30 {
            match rx.poll_recv_once() {
                Ok(Some(msg)) => {
                    received = Some(msg.payload);
                    break;
                }
                Ok(None) => std::thread::sleep(Duration::from_millis(5)),
                Err(err) => panic!("malformed peer must be isolated, not fail poll: {err:?}"),
            }
        }
        assert_eq!(received.as_deref(), Some(&b"healthy-peer-survives"[..]));
        assert_eq!(rx.diagnostics().dropped_malformed_frames, 2);
        assert_eq!(rx.connections.lock().unwrap().len(), 1);
    }

    #[cfg(feature = "tcp-transport")]
    #[tokio::test]
    async fn tcp_async_accept_batch_rewakes_task_and_isolates_bad_peer() {
        use tokio::io::AsyncWriteExt;

        let p_rx = PeerId::new([0x36u8; 32]);
        let cfg = TcpTransportConfigV1 {
            max_message_bytes: 1024,
            max_active_connections: 16,
            partial_frame_timeout: Duration::from_secs(5),
            idle_connection_timeout: Duration::from_secs(30),
            max_accept_batch: 2,
        };
        let Ok(rx) = TcpTransport::bind_with_config(p_rx, "127.0.0.1:0".parse().unwrap(), cfg)
        else {
            return;
        };

        // Connect 5 sockets before calling recv(): c1..c3 idle, c4 oversized header, c5 valid!
        let mut idles = Vec::new();
        for _ in 0..3 {
            idles.push(
                tokio::net::TcpStream::connect(rx.local_addr())
                    .await
                    .unwrap(),
            );
        }
        let mut bad = tokio::net::TcpStream::connect(rx.local_addr())
            .await
            .unwrap();
        bad.write_all(&8192u32.to_le_bytes()).await.unwrap();
        bad.flush().await.unwrap();

        let mut c5 = tokio::net::TcpStream::connect(rx.local_addr())
            .await
            .unwrap();
        let valid_frame = encode_length_prefixed_frame(b"fifth-beyond-batch-2", 1024).unwrap();
        c5.write_all(&valid_frame).await.unwrap();
        c5.flush().await.unwrap();

        let msg = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("exhausted accept batch must re-wake until 5th socket is accepted")
            .expect("bad peer frame must isolate rather than fail shared async recv");
        assert_eq!(msg.payload, b"fifth-beyond-batch-2");
        assert!(rx.diagnostics().dropped_malformed_frames >= 1);
        drop(idles);
    }

    #[test]
    fn tcp_encode_frame_and_bind_config_reject_zero_and_oversized() {
        assert!(matches!(
            encode_length_prefixed_frame(b"", 64),
            Err(Error::InvalidMessage)
        ));
        assert!(matches!(
            encode_length_prefixed_frame(b"12345", 4),
            Err(Error::InvalidMessage)
        ));
        let zero_cfg = TcpTransportConfigV1 {
            max_message_bytes: 0,
            ..TcpTransportConfigV1::default()
        };
        assert!(matches!(
            TcpTransport::bind_with_config(
                PeerId::new([0x37; 32]),
                "127.0.0.1:0".parse().unwrap(),
                zero_cfg,
            ),
            Err(Error::InvalidMessage)
        ));
        if let Some(over) = (u32::MAX as usize).checked_add(1) {
            let over_cfg = TcpTransportConfigV1 {
                max_message_bytes: over,
                ..TcpTransportConfigV1::default()
            };
            assert!(matches!(
                TcpTransport::bind_with_config(
                    PeerId::new([0x38; 32]),
                    "127.0.0.1:0".parse().unwrap(),
                    over_cfg,
                ),
                Err(Error::InvalidMessage)
            ));
        }
    }

    #[test]
    fn tcp_slowloris_continuous_dripping_does_not_reset_partial_frame_and_priority_tiers() {
        let Ok(listener) = TcpListener::bind("127.0.0.1:0") else {
            return;
        };
        let listen_addr = listener.local_addr().unwrap();
        let make_conn_pair = |port_tag: u16, now: Instant| -> (TcpStream, ActiveConnection) {
            let client = TcpStream::connect(listen_addr).unwrap();
            let (server, _) = listener.accept().unwrap();
            let addr: SocketAddr = format!("127.0.0.1:{port_tag}").parse().unwrap();
            (
                client,
                ActiveConnection::new(OsTcpStreamSlot::Std(server), addr, 1024, now),
            )
        };

        let t0 = Instant::now();
        let cfg = TcpTransportConfigV1 {
            max_message_bytes: 1024,
            max_active_connections: 3,
            partial_frame_timeout: Duration::from_millis(50),
            idle_connection_timeout: Duration::from_millis(100),
            max_accept_batch: 16,
        };
        let audit = TransportAuditBufferV1::new(PeerId::new([0x39; 32]));

        // Verify un-resettable partial_frame_started_at: dripping bytes at t0+10, t0+35, t0+55
        // updates last_progress_at but STILL expires at t0+60 >= t0+10+50!
        let (_c_drip, mut drip_conn) = make_conn_pair(5001, t0);
        let mut cursor_1 = PartialChunkReader {
            chunks: VecDeque::from(vec![vec![0x10u8]]),
        };
        assert!(drip_conn
            .reader_state
            .try_read_frame_with_progress(&mut cursor_1)
            .0
            .unwrap()
            .is_none());
        drip_conn.record_read_progress(1, false, t0 + Duration::from_millis(10));
        let started = drip_conn.partial_frame_started_at.unwrap();
        assert_eq!(started, t0 + Duration::from_millis(10));

        let mut cursor_2 = PartialChunkReader {
            chunks: VecDeque::from(vec![vec![0x00u8]]),
        };
        assert!(drip_conn
            .reader_state
            .try_read_frame_with_progress(&mut cursor_2)
            .0
            .unwrap()
            .is_none());
        drip_conn.record_read_progress(1, false, t0 + Duration::from_millis(35));
        assert_eq!(drip_conn.partial_frame_started_at, Some(started));
        assert_eq!(drip_conn.last_progress_at, t0 + Duration::from_millis(35));

        let mut conns = vec![drip_conn];
        TcpTransport::sweep_expired_connections(
            &mut conns,
            &cfg,
            t0 + Duration::from_millis(55),
            &audit,
        );
        assert_eq!(
            conns.len(),
            1,
            "not yet expired at t0+55 (45ms elapsed since start)"
        );

        TcpTransport::sweep_expired_connections(
            &mut conns,
            &cfg,
            t0 + Duration::from_millis(60),
            &audit,
        );
        assert_eq!(
            conns.len(),
            0,
            "continuous dripping must NOT reset un-resettable partial_frame_timeout"
        );
        assert_eq!(audit.snapshot().evicted_stalled_partial_connections, 1);

        // Verify 3-tier priority eviction order at capacity=3:
        // Tier 1: !is_idle && completed==0  >  Tier 2: !is_idle && completed>0  >  Tier 3: oldest idle
        let t1 = Instant::now();
        let (_c1, mut auth_idle) = make_conn_pair(5010, t1);
        auth_idle.completed_frames = 2;
        auth_idle.last_progress_at = t1 + Duration::from_millis(1);

        let (_c2, mut auth_partial) = make_conn_pair(5011, t1 + Duration::from_millis(2));
        auth_partial.completed_frames = 1;
        let mut p2 = PartialChunkReader {
            chunks: VecDeque::from(vec![vec![0x08u8]]),
        };
        let _ = auth_partial
            .reader_state
            .try_read_frame_with_progress(&mut p2)
            .0
            .unwrap();
        auth_partial.record_read_progress(1, false, t1 + Duration::from_millis(5));

        let (_c3, mut unauth_partial) = make_conn_pair(5012, t1 + Duration::from_millis(3));
        let mut p1 = PartialChunkReader {
            chunks: VecDeque::from(vec![vec![0x08u8]]),
        };
        let _ = unauth_partial
            .reader_state
            .try_read_frame_with_progress(&mut p1)
            .0
            .unwrap();
        unauth_partial.record_read_progress(1, false, t1 + Duration::from_millis(6));

        let mut cap_conns = vec![auth_idle, auth_partial, unauth_partial];
        let (_c4, incoming_1) = make_conn_pair(5013, t1 + Duration::from_millis(7));
        TcpTransport::push_bounded_connection(
            &mut cap_conns,
            incoming_1,
            &cfg,
            t1 + Duration::from_millis(8),
            &audit,
        );
        // Tier 1 (port 5012, unauth partial) must be evicted first!
        assert!(cap_conns.iter().all(|c| c.remote_addr.port() != 5012));
        assert!(cap_conns.iter().any(|c| c.remote_addr.port() == 5010));
        assert!(cap_conns.iter().any(|c| c.remote_addr.port() == 5011));

        // Next incoming evicts Tier 2 (port 5011, auth partial) before honest idle (5010)!
        let (_c5, incoming_2) = make_conn_pair(5014, t1 + Duration::from_millis(9));
        TcpTransport::push_bounded_connection(
            &mut cap_conns,
            incoming_2,
            &cfg,
            t1 + Duration::from_millis(10),
            &audit,
        );
        assert!(cap_conns.iter().all(|c| c.remote_addr.port() != 5011));
        assert!(cap_conns.iter().any(|c| c.remote_addr.port() == 5010));

        // Next incoming when all remaining sockets are idle evicts oldest idle (5010)!
        let (_c6, incoming_3) = make_conn_pair(5015, t1 + Duration::from_millis(11));
        TcpTransport::push_bounded_connection(
            &mut cap_conns,
            incoming_3,
            &cfg,
            t1 + Duration::from_millis(12),
            &audit,
        );
        assert!(cap_conns.iter().all(|c| c.remote_addr.port() != 5010));
        assert_eq!(audit.snapshot().evicted_idle_connections, 1);

        // Verify idle_connection_timeout sweep evicts all remaining idle connections and audits!
        TcpTransport::sweep_expired_connections(
            &mut cap_conns,
            &cfg,
            t1 + Duration::from_millis(150),
            &audit,
        );
        assert_eq!(cap_conns.len(), 0);
        assert_eq!(audit.snapshot().evicted_idle_connections, 4);
    }

    #[test]
    fn tcp_accept_error_classifier_distinguishes_transient_handshake_aborts_from_fatal() {
        let p_rx = PeerId::new([0x38u8; 32]);
        let hub = InMemoryTcpStreamHub::new();
        let rx = TcpTransport::bind_in_memory(p_rx, "127.0.0.1:41990".parse().unwrap(), 1024, hub)
            .unwrap();

        assert!(rx.classify_tcp_accept_error(&io::Error::new(
            io::ErrorKind::ConnectionReset,
            "handshake aborted"
        )));
        assert!(rx.classify_tcp_accept_error(&io::Error::new(
            io::ErrorKind::ConnectionAborted,
            "peer dropped SYN"
        )));
        assert!(rx.classify_tcp_accept_error(&io::Error::new(io::ErrorKind::Interrupted, "EINTR")));
        assert!(!rx.classify_tcp_accept_error(&io::Error::new(
            io::ErrorKind::PermissionDenied,
            "listener permission failure"
        )));
        assert_eq!(rx.diagnostics().dropped_connection_errors, 4);
    }

    #[test]
    fn tcp_poll_recv_once_caps_std_accept_batch() {
        let p_rx = PeerId::new([0x39u8; 32]);
        let cfg = TcpTransportConfigV1 {
            max_message_bytes: 1024,
            max_active_connections: 8,
            partial_frame_timeout: Duration::from_secs(5),
            idle_connection_timeout: Duration::from_secs(30),
            max_accept_batch: 2,
        };
        let Ok(rx) = TcpTransport::bind_with_config(p_rx, "127.0.0.1:0".parse().unwrap(), cfg)
        else {
            return;
        };

        let mut clients = Vec::new();
        for _ in 0..4 {
            let Ok(c) = TcpStream::connect(rx.local_addr()) else {
                return;
            };
            clients.push(c);
        }

        // First poll must accept at most max_accept_batch (=2) sockets.
        assert!(rx.poll_recv_once().unwrap().is_none());
        let after_first = rx.connections.lock().unwrap().len();
        assert!(
            after_first <= 2,
            "expected at most 2 accepted sockets per poll, got {after_first}"
        );

        // Subsequent polls drain all 4 sockets and deliver a frame from the 4th socket.
        let frame = encode_length_prefixed_frame(b"std-batch-4", 1024).unwrap();
        clients[3].write_all(&frame).unwrap();
        clients[3].flush().unwrap();
        let mut got = None;
        for _ in 0..30 {
            if let Ok(Some(msg)) = rx.poll_recv_once() {
                got = Some(msg.payload);
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(got.as_deref(), Some(&b"std-batch-4"[..]));
        assert_eq!(rx.connections.lock().unwrap().len(), 4);
        drop(clients);
    }

    #[cfg(feature = "tcp-transport")]
    #[tokio::test]
    async fn tcp_poll_recv_once_caps_tokio_accept_batch_and_async_evicts_slowloris_before_idle() {
        use tokio::io::AsyncWriteExt;

        // Part A: Promote listener to Tokio via `recv()`, then verify `poll_recv_once()` caps Tokio batch=2.
        let p_rx_a = PeerId::new([0x3Au8; 32]);
        let cfg_a = TcpTransportConfigV1 {
            max_message_bytes: 1024,
            max_active_connections: 8,
            partial_frame_timeout: Duration::from_secs(5),
            idle_connection_timeout: Duration::from_secs(30),
            max_accept_batch: 2,
        };
        let Ok(rx_a) =
            TcpTransport::bind_with_config(p_rx_a, "127.0.0.1:0".parse().unwrap(), cfg_a)
        else {
            return;
        };
        // Trigger listener promotion `OsTcpListenerSlot::Std -> Tokio` inside Tokio runtime.
        let _ = tokio::time::timeout(Duration::from_millis(5), rx_a.recv()).await;

        let mut async_clients = Vec::new();
        for _ in 0..4 {
            async_clients.push(
                tokio::net::TcpStream::connect(rx_a.local_addr())
                    .await
                    .unwrap(),
            );
        }
        assert!(rx_a.poll_recv_once().unwrap().is_none());
        let first_batch = rx_a.connections.lock().unwrap().len();
        assert!(
            first_batch <= 2,
            "promoted Tokio poll_recv_once must cap accept batch at 2, got {first_batch}"
        );
        let frame_a = encode_length_prefixed_frame(b"tokio-batch-4", 1024).unwrap();
        async_clients[3].write_all(&frame_a).await.unwrap();
        async_clients[3].flush().await.unwrap();
        let mut got_a = None;
        for _ in 0..30 {
            if let Ok(Some(msg)) = rx_a.poll_recv_once() {
                got_a = Some(msg.payload);
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(got_a.as_deref(), Some(&b"tokio-batch-4"[..]));
        assert_eq!(rx_a.connections.lock().unwrap().len(), 4);
        drop(async_clients);

        // Part B: Verify `poll_recv_os_async` evicts Slowloris partial-frame peer before honest idle.
        let p_rx_b = PeerId::new([0x3Bu8; 32]);
        let cfg_b = TcpTransportConfigV1 {
            max_message_bytes: 1024,
            max_active_connections: 2,
            partial_frame_timeout: Duration::from_secs(5),
            idle_connection_timeout: Duration::from_secs(30),
            max_accept_batch: 8,
        };
        let Ok(rx_b) =
            TcpTransport::bind_with_config(p_rx_b, "127.0.0.1:0".parse().unwrap(), cfg_b)
        else {
            return;
        };

        let mut honest_idle = tokio::net::TcpStream::connect(rx_b.local_addr())
            .await
            .unwrap();
        let mut slowloris = tokio::net::TcpStream::connect(rx_b.local_addr())
            .await
            .unwrap();
        slowloris.write_all(&[0x40]).await.unwrap();
        slowloris.flush().await.unwrap();

        for _ in 0..20 {
            let _ = tokio::time::timeout(Duration::from_millis(10), rx_b.recv()).await;
            let ready = {
                let conns = rx_b.connections.lock().unwrap();
                conns.len() == 2 && conns.iter().any(|c| !c.reader_state.is_idle())
            };
            if ready {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        let mut incoming = tokio::net::TcpStream::connect(rx_b.local_addr())
            .await
            .unwrap();
        let f_in = encode_length_prefixed_frame(b"async-incoming", 1024).unwrap();
        incoming.write_all(&f_in).await.unwrap();
        incoming.flush().await.unwrap();

        let rx_msg1 = tokio::time::timeout(Duration::from_secs(2), rx_b.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(rx_msg1.payload, b"async-incoming");

        let f_idle = encode_length_prefixed_frame(b"async-idle-survived", 1024).unwrap();
        honest_idle.write_all(&f_idle).await.unwrap();
        honest_idle.flush().await.unwrap();

        let rx_msg2 = tokio::time::timeout(Duration::from_secs(2), rx_b.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(rx_msg2.payload, b"async-idle-survived");
        assert!(rx_b.diagnostics().evicted_stalled_partial_connections >= 1);
        drop(slowloris);
    }
}
