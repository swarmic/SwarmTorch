//! Transport traits and types (`ADR-0022`, `W9-04`).
//!
//! Enforces Cargo feature additivity by separating:
//! - [`PacketTransport`]: always available (`no_std`, zero-allocation, caller-buffered,
//!   keyed by observed source `Endpoint` rather than unauthenticated `PeerId`).
//! - [`SwarmTransport`]: available under `alloc` (and unchanged by `std`), using
//!   named associated future types without `async-trait`.

#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use core::future::Future;

#[cfg(feature = "alloc")]
use crate::Result;
#[cfg(feature = "alloc")]
use swarm_torch_core::traits::PeerId;

/// Reliability classification for transports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReliabilityClass {
    /// No delivery guarantees (LoRa, BLE).
    BestEffort,
    /// At least once delivery with retries (UDP with ACK).
    AtLeastOnce,
    /// Reliable ordered delivery (TCP, QUIC).
    Reliable,
}

/// Bandwidth classification for transports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthClass {
    /// < 10 kbps (LoRa).
    UltraLow,
    /// 10 - 1000 kbps (BLE).
    Low,
    /// 1 - 100 Mbps (WiFi).
    Medium,
    /// > 100 Mbps (Ethernet, datacenter).
    High,
}

/// Capabilities reported by a high-level message transport.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportCapabilities {
    /// Reliability class of the transport.
    pub reliability: ReliabilityClass,
    /// Bandwidth class of the transport.
    pub bandwidth_class: BandwidthClass,
    /// Maximum reassembled message size in bytes.
    pub max_message_size: usize,
    /// Whether the transport supports multicast/broadcast.
    pub supports_multicast: bool,
}

/// Capabilities reported by a low-level [`PacketTransport`] (`ADR-0022` §1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketCapabilities {
    /// Reliability class of the underlying radio/link.
    pub reliability: ReliabilityClass,
    /// Bandwidth class of the underlying radio/link.
    pub bandwidth_class: BandwidthClass,
    /// Maximum codec-visible frame payload in bytes (after lower-layer link headers).
    pub max_frame_bytes: usize,
    /// Maximum reassembled application delivery in bytes.
    pub max_message_bytes: usize,
    /// Whether the link supports multicast/broadcast frames.
    pub supports_multicast: bool,
    /// Whether the link provides a cryptographically verified carrier principal.
    pub has_verified_link_principal: bool,
}

/// Metadata for a single frame received into a caller-supplied buffer via [`PacketTransport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceivedFrame<E> {
    /// Observed link/socket source endpoint (never conflated with authenticated `PeerId`).
    pub source: E,
    /// Number of valid frame bytes written into the caller's receive buffer.
    pub len: usize,
}

/// Non-blocking frame transmission error for [`PacketTransport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PacketSendError<E> {
    /// Transmit queue or radio is busy; frame was not accepted.
    WouldBlock,
    /// Frame exceeds `capabilities().max_frame_bytes`; rejected before transmission.
    FrameTooLarge { len: usize, max: usize },
    /// Underlying driver/link error.
    Transport(E),
}

/// Non-blocking frame receive error for [`PacketTransport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PacketRecvError<E> {
    /// Caller-supplied buffer is smaller than the incoming frame; frame was not truncated.
    BufferTooSmall { required: usize, available: usize },
    /// Incoming frame was truncated at the link layer.
    TruncatedFrame,
    /// Underlying driver/link error.
    Transport(E),
}

/// Allocation-free, non-blocking, caller-buffered packet transport trait (`ADR-0022` §1.2).
///
/// Always available in `no_std`, `alloc`, and `std` builds with an identical signature.
pub trait PacketTransport {
    /// Observed link/radio endpoint address type.
    type Endpoint;
    /// Driver-specific error type.
    type Error: core::fmt::Debug;

    /// Attempt to enqueue or transmit a single frame without blocking.
    fn try_send_frame(
        &mut self,
        destination: &Self::Endpoint,
        frame: &[u8],
    ) -> core::result::Result<(), PacketSendError<Self::Error>>;

    /// Attempt to receive a single frame into `buffer` without blocking.
    ///
    /// Returns `Ok(None)` when no frame is ready (`WouldBlock` equivalent on receive).
    fn try_recv_frame(
        &mut self,
        buffer: &mut [u8],
    ) -> core::result::Result<Option<ReceivedFrame<Self::Endpoint>>, PacketRecvError<Self::Error>>;

    /// Report static link/frame capabilities.
    fn capabilities(&self) -> PacketCapabilities;
}

/// Statistics from a broadcast operation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BroadcastStats {
    /// Number of peers the message was sent to.
    pub peers_sent: usize,
    /// Number of confirmed deliveries (if applicable).
    pub confirmed: usize,
    /// Number of failed sends.
    pub failed: usize,
}

/// Unauthenticated observed transport-level source locator (`ST-06`).
///
/// Represents a network or link endpoint observed directly at the transport layer and
/// MUST NEVER be treated as a verified canonical [`PeerId`]. Downstream protocol layers
/// must authenticate every payload cryptographically (or via a sealed carrier proof)
/// before attributing identity.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObservedSourceRef {
    /// IPv4 socket endpoint `(octets, port)`.
    Ipv4 {
        /// IPv4 octets.
        addr: [u8; 4],
        /// UDP/TCP port.
        port: u16,
    },
    /// IPv6 socket endpoint `(octets, port)`.
    Ipv6 {
        /// IPv6 octets.
        addr: [u8; 16],
        /// UDP/TCP port.
        port: u16,
    },
    /// Named transport or hub link hint (e.g. mock queue, topic, or local link handle).
    LinkLabel(String),
    /// Transport backend did not expose an observable source locator.
    Unspecified,
}

#[cfg(feature = "std")]
impl From<std::net::SocketAddr> for ObservedSourceRef {
    fn from(addr: std::net::SocketAddr) -> Self {
        match addr {
            std::net::SocketAddr::V4(v4) => Self::Ipv4 {
                addr: v4.ip().octets(),
                port: v4.port(),
            },
            std::net::SocketAddr::V6(v6) => Self::Ipv6 {
                addr: v6.ip().octets(),
                port: v6.port(),
            },
        }
    }
}

/// Complete message received by [`SwarmTransport::recv`] paired strictly with its
/// unauthenticated [`ObservedSourceRef`] (`ST-06`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedTransportMessage {
    /// Observed transport source hint (unauthenticated; NEVER a trusted `PeerId`).
    pub observed_source: ObservedSourceRef,
    /// Raw wire payload bytes awaiting downstream cryptographic verification.
    pub payload: Vec<u8>,
}

/// Asynchronous complete-message transport trait (`ADR-0022` §1.3, `ST-06`).
///
/// Available whenever `alloc` is enabled (and unchanged when `std` is enabled),
/// using Generic Associated Types instead of `async-trait`.
#[cfg(feature = "alloc")]
pub trait SwarmTransport: Send + Sync {
    /// Future returned by [`SwarmTransport::send`].
    type SendFuture<'a>: Future<Output = Result<()>> + Send + 'a
    where
        Self: 'a;
    /// Future returned by [`SwarmTransport::recv`].
    type RecvFuture<'a>: Future<Output = Result<ObservedTransportMessage>> + Send + 'a
    where
        Self: 'a;
    /// Future returned by [`SwarmTransport::broadcast`].
    type BroadcastFuture<'a>: Future<Output = Result<BroadcastStats>> + Send + 'a
    where
        Self: 'a;
    /// Future returned by [`SwarmTransport::discover`].
    type DiscoverFuture<'a>: Future<Output = Result<Vec<PeerId>>> + Send + 'a
    where
        Self: 'a;

    /// Send a message to a specific peer.
    fn send<'a>(&'a self, peer: PeerId, msg: &'a [u8]) -> Self::SendFuture<'a>;

    /// Receive the next message.
    fn recv(&self) -> Self::RecvFuture<'_>;

    /// Broadcast a message to all known peers (best-effort).
    fn broadcast<'a>(&'a self, msg: &'a [u8]) -> Self::BroadcastFuture<'a>;

    /// Discover peers on the network.
    fn discover(&self) -> Self::DiscoverFuture<'_>;

    /// Get transport capabilities.
    fn capabilities(&self) -> TransportCapabilities;
}

/// Priority level for multi-transport routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Priority(pub u8);

impl Priority {
    /// Highest priority (use first).
    pub const HIGH: Self = Self(0);
    /// Normal priority.
    pub const NORMAL: Self = Self(128);
    /// Low priority (fallback).
    pub const LOW: Self = Self(255);
}

/// Fallback policy for multi-transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FallbackPolicy {
    /// Prefer the transport with lowest latency.
    PreferLowLatency,
    /// Prefer the transport with highest reliability.
    PreferReliability,
    /// Try all transports in priority order.
    #[default]
    PriorityOrder,
    /// Try the transport with best battery efficiency.
    PreferPowerEfficient,
}
