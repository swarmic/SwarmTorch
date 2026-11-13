//! # SwarmTorch Network
//!
//! Network transport abstractions for SwarmTorch.
//!
//! This crate provides:
//! - Unified `SwarmTransport` trait for all transports
//! - Originator-authenticated gossip inner updates and recipient wrappers (`gossip`)
//! - Mock transport/network implementations for integration testing
//! - Placeholder feature flags for TCP/UDP/BLE/LoRa/WiFi backends (planned)
//! - Multi-transport policy surface with fallback error semantics
//! - Message framing and serialization

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod carrier;
pub mod chunked;
pub mod gossip;
pub mod protocol;
#[cfg(feature = "std")]
pub mod tcp;
pub mod traits;
#[cfg(feature = "std")]
pub mod udp;

mod mock;
pub use mock::MockTransport;
#[cfg(feature = "alloc")]
pub use mock::{MockNetwork, MockTransportError};
#[cfg(feature = "std")]
pub use mock::{MockNetworkHub, MockPeerTransport, Topology};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::carrier::*;
    pub use crate::chunked::*;
    pub use crate::gossip::*;
    pub use crate::protocol::*;
    #[cfg(feature = "std")]
    pub use crate::tcp::*;
    pub use crate::traits::*;
    #[cfg(feature = "std")]
    pub use crate::udp::*;
}

/// Result type for network operations
pub type Result<T> = core::result::Result<T, Error>;

/// Network error types
#[derive(Debug)]
pub enum Error {
    /// Connection failed
    ConnectionFailed,
    /// Send failed
    SendFailed,
    /// Receive failed
    ReceiveFailed,
    /// Timeout
    Timeout,
    /// Peer not found
    PeerNotFound,
    /// All transports failed
    AllTransportsFailed,
    /// Serialization error
    Serialization,
    /// Transport not available
    TransportUnavailable,
    /// Invalid message format
    InvalidMessage,
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::ConnectionFailed => write!(f, "connection failed"),
            Error::SendFailed => write!(f, "send failed"),
            Error::ReceiveFailed => write!(f, "receive failed"),
            Error::Timeout => write!(f, "timeout"),
            Error::PeerNotFound => write!(f, "peer not found"),
            Error::AllTransportsFailed => write!(f, "all transports failed"),
            Error::Serialization => write!(f, "serialization error"),
            Error::TransportUnavailable => write!(f, "transport unavailable"),
            Error::InvalidMessage => write!(f, "invalid message format"),
        }
    }
}
