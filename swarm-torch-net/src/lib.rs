//! # SwarmTorch Network
//!
//! Network transport abstractions for SwarmTorch.
//!
//! This crate provides:
//! - Unified `SwarmTransport` trait for all transports
//! - TCP/UDP implementations for server/edge
//! - BLE/LoRa/WiFi implementations for embedded
//! - Multi-transport support with fallback
//! - Message framing and serialization

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod protocol;
pub mod traits;

// TCP transport - placeholder
// #[cfg(feature = "tcp-transport")]
// pub mod tcp;

// UDP transport - placeholder
// #[cfg(feature = "udp-transport")]
// pub mod udp;

mod mock;
pub use mock::MockTransport;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::protocol::*;
    pub use crate::traits::*;
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
