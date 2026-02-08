//! # SwarmTorch Core
//!
//! Core swarm learning algorithms and primitives for SwarmTorch.
//!
//! This crate is `no_std` compatible and provides:
//! - Swarm optimization algorithms (PSO, ACO, Firefly)
//! - Robust aggregation (Krum, Bulyan, Trimmed Mean)
//! - Core traits and abstractions
//! - Gradient compression utilities
//!
//! ## Feature Flags
//!
//! - `std` (default): Enable standard library support
//! - `alloc`: Enable allocator for dynamic memory (included with `std`)
//! - `robust-aggregation`: Enable all robust aggregators
//! - `telemetry`: Enable tracing-based telemetry
//! - `defmt`: Enable defmt logging for embedded

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod aggregation;
pub mod algorithms;
pub mod compression;
pub mod consensus;
pub mod crypto;
pub mod identity;
pub mod traits;

#[cfg(feature = "telemetry")]
pub mod telemetry;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::aggregation::*;
    pub use crate::algorithms::*;
    pub use crate::traits::*;
}

/// Result type for SwarmTorch operations
pub type Result<T> = core::result::Result<T, Error>;

/// Error type for SwarmTorch core operations
#[derive(Debug)]
pub enum Error {
    /// Serialization/deserialization error
    Serialization,
    /// Cryptographic verification failed
    VerificationFailed,
    /// Gradient validation failed
    InvalidGradient,
    /// Aggregation failed
    AggregationFailed,
    /// Insufficient updates for aggregation
    InsufficientUpdates,
    /// Resource limit exceeded
    ResourceExhausted,
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::Serialization => write!(f, "serialization error"),
            Error::VerificationFailed => write!(f, "cryptographic verification failed"),
            Error::InvalidGradient => write!(f, "gradient validation failed"),
            Error::AggregationFailed => write!(f, "aggregation failed"),
            Error::InsufficientUpdates => write!(f, "insufficient updates for aggregation"),
            Error::ResourceExhausted => write!(f, "resource limit exceeded"),
        }
    }
}
