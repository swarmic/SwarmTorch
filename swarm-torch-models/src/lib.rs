//! # SwarmTorch Models
//!
//! Model utilities and backend integrations for SwarmTorch.
//!
//! This crate provides:
//! - Integration with Burn ML framework
//! - Model serialization/deserialization
//! - Reference model implementations
//! - ONNX import/export utilities
//!
//! ## Feature Flags
//!
//! - `burn` (default): Burn framework integration
//! - `tch`: PyTorch interop via tch-rs (requires libtorch)

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "burn")]
pub mod burn_integration;

pub mod simple;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::simple::*;

    #[cfg(feature = "burn")]
    pub use crate::burn_integration::*;
}

/// Model state for serialization
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelState {
    /// Model name/identifier
    pub name: alloc::string::String,
    /// Version of the model format
    pub version: u32,
    /// Flattened parameters
    pub parameters: alloc::vec::Vec<f32>,
    /// Parameter shapes for reconstruction
    pub shapes: alloc::vec::Vec<alloc::vec::Vec<usize>>,
}

#[cfg(feature = "alloc")]
impl ModelState {
    /// Create a new model state
    pub fn new(name: impl Into<alloc::string::String>, parameters: alloc::vec::Vec<f32>) -> Self {
        Self {
            name: name.into(),
            version: 1,
            parameters,
            shapes: alloc::vec::Vec::new(),
        }
    }

    /// Add parameter shape information
    pub fn with_shapes(mut self, shapes: alloc::vec::Vec<alloc::vec::Vec<usize>>) -> Self {
        self.shapes = shapes;
        self
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<alloc::vec::Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(bytes)
    }
}
