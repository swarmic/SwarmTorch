//! # SwarmTorch Models
//!
//! Reference models, analytical loss functions, and optimizers for SwarmTorch (`ADR-0024`).
//!
//! This crate is `no_std`-compatible by default and provides:
//! - Contiguous-parameter reference models ([`simple::LinearModel`], [`simple::MlpModel`])
//! - Analytical loss functions ([`loss::MseLoss`], [`loss::CrossEntropyLoss`], [`loss::HuberLoss`], [`loss::FedProxRegularizer`])
//! - Reference optimizers ([`optim::SgdOptimizer`])
//! - Model state serialization ([`ModelState`], requires `alloc`)

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
pub mod backend_bridge;
#[cfg(feature = "alloc")]
pub mod lora;
pub mod loss;
pub mod mlp;
pub mod optim;
#[cfg(feature = "alloc")]
pub mod safetensors_v1;
pub mod simple;

/// Prelude for convenient imports
pub mod prelude {
    #[cfg(feature = "alloc")]
    pub use crate::backend_bridge::*;
    #[cfg(feature = "alloc")]
    pub use crate::lora::*;
    pub use crate::loss::*;
    pub use crate::optim::*;
    #[cfg(feature = "alloc")]
    pub use crate::safetensors_v1::*;
    pub use crate::simple::*;
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
