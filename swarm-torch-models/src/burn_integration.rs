//! Burn framework integration
//!
//! This module provides wrappers for using Burn models with SwarmTorch.

// Placeholder for Burn integration
// Real implementation would wrap burn::module::Module

/// Marker trait for Burn-compatible models
pub trait BurnCompatible {}

/// Wrapper for Burn models in SwarmTorch
#[derive(Debug, Clone)]
pub struct BurnModelWrapper<M> {
    /// The underlying Burn model
    pub model: M,
}

impl<M> BurnModelWrapper<M> {
    /// Wrap a Burn model for use with SwarmTorch
    pub fn new(model: M) -> Self {
        Self { model }
    }

    /// Get the underlying model
    pub fn into_inner(self) -> M {
        self.model
    }
}
