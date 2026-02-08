//! Cryptographic utilities for authentication and verification
//!
//! This module provides Ed25519 signatures and message authentication.

use crate::traits::PeerId;

/// Key pair for signing messages
#[derive(Clone)]
pub struct KeyPair {
    /// Secret key bytes (32 bytes)
    secret: [u8; 32],
    /// Public key bytes (32 bytes)
    pub public: [u8; 32],
}

impl KeyPair {
    /// Generate a new key pair from random bytes
    ///
    /// # Safety
    /// The caller must ensure the seed is cryptographically random
    pub fn from_seed(seed: [u8; 32]) -> Self {
        // In a real implementation, this would use ed25519-dalek
        // For now, we just store the seed as secret and derive public key
        let mut public = [0u8; 32];
        // Simple derivation for placeholder (NOT SECURE - use ed25519-dalek in production)
        for (i, &b) in seed.iter().enumerate() {
            public[i] = b.wrapping_add(i as u8);
        }
        Self {
            secret: seed,
            public,
        }
    }

    /// Get the peer ID derived from this key pair
    pub fn peer_id(&self) -> PeerId {
        #[cfg(feature = "std")]
        {
            PeerId::from_public_key(&self.public)
        }
        #[cfg(not(feature = "std"))]
        {
            // Simple hash for no_std
            let mut id = [0u8; 32];
            id.copy_from_slice(&self.public);
            PeerId::new(id)
        }
    }

    /// Get the public key bytes
    pub fn public_key(&self) -> &[u8; 32] {
        &self.public
    }
}

/// Signature bytes (64 bytes for Ed25519)
#[derive(Debug, Clone, Copy)]
pub struct Signature(pub [u8; 64]);

impl Signature {
    /// Create from bytes
    pub const fn from_bytes(bytes: [u8; 64]) -> Self {
        Self(bytes)
    }

    /// Get the signature bytes
    pub const fn as_bytes(&self) -> &[u8; 64] {
        &self.0
    }
}

/// Message authentication helper
pub struct MessageAuth {
    key_pair: KeyPair,
}

impl MessageAuth {
    /// Create a new message authenticator
    pub fn new(key_pair: KeyPair) -> Self {
        Self { key_pair }
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> Signature {
        // Placeholder - in production use ed25519-dalek
        let mut sig = [0u8; 64];
        for (i, &b) in message.iter().take(32).enumerate() {
            sig[i] = b ^ self.key_pair.secret[i % 32];
        }
        Signature(sig)
    }

    /// Verify a signature
    pub fn verify(public_key: &[u8; 32], message: &[u8], signature: &Signature) -> bool {
        // Placeholder verification - in production use ed25519-dalek
        let expected = {
            let mut sig = [0u8; 64];
            for (i, &b) in message.iter().take(32).enumerate() {
                sig[i] = b ^ public_key[i % 32].wrapping_add(i as u8);
            }
            sig
        };
        // This is NOT a real verification - just a placeholder structure
        signature.0[0..8] == expected[0..8]
    }

    /// Get the key pair
    pub fn key_pair(&self) -> &KeyPair {
        &self.key_pair
    }
}

/// Configuration for security features
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Require message signatures
    pub require_signatures: bool,
    /// Encrypt network traffic
    pub encrypt_transport: bool,
    /// Validate gradient bounds
    pub validate_gradients: bool,
    /// Maximum clock skew for replay protection (seconds)
    pub max_clock_skew_secs: u32,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            require_signatures: true,
            encrypt_transport: true,
            validate_gradients: true,
            max_clock_skew_secs: 60,
        }
    }
}

/// Gradient validator for bounds checking
#[derive(Debug, Clone)]
pub struct GradientValidator {
    /// Maximum L2 norm for gradient updates
    pub max_gradient_norm: f32,
    /// Maximum absolute value for any coordinate
    pub max_coordinate_value: f32,
}

impl Default for GradientValidator {
    fn default() -> Self {
        Self {
            max_gradient_norm: 10.0,
            max_coordinate_value: 100.0,
        }
    }
}

impl GradientValidator {
    /// Validate a gradient vector
    pub fn validate(&self, gradients: &[f32]) -> Result<(), GradientValidationError> {
        // Check for NaN/Inf
        for (i, &g) in gradients.iter().enumerate() {
            if g.is_nan() {
                return Err(GradientValidationError::NaN { index: i });
            }
            if g.is_infinite() {
                return Err(GradientValidationError::Infinite { index: i });
            }
            if g.abs() > self.max_coordinate_value {
                return Err(GradientValidationError::CoordinateTooLarge {
                    index: i,
                    value: g,
                    max: self.max_coordinate_value,
                });
            }
        }

        // Check L2 norm
        let norm_sq: f32 = gradients.iter().map(|g| g * g).sum();
        let norm = norm_sq.sqrt();
        if norm > self.max_gradient_norm {
            return Err(GradientValidationError::NormTooLarge {
                norm,
                max: self.max_gradient_norm,
            });
        }

        Ok(())
    }
}

/// Gradient validation error
#[derive(Debug, Clone)]
pub enum GradientValidationError {
    /// NaN value found
    NaN { index: usize },
    /// Infinite value found
    Infinite { index: usize },
    /// Coordinate value too large
    CoordinateTooLarge { index: usize, value: f32, max: f32 },
    /// Gradient norm too large
    NormTooLarge { norm: f32, max: f32 },
}
