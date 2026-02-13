//! Cryptographic utilities for authentication and verification
//!
//! This module provides Ed25519 signatures and message authentication.

use crate::traits::PeerId;
use ed25519_dalek::{Signature as DalekSignature, Signer, SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};

/// Key pair for signing messages
#[derive(Clone)]
pub struct KeyPair {
    /// Secret signing key
    secret: SigningKey,
    /// Public key bytes (32 bytes) - kept for API compatibility
    pub public: [u8; 32],
}

impl KeyPair {
    /// Generate a new key pair from random bytes
    ///
    /// # Safety
    /// The caller must ensure the seed is cryptographically random
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let secret = SigningKey::from_bytes(&seed);
        let public = secret.verifying_key().to_bytes();
        Self { secret, public }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    /// Convert to internal Dalek signature
    pub fn to_dalek(&self) -> Result<DalekSignature, VerifyError> {
        DalekSignature::from_slice(&self.0).map_err(|_| VerifyError::InvalidSignatureEncoding)
    }
}

/// Errors that can occur during verification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyError {
    /// malformed signature bytes
    InvalidSignatureEncoding,
    /// malformed public key
    InvalidPublicKey,
    /// signature verification failed
    VerificationFailed,
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

    /// Sign a message envelope's components
    ///
    /// Binds the signature to the protocol version, metadata, and payload
    pub fn sign(
        &self,
        version: (u8, u8),
        message_type: u8,
        sequence: u64,
        timestamp: u32,
        payload: &[u8],
    ) -> Signature {
        // Domain separation tag
        let tag = b"swarmtorch.envelope.v0";

        // 1. Hash the payload first
        let payload_hash = Sha256::digest(payload);

        // 2. Construct canonical preimage
        let mut hasher = Sha256::new();
        hasher.update(tag);
        hasher.update([version.0, version.1]);
        hasher.update(self.key_pair.public); // Bind to sender (self)
        hasher.update(sequence.to_le_bytes());
        hasher.update(timestamp.to_le_bytes());
        hasher.update([message_type]);
        hasher.update(payload_hash);

        let canonical = hasher.finalize();

        // 3. Sign the canonical hash
        let sig = self.key_pair.secret.sign(&canonical);
        Signature(sig.to_bytes())
    }

    /// Verify a signature against envelope components
    pub fn verify(
        public_key: &[u8; 32],
        version: (u8, u8),
        message_type: u8,
        sequence: u64,
        timestamp: u32,
        payload: &[u8],
        signature: &Signature,
    ) -> Result<(), VerifyError> {
        // Parse public key
        let key =
            VerifyingKey::from_bytes(public_key).map_err(|_| VerifyError::InvalidPublicKey)?;

        // Parse signature
        let sig = signature.to_dalek()?;

        // Reconstruct canonical preimage
        let tag = b"swarmtorch.envelope.v0";
        let payload_hash = Sha256::digest(payload);

        let mut hasher = Sha256::new();
        hasher.update(tag);
        hasher.update([version.0, version.1]);
        hasher.update(public_key);
        hasher.update(sequence.to_le_bytes());
        hasher.update(timestamp.to_le_bytes());
        hasher.update([message_type]);
        hasher.update(payload_hash);

        let canonical = hasher.finalize();

        // Strict verification
        key.verify_strict(&canonical, &sig)
            .map_err(|_| VerifyError::VerificationFailed)
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
            let abs_g = if g < 0.0 { -g } else { g };
            if abs_g > self.max_coordinate_value {
                return Err(GradientValidationError::CoordinateTooLarge {
                    index: i,
                    value: g,
                    max: self.max_coordinate_value,
                });
            }
        }

        // Check L2 norm
        let norm_sq: f32 = gradients.iter().map(|g| g * g).sum();
        let norm = sqrt_f32(norm_sq);
        if norm > self.max_gradient_norm {
            return Err(GradientValidationError::NormTooLarge {
                norm,
                max: self.max_gradient_norm,
            });
        }

        Ok(())
    }
}

#[inline]
fn sqrt_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        // Newton-Raphson iterations: good enough for validation thresholds.
        if x <= 0.0 {
            return 0.0;
        }
        let mut y = x;
        // Fixed iteration count for determinism.
        for _ in 0..8 {
            y = 0.5 * (y + (x / y));
        }
        y
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

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to change one byte in a slice
    fn tamper(bytes: &mut [u8]) {
        bytes[0] ^= 0xFF;
    }

    #[test]
    fn signature_verification_succeeds_for_valid_keypair() {
        let seed = [1u8; 32];
        let pair = KeyPair::from_seed(seed);
        let auth = MessageAuth::new(pair.clone());

        let version = (0, 1);
        let msg_type = 1;
        let seq = 100;
        let ts = 1234567890;
        let payload = b"test payload";

        let sig = auth.sign(version, msg_type, seq, ts, payload);

        assert!(
            MessageAuth::verify(&pair.public, version, msg_type, seq, ts, payload, &sig).is_ok()
        );
    }

    #[test]
    fn signature_verification_fails_for_tampered_payload() {
        let seed = [2u8; 32];
        let pair = KeyPair::from_seed(seed);
        let auth = MessageAuth::new(pair.clone());

        let version = (0, 1);
        let msg_type = 1;
        let seq = 100;
        let ts = 1234567890;
        let payload = b"test payload";

        let sig = auth.sign(version, msg_type, seq, ts, payload);

        // Tamper payload
        let mut bad_payload = payload.to_vec();
        tamper(&mut bad_payload);

        let result =
            MessageAuth::verify(&pair.public, version, msg_type, seq, ts, &bad_payload, &sig);
        assert_eq!(result, Err(VerifyError::VerificationFailed));
    }

    #[test]
    fn signature_verification_fails_for_wrong_key() {
        let seed1 = [3u8; 32];
        let pair1 = KeyPair::from_seed(seed1);
        let auth1 = MessageAuth::new(pair1);

        let seed2 = [4u8; 32];
        let pair2 = KeyPair::from_seed(seed2);

        let version = (0, 1);
        let msg_type = 1;
        let seq = 100;
        let ts = 1234567890;
        let payload = b"test payload";

        // Sign with key1
        let sig = auth1.sign(version, msg_type, seq, ts, payload);

        // Verify with key2
        let result = MessageAuth::verify(&pair2.public, version, msg_type, seq, ts, payload, &sig);
        assert_eq!(result, Err(VerifyError::VerificationFailed));
    }

    #[test]
    fn signature_is_deterministic_for_fixed_seed_and_message() {
        let seed = [5u8; 32];
        let pair1 = KeyPair::from_seed(seed);
        let auth1 = MessageAuth::new(pair1);

        // Recreate same keypair
        let pair2 = KeyPair::from_seed(seed);
        let auth2 = MessageAuth::new(pair2);

        let version = (0, 1);
        let msg_type = 1;
        let seq = 100;
        let ts = 1234567890;
        let payload = b"variable payload";

        let sig1 = auth1.sign(version, msg_type, seq, ts, payload);
        let sig2 = auth2.sign(version, msg_type, seq, ts, payload);

        assert_eq!(sig1.0, sig2.0);
    }

    #[test]
    fn verification_fails_on_modified_sequence() {
        let seed = [6u8; 32];
        let pair = KeyPair::from_seed(seed);
        let auth = MessageAuth::new(pair.clone());

        let version = (0, 1);
        let msg_type = 1;
        let seq = 100;
        let ts = 1234567890;
        let payload = b"test";

        let sig = auth.sign(version, msg_type, seq, ts, payload);

        // Modify sequence
        let result =
            MessageAuth::verify(&pair.public, version, msg_type, seq + 1, ts, payload, &sig);
        assert_eq!(result, Err(VerifyError::VerificationFailed));
    }

    #[test]
    fn verification_fails_on_modified_timestamp() {
        let seed = [7u8; 32];
        let pair = KeyPair::from_seed(seed);
        let auth = MessageAuth::new(pair.clone());

        let version = (0, 1);
        let msg_type = 1;
        let seq = 100;
        let ts = 1234567890;
        let payload = b"test";

        let sig = auth.sign(version, msg_type, seq, ts, payload);

        // Modify timestamp
        let result =
            MessageAuth::verify(&pair.public, version, msg_type, seq, ts + 1, payload, &sig);
        assert_eq!(result, Err(VerifyError::VerificationFailed));
    }

    #[test]
    fn verification_fails_on_modified_sender() {
        let seed = [8u8; 32];
        let pair = KeyPair::from_seed(seed);
        let auth = MessageAuth::new(pair.clone());

        let version = (0, 1);
        let msg_type = 1;
        let seq = 100;
        let ts = 1234567890;
        let payload = b"test";

        let sig = auth.sign(version, msg_type, seq, ts, payload);

        // Modify sender (use wrong public key)
        let mut wrong_public = pair.public;
        tamper(&mut wrong_public);

        // This might fail as InvalidPublicKey if we tamper too much, or VerificationFailed
        // if the key is structurally valid but wrong.
        // Compressed points must be on curve. A random bit flip likely makes it off-curve.
        // Let's expect either InvalidPublicKey OR VerificationFailed, or check specifically.
        // Ideally we want to test "valid key but wrong one" (authentication failure)
        // versus "invalid key bytes" (parsing failure).
        // Let's generate a valid DIFFERENT key for authentication failure test (already done above).
        // This test is specifically modifying the bytes to be potentially invalid.

        let result = MessageAuth::verify(&wrong_public, version, msg_type, seq, ts, payload, &sig);
        // It's likely InvalidPublicKey, but could be failed verification.
        // Let's assert it is an Error.
        assert!(result.is_err());
    }

    #[test]
    fn verification_fails_on_modified_message_type() {
        let seed = [9u8; 32];
        let pair = KeyPair::from_seed(seed);
        let auth = MessageAuth::new(pair.clone());

        let version = (0, 1);
        let msg_type = 1;
        let seq = 100;
        let ts = 1234567890;
        let payload = b"test";

        let sig = auth.sign(version, msg_type, seq, ts, payload);

        let result =
            MessageAuth::verify(&pair.public, version, msg_type + 1, seq, ts, payload, &sig);
        assert_eq!(result, Err(VerifyError::VerificationFailed));
    }

    #[test]
    fn verification_fails_on_modified_version() {
        let seed = [10u8; 32];
        let pair = KeyPair::from_seed(seed);
        let auth = MessageAuth::new(pair.clone());

        let version = (0, 1);
        let msg_type = 1;
        let seq = 100;
        let ts = 1234567890;
        let payload = b"test";

        let sig = auth.sign(version, msg_type, seq, ts, payload);

        // Modify version
        let result = MessageAuth::verify(&pair.public, (1, 1), msg_type, seq, ts, payload, &sig);
        assert_eq!(result, Err(VerifyError::VerificationFailed));
    }

    #[test]
    fn verification_fails_on_invalid_public_key_bytes() {
        let seed = [11u8; 32];
        let pair = KeyPair::from_seed(seed);
        let auth = MessageAuth::new(pair.clone());

        let version = (0, 1);
        let msg_type = 1;
        let seq = 100;
        let ts = 1234567890;
        let payload = b"test";
        let sig = auth.sign(version, msg_type, seq, ts, payload);

        // Use obviously invalid public key
        // [0xFF; 32] was seemingly accepted by from_bytes (likely interpreted as valid Y)
        // but rejected by verify_strict.
        // We accept either InvalidPublicKey OR VerificationFailed as acceptable fail-closed behavior.
        // Fail-closed is what matters.
        let invalid_key = [0xFFu8; 32];

        let result = MessageAuth::verify(&invalid_key, version, msg_type, seq, ts, payload, &sig);
        // Either error means we rejected it.
        assert!(matches!(
            result,
            Err(VerifyError::InvalidPublicKey) | Err(VerifyError::VerificationFailed)
        ));
    }

    #[test]
    fn invalid_signature_encoding_rejected() {
        let seed = [12u8; 32];
        let pair = KeyPair::from_seed(seed);

        let version = (0, 1);
        let msg_type = 1;
        let seq = 100;
        let ts = 1234567890;
        let payload = b"test";

        // Create a signature with invalid encoding logic (s >= L)
        // This is caught by verify_strict, returning VerificationFailed (SignatureError).
        let bad_sig = Signature([0xFF; 64]);

        let result =
            MessageAuth::verify(&pair.public, version, msg_type, seq, ts, payload, &bad_sig);
        // VerificationFailed is the correct error for s >= L, as from_bytes is infallible for [u8; 64]
        assert_eq!(result, Err(VerifyError::VerificationFailed));
    }
}
