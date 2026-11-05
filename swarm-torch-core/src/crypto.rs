//! Cryptographic utilities for authentication and verification (`ADR-0008`, `ADR-0021`, `W9-02`).
//!
//! This module provides Ed25519 signatures, versioned domain-separated envelope signing
//! (`swarmtorch.envelope.v0` and `swarmtorch.envelope.v1`), and receiver-challenged
//! session-epoch recovery handshakes (`swarmtorch.session.handshake.v1`).

use crate::traits::PeerId;
use ed25519_dalek::{Signature as DalekSignature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Domain separation tag for legacy v0 envelopes (`(0, 1)`).
pub const ENVELOPE_DOMAIN_TAG_V0: &[u8] = b"swarmtorch.envelope.v0";

/// Domain separation tag for scoped v1 envelopes (`(0, 2)`).
pub const ENVELOPE_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.envelope.v1";

/// Domain separation tag for receiver-challenged session recovery handshakes.
pub const SESSION_HANDSHAKE_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.session.handshake.v1";

/// Scoped envelope v1 signing metadata (`ADR-0021`, `W9-02`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvelopeContextV1 {
    /// Protocol version `(major, minor)`.
    pub version: (u8, u8),
    /// Swarm deployment / cohort instance identifier (16 bytes).
    pub swarm_instance_id: [u8; 16],
    /// Monotonically increasing sender session epoch.
    pub session_epoch: u64,
    /// Training / coordination round identifier.
    pub round_id: u64,
    /// Message type discriminator byte.
    pub message_type: u8,
    /// Per-epoch monotonic sequence number.
    pub sequence: u64,
    /// Unix timestamp in seconds.
    pub timestamp: u32,
}

/// Receiver-challenged session recovery handshake statement (`ADR-0021`, `W9-02`).
///
/// Serde derives follow [`PeerId`]'s `alloc` gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "alloc", derive(Serialize, Deserialize))]
pub struct SessionHandshakeChallenge {
    /// Swarm deployment / cohort instance identifier.
    pub swarm_instance_id: [u8; 16],
    /// Receiver `PeerId` that issued the 256-bit challenge nonce.
    pub receiver_peer: PeerId,
    /// Restarting sender `PeerId` requesting epoch transition.
    pub sender_peer: PeerId,
    /// Requested new session epoch (`> current_epoch`).
    pub requested_epoch: u64,
    /// Initial sequence number in the new epoch.
    pub initial_sequence: u64,
    /// Receiver-generated 256-bit random challenge nonce.
    pub challenge_nonce: [u8; 32],
    /// Challenge issue timestamp in Unix seconds.
    pub issued_at_secs: u32,
}

/// Key pair for signing messages.
#[derive(Clone)]
pub struct KeyPair {
    /// Secret signing key.
    secret: SigningKey,
    /// Public key bytes (32 bytes) - kept for API compatibility.
    pub public: [u8; 32],
}

impl KeyPair {
    /// Generate a new key pair from seed bytes.
    ///
    /// Returns `Err(CryptoError::AllZeroSeed)` if `seed` is all zeros.
    pub fn from_seed(seed: [u8; 32]) -> Result<Self, CryptoError> {
        if seed == [0u8; 32] {
            return Err(CryptoError::AllZeroSeed);
        }
        let secret = SigningKey::from_bytes(&seed);
        let public = secret.verifying_key().to_bytes();
        Ok(Self { secret, public })
    }

    /// Get the peer ID derived from this key pair.
    ///
    /// Always uses the canonical SHA-256 hash derivation for cross-profile consistency.
    pub fn peer_id(&self) -> PeerId {
        PeerId::from_public_key(&self.public)
    }

    /// Get the public key bytes.
    pub const fn public_key(&self) -> &[u8; 32] {
        &self.public
    }
}

/// Signature bytes (64 bytes for Ed25519).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signature(pub [u8; 64]);

impl Signature {
    /// Create from bytes.
    pub const fn from_bytes(bytes: [u8; 64]) -> Self {
        Self(bytes)
    }

    /// Get the signature bytes.
    pub const fn as_bytes(&self) -> &[u8; 64] {
        &self.0
    }

    /// Convert to internal Dalek signature.
    pub fn to_dalek(&self) -> Result<DalekSignature, CryptoError> {
        DalekSignature::from_slice(&self.0).map_err(|_| CryptoError::InvalidSignatureEncoding)
    }
}

/// Errors that can occur during cryptographic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CryptoError {
    /// All-zero seed passed to `KeyPair::from_seed`.
    AllZeroSeed,
    /// Malformed signature bytes.
    InvalidSignatureEncoding,
    /// Malformed or all-zero public key.
    InvalidPublicKey,
    /// Signature verification failed.
    VerificationFailed,
}

impl core::fmt::Display for CryptoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AllZeroSeed => write!(f, "all-zero seed is not allowed"),
            Self::InvalidSignatureEncoding => write!(f, "invalid signature encoding"),
            Self::InvalidPublicKey => write!(f, "invalid public key"),
            Self::VerificationFailed => write!(f, "signature verification failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CryptoError {}

/// Legacy alias for backward compatibility.
pub type VerifyError = CryptoError;

fn canonical_envelope_v0_digest(
    public_key: &[u8; 32],
    version: (u8, u8),
    message_type: u8,
    sequence: u64,
    timestamp: u32,
    payload: &[u8],
) -> [u8; 32] {
    let payload_hash = Sha256::digest(payload);
    let mut hasher = Sha256::new();
    hasher.update(ENVELOPE_DOMAIN_TAG_V0);
    hasher.update([version.0, version.1]);
    hasher.update(public_key);
    hasher.update(sequence.to_le_bytes());
    hasher.update(timestamp.to_le_bytes());
    hasher.update([message_type]);
    hasher.update(payload_hash);
    hasher.finalize().into()
}

fn canonical_envelope_v1_digest(
    public_key: &[u8; 32],
    ctx: &EnvelopeContextV1,
    payload: &[u8],
) -> [u8; 32] {
    let payload_hash = Sha256::digest(payload);
    let mut hasher = Sha256::new();
    hasher.update(ENVELOPE_DOMAIN_TAG_V1);
    hasher.update([ctx.version.0, ctx.version.1]);
    hasher.update(ctx.swarm_instance_id);
    hasher.update(ctx.session_epoch.to_le_bytes());
    hasher.update(ctx.round_id.to_le_bytes());
    hasher.update(public_key);
    hasher.update(ctx.sequence.to_le_bytes());
    hasher.update(ctx.timestamp.to_le_bytes());
    hasher.update([ctx.message_type]);
    hasher.update(payload_hash);
    hasher.finalize().into()
}

fn canonical_session_handshake_digest(
    public_key: &[u8; 32],
    challenge: &SessionHandshakeChallenge,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SESSION_HANDSHAKE_DOMAIN_TAG_V1);
    hasher.update(challenge.swarm_instance_id);
    hasher.update(challenge.receiver_peer.as_bytes());
    hasher.update(challenge.sender_peer.as_bytes());
    hasher.update(public_key);
    hasher.update(challenge.requested_epoch.to_le_bytes());
    hasher.update(challenge.initial_sequence.to_le_bytes());
    hasher.update(challenge.challenge_nonce);
    hasher.update(challenge.issued_at_secs.to_le_bytes());
    hasher.finalize().into()
}

/// Message authentication helper.
pub struct MessageAuth {
    key_pair: KeyPair,
}

impl MessageAuth {
    /// Create a new message authenticator.
    pub fn new(key_pair: KeyPair) -> Self {
        Self { key_pair }
    }

    /// Sign a legacy v0 message envelope (`swarmtorch.envelope.v0`).
    pub fn sign(
        &self,
        version: (u8, u8),
        message_type: u8,
        sequence: u64,
        timestamp: u32,
        payload: &[u8],
    ) -> Signature {
        let canonical = canonical_envelope_v0_digest(
            &self.key_pair.public,
            version,
            message_type,
            sequence,
            timestamp,
            payload,
        );
        let sig = self.key_pair.secret.sign(&canonical);
        Signature(sig.to_bytes())
    }

    /// Verify a legacy v0 signature against envelope components (`swarmtorch.envelope.v0`).
    pub fn verify(
        public_key: &[u8; 32],
        version: (u8, u8),
        message_type: u8,
        sequence: u64,
        timestamp: u32,
        payload: &[u8],
        signature: &Signature,
    ) -> Result<(), CryptoError> {
        if *public_key == [0u8; 32] {
            return Err(CryptoError::InvalidPublicKey);
        }
        let key =
            VerifyingKey::from_bytes(public_key).map_err(|_| CryptoError::InvalidPublicKey)?;
        let sig = signature.to_dalek()?;
        let canonical = canonical_envelope_v0_digest(
            public_key,
            version,
            message_type,
            sequence,
            timestamp,
            payload,
        );
        key.verify_strict(&canonical, &sig)
            .map_err(|_| CryptoError::VerificationFailed)
    }

    /// Sign a scoped v1 message envelope (`swarmtorch.envelope.v1`, `ADR-0021`, `W9-02`).
    pub fn sign_v1(&self, ctx: &EnvelopeContextV1, payload: &[u8]) -> Signature {
        let canonical = canonical_envelope_v1_digest(&self.key_pair.public, ctx, payload);
        let sig = self.key_pair.secret.sign(&canonical);
        Signature(sig.to_bytes())
    }

    /// Verify a scoped v1 signature (`swarmtorch.envelope.v1`, `ADR-0021`, `W9-02`).
    pub fn verify_v1(
        public_key: &[u8; 32],
        ctx: &EnvelopeContextV1,
        payload: &[u8],
        signature: &Signature,
    ) -> Result<(), CryptoError> {
        if *public_key == [0u8; 32] {
            return Err(CryptoError::InvalidPublicKey);
        }
        let key =
            VerifyingKey::from_bytes(public_key).map_err(|_| CryptoError::InvalidPublicKey)?;
        let sig = signature.to_dalek()?;
        let canonical = canonical_envelope_v1_digest(public_key, ctx, payload);
        key.verify_strict(&canonical, &sig)
            .map_err(|_| CryptoError::VerificationFailed)
    }

    /// Sign a receiver-challenged session recovery handshake (`swarmtorch.session.handshake.v1`).
    pub fn sign_session_handshake(&self, challenge: &SessionHandshakeChallenge) -> Signature {
        let canonical = canonical_session_handshake_digest(&self.key_pair.public, challenge);
        let sig = self.key_pair.secret.sign(&canonical);
        Signature(sig.to_bytes())
    }

    /// Verify a receiver-challenged session recovery handshake (`swarmtorch.session.handshake.v1`).
    pub fn verify_session_handshake(
        public_key: &[u8; 32],
        challenge: &SessionHandshakeChallenge,
        signature: &Signature,
    ) -> Result<(), CryptoError> {
        let derived_sender = PeerId::try_from_public_key_bytes(public_key)?;
        if derived_sender != challenge.sender_peer || challenge.challenge_nonce == [0u8; 32] {
            return Err(CryptoError::VerificationFailed);
        }
        let key =
            VerifyingKey::from_bytes(public_key).map_err(|_| CryptoError::InvalidPublicKey)?;
        let sig = signature.to_dalek()?;
        let canonical = canonical_session_handshake_digest(public_key, challenge);
        key.verify_strict(&canonical, &sig)
            .map_err(|_| CryptoError::VerificationFailed)
    }

    /// Get the key pair.
    pub const fn key_pair(&self) -> &KeyPair {
        &self.key_pair
    }
}

/// Configuration surface for security features.
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Require message signatures.
    pub require_signatures: bool,
    /// Encrypt network traffic.
    pub encrypt_transport: bool,
    /// Validate gradient bounds.
    pub validate_gradients: bool,
    /// Maximum clock skew for replay protection (seconds).
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

/// Gradient validator for bounds checking.
#[derive(Debug, Clone)]
pub struct GradientValidator {
    /// Maximum L2 norm for gradient updates.
    pub max_gradient_norm: f32,
    /// Maximum absolute value for any coordinate.
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
    /// Validate a gradient vector.
    pub fn validate(&self, gradients: &[f32]) -> Result<(), GradientValidationError> {
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

        let norm_sq = gradients
            .iter()
            .map(|&g| (g as f64) * (g as f64))
            .sum::<f64>() as f32;
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

/// Software square root for gradient L2-norm validation only.
#[inline]
fn sqrt_f32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        if x <= 0.0 {
            return 0.0;
        }
        let mut y = x;
        for _ in 0..8 {
            y = 0.5 * (y + (x / y));
        }
        y
    }
}

/// Gradient validation error.
#[derive(Debug, Clone)]
pub enum GradientValidationError {
    /// NaN value found.
    NaN { index: usize },
    /// Infinite value found.
    Infinite { index: usize },
    /// Coordinate value too large.
    CoordinateTooLarge { index: usize, value: f32, max: f32 },
    /// Gradient norm too large.
    NormTooLarge { norm: f32, max: f32 },
}

#[cfg(test)]
#[path = "crypto_tests.rs"]
mod tests;
