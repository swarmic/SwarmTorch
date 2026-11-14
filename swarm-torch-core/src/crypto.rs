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

/// Domain separation tag for session-bound v2 envelopes (`(0, 3)`, `ST-03`).
pub const ENVELOPE_DOMAIN_TAG_V2: &[u8] = b"swarmtorch.envelope.v2";

/// Domain separation tag for receiver-nonce-bound session identifiers (`ST-03`).
pub const SESSION_ID_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.session.id.v1";

/// Domain separation tag for receiver-challenged session recovery handshakes.
pub const SESSION_HANDSHAKE_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.session.handshake.v1";

/// Domain separation tag for session-id-bound v2 recovery handshakes (`ST-03`).
pub const SESSION_HANDSHAKE_DOMAIN_TAG_V2: &[u8] = b"swarmtorch.session.handshake.v2";

/// Domain separation tag for attested transport link principal handshakes (`ST-08`).
pub const LINK_ATTESTATION_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.link.attestation.v1";

/// Domain separation tag for `ManifestSignatureV1` (`ADR-0023` §3.1, `W10-02`).
pub const MANIFEST_SIGNATURE_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.manifest.sig.v1";

/// Domain separation tag for `ManifestV1` canonical entry list hashing (`ADR-0023` §3.1, `W10-02`).
pub const MANIFEST_ENTRIES_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.manifest.entries.v1";

/// Domain separation tag for canonical inner update digests (`ADR-0022` §3.1, `W10-03`).
pub const INNER_UPDATE_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.update.v1";

/// Domain separation tag for run- and policy-bound canonical inner update digests (`ST-01`, `ST-03`).
pub const INNER_UPDATE_DOMAIN_TAG_V2: &[u8] = b"swarmtorch.update.v2";

/// Domain separation tag for originator-signed gossip recipient wrappers (`ADR-0022` §3.2, `W10-03`).
pub const RECIPIENT_WRAPPER_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.gossip.wrapper.v1";

/// Domain separation tag for session-id-bound gossip recipient wrappers (`ST-03`).
pub const RECIPIENT_WRAPPER_DOMAIN_TAG_V2: &[u8] = b"swarmtorch.gossip.wrapper.v2";

/// Domain separation tag for canonical `SwarmTorchCheckpointV1` model schema hashes (`ADR-0024` §3, `W11-04`).
pub const MODEL_SCHEMA_HASH_DOMAIN_TAG_V1: &[u8] = b"swarmtorch.schema.v1";

/// Compute a 32-byte SHA-256 digest over `data` in both `std` and `no_std` builds.
#[inline]
pub fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Derive a deterministic 256-bit receiver-bound `session_id` (`ST-03`).
///
/// Rejects an all-zero `receiver_nonce` with [`CryptoError::VerificationFailed`].
pub fn derive_session_id_v1(
    swarm_instance_id: &[u8; 16],
    run_id: &[u8; 16],
    policy_digest: &[u8; 32],
    receiver_peer: &PeerId,
    sender_peer: &PeerId,
    session_epoch: u64,
    receiver_nonce: &[u8; 32],
) -> Result<[u8; 32], CryptoError> {
    if *receiver_nonce == [0u8; 32] {
        return Err(CryptoError::VerificationFailed);
    }
    Ok(sha256_slices(&[
        SESSION_ID_DOMAIN_TAG_V1,
        swarm_instance_id,
        run_id,
        policy_digest,
        receiver_peer.as_bytes(),
        sender_peer.as_bytes(),
        &session_epoch.to_le_bytes(),
        receiver_nonce,
    ]))
}

/// Scoped and session-bound envelope v2 signing metadata (`(0, 3)`, `ST-03`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvelopeContextV2 {
    /// Protocol version `(major, minor)`.
    pub version: (u8, u8),
    /// Swarm deployment / cohort instance identifier (16 bytes).
    pub swarm_instance_id: [u8; 16],
    /// Bound training/coordination run identifier (16 bytes).
    pub run_id: [u8; 16],
    /// Monotonically increasing sender session epoch.
    pub session_epoch: u64,
    /// Receiver-nonce-derived session identifier ([`derive_session_id_v1`]).
    pub session_id: [u8; 32],
    /// Training / coordination round identifier.
    pub round_id: u64,
    /// Message type discriminator byte.
    pub message_type: u8,
    /// Per-session monotonic sequence number.
    pub sequence: u64,
    /// Unix timestamp in seconds.
    pub timestamp: u32,
}

/// Receiver-challenged v2 session recovery handshake statement (`ST-03`).
///
/// Binds `run_id`, `policy_digest`, and `receiver_nonce` so a fresh receiver process
/// invalidates stale pre-crash messages even within the same wall-clock second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "alloc", derive(Serialize, Deserialize))]
pub struct SessionHandshakeChallengeV2 {
    /// Swarm deployment / cohort instance identifier.
    pub swarm_instance_id: [u8; 16],
    /// Bound training/coordination run identifier.
    pub run_id: [u8; 16],
    /// Context policy digest governing the session.
    pub policy_digest: [u8; 32],
    /// Receiver `PeerId` that issued `receiver_nonce`.
    pub receiver_peer: PeerId,
    /// Restarting sender `PeerId` establishing or advancing the session.
    pub sender_peer: PeerId,
    /// Target session epoch (`>= 1`, strictly greater than any active epoch for the sender).
    pub requested_epoch: u64,
    /// Initial sequence floor reserved for the new session.
    pub initial_sequence: u64,
    /// Receiver-generated 256-bit random session challenge nonce.
    pub receiver_nonce: [u8; 32],
    /// Challenge issue timestamp in Unix seconds.
    pub issued_at_secs: u32,
}

impl SessionHandshakeChallengeV2 {
    /// Derive the deterministic v2 session identifier bound to this challenge.
    pub fn derive_session_id(&self) -> Result<[u8; 32], CryptoError> {
        derive_session_id_v1(
            &self.swarm_instance_id,
            &self.run_id,
            &self.policy_digest,
            &self.receiver_peer,
            &self.sender_peer,
            self.requested_epoch,
            &self.receiver_nonce,
        )
    }
}

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
    /// Cumulative differential privacy $(\varepsilon, \delta)$ budget exhausted (`ADR-0023` §4.2).
    PrivacyBudgetExhausted,
    /// Invalid differential privacy hyperparameter ($\tau \le 0$, $\sigma \le 0$, $q \notin (0, 1]$, or $\delta \notin (0, 1)$).
    InvalidPrivacyParameter,
}

impl core::fmt::Display for CryptoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AllZeroSeed => write!(f, "all-zero seed is not allowed"),
            Self::InvalidSignatureEncoding => write!(f, "invalid signature encoding"),
            Self::InvalidPublicKey => write!(f, "invalid public key"),
            Self::VerificationFailed => write!(f, "signature verification failed"),
            Self::PrivacyBudgetExhausted => {
                write!(f, "differential privacy budget exhausted")
            }
            Self::InvalidPrivacyParameter => {
                write!(f, "invalid differential privacy parameter")
            }
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

fn canonical_envelope_v2_digest(
    public_key: &[u8; 32],
    ctx: &EnvelopeContextV2,
    payload: &[u8],
) -> [u8; 32] {
    let payload_hash = Sha256::digest(payload);
    let mut hasher = Sha256::new();
    hasher.update(ENVELOPE_DOMAIN_TAG_V2);
    hasher.update([ctx.version.0, ctx.version.1]);
    hasher.update(ctx.swarm_instance_id);
    hasher.update(ctx.run_id);
    hasher.update(ctx.session_epoch.to_le_bytes());
    hasher.update(ctx.session_id);
    hasher.update(ctx.round_id.to_le_bytes());
    hasher.update(public_key);
    hasher.update(ctx.sequence.to_le_bytes());
    hasher.update(ctx.timestamp.to_le_bytes());
    hasher.update([ctx.message_type]);
    hasher.update(payload_hash);
    hasher.finalize().into()
}

fn canonical_session_handshake_v2_digest(
    public_key: &[u8; 32],
    challenge: &SessionHandshakeChallengeV2,
    session_id: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SESSION_HANDSHAKE_DOMAIN_TAG_V2);
    hasher.update(challenge.swarm_instance_id);
    hasher.update(challenge.run_id);
    hasher.update(challenge.policy_digest);
    hasher.update(challenge.receiver_peer.as_bytes());
    hasher.update(challenge.sender_peer.as_bytes());
    hasher.update(public_key);
    hasher.update(challenge.requested_epoch.to_le_bytes());
    hasher.update(challenge.initial_sequence.to_le_bytes());
    hasher.update(challenge.receiver_nonce);
    hasher.update(session_id);
    hasher.update(challenge.issued_at_secs.to_le_bytes());
    hasher.finalize().into()
}

/// Compute the canonical `ManifestSignatureV1` preimage digest (`ADR-0023` §3.1, `W10-02`):
/// `"swarmtorch.manifest.sig.v1" || u32_le(schema_version) || run_id_raw_16_bytes || u64_le(signed_unix_nanos) || entries_hash`.
pub fn canonical_manifest_sig_v1_digest(
    schema_version: u32,
    run_id_raw_16_bytes: &[u8; 16],
    signed_unix_nanos: u64,
    entries_hash: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(MANIFEST_SIGNATURE_DOMAIN_TAG_V1);
    hasher.update(schema_version.to_le_bytes());
    hasher.update(run_id_raw_16_bytes);
    hasher.update(signed_unix_nanos.to_le_bytes());
    hasher.update(entries_hash);
    hasher.finalize().into()
}

/// Compute the canonical SHA-256 digest of a slice sequence (`W10-03`).
pub fn sha256_slices(slices: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for s in slices {
        hasher.update(s);
    }
    hasher.finalize().into()
}

fn canonical_prehashed_domain_digest(
    domain_tag: &[u8],
    public_key: &[u8; 32],
    content_digest: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain_tag);
    hasher.update(public_key);
    hasher.update(content_digest);
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

    /// Sign a session-id-bound v2 message envelope (`swarmtorch.envelope.v2`, `ST-03`).
    pub fn sign_v2(&self, ctx: &EnvelopeContextV2, payload: &[u8]) -> Signature {
        let canonical = canonical_envelope_v2_digest(&self.key_pair.public, ctx, payload);
        let sig = self.key_pair.secret.sign(&canonical);
        Signature(sig.to_bytes())
    }

    /// Verify a session-id-bound v2 message envelope (`swarmtorch.envelope.v2`, `ST-03`).
    pub fn verify_v2(
        public_key: &[u8; 32],
        ctx: &EnvelopeContextV2,
        payload: &[u8],
        signature: &Signature,
    ) -> Result<(), CryptoError> {
        if *public_key == [0u8; 32] || ctx.session_id == [0u8; 32] {
            return Err(CryptoError::InvalidPublicKey);
        }
        let key =
            VerifyingKey::from_bytes(public_key).map_err(|_| CryptoError::InvalidPublicKey)?;
        let sig = signature.to_dalek()?;
        let canonical = canonical_envelope_v2_digest(public_key, ctx, payload);
        key.verify_strict(&canonical, &sig)
            .map_err(|_| CryptoError::VerificationFailed)
    }

    /// Sign a v2 session recovery handshake (`swarmtorch.session.handshake.v2`, `ST-03`).
    pub fn sign_session_handshake_v2(
        &self,
        challenge: &SessionHandshakeChallengeV2,
    ) -> Result<Signature, CryptoError> {
        let session_id = challenge.derive_session_id()?;
        let canonical =
            canonical_session_handshake_v2_digest(&self.key_pair.public, challenge, &session_id);
        let sig = self.key_pair.secret.sign(&canonical);
        Ok(Signature(sig.to_bytes()))
    }

    /// Verify a v2 session recovery handshake (`swarmtorch.session.handshake.v2`, `ST-03`),
    /// returning the derived deterministic `session_id`.
    pub fn verify_session_handshake_v2(
        public_key: &[u8; 32],
        challenge: &SessionHandshakeChallengeV2,
        signature: &Signature,
    ) -> Result<[u8; 32], CryptoError> {
        let derived_sender = PeerId::try_from_public_key_bytes(public_key)?;
        if derived_sender != challenge.sender_peer || challenge.requested_epoch == 0 {
            return Err(CryptoError::VerificationFailed);
        }
        let session_id = challenge.derive_session_id()?;
        let key =
            VerifyingKey::from_bytes(public_key).map_err(|_| CryptoError::InvalidPublicKey)?;
        let sig = signature.to_dalek()?;
        let canonical = canonical_session_handshake_v2_digest(public_key, challenge, &session_id);
        key.verify_strict(&canonical, &sig)
            .map_err(|_| CryptoError::VerificationFailed)?;
        Ok(session_id)
    }

    /// Sign a `ManifestSignatureV1` preimage (`"swarmtorch.manifest.sig.v1"`, `ADR-0023` §3.1, `W10-02`).
    pub fn sign_manifest_v1(
        &self,
        schema_version: u32,
        run_id_raw_16_bytes: &[u8; 16],
        signed_unix_nanos: u64,
        entries_hash: &[u8; 32],
    ) -> Signature {
        let canonical = canonical_manifest_sig_v1_digest(
            schema_version,
            run_id_raw_16_bytes,
            signed_unix_nanos,
            entries_hash,
        );
        let sig = self.key_pair.secret.sign(&canonical);
        Signature(sig.to_bytes())
    }

    /// Verify a `ManifestSignatureV1` signature (`"swarmtorch.manifest.sig.v1"`, `ADR-0023` §3.1, `W10-02`).
    pub fn verify_manifest_v1(
        public_key: &[u8; 32],
        schema_version: u32,
        run_id_raw_16_bytes: &[u8; 16],
        signed_unix_nanos: u64,
        entries_hash: &[u8; 32],
        signature: &Signature,
    ) -> Result<(), CryptoError> {
        let _ = PeerId::try_from_public_key_bytes(public_key)?;
        if run_id_raw_16_bytes.iter().all(|&b| b == 0) {
            return Err(CryptoError::VerificationFailed);
        }
        let key =
            VerifyingKey::from_bytes(public_key).map_err(|_| CryptoError::InvalidPublicKey)?;
        let sig = signature.to_dalek()?;
        let canonical = canonical_manifest_sig_v1_digest(
            schema_version,
            run_id_raw_16_bytes,
            signed_unix_nanos,
            entries_hash,
        );
        key.verify_strict(&canonical, &sig)
            .map_err(|_| CryptoError::VerificationFailed)
    }

    /// Sign a pre-hashed 32-byte digest under a specific domain separation tag (`ADR-0022` §3, `W10-03`).
    pub fn sign_prehashed_domain(&self, domain_tag: &[u8], content_digest: &[u8; 32]) -> Signature {
        let canonical =
            canonical_prehashed_domain_digest(domain_tag, &self.key_pair.public, content_digest);
        let sig = self.key_pair.secret.sign(&canonical);
        Signature(sig.to_bytes())
    }

    /// Verify a pre-hashed 32-byte digest under a specific domain separation tag (`ADR-0022` §3, `W10-03`).
    pub fn verify_prehashed_domain(
        public_key: &[u8; 32],
        domain_tag: &[u8],
        content_digest: &[u8; 32],
        signature: &Signature,
    ) -> Result<(), CryptoError> {
        let _ = PeerId::try_from_public_key_bytes(public_key)?;
        let key =
            VerifyingKey::from_bytes(public_key).map_err(|_| CryptoError::InvalidPublicKey)?;
        let sig = signature.to_dalek()?;
        let canonical = canonical_prehashed_domain_digest(domain_tag, public_key, content_digest);
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
            .sum::<f64>();
        let norm_f64 = crate::num::sqrt_f64(norm_sq);
        if !norm_f64.is_finite() || norm_f64 > (self.max_gradient_norm as f64) {
            return Err(GradientValidationError::NormTooLarge {
                norm: norm_f64 as f32,
                max: self.max_gradient_norm,
            });
        }

        Ok(())
    }
}

/// Software square root for gradient L2-norm validation.
#[inline]
#[allow(dead_code)]
fn sqrt_f32(x: f32) -> f32 {
    crate::num::sqrt_f32(x)
}

/// Wave 12 (`W12-03`) descriptor for Rényi Differential Privacy (`RdpAccountantV1`)
/// composition coupled to explicit $\ell_2$ gradient clipping bounds (`CenteredClipping` /
/// [`GradientValidator`]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct RdpBudgetDescriptorV1 {
    /// Rényi order $\alpha > 1.0$.
    pub alpha: f64,
    /// Cumulative privacy loss $\epsilon(\alpha)$ at order `alpha`.
    pub epsilon_alpha: f64,
    /// Target $(\epsilon, \delta)$-DP failure probability $\delta \in (0, 1)$.
    pub target_delta: f64,
    /// Per-round $\ell_2$ sensitivity / clipping radius $\tau > 0$.
    pub clipping_radius_tau: f32,
    /// Gaussian noise multiplier $\sigma / \tau$.
    pub noise_multiplier: f32,
}

/// Wave 12 (`W12-02`) extension hook for privacy budget accounting across rounds.
///
/// In Wave 11 (`v0.1.0`), `CenteredClipping` and [`GradientValidator`] enforce the geometric
/// $\ell_2$ sensitivity bound $\tau$, and implementations of this trait can observe or gate
/// per-round sensitivity expenditure ahead of full `RdpAccountant` integration in Wave 12.
pub trait PrivacyAccountantHook {
    /// Returns the current Rényi DP budget descriptor before committing a round.
    fn current_budget(&self) -> RdpBudgetDescriptorV1;

    /// Records a single round with clipping radius `tau` and cohort subsampling rate `q`,
    /// returning `Ok(())` if the cumulative budget remains within policy.
    fn record_clipped_round(
        &mut self,
        tau: f32,
        subsampling_rate_q: f64,
    ) -> Result<(), CryptoError>;
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
