//! Originator-authenticated gossip inner updates and recipient wrappers (`ADR-0022` §3, `W10-03`).
//!
//! Decouples immutable originator-signed inner round contributions ([`InnerUpdateV1`]) from
//! recipient-scoped delivery wrappers ([`RecipientWrapperV1`]) so multi-hop gossip relays can
//! forward updates without forging originator provenance or enabling cross-recipient replay.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
#[cfg(feature = "alloc")]
use swarm_torch_core::crypto::{
    sha256_slices, CryptoError, MessageAuth, Signature, INNER_UPDATE_DOMAIN_TAG_V1,
    INNER_UPDATE_DOMAIN_TAG_V2, RECIPIENT_WRAPPER_DOMAIN_TAG_V1, RECIPIENT_WRAPPER_DOMAIN_TAG_V2,
};
#[cfg(feature = "alloc")]
use swarm_torch_core::replay::{
    PreparedReplayAdmission, ReplayError, ReplayProtection, ReplayRecordKey, ReplayRecordKeyV2,
};
#[cfg(feature = "alloc")]
use swarm_torch_core::traits::PeerId;

/// Default maximum serialized byte size for [`BundledGossipDeliveryV1`] (`1,048,576` bytes = 1 MiB).
pub const DEFAULT_MAX_GOSSIP_DELIVERY_BYTES: usize = 1_048_576;

/// Payload kind discriminator for [`InnerUpdateV1`] (`ADR-0022` §3.1, `W10-03`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PayloadKindV1 {
    /// Multi-step local model delta $\Delta_i = w_i^{\text{local}} - w_{\text{parent}}$.
    ModelDeltaV1 = 1,
    /// Single-step gradient vector $g_i$.
    GradientV1 = 2,
    /// Stateful `EF21` compressed gradient innovation $c_i^{(r)}$.
    Ef21InnovationV1 = 3,
    /// Phase-1 pre-seed Merkle vector commitment for collective `SharedSeedSparse` (`ST-12`).
    SharedSeedCommitmentV1 = 4,
    /// Phase-2 shared-seed projected opening + Merkle proofs for collective `SharedSeedSparse` (`ST-12`).
    SharedSeedOpeningV1 = 5,
}

/// Originator-signed immutable round contribution (`ADR-0022` §3.1, `W10-03`).
///
/// Signed once by the originating participant over `"swarmtorch.update.v1"` and relayed
/// unchanged across one or more gossip hops.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InnerUpdateV1 {
    /// Originator's Ed25519 public key bytes (raw 32 bytes).
    pub originator_public_key: [u8; 32],
    /// Scoped swarm deployment / cohort instance identifier (16 bytes).
    pub swarm_instance_id: [u8; 16],
    /// Training round identifier.
    pub round_id: u64,
    /// Originator-assigned contribution identifier (monotonic per originator).
    pub contribution_id: u64,
    /// Expected parent checkpoint SHA-256 content digest for `round_id`.
    pub base_checkpoint_digest: [u8; 32],
    /// Payload kind discriminator.
    pub payload_kind: PayloadKindV1,
    /// Compression format schema version (`0` for uncompressed `f32` postcard slice, `1` for `CompressedGradient`).
    pub compression_schema_version: u16,
    /// Number of parameter/gradient coordinates represented in `encoded_payload`.
    pub num_elements: u32,
    /// Advisory local sample count (telemetry only; never trusted for uncertified weight inflation).
    pub reported_examples: u64,
    /// Encoded payload bytes (e.g., serialized `Vec<f32>` or `CompressedGradient`).
    pub encoded_payload: Vec<u8>,
    /// Originator Ed25519 signature (64 bytes, stored as `Vec<u8>` for serde portability).
    pub originator_signature: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl InnerUpdateV1 {
    /// Derive the originator's canonical [`PeerId`] after validating the Ed25519 public key.
    pub fn originator_peer_id(&self) -> Result<PeerId, CryptoError> {
        PeerId::try_from_public_key_bytes(&self.originator_public_key)
    }

    /// Compute the canonical SHA-256 content digest over all unsigned fields of [`InnerUpdateV1`]
    /// under [`INNER_UPDATE_DOMAIN_TAG_V1`] (`"swarmtorch.update.v1"`).
    pub fn canonical_digest(&self) -> [u8; 32] {
        let payload_sha = sha256_slices(&[&self.encoded_payload]);
        let round_le = self.round_id.to_le_bytes();
        let contrib_le = self.contribution_id.to_le_bytes();
        let kind_byte = [self.payload_kind as u8];
        let comp_le = self.compression_schema_version.to_le_bytes();
        let num_le = self.num_elements.to_le_bytes();
        let examples_le = self.reported_examples.to_le_bytes();
        let len_le = (self.encoded_payload.len() as u64).to_le_bytes();
        sha256_slices(&[
            INNER_UPDATE_DOMAIN_TAG_V1,
            &self.originator_public_key,
            &self.swarm_instance_id,
            &round_le,
            &contrib_le,
            &self.base_checkpoint_digest,
            &kind_byte,
            &comp_le,
            &num_le,
            &examples_le,
            &len_le,
            &payload_sha,
        ])
    }

    /// Sign this [`InnerUpdateV1`] with the originator's [`MessageAuth`].
    pub fn sign_with(mut self, auth: &MessageAuth) -> Self {
        let digest = self.canonical_digest();
        let sig = auth.sign_prehashed_domain(INNER_UPDATE_DOMAIN_TAG_V1, &digest);
        self.originator_signature = sig.as_bytes().to_vec();
        self
    }

    /// Verify the originator's Ed25519 signature over [`Self::canonical_digest`].
    pub fn verify_originator_signature(&self) -> Result<PeerId, GossipVerifyError> {
        let peer_id = self
            .originator_peer_id()
            .map_err(|_| GossipVerifyError::InvalidOriginatorKey)?;
        if self.originator_signature.len() != 64 {
            return Err(GossipVerifyError::InvalidSignatureLength {
                expected: 64,
                found: self.originator_signature.len(),
            });
        }
        let mut sig_arr = [0u8; 64];
        sig_arr.copy_from_slice(&self.originator_signature);
        let sig = Signature::from_bytes(sig_arr);
        let digest = self.canonical_digest();
        MessageAuth::verify_prehashed_domain(
            &self.originator_public_key,
            INNER_UPDATE_DOMAIN_TAG_V1,
            &digest,
            &sig,
        )
        .map_err(GossipVerifyError::Crypto)?;
        Ok(peer_id)
    }
}

/// Recipient-bound wrapper signed by the originator (`ADR-0022` §3.2, `W10-03`).
///
/// Binds `intended_recipient`, `recipient_session_epoch`, `delivery_seq`, and `inner_digest`
/// so a message addressed to recipient $A$ cannot be replayed against recipient $B$, while
/// allowing untrusted intermediate relays to forward the bundle without re-signing.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecipientWrapperV1 {
    /// Wrapper protocol version (`(0, 2)`).
    pub version: (u8, u8),
    /// Originator's Ed25519 public key bytes (raw 32 bytes).
    pub originator_public_key: [u8; 32],
    /// Intended final recipient `PeerId` (e.g. coordinator or gossip neighbor).
    pub intended_recipient: PeerId,
    /// Scoped swarm deployment / cohort instance identifier (16 bytes).
    pub swarm_instance_id: [u8; 16],
    /// Originator's session epoch with `intended_recipient`.
    pub recipient_session_epoch: u64,
    /// Monotonic delivery sequence number for `(swarm_instance_id, originator, recipient_session_epoch)`.
    pub delivery_seq: u64,
    /// Unix timestamp in seconds.
    pub timestamp_secs: u32,
    /// Training round identifier.
    pub round_id: u64,
    /// Canonical digest (`InnerUpdateV1::canonical_digest`) of the wrapped inner update.
    pub inner_digest: [u8; 32],
    /// Byte length of the wrapped `InnerUpdateV1::encoded_payload`.
    pub inner_length: u32,
    /// Originator Ed25519 signature over the wrapper preimage (64 bytes).
    pub wrapper_signature: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl RecipientWrapperV1 {
    /// Supported wrapper version (`(0, 2)`).
    pub const VERSION_V1: (u8, u8) = (0, 2);

    /// Compute the canonical SHA-256 digest of the unsigned wrapper fields.
    pub fn canonical_digest(&self) -> [u8; 32] {
        let ver_bytes = [self.version.0, self.version.1];
        let epoch_le = self.recipient_session_epoch.to_le_bytes();
        let seq_le = self.delivery_seq.to_le_bytes();
        let ts_le = self.timestamp_secs.to_le_bytes();
        let round_le = self.round_id.to_le_bytes();
        let len_le = self.inner_length.to_le_bytes();
        sha256_slices(&[
            RECIPIENT_WRAPPER_DOMAIN_TAG_V1,
            &ver_bytes,
            &self.originator_public_key,
            self.intended_recipient.as_bytes(),
            &self.swarm_instance_id,
            &epoch_le,
            &seq_le,
            &ts_le,
            &round_le,
            &self.inner_digest,
            &len_le,
        ])
    }

    /// Sign this [`RecipientWrapperV1`] with the originator's [`MessageAuth`].
    pub fn sign_with(mut self, auth: &MessageAuth) -> Self {
        let digest = self.canonical_digest();
        let sig = auth.sign_prehashed_domain(RECIPIENT_WRAPPER_DOMAIN_TAG_V1, &digest);
        self.wrapper_signature = sig.as_bytes().to_vec();
        self
    }

    /// Verify the originator's wrapper signature.
    pub fn verify_wrapper_signature(&self) -> Result<PeerId, GossipVerifyError> {
        if self.version != Self::VERSION_V1 {
            return Err(GossipVerifyError::UnsupportedVersion {
                major: self.version.0,
                minor: self.version.1,
            });
        }
        let peer_id = PeerId::try_from_public_key_bytes(&self.originator_public_key)
            .map_err(|_| GossipVerifyError::InvalidOriginatorKey)?;
        if self.wrapper_signature.len() != 64 {
            return Err(GossipVerifyError::InvalidSignatureLength {
                expected: 64,
                found: self.wrapper_signature.len(),
            });
        }
        let mut sig_arr = [0u8; 64];
        sig_arr.copy_from_slice(&self.wrapper_signature);
        let sig = Signature::from_bytes(sig_arr);
        let digest = self.canonical_digest();
        MessageAuth::verify_prehashed_domain(
            &self.originator_public_key,
            RECIPIENT_WRAPPER_DOMAIN_TAG_V1,
            &digest,
            &sig,
        )
        .map_err(GossipVerifyError::Crypto)?;
        Ok(peer_id)
    }
}

/// Bundled gossip delivery combining a [`RecipientWrapperV1`] and an [`InnerUpdateV1`]
/// (`ADR-0022` §3.3, `W10-03`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundledGossipDeliveryV1 {
    pub wrapper: RecipientWrapperV1,
    pub inner: InnerUpdateV1,
}

#[cfg(feature = "alloc")]
fn verify_prehashed_originator_sig(
    public_key: &[u8; 32],
    domain_tag: &[u8],
    digest: &[u8; 32],
    sig_bytes: &[u8],
) -> Result<PeerId, GossipVerifyError> {
    let peer_id = PeerId::try_from_public_key_bytes(public_key)
        .map_err(|_| GossipVerifyError::InvalidOriginatorKey)?;
    if sig_bytes.len() != 64 {
        return Err(GossipVerifyError::InvalidSignatureLength {
            expected: 64,
            found: sig_bytes.len(),
        });
    }
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(sig_bytes);
    let sig = Signature::from_bytes(sig_arr);
    MessageAuth::verify_prehashed_domain(public_key, domain_tag, digest, &sig)
        .map_err(GossipVerifyError::Crypto)?;
    Ok(peer_id)
}

/// Originator-signed immutable v2 round contribution (`ST-01`, `ST-03`).
///
/// Explicitly binds `run_id`, `base_checkpoint_digest`, and `context_policy_digest` under
/// [`INNER_UPDATE_DOMAIN_TAG_V2`] (`"swarmtorch.update.v2"`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InnerUpdateV2 {
    /// Originator's Ed25519 public key bytes (raw 32 bytes).
    pub originator_public_key: [u8; 32],
    /// Scoped swarm deployment / cohort instance identifier (16 bytes).
    pub swarm_instance_id: [u8; 16],
    /// Bound training/coordination run identifier (16 bytes).
    pub run_id: [u8; 16],
    /// Training round identifier.
    pub round_id: u64,
    /// Originator-assigned contribution identifier (monotonic per originator).
    pub contribution_id: u64,
    /// Expected parent checkpoint SHA-256 content digest for `round_id`.
    pub base_checkpoint_digest: [u8; 32],
    /// SHA-256 context/aggregation policy digest governing the round.
    pub context_policy_digest: [u8; 32],
    /// Payload kind discriminator.
    pub payload_kind: PayloadKindV1,
    /// Compression format schema version.
    pub compression_schema_version: u16,
    /// Number of parameter/gradient coordinates represented in `encoded_payload`.
    pub num_elements: u32,
    /// Advisory local sample count (telemetry only; never trusted for uncertified weight inflation).
    pub reported_examples: u64,
    /// Encoded payload bytes.
    pub encoded_payload: Vec<u8>,
    /// Originator Ed25519 signature (64 bytes).
    pub originator_signature: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl InnerUpdateV2 {
    /// Derive the originator's canonical [`PeerId`] after validating the Ed25519 public key.
    pub fn originator_peer_id(&self) -> Result<PeerId, CryptoError> {
        PeerId::try_from_public_key_bytes(&self.originator_public_key)
    }

    /// Compute the canonical SHA-256 content digest over all unsigned fields of [`InnerUpdateV2`]
    /// under [`INNER_UPDATE_DOMAIN_TAG_V2`] (`"swarmtorch.update.v2"`).
    pub fn canonical_digest(&self) -> [u8; 32] {
        let payload_sha = sha256_slices(&[&self.encoded_payload]);
        let round_le = self.round_id.to_le_bytes();
        let contrib_le = self.contribution_id.to_le_bytes();
        let kind_byte = [self.payload_kind as u8];
        let comp_le = self.compression_schema_version.to_le_bytes();
        let num_le = self.num_elements.to_le_bytes();
        let examples_le = self.reported_examples.to_le_bytes();
        let len_le = (self.encoded_payload.len() as u64).to_le_bytes();
        sha256_slices(&[
            INNER_UPDATE_DOMAIN_TAG_V2,
            &self.originator_public_key,
            &self.swarm_instance_id,
            &self.run_id,
            &round_le,
            &contrib_le,
            &self.base_checkpoint_digest,
            &self.context_policy_digest,
            &kind_byte,
            &comp_le,
            &num_le,
            &examples_le,
            &len_le,
            &payload_sha,
        ])
    }

    /// Sign this [`InnerUpdateV2`] with the originator's [`MessageAuth`].
    pub fn sign_with(mut self, auth: &MessageAuth) -> Self {
        let digest = self.canonical_digest();
        let sig = auth.sign_prehashed_domain(INNER_UPDATE_DOMAIN_TAG_V2, &digest);
        self.originator_signature = sig.as_bytes().to_vec();
        self
    }

    /// Verify the originator's Ed25519 signature over [`Self::canonical_digest`].
    pub fn verify_originator_signature(&self) -> Result<PeerId, GossipVerifyError> {
        let digest = self.canonical_digest();
        verify_prehashed_originator_sig(
            &self.originator_public_key,
            INNER_UPDATE_DOMAIN_TAG_V2,
            &digest,
            &self.originator_signature,
        )
    }
}

/// Recipient-bound and session-bound v2 wrapper signed by the originator (`ST-03`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecipientWrapperV2 {
    /// Wrapper protocol version (`(0, 3)`).
    pub version: (u8, u8),
    /// Originator's Ed25519 public key bytes (raw 32 bytes).
    pub originator_public_key: [u8; 32],
    /// Intended final recipient `PeerId`.
    pub intended_recipient: PeerId,
    /// Scoped swarm deployment / cohort instance identifier (16 bytes).
    pub swarm_instance_id: [u8; 16],
    /// Bound training/coordination run identifier (16 bytes).
    pub run_id: [u8; 16],
    /// Receiver-nonce-derived session identifier (`derive_session_id_v1`).
    pub session_id: [u8; 32],
    /// Originator's session epoch with `intended_recipient`.
    pub recipient_session_epoch: u64,
    /// Monotonic delivery sequence number within the session.
    pub delivery_seq: u64,
    /// Unix timestamp in seconds.
    pub timestamp_secs: u32,
    /// Training round identifier.
    pub round_id: u64,
    /// Canonical digest (`InnerUpdateV2::canonical_digest`) of the wrapped inner update.
    pub inner_digest: [u8; 32],
    /// Byte length of the wrapped `InnerUpdateV2::encoded_payload`.
    pub inner_length: u32,
    /// Originator Ed25519 signature over the wrapper preimage (64 bytes).
    pub wrapper_signature: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl RecipientWrapperV2 {
    /// Supported v2 wrapper version (`(0, 3)`).
    pub const VERSION_V2: (u8, u8) = (0, 3);

    /// Compute the canonical SHA-256 digest of the unsigned v2 wrapper fields.
    pub fn canonical_digest(&self) -> [u8; 32] {
        let ver_bytes = [self.version.0, self.version.1];
        let epoch_le = self.recipient_session_epoch.to_le_bytes();
        let seq_le = self.delivery_seq.to_le_bytes();
        let ts_le = self.timestamp_secs.to_le_bytes();
        let round_le = self.round_id.to_le_bytes();
        let len_le = self.inner_length.to_le_bytes();
        sha256_slices(&[
            RECIPIENT_WRAPPER_DOMAIN_TAG_V2,
            &ver_bytes,
            &self.originator_public_key,
            self.intended_recipient.as_bytes(),
            &self.swarm_instance_id,
            &self.run_id,
            &self.session_id,
            &epoch_le,
            &seq_le,
            &ts_le,
            &round_le,
            &self.inner_digest,
            &len_le,
        ])
    }

    /// Sign this [`RecipientWrapperV2`] with the originator's [`MessageAuth`].
    pub fn sign_with(mut self, auth: &MessageAuth) -> Self {
        let digest = self.canonical_digest();
        let sig = auth.sign_prehashed_domain(RECIPIENT_WRAPPER_DOMAIN_TAG_V2, &digest);
        self.wrapper_signature = sig.as_bytes().to_vec();
        self
    }

    /// Verify the originator's v2 wrapper signature.
    pub fn verify_wrapper_signature(&self) -> Result<PeerId, GossipVerifyError> {
        if self.version != Self::VERSION_V2 {
            return Err(GossipVerifyError::UnsupportedVersion {
                major: self.version.0,
                minor: self.version.1,
            });
        }
        let digest = self.canonical_digest();
        verify_prehashed_originator_sig(
            &self.originator_public_key,
            RECIPIENT_WRAPPER_DOMAIN_TAG_V2,
            &digest,
            &self.wrapper_signature,
        )
    }
}

/// Bundled v2 gossip delivery combining [`RecipientWrapperV2`] and [`InnerUpdateV2`] (`ST-01`, `ST-03`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundledGossipDeliveryV2 {
    pub wrapper: RecipientWrapperV2,
    pub inner: InnerUpdateV2,
}

/// Errors returned when verifying a [`BundledGossipDeliveryV1`] or [`BundledGossipDeliveryV2`] (`ADR-0022` §3).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GossipVerifyError {
    FrameTooLarge {
        len: usize,
        max: usize,
    },
    MalformedDelivery,
    UnsupportedVersion {
        major: u8,
        minor: u8,
    },
    RecipientMismatch {
        expected: PeerId,
        found: PeerId,
    },
    OriginatorKeyMismatch,
    SwarmInstanceMismatch,
    RunIdMismatch,
    RoundMismatch {
        wrapper_round: u64,
        inner_round: u64,
    },
    InnerLengthMismatch {
        expected: u32,
        found: usize,
    },
    InnerDigestMismatch,
    InvalidOriginatorKey,
    InvalidSignatureLength {
        expected: usize,
        found: usize,
    },
    Crypto(CryptoError),
    Replay(ReplayError),
}

#[cfg(feature = "alloc")]
impl core::fmt::Display for GossipVerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FrameTooLarge { len, max } => {
                write!(f, "gossip delivery frame size {len} exceeds max {max}")
            }
            Self::MalformedDelivery => write!(f, "malformed gossip delivery wire encoding"),
            Self::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported gossip wrapper version ({major}, {minor})")
            }
            Self::RecipientMismatch { expected, found } => {
                write!(
                    f,
                    "gossip recipient mismatch: expected {expected:?}, found {found:?}"
                )
            }
            Self::OriginatorKeyMismatch => {
                write!(
                    f,
                    "wrapper originator_public_key != inner originator_public_key"
                )
            }
            Self::SwarmInstanceMismatch => {
                write!(f, "wrapper swarm_instance_id != inner swarm_instance_id")
            }
            Self::RunIdMismatch => {
                write!(f, "wrapper run_id != inner run_id")
            }
            Self::RoundMismatch {
                wrapper_round,
                inner_round,
            } => write!(
                f,
                "wrapper round_id ({wrapper_round}) != inner round_id ({inner_round})"
            ),
            Self::InnerLengthMismatch { expected, found } => {
                write!(
                    f,
                    "wrapper inner_length ({expected}) != inner payload len ({found})"
                )
            }
            Self::InnerDigestMismatch => {
                write!(
                    f,
                    "wrapper inner_digest does not match inner canonical_digest"
                )
            }
            Self::InvalidOriginatorKey => write!(f, "invalid or all-zero originator public key"),
            Self::InvalidSignatureLength { expected, found } => {
                write!(
                    f,
                    "invalid signature length: expected {expected}, found {found}"
                )
            }
            Self::Crypto(e) => write!(f, "gossip signature verification failed: {e}"),
            Self::Replay(e) => write!(f, "gossip replay check failed: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GossipVerifyError {}

#[cfg(feature = "alloc")]
impl BundledGossipDeliveryV1 {
    /// Construct and sign both [`InnerUpdateV1`] and [`RecipientWrapperV1`] using the originator's
    /// [`MessageAuth`].
    pub fn new_signed(
        auth: &MessageAuth,
        mut inner: InnerUpdateV1,
        intended_recipient: PeerId,
        recipient_session_epoch: u64,
        delivery_seq: u64,
        timestamp_secs: u32,
    ) -> Self {
        inner = inner.sign_with(auth);
        let inner_digest = inner.canonical_digest();
        let inner_length = inner.encoded_payload.len() as u32;
        let wrapper = RecipientWrapperV1 {
            version: RecipientWrapperV1::VERSION_V1,
            originator_public_key: inner.originator_public_key,
            intended_recipient,
            swarm_instance_id: inner.swarm_instance_id,
            recipient_session_epoch,
            delivery_seq,
            timestamp_secs,
            round_id: inner.round_id,
            inner_digest,
            inner_length,
            wrapper_signature: Vec::new(),
        }
        .sign_with(auth);
        Self { wrapper, inner }
    }

    /// Serialize this delivery to postcard bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Deserialize from bytes enforcing `max_delivery_bytes` BEFORE decoding.
    pub fn from_bytes_bounded(
        bytes: &[u8],
        max_delivery_bytes: usize,
    ) -> Result<Self, GossipVerifyError> {
        if bytes.len() > max_delivery_bytes {
            return Err(GossipVerifyError::FrameTooLarge {
                len: bytes.len(),
                max: max_delivery_bytes,
            });
        }
        postcard::from_bytes(bytes).map_err(|_| GossipVerifyError::MalformedDelivery)
    }

    /// Verify recipient binding, structural consistency, cohort allowlist, replay window,
    /// wrapper signature, and originator inner signature (`ADR-0022` §3.3, `W10-03`).
    ///
    /// Note that `ReplayProtection` state is mutated **only after** both the wrapper signature
    /// and the inner originator signature verify cleanly, preventing unauthenticated state pollution.
    pub fn verify_for_recipient(
        &self,
        local_recipient: PeerId,
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
    ) -> Result<PeerId, GossipVerifyError> {
        if self.wrapper.version != RecipientWrapperV1::VERSION_V1 {
            return Err(GossipVerifyError::UnsupportedVersion {
                major: self.wrapper.version.0,
                minor: self.wrapper.version.1,
            });
        }
        if self.wrapper.intended_recipient != local_recipient {
            return Err(GossipVerifyError::RecipientMismatch {
                expected: local_recipient,
                found: self.wrapper.intended_recipient,
            });
        }
        if self.wrapper.originator_public_key != self.inner.originator_public_key {
            return Err(GossipVerifyError::OriginatorKeyMismatch);
        }
        if self.wrapper.swarm_instance_id != self.inner.swarm_instance_id {
            return Err(GossipVerifyError::SwarmInstanceMismatch);
        }
        if self.wrapper.round_id != self.inner.round_id {
            return Err(GossipVerifyError::RoundMismatch {
                wrapper_round: self.wrapper.round_id,
                inner_round: self.inner.round_id,
            });
        }
        if self.wrapper.inner_length as usize != self.inner.encoded_payload.len() {
            return Err(GossipVerifyError::InnerLengthMismatch {
                expected: self.wrapper.inner_length,
                found: self.inner.encoded_payload.len(),
            });
        }

        // Stateless timestamp check + cohort authorization + epoch continuity BEFORE crypto or state mutation.
        replay_guard
            .check_timestamp_only(self.wrapper.timestamp_secs, current_time_secs)
            .map_err(GossipVerifyError::Replay)?;
        let originator = PeerId::try_from_public_key_bytes(&self.wrapper.originator_public_key)
            .map_err(|_| GossipVerifyError::InvalidOriginatorKey)?;
        replay_guard
            .check_peer_authorized(&originator)
            .map_err(GossipVerifyError::Replay)?;
        let record_key = ReplayRecordKey::new(
            self.wrapper.swarm_instance_id,
            originator,
            self.wrapper.recipient_session_epoch,
        );
        replay_guard
            .check_epoch_continuity(&record_key)
            .map_err(GossipVerifyError::Replay)?;

        // Verify inner digest matches wrapper binding.
        let computed_inner_digest = self.inner.canonical_digest();
        if computed_inner_digest != self.wrapper.inner_digest {
            return Err(GossipVerifyError::InnerDigestMismatch);
        }

        // Verify both Ed25519 signatures BEFORE mutating replay_guard state!
        let _ = self.wrapper.verify_wrapper_signature()?;
        let _ = self.inner.verify_originator_signature()?;

        // Commit sequence number into replay_guard.
        replay_guard
            .validate_sequence_scoped_at(
                &record_key,
                self.wrapper.delivery_seq,
                self.wrapper.timestamp_secs,
                current_time_secs,
            )
            .map_err(GossipVerifyError::Replay)?;

        Ok(originator)
    }
}

#[cfg(feature = "alloc")]
impl BundledGossipDeliveryV2 {
    /// Construct and sign both [`InnerUpdateV2`] and [`RecipientWrapperV2`] using the originator's
    /// [`MessageAuth`].
    #[allow(clippy::too_many_arguments)]
    pub fn new_signed(
        auth: &MessageAuth,
        mut inner: InnerUpdateV2,
        intended_recipient: PeerId,
        session_id: [u8; 32],
        recipient_session_epoch: u64,
        delivery_seq: u64,
        timestamp_secs: u32,
    ) -> Self {
        inner = inner.sign_with(auth);
        let inner_digest = inner.canonical_digest();
        let inner_length = inner.encoded_payload.len() as u32;
        let wrapper = RecipientWrapperV2 {
            version: RecipientWrapperV2::VERSION_V2,
            originator_public_key: inner.originator_public_key,
            intended_recipient,
            swarm_instance_id: inner.swarm_instance_id,
            run_id: inner.run_id,
            session_id,
            recipient_session_epoch,
            delivery_seq,
            timestamp_secs,
            round_id: inner.round_id,
            inner_digest,
            inner_length,
            wrapper_signature: Vec::new(),
        }
        .sign_with(auth);
        Self { wrapper, inner }
    }

    /// Serialize this v2 delivery to postcard bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, postcard::Error> {
        postcard::to_allocvec(self)
    }

    /// Deserialize from bytes enforcing `max_delivery_bytes` BEFORE decoding.
    pub fn from_bytes_bounded(
        bytes: &[u8],
        max_delivery_bytes: usize,
    ) -> Result<Self, GossipVerifyError> {
        if bytes.len() > max_delivery_bytes {
            return Err(GossipVerifyError::FrameTooLarge {
                len: bytes.len(),
                max: max_delivery_bytes,
            });
        }
        postcard::from_bytes(bytes).map_err(|_| GossipVerifyError::MalformedDelivery)
    }

    /// Read-only verification & replay preflight (`ST-01`, `ST-03`).
    ///
    /// Returns `(verified_originator_peer, PreparedReplayAdmission)` with ZERO mutation to
    /// `replay_guard`, allowing downstream coordinator stages to validate round context before
    /// consuming replay state.
    pub fn verify_for_recipient_v2_prepared(
        &self,
        local_recipient: PeerId,
        replay_guard: &ReplayProtection,
        current_time_secs: u32,
    ) -> Result<(PeerId, PreparedReplayAdmission), GossipVerifyError> {
        if self.wrapper.version != RecipientWrapperV2::VERSION_V2 {
            return Err(GossipVerifyError::UnsupportedVersion {
                major: self.wrapper.version.0,
                minor: self.wrapper.version.1,
            });
        }
        if self.wrapper.intended_recipient != local_recipient {
            return Err(GossipVerifyError::RecipientMismatch {
                expected: local_recipient,
                found: self.wrapper.intended_recipient,
            });
        }
        if self.wrapper.originator_public_key != self.inner.originator_public_key {
            return Err(GossipVerifyError::OriginatorKeyMismatch);
        }
        if self.wrapper.swarm_instance_id != self.inner.swarm_instance_id {
            return Err(GossipVerifyError::SwarmInstanceMismatch);
        }
        if self.wrapper.run_id != self.inner.run_id {
            return Err(GossipVerifyError::RunIdMismatch);
        }
        if self.wrapper.round_id != self.inner.round_id {
            return Err(GossipVerifyError::RoundMismatch {
                wrapper_round: self.wrapper.round_id,
                inner_round: self.inner.round_id,
            });
        }
        if self.wrapper.inner_length as usize != self.inner.encoded_payload.len() {
            return Err(GossipVerifyError::InnerLengthMismatch {
                expected: self.wrapper.inner_length,
                found: self.inner.encoded_payload.len(),
            });
        }

        // Stateless timestamp/floor check, peer authorization, and v2 session continuity BEFORE crypto.
        replay_guard
            .check_timestamp_only(self.wrapper.timestamp_secs, current_time_secs)
            .map_err(GossipVerifyError::Replay)?;
        let originator = PeerId::try_from_public_key_bytes(&self.wrapper.originator_public_key)
            .map_err(|_| GossipVerifyError::InvalidOriginatorKey)?;
        replay_guard
            .check_peer_authorized(&originator)
            .map_err(GossipVerifyError::Replay)?;

        let record_key = ReplayRecordKeyV2::new(
            self.wrapper.swarm_instance_id,
            self.wrapper.run_id,
            originator,
            self.wrapper.recipient_session_epoch,
            self.wrapper.session_id,
        );
        replay_guard
            .check_v2_session_continuity(&record_key)
            .map_err(GossipVerifyError::Replay)?;

        // Verify inner digest binding + both v2 Ed25519 signatures.
        let computed_inner_digest = self.inner.canonical_digest();
        if computed_inner_digest != self.wrapper.inner_digest {
            return Err(GossipVerifyError::InnerDigestMismatch);
        }
        let _ = self.wrapper.verify_wrapper_signature()?;
        let _ = self.inner.verify_originator_signature()?;

        let prepared = replay_guard
            .check_v2_admission_readonly(
                &record_key,
                self.wrapper.delivery_seq,
                self.wrapper.timestamp_secs,
                current_time_secs,
            )
            .map_err(GossipVerifyError::Replay)?;
        Ok((originator, prepared))
    }

    /// End-to-end atomic verification and replay admission for v2 deliveries (`ST-03`).
    pub fn verify_for_recipient_v2(
        &self,
        local_recipient: PeerId,
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
    ) -> Result<PeerId, GossipVerifyError> {
        let (originator, prepared) = self.verify_for_recipient_v2_prepared(
            local_recipient,
            replay_guard,
            current_time_secs,
        )?;
        replay_guard
            .commit_prepared_admission(prepared)
            .map_err(GossipVerifyError::Replay)?;
        Ok(originator)
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use alloc::vec;
    use swarm_torch_core::crypto::{KeyPair, SessionHandshakeChallengeV2};
    use swarm_torch_core::replay::{CohortAllowlist, ReplayStartupPolicyV1};

    #[test]
    fn bundled_gossip_delivery_relays_intact_and_rejects_cross_recipient_or_relay_tamper() {
        let orig_kp = KeyPair::from_seed([0x11; 32]).unwrap();
        let orig_peer = orig_kp.peer_id();
        let orig_auth = MessageAuth::new(orig_kp.clone());

        let coord_kp = KeyPair::from_seed([0x22; 32]).unwrap();
        let coord_peer = coord_kp.peer_id();

        let other_kp = KeyPair::from_seed([0x33; 32]).unwrap();
        let other_peer = other_kp.peer_id();

        let inner = InnerUpdateV1 {
            originator_public_key: *orig_kp.public_key(),
            swarm_instance_id: [0xAB; 16],
            round_id: 3,
            contribution_id: 1,
            base_checkpoint_digest: [0xCD; 32],
            payload_kind: PayloadKindV1::ModelDeltaV1,
            compression_schema_version: 0,
            num_elements: 4,
            reported_examples: 128,
            encoded_payload: vec![1, 2, 3, 4],
            originator_signature: Vec::new(),
        };

        let delivery =
            BundledGossipDeliveryV1::new_signed(&orig_auth, inner, coord_peer, 1, 1, 500);
        let wire = delivery.to_bytes().unwrap();

        // Relay peer forwards `wire` unchanged to `coord_peer`.
        let decoded =
            BundledGossipDeliveryV1::from_bytes_bounded(&wire, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES)
                .unwrap();

        let allowlist = CohortAllowlist::from_peers([orig_peer]);
        let mut coord_replay = ReplayProtection::try_with_config(64, 60)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(allowlist.clone());

        // Cross-recipient replay against `other_peer` must fail closed!
        let mut other_replay = ReplayProtection::try_with_config(64, 60)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(allowlist);
        assert_eq!(
            decoded.verify_for_recipient(other_peer, &mut other_replay, 500),
            Err(GossipVerifyError::RecipientMismatch {
                expected: other_peer,
                found: coord_peer,
            })
        );

        // Tampering with inner payload by relay peer must fail closed without polluting replay_guard!
        let mut tampered = decoded.clone();
        tampered.inner.encoded_payload[0] ^= 0xFF;
        assert_eq!(
            tampered.verify_for_recipient(coord_peer, &mut coord_replay, 500),
            Err(GossipVerifyError::InnerDigestMismatch)
        );

        // Untampered delivery verifies cleanly at `coord_peer` and returns `orig_peer`!
        let verified_orig = decoded
            .verify_for_recipient(coord_peer, &mut coord_replay, 500)
            .unwrap();
        assert_eq!(verified_orig, orig_peer);
    }

    #[test]
    fn bundled_gossip_delivery_v2_binds_run_id_session_id_and_readonly_prepare() {
        let orig_kp = KeyPair::from_seed([0x41; 32]).unwrap();
        let orig_peer = orig_kp.peer_id();
        let orig_auth = MessageAuth::new(orig_kp.clone());
        let coord_kp = KeyPair::from_seed([0x42; 32]).unwrap();
        let coord_peer = coord_kp.peer_id();

        let swarm_id = [0xAA; 16];
        let run_id = [0xBB; 16];
        let policy_digest = [0xCC; 32];
        let now = 1_000u32;

        let mut replay = ReplayProtection::try_with_config(64, 60)
            .unwrap()
            .with_cohort_allowlist(CohortAllowlist::from_peers([orig_peer]));

        let challenge = SessionHandshakeChallengeV2 {
            swarm_instance_id: swarm_id,
            run_id,
            policy_digest,
            receiver_peer: coord_peer,
            sender_peer: orig_peer,
            requested_epoch: 1,
            initial_sequence: 0,
            receiver_nonce: [0x99; 32],
            issued_at_secs: now,
        };
        replay.register_recovery_challenge_v2(challenge).unwrap();
        let hs_sig = orig_auth.sign_session_handshake_v2(&challenge).unwrap();
        let active_key = replay
            .complete_recovery_handshake_v2(swarm_id, run_id, orig_kp.public_key(), &hs_sig, now)
            .unwrap();

        let inner = InnerUpdateV2 {
            originator_public_key: *orig_kp.public_key(),
            swarm_instance_id: swarm_id,
            run_id,
            round_id: 7,
            contribution_id: 2,
            base_checkpoint_digest: [0xDD; 32],
            context_policy_digest: policy_digest,
            payload_kind: PayloadKindV1::ModelDeltaV1,
            compression_schema_version: 0,
            num_elements: 4,
            reported_examples: 64,
            encoded_payload: vec![10, 20, 30, 40],
            originator_signature: Vec::new(),
        };
        let delivery = BundledGossipDeliveryV2::new_signed(
            &orig_auth,
            inner,
            coord_peer,
            active_key.session_id,
            1,
            1,
            now,
        );
        let wire = delivery.to_bytes().unwrap();
        let decoded =
            BundledGossipDeliveryV2::from_bytes_bounded(&wire, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES)
                .unwrap();

        // Wrong session_id must fail closed before consuming sequence state.
        let mut wrong_sid = decoded.clone();
        wrong_sid.wrapper.session_id = [0xEE; 32];
        wrong_sid.wrapper = wrong_sid.wrapper.sign_with(&orig_auth);
        assert_eq!(
            wrong_sid.verify_for_recipient_v2(coord_peer, &mut replay, now),
            Err(GossipVerifyError::Replay(ReplayError::SessionIdMismatch {
                peer: orig_peer
            }))
        );

        // Read-only prepare succeeds without consuming seq=1, allowing commit once.
        let (peer_prepared, prepared) = decoded
            .verify_for_recipient_v2_prepared(coord_peer, &replay, now)
            .unwrap();
        assert_eq!(peer_prepared, orig_peer);
        let _ = prepared;

        assert_eq!(
            decoded
                .verify_for_recipient_v2(coord_peer, &mut replay, now)
                .unwrap(),
            orig_peer
        );
        // Replaying seq=1 after commit fails.
        assert_eq!(
            decoded.verify_for_recipient_v2(coord_peer, &mut replay, now),
            Err(GossipVerifyError::Replay(ReplayError::Replay {
                peer: orig_peer,
                seq: 1,
            }))
        );
    }
}
