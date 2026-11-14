//! QUIC (`Iroh`) 1-RTT stream & `Zenoh` key-expression carrier security contracts (`ADR-0022` §4, `W12-02`).
//!
//! Enforces strict separation between **transport link principals** (QUIC/Iroh `EndpointId`, Zenoh
//! session) and **application-layer cryptographic contributor identity** (`PeerId` derived from
//! Ed25519 `originator_public_key`):
//! - Rejects QUIC 0-RTT / Half-RTT early data (`QuicHandshakeStageV1::OneRttComplete` required) so
//!   unconfirmed packets never reach replay, EF21, or coordinator state.
//! - Rejects ALPN mismatches against [`ALPN_SWARMTORCH_DELIVERY_V1`] (`b"swarmtorch/delivery/v1"`).
//! - Separates `DecoupledRelayAllowed` (relays may forward updates for multiple originators) from
//!   `StrictOneToOne` (direct peer-to-peer `link_peer_id == originator_peer_id`).
//! - Enforces canonical Zenoh topic structure
//!   `swarmtorch/v1/<scope_hex>/round/<round_id>/peer/<originator_hex>/update`, rejecting
//!   wildcards (`*`, `**`, `?`, `$`), ROS 2 `rmw_zenoh` prefixes (`rt/`, `rq/`, `rr/`), non-canonical
//!   hex/integers, and topic-vs-envelope identity mismatches.

#[cfg(feature = "alloc")]
use alloc::{format, string::String, vec::Vec};

#[cfg(feature = "alloc")]
use crate::gossip::{BundledGossipDeliveryV1, BundledGossipDeliveryV2};
#[cfg(feature = "alloc")]
use crate::protocol::{MessageEnvelope, DEFAULT_MAX_ENVELOPE_BYTES};
use swarm_torch_core::crypto::{
    sha256_slices, CryptoError, MessageAuth, Signature, LINK_ATTESTATION_DOMAIN_TAG_V1,
};
use swarm_torch_core::traits::PeerId;

/// Canonical ALPN protocol identifier negotiated on QUIC / Iroh connections (`ADR-0022` §4).
pub const ALPN_SWARMTORCH_DELIVERY_V1: &[u8] = b"swarmtorch/delivery/v1";

/// Canonical Zenoh key-expression prefix (`ADR-0022` §4).
pub const ZENOH_TOPIC_PREFIX_V1: &str = "swarmtorch/v1";

/// Default maximum length-prefixed QUIC stream frame size (`1,048,576` bytes = 1 MiB).
pub const DEFAULT_MAX_CARRIER_FRAME_BYTES: usize = 1_048_576;

/// Negotiated QUIC/TLS handshake stage observed at stream acceptance (`ADR-0022` §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuicHandshakeStageV1 {
    /// 0-RTT early data prior to handshake completion (MUST be rejected).
    ZeroRtt,
    /// Server-side half-RTT before client Finished verification (MUST be rejected).
    HalfRtt,
    /// Full mutual TLS 1.3 1-RTT handshake verified (`OneRttComplete`).
    OneRttComplete,
}

/// Transport link identity vs. application originator binding policy (`ADR-0022` §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LinkBindingPolicyV1 {
    /// Multi-hop relays may forward `InnerUpdateV1` / `RecipientWrapperV1` frames belonging to
    /// distinct cryptographic originators. Pre-auth rate limits attribute to the link principal
    /// while round attribution always uses the verified `originator_peer_id`.
    #[default]
    DecoupledRelayAllowed,
    /// Direct point-to-point mode requiring the authenticated QUIC/Iroh link principal to match
    /// the inner Ed25519 originator `PeerId`.
    StrictOneToOne,
}

/// Authenticated transport carrier descriptor (`ADR-0022` §4, `ST-08`).
///
/// Fields are private so unauthenticated callers cannot mint forged link principals without
/// verifying a signed handshake attestation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedLinkPrincipalV1 {
    /// Raw 32-byte transport public key / Iroh `EndpointId` authenticated by TLS/QUIC.
    endpoint_public_key: [u8; 32],
    /// Canonical SwarmTorch `PeerId` derived from `endpoint_public_key` (`SHA-256`).
    link_peer_id: PeerId,
}

/// Signed QUIC/TLS link-handshake attestation binding an endpoint key, negotiated ALPN, and
/// non-zero receiver challenge nonce (`ST-08`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkHandshakeAttestationV1<'a> {
    /// Raw 32-byte Ed25519 endpoint public key authenticated by TLS/QUIC.
    pub endpoint_public_key: [u8; 32],
    /// Negotiated ALPN protocol bytes carried under the handshake attestation.
    pub negotiated_alpn: &'a [u8],
    /// Non-zero receiver challenge nonce preventing replay or unauthenticated minting.
    pub receiver_nonce: [u8; 16],
    /// Ed25519 signature over `link_attestation_digest` under `LINK_ATTESTATION_DOMAIN_TAG_V1`.
    pub signature: Signature,
}

/// Compute the canonical SHA-256 attestation digest binding endpoint key, ALPN, and receiver nonce.
pub fn link_attestation_digest(
    endpoint_public_key: &[u8; 32],
    negotiated_alpn: &[u8],
    receiver_nonce: &[u8; 16],
) -> [u8; 32] {
    let alpn_len = (negotiated_alpn.len() as u64).to_le_bytes();
    sha256_slices(&[
        endpoint_public_key,
        &alpn_len,
        negotiated_alpn,
        receiver_nonce,
    ])
}

/// Sign a [`LinkHandshakeAttestationV1`] using the transport endpoint's [`MessageAuth`].
///
/// Fails with [`CarrierError::InvalidLinkAttestation`] if `receiver_nonce` is all-zero or
/// `negotiated_alpn` is empty.
pub fn sign_link_attestation<'a>(
    auth: &MessageAuth,
    negotiated_alpn: &'a [u8],
    receiver_nonce: [u8; 16],
) -> Result<LinkHandshakeAttestationV1<'a>, CarrierError> {
    if receiver_nonce == [0u8; 16] || negotiated_alpn.is_empty() {
        return Err(CarrierError::InvalidLinkAttestation);
    }
    let endpoint_public_key = *auth.key_pair().public_key();
    let digest = link_attestation_digest(&endpoint_public_key, negotiated_alpn, &receiver_nonce);
    let signature = auth.sign_prehashed_domain(LINK_ATTESTATION_DOMAIN_TAG_V1, &digest);
    Ok(LinkHandshakeAttestationV1 {
        endpoint_public_key,
        negotiated_alpn,
        receiver_nonce,
        signature,
    })
}

impl VerifiedLinkPrincipalV1 {
    /// Raw 32-byte transport endpoint public key / Iroh `EndpointId`.
    pub const fn endpoint_public_key(&self) -> &[u8; 32] {
        &self.endpoint_public_key
    }

    /// Canonical SwarmTorch `PeerId` derived from `endpoint_public_key` (`SHA-256`).
    pub const fn link_peer_id(&self) -> PeerId {
        self.link_peer_id
    }

    /// Construct a [`VerifiedLinkPrincipalV1`] strictly after verifying a signed
    /// [`LinkHandshakeAttestationV1`] against `expected_receiver_nonce` (`ST-08`).
    pub fn from_link_attestation(
        expected_receiver_nonce: &[u8; 16],
        attestation: &LinkHandshakeAttestationV1<'_>,
    ) -> Result<Self, CarrierError> {
        let link_peer_id = PeerId::try_from_public_key_bytes(&attestation.endpoint_public_key)
            .map_err(|_| CarrierError::InvalidLinkPrincipalKey)?;
        if *expected_receiver_nonce == [0u8; 16]
            || attestation.receiver_nonce == [0u8; 16]
            || attestation.receiver_nonce != *expected_receiver_nonce
            || attestation.negotiated_alpn.is_empty()
        {
            return Err(CarrierError::InvalidLinkAttestation);
        }
        let digest = link_attestation_digest(
            &attestation.endpoint_public_key,
            attestation.negotiated_alpn,
            &attestation.receiver_nonce,
        );
        MessageAuth::verify_prehashed_domain(
            &attestation.endpoint_public_key,
            LINK_ATTESTATION_DOMAIN_TAG_V1,
            &digest,
            &attestation.signature,
        )
        .map_err(|_| CarrierError::InvalidLinkAttestation)?;
        Ok(Self {
            endpoint_public_key: attestation.endpoint_public_key,
            link_peer_id,
        })
    }
}

/// Carrier-verified peer admission outcome separating transport hop from cryptographic originator (`ST-08`).
///
/// Fields are private so callers cannot construct bindings outside handshake policy checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedCarrierBindingV1 {
    /// Verified transport link principal (`EndpointId` / QUIC peer).
    link_principal: VerifiedLinkPrincipalV1,
    /// Cryptographically authenticated message originator (`PeerId`).
    originator_peer_id: PeerId,
    /// Applied binding policy.
    policy: LinkBindingPolicyV1,
    /// True if `link_principal.link_peer_id() != originator_peer_id` (forwarded via relay).
    relayed_via_intermediary: bool,
}

impl VerifiedCarrierBindingV1 {
    /// Verified transport link principal (`EndpointId` / QUIC peer).
    pub const fn link_principal(&self) -> VerifiedLinkPrincipalV1 {
        self.link_principal
    }

    /// Cryptographically authenticated message originator (`PeerId`).
    pub const fn originator_peer_id(&self) -> PeerId {
        self.originator_peer_id
    }

    /// Applied link-binding policy.
    pub const fn policy(&self) -> LinkBindingPolicyV1 {
        self.policy
    }

    /// True if `link_principal.link_peer_id() != originator_peer_id` (forwarded via relay).
    pub const fn relayed_via_intermediary(&self) -> bool {
        self.relayed_via_intermediary
    }
}

/// Errors returned by carrier handshake, framing, and topic binding verification (`ADR-0022` §4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CarrierError {
    /// Negotiated ALPN did not equal [`ALPN_SWARMTORCH_DELIVERY_V1`].
    AlpnMismatch,
    /// Packet or stream arrived prior to `QuicHandshakeStageV1::OneRttComplete` (0-RTT/Half-RTT rejected).
    EarlyDataRejected,
    /// Link endpoint key was all-zero or failed Ed25519 point validation.
    InvalidLinkPrincipalKey,
    /// Link handshake attestation failed nonce binding, ALPN non-emptiness, or signature verification.
    InvalidLinkAttestation,
    /// `StrictOneToOne` policy rejected a frame whose originator differs from the link principal.
    StrictBindingMismatch {
        link_peer: PeerId,
        originator: PeerId,
    },
    /// Length prefix exceeded configured `max_frame_bytes` prior to allocation.
    FrameTooLarge { len: usize, max: usize },
    /// Zero-length or truncated length-prefixed QUIC stream frame.
    EmptyOrTruncatedFrame,
    /// Zenoh topic string violated the canonical `swarmtorch/v1/...` grammar or contained wildcards/ROS 2 prefixes.
    InvalidZenohTopic,
    /// Authenticated envelope/gossip scope, round, or originator differed from the Zenoh topic key.
    TopicIdentityMismatch,
    /// Underlying originator Ed25519 key validation failed.
    Crypto(CryptoError),
}

impl core::fmt::Display for CarrierError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AlpnMismatch => write!(
                f,
                "carrier ALPN mismatch: expected 'swarmtorch/delivery/v1'"
            ),
            Self::EarlyDataRejected => write!(
                f,
                "QUIC early data (0-RTT / Half-RTT) rejected: 1-RTT completion required"
            ),
            Self::InvalidLinkPrincipalKey => {
                write!(f, "invalid or all-zero QUIC link principal public key")
            }
            Self::InvalidLinkAttestation => write!(
                f,
                "invalid link handshake attestation (missing nonce, mismatch, or bad signature)"
            ),
            Self::StrictBindingMismatch {
                link_peer,
                originator,
            } => write!(
                f,
                "strict 1:1 binding mismatch: link peer {link_peer:?} != originator {originator:?}"
            ),
            Self::FrameTooLarge { len, max } => {
                write!(f, "carrier frame length {len} exceeds maximum {max}")
            }
            Self::EmptyOrTruncatedFrame => write!(f, "empty or truncated carrier stream frame"),
            Self::InvalidZenohTopic => write!(f, "invalid or unsafe Zenoh topic key expression"),
            Self::TopicIdentityMismatch => write!(
                f,
                "Zenoh topic (scope, round, peer) does not match authenticated envelope identity"
            ),
            Self::Crypto(e) => write!(f, "carrier cryptographic error: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CarrierError {}

impl From<CryptoError> for CarrierError {
    fn from(value: CryptoError) -> Self {
        Self::Crypto(value)
    }
}

/// Policy validator for incoming QUIC / Iroh 1-RTT connections and stream bindings (`ADR-0022` §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuicStreamHandshakePolicyV1 {
    /// Expected ALPN identifier (defaults to [`ALPN_SWARMTORCH_DELIVERY_V1`]).
    pub expected_alpn: &'static [u8],
    /// Policy governing transport `EndpointId` vs. inner originator `PeerId`.
    pub binding_policy: LinkBindingPolicyV1,
    /// Maximum serialized delivery frame size accepted on a stream.
    pub max_frame_bytes: usize,
}

impl Default for QuicStreamHandshakePolicyV1 {
    fn default() -> Self {
        Self {
            expected_alpn: ALPN_SWARMTORCH_DELIVERY_V1,
            binding_policy: LinkBindingPolicyV1::DecoupledRelayAllowed,
            max_frame_bytes: DEFAULT_MAX_CARRIER_FRAME_BYTES,
        }
    }
}

impl QuicStreamHandshakePolicyV1 {
    /// Validate negotiated ALPN, 1-RTT handshake completion, and signed endpoint attestation
    /// against `expected_receiver_nonce` before accepting frames (`ST-08`).
    pub fn validate_attested_connection(
        &self,
        expected_receiver_nonce: &[u8; 16],
        handshake_stage: QuicHandshakeStageV1,
        attestation: &LinkHandshakeAttestationV1<'_>,
    ) -> Result<VerifiedLinkPrincipalV1, CarrierError> {
        if attestation.negotiated_alpn != self.expected_alpn {
            return Err(CarrierError::AlpnMismatch);
        }
        if handshake_stage != QuicHandshakeStageV1::OneRttComplete {
            return Err(CarrierError::EarlyDataRejected);
        }
        VerifiedLinkPrincipalV1::from_link_attestation(expected_receiver_nonce, attestation)
    }

    /// Enforce link-binding policy without minting a public binding proof (`ST-08`).
    ///
    /// Returns `Ok(true)` if forwarded via an intermediary (`link_peer != authenticated_originator`).
    pub fn check_link_binding(
        &self,
        link_peer: PeerId,
        authenticated_originator: PeerId,
    ) -> Result<bool, CarrierError> {
        let relayed = link_peer != authenticated_originator;
        if matches!(self.binding_policy, LinkBindingPolicyV1::StrictOneToOne) && relayed {
            return Err(CarrierError::StrictBindingMismatch {
                link_peer,
                originator: authenticated_originator,
            });
        }
        Ok(relayed)
    }

    /// Bind an already-verified transport connection to an authenticated message originator `PeerId`.
    pub fn bind_originator(
        &self,
        link_principal: VerifiedLinkPrincipalV1,
        authenticated_originator: PeerId,
    ) -> Result<VerifiedCarrierBindingV1, CarrierError> {
        let relayed =
            self.check_link_binding(link_principal.link_peer_id(), authenticated_originator)?;
        Ok(VerifiedCarrierBindingV1 {
            link_principal,
            originator_peer_id: authenticated_originator,
            policy: self.binding_policy,
            relayed_via_intermediary: relayed,
        })
    }
}

/// Bounded `[u32_le(len) || payload]` framing codec for QUIC bi-streams (`ADR-0022` §4).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuicLengthPrefixedCodecV1 {
    /// Maximum frame payload byte size enforced BEFORE allocation.
    pub max_frame_bytes: usize,
}

#[cfg(feature = "alloc")]
impl Default for QuicLengthPrefixedCodecV1 {
    fn default() -> Self {
        Self {
            max_frame_bytes: DEFAULT_MAX_CARRIER_FRAME_BYTES,
        }
    }
}

#[cfg(feature = "alloc")]
impl QuicLengthPrefixedCodecV1 {
    /// Encode a length-prefixed QUIC stream frame `[u32_le(len) || payload]`.
    pub fn encode_frame(&self, payload: &[u8]) -> Result<Vec<u8>, CarrierError> {
        if payload.is_empty() {
            return Err(CarrierError::EmptyOrTruncatedFrame);
        }
        if payload.len() > self.max_frame_bytes {
            return Err(CarrierError::FrameTooLarge {
                len: payload.len(),
                max: self.max_frame_bytes,
            });
        }
        let len_u32 = u32::try_from(payload.len()).map_err(|_| CarrierError::FrameTooLarge {
            len: payload.len(),
            max: self.max_frame_bytes,
        })?;
        let mut out = Vec::with_capacity(4 + payload.len());
        out.extend_from_slice(&len_u32.to_le_bytes());
        out.extend_from_slice(payload);
        Ok(out)
    }

    /// Decode one length-prefixed frame from the head of `stream_bytes`, returning
    /// `(payload_slice, remaining_bytes)` with zero over-allocation.
    pub fn decode_next_frame<'a>(
        &self,
        stream_bytes: &'a [u8],
    ) -> Result<(&'a [u8], &'a [u8]), CarrierError> {
        if stream_bytes.len() < 4 {
            return Err(CarrierError::EmptyOrTruncatedFrame);
        }
        let len = u32::from_le_bytes([
            stream_bytes[0],
            stream_bytes[1],
            stream_bytes[2],
            stream_bytes[3],
        ]) as usize;
        if len == 0 {
            return Err(CarrierError::EmptyOrTruncatedFrame);
        }
        if len > self.max_frame_bytes {
            return Err(CarrierError::FrameTooLarge {
                len,
                max: self.max_frame_bytes,
            });
        }
        let body = &stream_bytes[4..];
        if body.len() < len {
            return Err(CarrierError::EmptyOrTruncatedFrame);
        }
        Ok((&body[..len], &body[len..]))
    }
}

/// Canonical SwarmTorch Zenoh topic descriptor (`ADR-0022` §4, `W12-02`):
/// `swarmtorch/v1/<scope_hex>/round/<round_id>/peer/<originator_hex>/update`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ZenohTopicContractV1 {
    /// 16-byte scoped swarm deployment identifier (`<scope_hex>` = 32 lowercase hex chars).
    pub swarm_instance_id: [u8; 16],
    /// Canonical round identifier (`<round_id>`).
    pub round_id: u64,
    /// Canonical originator `PeerId` (`<originator_hex>` = 64 lowercase hex chars).
    pub originator_peer_id: PeerId,
}

#[cfg(feature = "alloc")]
impl ZenohTopicContractV1 {
    /// Construct a canonical topic descriptor.
    pub const fn new(
        swarm_instance_id: [u8; 16],
        round_id: u64,
        originator_peer_id: PeerId,
    ) -> Self {
        Self {
            swarm_instance_id,
            round_id,
            originator_peer_id,
        }
    }

    /// Format as canonical Zenoh key expression:
    /// `swarmtorch/v1/<scope_hex>/round/<round_id>/peer/<originator_hex>/update`.
    pub fn to_key_expr(&self) -> String {
        let scope_hex = encode_hex_lower(&self.swarm_instance_id);
        let peer_hex = encode_hex_lower(self.originator_peer_id.as_bytes());
        format!(
            "{}/v1/{}/round/{}/peer/{}/update",
            "swarmtorch", scope_hex, self.round_id, peer_hex
        )
    }

    /// Strictly parse and validate a concrete Zenoh publish topic string.
    ///
    /// Rejects:
    /// - Zenoh key-expression wildcards (`*`, `**`, `?`, `$`),
    /// - ROS 2 `rmw_zenoh` prefixes (`rt/`, `rq/`, `rr/`),
    /// - Control characters, whitespace, leading/trailing slashes, or duplicate slashes,
    /// - Non-canonical uppercase hex or leading-zero `round_id` aliases (`"01"`).
    pub fn parse_key_expr(topic: &str) -> Result<Self, CarrierError> {
        if topic.is_empty()
            || topic.len() > 256
            || topic.starts_with("rt/")
            || topic.starts_with("rq/")
            || topic.starts_with("rr/")
            || topic
                .bytes()
                .any(|b| b == b'*' || b == b'?' || b == b'$' || b.is_ascii_control() || b == b' ')
        {
            return Err(CarrierError::InvalidZenohTopic);
        }

        let parts: Vec<&str> = topic.split('/').collect();
        if parts.len() != 8
            || parts[0] != "swarmtorch"
            || parts[1] != "v1"
            || parts[3] != "round"
            || parts[5] != "peer"
            || parts[7] != "update"
        {
            return Err(CarrierError::InvalidZenohTopic);
        }

        let scope = decode_hex_lower_exact::<16>(parts[2])?;
        let round_str = parts[4];
        if round_str.is_empty()
            || (round_str.len() > 1 && round_str.starts_with('0'))
            || !round_str.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(CarrierError::InvalidZenohTopic);
        }
        let round_id = round_str
            .parse::<u64>()
            .map_err(|_| CarrierError::InvalidZenohTopic)?;

        let peer_bytes = decode_hex_lower_exact::<32>(parts[6])?;
        if peer_bytes == [0u8; 32] {
            return Err(CarrierError::InvalidZenohTopic);
        }

        Ok(Self {
            swarm_instance_id: scope,
            round_id,
            originator_peer_id: PeerId::new(peer_bytes),
        })
    }

    fn verify_topic_match_fields(
        &self,
        inner_scope: [u8; 16],
        wrapper_scope: [u8; 16],
        inner_round: u64,
        wrapper_round: u64,
        sender_peer: PeerId,
    ) -> Result<(), CarrierError> {
        if inner_scope != self.swarm_instance_id
            || inner_round != self.round_id
            || sender_peer != self.originator_peer_id
            || wrapper_scope != self.swarm_instance_id
            || wrapper_round != self.round_id
        {
            return Err(CarrierError::TopicIdentityMismatch);
        }
        Ok(())
    }

    /// Enforce that the Zenoh topic key agrees with an envelope's scoped v1/v2 `(swarm_instance_id, round_id, originator)`
    /// before admitting it into downstream signature/replay verification.
    pub fn verify_matches_envelope(&self, envelope: &MessageEnvelope) -> Result<(), CarrierError> {
        if !matches!(
            envelope.version,
            MessageEnvelope::VERSION_V1 | MessageEnvelope::VERSION_V2
        ) {
            return Err(CarrierError::TopicIdentityMismatch);
        }
        let sender_peer = envelope.sender_peer_id()?;
        if envelope.swarm_instance_id != self.swarm_instance_id
            || envelope.round_id != self.round_id
            || sender_peer != self.originator_peer_id
        {
            return Err(CarrierError::TopicIdentityMismatch);
        }
        Ok(())
    }

    /// Enforce that the Zenoh topic key agrees with a gossip [`BundledGossipDeliveryV1`].
    pub fn verify_matches_gossip_delivery(
        &self,
        delivery: &BundledGossipDeliveryV1,
    ) -> Result<(), CarrierError> {
        let sender_peer = delivery.inner.originator_peer_id()?;
        self.verify_topic_match_fields(
            delivery.inner.swarm_instance_id,
            delivery.wrapper.swarm_instance_id,
            delivery.inner.round_id,
            delivery.wrapper.round_id,
            sender_peer,
        )
    }

    /// Enforce that the Zenoh topic key agrees with a v2 gossip [`BundledGossipDeliveryV2`].
    pub fn verify_matches_gossip_delivery_v2(
        &self,
        delivery: &BundledGossipDeliveryV2,
    ) -> Result<(), CarrierError> {
        let sender_peer = delivery.inner.originator_peer_id()?;
        self.verify_topic_match_fields(
            delivery.inner.swarm_instance_id,
            delivery.wrapper.swarm_instance_id,
            delivery.inner.round_id,
            delivery.wrapper.round_id,
            sender_peer,
        )
    }
}

/// Convenience policy constant binding default max envelope bytes (`1 MiB`).
pub const fn default_max_carrier_bytes() -> usize {
    #[cfg(feature = "alloc")]
    {
        DEFAULT_MAX_ENVELOPE_BYTES
    }
    #[cfg(not(feature = "alloc"))]
    {
        DEFAULT_MAX_CARRIER_FRAME_BYTES
    }
}

#[cfg(feature = "alloc")]
fn encode_hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(feature = "alloc")]
fn decode_hex_lower_exact<const N: usize>(s: &str) -> Result<[u8; N], CarrierError> {
    if s.len() != N * 2 {
        return Err(CarrierError::InvalidZenohTopic);
    }
    let bytes = s.as_bytes();
    let mut out = [0u8; N];
    for i in 0..N {
        let hi = hex_nibble_lower(bytes[i * 2])?;
        let lo = hex_nibble_lower(bytes[i * 2 + 1])?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

#[cfg(feature = "alloc")]
fn hex_nibble_lower(b: u8) -> Result<u8, CarrierError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        _ => Err(CarrierError::InvalidZenohTopic),
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
    use crate::gossip::{BundledGossipDeliveryV2, InnerUpdateV2, PayloadKindV1};
    use crate::protocol::MessageType;
    use alloc::vec;
    use swarm_torch_core::crypto::KeyPair;

    #[test]
    fn quic_policy_rejects_early_data_alpn_mismatch_bad_attestation_and_enforces_binding_modes() {
        let relay_kp = KeyPair::from_seed([0x11; 32]).unwrap();
        let origin_kp = KeyPair::from_seed([0x22; 32]).unwrap();
        let relay_auth = MessageAuth::new(relay_kp.clone());
        let expected_nonce = [0x9A; 16];
        let valid_attestation =
            sign_link_attestation(&relay_auth, ALPN_SWARMTORCH_DELIVERY_V1, expected_nonce)
                .unwrap();

        let policy = QuicStreamHandshakePolicyV1::default();

        // Wrong ALPN rejected
        let wrong_alpn = LinkHandshakeAttestationV1 {
            negotiated_alpn: b"wrong/alpn",
            ..valid_attestation
        };
        assert_eq!(
            policy
                .validate_attested_connection(
                    &expected_nonce,
                    QuicHandshakeStageV1::OneRttComplete,
                    &wrong_alpn,
                )
                .unwrap_err(),
            CarrierError::AlpnMismatch
        );

        // 0-RTT and Half-RTT early data rejected
        for early in [QuicHandshakeStageV1::ZeroRtt, QuicHandshakeStageV1::HalfRtt] {
            assert_eq!(
                policy
                    .validate_attested_connection(&expected_nonce, early, &valid_attestation)
                    .unwrap_err(),
                CarrierError::EarlyDataRejected
            );
        }

        // All-zero link public key rejected
        let zero_key_attestation = LinkHandshakeAttestationV1 {
            endpoint_public_key: [0u8; 32],
            ..valid_attestation
        };
        assert_eq!(
            policy
                .validate_attested_connection(
                    &expected_nonce,
                    QuicHandshakeStageV1::OneRttComplete,
                    &zero_key_attestation,
                )
                .unwrap_err(),
            CarrierError::InvalidLinkPrincipalKey
        );

        // Zero receiver nonce, mismatched receiver nonce, empty ALPN, and tampered signature fail
        assert_eq!(
            sign_link_attestation(&relay_auth, ALPN_SWARMTORCH_DELIVERY_V1, [0u8; 16]).unwrap_err(),
            CarrierError::InvalidLinkAttestation
        );
        assert_eq!(
            sign_link_attestation(&relay_auth, b"", expected_nonce).unwrap_err(),
            CarrierError::InvalidLinkAttestation
        );
        let wrong_nonce = [0x7B; 16];
        assert_eq!(
            policy
                .validate_attested_connection(
                    &wrong_nonce,
                    QuicHandshakeStageV1::OneRttComplete,
                    &valid_attestation,
                )
                .unwrap_err(),
            CarrierError::InvalidLinkAttestation
        );
        let mut forged_sig = *valid_attestation.signature.as_bytes();
        forged_sig[0] ^= 0xFF;
        let bad_sig_attestation = LinkHandshakeAttestationV1 {
            signature: Signature::from_bytes(forged_sig),
            ..valid_attestation
        };
        assert_eq!(
            policy
                .validate_attested_connection(
                    &expected_nonce,
                    QuicHandshakeStageV1::OneRttComplete,
                    &bad_sig_attestation,
                )
                .unwrap_err(),
            CarrierError::InvalidLinkAttestation
        );

        // 1-RTT completes and allows relayed originator under DecoupledRelayAllowed
        let link = policy
            .validate_attested_connection(
                &expected_nonce,
                QuicHandshakeStageV1::OneRttComplete,
                &valid_attestation,
            )
            .unwrap();
        assert_eq!(link.endpoint_public_key(), relay_kp.public_key());
        assert_eq!(link.link_peer_id(), relay_kp.peer_id());
        assert!(policy
            .check_link_binding(link.link_peer_id(), origin_kp.peer_id())
            .unwrap());

        let relayed = policy.bind_originator(link, origin_kp.peer_id()).unwrap();
        assert!(relayed.relayed_via_intermediary());
        assert_eq!(relayed.originator_peer_id(), origin_kp.peer_id());
        assert_eq!(relayed.link_principal().link_peer_id(), relay_kp.peer_id());
        assert_eq!(relayed.policy(), LinkBindingPolicyV1::DecoupledRelayAllowed);

        // StrictOneToOne rejects relay forwarding, permits direct matching peer
        let strict_policy = QuicStreamHandshakePolicyV1 {
            binding_policy: LinkBindingPolicyV1::StrictOneToOne,
            ..Default::default()
        };
        assert!(matches!(
            strict_policy.check_link_binding(link.link_peer_id(), origin_kp.peer_id()),
            Err(CarrierError::StrictBindingMismatch { .. })
        ));
        assert!(matches!(
            strict_policy.bind_originator(link, origin_kp.peer_id()),
            Err(CarrierError::StrictBindingMismatch { .. })
        ));
        let direct = strict_policy
            .bind_originator(link, relay_kp.peer_id())
            .unwrap();
        assert!(!direct.relayed_via_intermediary());
    }

    #[test]
    fn quic_length_prefixed_codec_enforces_bounds_before_allocating() {
        let codec = QuicLengthPrefixedCodecV1 {
            max_frame_bytes: 64,
        };
        let frame1 = codec.encode_frame(b"hello").unwrap();
        let frame2 = codec.encode_frame(b"world!").unwrap();
        let mut combined = frame1;
        combined.extend_from_slice(&frame2);

        let (p1, rest) = codec.decode_next_frame(&combined).unwrap();
        assert_eq!(p1, b"hello");
        let (p2, tail) = codec.decode_next_frame(rest).unwrap();
        assert_eq!(p2, b"world!");
        assert!(tail.is_empty());

        // Oversized length prefix rejected BEFORE slice bounds
        let oversized_len = (1_000_000u32).to_le_bytes();
        assert_eq!(
            codec.decode_next_frame(&oversized_len).unwrap_err(),
            CarrierError::FrameTooLarge {
                len: 1_000_000,
                max: 64
            }
        );
    }

    #[test]
    fn zenoh_topic_contract_enforces_grammar_rejects_wildcards_ros2_and_spoofing() {
        let origin_kp = KeyPair::from_seed([0x33; 32]).unwrap();
        let other_kp = KeyPair::from_seed([0x44; 32]).unwrap();
        let scope = [0xab; 16];
        let topic = ZenohTopicContractV1::new(scope, 42, origin_kp.peer_id());
        let key_expr = topic.to_key_expr();
        let parsed = ZenohTopicContractV1::parse_key_expr(&key_expr).unwrap();
        assert_eq!(parsed, topic);

        // Reject wildcards, ROS 2 rmw_zenoh prefixes, leading zeros, and uppercase hex
        let bad_topics = [
            "swarmtorch/v1/*/round/42/peer/00/update",
            "swarmtorch/v1/**/update",
            "swarmtorch/v1/abababababababababababababababab/round/4?/peer/00/update",
            "rt/swarmtorch/v1/chatter",
            "rq/service/req",
            "rr/service/rep",
            "swarmtorch/v1/abababababababababababababababab/round/01/peer/0000000000000000000000000000000000000000000000000000000000000001/update",
            "swarmtorch/v1/ABABABABABABABABABABABABABABABAB/round/1/peer/0000000000000000000000000000000000000000000000000000000000000001/update",
        ];
        for bad in bad_topics {
            assert_eq!(
                ZenohTopicContractV1::parse_key_expr(bad).unwrap_err(),
                CarrierError::InvalidZenohTopic,
                "topic should fail: {bad}"
            );
        }

        // Verify topic-vs-envelope identity binding
        let good_env = MessageEnvelope::new_v1(
            *origin_kp.public_key(),
            scope,
            1,
            42,
            MessageType::GradientUpdate,
            vec![1, 2, 3],
        );
        parsed.verify_matches_envelope(&good_env).unwrap();

        let spoofed_sender_env = MessageEnvelope::new_v1(
            *other_kp.public_key(),
            scope,
            1,
            42,
            MessageType::GradientUpdate,
            vec![1, 2, 3],
        );
        assert_eq!(
            parsed
                .verify_matches_envelope(&spoofed_sender_env)
                .unwrap_err(),
            CarrierError::TopicIdentityMismatch
        );

        let wrong_round_env = MessageEnvelope::new_v1(
            *origin_kp.public_key(),
            scope,
            1,
            43,
            MessageType::GradientUpdate,
            vec![1, 2, 3],
        );
        assert_eq!(
            parsed
                .verify_matches_envelope(&wrong_round_env)
                .unwrap_err(),
            CarrierError::TopicIdentityMismatch
        );

        // Verify v2 envelopes and v2 gossip deliveries bind to Zenoh topic identity
        let good_env_v2 = MessageEnvelope::new_v2(
            *origin_kp.public_key(),
            scope,
            [0xcd; 16],
            1,
            [0x12; 32],
            42,
            MessageType::GradientUpdate,
            vec![1, 2, 3],
        );
        parsed.verify_matches_envelope(&good_env_v2).unwrap();

        let origin_auth = MessageAuth::new(origin_kp.clone());
        let delivery_v2 = BundledGossipDeliveryV2::new_signed(
            &origin_auth,
            InnerUpdateV2 {
                originator_public_key: *origin_kp.public_key(),
                swarm_instance_id: scope,
                run_id: [0xcd; 16],
                round_id: 42,
                contribution_id: 1,
                base_checkpoint_digest: [0x33; 32],
                context_policy_digest: [0x44; 32],
                payload_kind: PayloadKindV1::ModelDeltaV1,
                compression_schema_version: 0,
                num_elements: 1,
                reported_examples: 10,
                encoded_payload: vec![1, 2, 3, 4],
                originator_signature: Vec::new(),
            },
            other_kp.peer_id(),
            [0x12; 32],
            1,
            1,
            1_700_000_000,
        );
        parsed
            .verify_matches_gossip_delivery_v2(&delivery_v2)
            .unwrap();
        let mut wrong_round_v2 = delivery_v2.clone();
        wrong_round_v2.wrapper.round_id = 43;
        assert_eq!(
            parsed
                .verify_matches_gossip_delivery_v2(&wrong_round_v2)
                .unwrap_err(),
            CarrierError::TopicIdentityMismatch
        );
    }
}
