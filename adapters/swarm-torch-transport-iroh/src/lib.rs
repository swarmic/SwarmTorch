#![forbid(unsafe_code)]
//! Isolated QUIC / `Iroh` 1-RTT carrier adapter (`ADR-0022` §4, `W12-02`).

use std::collections::BTreeMap;

use swarm_torch_core::replay::ReplayProtection;
use swarm_torch_core::traits::PeerId;
use swarm_torch_net::carrier::{
    CarrierError, LinkBindingPolicyV1, QuicHandshakeStageV1, QuicLengthPrefixedCodecV1,
    QuicStreamHandshakePolicyV1, VerifiedCarrierBindingV1, VerifiedLinkPrincipalV1,
    ALPN_SWARMTORCH_DELIVERY_V1,
};
use swarm_torch_net::gossip::{
    BundledGossipDeliveryV1, GossipVerifyError, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
};
use swarm_torch_net::protocol::{MessageEnvelope, VerifyError};

/// Errors produced while decoding or authenticating an Iroh bi-stream frame.
#[derive(Debug)]
pub enum IrohCarrierError {
    /// Carrier ALPN, early-data, binding policy, or length-prefix contract failed.
    Carrier(CarrierError),
    /// Per-link principal rate limit exceeded before cryptographic admission.
    LinkRateLimited { link_peer: PeerId },
    /// Direct envelope verification failed.
    Envelope(VerifyError),
    /// Multi-hop gossip envelope verification failed.
    Gossip(GossipVerifyError),
}

impl From<CarrierError> for IrohCarrierError {
    fn from(value: CarrierError) -> Self {
        Self::Carrier(value)
    }
}

/// Verified frame received over an Iroh 1-RTT connection.
#[derive(Debug, Clone)]
pub struct AcceptedIrohEnvelopeV1 {
    /// Transport link principal vs. cryptographic originator binding.
    pub binding: VerifiedCarrierBindingV1,
    /// Authenticated message envelope.
    pub envelope: MessageEnvelope,
}

/// Verified gossip delivery frame received over an Iroh 1-RTT connection.
#[derive(Debug, Clone)]
pub struct AcceptedIrohGossipV1 {
    /// Transport link principal vs. cryptographic originator binding.
    pub binding: VerifiedCarrierBindingV1,
    /// Authenticated bundled gossip delivery (`InnerUpdateV1` + `RecipientWrapperV1`).
    pub delivery: BundledGossipDeliveryV1,
}

/// Per-link pre-auth frame budget tracker keyed strictly on the TLS/QUIC link principal (`PeerId`),
/// ensuring spoofed inner sender headers cannot poison another peer's quota (`ADR-0022` §4).
#[derive(Debug, Clone, Default)]
pub struct LinkFrameQuotaTrackerV1 {
    max_frames_per_link: u64,
    counts: BTreeMap<PeerId, u64>,
}

impl LinkFrameQuotaTrackerV1 {
    /// Construct a per-link frame quota tracker (`max_frames_per_link = 0` disables throttling).
    pub fn new(max_frames_per_link: u64) -> Self {
        Self {
            max_frames_per_link,
            counts: BTreeMap::new(),
        }
    }

    /// Record one frame arrival against `link_principal.link_peer_id`.
    pub fn record_frame(
        &mut self,
        link_principal: &VerifiedLinkPrincipalV1,
    ) -> Result<(), IrohCarrierError> {
        if self.max_frames_per_link == 0 {
            return Ok(());
        }
        let count = self.counts.entry(link_principal.link_peer_id).or_insert(0);
        if *count >= self.max_frames_per_link {
            return Err(IrohCarrierError::LinkRateLimited {
                link_peer: link_principal.link_peer_id,
            });
        }
        *count = count.saturating_add(1);
        Ok(())
    }
}

/// Adapter coordinating 1-RTT QUIC / Iroh handshake acceptance, stream framing, and delivery verification.
#[derive(Debug, Clone)]
pub struct IrohCarrierAdapterV1 {
    /// Policy governing ALPN, 1-RTT stage, max frame bytes, and link-vs-originator binding.
    pub policy: QuicStreamHandshakePolicyV1,
    /// Bounded length-prefixed framing codec.
    pub codec: QuicLengthPrefixedCodecV1,
}

impl Default for IrohCarrierAdapterV1 {
    fn default() -> Self {
        let policy = QuicStreamHandshakePolicyV1::default();
        Self {
            codec: QuicLengthPrefixedCodecV1 {
                max_frame_bytes: policy.max_frame_bytes,
            },
            policy,
        }
    }
}

impl IrohCarrierAdapterV1 {
    /// Canonical ALPN bytes expected on Iroh connections (`b"swarmtorch/delivery/v1"`).
    pub const fn expected_alpn() -> &'static [u8] {
        ALPN_SWARMTORCH_DELIVERY_V1
    }

    /// Construct an adapter with explicit link binding policy and max frame size.
    pub fn new(binding_policy: LinkBindingPolicyV1, max_frame_bytes: usize) -> Self {
        let policy = QuicStreamHandshakePolicyV1 {
            expected_alpn: ALPN_SWARMTORCH_DELIVERY_V1,
            binding_policy,
            max_frame_bytes,
        };
        Self {
            policy,
            codec: QuicLengthPrefixedCodecV1 { max_frame_bytes },
        }
    }

    /// Validate connection handshake (`ALPN_SWARMTORCH_DELIVERY_V1` + `OneRttComplete` only).
    pub fn accept_connection(
        &self,
        negotiated_alpn: &[u8],
        handshake_stage: QuicHandshakeStageV1,
        endpoint_public_key: &[u8; 32],
    ) -> Result<VerifiedLinkPrincipalV1, IrohCarrierError> {
        self.policy
            .validate_connection(negotiated_alpn, handshake_stage, endpoint_public_key)
            .map_err(IrohCarrierError::Carrier)
    }

    /// Frame an outbound serialized message (`MessageEnvelope` or `BundledGossipDeliveryV1`).
    pub fn encode_wire_frame(&self, payload: &[u8]) -> Result<Vec<u8>, IrohCarrierError> {
        self.codec
            .encode_frame(payload)
            .map_err(IrohCarrierError::Carrier)
    }

    /// Accept one length-prefixed direct [`MessageEnvelope`] frame from `stream_bytes`,
    /// verify signature & replay admission, bind originator policy, and return `(accepted, remaining_bytes)`.
    pub fn accept_envelope_frame<'a>(
        &self,
        link_principal: VerifiedLinkPrincipalV1,
        stream_bytes: &'a [u8],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<(AcceptedIrohEnvelopeV1, &'a [u8]), IrohCarrierError> {
        if let Some(tracker) = quota {
            tracker.record_frame(&link_principal)?;
        }
        let (frame, rest) = self.codec.decode_next_frame(stream_bytes)?;
        let envelope = MessageEnvelope::deserialize_bounded(frame, self.policy.max_frame_bytes)
            .map_err(IrohCarrierError::Envelope)?;
        envelope
            .verify_authenticated(replay_guard, current_time_secs)
            .map_err(IrohCarrierError::Envelope)?;
        let originator = envelope
            .sender_peer_id()
            .map_err(|_| IrohCarrierError::Envelope(VerifyError::InvalidSenderKey))?;
        let binding = self.policy.bind_originator(link_principal, originator)?;
        Ok((AcceptedIrohEnvelopeV1 { binding, envelope }, rest))
    }

    /// Accept one length-prefixed bundled gossip delivery frame (`BundledGossipDeliveryV1`),
    /// authenticating recipient wrapper + originator payload BEFORE replay commit.
    pub fn accept_gossip_frame<'a>(
        &self,
        link_principal: VerifiedLinkPrincipalV1,
        recipient_peer: PeerId,
        stream_bytes: &'a [u8],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<(AcceptedIrohGossipV1, &'a [u8]), IrohCarrierError> {
        if let Some(tracker) = quota {
            tracker.record_frame(&link_principal)?;
        }
        let (frame, rest) = self.codec.decode_next_frame(stream_bytes)?;
        let max_gossip = self
            .policy
            .max_frame_bytes
            .min(DEFAULT_MAX_GOSSIP_DELIVERY_BYTES);
        let delivery = BundledGossipDeliveryV1::from_bytes_bounded(frame, max_gossip)
            .map_err(IrohCarrierError::Gossip)?;
        let originator = delivery
            .verify_for_recipient(recipient_peer, replay_guard, current_time_secs)
            .map_err(IrohCarrierError::Gossip)?;
        let binding = self.policy.bind_originator(link_principal, originator)?;
        Ok((AcceptedIrohGossipV1 { binding, delivery }, rest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_torch_core::crypto::{KeyPair, MessageAuth};
    use swarm_torch_core::replay::CohortAllowlist;
    use swarm_torch_net::gossip::{InnerUpdateV1, PayloadKindV1};
    use swarm_torch_net::protocol::MessageType;

    #[test]
    fn iroh_carrier_rejects_early_data_enforces_binding_modes_and_authenticates_relay_gossip() {
        let relay_kp = KeyPair::from_seed([0x11; 32]).unwrap();
        let origin_kp = KeyPair::from_seed([0x22; 32]).unwrap();
        let coord_kp = KeyPair::from_seed([0x33; 32]).unwrap();

        let relay_adapter =
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::DecoupledRelayAllowed, 65_536);
        let strict_adapter = IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 65_536);

        // 0-RTT & wrong ALPN rejected pre-admission
        assert!(matches!(
            relay_adapter.accept_connection(
                ALPN_SWARMTORCH_DELIVERY_V1,
                QuicHandshakeStageV1::ZeroRtt,
                relay_kp.public_key()
            ),
            Err(IrohCarrierError::Carrier(CarrierError::EarlyDataRejected))
        ));
        assert!(matches!(
            relay_adapter.accept_connection(
                b"other/proto",
                QuicHandshakeStageV1::OneRttComplete,
                relay_kp.public_key()
            ),
            Err(IrohCarrierError::Carrier(CarrierError::AlpnMismatch))
        ));

        let link = relay_adapter
            .accept_connection(
                ALPN_SWARMTORCH_DELIVERY_V1,
                QuicHandshakeStageV1::OneRttComplete,
                relay_kp.public_key(),
            )
            .unwrap();

        // Envelope signed by originator arrives over relay link
        let origin_auth = MessageAuth::new(origin_kp.clone());
        let env = MessageEnvelope::new_v1(
            *origin_kp.public_key(),
            [0xAB; 16],
            1,
            5,
            MessageType::GradientUpdate,
            vec![10, 20, 30],
        )
        .sign_with(&origin_auth);
        let wire_frame = relay_adapter
            .encode_wire_frame(&env.serialize().unwrap())
            .unwrap();

        let allowlist = CohortAllowlist::from_peers([origin_kp.peer_id()]);
        let mut strict_replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(allowlist.clone());
        let mut relay_replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(allowlist.clone());

        // StrictOneToOne rejects relay link != originator
        assert!(matches!(
            strict_adapter.accept_envelope_frame(
                link,
                &wire_frame,
                &mut strict_replay,
                env.timestamp,
                None
            ),
            Err(IrohCarrierError::Carrier(
                CarrierError::StrictBindingMismatch { .. }
            ))
        ));

        // DecoupledRelayAllowed admits and flags relayed_via_intermediary = true
        let (accepted_env, rest) = relay_adapter
            .accept_envelope_frame(link, &wire_frame, &mut relay_replay, env.timestamp, None)
            .unwrap();
        assert!(rest.is_empty());
        assert!(accepted_env.binding.relayed_via_intermediary);
        assert_eq!(accepted_env.binding.originator_peer_id, origin_kp.peer_id());

        // Relayed BundledGossipDeliveryV1 bound to coordinator
        let inner = InnerUpdateV1 {
            originator_public_key: *origin_kp.public_key(),
            swarm_instance_id: [0xAB; 16],
            round_id: 5,
            contribution_id: 2,
            base_checkpoint_digest: [0xCD; 32],
            payload_kind: PayloadKindV1::ModelDeltaV1,
            compression_schema_version: 0,
            num_elements: 4,
            reported_examples: 16,
            encoded_payload: vec![1, 2, 3, 4],
            originator_signature: Vec::new(),
        };
        let gossip = BundledGossipDeliveryV1::new_signed(
            &origin_auth,
            inner,
            coord_kp.peer_id(),
            2,
            1,
            1000,
        );
        let gossip_frame = relay_adapter
            .encode_wire_frame(&gossip.to_bytes().unwrap())
            .unwrap();
        let mut coord_replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(allowlist);
        let (accepted_gossip, tail) = relay_adapter
            .accept_gossip_frame(
                link,
                coord_kp.peer_id(),
                &gossip_frame,
                &mut coord_replay,
                1000,
                None,
            )
            .unwrap();
        assert!(tail.is_empty());
        assert!(accepted_gossip.binding.relayed_via_intermediary);
        assert_eq!(
            accepted_gossip.binding.originator_peer_id,
            origin_kp.peer_id()
        );
    }
}
