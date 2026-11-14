#![forbid(unsafe_code)]
//! Isolated `Zenoh` topic contract carrier adapter (`ADR-0022` §4, `W12-02`, `ST-08`).
//!
//! **Maturity Tier:** `Simulation verified` — strict topic key-expression parsing and pre-replay
//! `(scope, round_id, originator)` enforcement are tested against serialized wire frames. Live `zenoh`
//! SDK session / pub-sub integration is deferred to Phase `13D-3`.

use swarm_torch_core::replay::ReplayProtection;
use swarm_torch_core::traits::PeerId;
use swarm_torch_net::carrier::{CarrierError, ZenohTopicContractV1, ZENOH_TOPIC_PREFIX_V1};
use swarm_torch_net::gossip::{
    BundledGossipDeliveryV1, GossipVerifyError, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
};
use swarm_torch_net::protocol::{MessageEnvelope, VerifyError, DEFAULT_MAX_ENVELOPE_BYTES};

/// Errors produced while validating a Zenoh publication against SwarmTorch topic and crypto contracts.
#[derive(Debug)]
pub enum ZenohCarrierError {
    /// Topic grammar, wildcard/ROS 2 guard, or topic-vs-envelope identity check failed.
    Carrier(CarrierError),
    /// Envelope deserialization or authentication/replay verification failed.
    Envelope(VerifyError),
    /// Bundled gossip deserialization or verification failed.
    Gossip(GossipVerifyError),
}

impl From<CarrierError> for ZenohCarrierError {
    fn from(value: CarrierError) -> Self {
        Self::Carrier(value)
    }
}

/// Canonical Zenoh carrier adapter enforcing `swarmtorch/v1/...` topic contracts prior to replay admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZenohCarrierAdapterV1 {
    /// Maximum bounded payload size accepted from Zenoh publications.
    pub max_payload_bytes: usize,
}

impl Default for ZenohCarrierAdapterV1 {
    fn default() -> Self {
        Self {
            max_payload_bytes: DEFAULT_MAX_ENVELOPE_BYTES,
        }
    }
}

impl ZenohCarrierAdapterV1 {
    /// Canonical Zenoh key-expression prefix (`"swarmtorch/v1"`).
    pub const fn topic_prefix() -> &'static str {
        ZENOH_TOPIC_PREFIX_V1
    }

    /// Format a validated topic string matching an outbound v1 [`MessageEnvelope`].
    pub fn topic_for_envelope(envelope: &MessageEnvelope) -> Result<String, ZenohCarrierError> {
        if envelope.version != MessageEnvelope::VERSION_V1 {
            return Err(ZenohCarrierError::Carrier(
                CarrierError::TopicIdentityMismatch,
            ));
        }
        let sender = envelope
            .sender_peer_id()
            .map_err(|_| ZenohCarrierError::Carrier(CarrierError::TopicIdentityMismatch))?;
        Ok(
            ZenohTopicContractV1::new(envelope.swarm_instance_id, envelope.round_id, sender)
                .to_key_expr(),
        )
    }

    /// Accept and authenticate a Zenoh message publication `(key_expr, payload_bytes)`:
    /// 1. Strictly parses `key_expr` before inspecting `payload_bytes`,
    /// 2. Boundedly deserializes [`MessageEnvelope`],
    /// 3. Enforces topic `(scope, round_id, originator)` agreement BEFORE touching `replay_guard`,
    /// 4. Runs full 6-stage [`MessageEnvelope::verify_authenticated`].
    pub fn receive_envelope(
        &self,
        key_expr: &str,
        payload_bytes: &[u8],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
    ) -> Result<(ZenohTopicContractV1, MessageEnvelope), ZenohCarrierError> {
        let topic = ZenohTopicContractV1::parse_key_expr(key_expr)?;
        let env = MessageEnvelope::deserialize_bounded(payload_bytes, self.max_payload_bytes)
            .map_err(ZenohCarrierError::Envelope)?;
        topic.verify_matches_envelope(&env)?;
        env.verify_authenticated(replay_guard, current_time_secs)
            .map_err(ZenohCarrierError::Envelope)?;
        Ok((topic, env))
    }

    /// Accept and authenticate a Zenoh bundled gossip publication `(key_expr, payload_bytes)`.
    pub fn receive_gossip_delivery(
        &self,
        key_expr: &str,
        recipient_peer: PeerId,
        payload_bytes: &[u8],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
    ) -> Result<(ZenohTopicContractV1, BundledGossipDeliveryV1), ZenohCarrierError> {
        let topic = ZenohTopicContractV1::parse_key_expr(key_expr)?;
        let cap = self
            .max_payload_bytes
            .min(DEFAULT_MAX_GOSSIP_DELIVERY_BYTES);
        let delivery = BundledGossipDeliveryV1::from_bytes_bounded(payload_bytes, cap)
            .map_err(ZenohCarrierError::Gossip)?;
        topic.verify_matches_gossip_delivery(&delivery)?;
        let _ = delivery
            .verify_for_recipient(recipient_peer, replay_guard, current_time_secs)
            .map_err(ZenohCarrierError::Gossip)?;
        Ok((topic, delivery))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_torch_core::crypto::{KeyPair, MessageAuth};
    use swarm_torch_core::replay::CohortAllowlist;
    use swarm_torch_net::protocol::MessageType;

    #[test]
    fn zenoh_carrier_rejects_topic_spoofing_before_replay_mutation() {
        let orig_kp = KeyPair::from_seed([0x44; 32]).unwrap();
        let spoof_kp = KeyPair::from_seed([0x55; 32]).unwrap();
        let scope = [0x22; 16];

        let adapter = ZenohCarrierAdapterV1::default();
        let env = MessageEnvelope::new_v1(
            *orig_kp.public_key(),
            scope,
            1,
            7,
            MessageType::GradientUpdate,
            vec![1, 2, 3, 4],
        )
        .sign_with(&MessageAuth::new(orig_kp.clone()));

        let canonical_topic = ZenohCarrierAdapterV1::topic_for_envelope(&env).unwrap();
        let payload = env.serialize().unwrap();

        let allowlist = CohortAllowlist::from_peers([orig_kp.peer_id(), spoof_kp.peer_id()]);
        let mut replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(allowlist);

        // Claiming another peer's topic fails TopicIdentityMismatch without exhausting sequence
        let spoof_topic = ZenohTopicContractV1::new(scope, 7, spoof_kp.peer_id()).to_key_expr();
        assert!(matches!(
            adapter.receive_envelope(&spoof_topic, &payload, &mut replay, env.timestamp),
            Err(ZenohCarrierError::Carrier(
                CarrierError::TopicIdentityMismatch
            ))
        ));

        // Wildcard topic fails before touching replay_guard
        assert!(matches!(
            adapter.receive_envelope(
                "swarmtorch/v1/*/round/7/peer/update",
                &payload,
                &mut replay,
                env.timestamp
            ),
            Err(ZenohCarrierError::Carrier(CarrierError::InvalidZenohTopic))
        ));

        // Authentic canonical topic succeeds cleanly
        let (parsed, verified_env) = adapter
            .receive_envelope(&canonical_topic, &payload, &mut replay, env.timestamp)
            .unwrap();
        assert_eq!(parsed.round_id, 7);
        assert_eq!(verified_env.sender_peer_id().unwrap(), orig_kp.peer_id());
    }
}
