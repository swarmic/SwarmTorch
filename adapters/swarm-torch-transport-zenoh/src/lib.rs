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
    BundledGossipDeliveryV1, BundledGossipDeliveryV2, GossipVerifyError,
    DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
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

    /// Format a validated topic string matching an outbound v1 or v2 [`MessageEnvelope`].
    pub fn topic_for_envelope(envelope: &MessageEnvelope) -> Result<String, ZenohCarrierError> {
        if !matches!(
            envelope.version,
            MessageEnvelope::VERSION_V1 | MessageEnvelope::VERSION_V2
        ) {
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

    /// Accept and authenticate a Zenoh v1 bundled gossip publication `(key_expr, payload_bytes)`.
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

    /// Accept and authenticate a Zenoh v2 bundled gossip publication `(key_expr, payload_bytes)`
    /// binding `run_id`, `session_id`, and topic `(scope, round_id, originator)` prior to replay
    /// admission (`ST-01`, `ST-03`, `ST-08`).
    pub fn receive_gossip_v2_delivery(
        &self,
        key_expr: &str,
        recipient_peer: PeerId,
        payload_bytes: &[u8],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
    ) -> Result<(ZenohTopicContractV1, BundledGossipDeliveryV2), ZenohCarrierError> {
        let topic = ZenohTopicContractV1::parse_key_expr(key_expr)?;
        let cap = self
            .max_payload_bytes
            .min(DEFAULT_MAX_GOSSIP_DELIVERY_BYTES);
        let delivery = BundledGossipDeliveryV2::from_bytes_bounded(payload_bytes, cap)
            .map_err(ZenohCarrierError::Gossip)?;
        topic.verify_matches_gossip_delivery_v2(&delivery)?;
        let _ = delivery
            .verify_for_recipient_v2(recipient_peer, replay_guard, current_time_secs)
            .map_err(ZenohCarrierError::Gossip)?;
        Ok((topic, delivery))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_torch_core::crypto::{KeyPair, MessageAuth, SessionHandshakeChallengeV2};
    use swarm_torch_core::replay::{CohortAllowlist, ReplayStartupPolicyV1};
    use swarm_torch_net::gossip::{InnerUpdateV2, PayloadKindV1};
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
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
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

    #[test]
    fn zenoh_carrier_v2_gossip_enforces_session_handshake_and_topic_binding() {
        let orig_kp = KeyPair::from_seed([0x61; 32]).unwrap();
        let coord_kp = KeyPair::from_seed([0x62; 32]).unwrap();
        let orig_auth = MessageAuth::new(orig_kp.clone());
        let scope = [0x31; 16];
        let run_id = [0x41; 16];
        let policy_digest = [0x51; 32];

        let adapter = ZenohCarrierAdapterV1::default();
        let allowlist = CohortAllowlist::from_peers([orig_kp.peer_id()]);
        let mut replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(allowlist);

        let challenge = SessionHandshakeChallengeV2 {
            swarm_instance_id: scope,
            run_id,
            policy_digest,
            receiver_peer: coord_kp.peer_id(),
            sender_peer: orig_kp.peer_id(),
            requested_epoch: 1,
            initial_sequence: 0,
            receiver_nonce: [0x88; 32],
            issued_at_secs: 1000,
        };
        let session_id = challenge.derive_session_id().unwrap();
        replay.register_recovery_challenge_v2(challenge).unwrap();
        let hs_sig = orig_auth.sign_session_handshake_v2(&challenge).unwrap();
        replay
            .complete_recovery_handshake_v2(scope, run_id, orig_kp.public_key(), &hs_sig, 1000)
            .unwrap();

        let env_v2 = MessageEnvelope::new_v2(
            *orig_kp.public_key(),
            scope,
            run_id,
            1,
            session_id,
            9,
            MessageType::GradientUpdate,
            vec![9, 8, 7],
        )
        .with_sequence(1)
        .with_timestamp(1000)
        .sign_with(&orig_auth);
        let topic_v2 = ZenohCarrierAdapterV1::topic_for_envelope(&env_v2).unwrap();
        let (_, accepted_env) = adapter
            .receive_envelope(&topic_v2, &env_v2.serialize().unwrap(), &mut replay, 1000)
            .unwrap();
        assert_eq!(accepted_env.version, MessageEnvelope::VERSION_V2);

        let inner_v2 = InnerUpdateV2 {
            originator_public_key: *orig_kp.public_key(),
            swarm_instance_id: scope,
            run_id,
            round_id: 9,
            contribution_id: 1,
            base_checkpoint_digest: [0x11; 32],
            context_policy_digest: policy_digest,
            payload_kind: PayloadKindV1::ModelDeltaV1,
            compression_schema_version: 0,
            num_elements: 2,
            reported_examples: 8,
            encoded_payload: vec![1, 2, 3, 4],
            originator_signature: Vec::new(),
        };
        let gossip_v2 = BundledGossipDeliveryV2::new_signed(
            &orig_auth,
            inner_v2,
            coord_kp.peer_id(),
            session_id,
            1,
            2,
            1000,
        );
        let gossip_bytes = gossip_v2.to_bytes().unwrap();

        // Wrong round topic must fail before mutating replay sequence
        let bad_round_topic = ZenohTopicContractV1::new(scope, 10, orig_kp.peer_id()).to_key_expr();
        assert!(matches!(
            adapter.receive_gossip_v2_delivery(
                &bad_round_topic,
                coord_kp.peer_id(),
                &gossip_bytes,
                &mut replay,
                1000,
            ),
            Err(ZenohCarrierError::Carrier(
                CarrierError::TopicIdentityMismatch
            ))
        ));

        // Canonical topic succeeds cleanly
        let (topic, accepted_gossip) = adapter
            .receive_gossip_v2_delivery(
                &topic_v2,
                coord_kp.peer_id(),
                &gossip_bytes,
                &mut replay,
                1000,
            )
            .unwrap();
        assert_eq!(topic.round_id, 9);
        assert_eq!(accepted_gossip.inner.run_id, run_id);
    }
}
