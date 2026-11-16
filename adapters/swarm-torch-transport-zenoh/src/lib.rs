#![forbid(unsafe_code)]
//! Isolated `Zenoh` topic contract carrier adapter (`ADR-0022` §4, `W12-02`, `ST-08`).
//!
//! **Maturity Tier:** `Simulation verified` — strict topic key-expression parsing and pre-replay
//! `(scope, round_id, originator)` enforcement are tested against serialized wire frames. Live `zenoh`
//! SDK session / pub-sub integration is deferred to Phase `13D-3`.

use core::fmt;

use swarm_torch_core::replay::ReplayProtection;
use swarm_torch_core::traits::PeerId;
use swarm_torch_net::carrier::{CarrierError, ZenohTopicContractV1, ZENOH_TOPIC_PREFIX_V1};
use swarm_torch_net::gossip::{
    BundledGossipDeliveryV1, BundledGossipDeliveryV2, GossipVerifyError,
    DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
};
use swarm_torch_net::protocol::{MessageEnvelope, VerifyError, DEFAULT_MAX_ENVELOPE_BYTES};
use zenoh::handlers::FifoChannelHandler;
use zenoh::pubsub::Subscriber;
use zenoh::sample::Sample;

/// Errors produced while validating a Zenoh publication against SwarmTorch topic and crypto contracts.
#[derive(Debug)]
pub enum ZenohCarrierError {
    /// Topic grammar, wildcard/ROS 2 guard, or topic-vs-envelope identity check failed.
    Carrier(CarrierError),
    /// Envelope deserialization or authentication/replay verification failed.
    Envelope(VerifyError),
    /// Bundled gossip deserialization or verification failed.
    Gossip(GossipVerifyError),
    /// Live Zenoh session configuration, publish, or subscribe transport failure.
    Transport(String),
}

impl fmt::Display for ZenohCarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Carrier(e) => write!(f, "zenoh carrier error: {e}"),
            Self::Envelope(e) => write!(f, "envelope verification failed: {e}"),
            Self::Gossip(e) => write!(f, "gossip verification failed: {e}"),
            Self::Transport(msg) => write!(f, "zenoh transport error: {msg}"),
        }
    }
}

impl std::error::Error for ZenohCarrierError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Carrier(e) => Some(e),
            Self::Envelope(e) => Some(e),
            Self::Gossip(e) => Some(e),
            Self::Transport(_) => None,
        }
    }
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

    /// Format a validated topic string matching an outbound v2 gossip [`BundledGossipDeliveryV2`].
    pub fn topic_for_gossip_v2(
        delivery: &BundledGossipDeliveryV2,
    ) -> Result<String, ZenohCarrierError> {
        if delivery.wrapper.swarm_instance_id != delivery.inner.swarm_instance_id
            || delivery.wrapper.round_id != delivery.inner.round_id
        {
            return Err(ZenohCarrierError::Carrier(
                CarrierError::TopicIdentityMismatch,
            ));
        }
        let sender = delivery
            .inner
            .originator_peer_id()
            .map_err(|_| ZenohCarrierError::Carrier(CarrierError::TopicIdentityMismatch))?;
        Ok(ZenohTopicContractV1::new(
            delivery.inner.swarm_instance_id,
            delivery.inner.round_id,
            sender,
        )
        .to_key_expr())
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

/// Live `zenoh 1.10` pub/sub session bridge enforcing canonical `swarmtorch/v1/...` topics,
/// bounded payload sizes, and cryptographic envelope/gossip verification before replay mutation (`ST-08`).
pub struct ZenohSessionBridgeV1 {
    session: zenoh::Session,
    adapter: ZenohCarrierAdapterV1,
}

/// Active Zenoh subscriber bound to a [`ZenohCarrierAdapterV1`] validator.
pub struct ZenohValidatedSubscriberV1 {
    adapter: ZenohCarrierAdapterV1,
    subscriber: Subscriber<FifoChannelHandler<Sample>>,
}

impl ZenohSessionBridgeV1 {
    /// Open a live Zenoh session from a caller-provided [`zenoh::Config`].
    pub async fn open(
        config: zenoh::Config,
        adapter: ZenohCarrierAdapterV1,
    ) -> Result<Self, ZenohCarrierError> {
        let session = zenoh::open(config)
            .await
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        Ok(Self { session, adapter })
    }

    /// Open a deterministic peer-mode Zenoh session over explicit TCP endpoints with multicast
    /// scouting and gossip disabled (recommended for isolated/reproducible deployments and CI tests).
    pub async fn open_tcp_peer(
        listen_endpoints: &[String],
        connect_endpoints: &[String],
        adapter: ZenohCarrierAdapterV1,
    ) -> Result<Self, ZenohCarrierError> {
        let mut cfg = zenoh::Config::default();
        cfg.insert_json5("mode", "\"peer\"")
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        cfg.insert_json5("scouting/multicast/enabled", "false")
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        cfg.insert_json5("scouting/gossip/enabled", "false")
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        if !listen_endpoints.is_empty() {
            cfg.insert_json5(
                "listen/endpoints",
                &encode_json5_string_array(listen_endpoints)?,
            )
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        }
        if !connect_endpoints.is_empty() {
            cfg.insert_json5(
                "connect/endpoints",
                &encode_json5_string_array(connect_endpoints)?,
            )
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        }
        Self::open(cfg, adapter).await
    }

    /// Access the underlying `zenoh::Session`.
    pub fn session(&self) -> &zenoh::Session {
        &self.session
    }

    /// Access the configured [`ZenohCarrierAdapterV1`].
    pub fn adapter(&self) -> &ZenohCarrierAdapterV1 {
        &self.adapter
    }

    /// Close the underlying `zenoh::Session`.
    pub async fn close(self) -> Result<(), ZenohCarrierError> {
        self.session
            .close()
            .await
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))
    }

    /// Validate topic parity + payload cap and publish a serialized [`MessageEnvelope`] over Zenoh.
    pub async fn publish_envelope(
        &self,
        envelope: &MessageEnvelope,
    ) -> Result<String, ZenohCarrierError> {
        let topic = ZenohCarrierAdapterV1::topic_for_envelope(envelope)?;
        let _ = ZenohTopicContractV1::parse_key_expr(&topic)?;
        let payload = envelope
            .serialize()
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        if payload.len() > self.adapter.max_payload_bytes {
            return Err(ZenohCarrierError::Carrier(CarrierError::FrameTooLarge {
                len: payload.len(),
                max: self.adapter.max_payload_bytes,
            }));
        }
        self.session
            .put(&topic, payload)
            .await
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        Ok(topic)
    }

    /// Validate topic parity + payload cap and publish a serialized [`BundledGossipDeliveryV2`].
    pub async fn publish_gossip_v2(
        &self,
        delivery: &BundledGossipDeliveryV2,
    ) -> Result<String, ZenohCarrierError> {
        let topic = ZenohCarrierAdapterV1::topic_for_gossip_v2(delivery)?;
        let _ = ZenohTopicContractV1::parse_key_expr(&topic)?;
        let cap = self
            .adapter
            .max_payload_bytes
            .min(DEFAULT_MAX_GOSSIP_DELIVERY_BYTES);
        let payload = delivery
            .to_bytes()
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        if payload.len() > cap {
            return Err(ZenohCarrierError::Carrier(CarrierError::FrameTooLarge {
                len: payload.len(),
                max: cap,
            }));
        }
        self.session
            .put(&topic, payload)
            .await
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        Ok(topic)
    }

    /// Declare a scoped subscriber rooted strictly under `swarmtorch/v1/<swarm_instance_id>/...`.
    pub async fn subscribe_swarm(
        &self,
        swarm_instance_id: [u8; 16],
        round_id: Option<u64>,
    ) -> Result<ZenohValidatedSubscriberV1, ZenohCarrierError> {
        let scope_hex: String = swarm_instance_id
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let pattern = match round_id {
            Some(round) => format!("{ZENOH_TOPIC_PREFIX_V1}/{scope_hex}/round/{round}/**"),
            None => format!("{ZENOH_TOPIC_PREFIX_V1}/{scope_hex}/**"),
        };
        let subscriber = self
            .session
            .declare_subscriber(&pattern)
            .await
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        Ok(ZenohValidatedSubscriberV1 {
            adapter: self.adapter,
            subscriber,
        })
    }
}

impl ZenohValidatedSubscriberV1 {
    /// Wait for the next published sample and validate topic + signature + replay contract
    /// before mutating `replay_guard`.
    pub async fn recv_envelope(
        &self,
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
    ) -> Result<(ZenohTopicContractV1, MessageEnvelope), ZenohCarrierError> {
        let sample = self
            .subscriber
            .recv_async()
            .await
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        let payload_len = sample.payload().len();
        if payload_len > self.adapter.max_payload_bytes {
            return Err(ZenohCarrierError::Carrier(CarrierError::FrameTooLarge {
                len: payload_len,
                max: self.adapter.max_payload_bytes,
            }));
        }
        let payload = sample.payload().to_bytes();
        self.adapter.receive_envelope(
            sample.key_expr().as_str(),
            payload.as_ref(),
            replay_guard,
            current_time_secs,
        )
    }

    /// Wait for the next published sample and validate topic + v2 gossip signature + receiver-bound
    /// `session_id` + replay contract before mutating `replay_guard`.
    pub async fn recv_gossip_v2(
        &self,
        recipient_peer: PeerId,
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
    ) -> Result<(ZenohTopicContractV1, BundledGossipDeliveryV2), ZenohCarrierError> {
        let sample = self
            .subscriber
            .recv_async()
            .await
            .map_err(|e| ZenohCarrierError::Transport(e.to_string()))?;
        let cap = self
            .adapter
            .max_payload_bytes
            .min(DEFAULT_MAX_GOSSIP_DELIVERY_BYTES);
        let payload_len = sample.payload().len();
        if payload_len > cap {
            return Err(ZenohCarrierError::Carrier(CarrierError::FrameTooLarge {
                len: payload_len,
                max: cap,
            }));
        }
        let payload = sample.payload().to_bytes();
        self.adapter.receive_gossip_v2_delivery(
            sample.key_expr().as_str(),
            recipient_peer,
            payload.as_ref(),
            replay_guard,
            current_time_secs,
        )
    }
}

fn encode_json5_string_array(items: &[String]) -> Result<String, ZenohCarrierError> {
    let mut out = String::from("[");
    for (idx, item) in items.iter().enumerate() {
        if item
            .bytes()
            .any(|b| b.is_ascii_control() || b == b'"' || b == b'\\')
        {
            return Err(ZenohCarrierError::Transport(format!(
                "invalid endpoint configuration string: {item}"
            )));
        }
        if idx > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(item);
        out.push('"');
    }
    out.push(']');
    Ok(out)
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn live_zenoh_tcp_peer_pub_sub_rejects_topic_spoof_and_admits_v2_gossip() {
        // Reserve an ephemeral localhost port for deterministic TCP peer-to-peer Zenoh test.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let endpoint = format!("tcp/127.0.0.1:{port}");

        let bridge_sub = ZenohSessionBridgeV1::open_tcp_peer(
            std::slice::from_ref(&endpoint),
            &[],
            ZenohCarrierAdapterV1::default(),
        )
        .await
        .unwrap();
        let bridge_pub =
            ZenohSessionBridgeV1::open_tcp_peer(&[], &[endpoint], ZenohCarrierAdapterV1::default())
                .await
                .unwrap();

        let orig_kp = KeyPair::from_seed([0x21; 32]).unwrap();
        let orig_auth = MessageAuth::new(orig_kp.clone());
        let coord_kp = KeyPair::from_seed([0x31; 32]).unwrap();

        let scope = [0x99; 16];
        let run_id = [0x88; 16];
        let policy_digest = [0x77; 32];
        let challenge = SessionHandshakeChallengeV2 {
            swarm_instance_id: scope,
            run_id,
            policy_digest,
            receiver_peer: coord_kp.peer_id(),
            sender_peer: orig_kp.peer_id(),
            requested_epoch: 1,
            initial_sequence: 0,
            receiver_nonce: [0x66; 32],
            issued_at_secs: 900,
        };
        let session_id = challenge.derive_session_id().unwrap();
        let hs_sig = orig_auth.sign_session_handshake_v2(&challenge).unwrap();

        let mut replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(CohortAllowlist::from_peers([orig_kp.peer_id()]));
        replay.register_recovery_challenge_v2(challenge).unwrap();
        replay
            .complete_recovery_handshake_v2(scope, run_id, orig_kp.public_key(), &hs_sig, 900)
            .unwrap();

        let sub = bridge_sub.subscribe_swarm(scope, None).await.unwrap();

        let mut env_v2 = MessageEnvelope::new_v2(
            *orig_kp.public_key(),
            scope,
            run_id,
            1,
            session_id,
            5,
            MessageType::GradientUpdate,
            vec![1, 2, 3, 4],
        );
        env_v2.sequence = 1;
        env_v2.timestamp = 1000;
        let env_v2 = env_v2.sign_with(&orig_auth);
        let env_topic = ZenohCarrierAdapterV1::topic_for_envelope(&env_v2).unwrap();

        // Wait deterministically for Zenoh subscription propagation before publishing.
        let readiness_pub = bridge_pub
            .session()
            .declare_publisher(&env_topic)
            .await
            .unwrap();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        while !readiness_pub.matching_status().await.unwrap().matching() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "timed out waiting for Zenoh peer subscriber matching"
            );
            tokio::time::sleep(std::time::Duration::from_millis(15)).await;
        }
        drop(readiness_pub);

        // 1. Live spoofed-round topic MUST fail BEFORE mutating replay seq=1!
        let spoofed_topic = ZenohTopicContractV1::new(scope, 999, orig_kp.peer_id()).to_key_expr();
        bridge_pub
            .session()
            .put(&spoofed_topic, env_v2.serialize().unwrap())
            .await
            .unwrap();
        let spoof_res = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            sub.recv_envelope(&mut replay, 1000),
        )
        .await
        .unwrap();
        assert!(matches!(
            spoof_res,
            Err(ZenohCarrierError::Carrier(
                CarrierError::TopicIdentityMismatch
            ))
        ));

        // 2. Canonical topic publication over live Zenoh TCP peer session MUST admit seq=1!
        let published_topic = bridge_pub.publish_envelope(&env_v2).await.unwrap();
        assert_eq!(published_topic, env_topic);
        let (parsed_topic, accepted_env) = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            sub.recv_envelope(&mut replay, 1000),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(parsed_topic.round_id, 5);
        assert_eq!(accepted_env.sequence, 1);

        // 3. Valid BundledGossipDeliveryV2 over live Zenoh pub/sub succeeds cleanly!
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
        let gossip_topic = ZenohCarrierAdapterV1::topic_for_gossip_v2(&gossip_v2).unwrap();
        let readiness_g = bridge_pub
            .session()
            .declare_publisher(&gossip_topic)
            .await
            .unwrap();
        while !readiness_g.matching_status().await.unwrap().matching() {
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(std::time::Duration::from_millis(15)).await;
        }
        drop(readiness_g);

        bridge_pub.publish_gossip_v2(&gossip_v2).await.unwrap();
        let (g_topic, accepted_gossip) = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            sub.recv_gossip_v2(coord_kp.peer_id(), &mut replay, 1000),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(g_topic.round_id, 9);
        assert_eq!(accepted_gossip.wrapper.delivery_seq, 2);

        drop(sub);
        bridge_pub.close().await.unwrap();
        bridge_sub.close().await.unwrap();
    }
}
