//! Integration tests for replay protection with message envelopes.

use swarm_torch_core::crypto::{CryptoError, KeyPair, MessageAuth, SessionHandshakeChallengeV2};
use swarm_torch_core::replay::{ReplayError, ReplayProtection, ReplayStartupPolicyV1};
use swarm_torch_core::traits::PeerId;
use swarm_torch_net::protocol::{
    AuthenticatedEnvelopeVerifier, MessageEnvelope, MessageType, VerifyError,
};

fn new_genesis_replay_guard() -> ReplayProtection {
    ReplayProtection::new().with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
}

fn signed_heartbeat(
    keypair: &KeyPair,
    auth: &MessageAuth,
    sequence: u64,
    timestamp: u32,
    payload: Vec<u8>,
) -> MessageEnvelope {
    let mut envelope = MessageEnvelope::new_with_public_key(
        *keypair.public_key(),
        MessageType::Heartbeat,
        payload,
    )
    .with_sequence(sequence)
    .with_timestamp(timestamp);

    let sig = auth.sign(
        envelope.version,
        envelope.message_type as u8,
        envelope.sequence,
        envelope.timestamp,
        &envelope.payload,
    );
    envelope = envelope.with_signature(sig.as_bytes().to_vec());
    envelope
}

#[test]
fn envelope_verify_authenticated_golden_path() {
    let keypair = KeyPair::from_seed([1u8; 32]).expect("non-zero seed");
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;

    let envelope = signed_heartbeat(&keypair, &auth, 1, now, b"test payload".to_vec());
    assert!(envelope
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());
}

#[test]
fn envelope_verify_authenticated_rejects_replay() {
    let keypair = KeyPair::from_seed([2u8; 32]).expect("non-zero seed");
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;

    let envelope = signed_heartbeat(&keypair, &auth, 1, now, b"test payload".to_vec());
    assert!(envelope
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());
    assert!(matches!(
        envelope.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::Replay(ReplayError::Replay { .. }))
    ));
}

#[test]
fn envelope_verify_authenticated_rejects_expired() {
    let keypair = KeyPair::from_seed([3u8; 32]).expect("non-zero seed");
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = new_genesis_replay_guard();
    let old_ts = 900;
    let now = 1000;

    let envelope = signed_heartbeat(&keypair, &auth, 1, old_ts, b"test payload".to_vec());
    assert!(matches!(
        envelope.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::Replay(ReplayError::Expired { .. }))
    ));
}

#[test]
fn envelope_verify_authenticated_signature_before_replay() {
    let keypair = KeyPair::from_seed([4u8; 32]).expect("non-zero seed");
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;

    let envelope = MessageEnvelope::new_with_public_key(
        *keypair.public_key(),
        MessageType::Heartbeat,
        b"test payload".to_vec(),
    )
    .with_sequence(1)
    .with_timestamp(now)
    .with_signature(vec![0xFF; 64]);

    assert!(matches!(
        envelope.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::Crypto(_))
    ));
    assert_eq!(replay_guard.cache_size(), 0);
}

#[test]
fn envelope_verify_authenticated_rejects_missing_signature() {
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;
    let keypair = KeyPair::from_seed([5u8; 32]).expect("non-zero seed");

    let envelope = MessageEnvelope::new_with_public_key(
        *keypair.public_key(),
        MessageType::Heartbeat,
        b"test".to_vec(),
    )
    .with_sequence(1)
    .with_timestamp(now);

    assert!(matches!(
        envelope.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::MissingSignature)
    ));
}

#[test]
fn envelope_verify_authenticated_rejects_wrong_signature_length() {
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;
    let keypair = KeyPair::from_seed([6u8; 32]).expect("non-zero seed");

    let envelope = MessageEnvelope::new_with_public_key(
        *keypair.public_key(),
        MessageType::Heartbeat,
        b"test".to_vec(),
    )
    .with_sequence(1)
    .with_timestamp(now)
    .with_signature(vec![0u8; 32]);

    assert!(matches!(
        envelope.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::InvalidSignatureLength {
            expected: 64,
            found: 32
        })
    ));
}

#[test]
fn envelope_verify_authenticated_rejects_tampered_payload() {
    let keypair = KeyPair::from_seed([7u8; 32]).expect("non-zero seed");
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;

    let mut envelope = signed_heartbeat(&keypair, &auth, 1, now, b"original payload".to_vec());
    envelope.payload = b"tampered payload".to_vec();

    assert!(matches!(
        envelope.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::Crypto(_))
    ));
}

#[test]
fn envelope_verify_authenticated_monotonic_sequences() {
    let keypair = KeyPair::from_seed([8u8; 32]).expect("non-zero seed");
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;

    for sequence in 1..=10 {
        let payload = format!("message {}", sequence).into_bytes();
        let envelope = signed_heartbeat(&keypair, &auth, sequence, now, payload);
        assert!(envelope
            .verify_authenticated(&mut replay_guard, now)
            .is_ok());
    }
}

#[test]
fn envelope_verify_authenticated_out_of_order_within_window() {
    let keypair = KeyPair::from_seed([9u8; 32]).expect("non-zero seed");
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;

    let make_envelope = |sequence: u64| {
        signed_heartbeat(
            &keypair,
            &auth,
            sequence,
            now,
            format!("message {}", sequence).into_bytes(),
        )
    };

    assert!(make_envelope(20)
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());
    assert!(make_envelope(15)
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());
    assert!(make_envelope(10)
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());
    assert!(make_envelope(5)
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());
    assert!(matches!(
        make_envelope(3).verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::Replay(ReplayError::TooOld { .. }))
    ));
}

#[test]
fn envelope_verify_authenticated_multi_peer_isolation() {
    let keypair_a = KeyPair::from_seed([10u8; 32]).expect("non-zero seed");
    let auth_a = MessageAuth::new(keypair_a.clone());

    let keypair_b = KeyPair::from_seed([11u8; 32]).expect("non-zero seed");
    let auth_b = MessageAuth::new(keypair_b.clone());

    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;

    let envelope_a = signed_heartbeat(&keypair_a, &auth_a, 5, now, b"peer a".to_vec());
    assert!(envelope_a
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());

    let envelope_b = signed_heartbeat(&keypair_b, &auth_b, 5, now, b"peer b".to_vec());
    assert!(envelope_b
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());

    assert!(matches!(
        envelope_a.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::Replay(ReplayError::Replay { .. }))
    ));
}

#[test]
fn new_with_public_key_uses_raw_bytes() {
    let keypair = KeyPair::from_seed([12u8; 32]).expect("non-zero seed");
    let envelope = MessageEnvelope::new_with_public_key(
        *keypair.public_key(),
        MessageType::Heartbeat,
        b"payload".to_vec(),
    );
    assert_eq!(envelope.sender, *keypair.public_key());
    assert_eq!(envelope.sender_public_key(), keypair.public_key());
}

#[test]
fn verify_authenticated_rejects_hashed_peer_id() {
    let keypair = KeyPair::from_seed([14u8; 32]).expect("non-zero seed");
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;

    let hashed_sender = PeerId::from_public_key(keypair.public_key());
    let mut envelope = MessageEnvelope::new_with_public_key(
        *hashed_sender.as_bytes(),
        MessageType::Heartbeat,
        b"payload".to_vec(),
    )
    .with_sequence(1)
    .with_timestamp(now);

    let sig = auth.sign(
        envelope.version,
        envelope.message_type as u8,
        envelope.sequence,
        envelope.timestamp,
        &envelope.payload,
    );
    envelope = envelope.with_signature(sig.as_bytes().to_vec());

    assert!(matches!(
        envelope.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::InvalidSenderKey) | Err(VerifyError::Crypto(_))
    ));
}

#[test]
fn current_unix_secs_returns_valid_timestamp() {
    let ts = MessageEnvelope::current_unix_secs().expect("current_unix_secs should succeed");
    assert!(ts >= 1_577_836_800);
}

#[test]
fn verify_authenticated_requires_unix_seconds_not_millis() {
    let keypair = KeyPair::from_seed([15u8; 32]).expect("non-zero seed");
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = new_genesis_replay_guard();
    let now_secs = 1_700_000_000;

    let envelope_secs = signed_heartbeat(&keypair, &auth, 1, now_secs, b"seconds".to_vec());
    assert!(envelope_secs
        .verify_authenticated(&mut replay_guard, now_secs)
        .is_ok());

    let envelope_ms = signed_heartbeat(
        &keypair,
        &auth,
        2,
        now_secs.saturating_add(1),
        b"milliseconds".to_vec(),
    );
    let now_millis = now_secs.saturating_mul(1000);
    assert!(matches!(
        envelope_ms.verify_authenticated(&mut replay_guard, now_millis),
        Err(VerifyError::Replay(ReplayError::Expired { .. }))
    ));
}

#[test]
fn verify_authenticated_rejects_unsupported_version() {
    let mut replay_guard = new_genesis_replay_guard();
    let mut envelope = MessageEnvelope::new_with_public_key(
        [16u8; 32],
        MessageType::Heartbeat,
        b"unsupported version".to_vec(),
    );
    envelope.version = (9, 9);

    let result = envelope.verify_authenticated(&mut replay_guard, 1000);
    assert!(matches!(
        result,
        Err(VerifyError::UnsupportedVersion { major: 9, minor: 9 })
    ));
    assert_eq!(replay_guard.cache_size(), 0);
}

#[test]
fn sender_peer_id_rejects_zero_sender() {
    let envelope =
        MessageEnvelope::new_with_public_key([0u8; 32], MessageType::Heartbeat, b"zero".to_vec());
    assert!(matches!(
        envelope.sender_peer_id(),
        Err(CryptoError::InvalidPublicKey)
    ));
}

#[test]
fn verify_authenticated_rejects_zero_sender_key() {
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;
    let envelope = MessageEnvelope::new_with_public_key(
        [0u8; 32],
        MessageType::Heartbeat,
        b"zero sender".to_vec(),
    )
    .with_sequence(1)
    .with_timestamp(now)
    .with_signature(vec![0u8; 64]);

    assert!(matches!(
        envelope.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::InvalidSenderKey)
    ));
}

#[test]
fn verify_error_source_chain_includes_crypto() {
    let keypair = KeyPair::from_seed([23u8; 32]).expect("non-zero seed");
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;

    let envelope = MessageEnvelope::new_with_public_key(
        *keypair.public_key(),
        MessageType::Heartbeat,
        b"invalid signature".to_vec(),
    )
    .with_sequence(1)
    .with_timestamp(now)
    .with_signature(vec![0xAA; 64]);

    let err = envelope
        .verify_authenticated(&mut replay_guard, now)
        .expect_err("invalid signature must fail");
    assert!(matches!(err, VerifyError::Crypto(_)));

    let source = std::error::Error::source(&err).expect("crypto source must be chained");
    assert!(
        source.to_string().contains("signature verification failed")
            || source.to_string().contains("invalid signature"),
        "unexpected source text: {}",
        source
    );
    // Keep `auth` referenced so this test mirrors signed-path setup and avoids future drift.
    let _ = auth;
}

#[test]
fn authenticated_verifier_verify_and_unwrap_with_time() {
    let keypair = KeyPair::from_seed([17u8; 32]).expect("non-zero seed");
    let auth = MessageAuth::new(keypair.clone());
    let now = 1000;
    let envelope = signed_heartbeat(&keypair, &auth, 1, now, b"wrapped".to_vec());

    let mut verifier = AuthenticatedEnvelopeVerifier::with_replay_guard(new_genesis_replay_guard());
    let result = verifier.verify_and_unwrap_with_time(envelope.clone(), now);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().sequence, envelope.sequence);
}

/// H-02 regression: `verify_authenticated()` must use the canonical
/// `PeerId::from_public_key()` derivation for the replay cache lookup,
/// not `PeerId::new(raw_sender_bytes)`.
///
/// Before fix, `PeerId::new()` (raw bytes) was used, creating a different
/// cache key than what `KeyPair::peer_id()` / `PeerId::from_public_key()`
/// produce. This meant the external PeerId and the internal replay PeerId
/// were different, breaking replay isolation accountability.
///
/// This test verifies that two messages from the same keypair are tracked
/// under the same replay cache entry (same PeerId) and that the replay
/// PeerId matches the canonical `sender_peer_id()` derivation.
#[test]
fn h02_verify_path_uses_canonical_peer_id_for_replay() {
    let keypair = KeyPair::from_seed([200u8; 32]).expect("non-zero seed");
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = new_genesis_replay_guard();
    let now = 1000;

    // First message accepted
    let envelope1 = signed_heartbeat(&keypair, &auth, 1, now, b"msg1".to_vec());
    assert!(envelope1
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());

    // The replay cache should have exactly 1 entry (the canonical PeerId)
    assert_eq!(replay_guard.cache_size(), 1);

    // Second message with same keypair, different seq: accepted under same peer entry
    let envelope2 = signed_heartbeat(&keypair, &auth, 2, now, b"msg2".to_vec());
    assert!(envelope2
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());

    // Still 1 entry — both messages are tracked under the same canonical PeerId
    assert_eq!(
        replay_guard.cache_size(),
        1,
        "both messages must use the same replay cache entry (canonical PeerId)"
    );

    // Replaying sequence 1 must be caught as Replay (proves same PeerId is used)
    let replay = signed_heartbeat(&keypair, &auth, 1, now, b"msg1".to_vec());
    assert!(
        matches!(
            replay.verify_authenticated(&mut replay_guard, now),
            Err(VerifyError::Replay(ReplayError::Replay { .. }))
        ),
        "replayed sequence must be detected — same PeerId in cache"
    );

    // Verify sender_peer_id() matches the canonical derivation
    assert_eq!(
        envelope1.sender_peer_id().expect("valid sender key"),
        keypair.peer_id(),
        "sender_peer_id() must match KeyPair::peer_id() (canonical hash)"
    );
}

#[test]
fn six_stage_pipeline_v1_envelope_allowlist_and_epoch_recovery_end_to_end() {
    use swarm_torch_core::crypto::SessionHandshakeChallenge;
    use swarm_torch_core::replay::CohortAllowlist;
    use swarm_torch_net::protocol::HeartbeatMessage;

    let swarm_id = [0x42u8; 16];
    let sender_kp = KeyPair::from_seed([81u8; 32]).unwrap();
    let unauthorized_kp = KeyPair::from_seed([82u8; 32]).unwrap();
    let receiver_kp = KeyPair::from_seed([83u8; 32]).unwrap();
    let sender_auth = MessageAuth::new(sender_kp.clone());
    let unauthorized_auth = MessageAuth::new(unauthorized_kp.clone());

    let allowlist = CohortAllowlist::from_peers([sender_kp.peer_id()]);
    let mut replay_guard = new_genesis_replay_guard().with_cohort_allowlist(allowlist);
    let now = 5000;

    // Stage 1: Oversized frame rejected before deserialization.
    assert_eq!(
        MessageEnvelope::deserialize_bounded(&[0u8; 128], 64).unwrap_err(),
        VerifyError::FrameTooLarge { len: 128, max: 64 }
    );

    // Stage 3: Unauthorized peer is rejected with UnauthorizedPeer before state mutation.
    let unauth_env = MessageEnvelope::new_v1(
        *unauthorized_kp.public_key(),
        swarm_id,
        1,
        10,
        MessageType::Heartbeat,
        b"unauth".to_vec(),
    )
    .with_sequence(1)
    .with_timestamp(now)
    .sign_with(&unauthorized_auth);

    assert_eq!(
        unauth_env.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::Replay(ReplayError::UnauthorizedPeer {
            peer: unauthorized_kp.peer_id()
        }))
    );
    assert_eq!(replay_guard.cache_size(), 0);

    // Stage 4-6: Authorized v1 envelope with serialized HeartbeatMessage verifies and decodes.
    let hb = HeartbeatMessage {
        current_round: 10,
        role: 1,
        known_peers: 7,
        load: 42,
    };
    let hb_bytes = postcard::to_allocvec(&hb).unwrap();
    let env_e1 = MessageEnvelope::new_v1(
        *sender_kp.public_key(),
        swarm_id,
        1,
        10,
        MessageType::Heartbeat,
        hb_bytes.clone(),
    )
    .with_sequence(5)
    .with_timestamp(now)
    .sign_with(&sender_auth);

    let wire = env_e1.serialize().unwrap();
    let decoded_env = MessageEnvelope::deserialize_bounded(&wire, 4096).unwrap();
    let decoded_hb: HeartbeatMessage = decoded_env
        .verify_and_decode_payload(&mut replay_guard, now)
        .unwrap();
    assert_eq!(decoded_hb, hb);

    // Cross-swarm replay of the signed v1 envelope fails signature verification.
    let mut cross_swarm = decoded_env.clone();
    cross_swarm.swarm_instance_id = [0x99u8; 16];
    assert_eq!(
        cross_swarm.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::Crypto(CryptoError::VerificationFailed))
    );

    // Unauthenticated epoch jump (epoch 2 without recovery handshake) fails at Stage 3 with RecoveryRequired.
    let env_e2_unrecovered = MessageEnvelope::new_v1(
        *sender_kp.public_key(),
        swarm_id,
        2,
        11,
        MessageType::Heartbeat,
        hb_bytes.clone(),
    )
    .with_sequence(1)
    .with_timestamp(now)
    .sign_with(&sender_auth);

    assert_eq!(
        env_e2_unrecovered.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::Replay(ReplayError::RecoveryRequired {
            peer: sender_kp.peer_id(),
            current_epoch: 1,
            requested_epoch: 2,
        }))
    );

    // Receiver issues recovery challenge; sender completes handshake; epoch 2 envelope now succeeds.
    let challenge = SessionHandshakeChallenge {
        swarm_instance_id: swarm_id,
        receiver_peer: receiver_kp.peer_id(),
        sender_peer: sender_kp.peer_id(),
        requested_epoch: 2,
        initial_sequence: 0,
        challenge_nonce: [0xAB; 32],
        issued_at_secs: now,
    };
    replay_guard.register_recovery_challenge(challenge).unwrap();
    let hs_sig = sender_auth.sign_session_handshake(&challenge);
    replay_guard
        .complete_recovery_handshake(swarm_id, sender_kp.public_key(), &hs_sig, now)
        .unwrap();

    assert!(env_e2_unrecovered
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());
}

#[test]
fn v2_envelope_session_id_and_restart_recovery_integration() {
    let swarm_id = [0x44u8; 16];
    let run_id = [0x55u8; 16];
    let policy_digest = [0x66u8; 32];
    let sender_kp = KeyPair::from_seed([0x71u8; 32]).unwrap();
    let sender_auth = MessageAuth::new(sender_kp.clone());
    let receiver_kp = KeyPair::from_seed([0x72u8; 32]).unwrap();
    let now = 1_700_005_000u32;

    let mut guard = ReplayProtection::try_with_config(64, 60)
        .unwrap()
        .with_cohort_allowlist(swarm_torch_core::replay::CohortAllowlist::from_peers([
            sender_kp.peer_id(),
        ]));

    // Unestablished peer fails closed on default RequireSessionHandshake policy.
    let fake_v2 = MessageEnvelope::new_v2(
        *sender_kp.public_key(),
        swarm_id,
        run_id,
        1,
        [0x88u8; 32],
        10,
        MessageType::Heartbeat,
        b"payload".to_vec(),
    )
    .with_sequence(1)
    .with_timestamp(now)
    .sign_with(&sender_auth);
    assert_eq!(
        fake_v2.verify_authenticated(&mut guard, now),
        Err(VerifyError::Replay(ReplayError::RecoveryRequired {
            peer: sender_kp.peer_id(),
            current_epoch: 0,
            requested_epoch: 1,
        }))
    );

    // Unsigned v2 fields smuggled inside a v1 envelope fail MalformedEnvelope.
    let mut smuggled_v1 = MessageEnvelope::new_v1(
        *sender_kp.public_key(),
        swarm_id,
        1,
        10,
        MessageType::Heartbeat,
        b"payload".to_vec(),
    )
    .with_sequence(1)
    .with_timestamp(now)
    .sign_with(&sender_auth);
    smuggled_v1.run_id = run_id;
    assert_eq!(
        smuggled_v1.verify_authenticated(&mut guard, now),
        Err(VerifyError::MalformedEnvelope)
    );

    // Complete v2 recovery handshake -> session_id_1 admitted.
    let ch1 = SessionHandshakeChallengeV2 {
        swarm_instance_id: swarm_id,
        run_id,
        policy_digest,
        receiver_peer: receiver_kp.peer_id(),
        sender_peer: sender_kp.peer_id(),
        requested_epoch: 1,
        initial_sequence: 0,
        receiver_nonce: [0xA1u8; 32],
        issued_at_secs: now,
    };
    guard.register_recovery_challenge_v2(ch1).unwrap();
    let hs_sig_1 = sender_auth.sign_session_handshake_v2(&ch1).unwrap();
    let active_1 = guard
        .complete_recovery_handshake_v2(swarm_id, run_id, sender_kp.public_key(), &hs_sig_1, now)
        .unwrap();

    let env1 = MessageEnvelope::new_v2(
        *sender_kp.public_key(),
        swarm_id,
        run_id,
        1,
        active_1.session_id,
        10,
        MessageType::Heartbeat,
        b"v2-ok".to_vec(),
    )
    .with_sequence(1)
    .with_timestamp(now)
    .sign_with(&sender_auth);
    let wire1 = env1.serialize().unwrap();
    let decoded1 = MessageEnvelope::deserialize_bounded(&wire1, 4096).unwrap();
    assert!(decoded1.verify_authenticated(&mut guard, now).is_ok());

    // Simulate process crash/restart in the SAME wall-clock second and establish a new nonce session.
    let mut restarted = ReplayProtection::try_with_config(64, 60)
        .unwrap()
        .with_timestamp_floor(now)
        .with_cohort_allowlist(swarm_torch_core::replay::CohortAllowlist::from_peers([
            sender_kp.peer_id(),
        ]));
    let ch2 = SessionHandshakeChallengeV2 {
        swarm_instance_id: swarm_id,
        run_id,
        policy_digest,
        receiver_peer: receiver_kp.peer_id(),
        sender_peer: sender_kp.peer_id(),
        requested_epoch: 1,
        initial_sequence: 0,
        receiver_nonce: [0xB2u8; 32],
        issued_at_secs: now,
    };
    restarted.register_recovery_challenge_v2(ch2).unwrap();
    let hs_sig_2 = sender_auth.sign_session_handshake_v2(&ch2).unwrap();
    let _active_2 = restarted
        .complete_recovery_handshake_v2(swarm_id, run_id, sender_kp.public_key(), &hs_sig_2, now)
        .unwrap();

    // Capturing & replaying pre-crash env1 in the same second MUST fail SessionIdMismatch!
    assert_eq!(
        decoded1.verify_authenticated(&mut restarted, now),
        Err(VerifyError::Replay(ReplayError::SessionIdMismatch {
            peer: sender_kp.peer_id(),
        }))
    );
}
