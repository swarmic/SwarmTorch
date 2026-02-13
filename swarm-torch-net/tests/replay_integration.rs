//! Integration tests for replay protection with message envelopes.

use swarm_torch_core::crypto::{KeyPair, MessageAuth};
use swarm_torch_core::replay::{ReplayError, ReplayProtection};
use swarm_torch_core::traits::PeerId;
use swarm_torch_net::protocol::{
    AuthenticatedEnvelopeVerifier, MessageEnvelope, MessageType, VerifyError,
};

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
    let keypair = KeyPair::from_seed([1u8; 32]);
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = ReplayProtection::new();
    let now = 1000;

    let envelope = signed_heartbeat(&keypair, &auth, 1, now, b"test payload".to_vec());
    assert!(envelope
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());
}

#[test]
fn envelope_verify_authenticated_rejects_replay() {
    let keypair = KeyPair::from_seed([2u8; 32]);
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = ReplayProtection::new();
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
    let keypair = KeyPair::from_seed([3u8; 32]);
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = ReplayProtection::new();
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
    let keypair = KeyPair::from_seed([4u8; 32]);
    let mut replay_guard = ReplayProtection::new();
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
    let mut replay_guard = ReplayProtection::new();
    let now = 1000;

    let envelope =
        MessageEnvelope::new_with_public_key([5u8; 32], MessageType::Heartbeat, b"test".to_vec())
            .with_sequence(1)
            .with_timestamp(now);

    assert!(matches!(
        envelope.verify_authenticated(&mut replay_guard, now),
        Err(VerifyError::MissingSignature)
    ));
}

#[test]
fn envelope_verify_authenticated_rejects_wrong_signature_length() {
    let mut replay_guard = ReplayProtection::new();
    let now = 1000;

    let envelope =
        MessageEnvelope::new_with_public_key([6u8; 32], MessageType::Heartbeat, b"test".to_vec())
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
    let keypair = KeyPair::from_seed([7u8; 32]);
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = ReplayProtection::new();
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
    let keypair = KeyPair::from_seed([8u8; 32]);
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = ReplayProtection::new();
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
    let keypair = KeyPair::from_seed([9u8; 32]);
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = ReplayProtection::new();
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
    let keypair_a = KeyPair::from_seed([10u8; 32]);
    let auth_a = MessageAuth::new(keypair_a.clone());

    let keypair_b = KeyPair::from_seed([11u8; 32]);
    let auth_b = MessageAuth::new(keypair_b.clone());

    let mut replay_guard = ReplayProtection::new();
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
    let keypair = KeyPair::from_seed([12u8; 32]);
    let envelope = MessageEnvelope::new_with_public_key(
        *keypair.public_key(),
        MessageType::Heartbeat,
        b"payload".to_vec(),
    );
    assert_eq!(envelope.sender, *keypair.public_key());
    assert_eq!(envelope.sender_public_key(), keypair.public_key());
}

#[test]
#[allow(deprecated)]
fn new_with_peer_id_deprecated_but_works() {
    let keypair = KeyPair::from_seed([13u8; 32]);
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = ReplayProtection::new();
    let now = 1000;

    let sender = PeerId::new(*keypair.public_key());
    let mut envelope = MessageEnvelope::new(sender, MessageType::Heartbeat, b"legacy".to_vec())
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

    assert!(envelope
        .verify_authenticated(&mut replay_guard, now)
        .is_ok());
}

#[test]
fn verify_authenticated_rejects_hashed_peer_id() {
    let keypair = KeyPair::from_seed([14u8; 32]);
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = ReplayProtection::new();
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
        Err(VerifyError::Crypto(_))
    ));
}

#[test]
fn current_unix_secs_returns_valid_timestamp() {
    let ts = MessageEnvelope::current_unix_secs().expect("current_unix_secs should succeed");
    assert!(ts >= 1_577_836_800);
}

#[test]
fn verify_authenticated_requires_unix_seconds_not_millis() {
    let keypair = KeyPair::from_seed([15u8; 32]);
    let auth = MessageAuth::new(keypair.clone());
    let mut replay_guard = ReplayProtection::new();
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
    let mut replay_guard = ReplayProtection::new();
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
fn authenticated_verifier_verify_and_unwrap_with_time() {
    let keypair = KeyPair::from_seed([17u8; 32]);
    let auth = MessageAuth::new(keypair.clone());
    let now = 1000;
    let envelope = signed_heartbeat(&keypair, &auth, 1, now, b"wrapped".to_vec());

    let mut verifier = AuthenticatedEnvelopeVerifier::new();
    let result = verifier.verify_and_unwrap_with_time(envelope.clone(), now);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().sequence, envelope.sequence);
}
