use super::*;

fn tamper(bytes: &mut [u8]) {
    bytes[0] ^= 0xFF;
}

#[test]
fn signature_verification_succeeds_for_valid_keypair() {
    let seed = [1u8; 32];
    let pair = KeyPair::from_seed(seed).expect("non-zero seed");
    let auth = MessageAuth::new(pair.clone());

    let version = (0, 1);
    let msg_type = 1;
    let seq = 100;
    let ts = 1234567890;
    let payload = b"test payload";

    let sig = auth.sign(version, msg_type, seq, ts, payload);

    assert!(
        MessageAuth::verify(pair.public_key(), version, msg_type, seq, ts, payload, &sig).is_ok()
    );
}

#[test]
fn signature_verification_fails_for_tampered_payload() {
    let seed = [2u8; 32];
    let pair = KeyPair::from_seed(seed).expect("non-zero seed");
    let auth = MessageAuth::new(pair.clone());

    let version = (0, 1);
    let msg_type = 1;
    let seq = 100;
    let ts = 1234567890;
    let payload = b"test payload";

    let sig = auth.sign(version, msg_type, seq, ts, payload);

    let mut bad_payload = *payload;
    tamper(&mut bad_payload);

    let result = MessageAuth::verify(
        pair.public_key(),
        version,
        msg_type,
        seq,
        ts,
        &bad_payload,
        &sig,
    );
    assert_eq!(result, Err(CryptoError::VerificationFailed));
}

#[test]
fn signature_verification_fails_for_wrong_key() {
    let seed1 = [3u8; 32];
    let pair1 = KeyPair::from_seed(seed1).expect("non-zero seed");
    let auth1 = MessageAuth::new(pair1);

    let seed2 = [4u8; 32];
    let pair2 = KeyPair::from_seed(seed2).expect("non-zero seed");

    let version = (0, 1);
    let msg_type = 1;
    let seq = 100;
    let ts = 1234567890;
    let payload = b"test payload";

    let sig = auth1.sign(version, msg_type, seq, ts, payload);

    let result = MessageAuth::verify(
        pair2.public_key(),
        version,
        msg_type,
        seq,
        ts,
        payload,
        &sig,
    );
    assert_eq!(result, Err(CryptoError::VerificationFailed));
}

#[test]
fn signature_is_deterministic_for_fixed_seed_and_message() {
    let seed = [5u8; 32];
    let pair1 = KeyPair::from_seed(seed).expect("non-zero seed");
    let auth1 = MessageAuth::new(pair1);

    let pair2 = KeyPair::from_seed(seed).expect("non-zero seed");
    let auth2 = MessageAuth::new(pair2);

    let version = (0, 1);
    let msg_type = 1;
    let seq = 100;
    let ts = 1234567890;
    let payload = b"variable payload";

    let sig1 = auth1.sign(version, msg_type, seq, ts, payload);
    let sig2 = auth2.sign(version, msg_type, seq, ts, payload);

    assert_eq!(sig1.0, sig2.0);
}

#[test]
fn verification_fails_on_modified_sequence() {
    let seed = [6u8; 32];
    let pair = KeyPair::from_seed(seed).expect("non-zero seed");
    let auth = MessageAuth::new(pair.clone());

    let version = (0, 1);
    let msg_type = 1;
    let seq = 100;
    let ts = 1234567890;
    let payload = b"test";

    let sig = auth.sign(version, msg_type, seq, ts, payload);

    let result = MessageAuth::verify(
        pair.public_key(),
        version,
        msg_type,
        seq + 1,
        ts,
        payload,
        &sig,
    );
    assert_eq!(result, Err(CryptoError::VerificationFailed));
}

#[test]
fn verification_fails_on_modified_timestamp() {
    let seed = [7u8; 32];
    let pair = KeyPair::from_seed(seed).expect("non-zero seed");
    let auth = MessageAuth::new(pair.clone());

    let version = (0, 1);
    let msg_type = 1;
    let seq = 100;
    let ts = 1234567890;
    let payload = b"test";

    let sig = auth.sign(version, msg_type, seq, ts, payload);

    let result = MessageAuth::verify(
        pair.public_key(),
        version,
        msg_type,
        seq,
        ts + 1,
        payload,
        &sig,
    );
    assert_eq!(result, Err(CryptoError::VerificationFailed));
}

#[test]
fn verification_fails_on_modified_sender() {
    let seed = [8u8; 32];
    let pair = KeyPair::from_seed(seed).expect("non-zero seed");
    let auth = MessageAuth::new(pair.clone());

    let version = (0, 1);
    let msg_type = 1;
    let seq = 100;
    let ts = 1234567890;
    let payload = b"test";

    let sig = auth.sign(version, msg_type, seq, ts, payload);

    let mut wrong_public = *pair.public_key();
    tamper(&mut wrong_public);

    let result = MessageAuth::verify(&wrong_public, version, msg_type, seq, ts, payload, &sig);
    assert!(result.is_err());
}

#[test]
fn verification_fails_on_modified_message_type() {
    let seed = [9u8; 32];
    let pair = KeyPair::from_seed(seed).expect("non-zero seed");
    let auth = MessageAuth::new(pair.clone());

    let version = (0, 1);
    let msg_type = 1;
    let seq = 100;
    let ts = 1234567890;
    let payload = b"test";

    let sig = auth.sign(version, msg_type, seq, ts, payload);

    let result = MessageAuth::verify(
        pair.public_key(),
        version,
        msg_type + 1,
        seq,
        ts,
        payload,
        &sig,
    );
    assert_eq!(result, Err(CryptoError::VerificationFailed));
}

#[test]
fn verification_fails_on_modified_version() {
    let seed = [10u8; 32];
    let pair = KeyPair::from_seed(seed).expect("non-zero seed");
    let auth = MessageAuth::new(pair.clone());

    let version = (0, 1);
    let msg_type = 1;
    let seq = 100;
    let ts = 1234567890;
    let payload = b"test";

    let sig = auth.sign(version, msg_type, seq, ts, payload);

    let result = MessageAuth::verify(pair.public_key(), (1, 1), msg_type, seq, ts, payload, &sig);
    assert_eq!(result, Err(CryptoError::VerificationFailed));
}

#[test]
fn verification_fails_on_invalid_public_key_bytes() {
    let seed = [11u8; 32];
    let pair = KeyPair::from_seed(seed).expect("non-zero seed");
    let auth = MessageAuth::new(pair);

    let version = (0, 1);
    let msg_type = 1;
    let seq = 100;
    let ts = 1234567890;
    let payload = b"test";
    let sig = auth.sign(version, msg_type, seq, ts, payload);

    let invalid_key = [0xFFu8; 32];
    let result = MessageAuth::verify(&invalid_key, version, msg_type, seq, ts, payload, &sig);
    assert!(matches!(
        result,
        Err(CryptoError::InvalidPublicKey) | Err(CryptoError::VerificationFailed)
    ));
}

#[test]
fn invalid_signature_encoding_rejected() {
    let seed = [12u8; 32];
    let pair = KeyPair::from_seed(seed).expect("non-zero seed");

    let version = (0, 1);
    let msg_type = 1;
    let seq = 100;
    let ts = 1234567890;
    let payload = b"test";
    let bad_sig = Signature([0xFF; 64]);

    let result = MessageAuth::verify(
        pair.public_key(),
        version,
        msg_type,
        seq,
        ts,
        payload,
        &bad_sig,
    );
    assert_eq!(result, Err(CryptoError::VerificationFailed));
}

#[test]
fn envelope_v1_and_session_handshake_bind_all_scope_fields() {
    let sender = KeyPair::from_seed([21u8; 32]).unwrap();
    let receiver = KeyPair::from_seed([22u8; 32]).unwrap();
    let auth = MessageAuth::new(sender.clone());

    let ctx = EnvelopeContextV1 {
        version: (0, 2),
        swarm_instance_id: [0xA5; 16],
        session_epoch: 3,
        round_id: 17,
        message_type: 1,
        sequence: 9,
        timestamp: 1_700_000_100,
    };
    let payload = b"scoped gradient update";
    let sig_v1 = auth.sign_v1(&ctx, payload);
    assert!(MessageAuth::verify_v1(sender.public_key(), &ctx, payload, &sig_v1).is_ok());

    // Tampering with swarm_instance_id, session_epoch, or round_id must fail verification.
    let mut wrong_swarm = ctx;
    wrong_swarm.swarm_instance_id = [0x5A; 16];
    assert_eq!(
        MessageAuth::verify_v1(sender.public_key(), &wrong_swarm, payload, &sig_v1),
        Err(CryptoError::VerificationFailed)
    );

    let mut wrong_epoch = ctx;
    wrong_epoch.session_epoch = 4;
    assert_eq!(
        MessageAuth::verify_v1(sender.public_key(), &wrong_epoch, payload, &sig_v1),
        Err(CryptoError::VerificationFailed)
    );

    // Session handshake challenge signing and verification.
    let challenge = SessionHandshakeChallenge {
        swarm_instance_id: [0xA5; 16],
        receiver_peer: receiver.peer_id(),
        sender_peer: sender.peer_id(),
        requested_epoch: 4,
        initial_sequence: 1,
        challenge_nonce: [0x77; 32],
        issued_at_secs: 1_700_000_200,
    };
    let hs_sig = auth.sign_session_handshake(&challenge);
    assert!(
        MessageAuth::verify_session_handshake(sender.public_key(), &challenge, &hs_sig).is_ok()
    );

    // Replaying with modified nonce must fail.
    let mut bad_challenge = challenge;
    bad_challenge.challenge_nonce[0] ^= 0x01;
    assert_eq!(
        MessageAuth::verify_session_handshake(sender.public_key(), &bad_challenge, &hs_sig),
        Err(CryptoError::VerificationFailed)
    );
}

#[test]
fn h01_peer_id_matches_from_public_key() {
    let seed = [42u8; 32];
    let pair = KeyPair::from_seed(seed).expect("non-zero seed");

    let via_method = pair.peer_id();
    let via_factory = PeerId::from_public_key(pair.public_key());

    assert_eq!(
        via_method, via_factory,
        "KeyPair::peer_id() must agree with PeerId::from_public_key()"
    );
    assert_ne!(
        *via_method.as_bytes(),
        *pair.public_key(),
        "PeerId must be a SHA-256 hash of the public key, not the raw key"
    );
}

#[test]
fn from_seed_rejects_zero_seed() {
    let result = KeyPair::from_seed([0u8; 32]);
    assert!(
        matches!(result, Err(CryptoError::AllZeroSeed)),
        "from_seed must reject all-zero seed"
    );
}

#[test]
fn from_seed_accepts_nonzero_seed() {
    let result = KeyPair::from_seed([1u8; 32]);
    assert!(result.is_ok());
}

#[test]
#[cfg(feature = "std")]
fn sqrt_f32_precision_within_1e6_relative_error() {
    let test_values: &[f32] = &[
        0.001, 0.01, 0.1, 1.0, 2.0, 4.0, 9.0, 16.0, 100.0, 1000.0, 1e6, 1e-6, 1e10, 1e-10, 1e20,
        1e-20, 1e30, 1e-30,
    ];
    for &x in test_values {
        let ours = sqrt_f32(x);
        let reference = x.sqrt();
        let rel_err = ((ours - reference) / reference).abs();
        assert!(
            rel_err < 1e-6,
            "sqrt_f32({}) = {} vs std {} (relative error {})",
            x,
            ours,
            reference,
            rel_err
        );
    }
}
