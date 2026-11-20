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
fn sqrt_f32_precision_within_1e6_relative_error() {
    let test_values: &[f32] = &[
        0.001, 0.01, 0.1, 1.0, 2.0, 4.0, 9.0, 16.0, 100.0, 1000.0, 1e6, 1e-6, 1e10, 1e-10, 1e20,
        1e-20, 1e30, 1e-30,
    ];
    for &x in test_values {
        let ours = sqrt_f32(x);
        let soft = crate::num::sqrt_f64_soft(x as f64) as f32;
        let rel_err = ((ours - soft) / soft).abs();
        assert!(
            rel_err < 1e-6,
            "sqrt_f32({}) = {} vs soft {} (relative error {})",
            x,
            ours,
            soft,
            rel_err
        );
    }
}

#[test]
fn gradient_validator_rejects_f32_norm_sq_overflow_without_nan_bypass() {
    // Regression test for F-01: two finite coordinates of 2e19 have norm_sq = 8e38 > f32::MAX
    // (~3.4028e38) and true L2 norm ~2.8284e19.
    // Casting norm_sq to f32 before sqrt previously produced +Inf -> NaN under no_std,
    // bypassing `norm > max_gradient_norm`.
    let validator = GradientValidator {
        max_coordinate_value: f32::MAX,
        max_gradient_norm: 2.5e19,
    };
    let gradients = [2e19_f32, 2e19_f32];
    let err = validator
        .validate(&gradients)
        .expect_err("norm ~2.828e19 > 2.5e19 must be rejected without NaN bypass");
    assert!(matches!(err, GradientValidationError::NormTooLarge { .. }));

    // Conversely, when max_gradient_norm = 3.0e19 (> 2.8284e19), evaluating sqrt in f64
    // avoids premature f32 overflow of norm_sq = 8e38.
    let permissive = GradientValidator {
        max_coordinate_value: f32::MAX,
        max_gradient_norm: 3.0e19,
    };
    assert!(permissive.validate(&gradients).is_ok());
}

#[test]
fn derive_session_id_v1_rejects_zero_nonce_and_matches_pinned_vector() {
    let swarm_id = [0x11u8; 16];
    let run_id = [0x22u8; 16];
    let policy_digest = [0x33u8; 32];
    let receiver = PeerId::new([0x44u8; 32]);
    let sender = PeerId::new([0x55u8; 32]);
    let epoch = 7u64;

    assert_eq!(
        derive_session_id_v1(
            &swarm_id,
            &run_id,
            &policy_digest,
            &receiver,
            &sender,
            epoch,
            &[0u8; 32]
        ),
        Err(CryptoError::VerificationFailed)
    );

    let nonce = [0x66u8; 32];
    let sid = derive_session_id_v1(
        &swarm_id,
        &run_id,
        &policy_digest,
        &receiver,
        &sender,
        epoch,
        &nonce,
    )
    .expect("non-zero nonce must derive session id");

    // Pinned SHA-256 preimage equivalence vector for cross-platform determinism (`ST-03`).
    let expected = sha256_slices(&[
        b"swarmtorch.session.id.v1",
        &swarm_id,
        &run_id,
        &policy_digest,
        receiver.as_bytes(),
        sender.as_bytes(),
        &7u64.to_le_bytes(),
        &nonce,
    ]);
    assert_eq!(sid, expected);
    assert_eq!(
        sid,
        [
            0xbf, 0x97, 0xcd, 0xfc, 0x4c, 0xfa, 0x2d, 0x6c, 0x80, 0x11, 0x37, 0xea, 0xe4, 0xef,
            0x3a, 0x3f, 0xcf, 0x12, 0xd9, 0x78, 0x28, 0x82, 0xec, 0x56, 0xbb, 0x86, 0xd1, 0x83,
            0xf6, 0x33, 0x2a, 0x95,
        ]
    );
}

#[test]
fn envelope_v2_and_handshake_v2_bind_run_policy_and_session_id() {
    let sender = KeyPair::from_seed([0x21; 32]).unwrap();
    let receiver = KeyPair::from_seed([0x22; 32]).unwrap();
    let auth = MessageAuth::new(sender.clone());

    let challenge = SessionHandshakeChallengeV2 {
        swarm_instance_id: [0xA5; 16],
        run_id: [0xB6; 16],
        policy_digest: [0xC7; 32],
        receiver_peer: receiver.peer_id(),
        sender_peer: sender.peer_id(),
        requested_epoch: 1,
        initial_sequence: 0,
        receiver_nonce: [0x77; 32],
        issued_at_secs: 1_700_000_000,
    };
    let session_id = challenge.derive_session_id().unwrap();
    let hs_sig = auth.sign_session_handshake_v2(&challenge).unwrap();
    assert_eq!(
        MessageAuth::verify_session_handshake_v2(sender.public_key(), &challenge, &hs_sig),
        Ok(session_id)
    );

    let mut bad_run = challenge;
    bad_run.run_id[0] ^= 0x01;
    assert_eq!(
        MessageAuth::verify_session_handshake_v2(sender.public_key(), &bad_run, &hs_sig),
        Err(CryptoError::VerificationFailed)
    );

    let ctx = EnvelopeContextV2 {
        version: (0, 3),
        swarm_instance_id: challenge.swarm_instance_id,
        run_id: challenge.run_id,
        session_epoch: challenge.requested_epoch,
        session_id,
        round_id: 4,
        message_type: 1,
        sequence: 12,
        timestamp: 1_700_000_010,
    };
    let payload = b"v2 session-bound payload";
    let sig_v2 = auth.sign_v2(&ctx, payload);
    assert!(MessageAuth::verify_v2(sender.public_key(), &ctx, payload, &sig_v2).is_ok());

    let mut wrong_session = ctx;
    wrong_session.session_id[31] ^= 0x80;
    assert_eq!(
        MessageAuth::verify_v2(sender.public_key(), &wrong_session, payload, &sig_v2),
        Err(CryptoError::VerificationFailed)
    );

    let mut wrong_run = ctx;
    wrong_run.run_id[0] ^= 0x01;
    assert_eq!(
        MessageAuth::verify_v2(sender.public_key(), &wrong_run, payload, &sig_v2),
        Err(CryptoError::VerificationFailed)
    );
}

#[test]
fn sha256_counter_drbg_is_deterministic_domain_separated_and_bounded() {
    let seed = [0x42u8; 32];
    let domain = b"swarmtorch.test.domain.a";
    let mut a = Sha256CounterDrbgV1::new(domain, seed, 10, 2);
    let mut a2 = Sha256CounterDrbgV1::new(domain, seed, 10, 2);
    let mut diff_domain = Sha256CounterDrbgV1::new(b"swarmtorch.test.domain.b", seed, 10, 2);
    let mut diff_param_a = Sha256CounterDrbgV1::new(domain, seed, 11, 2);
    let mut diff_param_b = Sha256CounterDrbgV1::new(domain, seed, 10, 3);

    // Independent SHA-256 known-answer test across block counter = 0 and counter = 1 (16 words).
    for counter in 0u64..2 {
        let mut h = Sha256::new();
        h.update(domain);
        h.update(seed);
        h.update(10u64.to_le_bytes());
        h.update(2u64.to_le_bytes());
        h.update(counter.to_le_bytes());
        let expected_block = h.finalize();
        for chunk in expected_block.chunks_exact(4) {
            let expected_word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let wa = a.next_u32();
            assert_eq!(wa, expected_word);
            assert_eq!(wa, a2.next_u32());
            assert_ne!(wa, diff_domain.next_u32());
            assert_ne!(wa, diff_param_a.next_u32());
            assert_ne!(wa, diff_param_b.next_u32());
        }
    }

    // Verify next_u64() little-endian word composition against two next_u32() draws.
    let mut u64_drbg = Sha256CounterDrbgV1::new(domain, seed, 7, 9);
    let mut u32_drbg = Sha256CounterDrbgV1::new(domain, seed, 7, 9);
    for _ in 0..8 {
        let lo = u32_drbg.next_u32() as u64;
        let hi = u32_drbg.next_u32() as u64;
        assert_eq!(u64_drbg.next_u64(), lo | (hi << 32));
    }

    // Verify uniform_u32_inclusive(u32::MAX) consumes 1 draw with threshold == 0.
    let mut max_inc_drbg = Sha256CounterDrbgV1::new(domain, seed, 1, 1);
    let mut raw_drbg = Sha256CounterDrbgV1::new(domain, seed, 1, 1);
    for _ in 0..16 {
        assert_eq!(
            max_inc_drbg.uniform_u32_inclusive(u32::MAX),
            raw_drbg.next_u32()
        );
    }

    // Regression test for Lemire 32-bit threshold (2^32 % s, NOT 2^64 % s):
    // For bound = 2^32 - 65535 = 4_294_901_761, true threshold is 65_535 (~1.000015 draws/sample),
    // whereas buggy 2^64 % s = 4_294_836_225 spun ~32,768 draws/sample.
    let mut large_bound_drbg = Sha256CounterDrbgV1::new(domain, seed, 99, 100);
    let large_bound: u32 = 4_294_901_761;
    for _ in 0..16 {
        let v = large_bound_drbg.uniform_u32_bounded(large_bound);
        assert!(v < large_bound);
    }
    assert!(
        large_bound_drbg.counter <= 4,
        "Lemire 32-bit rejection should consume ~2 blocks (<=4) for 16 draws, consumed {} blocks",
        large_bound_drbg.counter
    );

    assert_eq!(a.uniform_u32_bounded(1), 0);
    assert_eq!(a.uniform_u32_inclusive(0), 0);
    for _ in 0..128 {
        assert!(a.uniform_u32_bounded(7) < 7);
        assert!(a.uniform_u32_inclusive(7) <= 7);
        let u = a.next_f64_unit_half_open();
        assert!(u.is_finite() && (0.0..1.0).contains(&u));
        let s = a.next_f64_signed_unit();
        assert!(s.is_finite() && (-1.0..1.0).contains(&s));
    }
}

#[test]
#[should_panic(expected = "uniform_u32_bounded requires bound >= 1")]
fn sha256_counter_drbg_rejects_zero_bound() {
    let mut drbg = Sha256CounterDrbgV1::new(b"swarmtorch.test.zero", [0x22; 32], 0, 0);
    let _ = drbg.uniform_u32_bounded(0);
}

#[test]
#[should_panic(expected = "Sha256CounterDrbgV1 64-bit block counter overflow")]
fn sha256_counter_drbg_panics_instead_of_wrapping_on_counter_overflow() {
    let mut drbg = Sha256CounterDrbgV1::new(b"swarmtorch.test.overflow", [0x11; 32], 0, 0);
    drbg.counter = u64::MAX;
    let _ = drbg.next_u32();
}
