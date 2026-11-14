use super::*;
use crate::crypto::{KeyPair, MessageAuth, SessionHandshakeChallenge, SessionHandshakeChallengeV2};
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

fn make_peer(id: u8) -> PeerId {
    PeerId::new([id; 32])
}

fn new_genesis_guard() -> ReplayProtection {
    ReplayProtection::new().with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
}

#[test]
fn validate_accepts_first_message_from_peer_under_genesis_bootstrap() {
    let mut guard = new_genesis_guard();
    let peer = make_peer(1);
    let now = 1000;

    assert!(guard.validate(&peer, 1, now, now).is_ok());
}

#[test]
fn try_with_config_rejects_zero_capacity() {
    let result = ReplayProtection::try_with_config(0, 60);
    assert!(matches!(result, Err(ReplayConfigError::ZeroCapacity)));
}

#[test]
fn try_with_config_accepts_valid_capacity() {
    let result = ReplayProtection::try_with_config(32, 60);
    assert!(result.is_ok());
}

#[test]
fn validate_accepts_monotonic_sequence() {
    let mut guard = new_genesis_guard();
    let peer = make_peer(1);
    let now = 1000;

    assert!(guard.validate(&peer, 1, now, now).is_ok());
    assert!(guard.validate(&peer, 2, now, now).is_ok());
    assert!(guard.validate(&peer, 3, now, now).is_ok());
    assert!(guard.validate(&peer, 100, now, now).is_ok());
}

#[test]
fn validate_rejects_duplicate_sequence() {
    let mut guard = new_genesis_guard();
    let peer = make_peer(1);
    let now = 1000;

    assert!(guard.validate(&peer, 1, now, now).is_ok());

    let result = guard.validate(&peer, 1, now, now);
    assert_eq!(result, Err(ReplayError::Replay { peer, seq: 1 }));
}

#[test]
fn validate_rejects_retrograde_sequence() {
    let mut guard = new_genesis_guard();
    let peer = make_peer(1);
    let now = 1000;

    assert!(guard.validate(&peer, 100, now, now).is_ok());

    let result = guard.validate(&peer, 50, now, now);
    assert_eq!(
        result,
        Err(ReplayError::TooOld {
            peer,
            seq: 50,
            last_seen: 100,
        })
    );
}

#[test]
fn validate_accepts_out_of_order_within_window() {
    let mut guard = new_genesis_guard();
    let peer = make_peer(1);
    let now = 1000;

    assert!(guard.validate(&peer, 20, now, now).is_ok());
    assert!(guard.validate(&peer, 10, now, now).is_ok());
    assert!(guard.validate(&peer, 5, now, now).is_ok());
}

#[test]
fn sequence_window_check_no_overflow_at_u64_max() {
    let mut guard = new_genesis_guard();
    let peer = make_peer(42);
    let now = 1000;

    let near_max = u64::MAX - 5;
    assert!(guard.validate(&peer, near_max, now, now).is_ok());
    assert!(guard.validate(&peer, u64::MAX - 3, now, now).is_ok());
    assert!(guard.validate(&peer, u64::MAX, now, now).is_ok());

    let old_seq = near_max.saturating_sub(20);
    let result = guard.validate(&peer, old_seq, now, now);
    assert!(matches!(result, Err(ReplayError::TooOld { .. })));
}

#[test]
fn validate_rejects_timestamp_too_old() {
    let guard = ReplayProtection::try_with_config(100, 60).unwrap();
    let now = 1000;
    let old_ts = now - 100;

    let result = guard.check_timestamp_only(old_ts, now);
    assert_eq!(
        result,
        Err(ReplayError::Expired {
            ts: old_ts,
            now,
            window: 60,
        })
    );
}

#[test]
fn validate_rejects_timestamp_too_new() {
    let guard = ReplayProtection::try_with_config(100, 60).unwrap();
    let now = 1000;
    let future_ts = now + 100;

    let result = guard.check_timestamp_only(future_ts, now);
    assert_eq!(
        result,
        Err(ReplayError::Expired {
            ts: future_ts,
            now,
            window: 60,
        })
    );
}

#[test]
fn validate_accepts_timestamp_within_skew_window() {
    let guard = ReplayProtection::try_with_config(100, 60).unwrap();
    let now = 1000;

    assert!(guard.check_timestamp_only(now - 50, now).is_ok());
    assert!(guard.check_timestamp_only(now, now).is_ok());
    assert!(guard.check_timestamp_only(now + 50, now).is_ok());
    assert!(guard.check_timestamp_only(now - 60, now).is_ok());
    assert!(guard.check_timestamp_only(now + 60, now).is_ok());
}

#[test]
fn expiry_indexed_reclamation_prevents_in_window_eviction_and_reclaims_after_window() {
    let mut guard = ReplayProtection::try_with_config(10, 60)
        .unwrap()
        .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap);
    let now = 1000;

    // Fill cache with 10 peers at t = 1000 (expires_at = 1060).
    for i in 1..=10 {
        let peer = make_peer(i);
        assert!(guard.validate(&peer, 1, now, now).is_ok());
    }
    assert_eq!(guard.cache_size(), 10);

    // 11th peer while all 10 peers are still active in-window MUST NOT evict peer_1!
    let peer_11 = make_peer(11);
    assert_eq!(
        guard.validate(&peer_11, 1, now, now),
        Err(ReplayError::CacheFull { capacity: 10 })
    );
    assert_eq!(guard.cache_size(), 10);

    // Peer 1 is still protected against in-window replay at t = 1000!
    let peer_1 = make_peer(1);
    assert_eq!(
        guard.validate(&peer_1, 1, now, now),
        Err(ReplayError::Replay {
            peer: peer_1,
            seq: 1
        })
    );

    // Once current_time advances past t_max + A (1061 > 1000 + 60), expired records are reclaimed.
    let after_window = 1061;
    assert!(guard
        .validate(&peer_11, 1, after_window, after_window)
        .is_ok());
    assert_eq!(guard.cache_size(), 1);

    // Replaying peer_1's old message (ts = 1000) at now = 1061 fails the stateless timestamp check!
    assert_eq!(
        guard.validate(&peer_1, 1, now, after_window),
        Err(ReplayError::Expired {
            ts: now,
            now: after_window,
            window: 60
        })
    );
}

#[test]
fn cohort_allowlist_rejects_unauthorized_peer_without_state_mutation() {
    let peer_allowed = make_peer(1);
    let peer_blocked = make_peer(2);
    let allowlist = CohortAllowlist::from_peers([peer_allowed]);
    let mut guard = new_genesis_guard().with_cohort_allowlist(allowlist);

    assert!(guard.validate(&peer_allowed, 1, 1000, 1000).is_ok());
    assert_eq!(guard.cache_size(), 1);

    assert_eq!(
        guard.validate(&peer_blocked, 1, 1000, 1000),
        Err(ReplayError::UnauthorizedPeer { peer: peer_blocked })
    );
    assert_eq!(guard.cache_size(), 1);
}

#[test]
fn epoch_recovery_requires_signed_receiver_challenge_handshake() {
    let swarm_id = [0x11u8; 16];
    let sender_kp = KeyPair::from_seed([31u8; 32]).unwrap();
    let receiver_kp = KeyPair::from_seed([32u8; 32]).unwrap();
    let sender_auth = MessageAuth::new(sender_kp.clone());
    let sender_peer = sender_kp.peer_id();
    let mut guard = new_genesis_guard();
    let now = 2000;

    // Epoch 1 established at seq = 10.
    let key_e1 = ReplayRecordKey::new(swarm_id, sender_peer, 1);
    assert!(guard.validate_scoped(&key_e1, 10, now, now).is_ok());

    // Sender crashes and attempts to jump to epoch 2 at seq = 1 without handshake -> RecoveryRequired.
    let key_e2 = ReplayRecordKey::new(swarm_id, sender_peer, 2);
    assert_eq!(
        guard.validate_scoped(&key_e2, 1, now, now),
        Err(ReplayError::RecoveryRequired {
            peer: sender_peer,
            current_epoch: 1,
            requested_epoch: 2,
        })
    );

    // Receiver issues a 256-bit challenge nonce for epoch 2.
    let challenge = SessionHandshakeChallenge {
        swarm_instance_id: swarm_id,
        receiver_peer: receiver_kp.peer_id(),
        sender_peer,
        requested_epoch: 2,
        initial_sequence: 1,
        challenge_nonce: [0x99; 32],
        issued_at_secs: now,
    };
    guard.register_recovery_challenge(challenge).unwrap();

    // Sender signs the challenge under swarmtorch.session.handshake.v1.
    let sig = sender_auth.sign_session_handshake(&challenge);
    let recovered_key = guard
        .complete_recovery_handshake(swarm_id, sender_kp.public_key(), &sig, now)
        .unwrap();
    assert_eq!(recovered_key, key_e2);

    // Replaying the same handshake signature fails because the challenge was consumed.
    assert_eq!(
        guard.complete_recovery_handshake(swarm_id, sender_kp.public_key(), &sig, now),
        Err(ReplayError::HandshakeVerificationFailed)
    );

    // Stale epoch 1 is now rejected as StaleSessionEpoch, while epoch 2 seq = 2 succeeds!
    assert_eq!(
        guard.validate_scoped(&key_e1, 11, now, now),
        Err(ReplayError::StaleSessionEpoch {
            peer: sender_peer,
            current_epoch: 2,
            received_epoch: 1,
        })
    );
    assert!(guard.validate_scoped(&key_e2, 2, now, now).is_ok());
}

#[test]
fn peer_state_isolation() {
    let mut guard = new_genesis_guard();
    let peer_a = make_peer(1);
    let peer_b = make_peer(2);
    let now = 1000;

    assert!(guard.validate(&peer_a, 10, now, now).is_ok());
    assert!(guard.validate(&peer_b, 5, now, now).is_ok());

    assert!(guard.validate(&peer_a, 10, now, now).is_err());
    assert!(guard.validate(&peer_b, 10, now, now).is_ok());
}

#[test]
fn sequence_window_prunes_correctly() {
    let mut guard = new_genesis_guard();
    let peer = make_peer(1);
    let now = 1000;

    for seq in 1..=20 {
        assert!(guard.validate(&peer, seq, now, now).is_ok());
    }

    assert!(guard.validate(&peer, 100, now, now).is_ok());

    let result = guard.validate(&peer, 50, now, now);
    assert_eq!(
        result,
        Err(ReplayError::TooOld {
            peer,
            seq: 50,
            last_seen: 100,
        })
    );
}

#[test]
fn deterministic_btreeset_ordering() {
    let mut state = PeerReplayState::new(1, 1000, DEFAULT_MAX_CLOCK_SKEW_SECS);
    let peer = make_peer(1);

    assert!(state
        .validate_and_update(5, peer, DEFAULT_SEQUENCE_TOLERANCE_WINDOW)
        .is_ok());
    assert!(state
        .validate_and_update(3, peer, DEFAULT_SEQUENCE_TOLERANCE_WINDOW)
        .is_ok());
    assert!(state
        .validate_and_update(4, peer, DEFAULT_SEQUENCE_TOLERANCE_WINDOW)
        .is_ok());
    assert!(state
        .validate_and_update(2, peer, DEFAULT_SEQUENCE_TOLERANCE_WINDOW)
        .is_ok());

    let sequences: Vec<u64> = state.recent_sequences.iter().copied().collect();
    assert_eq!(sequences, vec![1, 2, 3, 4, 5]);
}

#[test]
fn check_timestamp_only_does_not_mutate_state() {
    let guard = ReplayProtection::new();
    let now = 1000;

    assert!(guard.check_timestamp_only(now, now).is_ok());
    assert!(guard.check_timestamp_only(now, now).is_ok());
    assert!(guard.check_timestamp_only(now - 30, now).is_ok());
    assert_eq!(guard.cache_size(), 0);
}

#[test]
fn validate_sequence_mutates_state() {
    let mut guard = new_genesis_guard();
    let peer = make_peer(1);

    assert!(guard.validate_sequence(&peer, 1).is_ok());
    assert_eq!(guard.cache_size(), 1);

    assert!(guard.validate_sequence(&peer, 1).is_err());
}

#[test]
fn display_error_formats_correctly() {
    let peer = make_peer(1);

    let expired = ReplayError::Expired {
        ts: 900,
        now: 1000,
        window: 60,
    };
    assert_eq!(
        format!("{}", expired),
        "timestamp expired: ts=900, now=1000, window=60s"
    );

    let replay = ReplayError::Replay { peer, seq: 42 };
    assert!(format!("{}", replay).contains("replay detected"));
    assert!(format!("{}", replay).contains("seq=42"));

    let too_old = ReplayError::TooOld {
        peer,
        seq: 10,
        last_seen: 100,
    };
    assert!(format!("{}", too_old).contains("sequence too old"));
    assert!(format!("{}", too_old).contains("seq=10"));
    assert!(format!("{}", too_old).contains("last_seen=100"));
}

#[test]
fn c01_boundary_sequence_rejected_on_second_submission() {
    let mut guard = new_genesis_guard();
    let peer = make_peer(99);
    let now = 1000;

    let n: u64 = 100;
    assert!(guard.validate(&peer, n, now, now).is_ok());

    let boundary_seq = n - DEFAULT_SEQUENCE_TOLERANCE_WINDOW as u64;

    let result = guard.validate(&peer, boundary_seq, now, now);
    assert_eq!(
        result,
        Err(ReplayError::TooOld {
            peer,
            seq: boundary_seq,
            last_seen: n,
        }),
        "boundary sequence must be rejected as TooOld"
    );
}

#[test]
fn c01_sequence_inside_window_still_accepted() {
    let mut guard = new_genesis_guard();
    let peer = make_peer(98);
    let now = 1000;

    let n: u64 = 100;
    assert!(guard.validate(&peer, n, now, now).is_ok());

    let inside_seq = n - DEFAULT_SEQUENCE_TOLERANCE_WINDOW as u64 + 1;
    assert!(
        guard.validate(&peer, inside_seq, now, now).is_ok(),
        "sequence one inside the window must be accepted"
    );

    let result = guard.validate(&peer, inside_seq, now, now);
    assert_eq!(
        result,
        Err(ReplayError::Replay {
            peer,
            seq: inside_seq,
        }),
        "duplicate of accepted sequence must be Replay"
    );
}

#[test]
fn configurable_tolerance_window_smaller_rejects_further_out_of_order() {
    let mut guard = ReplayProtection::try_with_tolerance_window(100, 60, 4)
        .unwrap()
        .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap);
    let peer = make_peer(200);
    let now = 1000;

    assert!(guard.validate(&peer, 50, now, now).is_ok());
    assert!(guard.validate(&peer, 47, now, now).is_ok());

    let result = guard.validate(&peer, 45, now, now);
    assert_eq!(
        result,
        Err(ReplayError::TooOld {
            peer,
            seq: 45,
            last_seen: 50,
        })
    );
}

#[test]
fn configurable_tolerance_window_zero_enforces_strict_monotonic() {
    let mut guard = ReplayProtection::try_with_tolerance_window(100, 60, 0)
        .unwrap()
        .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap);
    let peer = make_peer(201);
    let now = 1000;

    assert!(guard.validate(&peer, 10, now, now).is_ok());

    let result = guard.validate(&peer, 10, now, now);
    assert!(
        result.is_err(),
        "duplicate at zero tolerance must be rejected"
    );

    let result = guard.validate(&peer, 9, now, now);
    assert_eq!(
        result,
        Err(ReplayError::TooOld {
            peer,
            seq: 9,
            last_seen: 10,
        })
    );

    assert!(guard.validate(&peer, 11, now, now).is_ok());
}

#[test]
fn tolerance_window_too_large_rejected() {
    let result = ReplayProtection::try_with_tolerance_window(100, 60, 257);
    assert_eq!(
        result.unwrap_err(),
        ReplayConfigError::ToleranceWindowTooLarge {
            window: 257,
            max: MAX_SEQUENCE_TOLERANCE_WINDOW,
        }
    );

    assert!(ReplayProtection::try_with_tolerance_window(100, 60, 256).is_ok());
}

#[test]
fn default_startup_policy_requires_session_handshake_before_delivery() {
    let swarm_id = [0x11u8; 16];
    let run_id = [0x22u8; 16];
    let policy_digest = [0x33u8; 32];
    let sender_kp = KeyPair::from_seed([41u8; 32]).unwrap();
    let receiver_kp = KeyPair::from_seed([42u8; 32]).unwrap();
    let sender_peer = sender_kp.peer_id();
    let mut guard = ReplayProtection::new();
    assert_eq!(
        guard.startup_policy(),
        ReplayStartupPolicyV1::RequireSessionHandshake
    );

    let now = 2_000;
    let fake_sid = [0x99u8; 32];
    let v1_key = ReplayRecordKey::new(swarm_id, sender_peer, 1);
    assert_eq!(
        guard.validate_scoped(&v1_key, 1, now, now),
        Err(ReplayError::RecoveryRequired {
            peer: sender_peer,
            current_epoch: 0,
            requested_epoch: 1,
        })
    );

    let v2_key_unestablished = ReplayRecordKeyV2::new(swarm_id, run_id, sender_peer, 1, fake_sid);
    assert_eq!(
        guard.validate_session_v2_at(&v2_key_unestablished, 1, now, now),
        Err(ReplayError::RecoveryRequired {
            peer: sender_peer,
            current_epoch: 0,
            requested_epoch: 1,
        })
    );

    // Register & complete v2 handshake -> peer is now admitted for its bound session_id only.
    let challenge = SessionHandshakeChallengeV2 {
        swarm_instance_id: swarm_id,
        run_id,
        policy_digest,
        receiver_peer: receiver_kp.peer_id(),
        sender_peer,
        requested_epoch: 1,
        initial_sequence: 0,
        receiver_nonce: [0x55u8; 32],
        issued_at_secs: now,
    };
    guard.register_recovery_challenge_v2(challenge).unwrap();
    let hs_sig = MessageAuth::new(sender_kp.clone())
        .sign_session_handshake_v2(&challenge)
        .unwrap();
    let active_key = guard
        .complete_recovery_handshake_v2(swarm_id, run_id, sender_kp.public_key(), &hs_sig, now)
        .unwrap();

    assert_eq!(
        guard.validate_session_v2_at(&v2_key_unestablished, 1, now, now),
        Err(ReplayError::SessionIdMismatch { peer: sender_peer })
    );
    assert!(guard
        .validate_session_v2_at(&active_key, 1, now, now)
        .is_ok());
}

#[test]
fn crash_restart_invalidates_prior_session_id_even_within_same_wall_clock_second() {
    let swarm_id = [0x11u8; 16];
    let run_id = [0x22u8; 16];
    let policy_digest = [0x33u8; 32];
    let sender_kp = KeyPair::from_seed([43u8; 32]).unwrap();
    let receiver_kp = KeyPair::from_seed([44u8; 32]).unwrap();
    let sender_auth = MessageAuth::new(sender_kp.clone());
    let sender_peer = sender_kp.peer_id();
    let now = 5_000;

    let mut pre_crash_guard = ReplayProtection::new();
    let challenge_1 = SessionHandshakeChallengeV2 {
        swarm_instance_id: swarm_id,
        run_id,
        policy_digest,
        receiver_peer: receiver_kp.peer_id(),
        sender_peer,
        requested_epoch: 1,
        initial_sequence: 0,
        receiver_nonce: [0xAAu8; 32],
        issued_at_secs: now,
    };
    pre_crash_guard
        .register_recovery_challenge_v2(challenge_1)
        .unwrap();
    let sig_1 = sender_auth.sign_session_handshake_v2(&challenge_1).unwrap();
    let key_1 = pre_crash_guard
        .complete_recovery_handshake_v2(swarm_id, run_id, sender_kp.public_key(), &sig_1, now)
        .unwrap();
    assert!(pre_crash_guard
        .validate_session_v2_at(&key_1, 1, now, now)
        .is_ok());

    // Process crashes and restarts in the SAME wall-clock second (`now = 5_000`) with empty replay state.
    let mut restarted_guard = ReplayProtection::new().with_timestamp_floor(now);
    assert_eq!(
        restarted_guard.validate_session_v2_at(&key_1, 1, now, now),
        Err(ReplayError::RecoveryRequired {
            peer: sender_peer,
            current_epoch: 0,
            requested_epoch: 1,
        })
    );

    // Receiver issues a fresh 256-bit nonce (`0xBB`) even at same epoch=1 and same second `now = 5_000`.
    let challenge_2 = SessionHandshakeChallengeV2 {
        swarm_instance_id: swarm_id,
        run_id,
        policy_digest,
        receiver_peer: receiver_kp.peer_id(),
        sender_peer,
        requested_epoch: 1,
        initial_sequence: 0,
        receiver_nonce: [0xBBu8; 32],
        issued_at_secs: now,
    };
    restarted_guard
        .register_recovery_challenge_v2(challenge_2)
        .unwrap();
    let sig_2 = sender_auth.sign_session_handshake_v2(&challenge_2).unwrap();
    let key_2 = restarted_guard
        .complete_recovery_handshake_v2(swarm_id, run_id, sender_kp.public_key(), &sig_2, now)
        .unwrap();
    assert_ne!(key_1.session_id, key_2.session_id);

    // Pre-crash captured delivery carrying old `session_id` MUST still fail closed at same wall-clock second.
    assert_eq!(
        restarted_guard.validate_session_v2_at(&key_1, 1, now, now),
        Err(ReplayError::SessionIdMismatch { peer: sender_peer })
    );
    assert!(restarted_guard
        .validate_session_v2_at(&key_2, 1, now, now)
        .is_ok());
}

#[test]
fn atomic_handshake_preserves_challenge_and_prior_epoch_on_cache_full_or_bad_sig() {
    let swarm_id = [0x11u8; 16];
    let run_id = [0x22u8; 16];
    let policy_digest = [0x33u8; 32];
    let receiver_kp = KeyPair::from_seed([50u8; 32]).unwrap();
    let kp1 = KeyPair::from_seed([51u8; 32]).unwrap();
    let kp2 = KeyPair::from_seed([52u8; 32]).unwrap();
    let kp3 = KeyPair::from_seed([53u8; 32]).unwrap();
    let mut guard = ReplayProtection::try_with_config(2, 60).unwrap();

    // Fill cache to capacity=2 at t=1000 (`expires_at=1060`).
    for (idx, kp) in [&kp1, &kp2].iter().enumerate() {
        let ch = SessionHandshakeChallengeV2 {
            swarm_instance_id: swarm_id,
            run_id,
            policy_digest,
            receiver_peer: receiver_kp.peer_id(),
            sender_peer: kp.peer_id(),
            requested_epoch: 1,
            initial_sequence: 0,
            receiver_nonce: [(idx as u8) + 1; 32],
            issued_at_secs: 1000,
        };
        guard.register_recovery_challenge_v2(ch).unwrap();
        let sig = MessageAuth::new((*kp).clone())
            .sign_session_handshake_v2(&ch)
            .unwrap();
        guard
            .complete_recovery_handshake_v2(swarm_id, run_id, kp.public_key(), &sig, 1000)
            .unwrap();
    }
    assert_eq!(guard.cache_size(), 2);

    // New peer 3 registers challenge at t=1050; completion at t=1050 MUST fail closed with CacheFull
    // AND preserve pending challenge without mutating peers 1 & 2.
    let ch3 = SessionHandshakeChallengeV2 {
        swarm_instance_id: swarm_id,
        run_id,
        policy_digest,
        receiver_peer: receiver_kp.peer_id(),
        sender_peer: kp3.peer_id(),
        requested_epoch: 1,
        initial_sequence: 0,
        receiver_nonce: [0x77u8; 32],
        issued_at_secs: 1050,
    };
    guard.register_recovery_challenge_v2(ch3).unwrap();
    let sig3 = MessageAuth::new(kp3.clone())
        .sign_session_handshake_v2(&ch3)
        .unwrap();
    assert_eq!(
        guard.complete_recovery_handshake_v2(swarm_id, run_id, kp3.public_key(), &sig3, 1050),
        Err(ReplayError::CacheFull { capacity: 2 })
    );
    assert_eq!(guard.pending_challenge_count(), 1);
    assert_eq!(guard.cache_size(), 2);

    // Existing peer 1 rotating epoch 1->2 while full MUST preserve prior epoch if signature is wrong...
    let ch1_rot = SessionHandshakeChallengeV2 {
        swarm_instance_id: swarm_id,
        run_id,
        policy_digest,
        receiver_peer: receiver_kp.peer_id(),
        sender_peer: kp1.peer_id(),
        requested_epoch: 2,
        initial_sequence: 0,
        receiver_nonce: [0x88u8; 32],
        issued_at_secs: 1050,
    };
    guard.register_recovery_challenge_v2(ch1_rot).unwrap();
    let wrong_sig = MessageAuth::new(kp2.clone())
        .sign_session_handshake_v2(&ch1_rot)
        .unwrap();
    assert_eq!(
        guard.complete_recovery_handshake_v2(swarm_id, run_id, kp1.public_key(), &wrong_sig, 1050),
        Err(ReplayError::HandshakeVerificationFailed)
    );
    // Peer 1's prior epoch 1 remains active!
    assert_eq!(
        guard
            .active_v2_session(swarm_id, run_id, &kp1.peer_id())
            .map(|(ep, _)| ep),
        Some(1)
    );
    // ...and succeeds atomically once peer 1 presents the valid signature because its superseded slot frees!
    let good_sig1 = MessageAuth::new(kp1.clone())
        .sign_session_handshake_v2(&ch1_rot)
        .unwrap();
    let rotated_key1 = guard
        .complete_recovery_handshake_v2(swarm_id, run_id, kp1.public_key(), &good_sig1, 1050)
        .unwrap();
    assert_eq!(rotated_key1.session_epoch, 2);

    // At t=1061 peer 2 (last active t=1000, expires_at=1060) expired, so peer 3 can now complete its
    // previously preserved challenge ch3 without re-registration!
    let key3 = guard
        .complete_recovery_handshake_v2(swarm_id, run_id, kp3.public_key(), &sig3, 1061)
        .unwrap();
    assert_eq!(key3.peer, kp3.peer_id());
    assert_eq!(guard.pending_challenge_count(), 0);
}

#[test]
fn pending_challenges_table_is_bounded_and_fails_closed() {
    let swarm_id = [0x11u8; 16];
    let run_id = [0x22u8; 16];
    let policy_digest = [0x33u8; 32];
    let receiver_kp = KeyPair::from_seed([60u8; 32]).unwrap();
    assert_eq!(
        ReplayProtection::try_with_config(8, 60)
            .unwrap()
            .try_with_max_pending_challenges(0)
            .unwrap_err(),
        ReplayConfigError::ZeroPendingChallengesCapacity
    );

    let mut guard = ReplayProtection::try_with_config(8, 60)
        .unwrap()
        .try_with_max_pending_challenges(2)
        .unwrap();
    let kp1 = KeyPair::from_seed([61u8; 32]).unwrap();
    let kp2 = KeyPair::from_seed([62u8; 32]).unwrap();
    let kp3 = KeyPair::from_seed([63u8; 32]).unwrap();

    let mk = |kp: &KeyPair, nonce_byte: u8| SessionHandshakeChallengeV2 {
        swarm_instance_id: swarm_id,
        run_id,
        policy_digest,
        receiver_peer: receiver_kp.peer_id(),
        sender_peer: kp.peer_id(),
        requested_epoch: 1,
        initial_sequence: 0,
        receiver_nonce: [nonce_byte; 32],
        issued_at_secs: 1000,
    };

    guard.register_recovery_challenge_v2(mk(&kp1, 1)).unwrap();
    guard.register_recovery_challenge_v2(mk(&kp2, 2)).unwrap();
    // Refreshing existing pending key succeeds; new 3rd key fails closed with PendingChallengesFull.
    guard.register_recovery_challenge_v2(mk(&kp1, 11)).unwrap();
    assert_eq!(
        guard.register_recovery_challenge_v2(mk(&kp3, 3)),
        Err(ReplayError::PendingChallengesFull { capacity: 2 })
    );
    assert_eq!(guard.pending_challenge_count(), 2);
}

#[test]
fn timestamp_floor_and_readonly_preflight_enforce_bounds_without_mutation() {
    let swarm_id = [0x11u8; 16];
    let run_id = [0x22u8; 16];
    let policy_digest = [0x33u8; 32];
    let receiver_kp = KeyPair::from_seed([70u8; 32]).unwrap();
    let sender_kp = KeyPair::from_seed([71u8; 32]).unwrap();
    let sender_auth = MessageAuth::new(sender_kp.clone());
    let mut guard = ReplayProtection::try_with_config(8, 60)
        .unwrap()
        .with_timestamp_floor(2_000);

    let ch = SessionHandshakeChallengeV2 {
        swarm_instance_id: swarm_id,
        run_id,
        policy_digest,
        receiver_peer: receiver_kp.peer_id(),
        sender_peer: sender_kp.peer_id(),
        requested_epoch: 1,
        initial_sequence: 0,
        receiver_nonce: [0x12u8; 32],
        issued_at_secs: 2_000,
    };
    guard.register_recovery_challenge_v2(ch).unwrap();
    let hs_sig = sender_auth.sign_session_handshake_v2(&ch).unwrap();
    let key = guard
        .complete_recovery_handshake_v2(swarm_id, run_id, sender_kp.public_key(), &hs_sig, 2_000)
        .unwrap();

    // Timestamp 1999 is within 60s skew of now=2000, but MUST fail TimestampBelowFloor.
    assert_eq!(
        guard
            .check_v2_admission_readonly(&key, 1, 1_999, 2_000)
            .unwrap_err(),
        ReplayError::TimestampBelowFloor {
            ts: 1_999,
            floor: 2_000,
        }
    );

    // Read-only preflight at ts=2000 succeeds and does NOT advance sequence state.
    let prepared = guard
        .check_v2_admission_readonly(&key, 1, 2_000, 2_000)
        .unwrap();
    let _ = prepared;
    let prepared_again = guard
        .check_v2_admission_readonly(&key, 1, 2_000, 2_000)
        .unwrap();
    guard.commit_prepared_admission(prepared_again).unwrap();

    // Replaying sequence 1 after commit fails.
    assert_eq!(
        guard
            .check_v2_admission_readonly(&key, 1, 2_000, 2_000)
            .unwrap_err(),
        ReplayError::Replay {
            peer: sender_kp.peer_id(),
            seq: 1,
        }
    );
}
