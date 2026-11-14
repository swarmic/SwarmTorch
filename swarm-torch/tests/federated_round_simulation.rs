//! End-to-end deterministic multi-node honest & Byzantine federated training simulation
//! (`ADR-0022` §3, `ADR-0023` §1–3, `W10-03`).

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use swarm_torch::artifacts::{
    put_checkpoint_model, ArtifactWriteProfile, CheckpointContentDigestV1, DataOpsSession,
    LockAcquisitionPolicy, TrustPolicy,
};
use swarm_torch::coordinator::{
    encode_compressed_update_payload_v2, RoundCoordinatorConfig, RoundError, SubmissionDecision,
    SwarmRoundCoordinator, UpdateKindV1, WeightingPolicyV1,
    COMPRESSION_SCHEMA_COMPRESSED_GRADIENT_V1, COMPRESSION_SCHEMA_DENSE_F32_LE_V1,
};
use swarm_torch::native_runner::{NativeOpRunner, RunnerExecutionModeV1};
use swarm_torch::report::load_report_strict;
use swarm_torch::scheduler::execute_graph_sequential;
use swarm_torch_core::aggregation::{BulyanProfile, RobustAggregation};
use swarm_torch_core::compression::{ClassicResidualCompressor, CompressionMethod};
use swarm_torch_core::crypto::{KeyPair, MessageAuth, SessionHandshakeChallengeV2};
use swarm_torch_core::execution::PermissivePolicy;
use swarm_torch_core::observe::{AttrMap, EventRecord, RunEventEmitter, RunId, TraceId};
use swarm_torch_core::replay::ReplayError;
use swarm_torch_core::run_graph::{
    AssetRefV1, CachePolicyV0, CanonParams, ExecutionTrust, GraphV1, NodeV1, OpKind,
};
use swarm_torch_core::traits::{FitnessFunction, PeerId, SwarmModel, SwarmOptimizer};
use swarm_torch_models::loss::{FedProxRegularizer, MseLoss};
use swarm_torch_models::optim::SgdOptimizer;
use swarm_torch_models::simple::LinearModel;
use swarm_torch_net::gossip::{
    BundledGossipDeliveryV2, InnerUpdateV2, PayloadKindV1, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
};
use swarm_torch_net::traits::{ObservedSourceRef, SwarmTransport};
use swarm_torch_net::{MockNetworkHub, Topology};
use swarm_torch_runtime::mock_runtime::MockRuntime;
use swarm_torch_runtime::SwarmClock;

fn temp_dir(prefix: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut dir = std::env::temp_dir();
    dir.push(format!("swarmtorch_w10_sim_{prefix}_{pid}_{nanos}"));
    dir
}

struct PeerNode {
    keypair: KeyPair,
    peer_id: PeerId,
    auth: MessageAuth,
    shard: Vec<([f32; 2], [f32; 1])>,
}

impl PeerNode {
    fn new(seed: u8, shard: Vec<([f32; 2], [f32; 1])>) -> Self {
        let keypair = KeyPair::from_seed([seed; 32]).unwrap();
        let peer_id = keypair.peer_id();
        let auth = MessageAuth::new(keypair.clone());
        Self {
            keypair,
            peer_id,
            auth,
            shard,
        }
    }

    fn establish_session<M: SwarmModel + Clone>(
        &self,
        coord: &mut SwarmRoundCoordinator<M>,
        epoch: u64,
        nonce_seed: u8,
        now_secs: u32,
    ) -> [u8; 32] {
        let challenge = SessionHandshakeChallengeV2 {
            swarm_instance_id: coord.swarm_instance_id(),
            run_id: *coord.run_id().as_bytes(),
            policy_digest: coord.context_policy_digest(),
            receiver_peer: coord.coordinator_peer_id(),
            sender_peer: self.peer_id,
            requested_epoch: epoch,
            initial_sequence: 0,
            receiver_nonce: [nonce_seed; 32],
            issued_at_secs: now_secs,
        };
        let session_id = challenge.derive_session_id().unwrap();
        coord.register_session_challenge_v2(challenge).unwrap();
        let sig = self.auth.sign_session_handshake_v2(&challenge).unwrap();
        coord
            .complete_session_handshake_v2(self.keypair.public_key(), &sig, now_secs)
            .unwrap();
        session_id
    }

    #[allow(clippy::too_many_arguments)]
    fn build_dense_delivery<M: SwarmModel + Clone>(
        &self,
        coord: &SwarmRoundCoordinator<M>,
        session_id: [u8; 32],
        epoch: u64,
        seq: u64,
        ts_secs: u32,
        round_id: u64,
        base_digest: CheckpointContentDigestV1,
        values: &[f32],
        reported_examples: u64,
    ) -> Vec<u8> {
        let mut raw = Vec::with_capacity(values.len() * 4);
        for &v in values {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let inner = InnerUpdateV2 {
            originator_public_key: *self.keypair.public_key(),
            swarm_instance_id: coord.swarm_instance_id(),
            run_id: *coord.run_id().as_bytes(),
            round_id,
            contribution_id: seq,
            base_checkpoint_digest: *base_digest.as_bytes(),
            context_policy_digest: coord.context_policy_digest(),
            payload_kind: PayloadKindV1::ModelDeltaV1,
            compression_schema_version: COMPRESSION_SCHEMA_DENSE_F32_LE_V1,
            num_elements: values.len() as u32,
            reported_examples,
            encoded_payload: raw,
            originator_signature: Vec::new(),
        };
        BundledGossipDeliveryV2::new_signed(
            &self.auth,
            inner,
            coord.coordinator_peer_id(),
            session_id,
            epoch,
            seq,
            ts_secs,
        )
        .to_bytes()
        .unwrap()
    }

    /// Perform multi-step local SGD with `FedProx` regularization ($\mu > 0$) and return
    /// $\Delta_i = w_i^{\text{local}} - w_{\text{parent}}$.
    fn train_local_delta(
        &self,
        parent_model: &LinearModel,
        steps: usize,
        lr: f32,
        fedprox_mu: f32,
    ) -> Vec<f32> {
        let mut local_model = parent_model.clone();
        let parent_params = parent_model.parameters().to_vec();
        let fedprox = FedProxRegularizer::new(fedprox_mu).unwrap();
        let mut opt = SgdOptimizer::<LinearModel>::new(lr).unwrap();

        for _ in 0..steps {
            let mut batch_grad = [0.0f32; 3];
            for (x, y) in &self.shard {
                let pred = local_model.forward(x).unwrap();
                let mut grad_y = [0.0f32; 1];
                let _ = MseLoss::compute_and_grad(&pred, y, &mut grad_y).unwrap();
                let mut sample_grad = [0.0f32; 3];
                local_model
                    .backward_into(x, &grad_y, &mut sample_grad)
                    .unwrap();
                for (bg, sg) in batch_grad.iter_mut().zip(sample_grad.iter()) {
                    *bg += *sg / (self.shard.len() as f32);
                }
            }
            let _ = fedprox
                .apply_to_gradients(local_model.parameters(), &parent_params, &mut batch_grad)
                .unwrap();
            opt.step(&mut local_model, &batch_grad).unwrap();
        }

        local_model
            .parameters()
            .iter()
            .zip(parent_params.iter())
            .map(|(&w_local, &w_parent)| ((w_local as f64) - (w_parent as f64)) as f32)
            .collect()
    }
}

struct ValidationMseEvaluator {
    eval_samples: Vec<([f32; 2], [f32; 1])>,
}

impl FitnessFunction for ValidationMseEvaluator {
    type Model = LinearModel;
    type Error = String;

    fn evaluate(&self, model: &Self::Model) -> Result<f32, Self::Error> {
        let mut total = 0.0f64;
        for (x, y) in &self.eval_samples {
            let pred = model.forward(x).map_err(|e| e.to_string())?;
            let mut grad_y = [0.0f32; 1];
            let l = MseLoss::compute_and_grad(&pred, y, &mut grad_y).map_err(|e| e.to_string())?;
            total += l as f64;
        }
        Ok((total / (self.eval_samples.len() as f64)) as f32)
    }
}

fn synthetic_shards_7_peers() -> (Vec<PeerNode>, ValidationMseEvaluator) {
    // True linear relationship: y = 1.5 * x0 - 0.8 * x1 + 0.4
    let eval_samples = vec![
        ([1.0, 0.0], [1.9]),
        ([0.0, 1.0], [-0.4]),
        ([2.0, 1.0], [2.6]),
        ([-1.0, 2.0], [-2.7]),
        ([1.5, -1.0], [3.45]),
        ([-0.5, -0.5], [0.05]),
    ];
    let mut peers = Vec::new();
    for i in 0..7u8 {
        let offset = (i as f32 - 3.0) * 0.02;
        let shard = eval_samples
            .iter()
            .map(|(x, y)| (*x, [y[0] + offset]))
            .collect();
        peers.push(PeerNode::new(i + 1, shard));
    }
    (peers, ValidationMseEvaluator { eval_samples })
}

#[test]
fn scenario_1_honest_model_delta_fedprox_converges_and_strict_report_verifies() {
    let base = temp_dir("s1_honest_fedprox");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let (peers, evaluator) = synthetic_shards_7_peers();
    let cohort: BTreeSet<PeerId> = peers.iter().map(|p| p.peer_id).collect();
    let run_id = RunId::from_bytes([0xA1; 16]);
    let mut session = DataOpsSession::create(
        &base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();

    let config = RoundCoordinatorConfig {
        update_kind: UpdateKindV1::ModelDeltaV1 {
            server_learning_rate: 1.0,
        },
        aggregation: RobustAggregation::FedAvg,
        weighting_policy: WeightingPolicyV1::EqualPeerVote,
        authorized_cohort: cohort,
        min_quorum: 7,
        max_staleness: 0,
        max_clock_skew_secs: 120,
    };

    let mut init_model = LinearModel::new(2, 1).unwrap();
    init_model.load_parameters(&[0.0, 0.0, 0.0]).unwrap();
    let initial_mse = evaluator.evaluate(&init_model).unwrap();

    let rt = MockRuntime::with_deterministic_scheduler(64);
    let mut coord =
        SwarmRoundCoordinator::initialize(run_id, config, init_model, &mut session, 1_000_000)
            .unwrap();
    let sessions: Vec<[u8; 32]> = peers
        .iter()
        .enumerate()
        .map(|(i, p)| p.establish_session(&mut coord, 1, 0x10 + i as u8, 100))
        .collect();

    let mut last_mse = initial_mse;
    for r in 1..=10u64 {
        rt.advance(Duration::from_secs(1));
        let now_secs = (rt.now() / 1000) + 100;
        let parent_digest = coord.parent_digest();
        for (i, peer) in peers.iter().enumerate() {
            let delta = peer.train_local_delta(coord.model(), 4, 0.08, 0.05);
            let bytes = peer.build_dense_delivery(
                &coord,
                sessions[i],
                1,
                r,
                now_secs as u32,
                r,
                parent_digest,
                &delta,
                peer.shard.len() as u64,
            );
            coord
                .submit_verified_delivery(&bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, now_secs)
                .unwrap();
        }
        let summary = coord
            .finalize_round::<SgdOptimizer<LinearModel>, ValidationMseEvaluator>(
                &[],
                None,
                Some(&evaluator),
                &mut session,
                (r + 1) * 1_000_000,
            )
            .unwrap();
        let mse = summary.fitness_score.unwrap();
        assert!(
            mse <= last_mse + 1e-5,
            "round {r} MSE {mse} should decrease from {last_mse}"
        );
        last_mse = mse;
    }

    assert!(
        last_mse < initial_mse * 0.25,
        "10 rounds of honest FedProx training must reduce MSE by >75%: initial={initial_mse}, final={last_mse}"
    );

    let bundle_signer = KeyPair::from_seed([0x99; 32]).unwrap();
    let run_dir = session.sink().bundle().run_dir().to_path_buf();
    let receipt = session.commit_and_close(Some(&bundle_signer)).unwrap();
    assert!(receipt.manifest.signature.is_some());

    let policy = TrustPolicy::Strict {
        trusted_public_keys: vec![*bundle_signer.public_key()],
        expected_run_id: Some(run_id),
        min_signed_unix_nanos: None,
    };
    let (_report, warnings) = load_report_strict(&run_dir, &policy).unwrap();
    assert!(
        warnings.is_empty(),
        "strict signed bundle must load with zero warnings: {:?}",
        warnings
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn scenario_2_and_3_byzantine_bulyan_and_multikrum_defeat_outliers_and_weight_spoofing() {
    let base = temp_dir("s2_s3_byzantine");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let (peers, evaluator) = synthetic_shards_7_peers();
    let cohort: BTreeSet<PeerId> = peers.iter().map(|p| p.peer_id).collect();

    // Run Bulyan (n = 7, f = 1 >= 4f + 3 = 7) with 5 honest peers, 1 extreme sign-flipping + weight-spoofing
    // Byzantine peer (peers[5]), and 1 NaN-injecting Byzantine peer (peers[6]).
    let run_id = RunId::from_bytes([0xB2; 16]);
    let mut session = DataOpsSession::create(
        &base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();

    let bulyan_cfg = RoundCoordinatorConfig {
        update_kind: UpdateKindV1::ModelDeltaV1 {
            server_learning_rate: 1.0,
        },
        aggregation: RobustAggregation::Bulyan {
            max_byzantine: 1,
            profile: BulyanProfile::IterativeKrumV1,
        },
        weighting_policy: WeightingPolicyV1::EqualPeerVote,
        authorized_cohort: cohort.clone(),
        min_quorum: 7,
        max_staleness: 0,
        max_clock_skew_secs: 120,
    };

    let mut init_model = LinearModel::new(2, 1).unwrap();
    init_model.load_parameters(&[0.0, 0.0, 0.0]).unwrap();
    let initial_mse = evaluator.evaluate(&init_model).unwrap();

    let mut coord = SwarmRoundCoordinator::initialize(
        run_id,
        bulyan_cfg,
        init_model.clone(),
        &mut session,
        1_000,
    )
    .unwrap();
    let sessions: Vec<[u8; 32]> = peers
        .iter()
        .enumerate()
        .map(|(i, p)| p.establish_session(&mut coord, 1, 0x20 + i as u8, 100))
        .collect();

    for r in 1..=8u64 {
        let parent_digest = coord.parent_digest();
        // 6 honest peers (0..6) and 1 extreme Byzantine peer (6) that also spoofs reported_examples = 10_000_000_000!
        // First verify that if peer 6 sends NaN, it is rejected at Stage A without consuming sequence r:
        let nan_bytes = peers[6].build_dense_delivery(
            &coord,
            sessions[6],
            1,
            r,
            (100 + r) as u32,
            r,
            parent_digest,
            &[0.1, f32::NAN, 0.1],
            10_000_000_000,
        );
        let nan_err = coord
            .submit_verified_delivery(&nan_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100 + r)
            .unwrap_err();
        assert!(matches!(nan_err, RoundError::NonFiniteInput { .. }));

        for (i, peer) in peers[..6].iter().enumerate() {
            let delta = peer.train_local_delta(coord.model(), 4, 0.08, 0.05);
            let bytes = peer.build_dense_delivery(
                &coord,
                sessions[i],
                1,
                r,
                (100 + r) as u32,
                r,
                parent_digest,
                &delta,
                6,
            );
            coord
                .submit_verified_delivery(&bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100 + r)
                .unwrap();
        }

        // Byzantine peer 6 submits an extreme sign-flipped poison vector [-500.0, 500.0, -500.0]
        // reusing sequence r (not consumed by Stage A NaN rejection!) and claiming 10B examples:
        let poison_bytes = peers[6].build_dense_delivery(
            &coord,
            sessions[6],
            1,
            r,
            (100 + r) as u32,
            r,
            parent_digest,
            &[-500.0, 500.0, -500.0],
            10_000_000_000,
        );
        coord
            .submit_verified_delivery(&poison_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100 + r)
            .unwrap();

        let _ = coord
            .finalize_round::<SgdOptimizer<LinearModel>, ValidationMseEvaluator>(
                &[],
                None,
                Some(&evaluator),
                &mut session,
                (r + 1) * 1_000,
            )
            .unwrap();
    }

    let final_bulyan_mse = evaluator.evaluate(coord.model()).unwrap();
    assert!(
        final_bulyan_mse < initial_mse * 0.30,
        "Bulyan + EqualPeerVote must resist extreme Byzantine poison + weight spoofing: initial={initial_mse}, final={final_bulyan_mse}"
    );

    // Also verify Multi-Krum (n = 7, f = 1, m = 4) converges under the same attack:
    let run_id_mk = RunId::from_bytes([0xB3; 16]);
    let mut session_mk = DataOpsSession::create(
        &base,
        run_id_mk,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();
    let mk_cfg = RoundCoordinatorConfig {
        update_kind: UpdateKindV1::default(),
        aggregation: RobustAggregation::MultiKrum {
            num_byzantine: 1,
            num_selected: 4,
        },
        weighting_policy: WeightingPolicyV1::EqualPeerVote,
        authorized_cohort: cohort,
        min_quorum: 7,
        max_staleness: 0,
        max_clock_skew_secs: 120,
    };
    let mut coord_mk =
        SwarmRoundCoordinator::initialize(run_id_mk, mk_cfg, init_model, &mut session_mk, 1_000)
            .unwrap();
    let sessions_mk: Vec<[u8; 32]> = peers
        .iter()
        .enumerate()
        .map(|(i, p)| p.establish_session(&mut coord_mk, 1, 0x30 + i as u8, 100))
        .collect();
    for r in 1..=8u64 {
        let parent_digest = coord_mk.parent_digest();
        for (i, peer) in peers[..6].iter().enumerate() {
            let delta = peer.train_local_delta(coord_mk.model(), 4, 0.08, 0.05);
            let bytes = peer.build_dense_delivery(
                &coord_mk,
                sessions_mk[i],
                1,
                r,
                (100 + r) as u32,
                r,
                parent_digest,
                &delta,
                6,
            );
            coord_mk
                .submit_verified_delivery(&bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100 + r)
                .unwrap();
        }
        let poison_bytes = peers[6].build_dense_delivery(
            &coord_mk,
            sessions_mk[6],
            1,
            r,
            (100 + r) as u32,
            r,
            parent_digest,
            &[-500.0, 500.0, -500.0],
            10_000_000_000,
        );
        coord_mk
            .submit_verified_delivery(&poison_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 100 + r)
            .unwrap();
        coord_mk
            .finalize_round::<SgdOptimizer<LinearModel>, ValidationMseEvaluator>(
                &[],
                None,
                None,
                &mut session_mk,
                (r + 1) * 1_000,
            )
            .unwrap();
    }
    let final_mk_mse = evaluator.evaluate(coord_mk.model()).unwrap();
    assert!(
        (final_bulyan_mse - final_mk_mse).abs() < 0.05,
        "Bulyan ({final_bulyan_mse}) and Multi-Krum ({final_mk_mse}) should converge within 0.05 MSE"
    );

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn scenario_4_and_5_gossip_relay_forwarding_classic_residual_and_equivocation_defense() {
    let base = temp_dir("s4_s5_gossip_residual");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let (peers, _) = synthetic_shards_7_peers();
    let coord_kp = KeyPair::from_seed([0xEE; 32]).unwrap();
    let coord_peer = coord_kp.peer_id();

    // Build a Ring topology [peer0, relay_peer1, coord_peer] so peer0 must route through relay_peer1!
    let ring_peers = vec![peers[0].peer_id, peers[1].peer_id, coord_peer];
    let hub = MockNetworkHub::new(ring_peers, Topology::Ring, 64);
    let t_orig = hub.peer_transport(peers[0].peer_id);
    let t_relay = hub.peer_transport(peers[1].peer_id);
    let t_coord = hub.peer_transport(coord_peer);

    let run_id = RunId::from_bytes([0xC4; 16]);
    let mut session = DataOpsSession::create(
        &base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();
    let cohort: BTreeSet<PeerId> = [peers[0].peer_id, peers[1].peer_id].into_iter().collect();
    let config = RoundCoordinatorConfig {
        update_kind: UpdateKindV1::default(),
        aggregation: RobustAggregation::FedAvg,
        weighting_policy: WeightingPolicyV1::EqualPeerVote,
        authorized_cohort: cohort.clone(),
        min_quorum: 2,
        max_staleness: 0,
        max_clock_skew_secs: 120,
    };
    let mut model = LinearModel::new(2, 1).unwrap();
    model.load_parameters(&[0.5, -0.5, 0.1]).unwrap();
    let mut coord = SwarmRoundCoordinator::initialize(run_id, config, model, &mut session, 1_000)
        .unwrap()
        .with_coordinator_identity(*run_id.as_bytes(), coord_peer);
    let session0 = peers[0].establish_session(&mut coord, 1, 0x44, 200);

    // Originator (peer0) compresses its local delta ONCE using ClassicResidualCompressor (TopK 2/3)
    // and fans out the same signed InnerUpdateV2 through relay_peer1 without double-mutating residual!
    let mut compressor =
        ClassicResidualCompressor::new(3, CompressionMethod::TopK { k_ratio: 0.5 }).unwrap();
    let raw_delta = [0.6f32, -0.4f32, 0.05f32];
    let compressed = compressor.compress_step(&raw_delta).unwrap();
    let residual_snapshot = compressor.residual().to_vec();
    // Residual must hold the dropped smallest coordinate (0.05) while the larger 2 coordinates were sent:
    assert!((residual_snapshot[2] - 0.05).abs() < 1e-6);

    let encoded_payload = encode_compressed_update_payload_v2(&compressed).unwrap();
    let inner1 = InnerUpdateV2 {
        originator_public_key: *peers[0].keypair.public_key(),
        swarm_instance_id: coord.swarm_instance_id(),
        run_id: *run_id.as_bytes(),
        round_id: 1,
        contribution_id: 1,
        base_checkpoint_digest: *coord.parent_digest().as_bytes(),
        context_policy_digest: coord.context_policy_digest(),
        payload_kind: PayloadKindV1::ModelDeltaV1,
        compression_schema_version: COMPRESSION_SCHEMA_COMPRESSED_GRADIENT_V1,
        num_elements: 3,
        reported_examples: 6,
        encoded_payload: encoded_payload.clone(),
        originator_signature: Vec::new(),
    };
    let delivery1 = BundledGossipDeliveryV2::new_signed(
        &peers[0].auth,
        inner1,
        coord_peer,
        session0,
        1,
        1,
        200,
    );
    let wire_bytes1 = delivery1.to_bytes().unwrap();

    // Hop 1: peer0 -> relay_peer1 (residual state on peer0 is NOT touched again on re-hop/fanout)
    futures_poll_ready(t_orig.send(peers[1].peer_id, &wire_bytes1)).unwrap();
    assert_eq!(compressor.residual(), &residual_snapshot[..]);

    // Hop 2: relay_peer1 receives via observed link hint (NOT a trusted PeerId) and forwards unchanged to coord_peer
    let hop1_msg = futures_poll_ready(t_relay.recv()).unwrap();
    assert!(matches!(
        hop1_msg.observed_source,
        ObservedSourceRef::LinkLabel(_)
    ));
    futures_poll_ready(t_relay.send(coord_peer, &hop1_msg.payload)).unwrap();

    // Coordinator receives over untrusted relay link, cryptographically derives peer0 from InnerUpdateV2,
    // decompresses, validates Stage A/B, commits Stage C, and admits!
    let hop2_msg = futures_poll_ready(t_coord.recv()).unwrap();
    assert!(matches!(
        hop2_msg.observed_source,
        ObservedSourceRef::LinkLabel(_)
    ));
    assert_eq!(
        coord
            .submit_verified_delivery(&hop2_msg.payload, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 200)
            .unwrap(),
        SubmissionDecision::Admitted {
            admitted_count: 1,
            min_quorum: 2
        }
    );

    // Exact wire packet replay MUST fail with ReplayError::Replay!
    assert!(matches!(
        coord.submit_verified_delivery(&hop2_msg.payload, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 200),
        Err(RoundError::ReplayRejected(ReplayError::Replay {
            seq: 1,
            ..
        }))
    ));

    // Identical round content re-sent by originator under fresh delivery_seq = 2 -> DuplicateIdempotentAck!
    let inner2 = InnerUpdateV2 {
        originator_public_key: *peers[0].keypair.public_key(),
        swarm_instance_id: coord.swarm_instance_id(),
        run_id: *run_id.as_bytes(),
        round_id: 1,
        contribution_id: 2,
        base_checkpoint_digest: *coord.parent_digest().as_bytes(),
        context_policy_digest: coord.context_policy_digest(),
        payload_kind: PayloadKindV1::ModelDeltaV1,
        compression_schema_version: COMPRESSION_SCHEMA_COMPRESSED_GRADIENT_V1,
        num_elements: 3,
        reported_examples: 6,
        encoded_payload,
        originator_signature: Vec::new(),
    };
    let wire_bytes2 = BundledGossipDeliveryV2::new_signed(
        &peers[0].auth,
        inner2,
        coord_peer,
        session0,
        1,
        2,
        200,
    )
    .to_bytes()
    .unwrap();
    futures_poll_ready(t_orig.send(peers[1].peer_id, &wire_bytes2)).unwrap();
    let hop1_dup = futures_poll_ready(t_relay.recv()).unwrap();
    futures_poll_ready(t_relay.send(coord_peer, &hop1_dup.payload)).unwrap();
    let hop2_dup = futures_poll_ready(t_coord.recv()).unwrap();
    assert_eq!(
        coord
            .submit_verified_delivery(&hop2_dup.payload, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 200)
            .unwrap(),
        SubmissionDecision::DuplicateIdempotentAck
    );

    // Equivocation under fresh delivery_seq = 3 revokes peer0's admitted vote!
    let conflict_bytes = peers[0].build_dense_delivery(
        &coord,
        session0,
        1,
        3,
        200,
        1,
        coord.parent_digest(),
        &[-9.0, -9.0, -9.0],
        6,
    );
    assert!(matches!(
        coord.submit_verified_delivery(&conflict_bytes, DEFAULT_MAX_GOSSIP_DELIVERY_BYTES, 200),
        Err(RoundError::EquivocationRejected { .. })
    ));
    assert_eq!(coord.admitted_count(), 0);

    let _ = fs::remove_dir_all(&base);
}

#[test]
fn scenario_6_cas_cache_hit_bypass_tamper_rejection_and_post_seal_fence() {
    let base = temp_dir("s6_cas_and_seal");
    let _ = fs::remove_dir_all(&base);
    fs::create_dir_all(&base).unwrap();

    let run_id = RunId::from_bytes([0xD6; 16]);
    let mut session = DataOpsSession::create(
        &base,
        run_id,
        ArtifactWriteProfile::default(),
        LockAcquisitionPolicy::FailFast,
    )
    .unwrap();

    let mut model = LinearModel::new(2, 1).unwrap();
    model.load_parameters(&[1.0, 2.0, 0.5]).unwrap();
    let (digest, cas_uri, _) =
        put_checkpoint_model(session.sink().bundle().run_dir(), &model).unwrap();

    let graph = GraphV1 {
        schema_version: 1,
        graph_id: None,
        nodes: vec![NodeV1 {
            node_key: "stage_cached".to_string(),
            node_id: None,
            op_kind: OpKind::Data,
            op_type: "passthrough".to_string(),
            inputs: vec![],
            outputs: vec![AssetRefV1 {
                asset_key: "dataset://ns/stage_out".to_string(),
                fingerprint: None,
            }],
            params: CanonParams::new(),
            code_ref: Some("swarmtorch@0.1.0".to_string()),
            unsafe_surface: false,
            execution_trust: ExecutionTrust::Core,
            node_def_hash: None,
            execution_hint: None,
            cache_policy: Some(CachePolicyV0::PreferReuse),
            materialization_policy: None,
            resources: None,
            op_hash: None,
        }],
        edges: vec![],
    };

    let test_runner = NativeOpRunner::with_mode(RunnerExecutionModeV1::MetadataOnlyForTests);

    // First execution: Miss, then bind physical CAS blob
    let r1 = execute_graph_sequential(
        &graph,
        &mut session,
        &test_runner,
        &PermissivePolicy,
        run_id,
        || 1_000,
    )
    .unwrap();
    assert!(r1.cached_nodes.is_empty());
    session
        .bind_output_cas_uri(
            "dataset://ns/stage_out",
            &cas_uri,
            "application/octet-stream",
        )
        .unwrap();

    // Second execution with intact CAS blob: Hit!
    let r2 = execute_graph_sequential(
        &graph,
        &mut session,
        &test_runner,
        &PermissivePolicy,
        run_id,
        || 2_000,
    )
    .unwrap();
    assert_eq!(r2.cached_nodes.len(), 1);

    // Corrupt 1 byte inside the CAS blob -> verify_cache_hit & scheduler MUST return an explicit integrity failure!
    let blob_path = session
        .sink()
        .bundle()
        .run_dir()
        .join("blobs")
        .join("sha256")
        .join(digest.to_hex());
    let mut corrupted = fs::read(&blob_path).unwrap();
    corrupted[0] ^= 0xFF;
    fs::write(&blob_path, &corrupted).unwrap();

    let cache_err = session
        .verify_cache_hit(
            &graph.nodes[0],
            &[swarm_torch_core::dataops::OutputSpecCore {
                asset_key: "dataset://ns/stage_out".to_string(),
                schema: None,
            }],
        )
        .unwrap_err();
    assert_eq!(cache_err.kind(), std::io::ErrorKind::InvalidData);

    let r3 = execute_graph_sequential(
        &graph,
        &mut session,
        &test_runner,
        &PermissivePolicy,
        run_id,
        || 3_000,
    )
    .unwrap();
    assert!(r3.cached_nodes.is_empty());
    assert!(
        r3.executed_nodes.is_empty(),
        "corrupted CAS blob must not silently re-execute cached node"
    );
    assert_eq!(r3.failed_nodes, vec!["stage_cached"]);

    // Restore valid CAS blob before sealing and signing the bundle
    fs::remove_file(&blob_path).unwrap();
    let _ = put_checkpoint_model(session.sink().bundle().run_dir(), &model).unwrap();

    let signer = KeyPair::from_seed([0x77; 32]).unwrap();
    let cloned_sink = Arc::clone(session.sink());
    let run_dir = session.sink().bundle().run_dir().to_path_buf();
    let _ = session.commit_and_close(Some(&signer)).unwrap();

    // Cloned Arc<RunArtifactSink> must reject post-seal writes with PermissionDenied!
    let err = cloned_sink
        .emit_event(&EventRecord {
            schema_version: 1,
            ts_unix_nanos: 4_000,
            trace_id: TraceId::from_bytes([1; 16]),
            span_id: None,
            name: "post_seal_attempt".to_string(),
            attrs: AttrMap::new(),
        })
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);

    // Strict verification succeeds before tampering, and fails closed after 1-byte tampering of graphs/graph.json!
    let policy = TrustPolicy::Strict {
        trusted_public_keys: vec![*signer.public_key()],
        expected_run_id: Some(run_id),
        min_signed_unix_nanos: None,
    };
    assert!(load_report_strict(&run_dir, &policy).is_ok());

    let graph_path = run_dir.join("graph.json");
    let mut g_bytes = fs::read(&graph_path).unwrap();
    g_bytes.push(b'\n');
    fs::write(&graph_path, &g_bytes).unwrap();
    assert!(
        load_report_strict(&run_dir, &policy).is_err(),
        "TrustPolicy::Strict must reject tampered artifact file"
    );

    let _ = fs::remove_dir_all(&base);
}

fn futures_poll_ready<F: core::future::Future>(fut: F) -> F::Output {
    struct NoopWake;
    impl std::task::Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }
    let waker = std::task::Waker::from(Arc::new(NoopWake));
    let mut cx = std::task::Context::from_waker(&waker);
    let pinned = core::pin::pin!(fut);
    match pinned.poll(&mut cx) {
        core::task::Poll::Ready(out) => out,
        core::task::Poll::Pending => panic!("Ready future returned Pending"),
    }
}
