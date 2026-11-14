//! End-to-End Byzantine-Resilient Federated Round Example (`W10-03`, `W11-01`..`W11-04`).
//!
//! Demonstrates:
//! 1. Verifiable `SharedSeedSparse` compression with pre-seed binary SHA-256 Merkle vector
//!    commitments (`PreSeedVectorCommitmentV1`) and coordinate opening proofs (`W11-02`).
//! 2. Multi-chunk UDP gossip transport (`UdpGossipTransport` + `ChunkedFrameCodec` +
//!    `ObservedSourceReassembler`, `W11-03`).
//! 3. Conservative `$s$-Bucketing` + `CenteredClipping` (`CClip`) Byzantine-resilient aggregation (`W11-01`).
//! 4. `SwarmTorchCheckpointV1` SafeTensors v1 CAS checkpoint export and cryptographic verification (`W11-04`).

use swarm_torch::artifacts::{
    get_verified_safetensors_checkpoint_v1, put_safetensors_checkpoint_v1,
};
use swarm_torch::core::aggregation::{
    aggregate_verifiable_shared_seed_sparse_v1, aggregate_with_bucketing_v1, AggregationBudget,
    BucketRemainderPolicy, BucketingConfig, CenteredClipping, PostBucketingRule,
};
use swarm_torch::core::compression::{
    compute_commitments_root_v1, derive_context_bound_round_seed_v1, SharedSeedBandwidthReportV1,
    SharedSeedCommitmentMessageV1, SharedSeedOpeningMessageV1, SharedSeedResidualCompressor,
    SharedSeedScalingProfile, SparseRoundContextV1, SHARED_SEED_SAMPLER_VERSION_V1,
};
use swarm_torch::core::traits::{GradientUpdate, PeerId, SwarmModel};
use swarm_torch::models::safetensors_v1::{ModelTensorSchemaV1, SafeTensorsLimitsV1};
use swarm_torch::models::simple::LinearModel;
use swarm_torch::net::chunked::ReassemblerLimits;
use swarm_torch::net::udp::{InMemoryUdpDatagramHub, UdpGossipTransport};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("SwarmTorch Wave 11 Byzantine Federated Round Example");
    println!("====================================================");

    let dim = 10usize; // 4x2 LinearModel: 8 weights + 2 biases
    let top_k = 4usize;
    let round_id = 1u64;
    let run_id = [0x11u8; 16];

    // 1. Simulate 9 honest peers + 2 Byzantine sign-flipping peers using Verifiable SharedSeedSparse (W11-02 / ST-12)
    let total_peers = 11usize;
    let mut compressors: Vec<SharedSeedResidualCompressor> = (0..total_peers)
        .map(|_| SharedSeedResidualCompressor::new(dim, top_k).unwrap())
        .collect();

    let mut pre_seed_vectors = Vec::with_capacity(total_peers);
    let mut commitments = Vec::with_capacity(total_peers);
    let mut dense_updates = Vec::with_capacity(total_peers);

    for (i, comp) in compressors.iter().enumerate() {
        let sender = [(i as u8) + 1; 32];
        let raw_grad: Vec<f32> = if i < 9 {
            // Honest peer gradient around -0.20 (bucket norm ~0.60 > tau=0.5 so 3 honest buckets outweigh 2 clipped Byzantine buckets)
            (0..dim)
                .map(|j| -0.20 + (j as f32) * 0.002 + (i as f32) * 0.001)
                .collect()
        } else {
            // Byzantine peer attempting sign-flip + amplification attack
            vec![8.5f32; dim]
        };
        dense_updates.push(GradientUpdate {
            sender,
            sequence: 1,
            gradients: raw_grad.clone(),
            round_id,
        });
        let (u, commitment) = comp.prepare_pre_seed_vector(&raw_grad)?;
        pre_seed_vectors.push(u);
        commitments.push((PeerId::new(sender), commitment));
    }

    // Derive deterministic post-commitment round seed bound to cohort context + commitments_root
    let roster = commitments.iter().map(|(p, _)| *p).collect();
    let sparse_ctx = SparseRoundContextV1 {
        swarm_instance_id: [0x22; 16],
        run_id,
        round_id,
        base_checkpoint_digest: [0x33; 32],
        frozen_roster: roster,
        original_len: dim as u32,
        sparsity_k: top_k as u32,
        scaling: SharedSeedScalingProfile::UnscaledProjectionV1,
        sampler_version: SHARED_SEED_SAMPLER_VERSION_V1,
    };
    let ctx_digest = sparse_ctx.context_digest()?;
    let commitments_root = compute_commitments_root_v1(&ctx_digest, &commitments)?;
    let round_seed =
        derive_context_bound_round_seed_v1(&ctx_digest, &commitments_root, &[0x55; 32]);

    let mut verifiable_submissions = Vec::with_capacity(total_peers);
    let mut bw_report = SharedSeedBandwidthReportV1::new(dim as u32, top_k as u32);
    for (i, comp) in compressors.iter_mut().enumerate() {
        let (peer_id, commitment) = commitments[i];
        let (payload, proof_batch) =
            comp.finalize_with_seed(&pre_seed_vectors[i], &commitment, round_seed)?;
        let phase1_bytes = SharedSeedCommitmentMessageV1 {
            context_digest: ctx_digest,
            commitment,
        }
        .to_bytes()?;
        let phase2_bytes = SharedSeedOpeningMessageV1 {
            context_digest: ctx_digest,
            commitments_root,
            shared_round_seed: round_seed,
            payload: payload.clone(),
            openings: proof_batch.clone(),
        }
        .to_bytes()?;
        bw_report.record_peer_delivery(phase1_bytes.len(), phase2_bytes.len(), &proof_batch);
        verifiable_submissions.push((peer_id, commitment, payload, proof_batch));
    }

    let cclip = CenteredClipping::try_new(0.5, 3, Vec::new())?;
    let budget = AggregationBudget::default();
    let sparse_aggregated = aggregate_verifiable_shared_seed_sparse_v1(
        &verifiable_submissions,
        &round_seed,
        &cclip,
        &budget,
    )?;
    println!(
        "[W11-02 / ST-12] Verifiable SharedSeedSparse + CClip aggregated {} non-zero coordinates (values={}B, merkle_proofs={}B, total={}B vs dense={}B)",
        sparse_aggregated.iter().filter(|&&v| v != 0.0).count(),
        bw_report.projected_values_bytes_total,
        bw_report.merkle_proof_bytes_total,
        bw_report.total_two_phase_wire_bytes,
        bw_report.raw_dense_bytes_total,
    );

    // 2. Transmit a multi-chunk payload over UdpGossipTransport + ChunkedFrameCodec (W11-03)
    let hub = InMemoryUdpDatagramHub::new();
    let sender_addr = "127.0.0.1:19101".parse()?;
    let coord_addr = "127.0.0.1:19100".parse()?;
    let limits = ReassemblerLimits {
        mtu: 80, // 16-byte header + 64-byte payload stride to exercise multi-chunk framing
        ..ReassemblerLimits::default()
    };

    let sender_udp = UdpGossipTransport::bind_in_memory(
        PeerId::new([1u8; 32]),
        sender_addr,
        1,
        limits,
        hub.clone(),
    )?;
    let coord_udp =
        UdpGossipTransport::bind_in_memory(PeerId::new([99u8; 32]), coord_addr, 1, limits, hub)?;

    let wire_payload: Vec<u8> = dense_updates[0]
        .gradients
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let transfer_id = sender_udp.send_to_addr(coord_addr, &wire_payload)?;
    let (observed_from, reassembled) = coord_udp
        .poll_reassembled_once()?
        .expect("reassembled payload ready");
    assert_eq!(observed_from, sender_addr);
    assert_eq!(reassembled, wire_payload);
    println!(
        "[W11-03] UdpGossipTransport reassembled {} bytes (transfer_id={})",
        reassembled.len(),
        transfer_id
    );

    // 3. Run conservative s-Bucketing (5 buckets, 2 Byzantines < 5/2) + CenteredClipping (W11-01)
    let bucket_cfg = BucketingConfig {
        bucket_size: 2,
        byzantine_count: 2,
        round_seed,
        remainder_policy: BucketRemainderPolicy::DropRemainderDeterministic,
    };
    let post_rule = PostBucketingRule::CenteredClipping {
        tau: 0.5,
        max_iterations: 3,
    };
    let (bucketed_grad, report) =
        aggregate_with_bucketing_v1(&dense_updates, &bucket_cfg, post_rule, None, &budget)?;
    println!(
        "[W11-01] s-Bucketing ({} buckets, dropped_remainder={}) + CClip suppressed 2 Byzantine peers: grad[0]={:.4}",
        report.buckets.len(),
        report.dropped_remainder_peers.len(),
        bucketed_grad[0]
    );
    assert!(
        bucketed_grad[0] < 0.0,
        "honest negative gradient direction preserved"
    );

    // 4. Apply update to LinearModel and persist SwarmTorchCheckpointV1 SafeTensors to CAS (W11-04)
    let mut global_model = LinearModel::new(4, 2)?.with_random_init(2026);
    for (p, &g) in global_model
        .parameters_mut()
        .iter_mut()
        .zip(bucketed_grad.iter())
    {
        *p -= g;
    }

    let temp_dir =
        std::env::temp_dir().join(format!("swarmtorch_wave11_example_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir)?;

    let schema = ModelTensorSchemaV1::for_linear(4, 2)?;
    let st_limits = SafeTensorsLimitsV1::default();
    let (digest, cas_uri, size_bytes) =
        put_safetensors_checkpoint_v1(&temp_dir, &global_model, &schema, &st_limits)?;
    let restored = get_verified_safetensors_checkpoint_v1(
        &temp_dir,
        LinearModel::new(4, 2)?,
        &digest,
        &schema,
        &st_limits,
    )?;
    assert_eq!(restored.parameters(), global_model.parameters());
    let _ = std::fs::remove_dir_all(&temp_dir);

    println!(
        "[W11-04] SafeTensors v1 CAS checkpoint committed & verified: {} ({} bytes)",
        cas_uri, size_bytes
    );

    Ok(())
}
