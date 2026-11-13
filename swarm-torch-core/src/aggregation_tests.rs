use super::*;
use crate::compression::CompressionMethod;
use crate::traits::UpdateTransform;
use alloc::vec;

fn update(gradients: Vec<f32>) -> GradientUpdate {
    GradientUpdate {
        sender: [0u8; 32],
        sequence: 0,
        gradients,
        round_id: 0,
    }
}

fn update_with_meta(
    sender_byte: u8,
    sequence: u64,
    round_id: u64,
    gradients: Vec<f32>,
) -> GradientUpdate {
    GradientUpdate {
        sender: [sender_byte; 32],
        sequence,
        gradients,
        round_id,
    }
}

struct AddOneTransform;

impl UpdateTransform for AddOneTransform {
    fn transform(&self, mut update: GradientUpdate) -> GradientUpdate {
        for g in &mut update.gradients {
            *g += 1.0;
        }
        update
    }

    fn name(&self) -> &str {
        "add_one"
    }

    fn is_core_trusted(&self) -> bool {
        true
    }
}

struct UntrustedClipTransform;

impl UpdateTransform for UntrustedClipTransform {
    fn transform(&self, mut update: GradientUpdate) -> GradientUpdate {
        for g in &mut update.gradients {
            *g = g.clamp(-1.0, 1.0);
        }
        update
    }

    fn name(&self) -> &str {
        "clip"
    }
}

#[test]
fn fedavg_rejects_mismatched_gradient_dimensions() {
    let updates = vec![update(vec![1.0, 2.0]), update(vec![3.0])];
    let result = FedAvg.aggregate(&updates);
    assert!(matches!(result, Err(crate::Error::InvalidGradient)));
}

#[test]
fn trimmed_mean_rejects_mismatched_gradient_dimensions() {
    let updates = vec![update(vec![1.0, 2.0]), update(vec![3.0])];
    let result = TrimmedMean::new(0.2).aggregate(&updates);
    assert!(matches!(result, Err(crate::Error::InvalidGradient)));
}

#[test]
fn coordinate_median_rejects_empty_gradient_vectors() {
    let updates = vec![update(vec![]), update(vec![])];
    let result = CoordinateMedian.aggregate(&updates);
    assert!(matches!(result, Err(crate::Error::InvalidGradient)));
}

#[test]
fn aggregation_rejects_oversized_gradient_dim() {
    let oversized = vec![0.0; 10_000_001];
    let updates = vec![update(oversized)];
    let result = FedAvg.aggregate(&updates);
    assert!(matches!(result, Err(crate::Error::InvalidGradient)));
}

#[test]
#[cfg(feature = "krum")]
fn krum_rejects_oversized_peer_set() {
    let mut updates = Vec::new();
    for _ in 0..51 {
        updates.push(update(vec![1.0; 2]));
    }
    let krum = Krum::new(0);
    let result = krum.aggregate(&updates);
    assert!(matches!(result, Err(crate::Error::ResourceExhausted)));
}

#[test]
fn fedavg_sum_then_divide_precision() {
    let n = 100;
    let dim = 4;
    let updates: Vec<GradientUpdate> = (0..n)
        .map(|i| {
            let base = (i as f32) * 0.001 + 1e-7;
            update(vec![base; dim])
        })
        .collect();

    let result = FedAvg.aggregate(&updates).unwrap();

    for (d, &actual_f32) in result.iter().enumerate() {
        let oracle: f64 = updates.iter().map(|u| u.gradients[d] as f64).sum::<f64>() / (n as f64);
        let actual = actual_f32 as f64;
        let rel_err = if oracle.abs() > 1e-15 {
            (actual - oracle).abs() / oracle.abs()
        } else {
            (actual - oracle).abs()
        };
        assert!(
            rel_err < 1e-6,
            "FedAvg precision drift: dim={d}, oracle={oracle}, actual={actual}, rel_err={rel_err}"
        );
    }
}

#[test]
fn trimmed_mean_new_clamps_ratio() {
    let tm = TrimmedMean::new(0.6);
    assert!(
        (tm.trim_ratio - 0.49).abs() < f32::EPSILON,
        "ratio above 0.49 should clamp"
    );
    assert_eq!(
        TrimmedMean::try_new(0.5).unwrap_err(),
        crate::Error::InvalidThreshold
    );
    assert_eq!(
        TrimmedMean::try_new(-0.1).unwrap_err(),
        crate::Error::InvalidThreshold
    );
    assert_eq!(
        TrimmedMean::try_new(f32::NAN).unwrap_err(),
        crate::Error::InvalidThreshold
    );
}

#[test]
fn trimmed_mean_direct_construction_insufficient_updates() {
    let tm = TrimmedMean { trim_ratio: 0.6 };
    let updates = vec![update(vec![1.0, 2.0]), update(vec![3.0, 4.0])];
    let result = tm.aggregate(&updates);
    assert!(
        matches!(result, Err(crate::Error::InsufficientUpdates)),
        "guard should return InsufficientUpdates, got {result:?}"
    );
}

#[test]
fn aggregators_reject_nan_and_inf_explicitly() {
    let nan_updates = vec![
        update_with_meta(1, 0, 1, vec![1.0, 2.0]),
        update_with_meta(2, 0, 1, vec![1.0, f32::NAN]),
        update_with_meta(3, 0, 1, vec![1.0, 2.0]),
        update_with_meta(4, 0, 1, vec![1.0, 2.0]),
        update_with_meta(5, 0, 1, vec![1.0, 2.0]),
        update_with_meta(6, 0, 1, vec![1.0, 2.0]),
        update_with_meta(7, 0, 1, vec![1.0, 2.0]),
    ];
    let inf_updates = vec![
        update_with_meta(1, 0, 1, vec![1.0, 2.0]),
        update_with_meta(2, 0, 1, vec![f32::INFINITY, 2.0]),
        update_with_meta(3, 0, 1, vec![1.0, 2.0]),
        update_with_meta(4, 0, 1, vec![1.0, 2.0]),
        update_with_meta(5, 0, 1, vec![1.0, 2.0]),
        update_with_meta(6, 0, 1, vec![1.0, 2.0]),
        update_with_meta(7, 0, 1, vec![1.0, 2.0]),
    ];

    assert_eq!(
        FedAvg.aggregate(&nan_updates),
        Err(crate::Error::NanDetected)
    );
    assert_eq!(
        FedAvg.aggregate(&inf_updates),
        Err(crate::Error::InfDetected)
    );
    assert_eq!(
        CoordinateMedian.aggregate(&nan_updates),
        Err(crate::Error::NanDetected)
    );
    assert_eq!(
        CoordinateMedian.aggregate(&inf_updates),
        Err(crate::Error::InfDetected)
    );
    assert_eq!(
        TrimmedMean::new(0.2).aggregate(&nan_updates),
        Err(crate::Error::NanDetected)
    );
    assert_eq!(
        TrimmedMean::new(0.2).aggregate(&inf_updates),
        Err(crate::Error::InfDetected)
    );

    #[cfg(feature = "krum")]
    {
        assert_eq!(
            Krum::new(1).aggregate(&nan_updates),
            Err(crate::Error::NanDetected)
        );
        assert_eq!(
            Krum::new(1).aggregate(&inf_updates),
            Err(crate::Error::InfDetected)
        );
    }

    #[cfg(feature = "bulyan")]
    {
        assert_eq!(
            Bulyan::new(1).aggregate(&nan_updates),
            Err(crate::Error::NanDetected)
        );
        assert_eq!(
            Bulyan::new(1).aggregate(&inf_updates),
            Err(crate::Error::InfDetected)
        );
    }
}

#[test]
fn coordinate_median_even_midpoint_does_not_overflow_f32_max() {
    let updates = vec![
        update_with_meta(1, 0, 1, vec![f32::MAX, -f32::MAX]),
        update_with_meta(2, 0, 1, vec![f32::MAX, -f32::MAX]),
    ];
    let median = CoordinateMedian.aggregate(&updates).unwrap();
    assert_eq!(median, vec![f32::MAX, -f32::MAX]);
}

#[test]
#[cfg(feature = "krum")]
fn multi_krum_one_pass_selects_and_averages_top_m_candidates() {
    // 6 peers, f = 1 => k = n - f - 2 = 3.
    // Honest cluster at 1.0, 2.0, 3.0, 4.0; outlier at 100.0; intermediate at 10.0.
    let updates = vec![
        update_with_meta(1, 0, 1, vec![1.0]),
        update_with_meta(2, 0, 1, vec![2.0]),
        update_with_meta(3, 0, 1, vec![3.0]),
        update_with_meta(4, 0, 1, vec![4.0]),
        update_with_meta(5, 0, 1, vec![10.0]),
        update_with_meta(6, 0, 1, vec![100.0]),
    ];
    // Krum scores (sum of 3 closest sq distances):
    // 1.0 -> 1 + 4 + 9 = 14
    // 2.0 -> 1 + 1 + 4 = 6
    // 3.0 -> 1 + 1 + 4 = 6
    // 4.0 -> 1 + 4 + 9 = 14
    // Top m = 2 are 2.0 and 3.0 => mean = 2.5.
    let mk = Krum::try_multi(1, 2).unwrap();
    let res = mk.aggregate(&updates).unwrap();
    assert!((res[0] - 2.5).abs() < 1e-6);

    // Invalid m > n - f - 2 (m = 4 > 3) must return InvalidThreshold.
    let invalid_mk = Krum::multi(1, 4);
    assert_eq!(
        invalid_mk.aggregate(&updates),
        Err(crate::Error::InvalidThreshold)
    );
}

#[test]
#[cfg(feature = "bulyan")]
fn bulyan_stage2_nearest_to_median_golden_fixture() {
    // ADR-0019 / A3 golden fixture:
    // Selected scalar updates: [-100.0, -10.0, 0.0, 1.0, 2.0], f = 1 (theta = 5, beta = 3).
    // Median is 0.0; closest 3 values to 0.0 are [0.0, 1.0, 2.0]; mean = 1.0.
    let u0 = [-100.0f32];
    let u1 = [-10.0f32];
    let u2 = [0.0f32];
    let u3 = [1.0f32];
    let u4 = [2.0f32];
    let selected: [&[f32]; 5] = [&u0, &u1, &u2, &u3, &u4];
    let out = bulyan_stage2_nearest_to_median(&selected, 1).unwrap();
    assert_eq!(out, vec![1.0f32]);
}

#[test]
#[cfg(feature = "bulyan")]
fn bulyan_iterative_krum_v1_golden_selection_and_output_fixture() {
    // ADR-0019 / A3 golden selection-semantics fixture:
    // Scalar updates [0, 1, 2, 3, 4, 5, 6] with n = 7, f = 1 (theta = 3, beta = 1).
    // Stage 1 selects original indices [2, 4, 1] in selection order; Stage 2 outputs 2.0.
    let updates: Vec<GradientUpdate> = (0..7)
        .map(|i| update_with_meta(i as u8, 0, 1, vec![i as f32]))
        .collect();

    let bulyan = Bulyan::try_new(1, BulyanProfile::IterativeKrumV1).unwrap();
    let selected_indices = bulyan.select_indices(&updates).unwrap();
    assert_eq!(selected_indices, vec![2, 4, 1]);

    let aggregated = bulyan.aggregate(&updates).unwrap();
    assert_eq!(aggregated, vec![2.0f32]);
}

#[test]
#[cfg(feature = "bulyan")]
fn bulyan_enforces_preconditions_and_budget() {
    assert!(matches!(
        Bulyan::try_new(0, BulyanProfile::IterativeKrumV1),
        Err(crate::Error::InvalidThreshold)
    ));

    // n = 6 < 4f + 3 = 7 for f = 1
    let too_few: Vec<GradientUpdate> = (0..6)
        .map(|i| update_with_meta(i as u8, 0, 1, vec![i as f32]))
        .collect();
    assert_eq!(
        Bulyan::new(1).aggregate(&too_few),
        Err(crate::Error::InsufficientPeers)
    );

    // Budget enforcement
    let seven: Vec<GradientUpdate> = (0..7)
        .map(|i| update_with_meta(i as u8, 0, 1, vec![1.0, 2.0, 3.0, 4.0]))
        .collect();
    let tight_budget = AggregationBudget {
        max_peers: 7,
        max_dimension: 4,
        max_scratch_bytes: 16, // 7 * 7 * 8 = 392 > 16
        max_work_ops: 1000,
    };
    assert_eq!(
        Bulyan::new(1).with_budget(tight_budget).aggregate(&seven),
        Err(crate::Error::ResourceBudgetExceeded)
    );
}

#[test]
fn update_transform_preserves_sender_sequence_round_id() {
    let updates = vec![
        update_with_meta(1, 11, 101, vec![0.5, 1.5]),
        update_with_meta(2, 12, 101, vec![2.5, 3.5]),
    ];
    let (transformed, _) = apply_update_transforms(updates.clone(), &AddOneTransform);

    for (input, output) in updates.iter().zip(transformed.iter()) {
        assert_eq!(input.sender, output.sender);
        assert_eq!(input.sequence, output.sequence);
        assert_eq!(input.round_id, output.round_id);
    }
    assert_eq!(transformed[0].gradients, vec![1.5, 2.5]);
}

#[test]
fn untrusted_transform_audit_has_core_trusted_false() {
    let updates = vec![update_with_meta(7, 1, 9, vec![2.0, -4.0])];
    let (_, audit) = apply_update_transforms(updates, &UntrustedClipTransform);
    assert_eq!(audit.transform_name, "clip");
    assert!(!audit.core_trusted);
    assert_eq!(audit.round_id, 9);
}

#[test]
fn core_trusted_transform_audit_has_core_trusted_true() {
    let updates = vec![update_with_meta(7, 1, 3, vec![1.0])];
    let (_, audit) = apply_update_transforms(updates, &AddOneTransform);
    assert_eq!(audit.transform_name, "add_one");
    assert!(audit.core_trusted);
    assert_eq!(audit.round_id, 3);
}

#[test]
fn apply_transforms_then_aggregate_trimmed_mean_end_to_end() {
    let updates = vec![
        update_with_meta(1, 1, 4, vec![1.0, 2.0]),
        update_with_meta(2, 2, 4, vec![3.0, 4.0]),
    ];
    let (transformed, _) = apply_update_transforms(updates, &AddOneTransform);
    let aggregated = TrimmedMean::new(0.0).aggregate(&transformed).unwrap();
    assert_eq!(aggregated, vec![3.0, 4.0]);
}

#[test]
fn pipeline_trimmed_mean_no_compression_matches_direct() {
    let updates = vec![update(vec![1.0, 2.0]), update(vec![3.0, 4.0])];
    let direct = TrimmedMean::new(0.0).aggregate(&updates).unwrap();
    let piped = AggregationPipeline::new(TrimmedMean::new(0.0))
        .run(&updates)
        .unwrap();
    assert_eq!(piped, direct);
}

#[test]
fn pipeline_fedavg_topk_compression_roundtrip() {
    let updates = vec![update(vec![1.0, 2.0, 3.0]), update(vec![5.0, 6.0, 7.0])];
    let direct = FedAvg.aggregate(&updates).unwrap();
    let piped = AggregationPipeline::new(FedAvg)
        .with_compression(CompressionMethod::TopK { k_ratio: 1.0 })
        .run(&updates)
        .unwrap();
    assert_eq!(piped, direct);
}

#[test]
fn pipeline_returns_error_on_empty_updates() {
    let updates: Vec<GradientUpdate> = Vec::new();
    let err = AggregationPipeline::new(FedAvg).run(&updates).unwrap_err();
    assert!(matches!(
        err,
        AggregationPipelineError::Aggregation(crate::Error::InsufficientUpdates)
    ));
}

#[test]
fn pipeline_trace_nonzero_estimates_for_valid_inputs() {
    let trace = AggregationPipeline::new(FedAvg).trace(1024, 10);
    assert!(trace.estimated_memory_bytes > 0);
    assert!(trace.estimated_flops > 0);
}

#[test]
fn pipeline_trace_topk_compressed_less_than_raw() {
    let trace = AggregationPipeline::new(FedAvg)
        .with_compression(CompressionMethod::TopK { k_ratio: 0.1 })
        .trace(1000, 8);
    let compressed = trace.estimated_compressed_bytes.unwrap();
    assert!(compressed < trace.estimated_memory_bytes);
}

#[test]
fn pipeline_trace_memory_lower_bound_holds() {
    let gradient_len = 257usize;
    let num_peers = 19usize;
    let trace = AggregationPipeline::new(FedAvg).trace(gradient_len, num_peers);
    let lower_bound = (gradient_len as u64) * (num_peers as u64) * 4;
    assert!(trace.estimated_memory_bytes >= lower_bound);
}

#[test]
fn pipeline_trace_zero_peers_zero_estimates() {
    let trace = AggregationPipeline::new(FedAvg)
        .with_compression(CompressionMethod::TopK { k_ratio: 0.2 })
        .trace(0, 0);
    assert_eq!(trace.estimated_memory_bytes, 0);
    assert_eq!(trace.estimated_flops, 0);
    assert_eq!(trace.estimated_compressed_bytes, Some(0));
}

#[test]
fn centered_clipping_validates_hyperparameters_and_handles_zero_norm_and_outliers() {
    assert_eq!(
        CenteredClipping::try_new(0.0, 1, Vec::new()).unwrap_err(),
        Error::InvalidThreshold
    );
    assert_eq!(
        CenteredClipping::try_new(-1.0, 1, Vec::new()).unwrap_err(),
        Error::InvalidThreshold
    );
    assert_eq!(
        CenteredClipping::try_new(f32::NAN, 1, Vec::new()).unwrap_err(),
        Error::NanDetected
    );
    assert_eq!(
        CenteredClipping::try_new(f32::INFINITY, 1, Vec::new()).unwrap_err(),
        Error::InfDetected
    );
    assert_eq!(
        CenteredClipping::try_new(1.0, 0, Vec::new()).unwrap_err(),
        Error::InvalidThreshold
    );
    assert_eq!(
        CenteredClipping::try_new(1.0, MAX_CCLIP_ITERATIONS + 1, Vec::new()).unwrap_err(),
        Error::InvalidThreshold
    );

    // Zero-norm difference (all updates equal center) must never divide by zero.
    let center = vec![2.5f32, -1.5, 4.0];
    let updates = vec![
        update_with_meta(1, 0, 1, center.clone()),
        update_with_meta(2, 0, 1, center.clone()),
        update_with_meta(3, 0, 1, center.clone()),
    ];
    let cclip = CenteredClipping::try_new(0.5, 4, center.clone()).unwrap();
    let out = cclip.aggregate(&updates).unwrap();
    assert_eq!(out, center);

    // Byzantine outlier resistance around previous round center:
    // 4 honest peers at [1.0, 1.0], 1 Byzantine peer at [10_000.0, 10_000.0], center at [0.9, 0.9], tau = 0.25.
    let prev_center = vec![0.9f32, 0.9];
    let cohort = vec![
        update_with_meta(1, 0, 1, vec![1.0, 1.0]),
        update_with_meta(2, 0, 1, vec![1.0, 1.0]),
        update_with_meta(3, 0, 1, vec![1.0, 1.0]),
        update_with_meta(4, 0, 1, vec![1.0, 1.0]),
        update_with_meta(9, 0, 1, vec![10_000.0, 10_000.0]),
    ];
    let cclip_byz = CenteredClipping::try_new(0.25, 3, prev_center.clone()).unwrap();
    let res = cclip_byz
        .aggregate_with_center(&cohort, &prev_center)
        .unwrap();
    // Honest points are within distance sqrt(0.02) ~ 0.1414 <= tau=0.25; the Byzantine point is clipped to radius 0.25.
    assert!((res[0] - 1.0).abs() < 0.2);
    assert!((res[1] - 1.0).abs() < 0.2);
}

#[test]
fn s_bucketing_enforces_uniform_weights_arrival_order_invariance_and_conservative_quorum() {
    let seed = [0x42u8; 32];
    // 20 peers (17 honest, 3 Byzantine), bucket_size s = 2 -> B = 10 buckets, f_B = min(3, 10) = 3.
    let mut updates = Vec::new();
    for peer_byte in 1..=20u8 {
        let mut g = if peer_byte <= 17 {
            vec![1.0f32, 2.0, 3.0]
        } else {
            vec![500.0f32, -500.0, 500.0]
        };
        g[0] += (peer_byte as f32) * 0.001;
        updates.push(update_with_meta(peer_byte, peer_byte as u64, 1, g));
    }

    let cfg = BucketingConfig {
        bucket_size: 2,
        byzantine_count: 3,
        round_seed: seed,
        remainder_policy: BucketRemainderPolicy::StrictExactMultiple,
    };

    // Reversing network arrival order must produce the exact same bucket means and provenance.
    let mut reversed = updates.clone();
    reversed.reverse();
    let (b_fwd, rep_fwd) = bucket_updates_v1(&updates, &cfg).unwrap();
    let (b_rev, rep_rev) = bucket_updates_v1(&reversed, &cfg).unwrap();
    assert_eq!(rep_fwd, rep_rev);
    for (a, b) in b_fwd.iter().zip(b_rev.iter()) {
        assert_eq!(a.sender, b.sender);
        assert_eq!(a.gradients, b.gradients);
        assert_eq!(a.round_id, 1);
    }

    // Verify uniform 1/s weighting: in any bucket containing 1 honest (~1.0) and 1 Byzantine (500.0) peer,
    // the bucket mean is the exact unweighted average (~250.5).
    for b_upd in &b_fwd {
        assert!(b_upd.gradients[0] < 300.0 || (b_upd.gradients[0] - 500.0).abs() < 1.0);
    }

    // Conservative f_B = min(f, B) = 3 quorum check:
    // B = 10 satisfies Krum (B >= 2*3 + 3 = 9), so Krum succeeds:
    let budget = AggregationBudget::default_pairwise();
    let (krum_out, _) =
        aggregate_with_bucketing_v1(&updates, &cfg, PostBucketingRule::Krum, None, &budget)
            .unwrap();
    assert!((krum_out[0] - 1.0).abs() < 0.05);

    // B = 10 violates Bulyan (requires B >= 4*3 + 3 = 15), so Bulyan MUST fail closed with InsufficientPeers!
    let bulyan_err =
        aggregate_with_bucketing_v1(&updates, &cfg, PostBucketingRule::Bulyan, None, &budget)
            .unwrap_err();
    assert_eq!(bulyan_err, Error::InsufficientPeers);
}

#[test]
fn shared_seed_sparse_subspace_aggregation_matches_full_space_and_verifies_merkle_openings() {
    use crate::compression::derive_verifiable_round_seed_v1;
    let d = 128usize;
    let k = 16usize;
    let run_id = [0xAAu8; 16];
    let beacon = [0x55u8; 32];

    // 7 peers (f = 1 for Krum: n = 7 >= 2*1 + 3 = 5).
    let mut pre_seed_vectors = Vec::new();
    let mut peer_commitments = Vec::new();
    for p in 1..=7u8 {
        let vec_p: Vec<f32> = if p <= 6 {
            (0..d).map(|idx| (idx as f32) * 0.1 + 1.0).collect()
        } else {
            // Peer 7 is Byzantine in its pre-seed vector.
            (0..d).map(|_| -999.0f32).collect()
        };
        let comm = PreSeedVectorCommitmentV1::commit(&vec_p).unwrap();
        let peer_id = PeerId::new([p; 32]);
        pre_seed_vectors.push((peer_id, vec_p));
        peer_commitments.push((peer_id, comm));
    }

    let round_seed =
        derive_verifiable_round_seed_v1(&run_id, 1, &peer_commitments, &beacon).unwrap();
    let mut verifiable_submissions = Vec::new();
    for ((peer_id, vec_p), (_, comm)) in pre_seed_vectors.iter().zip(peer_commitments.iter()) {
        let (payload, openings) = comm
            .open_for_seed(
                vec_p,
                k,
                round_seed,
                crate::compression::SharedSeedScalingProfile::UnscaledProjectionV1,
            )
            .unwrap();
        verifiable_submissions.push((*peer_id, *comm, payload, openings));
    }

    let krum = Krum::new(1);
    let budget = AggregationBudget::default_pairwise();
    let agg = aggregate_verifiable_shared_seed_sparse_v1(
        &verifiable_submissions,
        &round_seed,
        &krum,
        &budget,
    )
    .unwrap();
    assert_eq!(agg.len(), d);

    // Check that all k sampled coordinates match the honest vector and non-sampled coordinates are 0.0.
    let sampled = sample_shared_indices_v1(d, k, &round_seed).unwrap();
    for (idx, &val) in agg.iter().enumerate() {
        if sampled.binary_search(&(idx as u32)).is_ok() {
            let expected = (idx as f32) * 0.1 + 1.0;
            assert!((val - expected).abs() < 1e-5);
        } else {
            assert_eq!(val, 0.0);
        }
    }

    // Tampering with a disclosed coordinate value after seed disclosure must fail VerificationFailed!
    let mut tampered = verifiable_submissions.clone();
    tampered[0].2.values[0] += 1.0;
    assert_eq!(
        aggregate_verifiable_shared_seed_sparse_v1(&tampered, &round_seed, &krum, &budget)
            .unwrap_err(),
        Error::VerificationFailed
    );
}

#[test]
fn test_fedasync_staleness_policy_and_drift_compensated_mixing() {
    let policy = FedAsyncStalenessDescriptorV1 {
        base_alpha: 0.6,
        max_staleness_rounds: 5,
        polynomial_decay_exponent: 0.5,
    };

    // Zero staleness -> base_alpha
    let alpha0 = policy.compute_mixing_alpha(10, 10).unwrap();
    assert!((alpha0 - 0.6).abs() < 1e-6);

    // Staleness tau = 3 -> 0.6 * (1 + 3)^(-0.5) = 0.6 * 0.5 = 0.3
    let alpha3 = policy.compute_mixing_alpha(7, 10).unwrap();
    assert!((alpha3 - 0.3).abs() < 1e-6);

    // Hinge threshold tau_hinge = 2 at tau = 3 -> excess = 1 -> 0.6 * (1 + 1)^(-0.5) = 0.6 / sqrt(2)
    let alpha_hinge = policy.compute_hinge_mixing_alpha(7, 10, 2).unwrap();
    let expected_hinge = 0.6f32 / core::f32::consts::SQRT_2;
    assert!((alpha_hinge - expected_hinge).abs() < 1e-6);

    // Future round or exceeding max_staleness_rounds fails closed
    assert_eq!(
        policy.compute_mixing_alpha(11, 10).unwrap_err(),
        Error::InvalidThreshold
    );
    assert_eq!(
        policy.compute_mixing_alpha(4, 10).unwrap_err(),
        Error::InvalidThreshold
    );

    // Drift-compensated mixing:
    // Suppose w_base (at round 7) = [10.0, 20.0], w_current (at round 10) = [14.0, 28.0].
    // Client trained from w_base and produced local weights w_local = [12.0, 24.0] => stale_delta = [2.0, 4.0].
    // Exact FedAsync convex combination at alpha(3) = 0.3:
    // w_new = (1 - 0.3) * w_current + 0.3 * (w_base + stale_delta)
    //       = 0.7 * [14.0, 28.0] + 0.3 * [12.0, 24.0] = [13.4, 26.8].
    // Notice naive uncompensated w_current + 0.3 * stale_delta would give [14.6, 29.2] (diverging!).
    let w_base = [10.0f32, 20.0f32];
    let w_current = [14.0f32, 28.0f32];
    let stale_delta = [2.0f32, 4.0f32];

    let outcome =
        apply_fedasync_stale_delta_v1(&policy, 7, 10, &w_current, &w_base, &stale_delta, None)
            .unwrap();
    assert_eq!(outcome.staleness_rounds, 3);
    assert!(!outcome.was_clipped);
    assert!((outcome.updated_weights[0] - 13.4).abs() < 1e-5);
    assert!((outcome.updated_weights[1] - 26.8).abs() < 1e-5);

    // With pre-drift L2 norm clipping: ||[3.0, 4.0]||_2 = 5.0, clipped to tau = 2.5 => [1.5, 2.0].
    let byz_delta = [3.0f32, 4.0f32];
    let clipped =
        apply_fedasync_stale_delta_v1(&policy, 7, 10, &w_current, &w_base, &byz_delta, Some(2.5))
            .unwrap();
    assert!(clipped.was_clipped);
    assert!((clipped.raw_delta_l2_norm - 5.0).abs() < 1e-9);
    assert!((clipped.clipped_delta_l2_norm - 2.5).abs() < 1e-9);
    // w_base + d_clipped = [11.5, 22.0], 0.7 * [14, 28] + 0.3 * [11.5, 22.0] = [13.25, 26.2]
    assert!((clipped.updated_weights[0] - 13.25).abs() < 1e-5);
    assert!((clipped.updated_weights[1] - 26.2).abs() < 1e-5);
}
