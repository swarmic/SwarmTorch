use super::*;

#[test]
fn exact_gaussian_rdp_and_composition_match_analytical_closed_form() {
    let profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 2.0, // sigma = 2.0 => rho(alpha) = alpha / 8
        target_delta: 1e-5,
        max_epsilon: 6.0,
        conversion_formula: DpConversionFormulaV1::Mironov2017,
        classification: PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
    };
    let mut acc = RdpAccountantV1::new(profile).unwrap();
    assert!(!acc.guarantees_differential_privacy());

    // Record 4 full-participation rounds (q = 1.0): cumulative rho(alpha) = 4 * alpha / 8 = alpha / 2.
    for _ in 0..4 {
        acc.record_clipped_round(1.0, 1.0).unwrap();
    }
    assert_eq!(acc.committed_rounds(), 4);
    let orders = acc.cumulative_rdp_orders();
    for (i, &alpha) in RDP_ALPHA_GRID_V1.iter().enumerate() {
        assert!(abs_f64(orders[i] - 0.5 * alpha) < 1e-12);
    }

    // At alpha = 6.0: rho(6) = 3.0, ln(1e5)/5 = 11.512925465 / 5 = 2.302585093 => eps = 5.302585093
    let (eps, delta, opt_alpha) = acc.current_approx_dp();
    assert_eq!(delta, 1e-5);
    assert_eq!(opt_alpha, 6.0);
    assert!(abs_f64(eps - (3.0 + 11.512_925_464_970_229 / 5.0)) < 1e-9);
}

#[test]
fn subsampled_gaussian_amplification_and_budget_exhaustion_gate() {
    let profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 1.5,
        target_delta: 1e-5,
        max_epsilon: 2.5,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
    };
    let mut acc = RdpAccountantV1::new(profile).unwrap();

    // Subsampling at q = 0.05 yields strictly smaller RDP than q = 1.0
    let sgm_sched = compute_round_rdp_schedule_v1(0.05, 1.5).unwrap();
    let full_sched = compute_round_rdp_schedule_v1(1.0, 1.5).unwrap();
    for i in 0..RDP_GRID_LEN {
        assert!(
            sgm_sched[i] < full_sched[i],
            "order {}: sgm {} >= full {}",
            RDP_ALPHA_GRID_V1[i],
            sgm_sched[i],
            full_sched[i]
        );
    }

    // Passing a clipping radius larger than calibrated profile.clipping_radius_tau fails closed
    assert_eq!(
        acc.record_clipped_round(1.5, 0.05).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // Repeated subsampled rounds succeed under amplification, then eventually hit PrivacyBudgetExhausted
    acc.record_clipped_round(1.0, 0.05).unwrap();
    let rounds_before = acc.committed_rounds();
    assert_eq!(rounds_before, 1);
    let mut hit_exhausted = false;
    for _ in 0..40 {
        if let Err(err) = acc.record_clipped_round(1.0, 0.25) {
            assert_eq!(err, CryptoError::PrivacyBudgetExhausted);
            hit_exhausted = true;
            break;
        }
    }
    assert!(hit_exhausted);
    assert!(acc.committed_rounds() >= rounds_before);
    let (eps, _delta, _alpha) = acc.current_approx_dp();
    assert!(eps <= profile.max_epsilon);
}

#[test]
fn certified_dp_classification_fails_closed_on_standalone_accountant() {
    let certified_attempt = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 1.5,
        target_delta: 1e-5,
        max_epsilon: 2.5,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
    };
    assert_eq!(
        RdpAccountantV1::new(certified_attempt).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        RdpAccountantV1::restore_from_orders(certified_attempt, [0.0; RDP_GRID_LEN], 0)
            .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // Even a certified accountant rejects the unsealed PrivacyAccountantHook::record_clipped_round!
    let mut certified_acc = RdpAccountantV1::new_verified_mechanism(certified_attempt).unwrap();
    assert!(certified_acc.guarantees_differential_privacy());
    assert_eq!(
        certified_acc.record_clipped_round(1.0, 0.5).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(certified_acc.committed_rounds(), 0);

    // ExampleLevel fails closed for certified ClientLevel DpFedAvgV1.
    let example_level = DpMechanismProfileV1 {
        unit: DpUnitV1::ExampleLevel,
        ..certified_attempt
    };
    assert_eq!(
        RdpAccountantV1::new_verified_mechanism(example_level).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        RdpAccountantV1::restore_verified_mechanism_from_orders(
            example_level,
            [0.0; RDP_GRID_LEN],
            0,
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
}

#[test]
fn restore_from_orders_rejects_inconsistent_round_count_and_handles_round_overflow() {
    let profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 1.5,
        target_delta: 1e-5,
        max_epsilon: 10.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
    };
    let mut nonzero = [0.0; RDP_GRID_LEN];
    nonzero[0] = 0.25;
    assert_eq!(
        RdpAccountantV1::restore_from_orders(profile, nonzero, 0).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        RdpAccountantV1::restore_from_orders(profile, [0.0; RDP_GRID_LEN], 1).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    let valid_sched = compute_round_rdp_schedule_v1(0.05, 1.5).unwrap();
    let mut maxed = RdpAccountantV1::restore_from_orders(profile, valid_sched, u64::MAX)
        .expect("finite schedule with non-zero count restores cleanly");
    assert_eq!(
        maxed.record_clipped_round(1.0, 0.05).unwrap_err(),
        CryptoError::PrivacyBudgetExhausted
    );
    assert_eq!(maxed.committed_rounds(), u64::MAX);
}

#[test]
fn restore_from_orders_and_preview_reject_non_finite_negative_and_budget_exceeded() {
    let profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 1.5,
        target_delta: 1e-5,
        max_epsilon: 0.5,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
    };
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01] {
        let mut bad_orders = [0.0; RDP_GRID_LEN];
        bad_orders[3] = bad;
        assert_eq!(
            RdpAccountantV1::restore_from_orders(profile, bad_orders, 1).unwrap_err(),
            CryptoError::InvalidPrivacyParameter
        );
    }

    // Restoring orders whose converted epsilon exceeds max_epsilon fails closed.
    let high_sched = [1.0; RDP_GRID_LEN];
    assert_eq!(
        RdpAccountantV1::restore_from_orders(profile, high_sched, 1).unwrap_err(),
        CryptoError::PrivacyBudgetExhausted
    );

    let mut acc = RdpAccountantV1::new(DpMechanismProfileV1 {
        max_epsilon: 10.0,
        ..profile
    })
    .unwrap();
    for bad_tau in [f32::NAN, f32::INFINITY, 0.0, -0.5] {
        assert_eq!(
            acc.record_clipped_round(bad_tau, 0.5).unwrap_err(),
            CryptoError::InvalidPrivacyParameter
        );
    }
    for bad_q in [f64::NAN, f64::INFINITY, 0.0, -0.1, 1.01] {
        assert_eq!(
            acc.record_clipped_round(1.0, bad_q).unwrap_err(),
            CryptoError::InvalidPrivacyParameter
        );
    }
    assert_eq!(acc.committed_rounds(), 0);
}

#[test]
fn discrete_gaussian_exact_sampler_matches_analytical_pmf_mean_and_variance() {
    // 1. Validate conservative ceiling compute_discrete_sigma_z_v1 bounds.
    assert_eq!(compute_discrete_sigma_z_v1(1.0).unwrap(), 65_536);
    assert_eq!(compute_discrete_sigma_z_v1(1.5).unwrap(), 98_304);
    assert!(compute_discrete_sigma_z_v1(1.000_01).unwrap() > 65_536);
    assert_eq!(
        compute_discrete_sigma_z_v1(0.0).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        compute_discrete_sigma_z_v1(f32::NAN).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        compute_discrete_sigma_z_v1(300.0).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // 2. Reject invalid parameters on integer samplers.
    let mut drbg = Sha256CounterDrbgV1::new(b"swarmtorch.test.dgauss.err", [0x13; 32], 0, 0);
    assert_eq!(
        sample_bernoulli_exp1_u128(2, 1, &mut drbg).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        sample_bernoulli_exp1_u128(0, 0, &mut drbg).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        sample_dgauss_u128(0, &mut drbg).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        sample_dgauss_u128(DP_MAX_DISCRETE_SIGMA_Z_V1 + 1, &mut drbg).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // 3. Exact PMF & moment conformance for sigma_z = 3 over N = 40_000 draws:
    //    Pr[X = x] = exp(-x^2 / 18) / Z, mean = 0, exact discrete variance ~ 9.0.
    let mut pmf_drbg = Sha256CounterDrbgV1::new(b"swarmtorch.test.dgauss.pmf", [0x42; 32], 3, 0);
    let sigma_z = 3u64;
    let n_draws = 40_000usize;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count_0 = 0usize;
    let mut count_3 = 0usize;
    let mut count_neg3 = 0usize;
    for _ in 0..n_draws {
        let x = sample_dgauss_u128(sigma_z, &mut pmf_drbg).unwrap();
        let xf = x as f64;
        sum += xf;
        sum_sq += xf * xf;
        if x == 0 {
            count_0 += 1;
        } else if x == 3 {
            count_3 += 1;
        } else if x == -3 {
            count_neg3 += 1;
        }
    }
    let mean = sum / (n_draws as f64);
    let var = sum_sq / (n_draws as f64) - mean * mean;
    // Standard error of mean = 3 / sqrt(40_000) = 0.015; allow 6 SE = 0.09.
    assert!(abs_f64(mean) < 0.09, "mean drift: {mean}");
    // Exact variance for N_Z(0, 9) is ~9.0000; SE of sample variance is ~9*sqrt(2/40_000) = 0.064; allow 6 SE.
    assert!(abs_f64(var - 9.0) < 0.40, "variance drift: {var}");
    // Analytical ratio Pr[3] / Pr[0] = exp(-9 / 18) = exp(-0.5) = 0.6065306597.
    let expected_ratio = crate::num::exp_f64_soft(-0.5);
    let emp_ratio_pos = (count_3 as f64) / (count_0 as f64);
    let emp_ratio_neg = (count_neg3 as f64) / (count_0 as f64);
    assert!(
        abs_f64(emp_ratio_pos - expected_ratio) < 0.05,
        "pos ratio {emp_ratio_pos} vs {expected_ratio}"
    );
    assert!(
        abs_f64(emp_ratio_neg - expected_ratio) < 0.05,
        "neg ratio {emp_ratio_neg} vs {expected_ratio}"
    );

    // 4. Verify large realistic lattice scale sigma_z = 98_304 (sigma = 1.5, B = 65_536) executes fast
    //    and matches variance ~ sigma_z^2.
    let mut big_drbg = Sha256CounterDrbgV1::new(b"swarmtorch.test.dgauss.big", [0x77; 32], 1, 0);
    let big_sigma = compute_discrete_sigma_z_v1(1.5).unwrap();
    let big_draws = 4_000usize;
    let mut b_sum = 0.0f64;
    let mut b_sq = 0.0f64;
    for _ in 0..big_draws {
        let z = sample_dgauss_u128(big_sigma, &mut big_drbg).unwrap() as f64;
        let scaled = z / (DP_DISCRETE_LATTICE_RADIUS_V1 as f64);
        b_sum += scaled;
        b_sq += scaled * scaled;
    }
    let b_mean = b_sum / (big_draws as f64);
    let b_var = b_sq / (big_draws as f64) - b_mean * b_mean;
    assert!(abs_f64(b_mean) < 0.15, "scaled mean drift: {b_mean}");
    // Expected variance = 1.5^2 = 2.25; SE ~ 2.25 * sqrt(2/4000) ~ 0.050; allow 6 SE = 0.30.
    assert!(abs_f64(b_var - 2.25) < 0.30, "scaled var drift: {b_var}");
}

#[test]
fn exact_bernoulli_f64_rejects_invalid_and_matches_target_rate() {
    let mut drbg = Sha256CounterDrbgV1::new(b"swarmtorch.test.bern", [0x51; 32], 0, 0);
    for bad_q in [
        0.0,
        -0.5,
        1.0001,
        f64::NAN,
        f64::INFINITY,
        f64::MIN_POSITIVE * 0.5,
    ] {
        assert_eq!(
            sample_bernoulli_f64_exact_v1(bad_q, &mut drbg).unwrap_err(),
            CryptoError::InvalidPrivacyParameter
        );
    }
    assert!(sample_bernoulli_f64_exact_v1(1.0, &mut drbg).unwrap());

    let trials = 20_000usize;
    let mut hits = 0usize;
    for _ in 0..trials {
        if sample_bernoulli_f64_exact_v1(0.25, &mut drbg).unwrap() {
            hits += 1;
        }
    }
    let rate = (hits as f64) / (trials as f64);
    // SE = sqrt(0.25 * 0.75 / 20_000) = 0.00306; allow 6 SE = 0.02.
    assert!(abs_f64(rate - 0.25) < 0.02, "bernoulli rate drift: {rate}");
}

#[cfg(feature = "alloc")]
#[test]
fn integer_lattice_clipping_strictly_enforces_exact_u128_norm_bound() {
    let b_sq = (DP_DISCRETE_LATTICE_RADIUS_V1 as u128) * (DP_DISCRETE_LATTICE_RADIUS_V1 as u128);

    // Rejects empty, non-finite, and invalid tau.
    assert_eq!(
        clip_and_quantize_l2_lattice_v1(&[], 1.0).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        clip_and_quantize_l2_lattice_v1(&[1.0, f32::NAN], 1.0).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        clip_and_quantize_l2_lattice_v1(&[1.0, f32::INFINITY], 1.0).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        clip_and_quantize_l2_lattice_v1(&[1.0, 2.0], 0.0).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // Zero vector & negative zero stay zero with was_clipped = false.
    let (z_vec, z_norm, z_clipped) =
        clip_and_quantize_l2_lattice_v1(&[0.0, -0.0, 0.0], 1.0).unwrap();
    assert_eq!(z_vec, alloc::vec![0, 0, 0]);
    assert_eq!(z_norm, 0.0);
    assert!(!z_clipped);

    // Exact boundary vector [tau, 0] is unclipped and maps to exact [B, 0].
    let (b_vec, b_norm, b_clipped) = clip_and_quantize_l2_lattice_v1(&[2.5, 0.0], 2.5).unwrap();
    assert_eq!(b_vec, alloc::vec![DP_DISCRETE_LATTICE_RADIUS_V1 as i64, 0]);
    assert_eq!(b_norm, 2.5);
    assert!(!b_clipped);

    // Extreme finite inputs (including f32::MAX) are clipped with exact u128 norm_sq <= B^2.
    let cases: [&[f32]; 4] = [
        &[300.0, -400.0],
        &[f32::MAX, -f32::MAX, f32::MAX],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        &[-1e-6, 1e6, -1e6, 0.5],
    ];
    for vec_in in cases {
        let (q, _orig_norm, was_clipped) = clip_and_quantize_l2_lattice_v1(vec_in, 1.5).unwrap();
        assert!(was_clipped);
        let norm_sq: u128 = q
            .iter()
            .map(|&v| {
                let a = v.unsigned_abs() as u128;
                a * a
            })
            .sum();
        assert!(
            norm_sq <= b_sq,
            "exact u128 norm_sq {norm_sq} exceeded B^2 {b_sq}"
        );
    }
}

#[cfg(feature = "alloc")]
#[test]
fn dp_fedavg_mechanism_enforces_receipt_binding_order_invariance_empty_draw_and_fail_closed_checks()
{
    let run_id = [0xA1u8; 16];
    let parent_digest = [0xB2u8; 32];
    let peers: alloc::vec::Vec<PeerId> = (1u8..=4).map(|i| PeerId::new([i; 32])).collect();

    let profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 1.5,
        target_delta: 1e-5,
        max_epsilon: 5.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
    };

    // Duplicate authorized peers & all-zero seeds fail closed.
    assert_eq!(
        SampledDpCohortV1::full_cohort_v1(&[peers[0], peers[0]], run_id, 1).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        SampledDpCohortV1::sample_poisson_with_explicit_seed_v1(&peers, 0.5, [0u8; 32], run_id, 1)
            .unwrap_err(),
        CryptoError::AllZeroSeed
    );
    assert_eq!(
        DpNoiseEntropyV1::from_explicit_seed_for_testing([0u8; 32]).unwrap_err(),
        CryptoError::AllZeroSeed
    );
    // q * N < 1.0 fails closed (0.1 * 4 = 0.4 < 1.0).
    assert_eq!(
        SampledDpCohortV1::sample_poisson_with_explicit_seed_v1(&peers, 0.1, [0x22; 32], run_id, 1)
            .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    let full_cohort = SampledDpCohortV1::full_cohort_v1(&peers, run_id, 1).unwrap();
    let probe_entropy = DpNoiseEntropyV1::from_explicit_seed_for_testing([0x33; 32]).unwrap();
    assert_eq!(
        probe_entropy.provenance(),
        DpEntropyProvenanceV1::ExplicitSeedNonProduction
    );
    assert_ne!(probe_entropy.commitment_sha256(), [0u8; 32]);

    let u1 = GradientUpdate {
        sender: *peers[0].as_bytes(),
        sequence: 1,
        gradients: alloc::vec![10.0, -10.0, 5.0],
        round_id: 1,
    };
    let u2 = GradientUpdate {
        sender: *peers[1].as_bytes(),
        sequence: 1,
        gradients: alloc::vec![-0.2, 0.3, 0.1],
        round_id: 1,
    };

    // Permutation invariance: exact i128 summation guarantees bit-identical output regardless of peer order.
    let out_12 = DpFedAvgMechanismV1::execute_round_v1(
        profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: run_id,
            round_id: 1,
            parent_digest_bytes: parent_digest,
            parameter_dim: 3,
            sampled_cohort: &full_cohort,
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x33; 32]).unwrap(),
            client_updates: &[u1.clone(), u2.clone()],
        },
    )
    .unwrap();
    let out_21 = DpFedAvgMechanismV1::execute_round_v1(
        profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: run_id,
            round_id: 1,
            parent_digest_bytes: parent_digest,
            parameter_dim: 3,
            sampled_cohort: &full_cohort,
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x33; 32]).unwrap(),
            client_updates: &[u2.clone(), u1.clone()],
        },
    )
    .unwrap();
    assert_eq!(out_12.noisy_mean_update, out_21.noisy_mean_update);
    assert_eq!(
        out_12.proof.sanitized_output_digest_sha256(),
        out_21.proof.sanitized_output_digest_sha256()
    );
    assert_eq!(out_12.clipped_count, 1);
    assert_eq!(out_12.participant_count, 2);
    assert_eq!(out_12.fixed_sensitivity_denominator, 4.0);
    assert!(!out_12.proof.is_production_certified_entropy());

    // Duplicate sender inside client_updates fails closed (prevents 2x sensitivity bypass).
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            profile,
            DpFedAvgRoundInputV1 {
                run_id_bytes: run_id,
                round_id: 1,
                parent_digest_bytes: parent_digest,
                parameter_dim: 3,
                sampled_cohort: &full_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x33; 32])
                    .unwrap(),
                client_updates: &[u1.clone(), u1.clone()],
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // Unsampled / unauthorized peer inside client_updates fails closed.
    let rogue = GradientUpdate {
        sender: [0x99; 32],
        sequence: 1,
        gradients: alloc::vec![0.1, 0.2, 0.3],
        round_id: 1,
    };
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            profile,
            DpFedAvgRoundInputV1 {
                run_id_bytes: run_id,
                round_id: 1,
                parent_digest_bytes: parent_digest,
                parameter_dim: 3,
                sampled_cohort: &full_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x33; 32])
                    .unwrap(),
                client_updates: &[rogue],
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // Wrong round_id or run_id against sampled_cohort fails closed.
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            profile,
            DpFedAvgRoundInputV1 {
                run_id_bytes: run_id,
                round_id: 2,
                parent_digest_bytes: parent_digest,
                parameter_dim: 3,
                sampled_cohort: &full_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x33; 32])
                    .unwrap(),
                client_updates: core::slice::from_ref(&u1),
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // Empty client_updates (empty Poisson draw / all dropouts) releases calibrated noise cleanly!
    let empty_out = DpFedAvgMechanismV1::execute_round_v1(
        profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: run_id,
            round_id: 1,
            parent_digest_bytes: parent_digest,
            parameter_dim: 4,
            sampled_cohort: &full_cohort,
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x33; 32]).unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    assert_eq!(empty_out.noisy_mean_update.len(), 4);
    assert_eq!(empty_out.participant_count, 0);
    assert_eq!(empty_out.clipped_count, 0);
    assert!(empty_out.noisy_mean_update.iter().all(|v| v.is_finite()));

    // Certified accountant accepts matching proof and rejects mismatched profile proof.
    let mut cert_acc = RdpAccountantV1::new_verified_mechanism(profile).unwrap();
    cert_acc
        .record_verified_mechanism_round(&out_12.proof)
        .unwrap();
    assert_eq!(cert_acc.committed_rounds(), 1);
    assert!(cert_acc.guarantees_differential_privacy());

    let diff_profile = DpMechanismProfileV1 {
        noise_multiplier: 2.0,
        ..profile
    };
    let mut diff_acc = RdpAccountantV1::new_verified_mechanism(diff_profile).unwrap();
    assert_eq!(
        diff_acc
            .record_verified_mechanism_round(&out_12.proof)
            .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(diff_acc.committed_rounds(), 0);
}

#[cfg(feature = "std")]
#[test]
fn os_csprng_dp_fedavg_mints_production_certified_entropy_proof() {
    let run_id = [0xD1u8; 16];
    let parent_digest = [0xD2u8; 32];
    let peers: Vec<PeerId> = (1u8..=8).map(|i| PeerId::new([i; 32])).collect();
    let profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::ReplaceOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 2.0,
        target_delta: 1e-5,
        max_epsilon: 10.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
    };

    let cohort = SampledDpCohortV1::sample_poisson_os_rng_v1(&peers, 0.5, run_id, 1).unwrap();
    assert_eq!(
        cohort.sampling_provenance(),
        DpSamplingProvenanceV1::OsCsprng
    );
    let entropy = DpNoiseEntropyV1::generate_os_csprng().unwrap();
    assert_eq!(entropy.provenance(), DpEntropyProvenanceV1::OsCsprng);

    let outcome = DpFedAvgMechanismV1::execute_round_v1(
        profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: run_id,
            round_id: 1,
            parent_digest_bytes: parent_digest,
            parameter_dim: 4,
            sampled_cohort: &cohort,
            noise_entropy: entropy,
            client_updates: &[],
        },
    )
    .unwrap();
    assert!(outcome.proof.is_production_certified_entropy());
    assert_eq!(outcome.fixed_sensitivity_denominator, 4.0);
    assert_eq!(outcome.discrete_sigma_z, 131_072);

    // ReplaceOne effective noise multiplier = sigma / 2 = 1.0 matches accountant preview.
    let mut acc = RdpAccountantV1::new_verified_mechanism(profile).unwrap();
    acc.record_verified_mechanism_round(&outcome.proof).unwrap();
    assert_eq!(acc.committed_rounds(), 1);
    let expected_orders = compute_round_rdp_schedule_v1(0.5, 1.0).unwrap();
    for (&actual, &expected) in acc
        .cumulative_rdp_orders()
        .iter()
        .zip(expected_orders.iter())
    {
        assert!(abs_f64(actual - expected) < 1e-12);
    }
}

#[test]
fn compute_discrete_sigma_z_boundary_floats_satisfy_conservative_ceiling() {
    let valid_cases = [
        (f32::from_bits(1), 1u64), // smallest positive f32 subnormal 2^-149
        (f32::MIN_POSITIVE, 1u64), // 2^-126
        (1.0f32 / 65_536.0, 1u64), // exact 2^-16
        (
            f32::from_bits((1.0f32 / 65_536.0).to_bits() + 1),
            2u64, // 2^-16 + 1 ulp -> ceil to 2
        ),
        (1.0 + f32::EPSILON, 65_537u64),
        (
            f32::from_bits(256.0f32.to_bits() - 1),
            DP_MAX_DISCRETE_SIGMA_Z_V1 - 1,
        ),
        (256.0, DP_MAX_DISCRETE_SIGMA_Z_V1),
    ];
    for (sigma, expected) in valid_cases {
        let got = compute_discrete_sigma_z_v1(sigma).unwrap();
        assert_eq!(got, expected, "sigma={sigma:e}");
        let raw = (sigma as f64) * (DP_DISCRETE_LATTICE_RADIUS_V1 as f64);
        assert!(
            (got as f64) >= raw,
            "ceiling violated for sigma={sigma:e}: {got} < {raw}"
        );
    }

    let invalid_cases = [
        0.0f32,
        -0.0f32,
        -1.0f32,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(256.0f32.to_bits() + 1),
    ];
    for bad in invalid_cases {
        assert_eq!(
            compute_discrete_sigma_z_v1(bad).unwrap_err(),
            CryptoError::InvalidPrivacyParameter,
            "expected failure for sigma={bad:e}"
        );
    }
}

#[test]
fn discrete_samplers_extreme_scales_and_invalid_inputs() {
    let mut drbg = Sha256CounterDrbgV1::new(b"swarmtorch.test.dgauss.bounds", [0x61; 32], 1, 0);

    // Zero denominator / zero scale fail closed.
    assert_eq!(
        sample_bernoulli_exp_u128(1, 0, &mut drbg).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        sample_geometric_exp_fast_u128(0, &mut drbg).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        sample_dlaplace_u128(0, &mut drbg).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // Zero numerator always returns true; huge integer part short-circuits in O(1) expected steps.
    assert!(sample_bernoulli_exp1_u128(0, 1, &mut drbg).unwrap());
    assert!(sample_bernoulli_exp_u128(0, u128::MAX, &mut drbg).unwrap());
    assert!(!sample_bernoulli_exp_u128(u128::MAX, 1, &mut drbg).unwrap());

    // Exact boundary scales for sample_dgauss_u128: sigma_z = 1 and sigma_z = DP_MAX_DISCRETE_SIGMA_Z_V1.
    let mut c0 = 0usize;
    let mut c1 = 0usize;
    let mut cm1 = 0usize;
    for _ in 0..20_000 {
        let x = sample_dgauss_u128(1, &mut drbg).unwrap();
        match x {
            0 => c0 += 1,
            1 => c1 += 1,
            -1 => cm1 += 1,
            _ => {}
        }
    }
    let exp_neg_half = crate::num::exp_f64_soft(-0.5);
    assert!(abs_f64((c1 as f64) / (c0 as f64) - exp_neg_half) < 0.05);
    assert!(abs_f64((cm1 as f64) / (c0 as f64) - exp_neg_half) < 0.05);

    for _ in 0..64 {
        let _ = sample_dgauss_u128(DP_MAX_DISCRETE_SIGMA_Z_V1, &mut drbg).unwrap();
    }

    // Geometric(1 - e^-1) at t = 1 executes cleanly without bias.
    let mut g_sum = 0.0f64;
    let g_trials = 10_000usize;
    for _ in 0..g_trials {
        g_sum += sample_geometric_exp_fast_u128(1, &mut drbg).unwrap() as f64;
    }
    let expected_g_mean = 1.0 / (core::f64::consts::E - 1.0);
    assert!(
        abs_f64(g_sum / (g_trials as f64) - expected_g_mean) < 0.06,
        "geom(t=1) mean drift"
    );
}

#[test]
fn sample_bernoulli_exp1_and_exp_full_u128_denominator_domain_no_overflow() {
    let mut drbg = Sha256CounterDrbgV1::new(b"swarmtorch.test.bexp.u128", [0x62; 32], 1, 0);
    // Full u128 denominator domain: 0 <= num <= den, den >= 1 must NEVER overflow `k * den`.
    for _ in 0..64 {
        let _ = sample_bernoulli_exp1_u128(u128::MAX - 1, u128::MAX, &mut drbg)
            .expect("den = u128::MAX must not overflow k * den");
        let _ = sample_bernoulli_exp1_u128(1u128 << 126, (1u128 << 127) + 1, &mut drbg)
            .expect("den > u128::MAX / 2 must not probabilistically overflow k * den");
        let _ = sample_bernoulli_exp_u128(u128::MAX - 1, u128::MAX, &mut drbg)
            .expect("sample_bernoulli_exp_u128 must handle den = u128::MAX");
    }
}

#[test]
fn exact_bernoulli_f64_bit_level_thresholds_and_exponent_boundaries() {
    let q_max_sub_one = f64::from_bits(1.0f64.to_bits() - 1); // 1 - 2^-53
    let q_one_minus_eps = 1.0 - f64::EPSILON; // 1 - 2^-52
    let q_two_neg_53 = f64::EPSILON * 0.5; // 2^-53
    let q_two_neg_76 = f64::from_bits((1023u64 - 76) << 52); // 2^-76
    let q_two_neg_76_plus_ulp = f64::from_bits(((1023u64 - 76) << 52) | 1); // 2^-76 + 2^-128
    let q_two_neg_77 = f64::from_bits((1023u64 - 77) << 52); // 2^-77 (rejected)

    let cases: [(f64, u128); 6] = [
        (0.5, 1u128 << 127),
        (q_one_minus_eps, ((1u128 << 53) - 2) << 75),
        (q_max_sub_one, ((1u128 << 53) - 1) << 75),
        (q_two_neg_53, 1u128 << 75),
        (q_two_neg_76, 1u128 << 52),
        (q_two_neg_76_plus_ulp, (1u128 << 52) + 1),
    ];

    for (idx, &(q, expected_threshold)) in cases.iter().enumerate() {
        let mut drbg_fn =
            Sha256CounterDrbgV1::new(b"swarmtorch.test.bern.bits", [0x52; 32], idx as u64, 0);
        let mut drbg_oracle =
            Sha256CounterDrbgV1::new(b"swarmtorch.test.bern.bits", [0x52; 32], idx as u64, 0);
        for _ in 0..256 {
            let got = sample_bernoulli_f64_exact_v1(q, &mut drbg_fn).unwrap();
            let expected = drbg_oracle.next_u128() < expected_threshold;
            assert_eq!(got, expected, "threshold mismatch for q={q:e}");
        }
    }

    let mut drbg = Sha256CounterDrbgV1::new(b"swarmtorch.test.bern.bad", [0x53; 32], 0, 0);
    for bad in [
        -0.0,
        f64::NEG_INFINITY,
        f64::from_bits(1),
        q_two_neg_77,
        f64::from_bits(1.0f64.to_bits() + 1),
    ] {
        assert_eq!(
            sample_bernoulli_f64_exact_v1(bad, &mut drbg).unwrap_err(),
            CryptoError::InvalidPrivacyParameter,
            "expected failure for q={bad:e}"
        );
    }
}

#[cfg(feature = "alloc")]
#[test]
fn integer_lattice_clipping_handles_extreme_tiny_huge_subnormal_and_high_dim_boundaries() {
    let b_sq = (DP_DISCRETE_LATTICE_RADIUS_V1 as u128) * (DP_DISCRETE_LATTICE_RADIUS_V1 as u128);
    let subnormal = f32::from_bits(1);

    let max_100 = alloc::vec![f32::MAX; 100];
    let ones_65536 = alloc::vec![1.0f32; 65_536];
    let ones_65537 = alloc::vec![1.0f32; 65_537];
    let tiny_pair = [1e-38_f32, -1e-38_f32];
    let cases: [(&[f32], f32); 13] = [
        (&[1e-38_f32], 1e-38_f32),
        (&tiny_pair, 1e-38_f32),
        (&[f32::MIN_POSITIVE], f32::MIN_POSITIVE),
        (&[subnormal], subnormal),
        (&[f32::MAX], subnormal),
        (&[0.0, -0.0], subnormal),
        (&[f32::MAX], f32::MAX),
        (&max_100, f32::MAX),
        (&max_100, 1.0),
        (&[3.0, 4.0], 5.0),
        (&[1.0, 1.0, 1.0, 1.0], 2.0),
        (&ones_65536, 1.0),
        (&ones_65537, 1.0),
    ];

    for (vec_in, tau) in cases {
        let (q, norm_f64, _clipped) = clip_and_quantize_l2_lattice_v1(vec_in, tau)
            .expect("all finite non-empty updates with finite tau > 0 must succeed");
        assert!(norm_f64.is_finite());
        let norm_sq: u128 = q
            .iter()
            .map(|&v| {
                let a = v.unsigned_abs() as u128;
                a * a
            })
            .sum();
        assert!(
            norm_sq <= b_sq,
            "tau={tau:e} dim={}: norm_sq {norm_sq} > B^2 {b_sq}",
            vec_in.len()
        );
    }

    for bad_tau in [-0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            clip_and_quantize_l2_lattice_v1(&[1.0], bad_tau).unwrap_err(),
            CryptoError::InvalidPrivacyParameter
        );
    }
    assert_eq!(
        clip_and_quantize_l2_lattice_v1(&[f32::NEG_INFINITY], 1.0).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
}

#[cfg(feature = "alloc")]
#[test]
fn dp_fedavg_sealing_redaction_empty_draw_and_fail_closed_matrix() {
    use alloc::format;

    let run_id = [0xE1u8; 16];
    let other_run_id = [0xE2u8; 16];
    let parent_digest = [0xF1u8; 32];
    let peers: alloc::vec::Vec<PeerId> = (1u8..=4).map(|i| PeerId::new([i; 32])).collect();

    // 1. DpNoiseEntropyV1 Debug output must redact raw seed bytes.
    let secret_entropy = DpNoiseEntropyV1::from_explicit_seed_for_testing([0xAB; 32]).unwrap();
    let dbg_str = format!("{secret_entropy:?}");
    assert!(
        dbg_str.contains("[REDACTED]"),
        "debug must redact: {dbg_str}"
    );
    assert!(
        !dbg_str.contains("171"),
        "debug leaked decimal seed byte 0xAB (171): {dbg_str}"
    );

    // 2. SampledDpCohortV1 rejects empty authorized_peers and is order-invariant.
    assert_eq!(
        SampledDpCohortV1::full_cohort_v1(&[], run_id, 1).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        SampledDpCohortV1::sample_poisson_with_explicit_seed_v1(&[], 0.5, [0x11; 32], run_id, 1)
            .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    let reversed: alloc::vec::Vec<PeerId> = peers.iter().rev().copied().collect();
    let c_fwd =
        SampledDpCohortV1::sample_poisson_with_explicit_seed_v1(&peers, 0.5, [0x44; 32], run_id, 1)
            .unwrap();
    let c_rev = SampledDpCohortV1::sample_poisson_with_explicit_seed_v1(
        &reversed, 0.5, [0x44; 32], run_id, 1,
    )
    .unwrap();
    assert_eq!(c_fwd, c_rev);

    // 3. Find a deterministic empty Poisson draw (q = 0.25, N = 4 => q*N = 1.0) and verify
    //    execute_round_v1 succeeds cleanly without re-rolling.
    let mut empty_draw_cohort = None;
    for seed_byte in 1u8..=255 {
        let candidate = SampledDpCohortV1::sample_poisson_with_explicit_seed_v1(
            &peers,
            0.25,
            [seed_byte; 32],
            run_id,
            1,
        )
        .unwrap();
        if candidate.sampled_peers().is_empty() {
            empty_draw_cohort = Some(candidate);
            break;
        }
    }
    let empty_draw_cohort =
        empty_draw_cohort.expect("q=0.25, N=4 has 31.6% empty draw rate across 255 seeds");

    let profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 1.5,
        target_delta: 1e-5,
        max_epsilon: 5.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
    };

    let empty_draw_out = DpFedAvgMechanismV1::execute_round_v1(
        profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: run_id,
            round_id: 1,
            parent_digest_bytes: parent_digest,
            parameter_dim: 3,
            sampled_cohort: &empty_draw_cohort,
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x55; 32]).unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    assert_eq!(empty_draw_out.participant_count, 0);
    assert_eq!(empty_draw_out.fixed_sensitivity_denominator, 1.0);

    let full_cohort = SampledDpCohortV1::full_cohort_v1(&peers, run_id, 1).unwrap();
    let valid_u1 = GradientUpdate {
        sender: *peers[0].as_bytes(),
        sequence: 1,
        gradients: alloc::vec![0.5, -0.5, 0.25],
        round_id: 1,
    };

    // 4. Reject ExperimentalHypotheticalBudgetOnly, ExampleLevel, parameter_dim == 0,
    //    wrong run_id, dim mismatch, and NaN/Inf in updates.
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            DpMechanismProfileV1 {
                classification:
                    PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
                ..profile
            },
            DpFedAvgRoundInputV1 {
                run_id_bytes: run_id,
                round_id: 1,
                parent_digest_bytes: parent_digest,
                parameter_dim: 3,
                sampled_cohort: &full_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x55; 32])
                    .unwrap(),
                client_updates: core::slice::from_ref(&valid_u1),
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            DpMechanismProfileV1 {
                unit: DpUnitV1::ExampleLevel,
                ..profile
            },
            DpFedAvgRoundInputV1 {
                run_id_bytes: run_id,
                round_id: 1,
                parent_digest_bytes: parent_digest,
                parameter_dim: 3,
                sampled_cohort: &full_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x55; 32])
                    .unwrap(),
                client_updates: core::slice::from_ref(&valid_u1),
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            profile,
            DpFedAvgRoundInputV1 {
                run_id_bytes: run_id,
                round_id: 1,
                parent_digest_bytes: parent_digest,
                parameter_dim: 0,
                sampled_cohort: &full_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x55; 32])
                    .unwrap(),
                client_updates: &[],
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            profile,
            DpFedAvgRoundInputV1 {
                run_id_bytes: other_run_id,
                round_id: 1,
                parent_digest_bytes: parent_digest,
                parameter_dim: 3,
                sampled_cohort: &full_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x55; 32])
                    .unwrap(),
                client_updates: &[],
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    let wrong_dim_u = GradientUpdate {
        gradients: alloc::vec![0.5, -0.5],
        ..valid_u1.clone()
    };
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            profile,
            DpFedAvgRoundInputV1 {
                run_id_bytes: run_id,
                round_id: 1,
                parent_digest_bytes: parent_digest,
                parameter_dim: 3,
                sampled_cohort: &full_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x55; 32])
                    .unwrap(),
                client_updates: &[wrong_dim_u],
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    let nan_u = GradientUpdate {
        gradients: alloc::vec![0.5, f32::NAN, 0.25],
        ..valid_u1.clone()
    };
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            profile,
            DpFedAvgRoundInputV1 {
                run_id_bytes: run_id,
                round_id: 1,
                parent_digest_bytes: parent_digest,
                parameter_dim: 3,
                sampled_cohort: &full_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x55; 32])
                    .unwrap(),
                client_updates: &[nan_u],
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // 5. Certified accountant restore & hypothetical accountant rejecting verified proof.
    let mut hyp_acc = RdpAccountantV1::new(DpMechanismProfileV1 {
        classification: PrivacyAccountingClassificationV1::ExperimentalHypotheticalBudgetOnly,
        ..profile
    })
    .unwrap();
    assert_eq!(
        hyp_acc
            .record_verified_mechanism_round(&empty_draw_out.proof)
            .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    let valid_sched = compute_round_rdp_schedule_v1(0.25, 1.5).unwrap();
    let mut restored_max =
        RdpAccountantV1::restore_verified_mechanism_from_orders(profile, valid_sched, u64::MAX)
            .unwrap();
    assert_eq!(
        restored_max
            .record_verified_mechanism_round(&empty_draw_out.proof)
            .unwrap_err(),
        CryptoError::PrivacyBudgetExhausted
    );
    assert_eq!(restored_max.committed_rounds(), u64::MAX);
}

#[cfg(feature = "alloc")]
#[test]
fn dp_fedavg_rejects_stale_cross_round_gradient_update() {
    let run_id = [0xE3u8; 16];
    let parent_digest = [0xF2u8; 32];
    let peers = [PeerId::new([1u8; 32]), PeerId::new([2u8; 32])];
    let full_cohort = SampledDpCohortV1::full_cohort_v1(&peers, run_id, 1).unwrap();
    let profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 1.5,
        target_delta: 1e-5,
        max_epsilon: 5.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
    };
    let stale_round_u = GradientUpdate {
        sender: *peers[0].as_bytes(),
        sequence: 1,
        gradients: alloc::vec![0.5, -0.5, 0.25],
        round_id: 99,
    };
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            profile,
            DpFedAvgRoundInputV1 {
                run_id_bytes: run_id,
                round_id: 1,
                parent_digest_bytes: parent_digest,
                parameter_dim: 3,
                sampled_cohort: &full_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x55; 32])
                    .unwrap(),
                client_updates: &[stale_round_u],
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
}

#[cfg(feature = "alloc")]
#[test]
fn dp_fedavg_post_processing_is_total_under_max_tau_without_data_dependent_abort() {
    let run_id = [0xE4u8; 16];
    let parent_digest = [0xF3u8; 32];
    let peers = [PeerId::new([1u8; 32])];
    let single_cohort = SampledDpCohortV1::full_cohort_v1(&peers, run_id, 1).unwrap();
    let max_tau_profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: f32::MAX,
        noise_multiplier: 2.0,
        target_delta: 1e-5,
        max_epsilon: 5.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
    };
    let max_tau_out = DpFedAvgMechanismV1::execute_round_v1(
        max_tau_profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: run_id,
            round_id: 1,
            parent_digest_bytes: parent_digest,
            parameter_dim: 16,
            sampled_cohort: &single_cohort,
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x66; 32]).unwrap(),
            client_updates: &[],
        },
    )
    .expect("post-processing must be total without data-dependent f32 overflow aborts");
    assert!(max_tau_out.noisy_mean_update.iter().all(|v| v.is_finite()));
}

#[cfg(feature = "alloc")]
#[test]
fn dp_fedavg_post_processing_canonicalizes_negative_zero_to_match_output_digest() {
    let run_id = [0xE5u8; 16];
    let parent_digest = [0xF4u8; 32];
    let peers = [PeerId::new([1u8; 32])];
    let single_cohort = SampledDpCohortV1::full_cohort_v1(&peers, run_id, 1).unwrap();
    let tiny_tau_profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: f32::from_bits(1),
        noise_multiplier: 1.0 / 65_536.0,
        target_delta: 1e-5,
        max_epsilon: 5.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
    };
    let tiny_tau_out = DpFedAvgMechanismV1::execute_round_v1(
        tiny_tau_profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: run_id,
            round_id: 1,
            parent_digest_bytes: parent_digest,
            parameter_dim: 64,
            sampled_cohort: &single_cohort,
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x77; 32]).unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    assert!(
        tiny_tau_out
            .noisy_mean_update
            .iter()
            .all(|v| v.to_bits() != (-0.0f32).to_bits()),
        "emitted sanitized update must canonicalize -0.0 to +0.0 to match output digest"
    );
}

#[cfg(feature = "std")]
#[test]
fn os_csprng_cohort_q1_and_fail_closed_boundary_checks() {
    let run_id = [0xE6u8; 16];
    let peers = [PeerId::new([1u8; 32]), PeerId::new([2u8; 32])];
    assert_eq!(
        SampledDpCohortV1::sample_poisson_os_rng_v1(&[], 0.5, run_id, 1).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        SampledDpCohortV1::sample_poisson_os_rng_v1(&[peers[0], peers[0]], 1.0, run_id, 1)
            .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    let q1 = SampledDpCohortV1::sample_poisson_os_rng_v1(&peers, 1.0, run_id, 1).unwrap();
    assert_eq!(
        q1.sampling_provenance(),
        DpSamplingProvenanceV1::DeterministicFullCohort
    );
    assert_eq!(q1.sampled_peers().len(), 2);
}

#[test]
fn sample_bernoulli_exp1_u128_huge_coprime_denominator_matches_exp_rate() {
    // den = 2^128 - 1, num = 2^127 - 1 => gcd(num, den) == 1 (unreducible; k * den would overflow at k = 2).
    let den = u128::MAX;
    let num = (1u128 << 127) - 1;
    assert_eq!(gcd_u128(num, den), 1);

    let mut drbg = Sha256CounterDrbgV1::new(b"swarmtorch.test.bexp.u128.rate", [0x63; 32], 1, 0);
    let trials = 10_000usize;
    let mut hits = 0usize;
    for _ in 0..trials {
        if sample_bernoulli_exp1_u128(num, den, &mut drbg).unwrap() {
            hits += 1;
        }
    }
    let emp_rate = (hits as f64) / (trials as f64);
    let expected = crate::num::exp_f64_soft(-0.5);
    // SE = sqrt(0.6065 * 0.3935 / 10_000) = 0.0049; allow 6 SE = 0.03.
    assert!(
        abs_f64(emp_rate - expected) < 0.03,
        "huge-den Bernoulli(exp(-num/den)) rate drift: {emp_rate} vs {expected}"
    );
}

#[cfg(feature = "alloc")]
#[test]
fn sampled_cohort_digest_is_salted_with_derived_seed_preventing_subset_bruteforce() {
    let run_id = [0xE7u8; 16];
    let parent_digest = [0xF5u8; 32];
    let peers = [PeerId::new([1u8; 32]), PeerId::new([2u8; 32])];

    let mut empty_draws = alloc::vec::Vec::new();
    let mut full_draws = alloc::vec::Vec::new();
    for seed_byte in 1u8..=255 {
        let cohort = SampledDpCohortV1::sample_poisson_with_explicit_seed_v1(
            &peers,
            0.5,
            [seed_byte; 32],
            run_id,
            7,
        )
        .unwrap();
        if cohort.sampled_peers().is_empty() && empty_draws.len() < 2 {
            empty_draws.push(cohort);
        } else if cohort.sampled_peers().len() == peers.len() && full_draws.len() < 2 {
            full_draws.push(cohort);
        }
        if empty_draws.len() == 2 && full_draws.len() == 2 {
            break;
        }
    }
    assert_eq!(empty_draws.len(), 2);
    assert_eq!(full_draws.len(), 2);

    // Identical realized subsets under distinct seeds MUST yield distinct salted digests,
    // and MUST NOT equal the unsalted peer-set digest.
    let unsalted_empty = compute_peer_set_digest_v1(b"sampled", empty_draws[0].sampled_peers());
    assert_eq!(
        empty_draws[0].sampled_peers(),
        empty_draws[1].sampled_peers()
    );
    assert_ne!(
        empty_draws[0].sampled_cohort_digest_sha256(),
        empty_draws[1].sampled_cohort_digest_sha256()
    );
    assert_ne!(
        empty_draws[0].sampled_cohort_digest_sha256(),
        &unsalted_empty
    );
    assert_ne!(
        empty_draws[1].sampled_cohort_digest_sha256(),
        &unsalted_empty
    );

    let unsalted_full = compute_peer_set_digest_v1(b"sampled", full_draws[0].sampled_peers());
    assert_eq!(full_draws[0].sampled_peers(), full_draws[1].sampled_peers());
    assert_ne!(
        full_draws[0].sampled_cohort_digest_sha256(),
        full_draws[1].sampled_cohort_digest_sha256()
    );
    assert_ne!(full_draws[0].sampled_cohort_digest_sha256(), &unsalted_full);
    assert_ne!(full_draws[1].sampled_cohort_digest_sha256(), &unsalted_full);

    // Deterministic full_cohort_v1 uses unsalted digest (full cohort is public at q = 1.0),
    // whereas explicit-seed q = 1.0 still salts with derived_seed.
    let det_full = SampledDpCohortV1::full_cohort_v1(&peers, run_id, 7).unwrap();
    assert_eq!(det_full.sampled_cohort_digest_sha256(), &unsalted_full);
    let exp_q1 =
        SampledDpCohortV1::sample_poisson_with_explicit_seed_v1(&peers, 1.0, [0x42; 32], run_id, 7)
            .unwrap();
    assert_eq!(
        exp_q1.sampling_provenance(),
        DpSamplingProvenanceV1::ExplicitSeedNonProduction
    );
    assert_ne!(exp_q1.sampled_cohort_digest_sha256(), &unsalted_full);

    // Proof minted by execute_round_v1 carries the exact salted sampled_cohort_digest_sha256.
    let profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 1.5,
        target_delta: 1e-5,
        max_epsilon: 5.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
    };
    let out = DpFedAvgMechanismV1::execute_round_v1(
        profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: run_id,
            round_id: 7,
            parent_digest_bytes: parent_digest,
            parameter_dim: 4,
            sampled_cohort: &empty_draws[0],
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x88; 32]).unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    assert_eq!(
        out.proof.sampled_cohort_digest_sha256(),
        empty_draws[0].sampled_cohort_digest_sha256()
    );
}

#[cfg(feature = "alloc")]
#[test]
fn dp_fedavg_post_processing_digest_matches_emitted_f32_under_clamping_and_neg_zero_underflow() {
    use sha2::Digest;

    let recompute_output_digest = |run_id: [u8; 16], round_id: u64, vals: &[f32]| -> [u8; 32] {
        let mut h = sha2::Sha256::new();
        h.update(DP_OUTPUT_DIGEST_DOMAIN_V1);
        h.update(run_id);
        h.update(round_id.to_le_bytes());
        h.update((vals.len() as u64).to_le_bytes());
        for &v in vals {
            h.update(v.to_bits().to_le_bytes());
        }
        h.finalize().into()
    };

    let run_id = [0xE8u8; 16];
    let parent_digest = [0xF6u8; 32];
    let peers = [PeerId::new([1u8; 32])];
    let single_cohort = SampledDpCohortV1::full_cohort_v1(&peers, run_id, 3).unwrap();

    // Case A: Non-empty extreme update + max tau forces saturation clamping to +/- f32::MAX,
    // without aborting, and sanitized_output_digest_sha256 matches the emitted f32 bits.
    let max_tau_profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: f32::MAX,
        noise_multiplier: 2.0,
        target_delta: 1e-5,
        max_epsilon: 5.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
    };
    let extreme_upd = GradientUpdate {
        sender: *peers[0].as_bytes(),
        sequence: 1,
        gradients: alloc::vec![f32::MAX; 32],
        round_id: 3,
    };
    let max_out = DpFedAvgMechanismV1::execute_round_v1(
        max_tau_profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: run_id,
            round_id: 3,
            parent_digest_bytes: parent_digest,
            parameter_dim: 32,
            sampled_cohort: &single_cohort,
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x69; 32]).unwrap(),
            client_updates: &[extreme_upd],
        },
    )
    .unwrap();
    assert!(max_out.noisy_mean_update.iter().all(|v| v.is_finite()));
    assert!(
        max_out
            .noisy_mean_update
            .iter()
            .any(|&v| v == f32::MAX || v == -f32::MAX),
        "expected saturation clamping to engage on f32::MAX profile"
    );
    assert_eq!(
        *max_out.proof.sanitized_output_digest_sha256(),
        recompute_output_digest(run_id, 3, &max_out.noisy_mean_update)
    );

    // Case B: Tiny subnormal tau forces underflow to zero; all zeros must be canonical +0.0
    // and match sanitized_output_digest_sha256 bit-for-bit.
    let tiny_tau_profile = DpMechanismProfileV1 {
        clipping_radius_tau: f32::from_bits(1),
        noise_multiplier: 1.0 / 65_536.0,
        ..max_tau_profile
    };
    let tiny_out = DpFedAvgMechanismV1::execute_round_v1(
        tiny_tau_profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: run_id,
            round_id: 3,
            parent_digest_bytes: parent_digest,
            parameter_dim: 64,
            sampled_cohort: &single_cohort,
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x79; 32]).unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    assert!(tiny_out
        .noisy_mean_update
        .iter()
        .any(|v| v.to_bits() == 0.0f32.to_bits()));
    assert!(tiny_out
        .noisy_mean_update
        .iter()
        .all(|v| v.to_bits() != (-0.0f32).to_bits()));
    assert_eq!(
        *tiny_out.proof.sanitized_output_digest_sha256(),
        recompute_output_digest(run_id, 3, &tiny_out.noisy_mean_update)
    );
}

#[cfg(feature = "alloc")]
#[test]
fn certified_dp_entrypoints_enforce_sigma_cap_empty_draw_overcount_and_adjacency_binding() {
    let run_id = [0xE9u8; 16];
    let parent_digest = [0xF7u8; 32];
    let peers = [PeerId::new([1u8; 32]), PeerId::new([2u8; 32])];

    let over_cap_profile = DpMechanismProfileV1 {
        unit: DpUnitV1::ClientLevel,
        adjacency: DpAdjacencyV1::AddRemoveOne,
        clipping_radius_tau: 1.0,
        noise_multiplier: 257.0, // > DP_MAX_DISCRETE_SIGMA_Z_V1 / B = 256.0
        target_delta: 1e-5,
        max_epsilon: 5.0,
        conversion_formula: DpConversionFormulaV1::Balle2020Tight,
        classification: PrivacyAccountingClassificationV1::VerifiedMechanismProtected,
    };
    assert_eq!(
        RdpAccountantV1::new_verified_mechanism(over_cap_profile).unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        RdpAccountantV1::restore_verified_mechanism_from_orders(
            over_cap_profile,
            [0.0; RDP_GRID_LEN],
            0,
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    let full_cohort = SampledDpCohortV1::full_cohort_v1(&peers, run_id, 1).unwrap();
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            over_cap_profile,
            DpFedAvgRoundInputV1 {
                run_id_bytes: run_id,
                round_id: 1,
                parent_digest_bytes: parent_digest,
                parameter_dim: 2,
                sampled_cohort: &full_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x91; 32])
                    .unwrap(),
                client_updates: &[],
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // Submitting a non-empty update against an empty Poisson draw fails closed.
    let empty_cohort = (1u8..=255)
        .map(|b| {
            SampledDpCohortV1::sample_poisson_with_explicit_seed_v1(&peers, 0.5, [b; 32], run_id, 1)
                .unwrap()
        })
        .find(|c| c.sampled_peers().is_empty())
        .expect("q=0.5, N=2 has 25% empty draw rate");
    let valid_profile = DpMechanismProfileV1 {
        noise_multiplier: 1.5,
        ..over_cap_profile
    };
    let upd = GradientUpdate {
        sender: *peers[0].as_bytes(),
        sequence: 1,
        gradients: alloc::vec![0.1, 0.2],
        round_id: 1,
    };
    assert_eq!(
        DpFedAvgMechanismV1::execute_round_v1(
            valid_profile,
            DpFedAvgRoundInputV1 {
                run_id_bytes: run_id,
                round_id: 1,
                parent_digest_bytes: parent_digest,
                parameter_dim: 2,
                sampled_cohort: &empty_cohort,
                noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x92; 32])
                    .unwrap(),
                client_updates: &[upd],
            },
        )
        .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );

    // Proof minted under AddRemoveOne cannot be previewed or recorded into a ReplaceOne accountant.
    let outcome = DpFedAvgMechanismV1::execute_round_v1(
        valid_profile,
        DpFedAvgRoundInputV1 {
            run_id_bytes: run_id,
            round_id: 1,
            parent_digest_bytes: parent_digest,
            parameter_dim: 2,
            sampled_cohort: &full_cohort,
            noise_entropy: DpNoiseEntropyV1::from_explicit_seed_for_testing([0x93; 32]).unwrap(),
            client_updates: &[],
        },
    )
    .unwrap();
    let replace_profile = DpMechanismProfileV1 {
        adjacency: DpAdjacencyV1::ReplaceOne,
        ..valid_profile
    };
    let mut replace_acc = RdpAccountantV1::new_verified_mechanism(replace_profile).unwrap();
    assert_eq!(
        replace_acc
            .preview_verified_mechanism_round(&outcome.proof)
            .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(
        replace_acc
            .record_verified_mechanism_round(&outcome.proof)
            .unwrap_err(),
        CryptoError::InvalidPrivacyParameter
    );
    assert_eq!(replace_acc.committed_rounds(), 0);
}
