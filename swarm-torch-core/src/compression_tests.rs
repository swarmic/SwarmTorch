use super::*;
use alloc::collections::BTreeSet;
use alloc::vec;

fn topk_reference_sparse(gradients: &[f32], k_ratio: f32) -> (Vec<u32>, Vec<u8>) {
    let raw = (gradients.len() as f64) * (k_ratio as f64);
    let mut k = raw as usize;
    if (k as f64) < raw {
        k = k.saturating_add(1);
    }
    let k = k.max(1).min(gradients.len());

    let mut indexed: Vec<(usize, f32)> =
        gradients.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(topk_priority_cmp);
    indexed.truncate(k);
    indexed.sort_unstable_by_key(|(idx, _)| *idx);

    (
        indexed.iter().map(|(i, _)| *i as u32).collect(),
        indexed.iter().flat_map(|(_, v)| v.to_le_bytes()).collect(),
    )
}

#[test]
fn decompress_rejects_sparse_index_value_mismatch() {
    let compressed = CompressedGradient {
        method: CompressionMethod::TopK { k_ratio: 0.5 },
        shape: vec![2],
        num_elements: 2,
        data: CompressedData::Sparse {
            indices: vec![0, 1],
            values: vec![1, 0, 0, 0],
        },
    };
    assert_eq!(
        compressed.decompress().unwrap_err(),
        CompressionError::InvalidData
    );
}

#[test]
fn decompress_sparse_returns_error_for_out_of_bounds_indices() {
    let compressed = CompressedGradient {
        method: CompressionMethod::TopK { k_ratio: 0.5 },
        shape: vec![2],
        num_elements: 2,
        data: CompressedData::Sparse {
            indices: vec![0, 99],
            values: vec![1, 0, 0, 0, 2, 0, 0, 0],
        },
    };
    assert_eq!(
        compressed.decompress().unwrap_err(),
        CompressionError::IndexOutOfBounds
    );
}

#[test]
fn decompress_sparse_rejects_unsorted_or_duplicate_indices() {
    let dup = CompressedGradient {
        method: CompressionMethod::TopK { k_ratio: 0.5 },
        shape: vec![4],
        num_elements: 4,
        data: CompressedData::Sparse {
            indices: vec![1, 1],
            values: vec![0, 0, 128, 63, 0, 0, 0, 64],
        },
    };
    assert_eq!(
        dup.decompress().unwrap_err(),
        CompressionError::UnsortedOrDuplicateIndices
    );

    let unsorted = CompressedGradient {
        method: CompressionMethod::TopK { k_ratio: 0.5 },
        shape: vec![4],
        num_elements: 4,
        data: CompressedData::Sparse {
            indices: vec![2, 1],
            values: vec![0, 0, 128, 63, 0, 0, 0, 64],
        },
    };
    assert_eq!(
        unsorted.decompress().unwrap_err(),
        CompressionError::UnsortedOrDuplicateIndices
    );
}

#[test]
fn decompress_enforces_pre_allocation_limit_and_shape_product() {
    let huge = CompressedGradient {
        method: CompressionMethod::None,
        shape: vec![usize::MAX],
        num_elements: usize::MAX,
        data: CompressedData::Dense(vec![]),
    };
    assert_eq!(
        huge.decompress().unwrap_err(),
        CompressionError::ResourceLimitExceeded
    );

    let shape_mismatch = CompressedGradient {
        method: CompressionMethod::None,
        shape: vec![2, 3],
        num_elements: 4,
        data: CompressedData::Dense(vec![0; 16]),
    };
    assert_eq!(
        shape_mismatch.decompress().unwrap_err(),
        CompressionError::ShapeMismatch
    );
}

#[test]
fn compress_and_decompress_reject_non_finite_values() {
    assert_eq!(
        CompressedGradient::compress(&[1.0, f32::NAN], CompressionMethod::None).unwrap_err(),
        CompressionError::NonFiniteValue
    );
    assert_eq!(
        CompressedGradient::compress(
            &[1.0, f32::INFINITY],
            CompressionMethod::TopK { k_ratio: 0.5 }
        )
        .unwrap_err(),
        CompressionError::NonFiniteValue
    );

    let mut nan_bytes = Vec::new();
    nan_bytes.extend_from_slice(&1.0f32.to_le_bytes());
    nan_bytes.extend_from_slice(&f32::NAN.to_le_bytes());
    let crafted = CompressedGradient {
        method: CompressionMethod::None,
        shape: vec![2],
        num_elements: 2,
        data: CompressedData::Dense(nan_bytes),
    };
    assert_eq!(
        crafted.decompress().unwrap_err(),
        CompressionError::NonFiniteValue
    );
}

#[test]
fn decompress_rejects_misaligned_dense_data() {
    let compressed = CompressedGradient {
        method: CompressionMethod::None,
        shape: vec![2],
        num_elements: 2,
        data: CompressedData::Dense(vec![1, 0, 0, 0, 2, 0, 0]),
    };
    assert_eq!(
        compressed.decompress().unwrap_err(),
        CompressionError::InvalidData
    );
}

#[test]
fn compression_ratio_is_well_defined_for_empty_input() {
    let compressed = CompressedGradient::compress(&[], CompressionMethod::None).unwrap();
    assert_eq!(compressed.compression_ratio(), 1.0);
}

#[test]
fn try_usize_to_u32_rejects_overflow() {
    assert_eq!(try_usize_to_u32(0), Ok(0));
    assert_eq!(try_usize_to_u32(u32::MAX as usize), Ok(u32::MAX));
    assert_eq!(
        try_usize_to_u32((u32::MAX as usize) + 1),
        Err(CompressionError::IndexOverflow)
    );
}

#[test]
fn decompress_rejects_unmatched_method_data() {
    let compressed = CompressedGradient {
        method: CompressionMethod::TopK { k_ratio: 0.5 },
        shape: vec![2],
        num_elements: 2,
        data: CompressedData::Dense(vec![1, 0, 0, 0, 2, 0, 0, 0]),
    };
    assert_eq!(
        compressed.decompress().unwrap_err(),
        CompressionError::InvalidData
    );
}

#[test]
fn compress_random_sparse_returns_unsupported() {
    let gradients = vec![1.0, 2.0, 3.0];
    let result = CompressedGradient::compress(
        &gradients,
        CompressionMethod::RandomSparse { p: 0.5, seed: 42 },
    );
    assert_eq!(result.unwrap_err(), CompressionError::UnsupportedMethod);
}

#[test]
fn compress_topk_quantized_returns_unsupported() {
    let gradients = vec![1.0, 2.0, 3.0];
    let result = CompressedGradient::compress(
        &gradients,
        CompressionMethod::TopKQuantized {
            k_ratio: 0.5,
            scale: 1.0,
        },
    );
    assert_eq!(result.unwrap_err(), CompressionError::UnsupportedMethod);
}

#[test]
fn decompress_sparse_random_sparse_returns_unsupported() {
    let compressed = CompressedGradient {
        method: CompressionMethod::RandomSparse { p: 0.5, seed: 42 },
        shape: vec![4],
        num_elements: 4,
        data: CompressedData::Sparse {
            indices: vec![0, 1],
            values: vec![0, 0, 128, 63, 0, 0, 0, 64],
        },
    };
    assert_eq!(
        compressed.decompress().unwrap_err(),
        CompressionError::UnsupportedMethod,
        "sparse data with RandomSparse method must be rejected on decompress"
    );
}

#[test]
fn decompress_sparse_topk_quantized_returns_unsupported() {
    let compressed = CompressedGradient {
        method: CompressionMethod::TopKQuantized {
            k_ratio: 0.5,
            scale: 1.0,
        },
        shape: vec![4],
        num_elements: 4,
        data: CompressedData::Sparse {
            indices: vec![0, 1],
            values: vec![0, 0, 128, 63, 0, 0, 0, 64],
        },
    };
    assert_eq!(
        compressed.decompress().unwrap_err(),
        CompressionError::UnsupportedMethod,
        "sparse data with TopKQuantized method must be rejected on decompress"
    );
}

#[test]
fn compress_quantized_rejects_zero_scale() {
    let data = vec![1.0f32; 4];
    let result = CompressedGradient::compress(&data, CompressionMethod::Quantized { scale: 0.0 });
    assert_eq!(result.unwrap_err(), CompressionError::InvalidScale);
}

#[test]
fn compress_quantized_rejects_nan_scale() {
    let data = vec![1.0f32; 4];
    let result =
        CompressedGradient::compress(&data, CompressionMethod::Quantized { scale: f32::NAN });
    assert_eq!(result.unwrap_err(), CompressionError::InvalidScale);
}

#[test]
fn compress_quantized_rejects_negative_scale() {
    let data = vec![1.0f32; 4];
    let result = CompressedGradient::compress(&data, CompressionMethod::Quantized { scale: -1.0 });
    assert_eq!(result.unwrap_err(), CompressionError::InvalidScale);
}

#[test]
fn compress_quantized_rejects_infinity_scale() {
    let data = vec![1.0f32; 4];
    let result = CompressedGradient::compress(
        &data,
        CompressionMethod::Quantized {
            scale: f32::INFINITY,
        },
    );
    assert_eq!(result.unwrap_err(), CompressionError::InvalidScale);
}

#[test]
fn compress_quantized_rejects_overflowing_scale() {
    let data = vec![1.0f32; 4];
    let result = CompressedGradient::compress(
        &data,
        CompressionMethod::Quantized {
            scale: f32::MAX * 0.5, // 128 * scale overflows f32::MAX
        },
    );
    assert_eq!(result.unwrap_err(), CompressionError::InvalidScale);
}

#[test]
fn decompress_quantized_rejects_zero_scale_crafted() {
    let compressed = CompressedGradient {
        method: CompressionMethod::Quantized { scale: 0.0 },
        shape: vec![4],
        num_elements: 4,
        data: CompressedData::Dense(vec![10, 20, 30, 40]),
    };
    assert_eq!(
        compressed.decompress().unwrap_err(),
        CompressionError::InvalidScale,
        "decompress must reject zero scale (crafted payload)"
    );
}

#[test]
fn decompress_quantized_rejects_nan_scale_crafted() {
    let compressed = CompressedGradient {
        method: CompressionMethod::Quantized { scale: f32::NAN },
        shape: vec![4],
        num_elements: 4,
        data: CompressedData::Dense(vec![10, 20, 30, 40]),
    };
    assert_eq!(
        compressed.decompress().unwrap_err(),
        CompressionError::InvalidScale,
        "decompress must reject NaN scale (crafted payload)"
    );
}

#[test]
fn topk_selection_matches_sort_semantics() {
    let gradients = vec![
        1.0, -3.0, 3.0, 0.2, -0.2, 8.0, -8.0, 5.0, -5.0, 0.0, -1.5, 1.5,
    ];
    let k_ratio = 0.25;
    let compressed =
        CompressedGradient::compress(&gradients, CompressionMethod::TopK { k_ratio }).unwrap();
    let (expected_indices, expected_values) = topk_reference_sparse(&gradients, k_ratio);

    match compressed.data {
        CompressedData::Sparse { indices, values } => {
            assert_eq!(indices, expected_indices);
            assert_eq!(values, expected_values);
        }
        _ => panic!("expected sparse data for TopK"),
    }
}

#[test]
fn topk_selection_deterministic_with_ties() {
    let gradients = vec![2.0, -2.0, 2.0, -2.0, 1.0, -1.0, 1.0, -1.0];
    let method = CompressionMethod::TopK { k_ratio: 0.5 };

    let a = CompressedGradient::compress(&gradients, method.clone()).unwrap();
    let b = CompressedGradient::compress(&gradients, method).unwrap();

    match (&a.data, &b.data) {
        (
            CompressedData::Sparse {
                indices: a_i,
                values: a_v,
            },
            CompressedData::Sparse {
                indices: b_i,
                values: b_v,
            },
        ) => {
            assert_eq!(a_i, b_i);
            assert_eq!(a_v, b_v);
            let uniq: BTreeSet<u32> = a_i.iter().copied().collect();
            assert_eq!(uniq.len(), a_i.len());
        }
        _ => panic!("expected sparse TopK outputs"),
    }
}

#[test]
fn topk_selection_matches_sort_semantics_across_shapes() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let sizes = [1usize, 2, 3, 7, 16, 33, 127];
    let ratios = [0.01f32, 0.1, 0.25, 0.5, 0.9, 1.0];

    for len in sizes {
        let mut gradients = Vec::with_capacity(len);
        for i in 0..len {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((state >> 16) as u32) as f32 / (u32::MAX as f32);
            let sign = if (i & 1) == 0 { 1.0 } else { -1.0 };
            gradients.push(sign * (x - 0.5) * 10.0);
        }

        for k_ratio in ratios {
            let compressed =
                CompressedGradient::compress(&gradients, CompressionMethod::TopK { k_ratio })
                    .expect("topk compression should succeed");
            let (expected_indices, expected_values) = topk_reference_sparse(&gradients, k_ratio);
            match compressed.data {
                CompressedData::Sparse { indices, values } => {
                    assert_eq!(indices, expected_indices, "len={len} k_ratio={k_ratio}");
                    assert_eq!(values, expected_values, "len={len} k_ratio={k_ratio}");
                }
                _ => panic!("expected sparse data for TopK"),
            }
        }
    }
}

#[test]
fn classic_residual_accumulates_error_and_preserves_state_on_failure() {
    let mut ef = ClassicResidualCompressor::new(2, CompressionMethod::TopK { k_ratio: 0.5 })
        .expect("valid residual compressor");

    // Step 1: [10.0, 4.0] -> Top-1 selects index 0 (10.0), residual becomes [0.0, 4.0].
    let c1 = ef.compress_step(&[10.0, 4.0]).unwrap();
    assert_eq!(c1.decompress().unwrap(), vec![10.0, 0.0]);
    assert_eq!(ef.residual(), &[0.0, 4.0]);

    // Non-finite input must fail without mutating residual [0.0, 4.0].
    assert_eq!(
        ef.compress_step(&[f32::NAN, 1.0]).unwrap_err(),
        CompressionError::NonFiniteValue
    );
    assert_eq!(ef.residual(), &[0.0, 4.0]);

    // Step 2: [1.0, 4.0] + residual [0.0, 4.0] = [1.0, 8.0] -> Top-1 selects index 1 (8.0),
    // residual becomes [1.0, 0.0].
    let c2 = ef.compress_step(&[1.0, 4.0]).unwrap();
    assert_eq!(c2.decompress().unwrap(), vec![0.0, 8.0]);
    assert_eq!(ef.residual(), &[1.0, 0.0]);
}

#[test]
fn ef21_sender_and_receiver_stay_synchronized_and_reject_dropped_step() {
    let mut sender = Ef21SenderState::new(3, CompressionMethod::TopK { k_ratio: 0.34 }).unwrap();
    let mut receiver = Ef21ReceiverState::new(3).unwrap();

    let (seq0, delta0) = sender.step(&[6.0, 2.0, 1.0]).unwrap();
    assert_eq!(seq0, 0);
    let rec0 = receiver.apply_step(seq0, &delta0).unwrap();
    assert_eq!(rec0, sender.estimate());

    let (seq1, delta1) = sender.step(&[6.0, 5.0, 1.0]).unwrap();
    assert_eq!(seq1, 1);

    // Out-of-order / dropped packet (passing sequence 2 when receiver expects 1) must fail closed.
    assert_eq!(
        receiver.apply_step(2, &delta1).unwrap_err(),
        CompressionError::InvalidData
    );
    assert_eq!(receiver.expected_sequence(), 1);

    // Applying the expected sequence 1 succeeds and restores exact sender/receiver agreement.
    let rec1 = receiver.apply_step(seq1, &delta1).unwrap();
    assert_eq!(rec1, sender.estimate());
}
