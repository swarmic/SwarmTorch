//! Gradient compression utilities
//!
//! This module provides compression algorithms for efficient gradient
//! transmission over bandwidth-constrained networks.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

#[cfg(feature = "alloc")]
fn topk_priority_cmp(a: &(usize, f32), b: &(usize, f32)) -> core::cmp::Ordering {
    // Deterministic total order:
    // 1) descending |value| (largest first)
    // 2) ascending index (tie-breaker)
    let abs_cmp = b.1.abs().total_cmp(&a.1.abs());
    if abs_cmp == core::cmp::Ordering::Equal {
        a.0.cmp(&b.0)
    } else {
        abs_cmp
    }
}

/// Error type for compression operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionError {
    /// Index value exceeds u32 capacity
    IndexOverflow,
    /// Invalid compressed data format
    InvalidData,
    /// Decompressed index out of bounds
    IndexOutOfBounds,
    /// Compression method is not yet implemented
    UnsupportedMethod,
    /// Quantization scale is invalid (non-finite, zero, or negative)
    InvalidScale,
}

impl core::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::IndexOverflow => write!(f, "index value exceeds u32 capacity"),
            Self::InvalidData => write!(f, "invalid compressed data format"),
            Self::IndexOutOfBounds => write!(f, "decompressed index out of bounds"),
            Self::UnsupportedMethod => write!(f, "compression method is not yet implemented"),
            Self::InvalidScale => write!(
                f,
                "quantization scale is invalid (non-finite, zero, or negative)"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CompressionError {}

/// Helper to convert usize length/index to u32 safely
pub fn try_usize_to_u32(val: usize) -> core::result::Result<u32, CompressionError> {
    u32::try_from(val).map_err(|_| CompressionError::IndexOverflow)
}

/// Compression method for gradients
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CompressionMethod {
    /// No compression (full gradient)
    #[default]
    None,
    /// Top-K sparsification (send only top K% of gradient values)
    TopK {
        /// Ratio of values to keep (e.g., 0.01 for 1%)
        k_ratio: f32,
    },
    /// Random sparsification with seed
    RandomSparse {
        /// Probability of keeping each value
        p: f32,
        /// Random seed for reproducibility
        seed: u64,
    },
    /// Quantized to INT8 with scale
    Quantized {
        /// Scale factor for dequantization
        scale: f32,
    },
    /// Combined: TopK + Quantization
    TopKQuantized {
        /// Ratio of values to keep
        k_ratio: f32,
        /// Scale factor for dequantization
        scale: f32,
    },
}

/// Compressed gradient data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressedData {
    /// Dense representation (full or quantized)
    #[cfg(feature = "alloc")]
    Dense(Vec<u8>),
    /// Sparse representation (indices + values)
    #[cfg(feature = "alloc")]
    Sparse {
        /// Indices of non-zero values
        indices: Vec<u32>,
        /// Compressed values
        values: Vec<u8>,
    },
}

/// A compressed gradient update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedGradient {
    /// Compression method used
    pub method: CompressionMethod,
    /// Original tensor shape
    #[cfg(feature = "alloc")]
    pub shape: Vec<usize>,
    /// Total number of elements
    pub num_elements: usize,
    /// Compressed data
    #[cfg(feature = "alloc")]
    pub data: CompressedData,
}

#[cfg(feature = "alloc")]
impl CompressedGradient {
    /// Compress a gradient using the specified method
    pub fn compress(
        gradients: &[f32],
        method: CompressionMethod,
    ) -> core::result::Result<Self, CompressionError> {
        match method {
            CompressionMethod::None => {
                // Convert f32 to bytes
                let bytes: Vec<u8> = gradients.iter().flat_map(|f| f.to_le_bytes()).collect();
                Ok(Self {
                    method,
                    shape: alloc::vec![gradients.len()],
                    num_elements: gradients.len(),
                    data: CompressedData::Dense(bytes),
                })
            }
            CompressionMethod::TopK { k_ratio } => {
                // Avoid `f32::ceil()` so `no_std + alloc` builds don't require libm.
                let raw = (gradients.len() as f32) * k_ratio;
                let mut k = raw as usize;
                if (k as f32) < raw {
                    k = k.saturating_add(1);
                }
                let k = k.max(1).min(gradients.len());

                // E-01: selection-based TopK with deterministic tie handling.
                let mut indexed: Vec<(usize, f32)> =
                    gradients.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                if k < indexed.len() {
                    indexed.select_nth_unstable_by(k, topk_priority_cmp);
                    indexed.truncate(k);
                }

                // Stabilize wire output regardless of selection internals.
                indexed.sort_unstable_by_key(|(idx, _)| *idx);

                let indices: core::result::Result<Vec<u32>, CompressionError> =
                    indexed.iter().map(|(i, _)| try_usize_to_u32(*i)).collect();
                let indices = indices?;
                let values: Vec<u8> = indexed.iter().flat_map(|(_, v)| v.to_le_bytes()).collect();

                Ok(Self {
                    method,
                    shape: alloc::vec![gradients.len()],
                    num_elements: gradients.len(),
                    data: CompressedData::Sparse { indices, values },
                })
            }
            CompressionMethod::Quantized { scale } => {
                // L-03: reject non-finite, zero, or negative scales
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(CompressionError::InvalidScale);
                }
                // Quantize to INT8
                let bytes: Vec<u8> = gradients
                    .iter()
                    .map(|&v| {
                        let quantized = (v / scale).clamp(-128.0, 127.0) as i8;
                        quantized as u8
                    })
                    .collect();
                Ok(Self {
                    method: CompressionMethod::Quantized { scale },
                    shape: alloc::vec![gradients.len()],
                    num_elements: gradients.len(),
                    data: CompressedData::Dense(bytes),
                })
            }
            CompressionMethod::RandomSparse { .. } => Err(CompressionError::UnsupportedMethod),
            CompressionMethod::TopKQuantized { .. } => Err(CompressionError::UnsupportedMethod),
        }
    }

    /// Decompress back to a gradient vector
    pub fn decompress(&self) -> core::result::Result<Vec<f32>, CompressionError> {
        match (&self.data, &self.method) {
            (CompressedData::Dense(bytes), CompressionMethod::None) => {
                if bytes.len() % 4 != 0 || bytes.len() / 4 != self.num_elements {
                    return Err(CompressionError::InvalidData);
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            }
            (CompressedData::Dense(bytes), CompressionMethod::Quantized { scale }) => {
                // L-03: reject non-finite, zero, or negative scales on decompress too
                if !scale.is_finite() || *scale <= 0.0 {
                    return Err(CompressionError::InvalidScale);
                }
                if bytes.len() != self.num_elements {
                    return Err(CompressionError::InvalidData);
                }
                Ok(bytes.iter().map(|&b| (b as i8 as f32) * scale).collect())
            }
            (CompressedData::Sparse { indices, values }, CompressionMethod::TopK { .. }) => {
                if values.len() % 4 != 0 || values.len() / 4 != indices.len() {
                    return Err(CompressionError::InvalidData);
                }
                let mut result = alloc::vec![0.0f32; self.num_elements];
                for (idx, chunk) in indices.iter().zip(values.chunks_exact(4)) {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let idx = *idx as usize;
                    if idx >= result.len() {
                        return Err(CompressionError::IndexOutOfBounds);
                    }
                    result[idx] = value;
                }
                Ok(result)
            }
            // NEW-06: unsupported methods must fail closed on decompress too.
            (CompressedData::Sparse { .. }, CompressionMethod::RandomSparse { .. })
            | (CompressedData::Sparse { .. }, CompressionMethod::TopKQuantized { .. }) => {
                Err(CompressionError::UnsupportedMethod)
            }
            _ => Err(CompressionError::InvalidData),
        }
    }

    /// Get the compressed size in bytes
    pub fn compressed_size(&self) -> usize {
        match &self.data {
            CompressedData::Dense(bytes) => bytes.len(),
            CompressedData::Sparse { indices, values } => indices.len() * 4 + values.len(),
        }
    }

    /// Get the compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.num_elements * 4; // f32 = 4 bytes
        let compressed_size = self.compressed_size();
        if original_size == 0 || compressed_size == 0 {
            return 1.0;
        }
        original_size as f32 / compressed_size as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    fn topk_reference_sparse(gradients: &[f32], k_ratio: f32) -> (Vec<u32>, Vec<u8>) {
        let raw = (gradients.len() as f32) * k_ratio;
        let mut k = raw as usize;
        if (k as f32) < raw {
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
                indices: vec![0, 1],      // Two indices
                values: vec![1, 0, 0, 0], // Four bytes = one value, length mismatch!
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
                values: vec![
                    1, 0, 0, 0, // 1.4e-45-ish placeholder bytes
                    2, 0, 0, 0, // would be OOB if applied at idx 99
                ],
            },
        };
        assert_eq!(
            compressed.decompress().unwrap_err(),
            CompressionError::IndexOutOfBounds
        );
    }

    #[test]
    fn decompress_rejects_misaligned_dense_data() {
        // Missing one byte to be modulo-4 aligned
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
        // Dense data paired with TopK method — no valid decode path
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

    /// NEW-06: RandomSparse is not yet implemented and returns UnsupportedMethod.
    #[test]
    fn compress_random_sparse_returns_unsupported() {
        let gradients = vec![1.0, 2.0, 3.0];
        let result = CompressedGradient::compress(
            &gradients,
            CompressionMethod::RandomSparse { p: 0.5, seed: 42 },
        );
        assert_eq!(result.unwrap_err(), CompressionError::UnsupportedMethod);
    }

    /// NEW-06: TopKQuantized is not yet implemented and returns UnsupportedMethod.
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

    /// NEW-06: decompress must also reject sparse data paired with RandomSparse method.
    #[test]
    fn decompress_sparse_random_sparse_returns_unsupported() {
        let compressed = CompressedGradient {
            method: CompressionMethod::RandomSparse { p: 0.5, seed: 42 },
            shape: vec![4],
            num_elements: 4,
            data: CompressedData::Sparse {
                indices: vec![0, 1],
                values: vec![0, 0, 128, 63, 0, 0, 0, 64], // 1.0f32, 2.0f32 in LE
            },
        };
        assert_eq!(
            compressed.decompress().unwrap_err(),
            CompressionError::UnsupportedMethod,
            "sparse data with RandomSparse method must be rejected on decompress"
        );
    }

    /// NEW-06: decompress must also reject sparse data paired with TopKQuantized method.
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
                values: vec![0, 0, 128, 63, 0, 0, 0, 64], // 1.0f32, 2.0f32 in LE
            },
        };
        assert_eq!(
            compressed.decompress().unwrap_err(),
            CompressionError::UnsupportedMethod,
            "sparse data with TopKQuantized method must be rejected on decompress"
        );
    }

    // L-03: compress rejects invalid scales
    #[test]
    fn compress_quantized_rejects_zero_scale() {
        let data = vec![1.0f32; 4];
        let result =
            CompressedGradient::compress(&data, CompressionMethod::Quantized { scale: 0.0 });
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
        let result =
            CompressedGradient::compress(&data, CompressionMethod::Quantized { scale: -1.0 });
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

    // L-03: decompress rejects crafted invalid scales
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
        // Absolute-value ties at the boundary stress deterministic selection.
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
                let (expected_indices, expected_values) =
                    topk_reference_sparse(&gradients, k_ratio);
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
}
