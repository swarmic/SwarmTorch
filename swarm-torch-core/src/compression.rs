//! Gradient compression utilities
//!
//! This module provides compression algorithms for efficient gradient
//! transmission over bandwidth-constrained networks.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

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
    pub fn compress(gradients: &[f32], method: CompressionMethod) -> Self {
        match method {
            CompressionMethod::None => {
                // Convert f32 to bytes
                let bytes: Vec<u8> = gradients.iter().flat_map(|f| f.to_le_bytes()).collect();
                Self {
                    method,
                    shape: alloc::vec![gradients.len()],
                    num_elements: gradients.len(),
                    data: CompressedData::Dense(bytes),
                }
            }
            CompressionMethod::TopK { k_ratio } => {
                // Avoid `f32::ceil()` so `no_std + alloc` builds don't require libm.
                let raw = (gradients.len() as f32) * k_ratio;
                let mut k = raw as usize;
                if (k as f32) < raw {
                    k = k.saturating_add(1);
                }
                let k = k.max(1).min(gradients.len());

                // Find top-k by absolute value
                let mut indexed: Vec<(usize, f32)> =
                    gradients.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| {
                    b.1.abs()
                        .partial_cmp(&a.1.abs())
                        .unwrap_or(core::cmp::Ordering::Equal)
                });

                let top_k: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
                let indices: Vec<u32> = top_k.iter().map(|(i, _)| *i as u32).collect();
                let values: Vec<u8> = top_k.iter().flat_map(|(_, v)| v.to_le_bytes()).collect();

                Self {
                    method,
                    shape: alloc::vec![gradients.len()],
                    num_elements: gradients.len(),
                    data: CompressedData::Sparse { indices, values },
                }
            }
            CompressionMethod::Quantized { scale } => {
                // Quantize to INT8
                let bytes: Vec<u8> = gradients
                    .iter()
                    .map(|&v| {
                        let quantized = (v / scale).clamp(-128.0, 127.0) as i8;
                        quantized as u8
                    })
                    .collect();
                Self {
                    method: CompressionMethod::Quantized { scale },
                    shape: alloc::vec![gradients.len()],
                    num_elements: gradients.len(),
                    data: CompressedData::Dense(bytes),
                }
            }
            _ => {
                // Fallback to no compression for other methods
                Self::compress(gradients, CompressionMethod::None)
            }
        }
    }

    /// Decompress back to a gradient vector
    pub fn decompress(&self) -> Vec<f32> {
        match (&self.data, &self.method) {
            (CompressedData::Dense(bytes), CompressionMethod::None) => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            (CompressedData::Dense(bytes), CompressionMethod::Quantized { scale }) => {
                bytes.iter().map(|&b| (b as i8 as f32) * scale).collect()
            }
            (CompressedData::Sparse { indices, values }, _) => {
                let mut result = alloc::vec![0.0f32; self.num_elements];
                for (idx, chunk) in indices.iter().zip(values.chunks_exact(4)) {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let idx = *idx as usize;
                    if idx < result.len() {
                        result[idx] = value;
                    }
                }
                result
            }
            _ => alloc::vec![0.0f32; self.num_elements],
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

    #[test]
    fn decompress_sparse_ignores_out_of_bounds_indices() {
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
        let decompressed = compressed.decompress();
        assert_eq!(decompressed.len(), 2);
    }

    #[test]
    fn compression_ratio_is_well_defined_for_empty_input() {
        let compressed = CompressedGradient::compress(&[], CompressionMethod::None);
        assert_eq!(compressed.compression_ratio(), 1.0);
    }
}
