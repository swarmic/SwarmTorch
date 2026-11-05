//! Gradient compression and error-feedback utilities (`ADR-0020`, `W9-03`).
//!
//! Provides source-accurate wire validation for [`CompressedGradient`],
//! receiver-stateless [`ClassicResidualCompressor`] (default error feedback),
//! and synchronized-cohort [`Ef21SenderState`] / [`Ef21ReceiverState`] (Richtárik et al., 2021).

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

/// Default pre-allocation limit on `CompressedGradient::num_elements` (`16,777,216` elements = 64 MiB).
pub const DEFAULT_MAX_COMPRESSION_ELEMENTS: usize = 16_777_216;

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

/// Error type for compression and error-feedback operations (`ADR-0020`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionError {
    /// Index value exceeds `u32` capacity.
    IndexOverflow,
    /// Invalid compressed data format or sequence synchronization mismatch.
    InvalidData,
    /// Decompressed index out of bounds (`idx >= num_elements`).
    IndexOutOfBounds,
    /// Compression method is not yet implemented.
    UnsupportedMethod,
    /// Quantization scale is invalid (non-finite, zero, negative, or overflows `f32` on dequantize).
    InvalidScale,
    /// Sparsification ratio `k_ratio` is invalid (non-finite, `<= 0.0`, or `> 1.0`).
    InvalidRatio,
    /// Non-finite (`NaN` or `±Inf`) value encountered during compression or decompression.
    NonFiniteValue,
    /// `shape` product overflows or does not equal `num_elements`.
    ShapeMismatch,
    /// `num_elements` exceeds configured pre-allocation limit.
    ResourceLimitExceeded,
    /// Sparse indices are not strictly ascending (`indices[i] <= indices[i - 1]`).
    UnsortedOrDuplicateIndices,
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
                "quantization scale is invalid (non-finite, zero, negative, or overflows f32)"
            ),
            Self::InvalidRatio => write!(f, "sparsification ratio k_ratio must be in (0.0, 1.0]"),
            Self::NonFiniteValue => write!(f, "non-finite (NaN or Inf) gradient value detected"),
            Self::ShapeMismatch => write!(f, "shape product does not match num_elements"),
            Self::ResourceLimitExceeded => {
                write!(f, "num_elements exceeds pre-allocation resource limit")
            }
            Self::UnsortedOrDuplicateIndices => {
                write!(
                    f,
                    "sparse indices must be strictly ascending without duplicates"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CompressionError {}

/// Helper to convert `usize` length/index to `u32` safely.
pub fn try_usize_to_u32(val: usize) -> core::result::Result<u32, CompressionError> {
    u32::try_from(val).map_err(|_| CompressionError::IndexOverflow)
}

#[cfg(feature = "alloc")]
fn validate_method_params(
    method: &CompressionMethod,
) -> core::result::Result<(), CompressionError> {
    match method {
        CompressionMethod::None => Ok(()),
        CompressionMethod::TopK { k_ratio } => {
            if !k_ratio.is_finite() || *k_ratio <= 0.0 || *k_ratio > 1.0 {
                return Err(CompressionError::InvalidRatio);
            }
            Ok(())
        }
        CompressionMethod::Quantized { scale } => {
            if !scale.is_finite()
                || *scale <= 0.0
                || (128.0f64 * (*scale as f64)) > (f32::MAX as f64)
            {
                return Err(CompressionError::InvalidScale);
            }
            Ok(())
        }
        CompressionMethod::RandomSparse { .. } | CompressionMethod::TopKQuantized { .. } => {
            Err(CompressionError::UnsupportedMethod)
        }
    }
}

/// Compression method for gradients.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum CompressionMethod {
    /// No compression (full gradient).
    #[default]
    None,
    /// Top-K sparsification (send only top `k_ratio` fraction of gradient values).
    TopK {
        /// Ratio of values to keep in `(0.0, 1.0]` (e.g., `0.01` for 1%).
        k_ratio: f32,
    },
    /// Random sparsification with seed (unsupported in v0.1; returns `UnsupportedMethod`).
    RandomSparse {
        /// Probability of keeping each value.
        p: f32,
        /// Random seed for reproducibility.
        seed: u64,
    },
    /// Quantized to `i8` with scale.
    Quantized {
        /// Scale factor for dequantization (`> 0.0`, finite).
        scale: f32,
    },
    /// Combined: TopK + Quantization (unsupported in v0.1; returns `UnsupportedMethod`).
    TopKQuantized {
        /// Ratio of values to keep.
        k_ratio: f32,
        /// Scale factor for dequantization.
        scale: f32,
    },
}

/// Compressed gradient data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompressedData {
    /// Dense representation (full `f32` LE bytes or quantized `i8` bytes).
    #[cfg(feature = "alloc")]
    Dense(Vec<u8>),
    /// Sparse representation (strictly ascending indices + LE `f32` values).
    #[cfg(feature = "alloc")]
    Sparse {
        /// Strictly ascending indices of non-zero values.
        indices: Vec<u32>,
        /// Compressed `f32` values in little-endian byte order.
        values: Vec<u8>,
    },
}

/// A compressed gradient update.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompressedGradient {
    /// Compression method used.
    pub method: CompressionMethod,
    /// Original tensor shape (`product(shape) == num_elements`).
    #[cfg(feature = "alloc")]
    pub shape: Vec<usize>,
    /// Total number of elements.
    pub num_elements: usize,
    /// Compressed data.
    #[cfg(feature = "alloc")]
    pub data: CompressedData,
}

#[cfg(feature = "alloc")]
impl CompressedGradient {
    /// Validate pre-allocation `num_elements` bound and `product(shape) == num_elements`.
    pub fn validate_shape_and_bounds(
        &self,
        max_elements: usize,
    ) -> core::result::Result<(), CompressionError> {
        if self.num_elements > max_elements {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        if self.num_elements == 0 {
            if self.shape.is_empty() {
                return Ok(());
            }
            let prod = self
                .shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or(CompressionError::ShapeMismatch)?;
            if prod != 0 {
                return Err(CompressionError::ShapeMismatch);
            }
            return Ok(());
        }
        if self.shape.is_empty() {
            return Err(CompressionError::ShapeMismatch);
        }
        let prod = self
            .shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(CompressionError::ShapeMismatch)?;
        if prod != self.num_elements {
            return Err(CompressionError::ShapeMismatch);
        }
        Ok(())
    }

    /// Compress a gradient slice using the specified method.
    pub fn compress(
        gradients: &[f32],
        method: CompressionMethod,
    ) -> core::result::Result<Self, CompressionError> {
        validate_method_params(&method)?;
        if gradients.len() > DEFAULT_MAX_COMPRESSION_ELEMENTS {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        for &v in gradients {
            if !v.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
        }

        match method {
            CompressionMethod::None => {
                let bytes: Vec<u8> = gradients.iter().flat_map(|f| f.to_le_bytes()).collect();
                Ok(Self {
                    method,
                    shape: alloc::vec![gradients.len()],
                    num_elements: gradients.len(),
                    data: CompressedData::Dense(bytes),
                })
            }
            CompressionMethod::TopK { k_ratio } => {
                if gradients.is_empty() {
                    return Ok(Self {
                        method,
                        shape: alloc::vec![0],
                        num_elements: 0,
                        data: CompressedData::Sparse {
                            indices: Vec::new(),
                            values: Vec::new(),
                        },
                    });
                }

                let raw = (gradients.len() as f64) * (k_ratio as f64);
                let mut k = raw as usize;
                if (k as f64) < raw {
                    k = k.saturating_add(1);
                }
                let k = k.max(1).min(gradients.len());

                let mut indexed: Vec<(usize, f32)> =
                    gradients.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                if k < indexed.len() {
                    indexed.select_nth_unstable_by(k, topk_priority_cmp);
                    indexed.truncate(k);
                }

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
                let inv_scale = 1.0f64 / (scale as f64);
                let bytes: Vec<u8> = gradients
                    .iter()
                    .map(|&v| {
                        let quantized = ((v as f64) * inv_scale).clamp(-128.0, 127.0) as i8;
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
            CompressionMethod::RandomSparse { .. } | CompressionMethod::TopKQuantized { .. } => {
                Err(CompressionError::UnsupportedMethod)
            }
        }
    }

    /// Decompress back to a gradient vector enforcing [`DEFAULT_MAX_COMPRESSION_ELEMENTS`].
    pub fn decompress(&self) -> core::result::Result<Vec<f32>, CompressionError> {
        self.decompress_with_limit(DEFAULT_MAX_COMPRESSION_ELEMENTS)
    }

    /// Decompress back to a gradient vector with an explicit pre-allocation `max_elements` limit.
    pub fn decompress_with_limit(
        &self,
        max_elements: usize,
    ) -> core::result::Result<Vec<f32>, CompressionError> {
        validate_method_params(&self.method)?;
        self.validate_shape_and_bounds(max_elements)?;

        match (&self.data, &self.method) {
            (CompressedData::Dense(bytes), CompressionMethod::None) => {
                if bytes.len() % 4 != 0 || bytes.len() / 4 != self.num_elements {
                    return Err(CompressionError::InvalidData);
                }
                let mut out = Vec::with_capacity(self.num_elements);
                for chunk in bytes.chunks_exact(4) {
                    let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if !v.is_finite() {
                        return Err(CompressionError::NonFiniteValue);
                    }
                    out.push(v);
                }
                Ok(out)
            }
            (CompressedData::Dense(bytes), CompressionMethod::Quantized { scale }) => {
                if bytes.len() != self.num_elements {
                    return Err(CompressionError::InvalidData);
                }
                let scale_f64 = *scale as f64;
                let mut out = Vec::with_capacity(self.num_elements);
                for &b in bytes {
                    let val = ((b as i8 as f64) * scale_f64) as f32;
                    if !val.is_finite() {
                        return Err(CompressionError::NonFiniteValue);
                    }
                    out.push(val);
                }
                Ok(out)
            }
            (CompressedData::Sparse { indices, values }, CompressionMethod::TopK { .. }) => {
                if values.len() % 4 != 0
                    || values.len() / 4 != indices.len()
                    || indices.len() > self.num_elements
                {
                    return Err(CompressionError::InvalidData);
                }
                // Validate indices and values before allocating the dense output vector.
                let mut prev_idx: Option<u32> = None;
                for (&raw_idx, chunk) in indices.iter().zip(values.chunks_exact(4)) {
                    let idx = raw_idx as usize;
                    if idx >= self.num_elements {
                        return Err(CompressionError::IndexOutOfBounds);
                    }
                    if let Some(prev) = prev_idx {
                        if raw_idx <= prev {
                            return Err(CompressionError::UnsortedOrDuplicateIndices);
                        }
                    }
                    prev_idx = Some(raw_idx);
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    if !value.is_finite() {
                        return Err(CompressionError::NonFiniteValue);
                    }
                }

                let mut result = alloc::vec![0.0f32; self.num_elements];
                for (&raw_idx, chunk) in indices.iter().zip(values.chunks_exact(4)) {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    result[raw_idx as usize] = value;
                }
                Ok(result)
            }
            _ => Err(CompressionError::InvalidData),
        }
    }

    /// Get the compressed size in bytes.
    pub fn compressed_size(&self) -> usize {
        match &self.data {
            CompressedData::Dense(bytes) => bytes.len(),
            CompressedData::Sparse { indices, values } => {
                indices.len().saturating_mul(4).saturating_add(values.len())
            }
        }
    }

    /// Get the compression ratio (`original_bytes / compressed_bytes`).
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.num_elements.saturating_mul(4);
        let compressed_size = self.compressed_size();
        if original_size == 0 || compressed_size == 0 {
            return 1.0;
        }
        original_size as f32 / compressed_size as f32
    }
}

/// Default receiver-stateless error-feedback compressor (`ClassicResidual`, `ADR-0020`, `W9-03`).
///
/// Accumulates local compression error $e_{t+1} = (g_t + e_t) - \mathcal{C}(g_t + e_t)$ on the
/// sender while emitting standard self-contained [`CompressedGradient`] payloads that any
/// receiver can decompress without maintaining per-peer state.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ClassicResidualCompressor {
    residual: Vec<f32>,
    method: CompressionMethod,
}

#[cfg(feature = "alloc")]
impl ClassicResidualCompressor {
    /// Create a new `ClassicResidualCompressor` for a fixed gradient dimension.
    pub fn new(
        dim: usize,
        method: CompressionMethod,
    ) -> core::result::Result<Self, CompressionError> {
        validate_method_params(&method)?;
        if dim == 0 || dim > DEFAULT_MAX_COMPRESSION_ELEMENTS {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        Ok(Self {
            residual: alloc::vec![0.0f32; dim],
            method,
        })
    }

    /// Return a read-only view of the current sender residual vector $e_t$.
    pub fn residual(&self) -> &[f32] {
        &self.residual
    }

    /// Compress `gradient` with error feedback, updating `self.residual` transactionally
    /// only after compression and reconstruction succeed.
    pub fn compress_step(
        &mut self,
        gradient: &[f32],
    ) -> core::result::Result<CompressedGradient, CompressionError> {
        if gradient.len() != self.residual.len() {
            return Err(CompressionError::ShapeMismatch);
        }
        let mut compensated = Vec::with_capacity(gradient.len());
        for (&g, &r) in gradient.iter().zip(self.residual.iter()) {
            if !g.is_finite() || !r.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
            let u = ((g as f64) + (r as f64)) as f32;
            if !u.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
            compensated.push(u);
        }

        let compressed = CompressedGradient::compress(&compensated, self.method.clone())?;
        let reconstructed = compressed.decompress()?;

        for ((slot, &u), &rec) in self
            .residual
            .iter_mut()
            .zip(compensated.iter())
            .zip(reconstructed.iter())
        {
            let next_res = ((u as f64) - (rec as f64)) as f32;
            if !next_res.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
            *slot = next_res;
        }

        Ok(compressed)
    }
}

/// Synchronized-cohort EF21 sender state (Richtárik et al., NeurIPS 2021; `ADR-0020`, `W9-03`).
///
/// Requires reliable in-order delivery of every step to a paired [`Ef21ReceiverState`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct Ef21SenderState {
    estimate: Vec<f32>,
    sequence: u64,
    method: CompressionMethod,
}

#[cfg(feature = "alloc")]
impl Ef21SenderState {
    /// Create a new EF21 sender state initialized at zero (`sequence = 0`).
    pub fn new(
        dim: usize,
        method: CompressionMethod,
    ) -> core::result::Result<Self, CompressionError> {
        validate_method_params(&method)?;
        if dim == 0 || dim > DEFAULT_MAX_COMPRESSION_ELEMENTS {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        Ok(Self {
            estimate: alloc::vec![0.0f32; dim],
            sequence: 0,
            method,
        })
    }

    /// Current sender estimate $v_t$.
    pub fn estimate(&self) -> &[f32] {
        &self.estimate
    }

    /// Next expected sequence number.
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Compress the difference $g_t - v_t$, update $v_{t+1} = v_t + \mathcal{C}(g_t - v_t)$
    /// transactionally, and return `(sequence, compressed_delta)`.
    pub fn step(
        &mut self,
        gradient: &[f32],
    ) -> core::result::Result<(u64, CompressedGradient), CompressionError> {
        if gradient.len() != self.estimate.len() {
            return Err(CompressionError::ShapeMismatch);
        }
        let mut diff = Vec::with_capacity(gradient.len());
        for (&g, &est) in gradient.iter().zip(self.estimate.iter()) {
            if !g.is_finite() || !est.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
            let d = ((g as f64) - (est as f64)) as f32;
            if !d.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
            diff.push(d);
        }

        let compressed_delta = CompressedGradient::compress(&diff, self.method.clone())?;
        let reconstructed_delta = compressed_delta.decompress()?;

        let mut next_estimate = Vec::with_capacity(self.estimate.len());
        for (&est, &delta) in self.estimate.iter().zip(reconstructed_delta.iter()) {
            let next = ((est as f64) + (delta as f64)) as f32;
            if !next.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
            next_estimate.push(next);
        }

        let seq = self.sequence;
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or(CompressionError::IndexOverflow)?;
        self.estimate = next_estimate;
        Ok((seq, compressed_delta))
    }
}

/// Synchronized-cohort EF21 receiver state (`ADR-0020`, `W9-03`).
///
/// Enforces strict sequential step application (`sequence == expected_sequence`) so dropped or
/// reordered packets fail closed with [`CompressionError::InvalidData`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct Ef21ReceiverState {
    estimate: Vec<f32>,
    expected_sequence: u64,
}

#[cfg(feature = "alloc")]
impl Ef21ReceiverState {
    /// Create a new EF21 receiver state initialized at zero (`expected_sequence = 0`).
    pub fn new(dim: usize) -> core::result::Result<Self, CompressionError> {
        if dim == 0 || dim > DEFAULT_MAX_COMPRESSION_ELEMENTS {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        Ok(Self {
            estimate: alloc::vec![0.0f32; dim],
            expected_sequence: 0,
        })
    }

    /// Current reconstructed gradient estimate $v_t$.
    pub fn estimate(&self) -> &[f32] {
        &self.estimate
    }

    /// Next expected sequence number.
    pub const fn expected_sequence(&self) -> u64 {
        self.expected_sequence
    }

    /// Apply an incoming compressed delta packet for `sequence`, updating and returning $v_{t+1}$.
    pub fn apply_step(
        &mut self,
        sequence: u64,
        compressed_delta: &CompressedGradient,
    ) -> core::result::Result<&[f32], CompressionError> {
        if sequence != self.expected_sequence {
            return Err(CompressionError::InvalidData);
        }
        let delta = compressed_delta.decompress()?;
        if delta.len() != self.estimate.len() {
            return Err(CompressionError::ShapeMismatch);
        }

        let mut next_estimate = Vec::with_capacity(self.estimate.len());
        for (&est, &d) in self.estimate.iter().zip(delta.iter()) {
            let next = ((est as f64) + (d as f64)) as f32;
            if !next.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
            next_estimate.push(next);
        }

        self.expected_sequence = self
            .expected_sequence
            .checked_add(1)
            .ok_or(CompressionError::IndexOverflow)?;
        self.estimate = next_estimate;
        Ok(&self.estimate)
    }
}

#[cfg(all(test, feature = "alloc"))]
#[path = "compression_tests.rs"]
mod tests;
