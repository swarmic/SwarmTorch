//! Gradient compression and error-feedback utilities (`ADR-0020`, `W9-03`).
//!
//! Provides source-accurate wire validation for [`CompressedGradient`],
//! receiver-stateless [`ClassicResidualCompressor`] (default error feedback),
//! and synchronized-cohort [`Ef21SenderState`] / [`Ef21ReceiverState`] (Richtárik et al., 2021).

#[cfg(feature = "alloc")]
use alloc::collections::BTreeSet;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};
#[cfg(feature = "alloc")]
use sha2::{Digest, Sha256};

/// Default pre-allocation limit on `CompressedGradient::num_elements` (`16,777,216` elements = 64 MiB).
pub const DEFAULT_MAX_COMPRESSION_ELEMENTS: usize = 16_777_216;

/// Domain separation tag for `SharedSeedSparse` deterministic subset sampling (`ADR-0021` §3, `W11-02`).
pub const SHARED_SPARSE_SEED_DOMAIN_V1: &[u8] = b"swarmtorch.shared_sparse.seed.v1";

/// Domain separation tag for pre-seed Merkle coordinate leaf hashing (`ADR-0021` §3, `W11-02`).
pub const MERKLE_LEAF_DOMAIN_V1: &[u8] = b"swarmtorch.merkle.leaf.v1";

/// Domain separation tag for power-of-two virtual padding leaves (`ADR-0021` §3, `W11-02`).
pub const MERKLE_PAD_DOMAIN_V1: &[u8] = b"swarmtorch.merkle.pad.v1";

/// Domain separation tag for internal Merkle tree nodes (`ADR-0021` §3, `W11-02`).
pub const MERKLE_NODE_DOMAIN_V1: &[u8] = b"swarmtorch.merkle.node.v1";

/// Domain separation tag for binding a round seed to cohort pre-seed commitments (`ADR-0021` §3, `W11-02`).
pub const VERIFIABLE_ROUND_SEED_DOMAIN_V1: &[u8] = b"swarmtorch.round.seed.v1";

#[cfg(feature = "alloc")]
fn topk_priority_cmp(a: &(usize, f32), b: &(usize, f32)) -> core::cmp::Ordering {
    // Deterministic total order:
    // 1) descending |value| (largest first)
    // 2) ascending index (tie-breaker)
    let abs_cmp = crate::num::abs_f32(b.1).total_cmp(&crate::num::abs_f32(a.1));
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
    /// Pre-seed Merkle vector commitment or coordinate opening proof verification failed (`ADR-0021` §3).
    CommitmentVerificationFailed,
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
            Self::CommitmentVerificationFailed => {
                write!(
                    f,
                    "pre-seed Merkle commitment or coordinate opening verification failed"
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

#[cfg(feature = "alloc")]
fn canonical_f32_bits(v: f32) -> core::result::Result<u32, CompressionError> {
    if !v.is_finite() {
        return Err(CompressionError::NonFiniteValue);
    }
    if v == 0.0 {
        Ok(0x0000_0000)
    } else {
        Ok(v.to_bits())
    }
}

#[cfg(feature = "alloc")]
struct SharedSparseWordStream {
    seed: [u8; 32],
    d: u64,
    k: u64,
    counter: u64,
    block: [u8; 32],
    offset: usize,
}

#[cfg(feature = "alloc")]
impl SharedSparseWordStream {
    fn new(seed: [u8; 32], d: usize, k: usize) -> Self {
        Self {
            seed,
            d: d as u64,
            k: k as u64,
            counter: 0,
            block: [0u8; 32],
            offset: 32,
        }
    }

    fn next_u32(&mut self) -> u32 {
        if self.offset + 4 > 32 {
            let mut hasher = Sha256::new();
            hasher.update(SHARED_SPARSE_SEED_DOMAIN_V1);
            hasher.update(self.seed);
            hasher.update(self.d.to_le_bytes());
            hasher.update(self.k.to_le_bytes());
            hasher.update(self.counter.to_le_bytes());
            self.counter = self.counter.wrapping_add(1);
            let digest = hasher.finalize();
            self.block.copy_from_slice(&digest);
            self.offset = 0;
        }
        let word = u32::from_le_bytes([
            self.block[self.offset],
            self.block[self.offset + 1],
            self.block[self.offset + 2],
            self.block[self.offset + 3],
        ]);
        self.offset += 4;
        word
    }

    /// Unbiased integer in `0..=inclusive_max` via Lemire's divisionless rejection method.
    fn uniform_u32_inclusive(&mut self, inclusive_max: u32) -> u32 {
        let bound = (inclusive_max as u64) + 1;
        if bound == 1 {
            return 0;
        }
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let x = self.next_u32() as u64;
            let m = x * bound;
            let l = m as u32 as u64;
            if l >= threshold {
                return (m >> 32) as u32;
            }
        }
    }
}

/// Sample an unbiased $k$-of-$d$ coordinate subset in $\mathcal{O}(k \log k)$ time and
/// $\mathcal{O}(k)$ auxiliary memory using Robert Floyd's selection algorithm (`ADR-0021` §3, `W11-02`).
///
/// Unlike a partial Fisher-Yates shuffle over `[0..d]`, this function never allocates an
/// $\mathcal{O}(d)$ index buffer (`F-04`) and returns the $k$ indices in strictly ascending
/// canonical order.
#[cfg(feature = "alloc")]
pub fn sample_shared_indices_v1(
    d: usize,
    k: usize,
    seed: &[u8; 32],
) -> core::result::Result<Vec<u32>, CompressionError> {
    if d == 0 || d > DEFAULT_MAX_COMPRESSION_ELEMENTS {
        return Err(CompressionError::ResourceLimitExceeded);
    }
    if d > u32::MAX as usize {
        return Err(CompressionError::IndexOverflow);
    }
    if k == 0 || k > d {
        return Err(CompressionError::InvalidRatio);
    }

    let mut selected = BTreeSet::new();
    let mut rng = SharedSparseWordStream::new(*seed, d, k);

    for j in (d - k)..d {
        let t = rng.uniform_u32_inclusive(j as u32);
        if !selected.insert(t) {
            selected.insert(j as u32);
        }
    }

    Ok(selected.into_iter().collect())
}

/// Projection scaling profile for [`SharedSeedSparsePayloadV1`] (`ADR-0021` §3, `W11-02`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SharedSeedScalingProfile {
    /// Unscaled projection $C_S(x) = P_S x$. Required when paired with local error-feedback
    /// residual accumulation (`ADR-0020` / [`SharedSeedResidualCompressor`]) so the contraction
    /// bound $\|x - C_S(x)\|_2^2 \le (1 - k/d)\|x\|_2^2$ holds in expectation.
    #[default]
    UnscaledProjectionV1,
    /// Unbiased scaled projection $C_S(x) = \frac{d}{k} P_S x$ ($\mathbb{E}[C_S(x)] = x$).
    /// Use only when local residual accumulation is disabled.
    ScaledUnbiasedV1,
}

/// Compact wire payload for `SharedSeedSparse` (`ADR-0021` §3, `W11-02`).
///
/// Because all peers and the coordinator derive the identical ascending index subset
/// `S = sample_shared_indices_v1(original_len, k, &seed)`, the wire payload carries only
/// the $k$ retained `f32` values (`4 * k` bytes) without per-element `u32` index overhead.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedSeedSparsePayloadV1 {
    /// Full uncompressed vector dimension $d$.
    pub original_len: u32,
    /// Number of retained coordinates $k$ ($1 \le k \le d$).
    pub k: u32,
    /// Shared 32-byte round seed determining the $k$-of-$d$ support $S$.
    pub seed: [u8; 32],
    /// Projection scaling profile applied to `values`.
    pub scaling: SharedSeedScalingProfile,
    /// Retained coordinate values ordered by `sample_shared_indices_v1(original_len, k, &seed)`.
    pub values: Vec<f32>,
}

#[cfg(feature = "alloc")]
impl SharedSeedSparsePayloadV1 {
    /// Project a dense $d$-dimensional vector onto the shared-seed $k$-coordinate subspace.
    pub fn compress(
        gradient: &[f32],
        k: usize,
        seed: [u8; 32],
        scaling: SharedSeedScalingProfile,
    ) -> core::result::Result<Self, CompressionError> {
        let d = gradient.len();
        for &v in gradient {
            if !v.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
        }
        let indices = sample_shared_indices_v1(d, k, &seed)?;
        let scale_f64 = match scaling {
            SharedSeedScalingProfile::UnscaledProjectionV1 => 1.0f64,
            SharedSeedScalingProfile::ScaledUnbiasedV1 => (d as f64) / (k as f64),
        };

        let mut values = Vec::with_capacity(k);
        for idx in indices {
            let raw = gradient[idx as usize];
            let projected = if scaling == SharedSeedScalingProfile::UnscaledProjectionV1 {
                raw
            } else {
                let s = ((raw as f64) * scale_f64) as f32;
                if !s.is_finite() {
                    return Err(CompressionError::NonFiniteValue);
                }
                s
            };
            values.push(projected);
        }

        Ok(Self {
            original_len: d as u32,
            k: k as u32,
            seed,
            scaling,
            values,
        })
    }

    /// Reconstruct the $d$-dimensional sparse vector by scattering `values` into `sample_shared_indices_v1`.
    pub fn decompress(&self) -> core::result::Result<Vec<f32>, CompressionError> {
        let d = self.original_len as usize;
        let k = self.k as usize;
        let indices = sample_shared_indices_v1(d, k, &self.seed)?;
        if self.values.len() != k {
            return Err(CompressionError::InvalidData);
        }
        for &v in &self.values {
            if !v.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
        }

        let mut out = alloc::vec![0.0f32; d];
        for (idx, &val) in indices.into_iter().zip(self.values.iter()) {
            out[idx as usize] = val;
        }
        Ok(out)
    }
}

#[cfg(feature = "alloc")]
fn merkle_leaf_hash_v1(
    index: u32,
    raw_value: f32,
) -> core::result::Result<[u8; 32], CompressionError> {
    let bits = canonical_f32_bits(raw_value)?;
    let mut hasher = Sha256::new();
    hasher.update(MERKLE_LEAF_DOMAIN_V1);
    hasher.update(index.to_le_bytes());
    hasher.update(bits.to_le_bytes());
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    Ok(out)
}

#[cfg(feature = "alloc")]
fn merkle_pad_hash_v1(index: u32) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(MERKLE_PAD_DOMAIN_V1);
    hasher.update(index.to_le_bytes());
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

#[cfg(feature = "alloc")]
fn merkle_node_hash_v1(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(MERKLE_NODE_DOMAIN_V1);
    hasher.update(left);
    hasher.update(right);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Pre-seed binary SHA-256 Merkle vector commitment (`ADR-0021` §3, `W11-02`).
///
/// Published and signed by each cohort peer *before* the round's shared seed is revealed so a
/// Byzantine peer cannot adaptively fabricate coordinates on the $k$ sampled indices (`F-04`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreSeedVectorCommitmentV1 {
    /// Full uncompressed vector dimension $d$.
    pub original_len: u32,
    /// Binary SHA-256 Merkle root over the $d$ coordinates (padded to $\max(2, \text{next\_power\_of\_two}(d))$).
    pub merkle_root: [u8; 32],
}

/// Single coordinate opening proof against a [`PreSeedVectorCommitmentV1`] (`ADR-0021` §3, `W11-02`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MerkleCoordinateProofV1 {
    /// Coordinate index $j \in [0, d)$.
    pub index: u32,
    /// Unscaled pre-seed coordinate value $u_t^{(i, j)}$ committed at leaf $j$.
    pub raw_value: f32,
    /// Sibling hashes from leaf level (`level = 0`) to the child of `merkle_root`.
    pub siblings: Vec<[u8; 32]>,
}

/// Batch of $k$ Merkle coordinate opening proofs accompanying a [`SharedSeedSparsePayloadV1`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MerkleOpeningBatchV1 {
    /// Opening proofs in canonical ascending index order of `sample_shared_indices_v1`.
    pub proofs: Vec<MerkleCoordinateProofV1>,
}

#[cfg(feature = "alloc")]
impl PreSeedVectorCommitmentV1 {
    /// Build the full Merkle tree layers (leaf layer at index `0`, root at the last layer)
    /// for a finite $d$-dimensional vector.
    fn build_tree_layers(
        vector: &[f32],
    ) -> core::result::Result<Vec<Vec<[u8; 32]>>, CompressionError> {
        let d = vector.len();
        if d == 0 || d > DEFAULT_MAX_COMPRESSION_ELEMENTS {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        if d > u32::MAX as usize {
            return Err(CompressionError::IndexOverflow);
        }
        let leaf_count = d
            .checked_next_power_of_two()
            .ok_or(CompressionError::ResourceLimitExceeded)?
            .max(2);

        let mut current = Vec::with_capacity(leaf_count);
        for (idx, &val) in vector.iter().enumerate() {
            current.push(merkle_leaf_hash_v1(idx as u32, val)?);
        }
        for idx in d..leaf_count {
            current.push(merkle_pad_hash_v1(idx as u32));
        }

        let mut layers = Vec::new();
        while current.len() > 1 {
            let mut next = Vec::with_capacity(current.len() / 2);
            for pair in current.chunks_exact(2) {
                next.push(merkle_node_hash_v1(&pair[0], &pair[1]));
            }
            layers.push(current);
            current = next;
        }
        layers.push(current);
        Ok(layers)
    }

    /// Compute the pre-seed Merkle commitment for `vector` before the shared seed is revealed.
    pub fn commit(vector: &[f32]) -> core::result::Result<Self, CompressionError> {
        let layers = Self::build_tree_layers(vector)?;
        let root = layers
            .last()
            .and_then(|top| top.first())
            .copied()
            .ok_or(CompressionError::InvalidData)?;
        Ok(Self {
            original_len: vector.len() as u32,
            merkle_root: root,
        })
    }

    /// Produce the [`SharedSeedSparsePayloadV1`] and [`MerkleOpeningBatchV1`] for a previously
    /// committed pre-seed `vector` once the round `seed` is revealed.
    pub fn open_for_seed(
        &self,
        vector: &[f32],
        k: usize,
        seed: [u8; 32],
        scaling: SharedSeedScalingProfile,
    ) -> core::result::Result<(SharedSeedSparsePayloadV1, MerkleOpeningBatchV1), CompressionError>
    {
        if vector.len() != self.original_len as usize {
            return Err(CompressionError::ShapeMismatch);
        }
        let layers = Self::build_tree_layers(vector)?;
        let root = layers
            .last()
            .and_then(|top| top.first())
            .copied()
            .ok_or(CompressionError::InvalidData)?;
        if root != self.merkle_root {
            return Err(CompressionError::CommitmentVerificationFailed);
        }

        let payload = SharedSeedSparsePayloadV1::compress(vector, k, seed, scaling)?;
        let indices = sample_shared_indices_v1(vector.len(), k, &seed)?;
        let tree_depth = layers.len() - 1;
        let mut proofs = Vec::with_capacity(k);

        for idx in indices {
            let mut siblings = Vec::with_capacity(tree_depth);
            let mut pos = idx as usize;
            for layer in layers.iter().take(tree_depth) {
                let sibling_pos = pos ^ 1;
                siblings.push(layer[sibling_pos]);
                pos >>= 1;
            }
            proofs.push(MerkleCoordinateProofV1 {
                index: idx,
                raw_value: vector[idx as usize],
                siblings,
            });
        }

        Ok((payload, MerkleOpeningBatchV1 { proofs }))
    }

    /// Commit to `vector` and open the $k$ coordinates for `seed` in one call.
    pub fn commit_and_open(
        vector: &[f32],
        k: usize,
        seed: [u8; 32],
        scaling: SharedSeedScalingProfile,
    ) -> core::result::Result<
        (Self, SharedSeedSparsePayloadV1, MerkleOpeningBatchV1),
        CompressionError,
    > {
        let commitment = Self::commit(vector)?;
        let (payload, openings) = commitment.open_for_seed(vector, k, seed, scaling)?;
        Ok((commitment, payload, openings))
    }
}

/// Verify that a [`SharedSeedSparsePayloadV1`] and its [`MerkleOpeningBatchV1`] match a peer's
/// pre-seed [`PreSeedVectorCommitmentV1`] (`ADR-0021` §3, `W11-02`).
#[cfg(feature = "alloc")]
pub fn verify_shared_seed_openings_v1(
    commitment: &PreSeedVectorCommitmentV1,
    payload: &SharedSeedSparsePayloadV1,
    openings: &MerkleOpeningBatchV1,
) -> core::result::Result<(), CompressionError> {
    if commitment.original_len != payload.original_len {
        return Err(CompressionError::CommitmentVerificationFailed);
    }
    let d = payload.original_len as usize;
    let k = payload.k as usize;
    let expected_indices = sample_shared_indices_v1(d, k, &payload.seed)?;

    if payload.values.len() != k || openings.proofs.len() != k {
        return Err(CompressionError::CommitmentVerificationFailed);
    }

    let leaf_count = d
        .checked_next_power_of_two()
        .ok_or(CompressionError::ResourceLimitExceeded)?
        .max(2);
    let expected_depth = leaf_count.trailing_zeros() as usize;
    let scale_f64 = match payload.scaling {
        SharedSeedScalingProfile::UnscaledProjectionV1 => 1.0f64,
        SharedSeedScalingProfile::ScaledUnbiasedV1 => (d as f64) / (k as f64),
    };

    for ((&expected_idx, &wire_val), proof) in expected_indices
        .iter()
        .zip(payload.values.iter())
        .zip(openings.proofs.iter())
    {
        if proof.index != expected_idx || proof.siblings.len() != expected_depth {
            return Err(CompressionError::CommitmentVerificationFailed);
        }

        let expected_wire_val = if payload.scaling == SharedSeedScalingProfile::UnscaledProjectionV1
        {
            proof.raw_value
        } else {
            let s = ((proof.raw_value as f64) * scale_f64) as f32;
            if !s.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
            s
        };

        if canonical_f32_bits(wire_val)? != canonical_f32_bits(expected_wire_val)? {
            return Err(CompressionError::CommitmentVerificationFailed);
        }

        let mut current = merkle_leaf_hash_v1(proof.index, proof.raw_value)?;
        let mut pos = proof.index as usize;
        for sibling in &proof.siblings {
            if (pos & 1) == 0 {
                current = merkle_node_hash_v1(&current, sibling);
            } else {
                current = merkle_node_hash_v1(sibling, &current);
            }
            pos >>= 1;
        }

        if current != commitment.merkle_root {
            return Err(CompressionError::CommitmentVerificationFailed);
        }
    }

    Ok(())
}

/// Derive a verifiable 32-byte round seed cryptographically bound to the cohort's pre-seed
/// [`PreSeedVectorCommitmentV1`] roots and an optional external randomness beacon (`ADR-0021` §3).
#[cfg(feature = "alloc")]
pub fn derive_verifiable_round_seed_v1(
    run_id: &[u8; 16],
    round_id: u64,
    commitments: &[(crate::traits::PeerId, PreSeedVectorCommitmentV1)],
    external_beacon: &[u8; 32],
) -> core::result::Result<[u8; 32], CompressionError> {
    if commitments.is_empty() {
        return Err(CompressionError::InvalidData);
    }

    let mut sorted = commitments.to_vec();
    sorted.sort_by_key(|a| a.0);
    for window in sorted.windows(2) {
        if window[0].0 == window[1].0 {
            return Err(CompressionError::InvalidData);
        }
    }

    let mut hasher = Sha256::new();
    hasher.update(VERIFIABLE_ROUND_SEED_DOMAIN_V1);
    hasher.update(run_id);
    hasher.update(round_id.to_le_bytes());
    hasher.update((sorted.len() as u32).to_le_bytes());
    for (peer_id, commitment) in &sorted {
        hasher.update(peer_id.as_bytes());
        hasher.update(commitment.original_len.to_le_bytes());
        hasher.update(commitment.merkle_root);
    }
    hasher.update(external_beacon);

    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    Ok(out)
}

/// Local error-feedback residual state co-designed for `SharedSeedSparse` (`ADR-0021` §3, `W11-02`).
///
/// Implements the two-phase verifiable commitment workflow:
/// 1. [`Self::prepare_pre_seed_vector`] forms $u_t = g_t + e_t$ and computes its
///    [`PreSeedVectorCommitmentV1`] *before* the round seed is revealed.
/// 2. [`Self::finalize_with_seed`] projects $u_t$ onto the shared-seed $k$-coordinate support $S$
///    using [`SharedSeedScalingProfile::UnscaledProjectionV1`], updates the local residual
///    $e_{t+1} = u_t - P_S(u_t)$ transactionally, and emits `(payload, openings)`.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct SharedSeedResidualCompressor {
    residual: Vec<f32>,
    k: usize,
}

#[cfg(feature = "alloc")]
impl SharedSeedResidualCompressor {
    /// Create a new residual compressor for dimension `dim` retaining `k` coordinates per round.
    pub fn new(dim: usize, k: usize) -> core::result::Result<Self, CompressionError> {
        if dim == 0 || dim > DEFAULT_MAX_COMPRESSION_ELEMENTS {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        if k == 0 || k > dim {
            return Err(CompressionError::InvalidRatio);
        }
        Ok(Self {
            residual: alloc::vec![0.0f32; dim],
            k,
        })
    }

    /// Borrow the current residual vector $e_t$.
    pub fn residual(&self) -> &[f32] {
        &self.residual
    }

    /// Phase 1 (Pre-Seed): Compute $u_t = g_t + e_t$ in widened `f64` and commit to $u_t$ via
    /// [`PreSeedVectorCommitmentV1`].
    pub fn prepare_pre_seed_vector(
        &self,
        gradient: &[f32],
    ) -> core::result::Result<(Vec<f32>, PreSeedVectorCommitmentV1), CompressionError> {
        if gradient.len() != self.residual.len() {
            return Err(CompressionError::ShapeMismatch);
        }
        let mut u = Vec::with_capacity(gradient.len());
        for (&g, &r) in gradient.iter().zip(self.residual.iter()) {
            if !g.is_finite() || !r.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
            let sum = ((g as f64) + (r as f64)) as f32;
            if !sum.is_finite() {
                return Err(CompressionError::NonFiniteValue);
            }
            u.push(sum);
        }
        let commitment = PreSeedVectorCommitmentV1::commit(&u)?;
        Ok((u, commitment))
    }

    /// Phase 2 (Post-Seed): Open the $k$ coordinates selected by `seed` from `pre_seed_u` and
    /// update the local residual $e_{t+1} = u_t - P_S(u_t)$.
    pub fn finalize_with_seed(
        &mut self,
        pre_seed_u: &[f32],
        commitment: &PreSeedVectorCommitmentV1,
        seed: [u8; 32],
    ) -> core::result::Result<(SharedSeedSparsePayloadV1, MerkleOpeningBatchV1), CompressionError>
    {
        if pre_seed_u.len() != self.residual.len() {
            return Err(CompressionError::ShapeMismatch);
        }
        let (payload, openings) = commitment.open_for_seed(
            pre_seed_u,
            self.k,
            seed,
            SharedSeedScalingProfile::UnscaledProjectionV1,
        )?;
        let indices = sample_shared_indices_v1(pre_seed_u.len(), self.k, &seed)?;

        let mut next_residual = pre_seed_u.to_vec();
        for idx in indices {
            next_residual[idx as usize] = 0.0;
        }
        self.residual = next_residual;
        Ok((payload, openings))
    }
}

/// Domain separation tag for canonical [`SparseRoundContextV1`] hashing (`ST-12`).
pub const SPARSE_ROUND_CONTEXT_DOMAIN_V1: &[u8] = b"swarmtorch.sparse_round.context.v1";

/// Domain separation tag for Phase-1 pre-seed commitment transcript hashing (`ST-12`).
pub const COMMITMENTS_ROOT_DOMAIN_V1: &[u8] = b"swarmtorch.shared_sparse.commitments_root.v1";

/// Domain separation tag for collective post-commitment `SharedSeedSparse` seed derivation (`ST-12`).
pub const COLLECTIVE_ROUND_SEED_DOMAIN_V1: &[u8] = b"swarmtorch.shared_sparse.collective_seed.v1";

/// Version identifier for [`sample_shared_indices_v1`] deterministic coordinate sampling (`ST-12`).
pub const SHARED_SEED_SAMPLER_VERSION_V1: u16 = 1;

/// Frozen cohort context published before Phase-1 pre-seed commitments begin (`ST-12`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseRoundContextV1 {
    /// Scoped swarm deployment / cohort instance identifier.
    pub swarm_instance_id: [u8; 16],
    /// Bound training/coordination run identifier.
    pub run_id: [u8; 16],
    /// Round identifier for which this sparse projection context is valid.
    pub round_id: u64,
    /// Expected parent checkpoint SHA-256 content digest for `round_id`.
    pub base_checkpoint_digest: [u8; 32],
    /// Frozen set of peers authorized to commit and open in this round.
    pub frozen_roster: BTreeSet<crate::traits::PeerId>,
    /// Original dense vector dimension $d$.
    pub original_len: u32,
    /// Shared projection support cardinality $k \in [1, d]$.
    pub sparsity_k: u32,
    /// Projection scaling profile shared identically by all contributors in this round.
    pub scaling: SharedSeedScalingProfile,
    /// Sampler version (`SHARED_SEED_SAMPLER_VERSION_V1 = 1`).
    pub sampler_version: u16,
}

#[cfg(feature = "alloc")]
impl SparseRoundContextV1 {
    /// Validate invariants of this [`SparseRoundContextV1`].
    pub fn validate(&self) -> core::result::Result<(), CompressionError> {
        let d = self.original_len as usize;
        let k = self.sparsity_k as usize;
        if d == 0 || d > DEFAULT_MAX_COMPRESSION_ELEMENTS {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        if k == 0 || k > d {
            return Err(CompressionError::InvalidRatio);
        }
        if self.frozen_roster.is_empty() {
            return Err(CompressionError::InvalidData);
        }
        if self.sampler_version != SHARED_SEED_SAMPLER_VERSION_V1 {
            return Err(CompressionError::UnsupportedMethod);
        }
        Ok(())
    }

    /// Compute the canonical SHA-256 digest of this validated [`SparseRoundContextV1`].
    pub fn context_digest(&self) -> core::result::Result<[u8; 32], CompressionError> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(SPARSE_ROUND_CONTEXT_DOMAIN_V1);
        hasher.update(self.swarm_instance_id);
        hasher.update(self.run_id);
        hasher.update(self.round_id.to_le_bytes());
        hasher.update(self.base_checkpoint_digest);
        hasher.update((self.frozen_roster.len() as u32).to_le_bytes());
        for peer in &self.frozen_roster {
            hasher.update(peer.as_bytes());
        }
        hasher.update(self.original_len.to_le_bytes());
        hasher.update(self.sparsity_k.to_le_bytes());
        let scaling_tag: u8 = match self.scaling {
            SharedSeedScalingProfile::UnscaledProjectionV1 => 1,
            SharedSeedScalingProfile::ScaledUnbiasedV1 => 2,
        };
        hasher.update([scaling_tag]);
        hasher.update(self.sampler_version.to_le_bytes());
        Ok(hasher.finalize().into())
    }
}

/// Authenticated Phase-1 wire payload binding a peer's [`PreSeedVectorCommitmentV1`] to
/// `context_digest` (`ST-12`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedSeedCommitmentMessageV1 {
    /// Canonical [`SparseRoundContextV1::context_digest`].
    pub context_digest: [u8; 32],
    /// Pre-seed binary SHA-256 Merkle vector commitment over $u_r^{(i)} \in \mathbb{R}^d$.
    pub commitment: PreSeedVectorCommitmentV1,
}

#[cfg(feature = "alloc")]
impl SharedSeedCommitmentMessageV1 {
    /// Serialize this Phase-1 commitment payload using canonical `postcard` encoding.
    pub fn to_bytes(&self) -> core::result::Result<Vec<u8>, CompressionError> {
        postcard::to_allocvec(self).map_err(|_| CompressionError::InvalidData)
    }

    /// Bounded-deserialize and structurally validate a Phase-1 commitment payload.
    pub fn from_bytes_bounded(
        bytes: &[u8],
        max_bytes: usize,
    ) -> core::result::Result<Self, CompressionError> {
        if bytes.len() > max_bytes {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        let msg: Self = postcard::from_bytes(bytes).map_err(|_| CompressionError::InvalidData)?;
        let d = msg.commitment.original_len as usize;
        if d == 0 || d > DEFAULT_MAX_COMPRESSION_ELEMENTS {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        Ok(msg)
    }
}

/// Authenticated Phase-2 wire payload binding a peer's projected coordinates and Merkle
/// opening proofs to `(context_digest, commitments_root, shared_round_seed)` (`ST-12`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedSeedOpeningMessageV1 {
    /// Canonical [`SparseRoundContextV1::context_digest`].
    pub context_digest: [u8; 32],
    /// Canonical [`compute_commitments_root_v1`] transcript digest locking Phase-1 commitments.
    pub commitments_root: [u8; 32],
    /// Single collective [`derive_context_bound_round_seed_v1`] seed shared by the entire cohort.
    pub shared_round_seed: [u8; 32],
    /// Compact `SharedSeedSparse` payload evaluated strictly on `shared_round_seed`.
    pub payload: SharedSeedSparsePayloadV1,
    /// Merkle coordinate opening proofs linking every opened value to the Phase-1 commitment.
    pub openings: MerkleOpeningBatchV1,
}

#[cfg(feature = "alloc")]
impl SharedSeedOpeningMessageV1 {
    /// Serialize this Phase-2 opening payload using canonical `postcard` encoding.
    pub fn to_bytes(&self) -> core::result::Result<Vec<u8>, CompressionError> {
        postcard::to_allocvec(self).map_err(|_| CompressionError::InvalidData)
    }

    /// Bounded-deserialize and structurally validate a Phase-2 opening payload.
    pub fn from_bytes_bounded(
        bytes: &[u8],
        max_bytes: usize,
    ) -> core::result::Result<Self, CompressionError> {
        if bytes.len() > max_bytes {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        let msg: Self = postcard::from_bytes(bytes).map_err(|_| CompressionError::InvalidData)?;
        let d = msg.payload.original_len as usize;
        let k = msg.payload.k as usize;
        if d == 0 || d > DEFAULT_MAX_COMPRESSION_ELEMENTS {
            return Err(CompressionError::ResourceLimitExceeded);
        }
        if k == 0 || k > d {
            return Err(CompressionError::InvalidRatio);
        }
        if msg.payload.values.len() != k || msg.openings.proofs.len() != k {
            return Err(CompressionError::CommitmentVerificationFailed);
        }
        if msg.payload.seed != msg.shared_round_seed {
            return Err(CompressionError::CommitmentVerificationFailed);
        }
        if msg.payload.values.iter().any(|v| !v.is_finite())
            || msg.openings.proofs.iter().any(|p| !p.raw_value.is_finite())
        {
            return Err(CompressionError::NonFiniteValue);
        }
        Ok(msg)
    }
}

/// Compute the deterministic `commitments_root` binding `context_digest` and the canonical
/// `PeerId`-sorted Phase-1 pre-seed commitments (`ST-12`).
#[cfg(feature = "alloc")]
pub fn compute_commitments_root_v1(
    context_digest: &[u8; 32],
    commitments: &[(crate::traits::PeerId, PreSeedVectorCommitmentV1)],
) -> core::result::Result<[u8; 32], CompressionError> {
    if commitments.is_empty() {
        return Err(CompressionError::InvalidData);
    }

    let mut sorted = commitments.to_vec();
    sorted.sort_by_key(|a| a.0);
    for window in sorted.windows(2) {
        if window[0].0 == window[1].0 {
            return Err(CompressionError::InvalidData);
        }
    }

    let mut hasher = Sha256::new();
    hasher.update(COMMITMENTS_ROOT_DOMAIN_V1);
    hasher.update(context_digest);
    hasher.update((sorted.len() as u32).to_le_bytes());
    for (peer_id, commitment) in &sorted {
        hasher.update(peer_id.as_bytes());
        hasher.update(commitment.original_len.to_le_bytes());
        hasher.update(commitment.merkle_root);
    }
    Ok(hasher.finalize().into())
}

/// Derive the single collective 32-byte round seed shared by every participant in a two-phase
/// `SharedSeedSparse` round from `(context_digest, commitments_root, beacon_r)` (`ST-12`).
///
/// Because `beacon_r` is revealed only after `commitments_root` closes and no per-sender nonce or
/// `PeerId` enters the seed derivation, all honest peers project onto an identical support $S_r$.
#[cfg(feature = "alloc")]
pub fn derive_context_bound_round_seed_v1(
    context_digest: &[u8; 32],
    commitments_root: &[u8; 32],
    beacon_r: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(COLLECTIVE_ROUND_SEED_DOMAIN_V1);
    hasher.update(context_digest);
    hasher.update(commitments_root);
    hasher.update(beacon_r);
    hasher.finalize().into()
}

/// Coordinator-side verifier enforcing one common shared-seed support $S_{\text{round}}$ and
/// validating Phase-2 Merkle openings against frozen Phase-1 commitments (`ST-12`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct SharedSeedCoordinateVerifier {
    context: SparseRoundContextV1,
    context_digest: [u8; 32],
    commitments_root: [u8; 32],
    shared_round_seed: [u8; 32],
    expected_indices: Vec<u32>,
}

#[cfg(feature = "alloc")]
impl SharedSeedCoordinateVerifier {
    /// Construct a [`SharedSeedCoordinateVerifier`] and precompute the single cohort-wide
    /// projection support $S_{\text{round}} = \text{sample\_shared\_indices\_v1}(d, k, R)$.
    pub fn try_new(
        context: SparseRoundContextV1,
        commitments_root: [u8; 32],
        shared_round_seed: [u8; 32],
    ) -> core::result::Result<Self, CompressionError> {
        let context_digest = context.context_digest()?;
        let expected_indices = sample_shared_indices_v1(
            context.original_len as usize,
            context.sparsity_k as usize,
            &shared_round_seed,
        )?;
        Ok(Self {
            context,
            context_digest,
            commitments_root,
            shared_round_seed,
            expected_indices,
        })
    }

    /// Borrow the frozen round context.
    pub const fn context(&self) -> &SparseRoundContextV1 {
        &self.context
    }

    /// Canonical SHA-256 digest of `context`.
    pub const fn context_digest(&self) -> [u8; 32] {
        self.context_digest
    }

    /// Frozen Phase-1 commitment transcript root.
    pub const fn commitments_root(&self) -> [u8; 32] {
        self.commitments_root
    }

    /// Single cohort-wide shared round seed.
    pub const fn shared_round_seed(&self) -> [u8; 32] {
        self.shared_round_seed
    }

    /// Strictly ascending shared coordinate indices $S_{\text{round}}$ opened by all cohort peers.
    pub fn expected_indices(&self) -> &[u32] {
        &self.expected_indices
    }

    /// Verify a peer's Phase-2 [`SharedSeedOpeningMessageV1`] against its frozen Phase-1
    /// `recorded_commitment`, failing closed on any transcript divergence, scaling mismatch,
    /// non-finite coordinate, wrong coordinate index, or Merkle proof forgery.
    pub fn verify_opening_message(
        &self,
        recorded_commitment: &PreSeedVectorCommitmentV1,
        msg: &SharedSeedOpeningMessageV1,
    ) -> core::result::Result<(), CompressionError> {
        if msg.context_digest != self.context_digest
            || msg.commitments_root != self.commitments_root
            || msg.shared_round_seed != self.shared_round_seed
        {
            return Err(CompressionError::CommitmentVerificationFailed);
        }
        if recorded_commitment.original_len != self.context.original_len
            || msg.payload.original_len != self.context.original_len
            || msg.payload.k != self.context.sparsity_k
            || msg.payload.seed != self.shared_round_seed
            || msg.payload.scaling != self.context.scaling
        {
            return Err(CompressionError::CommitmentVerificationFailed);
        }
        verify_shared_seed_openings_v1(recorded_commitment, &msg.payload, &msg.openings)?;
        if msg.openings.proofs.len() != self.expected_indices.len() {
            return Err(CompressionError::CommitmentVerificationFailed);
        }
        for (proof, &expected_idx) in msg.openings.proofs.iter().zip(self.expected_indices.iter()) {
            if proof.index != expected_idx {
                return Err(CompressionError::CommitmentVerificationFailed);
            }
        }
        Ok(())
    }
}

/// Honest two-phase wire bandwidth breakdown for collective `SharedSeedSparse` delivery (`ST-12`).
///
/// Explicitly separates $4k$ projected scalar value bytes from Merkle opening proof overhead
/// ($\approx k \lceil \log_2 d \rceil \times 32\text{ B}$) and Phase-1/2 authenticated envelope
/// framing so small-dimension regimes (`total_two_phase_wire_bytes > raw_dense_bytes_total`) are
/// never masked behind compressed scalar-only claims.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedSeedBandwidthReportV1 {
    /// Number of verified Phase-2 peer deliveries accounted for in this report.
    pub peer_count: usize,
    /// Dense parameter dimension $d$.
    pub original_len: u32,
    /// Shared sparse coordinate count $k$.
    pub sparsity_k: u32,
    /// Uncompressed raw dense scalar payload baseline across the cohort (`peer_count * 4 * d`).
    pub raw_dense_bytes_total: usize,
    /// Projected scalar value bytes across the cohort (`peer_count * 4 * k`).
    pub projected_values_bytes_total: usize,
    /// Merkle coordinate opening proof payload bytes (`sum_proofs (8 + siblings.len() * 32)`).
    pub merkle_proof_bytes_total: usize,
    /// Serialized Phase-1 commitment delivery bytes across the cohort (`BundledGossipDeliveryV2`).
    pub phase1_commitment_wire_bytes_total: usize,
    /// Serialized Phase-2 opening delivery bytes across the cohort (`BundledGossipDeliveryV2`).
    pub phase2_opening_wire_bytes_total: usize,
    /// Remaining protocol/envelope/signature framing bytes (`total - values - merkle_proofs`).
    pub envelope_and_framing_bytes_total: usize,
    /// Total wire bytes across both Phase-1 commitments and Phase-2 openings.
    pub total_two_phase_wire_bytes: usize,
}

#[cfg(feature = "alloc")]
impl SharedSeedBandwidthReportV1 {
    /// Build a zeroed report header for `(original_len, sparsity_k)`.
    pub const fn new(original_len: u32, sparsity_k: u32) -> Self {
        Self {
            peer_count: 0,
            original_len,
            sparsity_k,
            raw_dense_bytes_total: 0,
            projected_values_bytes_total: 0,
            merkle_proof_bytes_total: 0,
            phase1_commitment_wire_bytes_total: 0,
            phase2_opening_wire_bytes_total: 0,
            envelope_and_framing_bytes_total: 0,
            total_two_phase_wire_bytes: 0,
        }
    }

    /// Accumulate one peer's Phase-1 commitment frame, Phase-2 opening frame, and opening batch.
    pub fn record_peer_delivery(
        &mut self,
        phase1_wire_bytes: usize,
        phase2_wire_bytes: usize,
        openings: &MerkleOpeningBatchV1,
    ) {
        let k = self.sparsity_k as usize;
        let d = self.original_len as usize;
        let peer_raw_dense = d.saturating_mul(4);
        let peer_projected = k.saturating_mul(4);
        let peer_proof_bytes: usize = openings
            .proofs
            .iter()
            .map(|p| 8usize.saturating_add(p.siblings.len().saturating_mul(32)))
            .sum();
        let peer_total = phase1_wire_bytes.saturating_add(phase2_wire_bytes);

        self.peer_count = self.peer_count.saturating_add(1);
        self.raw_dense_bytes_total = self.raw_dense_bytes_total.saturating_add(peer_raw_dense);
        self.projected_values_bytes_total = self
            .projected_values_bytes_total
            .saturating_add(peer_projected);
        self.merkle_proof_bytes_total = self
            .merkle_proof_bytes_total
            .saturating_add(peer_proof_bytes);
        self.phase1_commitment_wire_bytes_total = self
            .phase1_commitment_wire_bytes_total
            .saturating_add(phase1_wire_bytes);
        self.phase2_opening_wire_bytes_total = self
            .phase2_opening_wire_bytes_total
            .saturating_add(phase2_wire_bytes);
        self.total_two_phase_wire_bytes =
            self.total_two_phase_wire_bytes.saturating_add(peer_total);
        self.envelope_and_framing_bytes_total = self.total_two_phase_wire_bytes.saturating_sub(
            self.projected_values_bytes_total
                .saturating_add(self.merkle_proof_bytes_total),
        );
    }
}

#[cfg(all(test, feature = "alloc"))]
#[path = "compression_tests.rs"]
mod tests;
