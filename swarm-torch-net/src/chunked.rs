//! Normative 16-byte `ChunkedFrameCodec` and pre-authentication `ObservedSourceReassembler`
//! (`ADR-0022` §2, `W11-03`).
//!
//! Provides:
//! - Frozen 16-byte little-endian frame header with `[0x53, 0x54]` (`"ST"`) magic, version `0x01`,
//!   strict `first`/`last` flags, and **CRC-16/IBM-3740 (CCITT-FALSE)** integrity check.
//! - Pinned 19-byte golden test vector verification.
//! - Fixed-stride fragmentation ($P = M - 16$) and pre-allocation upper bound check
//!   $(N - 1)P + 1 \le L_{\max}$.
//! - Pre-authentication reassembly keyed strictly by `(adapter_generation, observed_source, transfer_id)`
//!   with per-source and global byte/transfer budgets, non-extending hard deadlines, and conflict
//!   quarantine tombstones (`F-05`).

#[cfg(feature = "alloc")]
use alloc::collections::BTreeMap;
#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Fixed byte length of the normative `ChunkedFrameCodec` v1 header (`16` bytes).
pub const CHUNK_HEADER_LEN: usize = 16;

/// Magic bytes (`"ST"`) at offset `0..2` of every chunked frame header.
pub const CHUNK_MAGIC: [u8; 2] = [0x53, 0x54];

/// Protocol version byte at offset `2` of every v1 chunked frame header.
pub const CHUNK_VERSION_V1: u8 = 0x01;

/// Flag bit `0` (`0x01`): set if and only if `chunk_idx == 0`.
pub const CHUNK_FLAG_FIRST: u8 = 0x01;

/// Flag bit `1` (`0x02`): set if and only if `chunk_idx + 1 == total_chunks`.
pub const CHUNK_FLAG_LAST: u8 = 0x02;

/// Reserved flag bits mask (`bits 2..7`, `0xFC`): MUST be `0` in v1 frames.
pub const CHUNK_FLAGS_RESERVED_MASK: u8 = 0xFC;

/// Compute **CRC-16/IBM-3740 (CCITT-FALSE)** over a byte slice (`poly = 0x1021`, `init = 0xFFFF`,
/// `refin = false`, `refout = false`, `xorout = 0x0000`, `check("123456789") == 0x29B1`).
#[inline]
pub fn crc16_ibm3740(bytes: &[u8]) -> u16 {
    crc16_ibm3740_update(0xFFFF, bytes)
}

/// Update an in-progress **CRC-16/IBM-3740 (CCITT-FALSE)** accumulator with `bytes`.
#[inline]
pub fn crc16_ibm3740_update(mut crc: u16, bytes: &[u8]) -> u16 {
    for &b in bytes {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            if (crc & 0x8000) != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// Errors returned by [`ChunkedFrameCodec`] and [`ObservedSourceReassembler`] (`ADR-0022` §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkError {
    /// MTU must exceed [`CHUNK_HEADER_LEN`] (`16` bytes) and fit within `u16::MAX`.
    InvalidMtu,
    /// Frame is shorter than [`CHUNK_HEADER_LEN`] or does not match `16 + payload_len`.
    FrameLengthMismatch,
    /// Header magic bytes do not equal `[0x53, 0x54]` (`"ST"`).
    InvalidMagic,
    /// Unsupported chunk header version.
    UnsupportedVersion(u8),
    /// Reserved flag bits are non-zero or `first`/`last` flags contradict `chunk_idx`/`total_chunks`.
    InvalidFlags(u8),
    /// `total_chunks == 0`, `total_chunks > max_chunks_per_transfer`, or `chunk_idx >= total_chunks`.
    InvalidChunkBounds,
    /// `payload_len == 0`, non-final chunk has `payload_len != P`, or final chunk has `payload_len > P`.
    InvalidPayloadStride,
    /// Message is empty or exceeds `max_message_bytes` ($L_{\max}$) or $(N - 1)P + 1 > L_{\max}$.
    MessageTooLarge,
    /// CRC-16/IBM-3740 check failed (`expected` in header vs. `computed` over frame).
    CrcMismatch { expected: u16, computed: u16 },
    /// Caller-supplied output buffer is too small.
    BufferTooSmall { required: usize, available: usize },
    /// Transfer key is currently quarantined due to a prior conflicting chunk or `total_chunks` mismatch.
    TransferQuarantined,
    /// Incoming chunk for an active transfer declared a different `total_chunks` than the transfer's initial chunk.
    ConflictingTotalChunks,
    /// Incoming duplicate chunk index carried different payload bytes than the previously admitted chunk.
    ConflictingChunkPayload,
    /// Per-source or global active-transfer, reserved-byte, source-table, or tombstone budget exceeded.
    ResourceBudgetExceeded,
}

impl core::fmt::Display for ChunkError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidMtu => write!(f, "invalid MTU for ChunkedFrameCodec"),
            Self::FrameLengthMismatch => write!(f, "chunk frame byte length mismatch"),
            Self::InvalidMagic => write!(f, "invalid chunk magic bytes (expected 'ST')"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported chunk frame version: {v}"),
            Self::InvalidFlags(fl) => write!(f, "invalid chunk header flags: 0x{fl:02x}"),
            Self::InvalidChunkBounds => write!(f, "invalid chunk_idx or total_chunks bounds"),
            Self::InvalidPayloadStride => write!(f, "invalid chunk payload_len for fixed stride P"),
            Self::MessageTooLarge => write!(f, "message empty or exceeds max_message_bytes"),
            Self::CrcMismatch { expected, computed } => write!(
                f,
                "chunk CRC-16/IBM-3740 mismatch: expected 0x{expected:04x}, computed 0x{computed:04x}"
            ),
            Self::BufferTooSmall {
                required,
                available,
            } => write!(
                f,
                "output buffer too small: required {required}, available {available}"
            ),
            Self::TransferQuarantined => {
                write!(f, "transfer key is quarantined after conflict detection")
            }
            Self::ConflictingTotalChunks => write!(
                f,
                "conflicting total_chunks for active transfer; transfer quarantined"
            ),
            Self::ConflictingChunkPayload => write!(
                f,
                "conflicting duplicate chunk payload for active transfer; transfer quarantined"
            ),
            Self::ResourceBudgetExceeded => {
                write!(f, "pre-authentication reassembly resource budget exceeded")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ChunkError {}

/// Decoded 16-byte `ChunkedFrameCodec` v1 header (`ADR-0022` §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkHeaderV1 {
    /// Flags byte (`CHUNK_FLAG_FIRST | CHUNK_FLAG_LAST`).
    pub flags: u8,
    /// Sender-scoped transfer identifier.
    pub transfer_id: u32,
    /// Zero-based chunk index (`0 <= chunk_idx < total_chunks`).
    pub chunk_idx: u16,
    /// Total chunks in this transfer (`1 <= total_chunks <= 65535`).
    pub total_chunks: u16,
    /// Byte length of this chunk's payload (`1 <= payload_len <= P`).
    pub payload_len: u16,
    /// CRC-16/IBM-3740 over `header[0..14] || payload`.
    pub crc16: u16,
}

impl ChunkHeaderV1 {
    /// Compute the canonical `flags` byte for `(chunk_idx, total_chunks)`.
    #[inline]
    pub const fn canonical_flags(chunk_idx: u16, total_chunks: u16) -> u8 {
        let mut flags = 0u8;
        if chunk_idx == 0 {
            flags |= CHUNK_FLAG_FIRST;
        }
        if total_chunks > 0 && chunk_idx + 1 == total_chunks {
            flags |= CHUNK_FLAG_LAST;
        }
        flags
    }
}

/// Borrowed view of a decoded and CRC-verified chunk frame (`ADR-0022` §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodedChunkFrame<'a> {
    /// Verified 16-byte header.
    pub header: ChunkHeaderV1,
    /// Verified chunk payload slice of length `header.payload_len as usize`.
    pub payload: &'a [u8],
}

/// Stateless encoder/decoder for the frozen 16-byte `ChunkedFrameCodec` (`ADR-0022` §2, `W11-03`).
pub struct ChunkedFrameCodec;

impl ChunkedFrameCodec {
    /// Compute nominal chunk payload stride $P = M - 16$ from link MTU $M$.
    #[inline]
    pub fn stride_for_mtu(mtu: usize) -> Result<usize, ChunkError> {
        if mtu <= CHUNK_HEADER_LEN {
            return Err(ChunkError::InvalidMtu);
        }
        let p = mtu - CHUNK_HEADER_LEN;
        if p > u16::MAX as usize {
            return Err(ChunkError::InvalidMtu);
        }
        Ok(p)
    }

    /// Encode a single chunk frame into `out_buf`, returning the total frame length (`16 + payload.len()`).
    pub fn encode_frame_into(
        transfer_id: u32,
        chunk_idx: u16,
        total_chunks: u16,
        payload: &[u8],
        out_buf: &mut [u8],
    ) -> Result<usize, ChunkError> {
        if total_chunks == 0 || chunk_idx >= total_chunks {
            return Err(ChunkError::InvalidChunkBounds);
        }
        if payload.is_empty() || payload.len() > u16::MAX as usize {
            return Err(ChunkError::InvalidPayloadStride);
        }
        let total_len = CHUNK_HEADER_LEN + payload.len();
        if out_buf.len() < total_len {
            return Err(ChunkError::BufferTooSmall {
                required: total_len,
                available: out_buf.len(),
            });
        }

        let flags = ChunkHeaderV1::canonical_flags(chunk_idx, total_chunks);
        let payload_len = payload.len() as u16;

        out_buf[0..2].copy_from_slice(&CHUNK_MAGIC);
        out_buf[2] = CHUNK_VERSION_V1;
        out_buf[3] = flags;
        out_buf[4..8].copy_from_slice(&transfer_id.to_le_bytes());
        out_buf[8..10].copy_from_slice(&chunk_idx.to_le_bytes());
        out_buf[10..12].copy_from_slice(&total_chunks.to_le_bytes());
        out_buf[12..14].copy_from_slice(&payload_len.to_le_bytes());

        let mut crc = crc16_ibm3740_update(0xFFFF, &out_buf[0..14]);
        crc = crc16_ibm3740_update(crc, payload);
        out_buf[14..16].copy_from_slice(&crc.to_le_bytes());
        out_buf[16..total_len].copy_from_slice(payload);

        Ok(total_len)
    }

    /// Decode and validate a single chunk frame against `mtu`, `max_message_bytes` ($L_{\max}$),
    /// and `max_chunks_per_transfer`.
    pub fn decode_frame_with_limits(
        frame: &[u8],
        mtu: usize,
        max_message_bytes: usize,
        max_chunks_per_transfer: u16,
    ) -> Result<DecodedChunkFrame<'_>, ChunkError> {
        let stride_p = Self::stride_for_mtu(mtu)?;
        if frame.len() < CHUNK_HEADER_LEN || frame.len() > mtu {
            return Err(ChunkError::FrameLengthMismatch);
        }
        if frame[0..2] != CHUNK_MAGIC {
            return Err(ChunkError::InvalidMagic);
        }
        let version = frame[2];
        if version != CHUNK_VERSION_V1 {
            return Err(ChunkError::UnsupportedVersion(version));
        }
        let flags = frame[3];
        if (flags & CHUNK_FLAGS_RESERVED_MASK) != 0 {
            return Err(ChunkError::InvalidFlags(flags));
        }

        let transfer_id = u32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]]);
        let chunk_idx = u16::from_le_bytes([frame[8], frame[9]]);
        let total_chunks = u16::from_le_bytes([frame[10], frame[11]]);
        let payload_len = u16::from_le_bytes([frame[12], frame[13]]);
        let expected_crc = u16::from_le_bytes([frame[14], frame[15]]);

        if total_chunks == 0 || total_chunks > max_chunks_per_transfer || chunk_idx >= total_chunks
        {
            return Err(ChunkError::InvalidChunkBounds);
        }

        let expected_flags = ChunkHeaderV1::canonical_flags(chunk_idx, total_chunks);
        if flags != expected_flags {
            return Err(ChunkError::InvalidFlags(flags));
        }

        let payload_len_usize = payload_len as usize;
        if payload_len_usize == 0 || CHUNK_HEADER_LEN + payload_len_usize != frame.len() {
            return Err(ChunkError::FrameLengthMismatch);
        }

        // Fixed-stride validation (ADR-0022 §2):
        // - Non-final chunk must have payload_len == P.
        // - Final chunk must have 1 <= payload_len <= P.
        let is_last = chunk_idx + 1 == total_chunks;
        if !is_last && payload_len_usize != stride_p {
            return Err(ChunkError::InvalidPayloadStride);
        }
        if is_last && payload_len_usize > stride_p {
            return Err(ChunkError::InvalidPayloadStride);
        }

        // Pre-allocation upper bound check: minimum message size for N chunks is (N - 1)*P + 1.
        let min_message_bytes = ((total_chunks as usize) - 1)
            .checked_mul(stride_p)
            .and_then(|prefix| prefix.checked_add(1))
            .ok_or(ChunkError::MessageTooLarge)?;
        if min_message_bytes > max_message_bytes {
            return Err(ChunkError::MessageTooLarge);
        }
        if is_last {
            let exact_message_bytes = ((chunk_idx as usize) * stride_p) + payload_len_usize;
            if exact_message_bytes > max_message_bytes {
                return Err(ChunkError::MessageTooLarge);
            }
        }

        let payload = &frame[CHUNK_HEADER_LEN..];
        let mut computed_crc = crc16_ibm3740_update(0xFFFF, &frame[0..14]);
        computed_crc = crc16_ibm3740_update(computed_crc, payload);
        if computed_crc != expected_crc {
            return Err(ChunkError::CrcMismatch {
                expected: expected_crc,
                computed: computed_crc,
            });
        }

        Ok(DecodedChunkFrame {
            header: ChunkHeaderV1 {
                flags,
                transfer_id,
                chunk_idx,
                total_chunks,
                payload_len,
                crc16: expected_crc,
            },
            payload,
        })
    }

    /// Fragment `message` into a sequence of chunked frames for `mtu` and `max_message_bytes`.
    #[cfg(feature = "alloc")]
    pub fn encode_message(
        transfer_id: u32,
        mtu: usize,
        max_message_bytes: usize,
        max_chunks_per_transfer: u16,
        message: &[u8],
    ) -> Result<Vec<Vec<u8>>, ChunkError> {
        let stride_p = Self::stride_for_mtu(mtu)?;
        if message.is_empty() || message.len() > max_message_bytes {
            return Err(ChunkError::MessageTooLarge);
        }
        let num_chunks = message.len().div_ceil(stride_p);
        if num_chunks == 0
            || num_chunks > (u16::MAX as usize)
            || num_chunks > (max_chunks_per_transfer as usize)
        {
            return Err(ChunkError::InvalidChunkBounds);
        }

        let total_chunks = num_chunks as u16;
        let mut frames = Vec::with_capacity(num_chunks);
        for (idx, chunk) in message.chunks(stride_p).enumerate() {
            let mut frame = vec![0u8; CHUNK_HEADER_LEN + chunk.len()];
            let written =
                Self::encode_frame_into(transfer_id, idx as u16, total_chunks, chunk, &mut frame)?;
            debug_assert_eq!(written, frame.len());
            frames.push(frame);
        }
        Ok(frames)
    }
}

/// Composite pre-authentication reassembly key (`ADR-0022` §2, `W11-03`).
///
/// Keyed strictly by `(adapter_generation, observed_source, transfer_id)` — never by an
/// unauthenticated `PeerId` read from an unverified payload.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReassemblyKey<E> {
    /// Monotonic adapter bind/epoch generation counter.
    pub adapter_generation: u64,
    /// Kernel/driver observed source endpoint (e.g. `SocketAddr` or radio address).
    pub observed_source: E,
    /// Sender-chosen 32-bit transfer identifier.
    pub transfer_id: u32,
}

/// Resource and timeout limits for [`ObservedSourceReassembler`] (`ADR-0022` §2, `W11-03`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReassemblerLimits {
    /// Maximum frame size $M$ in bytes (default `1400`, giving stride $P = 1384$).
    pub mtu: usize,
    /// Maximum reassembled message size $L_{\max}$ in bytes (default `1_048_576`).
    pub max_message_bytes: usize,
    /// Maximum chunks allowed per transfer (default `768`).
    pub max_chunks_per_transfer: u16,
    /// Maximum concurrent in-flight transfers per observed source (default `4`).
    pub per_source_max_active_transfers: usize,
    /// Maximum reserved reassembly bytes per observed source (default `2_097_152`).
    pub per_source_max_reserved_bytes: usize,
    /// Maximum concurrent in-flight transfers across all sources (default `64`).
    pub global_max_active_transfers: usize,
    /// Maximum reserved reassembly bytes across all sources (default `16_777_216`).
    pub global_max_reserved_bytes: usize,
    /// Maximum distinct `(adapter_generation, observed_source)` entries tracked simultaneously (default `256`).
    pub max_tracked_sources: usize,
    /// Maximum conflict quarantine tombstones retained simultaneously (default `512`).
    pub max_quarantine_tombstones: usize,
    /// Hard monotonic reassembly timeout in milliseconds from the first admitted chunk (default `5_000`).
    pub transfer_timeout_ms: u64,
}

impl Default for ReassemblerLimits {
    fn default() -> Self {
        Self {
            mtu: 1400,
            max_message_bytes: 1_048_576,
            max_chunks_per_transfer: 768,
            per_source_max_active_transfers: 4,
            per_source_max_reserved_bytes: 2 * 1024 * 1024,
            global_max_active_transfers: 64,
            global_max_reserved_bytes: 16 * 1024 * 1024,
            max_tracked_sources: 256,
            max_quarantine_tombstones: 512,
            transfer_timeout_ms: 5_000,
        }
    }
}

/// Completed message reassembled by [`ObservedSourceReassembler`].
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReassembledTransfer<E> {
    /// Pre-authentication source key that produced this message.
    pub key: ReassemblyKey<E>,
    /// Complete reassembled message bytes ready for 6-stage `MessageEnvelope` verification.
    pub payload: Vec<u8>,
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
struct SourceAccounting {
    active_transfers: usize,
    reserved_bytes: usize,
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
struct ActiveTransferState {
    total_chunks: u16,
    reserved_bytes: usize,
    deadline_mono_ms: u64,
    received_count: usize,
    chunks: Vec<Option<Vec<u8>>>,
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy)]
struct QuarantineTombstone {
    deadline_mono_ms: u64,
}

/// Pre-authentication bounded chunk reassembler (`ADR-0022` §2, `W11-03`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ObservedSourceReassembler<E: Clone + Eq + Ord> {
    limits: ReassemblerLimits,
    active_transfers: BTreeMap<ReassemblyKey<E>, ActiveTransferState>,
    source_stats: BTreeMap<(u64, E), SourceAccounting>,
    quarantine_tombstones: BTreeMap<ReassemblyKey<E>, QuarantineTombstone>,
    global_reserved_bytes: usize,
}

#[cfg(feature = "alloc")]
impl<E: Clone + Eq + Ord> ObservedSourceReassembler<E> {
    /// Create a new pre-authentication reassembler with `limits`.
    pub fn new(limits: ReassemblerLimits) -> Result<Self, ChunkError> {
        let _ = ChunkedFrameCodec::stride_for_mtu(limits.mtu)?;
        if limits.max_message_bytes == 0 || limits.max_chunks_per_transfer == 0 {
            return Err(ChunkError::MessageTooLarge);
        }
        Ok(Self {
            limits,
            active_transfers: BTreeMap::new(),
            source_stats: BTreeMap::new(),
            quarantine_tombstones: BTreeMap::new(),
            global_reserved_bytes: 0,
        })
    }

    /// Current number of active in-flight multi-chunk transfers.
    pub fn active_transfer_count(&self) -> usize {
        self.active_transfers.len()
    }

    /// Current global reserved bytes across all active transfers.
    pub const fn global_reserved_bytes(&self) -> usize {
        self.global_reserved_bytes
    }

    /// Current number of active conflict quarantine tombstones.
    pub fn quarantine_tombstone_count(&self) -> usize {
        self.quarantine_tombstones.len()
    }

    /// Evict all active transfers and quarantine tombstones whose hard deadline has expired
    /// (`now_mono_ms >= deadline_mono_ms`).
    pub fn evict_expired(&mut self, now_mono_ms: u64) {
        let expired_keys: Vec<ReassemblyKey<E>> = self
            .active_transfers
            .iter()
            .filter_map(|(k, v)| {
                if now_mono_ms >= v.deadline_mono_ms {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();

        for key in expired_keys {
            self.remove_active_transfer(&key);
        }

        let expired_tombstones: Vec<ReassemblyKey<E>> = self
            .quarantine_tombstones
            .iter()
            .filter_map(|(k, t)| {
                if now_mono_ms >= t.deadline_mono_ms {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();

        for key in expired_tombstones {
            self.quarantine_tombstones.remove(&key);
        }
    }

    fn remove_active_transfer(&mut self, key: &ReassemblyKey<E>) -> Option<ActiveTransferState> {
        let removed = self.active_transfers.remove(key)?;
        self.global_reserved_bytes = self
            .global_reserved_bytes
            .saturating_sub(removed.reserved_bytes);

        let src_key = (key.adapter_generation, key.observed_source.clone());
        if let Some(stats) = self.source_stats.get_mut(&src_key) {
            stats.active_transfers = stats.active_transfers.saturating_sub(1);
            stats.reserved_bytes = stats.reserved_bytes.saturating_sub(removed.reserved_bytes);
            if stats.active_transfers == 0 && stats.reserved_bytes == 0 {
                self.source_stats.remove(&src_key);
            }
        }
        Some(removed)
    }

    fn quarantine_key(&mut self, key: ReassemblyKey<E>, deadline_mono_ms: u64) {
        if self.limits.max_quarantine_tombstones == 0 {
            return;
        }
        if self.quarantine_tombstones.len() >= self.limits.max_quarantine_tombstones
            && !self.quarantine_tombstones.contains_key(&key)
        {
            if let Some(oldest_key) = self
                .quarantine_tombstones
                .iter()
                .min_by_key(|(_, t)| t.deadline_mono_ms)
                .map(|(k, _)| k.clone())
            {
                self.quarantine_tombstones.remove(&oldest_key);
            }
        }
        self.quarantine_tombstones
            .insert(key, QuarantineTombstone { deadline_mono_ms });
    }

    /// Ingest a single raw frame from `(adapter_generation, observed_source)` at monotonic
    /// timestamp `now_mono_ms` (`ADR-0022` §2).
    pub fn ingest_frame(
        &mut self,
        adapter_generation: u64,
        observed_source: E,
        frame_bytes: &[u8],
        now_mono_ms: u64,
    ) -> Result<Option<ReassembledTransfer<E>>, ChunkError> {
        // 1. Reclaim expired transfers and tombstones first.
        self.evict_expired(now_mono_ms);

        // 2. Decode and verify header + CRC-16/IBM-3740 BEFORE touching any transfer state.
        let decoded = ChunkedFrameCodec::decode_frame_with_limits(
            frame_bytes,
            self.limits.mtu,
            self.limits.max_message_bytes,
            self.limits.max_chunks_per_transfer,
        )?;
        let stride_p = ChunkedFrameCodec::stride_for_mtu(self.limits.mtu)?;

        let key = ReassemblyKey {
            adapter_generation,
            observed_source: observed_source.clone(),
            transfer_id: decoded.header.transfer_id,
        };

        // 3. Reject if transfer key is under conflict quarantine.
        if self.quarantine_tombstones.contains_key(&key) {
            return Err(ChunkError::TransferQuarantined);
        }

        // 4. Existing active transfer handling.
        if let Some(existing) = self.active_transfers.get_mut(&key) {
            if decoded.header.total_chunks != existing.total_chunks {
                let deadline = existing.deadline_mono_ms;
                self.remove_active_transfer(&key);
                self.quarantine_key(key, deadline);
                return Err(ChunkError::ConflictingTotalChunks);
            }

            let idx = decoded.header.chunk_idx as usize;
            if let Some(ref prev_bytes) = existing.chunks[idx] {
                if prev_bytes.as_slice() == decoded.payload {
                    // Byte-identical duplicate chunk: idempotent no-op without extending deadline.
                    return Ok(None);
                } else {
                    let deadline = existing.deadline_mono_ms;
                    self.remove_active_transfer(&key);
                    self.quarantine_key(key, deadline);
                    return Err(ChunkError::ConflictingChunkPayload);
                }
            }

            existing.chunks[idx] = Some(decoded.payload.to_vec());
            existing.received_count += 1;

            if existing.received_count == existing.total_chunks as usize {
                let completed = self
                    .remove_active_transfer(&key)
                    .expect("active transfer existed");
                let total_len: usize = completed
                    .chunks
                    .iter()
                    .map(|c| c.as_ref().map_or(0, Vec::len))
                    .sum();
                if total_len > self.limits.max_message_bytes {
                    return Err(ChunkError::MessageTooLarge);
                }
                let mut payload = Vec::with_capacity(total_len);
                for chunk in completed.chunks.into_iter().flatten() {
                    payload.extend_from_slice(&chunk);
                }
                return Ok(Some(ReassembledTransfer { key, payload }));
            }

            return Ok(None);
        }

        // 5. New transfer admission: compute bounded reservation min(N * P, L_max).
        let reserved_bytes = (decoded.header.total_chunks as usize)
            .saturating_mul(stride_p)
            .min(self.limits.max_message_bytes);

        let src_key = (adapter_generation, observed_source);
        let current_src = self
            .source_stats
            .get(&src_key)
            .cloned()
            .unwrap_or(SourceAccounting {
                active_transfers: 0,
                reserved_bytes: 0,
            });

        if !self.source_stats.contains_key(&src_key)
            && self.source_stats.len() >= self.limits.max_tracked_sources
        {
            return Err(ChunkError::ResourceBudgetExceeded);
        }
        if current_src.active_transfers >= self.limits.per_source_max_active_transfers
            || current_src.reserved_bytes.saturating_add(reserved_bytes)
                > self.limits.per_source_max_reserved_bytes
            || self.active_transfers.len() >= self.limits.global_max_active_transfers
            || self.global_reserved_bytes.saturating_add(reserved_bytes)
                > self.limits.global_max_reserved_bytes
        {
            return Err(ChunkError::ResourceBudgetExceeded);
        }

        // Fast-path for a 1-chunk message after budget verification.
        if decoded.header.total_chunks == 1 {
            return Ok(Some(ReassembledTransfer {
                key,
                payload: decoded.payload.to_vec(),
            }));
        }

        let deadline_mono_ms = now_mono_ms.saturating_add(self.limits.transfer_timeout_ms);
        let mut chunks = vec![None; decoded.header.total_chunks as usize];
        chunks[decoded.header.chunk_idx as usize] = Some(decoded.payload.to_vec());

        self.active_transfers.insert(
            key,
            ActiveTransferState {
                total_chunks: decoded.header.total_chunks,
                reserved_bytes,
                deadline_mono_ms,
                received_count: 1,
                chunks,
            },
        );
        self.global_reserved_bytes += reserved_bytes;
        let entry = self
            .source_stats
            .entry(src_key)
            .or_insert(SourceAccounting {
                active_transfers: 0,
                reserved_bytes: 0,
            });
        entry.active_transfers += 1;
        entry.reserved_bytes += reserved_bytes;

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc16_ibm3740_check_value_and_pinned_19_byte_golden_vector() {
        // Standard CRC-16/IBM-3740 check string "123456789" -> 0x29B1.
        assert_eq!(crc16_ibm3740(b"123456789"), 0x29B1);

        // ADR-0022 §2 pinned golden vector:
        // transfer_id = 0x01020304, single chunk (chunk_idx = 0, total_chunks = 1, flags = 0x03),
        // payload = [0xAA, 0xBB, 0xCC] -> exact 19 bytes:
        // 53 54 01 03 04 03 02 01 00 00 01 00 03 00 7D A5 AA BB CC
        let expected_golden: [u8; 19] = [
            0x53, 0x54, 0x01, 0x03, 0x04, 0x03, 0x02, 0x01, 0x00, 0x00, 0x01, 0x00, 0x03, 0x00,
            0x7D, 0xA5, 0xAA, 0xBB, 0xCC,
        ];

        let mut buf = [0u8; 19];
        let n =
            ChunkedFrameCodec::encode_frame_into(0x0102_0304, 0, 1, &[0xAA, 0xBB, 0xCC], &mut buf)
                .unwrap();
        assert_eq!(n, 19);
        assert_eq!(buf, expected_golden);

        let decoded = ChunkedFrameCodec::decode_frame_with_limits(&buf, 64, 1024, 16).unwrap();
        assert_eq!(decoded.header.flags, 0x03);
        assert_eq!(decoded.header.transfer_id, 0x0102_0304);
        assert_eq!(decoded.header.chunk_idx, 0);
        assert_eq!(decoded.header.total_chunks, 1);
        assert_eq!(decoded.header.payload_len, 3);
        assert_eq!(decoded.header.crc16, 0xA57D);
        assert_eq!(decoded.payload, &[0xAA, 0xBB, 0xCC]);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn reassembler_enforces_pre_auth_caps_duplicate_idempotency_and_conflict_quarantine() {
        let limits = ReassemblerLimits {
            mtu: 24, // stride P = 8 bytes
            max_message_bytes: 32,
            max_chunks_per_transfer: 8,
            per_source_max_active_transfers: 2,
            per_source_max_reserved_bytes: 48,
            global_max_active_transfers: 4,
            global_max_reserved_bytes: 96,
            max_tracked_sources: 8,
            max_quarantine_tombstones: 8,
            transfer_timeout_ms: 1_000,
        };
        let mut reasm = ObservedSourceReassembler::<u32>::new(limits).unwrap();

        let msg = b"0123456789abcdefXYZ"; // 19 bytes -> 3 chunks (8, 8, 3)
        let frames =
            ChunkedFrameCodec::encode_message(42, limits.mtu, limits.max_message_bytes, 8, msg)
                .unwrap();
        assert_eq!(frames.len(), 3);

        // Ingest chunk 0 at t = 100ms.
        assert!(reasm
            .ingest_frame(1, 10, &frames[0], 100)
            .unwrap()
            .is_none());
        assert_eq!(reasm.active_transfer_count(), 1);
        assert_eq!(reasm.global_reserved_bytes(), 24); // min(3 * 8, 32) = 24

        // Corrupt CRC on a spoofed packet for the same transfer must be dropped WITHOUT evicting or quarantining!
        let mut bad_crc = frames[1].clone();
        bad_crc[16] ^= 0xFF;
        assert!(matches!(
            reasm.ingest_frame(1, 10, &bad_crc, 200),
            Err(ChunkError::CrcMismatch { .. })
        ));
        assert_eq!(reasm.active_transfer_count(), 1);
        assert_eq!(reasm.quarantine_tombstone_count(), 0);

        // Byte-identical duplicate chunk 0 at t = 900ms is an idempotent no-op and does NOT extend the 1100ms deadline.
        assert!(reasm
            .ingest_frame(1, 10, &frames[0], 900)
            .unwrap()
            .is_none());

        // Ingest chunk 1 and chunk 2 before deadline (t = 950ms) -> completes!
        assert!(reasm
            .ingest_frame(1, 10, &frames[1], 950)
            .unwrap()
            .is_none());
        let completed = reasm
            .ingest_frame(1, 10, &frames[2], 950)
            .unwrap()
            .expect("transfer must complete");
        assert_eq!(completed.payload, msg);
        assert_eq!(reasm.active_transfer_count(), 0);
        assert_eq!(reasm.global_reserved_bytes(), 0);

        // Conflict quarantine test: start transfer 77, then send a valid-CRC chunk 0 with different payload bytes.
        let frames_a = ChunkedFrameCodec::encode_message(
            77,
            limits.mtu,
            limits.max_message_bytes,
            8,
            b"AAAAAAAABBBBBBBB",
        )
        .unwrap();
        let frames_b = ChunkedFrameCodec::encode_message(
            77,
            limits.mtu,
            limits.max_message_bytes,
            8,
            b"CCCCCCCCBBBBBBBB",
        )
        .unwrap();
        assert!(reasm
            .ingest_frame(1, 10, &frames_a[0], 2_000)
            .unwrap()
            .is_none());
        assert_eq!(
            reasm.ingest_frame(1, 10, &frames_b[0], 2_100).unwrap_err(),
            ChunkError::ConflictingChunkPayload
        );
        // Active transfer was freed immediately and key is now quarantined until t = 3000ms!
        assert_eq!(reasm.active_transfer_count(), 0);
        assert_eq!(reasm.global_reserved_bytes(), 0);
        assert_eq!(reasm.quarantine_tombstone_count(), 1);
        assert_eq!(
            reasm.ingest_frame(1, 10, &frames_a[1], 2_500).unwrap_err(),
            ChunkError::TransferQuarantined
        );

        // At t = 3_000ms, the quarantine tombstone expires and transfer 77 can be reused cleanly.
        assert!(reasm
            .ingest_frame(1, 10, &frames_a[0], 3_000)
            .unwrap()
            .is_none());
        let ok_after_expiry = reasm
            .ingest_frame(1, 10, &frames_a[1], 3_010)
            .unwrap()
            .unwrap();
        assert_eq!(ok_after_expiry.payload, b"AAAAAAAABBBBBBBB");
    }
}
