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
    /// Observed source exceeded conflict quarantine threshold or saturated global tombstones (`ST-13`).
    SourceQuarantined,
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
            Self::SourceQuarantined => write!(
                f,
                "observed source is quarantined after repeated transfer conflicts"
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

/// Conservative fixed heap/map-node overhead charged per `BTreeMap` entry (`ST-13`).
pub const ESTIMATED_ENTRY_OVERHEAD_BYTES: usize = 64;
/// Fixed per-transfer metadata + map entry overhead charged to `global_reserved_bytes` (`ST-13`).
pub const ESTIMATED_TRANSFER_META_BYTES: usize = 64 + ESTIMATED_ENTRY_OVERHEAD_BYTES;
/// Fixed per-source accounting entry overhead charged to `global_reserved_bytes` (`ST-13`).
pub const ESTIMATED_SOURCE_ENTRY_BYTES: usize = 32 + ESTIMATED_ENTRY_OVERHEAD_BYTES;
/// Fixed per-tombstone / per-source-quarantine map entry overhead (`ST-13`).
pub const ESTIMATED_QUARANTINE_ENTRY_BYTES: usize = 16 + ESTIMATED_ENTRY_OVERHEAD_BYTES;

/// Chunk deduplication bitmap storage strategy (`ST-13`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReassemblyBitmapMode {
    /// Dynamically dimensioned `Vec<u64>` of `ceil(total_chunks / 64)` words supporting full
    /// `max_chunks_per_transfer = 768` (or larger configured `u16` caps).
    Dimensioned,
    /// Constrained embedded mode enforcing `max_chunks_per_transfer <= 256`.
    ConstrainedU256,
}

/// Resource and timeout limits for [`ObservedSourceReassembler`] (`ADR-0022` §2, `W11-03`, `ST-13`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReassemblerLimits {
    /// Maximum frame size $M$ in bytes (default `1400`, giving stride $P = 1384$).
    pub mtu: usize,
    /// Maximum reassembled message size $L_{\max}$ in bytes (default `1_048_576`).
    pub max_message_bytes: usize,
    /// Maximum chunks allowed per transfer (default `768`).
    pub max_chunks_per_transfer: u16,
    /// Deduplication bitmap storage strategy (default [`ReassemblyBitmapMode::Dimensioned`]).
    pub bitmap_mode: ReassemblyBitmapMode,
    /// Maximum concurrent in-flight transfers per observed source (default `4`).
    pub per_source_max_active_transfers: usize,
    /// Maximum reserved peak heap bytes (payload + bitmap + metadata + quarantine entries) per source (default `2_097_152`).
    pub per_source_max_reserved_bytes: usize,
    /// Maximum concurrent in-flight transfers across all sources (default `64`).
    pub global_max_active_transfers: usize,
    /// Maximum reserved peak heap bytes across all sources (default `16_777_216`).
    pub global_max_reserved_bytes: usize,
    /// Maximum distinct `(adapter_generation, observed_source)` entries tracked simultaneously (default `256`).
    pub max_tracked_sources: usize,
    /// Maximum conflict quarantine tombstones retained simultaneously (default `512`).
    pub max_quarantine_tombstones: usize,
    /// Conflicts per source before escalating the source into $O(\log S)$ cooldown quarantine (default `8`).
    pub max_quarantines_per_source: usize,
    /// Duration (in milliseconds) that an abusive source remains quarantined once escalated (default `10_000`).
    pub source_quarantine_ms: u64,
    /// Hard monotonic reassembly timeout in milliseconds from the first admitted chunk (default `5_000`).
    pub transfer_timeout_ms: u64,
}

impl Default for ReassemblerLimits {
    fn default() -> Self {
        Self {
            mtu: 1400,
            max_message_bytes: 1_048_576,
            max_chunks_per_transfer: 768,
            bitmap_mode: ReassemblyBitmapMode::Dimensioned,
            per_source_max_active_transfers: 4,
            per_source_max_reserved_bytes: 2 * 1024 * 1024,
            global_max_active_transfers: 64,
            global_max_reserved_bytes: 16 * 1024 * 1024,
            max_tracked_sources: 256,
            max_quarantine_tombstones: 512,
            max_quarantines_per_source: 8,
            source_quarantine_ms: 10_000,
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
    conflict_count: usize,
    active_tombstones: usize,
    quarantine_cooldown_active: bool,
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
struct ActiveTransferState {
    total_chunks: u16,
    reserved_bytes: usize,
    deadline_mono_ms: u64,
    received_count: usize,
    exact_total_len: Option<usize>,
    received_bitmap: Vec<u64>,
    payload_buf: Vec<u8>,
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy)]
struct QuarantineTombstone {
    deadline_mono_ms: u64,
}

/// Pre-authentication bounded chunk reassembler (`ADR-0022` §2, `W11-03`, `ST-13`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone)]
pub struct ObservedSourceReassembler<E: Clone + Eq + Ord> {
    limits: ReassemblerLimits,
    active_transfers: BTreeMap<ReassemblyKey<E>, ActiveTransferState>,
    source_stats: BTreeMap<(u64, E), SourceAccounting>,
    quarantine_tombstones: BTreeMap<ReassemblyKey<E>, QuarantineTombstone>,
    source_quarantine_until: BTreeMap<(u64, E), u64>,
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
        if matches!(limits.bitmap_mode, ReassemblyBitmapMode::ConstrainedU256)
            && limits.max_chunks_per_transfer > 256
        {
            return Err(ChunkError::InvalidChunkBounds);
        }
        Ok(Self {
            limits,
            active_transfers: BTreeMap::new(),
            source_stats: BTreeMap::new(),
            quarantine_tombstones: BTreeMap::new(),
            source_quarantine_until: BTreeMap::new(),
            global_reserved_bytes: 0,
        })
    }

    #[inline]
    fn bitmap_word_count(total_chunks: u16) -> usize {
        (total_chunks as usize).div_ceil(64)
    }

    /// Compute peak heap reservation for an active transfer including payload buffer, dimensioned
    /// `Vec<u64>` bitmap, and transfer metadata/entry overhead (`ST-13`).
    pub fn transfer_memory_charge(
        reserved_payload_bytes: usize,
        total_chunks: u16,
    ) -> Option<usize> {
        let bitmap_bytes = Self::bitmap_word_count(total_chunks).checked_mul(8)?;
        reserved_payload_bytes
            .checked_add(bitmap_bytes)?
            .checked_add(ESTIMATED_TRANSFER_META_BYTES)
    }

    /// Current number of active in-flight multi-chunk transfers.
    pub fn active_transfer_count(&self) -> usize {
        self.active_transfers.len()
    }

    /// Current global reserved peak heap bytes across transfers, bitmaps, sources, and quarantines.
    pub const fn global_reserved_bytes(&self) -> usize {
        self.global_reserved_bytes
    }

    /// Current number of active conflict quarantine tombstones.
    pub fn quarantine_tombstone_count(&self) -> usize {
        self.quarantine_tombstones.len()
    }

    /// Current number of observed sources under active cooldown quarantine.
    pub fn quarantined_source_count(&self) -> usize {
        self.source_quarantine_until.len()
    }

    fn prune_source_if_idle(&mut self, src_key: &(u64, E)) {
        let should_remove = self.source_stats.get(src_key).is_some_and(|stats| {
            stats.active_transfers == 0
                && stats.active_tombstones == 0
                && !stats.quarantine_cooldown_active
        });
        if should_remove {
            if let Some(removed) = self.source_stats.remove(src_key) {
                debug_assert_eq!(removed.reserved_bytes, ESTIMATED_SOURCE_ENTRY_BYTES);
                self.global_reserved_bytes = self
                    .global_reserved_bytes
                    .saturating_sub(ESTIMATED_SOURCE_ENTRY_BYTES);
            }
        }
    }

    /// Evict all active transfers, quarantine tombstones, and source cooldowns whose hard deadline
    /// has expired (`now_mono_ms >= deadline_mono_ms`).
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
            if self.quarantine_tombstones.remove(&key).is_some() {
                self.global_reserved_bytes = self
                    .global_reserved_bytes
                    .saturating_sub(ESTIMATED_QUARANTINE_ENTRY_BYTES);
                let src_key = (key.adapter_generation, key.observed_source);
                if let Some(stats) = self.source_stats.get_mut(&src_key) {
                    stats.active_tombstones = stats.active_tombstones.saturating_sub(1);
                    stats.reserved_bytes = stats
                        .reserved_bytes
                        .saturating_sub(ESTIMATED_QUARANTINE_ENTRY_BYTES);
                }
                self.prune_source_if_idle(&src_key);
            }
        }

        let expired_sources: Vec<(u64, E)> = self
            .source_quarantine_until
            .iter()
            .filter_map(|(src, &until)| {
                if now_mono_ms >= until {
                    Some(src.clone())
                } else {
                    None
                }
            })
            .collect();

        for src_key in expired_sources {
            if self.source_quarantine_until.remove(&src_key).is_some() {
                self.global_reserved_bytes = self
                    .global_reserved_bytes
                    .saturating_sub(ESTIMATED_QUARANTINE_ENTRY_BYTES);
                if let Some(stats) = self.source_stats.get_mut(&src_key) {
                    stats.quarantine_cooldown_active = false;
                    stats.conflict_count = 0;
                    stats.reserved_bytes = stats
                        .reserved_bytes
                        .saturating_sub(ESTIMATED_QUARANTINE_ENTRY_BYTES);
                }
                self.prune_source_if_idle(&src_key);
            }
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
        }
        self.prune_source_if_idle(&src_key);
        Some(removed)
    }

    /// Quarantine a conflicting transfer WITHOUT evicting unexpired tombstones (`ST-13`).
    ///
    /// Escalates `(adapter_generation, observed_source)` to `source_quarantine_until` whenever
    /// per-source conflicts reach `max_quarantines_per_source` OR global `quarantine_tombstones`
    /// is saturated with unexpired tombstones.
    fn quarantine_conflict(
        &mut self,
        key: ReassemblyKey<E>,
        deadline_mono_ms: u64,
        now_mono_ms: u64,
    ) {
        // Remove the active transfer first so its reserved_bytes are freed, while retaining
        // conflict_count on SourceAccounting below.
        let src_key = (key.adapter_generation, key.observed_source.clone());
        if let Some(removed) = self.active_transfers.remove(&key) {
            self.global_reserved_bytes = self
                .global_reserved_bytes
                .saturating_sub(removed.reserved_bytes);
            if let Some(stats) = self.source_stats.get_mut(&src_key) {
                stats.active_transfers = stats.active_transfers.saturating_sub(1);
                stats.reserved_bytes = stats.reserved_bytes.saturating_sub(removed.reserved_bytes);
            }
        }

        let Some(stats) = self.source_stats.get_mut(&src_key) else {
            return;
        };
        stats.conflict_count = stats.conflict_count.saturating_add(1);

        let tombstones_saturated = if self.limits.max_quarantine_tombstones == 0 {
            true
        } else if self.quarantine_tombstones.contains_key(&key) {
            false
        } else if self.quarantine_tombstones.len() < self.limits.max_quarantine_tombstones {
            self.quarantine_tombstones
                .insert(key, QuarantineTombstone { deadline_mono_ms });
            stats.active_tombstones += 1;
            stats.reserved_bytes = stats
                .reserved_bytes
                .saturating_add(ESTIMATED_QUARANTINE_ENTRY_BYTES);
            self.global_reserved_bytes = self
                .global_reserved_bytes
                .saturating_add(ESTIMATED_QUARANTINE_ENTRY_BYTES);
            false
        } else {
            // Never evict unexpired tombstones when an attacker floods conflict IDs!
            true
        };

        let exceeded_source_conflicts = self.limits.max_quarantines_per_source > 0
            && stats.conflict_count >= self.limits.max_quarantines_per_source;

        if tombstones_saturated || exceeded_source_conflicts {
            let cooldown_until = now_mono_ms.saturating_add(self.limits.source_quarantine_ms);
            if cooldown_until > now_mono_ms {
                if let Some(existing_until) = self.source_quarantine_until.get_mut(&src_key) {
                    *existing_until = (*existing_until).max(cooldown_until);
                } else {
                    self.source_quarantine_until
                        .insert(src_key.clone(), cooldown_until);
                    stats.quarantine_cooldown_active = true;
                    stats.reserved_bytes = stats
                        .reserved_bytes
                        .saturating_add(ESTIMATED_QUARANTINE_ENTRY_BYTES);
                    self.global_reserved_bytes = self
                        .global_reserved_bytes
                        .saturating_add(ESTIMATED_QUARANTINE_ENTRY_BYTES);
                }
            }
        }

        self.prune_source_if_idle(&src_key);
    }

    /// Ingest a single raw frame from `(adapter_generation, observed_source)` at monotonic
    /// timestamp `now_mono_ms` (`ADR-0022` §2, `ST-13`).
    pub fn ingest_frame(
        &mut self,
        adapter_generation: u64,
        observed_source: E,
        frame_bytes: &[u8],
        now_mono_ms: u64,
    ) -> Result<Option<ReassembledTransfer<E>>, ChunkError> {
        // 1. Reclaim expired transfers, tombstones, and source cooldowns first.
        self.evict_expired(now_mono_ms);

        // 2. O(log S) source quarantine check BEFORE decoding or allocating buffers.
        let src_key = (adapter_generation, observed_source.clone());
        if let Some(&until) = self.source_quarantine_until.get(&src_key) {
            if now_mono_ms < until {
                return Err(ChunkError::SourceQuarantined);
            }
        }

        // 3. Decode and verify header + CRC-16/IBM-3740 BEFORE touching any transfer state.
        let max_chunks = match self.limits.bitmap_mode {
            ReassemblyBitmapMode::Dimensioned => self.limits.max_chunks_per_transfer,
            ReassemblyBitmapMode::ConstrainedU256 => self.limits.max_chunks_per_transfer.min(256),
        };
        let decoded = ChunkedFrameCodec::decode_frame_with_limits(
            frame_bytes,
            self.limits.mtu,
            self.limits.max_message_bytes,
            max_chunks,
        )?;
        let stride_p = ChunkedFrameCodec::stride_for_mtu(self.limits.mtu)?;

        let key = ReassemblyKey {
            adapter_generation,
            observed_source,
            transfer_id: decoded.header.transfer_id,
        };

        // 4. Reject if transfer key is under conflict quarantine.
        if self.quarantine_tombstones.contains_key(&key) {
            return Err(ChunkError::TransferQuarantined);
        }

        // 5. Existing active transfer handling.
        if let Some(existing) = self.active_transfers.get_mut(&key) {
            if decoded.header.total_chunks != existing.total_chunks {
                let deadline = existing.deadline_mono_ms;
                self.quarantine_conflict(key, deadline, now_mono_ms);
                return Err(ChunkError::ConflictingTotalChunks);
            }

            let idx = decoded.header.chunk_idx as usize;
            let start = idx
                .checked_mul(stride_p)
                .ok_or(ChunkError::MessageTooLarge)?;
            let end = start
                .checked_add(decoded.payload.len())
                .ok_or(ChunkError::MessageTooLarge)?;
            if end > existing.payload_buf.len() {
                let deadline = existing.deadline_mono_ms;
                self.quarantine_conflict(key, deadline, now_mono_ms);
                return Err(ChunkError::MessageTooLarge);
            }

            let word_idx = idx / 64;
            let bit_mask = 1u64 << (idx % 64);
            if (existing.received_bitmap[word_idx] & bit_mask) != 0 {
                if existing.payload_buf[start..end] == *decoded.payload {
                    // Byte-identical duplicate chunk: idempotent no-op without extending deadline.
                    return Ok(None);
                } else {
                    let deadline = existing.deadline_mono_ms;
                    self.quarantine_conflict(key, deadline, now_mono_ms);
                    return Err(ChunkError::ConflictingChunkPayload);
                }
            }

            existing.payload_buf[start..end].copy_from_slice(decoded.payload);
            existing.received_bitmap[word_idx] |= bit_mask;
            existing.received_count += 1;
            if decoded.header.chunk_idx + 1 == existing.total_chunks {
                existing.exact_total_len = Some(end);
            }

            if existing.received_count == existing.total_chunks as usize {
                let final_len = existing
                    .exact_total_len
                    .ok_or(ChunkError::MessageTooLarge)?;
                if final_len > self.limits.max_message_bytes {
                    self.remove_active_transfer(&key);
                    return Err(ChunkError::MessageTooLarge);
                }
                // Zero-copy ownership move via mem::take (no second payload buffer allocation).
                let mut payload = core::mem::take(&mut existing.payload_buf);
                payload.truncate(final_len);
                self.remove_active_transfer(&key);
                return Ok(Some(ReassembledTransfer { key, payload }));
            }

            return Ok(None);
        }

        // 6. New transfer admission: account payload, bitmap, metadata, and source entry inside caps.
        let is_new_source = !self.source_stats.contains_key(&src_key);
        if is_new_source && self.source_stats.len() >= self.limits.max_tracked_sources {
            return Err(ChunkError::ResourceBudgetExceeded);
        }

        let current_src = self
            .source_stats
            .get(&src_key)
            .cloned()
            .unwrap_or(SourceAccounting {
                active_transfers: 0,
                reserved_bytes: 0,
                conflict_count: 0,
                active_tombstones: 0,
                quarantine_cooldown_active: false,
            });
        if current_src.active_transfers >= self.limits.per_source_max_active_transfers
            || self.active_transfers.len() >= self.limits.global_max_active_transfers
        {
            return Err(ChunkError::ResourceBudgetExceeded);
        }

        let new_source_charge = if is_new_source {
            ESTIMATED_SOURCE_ENTRY_BYTES
        } else {
            0
        };

        // Fast-path for a 1-chunk message after verifying admission budgets.
        if decoded.header.total_chunks == 1 {
            let one_chunk_peak = decoded
                .payload
                .len()
                .checked_add(new_source_charge)
                .ok_or(ChunkError::ResourceBudgetExceeded)?;
            if current_src.reserved_bytes.saturating_add(one_chunk_peak)
                > self.limits.per_source_max_reserved_bytes
                || self.global_reserved_bytes.saturating_add(one_chunk_peak)
                    > self.limits.global_max_reserved_bytes
            {
                return Err(ChunkError::ResourceBudgetExceeded);
            }
            return Ok(Some(ReassembledTransfer {
                key,
                payload: decoded.payload.to_vec(),
            }));
        }

        let payload_capacity = (decoded.header.total_chunks as usize)
            .checked_mul(stride_p)
            .map_or(self.limits.max_message_bytes, |n| {
                n.min(self.limits.max_message_bytes)
            });
        let transfer_charge =
            Self::transfer_memory_charge(payload_capacity, decoded.header.total_chunks)
                .ok_or(ChunkError::ResourceBudgetExceeded)?;
        let total_admission_charge = transfer_charge
            .checked_add(new_source_charge)
            .ok_or(ChunkError::ResourceBudgetExceeded)?;

        if current_src
            .reserved_bytes
            .saturating_add(total_admission_charge)
            > self.limits.per_source_max_reserved_bytes
            || self
                .global_reserved_bytes
                .saturating_add(total_admission_charge)
                > self.limits.global_max_reserved_bytes
        {
            return Err(ChunkError::ResourceBudgetExceeded);
        }

        let idx = decoded.header.chunk_idx as usize;
        let start = idx
            .checked_mul(stride_p)
            .ok_or(ChunkError::MessageTooLarge)?;
        let end = start
            .checked_add(decoded.payload.len())
            .ok_or(ChunkError::MessageTooLarge)?;
        if end > payload_capacity {
            return Err(ChunkError::MessageTooLarge);
        }

        let mut payload_buf = vec![0u8; payload_capacity];
        payload_buf[start..end].copy_from_slice(decoded.payload);
        let mut received_bitmap = vec![0u64; Self::bitmap_word_count(decoded.header.total_chunks)];
        received_bitmap[idx / 64] |= 1u64 << (idx % 64);
        let exact_total_len = if decoded.header.chunk_idx + 1 == decoded.header.total_chunks {
            Some(end)
        } else {
            None
        };

        let deadline_mono_ms = now_mono_ms.saturating_add(self.limits.transfer_timeout_ms);
        self.active_transfers.insert(
            key,
            ActiveTransferState {
                total_chunks: decoded.header.total_chunks,
                reserved_bytes: transfer_charge,
                deadline_mono_ms,
                received_count: 1,
                exact_total_len,
                received_bitmap,
                payload_buf,
            },
        );
        self.global_reserved_bytes += total_admission_charge;
        let entry = self
            .source_stats
            .entry(src_key)
            .or_insert_with(|| SourceAccounting {
                active_transfers: 0,
                reserved_bytes: ESTIMATED_SOURCE_ENTRY_BYTES,
                conflict_count: 0,
                active_tombstones: 0,
                quarantine_cooldown_active: false,
            });
        entry.active_transfers += 1;
        entry.reserved_bytes += transfer_charge;

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
            per_source_max_reserved_bytes: 1024,
            global_max_active_transfers: 4,
            global_max_reserved_bytes: 2048,
            max_tracked_sources: 8,
            max_quarantine_tombstones: 8,
            ..ReassemblerLimits::default()
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
        // Reserved bytes include payload (min(3*8, 32)=24) + 1*8 bitmap + transfer meta + source entry.
        let expected_active_charge =
            ObservedSourceReassembler::<u32>::transfer_memory_charge(24, 3).unwrap()
                + ESTIMATED_SOURCE_ENTRY_BYTES;
        assert_eq!(reasm.global_reserved_bytes(), expected_active_charge);

        // Corrupt CRC on a spoofed packet for the same transfer must be dropped WITHOUT evicting or quarantining!
        let mut bad_crc = frames[1].clone();
        bad_crc[16] ^= 0xFF;
        assert!(matches!(
            reasm.ingest_frame(1, 10, &bad_crc, 200),
            Err(ChunkError::CrcMismatch { .. })
        ));
        assert_eq!(reasm.active_transfer_count(), 1);
        assert_eq!(reasm.quarantine_tombstone_count(), 0);

        // Byte-identical duplicate chunk 0 at t = 900ms is an idempotent no-op and does NOT extend the 5100ms deadline.
        assert!(reasm
            .ingest_frame(1, 10, &frames[0], 900)
            .unwrap()
            .is_none());

        // Ingest chunk 1 and chunk 2 out-of-order before deadline (t = 950ms) -> completes!
        assert!(reasm
            .ingest_frame(1, 10, &frames[2], 940)
            .unwrap()
            .is_none());
        let completed = reasm
            .ingest_frame(1, 10, &frames[1], 950)
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
        // Active transfer was freed immediately and only source + quarantine tombstone remain reserved!
        assert_eq!(reasm.active_transfer_count(), 0);
        assert_eq!(reasm.quarantine_tombstone_count(), 1);
        assert_eq!(
            reasm.global_reserved_bytes(),
            ESTIMATED_SOURCE_ENTRY_BYTES + ESTIMATED_QUARANTINE_ENTRY_BYTES
        );
        assert_eq!(
            reasm.ingest_frame(1, 10, &frames_a[1], 2_500).unwrap_err(),
            ChunkError::TransferQuarantined
        );

        // At t = 7_000ms (2_000 + default 5_000 timeout), tombstone expires and reservation returns to 0 on reuse.
        assert!(reasm
            .ingest_frame(1, 10, &frames_a[0], 7_000)
            .unwrap()
            .is_none());
        let ok_after_expiry = reasm
            .ingest_frame(1, 10, &frames_a[1], 7_010)
            .unwrap()
            .unwrap();
        assert_eq!(ok_after_expiry.payload, b"AAAAAAAABBBBBBBB");
        assert_eq!(reasm.global_reserved_bytes(), 0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn reassembler_dimensioned_bitmap_supports_768_chunks_at_1_byte_stride_and_constrained_u256_rejects(
    ) {
        // Adversarial 1-byte stride (mtu = 17 => P = 1) with N = 768 chunks (12 u64 bitmap words)
        let limits = ReassemblerLimits {
            mtu: 17,
            max_message_bytes: 1024,
            max_chunks_per_transfer: 768,
            bitmap_mode: ReassemblyBitmapMode::Dimensioned,
            per_source_max_reserved_bytes: 4096,
            global_max_reserved_bytes: 8192,
            ..ReassemblerLimits::default()
        };
        let mut reasm = ObservedSourceReassembler::<u32>::new(limits).unwrap();
        let msg: Vec<u8> = (0..768).map(|i| (i % 251) as u8).collect();
        let frames =
            ChunkedFrameCodec::encode_message(901, limits.mtu, limits.max_message_bytes, 768, &msg)
                .unwrap();
        assert_eq!(frames.len(), 768);

        // Deliver out-of-order (reverse) plus duplicate on chunk 700 (word 10, beyond U256's 4 words!)
        for f in frames.iter().rev() {
            if let Some(done) = reasm.ingest_frame(1, 7, f, 100).unwrap() {
                assert_eq!(done.payload, msg);
                assert_eq!(reasm.global_reserved_bytes(), 0);
                return;
            }
            // Inject duplicate on chunk 700 mid-stream before transfer finishes
            if f == &frames[700] {
                assert!(reasm
                    .ingest_frame(1, 7, &frames[700], 101)
                    .unwrap()
                    .is_none());
            }
        }
        panic!("768-chunk 1-byte-stride transfer did not complete");
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn reassembler_constrained_u256_mode_and_non_evicting_quarantine_flood() {
        // ConstrainedU256 rejects >256 configuration at init and rejects 257-chunk frames
        let bad_cfg = ReassemblerLimits {
            bitmap_mode: ReassemblyBitmapMode::ConstrainedU256,
            max_chunks_per_transfer: 257,
            ..ReassemblerLimits::default()
        };
        assert_eq!(
            ObservedSourceReassembler::<u32>::new(bad_cfg).unwrap_err(),
            ChunkError::InvalidChunkBounds
        );

        let u256_cfg = ReassemblerLimits {
            mtu: 17,
            max_message_bytes: 512,
            max_chunks_per_transfer: 256,
            bitmap_mode: ReassemblyBitmapMode::ConstrainedU256,
            ..ReassemblerLimits::default()
        };
        let mut u256_reasm = ObservedSourceReassembler::<u32>::new(u256_cfg).unwrap();
        let big_msg = vec![0xAAu8; 257];
        let big_frames = ChunkedFrameCodec::encode_message(1, 17, 512, 257, &big_msg).unwrap();
        assert_eq!(
            u256_reasm
                .ingest_frame(1, 1, &big_frames[0], 100)
                .unwrap_err(),
            ChunkError::InvalidChunkBounds
        );

        // Non-evicting quarantine flood across sources:
        let limits = ReassemblerLimits {
            mtu: 24,
            max_message_bytes: 64,
            max_chunks_per_transfer: 8,
            max_quarantine_tombstones: 2,
            max_quarantines_per_source: 8,
            source_quarantine_ms: 5_000,
            transfer_timeout_ms: 2_000,
            per_source_max_reserved_bytes: 2048,
            global_max_reserved_bytes: 8192,
            ..ReassemblerLimits::default()
        };
        let mut reasm = ObservedSourceReassembler::<u32>::new(limits).unwrap();

        let mk_conflict =
            |reasm: &mut ObservedSourceReassembler<u32>, src: u32, tid: u32, ts: u64| {
                let fa =
                    ChunkedFrameCodec::encode_message(tid, 24, 64, 8, b"AAAAAAAABBBBBBBB").unwrap();
                let fb =
                    ChunkedFrameCodec::encode_message(tid, 24, 64, 8, b"CCCCCCCCBBBBBBBB").unwrap();
                assert!(reasm.ingest_frame(1, src, &fa[0], ts).unwrap().is_none());
                assert_eq!(
                    reasm.ingest_frame(1, src, &fb[0], ts + 1).unwrap_err(),
                    ChunkError::ConflictingChunkPayload
                );
                fa
            };

        let s1_frames = mk_conflict(&mut reasm, 101, 1, 100);
        let s2_frames = mk_conflict(&mut reasm, 102, 2, 110);
        assert_eq!(reasm.quarantine_tombstone_count(), 2);

        // Attacker 103 conflicts when tombstone table (cap=2) is already full:
        // unexpired tombstones for 101 & 102 MUST NOT be evicted, and 103 is source-quarantined!
        let _s3_frames = mk_conflict(&mut reasm, 103, 3, 120);
        assert_eq!(reasm.quarantine_tombstone_count(), 2);
        assert_eq!(reasm.quarantined_source_count(), 1);
        assert_eq!(
            reasm.ingest_frame(1, 101, &s1_frames[1], 130).unwrap_err(),
            ChunkError::TransferQuarantined
        );
        assert_eq!(
            reasm.ingest_frame(1, 102, &s2_frames[1], 130).unwrap_err(),
            ChunkError::TransferQuarantined
        );
        // Any subsequent frame from 103 (even brand new transfer_id=99 or corrupt CRC) hits O(log S) SourceQuarantined!
        assert_eq!(
            reasm.ingest_frame(1, 103, b"junk", 131).unwrap_err(),
            ChunkError::SourceQuarantined
        );

        // Honest source 200 completes a multi-chunk transfer unaffected!
        let honest = ChunkedFrameCodec::encode_message(77, 24, 64, 8, b"HONEST__PAYLOAD!").unwrap();
        assert!(reasm
            .ingest_frame(1, 200, &honest[0], 140)
            .unwrap()
            .is_none());
        let done = reasm
            .ingest_frame(1, 200, &honest[1], 141)
            .unwrap()
            .unwrap();
        assert_eq!(done.payload, b"HONEST__PAYLOAD!");

        // After all quarantine deadlines expire (t = 6000ms >= 121 + 5000ms), all reserved bytes return to 0!
        reasm.evict_expired(6_000);
        assert_eq!(reasm.quarantine_tombstone_count(), 0);
        assert_eq!(reasm.quarantined_source_count(), 0);
        assert_eq!(reasm.global_reserved_bytes(), 0);
    }
}
