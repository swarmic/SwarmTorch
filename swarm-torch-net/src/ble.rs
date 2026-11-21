//! `no_std` BLE GATT packet transport (`ADR-0022`, `Wave 16E-2`).
//!
//! **Maturity Tier:** `Simulation verified` (hardware-agnostic HAL trait + in-memory simulated
//! GATT link tests; physical BLE controller/GATT stacks such as `trouble-host`, `nrf-softdevice`,
//! or NimBLE implement [`BleGattHal`] out of tree).
//!
//! Enforces **Bluetooth Core Specification Vol. 3, Part F §3.2.8–§3.2.9**:
//! - Minimum LE `ATT_MTU` is [`BLE_MIN_ATT_MTU_BYTES`] (`23` bytes -> `20`-byte notification payload).
//! - Canonical BLE 4.2/5.x DLE `ATT_MTU` is [`BLE_CANONICAL_ATT_MTU_BYTES`] (`247` bytes ->
//!   `244`-byte notification payload -> [`BLE_CANONICAL_CHUNK_STRIDE_BYTES`] = `228`-byte
//!   [`ChunkedFrameCodec`] stride).
//! - Maximum ATT attribute value length is strictly capped at [`BLE_MAX_ATT_VALUE_BYTES`]
//!   (`512` bytes) per Vol. 3 Part F §3.2.9 — even when `ATT_MTU = 517` (`517 - 3 = 514 > 512`),
//!   [`effective_ble_payload_limit_v1`] returns `512` and rejects `513`/`514`-byte frames before
//!   touching HAL TX queues or credits.
//! - Static [`PacketCapabilities::max_frame_bytes`] reports the conservative transport floor
//!   derived from `min_peer_att_mtu`, while [`BlePacketTransportV1::try_send_frame`] and
//!   [`BlePacketTransportV1::send_chunked_message_v1`] enforce each peer's live negotiated
//!   effective payload ceiling and available TX credits pre-TX.

use crate::chunked::{ChunkError, ChunkedFrameCodec, CHUNK_HEADER_LEN};
use crate::traits::{
    BandwidthClass, PacketCapabilities, PacketRecvError, PacketSendError, PacketTransport,
    ReceivedFrame, ReliabilityClass,
};

/// Minimum LE `ATT_MTU` permitted by Bluetooth Core Spec Vol. 3, Part F §3.2.8 (`23` bytes).
pub const BLE_MIN_ATT_MTU_BYTES: u16 = 23;

/// Canonical BLE 4.2/5.x Data Length Extension (DLE) `ATT_MTU` (`247` bytes = 251B LL PDU - 4B L2CAP).
pub const BLE_CANONICAL_ATT_MTU_BYTES: u16 = 247;

/// Upper `ATT_MTU` cap used for single-notification framing (`517` bytes).
pub const BLE_MAX_ATT_MTU_BYTES: u16 = 517;

/// ATT opcode (`1` byte) + attribute handle (`2` bytes) overhead for `ATT_HANDLE_VALUE_NTF` and
/// `ATT_WRITE_CMD` (`3` bytes).
pub const BLE_ATT_NOTIFY_OVERHEAD_BYTES: u16 = 3;

/// Hard maximum GATT attribute value length (`512` bytes) per Bluetooth Core Spec Vol. 3, Part F §3.2.9.
///
/// Even at `ATT_MTU = 517` (`517 - 3 = 514`), a GATT characteristic value MUST NOT exceed `512` bytes.
pub const BLE_MAX_ATT_VALUE_BYTES: usize = 512;

/// Minimum characteristic value capacity (`17` bytes) required to carry a 16-byte `ChunkedFrameCodec`
/// header plus at least 1 byte of payload.
pub const BLE_MIN_CHARACTERISTIC_VALUE_BYTES: usize = CHUNK_HEADER_LEN + 1;

/// Canonical BLE 247-byte `ATT_MTU` notification payload capacity (`247 - 3 = 244` bytes).
pub const BLE_CANONICAL_NOTIFY_PAYLOAD_BYTES: usize =
    (BLE_CANONICAL_ATT_MTU_BYTES - BLE_ATT_NOTIFY_OVERHEAD_BYTES) as usize;

/// Canonical [`ChunkedFrameCodec`] payload stride for `ATT_MTU = 247` ($P = 244 - 16 = 228$ bytes).
pub const BLE_CANONICAL_CHUNK_STRIDE_BYTES: usize =
    BLE_CANONICAL_NOTIFY_PAYLOAD_BYTES - CHUNK_HEADER_LEN;

/// Default maximum reassembled message size advertised by [`BlePacketTransportV1`] (`64 KiB`).
pub const BLE_DEFAULT_MAX_MESSAGE_BYTES: usize = 65_536;

/// Unauthenticated 16-bit observed BLE GATT connection handle (`ADR-0022` §1.2).
///
/// Never conflated with a cryptographically verified `PeerId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BleEndpointV1(pub u16);

impl BleEndpointV1 {
    /// Maximum standard 12-bit Bluetooth Core Spec HCI LE connection handle (`0x0EFF`).
    pub const MAX_HCI_CONNECTION_HANDLE: u16 = 0x0EFF;

    /// Return `true` if this handle falls inside the standard Bluetooth HCI LE handle range (`0x0000..=0x0EFF`).
    #[inline]
    pub const fn is_standard_hci_handle(self) -> bool {
        self.0 <= Self::MAX_HCI_CONNECTION_HANDLE
    }
}

/// Errors returned by BLE configuration, per-peer ATT MTU enforcement, and HAL integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BleTransportError<E> {
    /// Invalid transport configuration (`min_peer_att_mtu`, `characteristic_max_bytes`, or `max_message_bytes`).
    InvalidConfig,
    /// Peer reported a spec-violating `ATT_MTU < 23`.
    InvalidAttMtu {
        /// Reported peer `ATT_MTU` in octets.
        att_mtu: u16,
    },
    /// Peer's negotiated `ATT_MTU` fell below this transport's configured static floor.
    PeerMtuBelowFloor {
        /// Observed GATT peer connection handle.
        peer: BleEndpointV1,
        /// Peer's negotiated `ATT_MTU`.
        peer_att_mtu: u16,
        /// Transport's configured minimum required `ATT_MTU`.
        min_required_att_mtu: u16,
    },
    /// Target GATT peer is unknown or disconnected.
    PeerDisconnected(BleEndpointV1),
    /// Chunked framing/bounds error in [`BlePacketTransportV1::send_chunked_message_v1`].
    Chunk(ChunkError),
    /// Underlying BLE controller/GATT HAL error.
    Hal(E),
}

impl<E: core::fmt::Debug> core::fmt::Display for BleTransportError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid BLE GATT transport configuration"),
            Self::InvalidAttMtu { att_mtu } => write!(
                f,
                "invalid BLE ATT_MTU {att_mtu}: minimum spec ATT_MTU is {BLE_MIN_ATT_MTU_BYTES}"
            ),
            Self::PeerMtuBelowFloor {
                peer,
                peer_att_mtu,
                min_required_att_mtu,
            } => write!(
                f,
                "BLE peer 0x{:04x} ATT_MTU {peer_att_mtu} is below configured floor {min_required_att_mtu}",
                peer.0
            ),
            Self::PeerDisconnected(peer) => {
                write!(f, "BLE peer 0x{:04x} is disconnected", peer.0)
            }
            Self::Chunk(err) => write!(f, "BLE chunked framing error: {err}"),
            Self::Hal(err) => write!(f, "BLE GATT HAL error: {err:?}"),
        }
    }
}

#[cfg(feature = "std")]
impl<E: core::fmt::Debug> std::error::Error for BleTransportError<E> {}

/// Compute the effective single-notification / write-without-response GATT payload limit in bytes
/// per **Bluetooth Core Spec Vol. 3, Part F §3.2.8–§3.2.9**:
///
/// $$\min\bigl(\min(\text{att\_mtu},\ 517) - 3,\ 512,\ \text{characteristic\_max\_bytes}\bigr)$$
///
/// Fails closed if `att_mtu < 23` ([`BleTransportError::InvalidAttMtu`]) or
/// `characteristic_max_bytes == 0` ([`BleTransportError::InvalidConfig`]).
/// Even when `att_mtu == 517` (`517 - 3 = 514`) and `characteristic_max_bytes > 512`, the returned
/// ceiling is strictly clamped to [`BLE_MAX_ATT_VALUE_BYTES`] (`512` bytes).
pub fn effective_ble_payload_limit_v1(
    att_mtu: u16,
    characteristic_max_bytes: usize,
) -> Result<usize, BleTransportError<()>> {
    if att_mtu < BLE_MIN_ATT_MTU_BYTES {
        return Err(BleTransportError::InvalidAttMtu { att_mtu });
    }
    if characteristic_max_bytes == 0 {
        return Err(BleTransportError::InvalidConfig);
    }
    let att_payload =
        usize::from(att_mtu.min(BLE_MAX_ATT_MTU_BYTES) - BLE_ATT_NOTIFY_OVERHEAD_BYTES);
    Ok(att_payload
        .min(BLE_MAX_ATT_VALUE_BYTES)
        .min(characteristic_max_bytes))
}

/// Non-blocking error returned by a low-level [`BleGattHal`] stack driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BleHalError<E> {
    /// Peer notification/write TX credit pool or HCI buffer is exhausted; frame was not sent.
    WouldBlock,
    /// Connection handle dropped or notifications are not subscribed (CCCD disabled).
    Disconnected,
    /// Hardware/HCI/GATT driver error.
    Gatt(E),
}

/// Hardware-agnostic, non-blocking `no_std` BLE GATT HAL trait (`ADR-0022`, `Wave 16E-2`).
///
/// Downstream firmware crates implement this trait over `trouble-host`, `nrf-softdevice`,
/// `esp-radio`, or NimBLE GATT characteristics (unacknowledged `ATT_HANDLE_VALUE_NTF` /
/// `ATT_WRITE_CMD`).
pub trait BleGattHal {
    /// Stack-specific GATT/HCI error type.
    type Error: core::fmt::Debug;

    /// Query the peer's currently negotiated `ATT_MTU` in octets, or `None` if disconnected.
    fn peer_att_mtu(&self, peer: BleEndpointV1) -> Option<u16>;

    /// Query the number of currently available TX notification/write credits for `peer`,
    /// or `None` if `peer` is disconnected.
    fn tx_credits(&self, peer: BleEndpointV1) -> Option<u16>;

    /// Attempt to send one GATT notification or Write Without Response payload (`frame.len() <= 512`)
    /// to `peer` without blocking.
    ///
    /// Returning `Err(BleHalError::WouldBlock)`, `Err(BleHalError::Disconnected)`, or
    /// `Err(BleHalError::Gatt(_))` signals that no ATT PDU was transmitted.
    fn try_notify_frame(
        &mut self,
        peer: BleEndpointV1,
        frame: &[u8],
    ) -> Result<(), BleHalError<Self::Error>>;

    /// Attempt to receive one incoming GATT characteristic payload into `buf`
    /// (`buf.len() >= BLE_MAX_ATT_VALUE_BYTES`) without blocking.
    ///
    /// Returns `Ok(None)` when no payload is queued, or `Ok(Some((source, len)))` when `len` bytes
    /// were written into `buf[..len]`.
    fn try_recv_frame(
        &mut self,
        buf: &mut [u8],
    ) -> Result<Option<(BleEndpointV1, usize)>, Self::Error>;
}

/// Configuration for [`BlePacketTransportV1`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BleTransportConfigV1 {
    /// Minimum peer `ATT_MTU` accepted by this transport (`23..=517`, default `23`).
    ///
    /// Determines the conservative static floor reported by [`PacketCapabilities::max_frame_bytes`].
    /// Peers negotiating a lower `ATT_MTU` fail closed with [`BleTransportError::PeerMtuBelowFloor`].
    pub min_peer_att_mtu: u16,
    /// Maximum byte length of the underlying GATT characteristic value (`17..=512`, default `512`).
    pub characteristic_max_bytes: usize,
    /// Maximum reassembled message size advertised by [`PacketCapabilities::max_message_bytes`].
    pub max_message_bytes: usize,
}

impl Default for BleTransportConfigV1 {
    fn default() -> Self {
        Self {
            min_peer_att_mtu: BLE_MIN_ATT_MTU_BYTES,
            characteristic_max_bytes: BLE_MAX_ATT_VALUE_BYTES,
            max_message_bytes: BLE_DEFAULT_MAX_MESSAGE_BYTES,
        }
    }
}

impl BleTransportConfigV1 {
    /// Canonical BLE 4.2/5.x DLE configuration requiring `ATT_MTU >= 247` (`244`B frame floor,
    /// `228`B [`ChunkedFrameCodec`] stride, `512`B characteristic cap).
    pub const fn canonical_dle_247() -> Self {
        Self {
            min_peer_att_mtu: BLE_CANONICAL_ATT_MTU_BYTES,
            characteristic_max_bytes: BLE_MAX_ATT_VALUE_BYTES,
            max_message_bytes: BLE_DEFAULT_MAX_MESSAGE_BYTES,
        }
    }

    /// Validate configuration parameters and compute the conservative static frame floor.
    pub fn validate_and_static_floor(&self) -> Result<usize, BleTransportError<()>> {
        if self.min_peer_att_mtu < BLE_MIN_ATT_MTU_BYTES {
            return Err(BleTransportError::InvalidAttMtu {
                att_mtu: self.min_peer_att_mtu,
            });
        }
        if self.min_peer_att_mtu > BLE_MAX_ATT_MTU_BYTES
            || self.characteristic_max_bytes < BLE_MIN_CHARACTERISTIC_VALUE_BYTES
            || self.characteristic_max_bytes > BLE_MAX_ATT_VALUE_BYTES
            || self.max_message_bytes == 0
        {
            return Err(BleTransportError::InvalidConfig);
        }
        effective_ble_payload_limit_v1(self.min_peer_att_mtu, self.characteristic_max_bytes)
    }
}

/// Preserved incoming GATT frame staged when caller passed an undersized RX buffer (`BufferTooSmall`).
#[derive(Debug, Clone)]
struct PendingBleRxFrame {
    source: BleEndpointV1,
    len: usize,
    payload: [u8; BLE_MAX_ATT_VALUE_BYTES],
}

/// Allocation-free `no_std` BLE GATT [`PacketTransport`] enforcing Bluetooth Core Spec Vol. 3
/// Part F §3.2.9 (`512`-byte attribute ceiling), per-peer negotiated `ATT_MTU`, pre-TX credit
/// gating, and lossless `BufferTooSmall` receive staging (`ADR-0022`, `Wave 16E-2`).
#[derive(Debug)]
pub struct BlePacketTransportV1<H: BleGattHal> {
    hal: H,
    config: BleTransportConfigV1,
    static_max_frame_bytes: usize,
    pending_rx: Option<PendingBleRxFrame>,
}

impl<H: BleGattHal> BlePacketTransportV1<H> {
    /// Construct a validated [`BlePacketTransportV1`] over `hal`.
    pub fn try_new(
        hal: H,
        config: BleTransportConfigV1,
    ) -> Result<Self, BleTransportError<H::Error>> {
        let static_max_frame_bytes = config
            .validate_and_static_floor()
            .map_err(map_unit_ble_err)?;
        Ok(Self {
            hal,
            config,
            static_max_frame_bytes,
            pending_rx: None,
        })
    }

    /// Borrow the underlying [`BleGattHal`].
    pub const fn hal(&self) -> &H {
        &self.hal
    }

    /// Mutably borrow the underlying [`BleGattHal`].
    pub fn hal_mut(&mut self) -> &mut H {
        &mut self.hal
    }

    /// Active transport configuration.
    pub const fn config(&self) -> &BleTransportConfigV1 {
        &self.config
    }

    /// Compute the effective GATT payload ceiling for `peer` based on its live negotiated
    /// `ATT_MTU`, enforcing Bluetooth Core Spec Vol. 3 Part F §3.2.8–§3.2.9 (`min(mtu.min(517)-3, 512, char_max)`)
    /// and this transport's `min_peer_att_mtu` floor.
    pub fn effective_peer_payload_limit(
        &self,
        peer: BleEndpointV1,
    ) -> Result<usize, BleTransportError<H::Error>> {
        let Some(peer_att_mtu) = self.hal.peer_att_mtu(peer) else {
            return Err(BleTransportError::PeerDisconnected(peer));
        };
        if peer_att_mtu < BLE_MIN_ATT_MTU_BYTES {
            return Err(BleTransportError::InvalidAttMtu {
                att_mtu: peer_att_mtu,
            });
        }
        if peer_att_mtu < self.config.min_peer_att_mtu {
            return Err(BleTransportError::PeerMtuBelowFloor {
                peer,
                peer_att_mtu,
                min_required_att_mtu: self.config.min_peer_att_mtu,
            });
        }
        effective_ble_payload_limit_v1(peer_att_mtu, self.config.characteristic_max_bytes)
            .map_err(map_unit_ble_err)
    }

    /// Fragment and transmit a complete `message` to `destination` using [`ChunkedFrameCodec`]
    /// sized strictly to `destination`'s live effective ATT payload limit (`Wave 16E-2`).
    ///
    /// Performs full pre-TX admission checks (peer connectivity, valid `ATT_MTU >= min_peer_att_mtu`,
    /// message/chunk bounds, and `tx_credits >= total_chunks`) BEFORE sending chunk 0 so missing
    /// credits or bad bounds fail closed with zero partial transmission on the wire.
    ///
    /// **Post-admission failure semantics:** BLE notifications are unacknowledged, so this method
    /// cannot be transactional once chunk 0 is handed to the HAL. If the HAL faults on chunk `k > 0`
    /// (`WouldBlock`, `Disconnected`, `Gatt`) or the peer's MTU shrinks mid-transfer, chunks `0..k`
    /// are already on the wire and the error is returned immediately without sending chunks `k..`.
    /// The receiver's [`crate::chunked::ObservedSourceReassembler`] discards the incomplete transfer
    /// at its hard deadline; callers MUST retry with a fresh `transfer_id` (reusing the same
    /// `transfer_id` with different chunk bytes is quarantined as a conflict by the reassembler).
    ///
    /// Returns the number of chunk frames transmitted (`1..=max_chunks_per_transfer`).
    /// For messages exceeding a peer's single-burst credit pool, callers should encode individual
    /// chunks with [`ChunkedFrameCodec::encode_frame_into`] sized to
    /// [`Self::effective_peer_payload_limit`] and pace them via [`PacketTransport::try_send_frame`].
    pub fn send_chunked_message_v1(
        &mut self,
        destination: &BleEndpointV1,
        transfer_id: u32,
        max_chunks_per_transfer: u16,
        message: &[u8],
    ) -> Result<u16, PacketSendError<BleTransportError<H::Error>>> {
        let peer_limit = self
            .effective_peer_payload_limit(*destination)
            .map_err(PacketSendError::Transport)?;
        let stride = ChunkedFrameCodec::stride_for_mtu(peer_limit)
            .map_err(|e| PacketSendError::Transport(BleTransportError::Chunk(e)))?;

        if message.is_empty() || message.len() > self.config.max_message_bytes {
            return Err(PacketSendError::Transport(BleTransportError::Chunk(
                ChunkError::MessageTooLarge,
            )));
        }

        let num_chunks = message.len().div_ceil(stride);
        if num_chunks == 0
            || num_chunks > (u16::MAX as usize)
            || num_chunks > (max_chunks_per_transfer as usize)
        {
            return Err(PacketSendError::Transport(BleTransportError::Chunk(
                ChunkError::InvalidChunkBounds,
            )));
        }

        let Some(avail_credits) = self.hal.tx_credits(*destination) else {
            return Err(PacketSendError::Transport(
                BleTransportError::PeerDisconnected(*destination),
            ));
        };
        if usize::from(avail_credits) < num_chunks {
            return Err(PacketSendError::WouldBlock);
        }

        let total_chunks = num_chunks as u16;
        let mut frame_buf = [0u8; BLE_MAX_ATT_VALUE_BYTES];
        for (idx, chunk) in message.chunks(stride).enumerate() {
            let written = ChunkedFrameCodec::encode_frame_into(
                transfer_id,
                idx as u16,
                total_chunks,
                chunk,
                &mut frame_buf[..peer_limit],
            )
            .map_err(|e| PacketSendError::Transport(BleTransportError::Chunk(e)))?;
            self.try_send_frame(destination, &frame_buf[..written])?;
        }
        Ok(total_chunks)
    }
}

fn map_unit_ble_err<E>(err: BleTransportError<()>) -> BleTransportError<E> {
    match err {
        BleTransportError::InvalidConfig => BleTransportError::InvalidConfig,
        BleTransportError::InvalidAttMtu { att_mtu } => {
            BleTransportError::InvalidAttMtu { att_mtu }
        }
        BleTransportError::PeerMtuBelowFloor {
            peer,
            peer_att_mtu,
            min_required_att_mtu,
        } => BleTransportError::PeerMtuBelowFloor {
            peer,
            peer_att_mtu,
            min_required_att_mtu,
        },
        BleTransportError::PeerDisconnected(peer) => BleTransportError::PeerDisconnected(peer),
        BleTransportError::Chunk(err) => BleTransportError::Chunk(err),
        BleTransportError::Hal(()) => BleTransportError::InvalidConfig,
    }
}

impl<H: BleGattHal> PacketTransport for BlePacketTransportV1<H> {
    type Endpoint = BleEndpointV1;
    type Error = BleTransportError<H::Error>;

    fn try_send_frame(
        &mut self,
        destination: &Self::Endpoint,
        frame: &[u8],
    ) -> Result<(), PacketSendError<Self::Error>> {
        // Phase 1 (`prepare`, zero mutation): verify live peer ATT_MTU, 512B Core Spec cap,
        // characteristic cap, frame length, and available TX credits before touching HAL TX.
        let peer_limit = self
            .effective_peer_payload_limit(*destination)
            .map_err(PacketSendError::Transport)?;
        if frame.len() > peer_limit {
            return Err(PacketSendError::FrameTooLarge {
                len: frame.len(),
                max: peer_limit,
            });
        }
        let Some(credits) = self.hal.tx_credits(*destination) else {
            return Err(PacketSendError::Transport(
                BleTransportError::PeerDisconnected(*destination),
            ));
        };
        if credits == 0 {
            return Err(PacketSendError::WouldBlock);
        }

        // Phase 2 (`tx`): hand bounded frame to GATT HAL.
        match self.hal.try_notify_frame(*destination, frame) {
            Ok(()) => Ok(()),
            Err(BleHalError::WouldBlock) => Err(PacketSendError::WouldBlock),
            Err(BleHalError::Disconnected) => Err(PacketSendError::Transport(
                BleTransportError::PeerDisconnected(*destination),
            )),
            Err(BleHalError::Gatt(err)) => {
                Err(PacketSendError::Transport(BleTransportError::Hal(err)))
            }
        }
    }

    fn try_recv_frame(
        &mut self,
        buffer: &mut [u8],
    ) -> Result<Option<ReceivedFrame<Self::Endpoint>>, PacketRecvError<Self::Error>> {
        // First deliver any frame preserved from a prior `BufferTooSmall` call.
        if let Some(ref staged) = self.pending_rx {
            if buffer.len() < staged.len {
                return Err(PacketRecvError::BufferTooSmall {
                    required: staged.len,
                    available: buffer.len(),
                });
            }
            let staged = self.pending_rx.take().expect("checked Some above");
            buffer[..staged.len].copy_from_slice(&staged.payload[..staged.len]);
            return Ok(Some(ReceivedFrame {
                source: staged.source,
                len: staged.len,
            }));
        }

        let mut rx_buf = [0u8; BLE_MAX_ATT_VALUE_BYTES];
        let (source, len) = match self
            .hal
            .try_recv_frame(&mut rx_buf)
            .map_err(|e| PacketRecvError::Transport(BleTransportError::Hal(e)))?
        {
            Some(pair) => pair,
            None => return Ok(None),
        };

        if len > self.config.characteristic_max_bytes || len > BLE_MAX_ATT_VALUE_BYTES {
            return Err(PacketRecvError::TruncatedFrame);
        }

        if buffer.len() < len {
            rx_buf[len..].fill(0);
            self.pending_rx = Some(PendingBleRxFrame {
                source,
                len,
                payload: rx_buf,
            });
            return Err(PacketRecvError::BufferTooSmall {
                required: len,
                available: buffer.len(),
            });
        }

        buffer[..len].copy_from_slice(&rx_buf[..len]);
        Ok(Some(ReceivedFrame { source, len }))
    }

    fn capabilities(&self) -> PacketCapabilities {
        PacketCapabilities {
            reliability: ReliabilityClass::BestEffort,
            bandwidth_class: BandwidthClass::Low,
            max_frame_bytes: self.static_max_frame_bytes,
            max_message_bytes: self.config.max_message_bytes,
            supports_multicast: false,
            has_verified_link_principal: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    use alloc::collections::{BTreeMap, VecDeque};
    #[cfg(feature = "alloc")]
    use alloc::vec::Vec;

    #[cfg(feature = "alloc")]
    #[derive(Debug, Clone)]
    struct MockPeerState {
        att_mtu: u16,
        credits: u16,
    }

    /// Deterministic in-memory BLE GATT HAL supporting per-peer ATT_MTU, TX credits, and fault injection.
    #[cfg(feature = "alloc")]
    #[derive(Debug, Default)]
    struct MockBleGattHal {
        peers: BTreeMap<BleEndpointV1, MockPeerState>,
        tx_attempts: usize,
        tx_queue: VecDeque<(BleEndpointV1, Vec<u8>)>,
        rx_queue: VecDeque<(BleEndpointV1, Vec<u8>)>,
        fail_next_would_block: bool,
        fail_next_disconnect: bool,
        fail_next_gatt_err: Option<u8>,
        /// Inject a GATT error exactly when `tx_attempts` reaches the given (1-based) attempt number.
        fail_gatt_on_attempt: Option<(usize, u8)>,
        fail_next_rx_err: Option<u8>,
    }

    #[cfg(feature = "alloc")]
    impl MockBleGattHal {
        fn connect(&mut self, peer: BleEndpointV1, att_mtu: u16, credits: u16) {
            self.peers.insert(peer, MockPeerState { att_mtu, credits });
        }
    }

    #[cfg(feature = "alloc")]
    impl BleGattHal for MockBleGattHal {
        type Error = u8;

        fn peer_att_mtu(&self, peer: BleEndpointV1) -> Option<u16> {
            self.peers.get(&peer).map(|p| p.att_mtu)
        }

        fn tx_credits(&self, peer: BleEndpointV1) -> Option<u16> {
            self.peers.get(&peer).map(|p| p.credits)
        }

        fn try_notify_frame(
            &mut self,
            peer: BleEndpointV1,
            frame: &[u8],
        ) -> Result<(), BleHalError<Self::Error>> {
            self.tx_attempts += 1;
            if self.fail_next_would_block {
                self.fail_next_would_block = false;
                return Err(BleHalError::WouldBlock);
            }
            if self.fail_next_disconnect {
                self.fail_next_disconnect = false;
                return Err(BleHalError::Disconnected);
            }
            if let Some(code) = self.fail_next_gatt_err.take() {
                return Err(BleHalError::Gatt(code));
            }
            if let Some((attempt, code)) = self.fail_gatt_on_attempt {
                if self.tx_attempts == attempt {
                    self.fail_gatt_on_attempt = None;
                    return Err(BleHalError::Gatt(code));
                }
            }
            let Some(state) = self.peers.get_mut(&peer) else {
                return Err(BleHalError::Disconnected);
            };
            if state.credits == 0 {
                return Err(BleHalError::WouldBlock);
            }
            state.credits -= 1;
            self.tx_queue.push_back((peer, frame.to_vec()));
            Ok(())
        }

        fn try_recv_frame(
            &mut self,
            buf: &mut [u8],
        ) -> Result<Option<(BleEndpointV1, usize)>, Self::Error> {
            if let Some(code) = self.fail_next_rx_err.take() {
                return Err(code);
            }
            let Some((src, pkt)) = self.rx_queue.pop_front() else {
                return Ok(None);
            };
            let n = pkt.len().min(buf.len());
            buf[..n].copy_from_slice(&pkt[..n]);
            Ok(Some((src, pkt.len())))
        }
    }

    #[test]
    fn wave16e2_effective_ble_payload_limit_enforces_core_spec_512_byte_att_value_ceiling() {
        // Spec floor: ATT_MTU = 23 -> 20 bytes
        assert_eq!(effective_ble_payload_limit_v1(23, 512).unwrap(), 20);
        // Canonical DLE: ATT_MTU = 247 -> 244 bytes -> 228B chunk stride
        assert_eq!(
            effective_ble_payload_limit_v1(BLE_CANONICAL_ATT_MTU_BYTES, 512).unwrap(),
            244
        );
        assert_eq!(
            ChunkedFrameCodec::stride_for_mtu(244).unwrap(),
            BLE_CANONICAL_CHUNK_STRIDE_BYTES
        );
        assert_eq!(BLE_CANONICAL_CHUNK_STRIDE_BYTES, 228);

        // Bluetooth Core Spec Vol. 3 Part F §3.2.9 hard 512B attribute cap:
        // Even at ATT_MTU = 515, 516, 517, or u16::MAX (and even if char_max > 512), ceiling is strictly 512!
        assert_eq!(effective_ble_payload_limit_v1(515, 512).unwrap(), 512);
        assert_eq!(effective_ble_payload_limit_v1(516, 512).unwrap(), 512);
        assert_eq!(effective_ble_payload_limit_v1(517, 512).unwrap(), 512);
        assert_eq!(effective_ble_payload_limit_v1(517, 517).unwrap(), 512);
        assert_eq!(
            effective_ble_payload_limit_v1(u16::MAX, usize::MAX).unwrap(),
            512
        );

        // Characteristic smaller than ATT_MTU - 3 caps effective payload.
        assert_eq!(effective_ble_payload_limit_v1(247, 100).unwrap(), 100);
        assert_eq!(effective_ble_payload_limit_v1(517, 64).unwrap(), 64);

        // Spec-violating ATT_MTU < 23 or characteristic_max_bytes == 0 fail closed.
        assert_eq!(
            effective_ble_payload_limit_v1(22, 512).unwrap_err(),
            BleTransportError::InvalidAttMtu { att_mtu: 22 }
        );
        assert_eq!(
            effective_ble_payload_limit_v1(0, 512).unwrap_err(),
            BleTransportError::InvalidAttMtu { att_mtu: 0 }
        );
        assert_eq!(
            effective_ble_payload_limit_v1(247, 0).unwrap_err(),
            BleTransportError::InvalidConfig
        );

        // Config validation rejects out-of-range ATT_MTU, char_max (< 17 or > 512), or zero max_message_bytes.
        assert_eq!(
            BleTransportConfigV1 {
                min_peer_att_mtu: 22,
                ..BleTransportConfigV1::default()
            }
            .validate_and_static_floor()
            .unwrap_err(),
            BleTransportError::InvalidAttMtu { att_mtu: 22 }
        );
        assert_eq!(
            BleTransportConfigV1 {
                min_peer_att_mtu: 518,
                ..BleTransportConfigV1::default()
            }
            .validate_and_static_floor()
            .unwrap_err(),
            BleTransportError::InvalidConfig
        );
        assert_eq!(
            BleTransportConfigV1 {
                characteristic_max_bytes: 16,
                ..BleTransportConfigV1::default()
            }
            .validate_and_static_floor()
            .unwrap_err(),
            BleTransportError::InvalidConfig
        );
        assert_eq!(
            BleTransportConfigV1 {
                characteristic_max_bytes: 513,
                ..BleTransportConfigV1::default()
            }
            .validate_and_static_floor()
            .unwrap_err(),
            BleTransportError::InvalidConfig
        );
        assert_eq!(
            BleTransportConfigV1 {
                max_message_bytes: 0,
                ..BleTransportConfigV1::default()
            }
            .validate_and_static_floor()
            .unwrap_err(),
            BleTransportError::InvalidConfig
        );

        // HCI handle range helper check.
        assert!(BleEndpointV1(0x0001).is_standard_hci_handle());
        assert!(BleEndpointV1(0x0EFF).is_standard_hci_handle());
        assert!(!BleEndpointV1(0x0F00).is_standard_hci_handle());
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn wave16e2_att_mtu_517_accepts_512_rejects_513_and_514_pre_tx_and_enforces_credits_and_floor()
    {
        let mut hal = MockBleGattHal::default();
        let peer_517 = BleEndpointV1(0x0040);
        let peer_23 = BleEndpointV1(0x0041);
        let peer_bad_mtu = BleEndpointV1(0x0042);
        let peer_no_credits = BleEndpointV1(0x0043);
        let peer_missing = BleEndpointV1(0x0099);

        hal.connect(peer_517, 517, 2);
        hal.connect(peer_23, 23, 2);
        hal.connect(peer_bad_mtu, 22, 2);
        hal.connect(peer_no_credits, 247, 0);

        let mut tx = BlePacketTransportV1::try_new(hal, BleTransportConfigV1::default()).unwrap();

        // Static capabilities report conservative floor (20B for min_peer_att_mtu = 23).
        let caps = tx.capabilities();
        assert_eq!(caps.max_frame_bytes, 20);
        assert_eq!(caps.bandwidth_class, BandwidthClass::Low);
        assert_eq!(caps.reliability, ReliabilityClass::BestEffort);
        assert!(!caps.supports_multicast);
        assert!(!caps.has_verified_link_principal);

        // 1. At ATT_MTU = 517, char_max = 512: 513B and 514B MUST be rejected pre-TX!
        let attempts_before = tx.hal().tx_attempts;
        assert_eq!(
            tx.try_send_frame(&peer_517, &[0xAB; 513]).unwrap_err(),
            PacketSendError::FrameTooLarge { len: 513, max: 512 }
        );
        assert_eq!(
            tx.try_send_frame(&peer_517, &[0xAB; 514]).unwrap_err(),
            PacketSendError::FrameTooLarge { len: 514, max: 512 }
        );
        assert_eq!(
            tx.hal().tx_attempts,
            attempts_before,
            "513B/514B must fail before touching HAL TX"
        );
        assert_eq!(tx.hal().tx_credits(peer_517), Some(2));

        // 2. Exactly 512B succeeds and consumes 1 credit.
        tx.try_send_frame(&peer_517, &[0xCD; 512]).unwrap();
        assert_eq!(tx.hal().tx_credits(peer_517), Some(1));
        assert_eq!(tx.hal().tx_queue.len(), 1);
        assert_eq!(tx.hal().tx_queue[0].1.len(), 512);

        // 3. Peer with ATT_MTU = 23 accepts 20B and rejects 21B pre-TX.
        tx.try_send_frame(&peer_23, &[0x11; 20]).unwrap();
        let attempts_before_21 = tx.hal().tx_attempts;
        assert_eq!(
            tx.try_send_frame(&peer_23, &[0x11; 21]).unwrap_err(),
            PacketSendError::FrameTooLarge { len: 21, max: 20 }
        );
        assert_eq!(tx.hal().tx_attempts, attempts_before_21);

        // 4. Peer with invalid ATT_MTU = 22 fails closed pre-TX.
        assert_eq!(
            tx.try_send_frame(&peer_bad_mtu, &[0x22; 10]).unwrap_err(),
            PacketSendError::Transport(BleTransportError::InvalidAttMtu { att_mtu: 22 })
        );

        // 5. Missing TX credit returns WouldBlock BEFORE touching HAL!
        let attempts_before_wb = tx.hal().tx_attempts;
        assert_eq!(
            tx.try_send_frame(&peer_no_credits, &[0x33; 10])
                .unwrap_err(),
            PacketSendError::WouldBlock
        );
        assert_eq!(tx.hal().tx_attempts, attempts_before_wb);

        // 6. Disconnected peer returns PeerDisconnected BEFORE touching HAL!
        assert_eq!(
            tx.try_send_frame(&peer_missing, &[0x44; 10]).unwrap_err(),
            PacketSendError::Transport(BleTransportError::PeerDisconnected(peer_missing))
        );
        assert_eq!(tx.hal().tx_attempts, attempts_before_wb);

        // 7. HAL-level WouldBlock, Disconnected, and Gatt errors propagate cleanly without burning credits.
        tx.hal_mut().fail_next_would_block = true;
        assert_eq!(
            tx.try_send_frame(&peer_517, b"retry").unwrap_err(),
            PacketSendError::WouldBlock
        );
        assert_eq!(tx.hal().tx_credits(peer_517), Some(1));

        tx.hal_mut().fail_next_disconnect = true;
        assert_eq!(
            tx.try_send_frame(&peer_517, b"drop").unwrap_err(),
            PacketSendError::Transport(BleTransportError::PeerDisconnected(peer_517))
        );
        assert_eq!(tx.hal().tx_credits(peer_517), Some(1));

        tx.hal_mut().fail_next_gatt_err = Some(0x7F);
        assert_eq!(
            tx.try_send_frame(&peer_517, b"err").unwrap_err(),
            PacketSendError::Transport(BleTransportError::Hal(0x7F))
        );
        assert_eq!(tx.hal().tx_credits(peer_517), Some(1));

        // 8. Canonical DLE transport (min_peer_att_mtu = 247) reports 244B floor and rejects 23B peer.
        let mut dle_hal = MockBleGattHal::default();
        dle_hal.connect(peer_23, 23, 4);
        dle_hal.connect(peer_517, 247, 4);
        let mut dle_tx =
            BlePacketTransportV1::try_new(dle_hal, BleTransportConfigV1::canonical_dle_247())
                .unwrap();
        assert_eq!(dle_tx.capabilities().max_frame_bytes, 244);
        assert_eq!(
            dle_tx.try_send_frame(&peer_23, &[0x55; 10]).unwrap_err(),
            PacketSendError::Transport(BleTransportError::PeerMtuBelowFloor {
                peer: peer_23,
                peer_att_mtu: 23,
                min_required_att_mtu: 247,
            })
        );
        assert_eq!(dle_tx.hal().tx_attempts, 0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn wave16e2_ble_chunked_247_mtu_reassembly_atomic_credit_precheck_and_buffer_too_small() {
        use crate::chunked::{ObservedSourceReassembler, ReassemblerLimits};

        let peer = BleEndpointV1(0x0010);
        let mut tx_hal = MockBleGattHal::default();
        // Start with only 2 credits — a 600B message at 247 MTU (244B notify -> 228B stride) needs 3 chunks!
        tx_hal.connect(peer, BLE_CANONICAL_ATT_MTU_BYTES, 2);

        let mut tx =
            BlePacketTransportV1::try_new(tx_hal, BleTransportConfigV1::canonical_dle_247())
                .unwrap();
        let mut rx = BlePacketTransportV1::try_new(
            MockBleGattHal::default(),
            BleTransportConfigV1::default(),
        )
        .unwrap();

        let message: Vec<u8> = (0..600usize)
            .map(|i| (i.wrapping_mul(17).wrapping_add(5) & 0xFF) as u8)
            .collect();

        // 1. Atomic pre-TX credit check: 2 credits < 3 chunks -> WouldBlock with ZERO packets sent!
        assert_eq!(
            tx.send_chunked_message_v1(&peer, 42, 16, &message)
                .unwrap_err(),
            PacketSendError::WouldBlock
        );
        assert_eq!(tx.hal().tx_attempts, 0);
        assert_eq!(tx.hal().tx_credits(peer), Some(2));

        // 2. Empty message and chunk-cap overflow fail closed pre-TX.
        assert_eq!(
            tx.send_chunked_message_v1(&peer, 42, 16, &[]).unwrap_err(),
            PacketSendError::Transport(BleTransportError::Chunk(ChunkError::MessageTooLarge))
        );
        assert_eq!(
            tx.send_chunked_message_v1(&peer, 42, 2, &message)
                .unwrap_err(),
            PacketSendError::Transport(BleTransportError::Chunk(ChunkError::InvalidChunkBounds))
        );
        assert_eq!(tx.hal().tx_attempts, 0);

        // 3. Replenish credits to 3 -> all 3 chunks (244, 244, 160 bytes) transmit cleanly!
        tx.hal_mut().peers.get_mut(&peer).unwrap().credits = 3;
        let sent_chunks = tx.send_chunked_message_v1(&peer, 42, 16, &message).unwrap();
        assert_eq!(sent_chunks, 3);
        assert_eq!(tx.hal().tx_credits(peer), Some(0));

        // Move wire frames into RX queue.
        while let Some((src, frame)) = tx.hal_mut().tx_queue.pop_front() {
            rx.hal_mut().rx_queue.push_back((src, frame));
        }

        // 4. Repeated BufferTooSmall (32 -> 128) preserves staged frame without dropping it!
        let mut tiny = [0u8; 32];
        assert_eq!(
            rx.try_recv_frame(&mut tiny).unwrap_err(),
            PacketRecvError::BufferTooSmall {
                required: 244,
                available: 32,
            }
        );
        let mut medium = [0u8; 128];
        assert_eq!(
            rx.try_recv_frame(&mut medium).unwrap_err(),
            PacketRecvError::BufferTooSmall {
                required: 244,
                available: 128,
            }
        );

        // 5. Reassemble full 600-byte payload via ObservedSourceReassembler (MTU = 244, stride = 228).
        let mut reassembler = ObservedSourceReassembler::new(ReassemblerLimits {
            mtu: BLE_CANONICAL_NOTIFY_PAYLOAD_BYTES,
            max_message_bytes: BLE_DEFAULT_MAX_MESSAGE_BYTES,
            ..ReassemblerLimits::default()
        })
        .unwrap();

        let mut rx_buf = [0u8; BLE_MAX_ATT_VALUE_BYTES];
        let mut frame_lens = Vec::new();
        let mut assembled = None;
        while let Some(rec) = rx.try_recv_frame(&mut rx_buf).unwrap() {
            assert_eq!(rec.source, peer);
            frame_lens.push(rec.len);
            if let Some(done) = reassembler
                .ingest_frame(1, rec.source, &rx_buf[..rec.len], 2_000)
                .unwrap()
            {
                assembled = Some(done);
            }
        }
        assert_eq!(frame_lens, alloc::vec![244, 244, 160]);
        assert_eq!(assembled.unwrap().payload, message);

        // 6. Oversized RX payload (> 512 bytes) fails closed with TruncatedFrame; HAL RX error propagates.
        rx.hal_mut()
            .rx_queue
            .push_back((peer, alloc::vec![0xFF; BLE_MAX_ATT_VALUE_BYTES + 1]));
        assert_eq!(
            rx.try_recv_frame(&mut rx_buf).unwrap_err(),
            PacketRecvError::TruncatedFrame
        );
        rx.hal_mut().fail_next_rx_err = Some(0x91);
        assert_eq!(
            rx.try_recv_frame(&mut rx_buf).unwrap_err(),
            PacketRecvError::Transport(BleTransportError::Hal(0x91))
        );

        // 7. Zero-byte RX notification succeeds even into an empty caller buffer `&mut []`.
        rx.hal_mut().rx_queue.push_back((peer, Vec::new()));
        let rec_zero = rx.try_recv_frame(&mut []).unwrap().unwrap();
        assert_eq!(rec_zero.source, peer);
        assert_eq!(rec_zero.len, 0);

        // 8. Smaller characteristic_max_bytes (100 < 247 - 3) caps static floor, TX (even at MTU=517), and RX.
        let mut small_char_hal = MockBleGattHal::default();
        let peer_over_517 = BleEndpointV1(0x0050);
        small_char_hal.connect(peer_over_517, u16::MAX, 4);
        let mut small_tx = BlePacketTransportV1::try_new(
            small_char_hal,
            BleTransportConfigV1 {
                min_peer_att_mtu: BLE_CANONICAL_ATT_MTU_BYTES,
                characteristic_max_bytes: 100,
                max_message_bytes: 256,
            },
        )
        .unwrap();
        assert_eq!(small_tx.capabilities().max_frame_bytes, 100);
        assert_eq!(
            small_tx
                .effective_peer_payload_limit(peer_over_517)
                .unwrap(),
            100
        );
        assert_eq!(
            small_tx
                .try_send_frame(&peer_over_517, &[0xAA; 101])
                .unwrap_err(),
            PacketSendError::FrameTooLarge { len: 101, max: 100 }
        );
        small_tx
            .hal_mut()
            .rx_queue
            .push_back((peer_over_517, alloc::vec![0xBB; 101]));
        assert_eq!(
            small_tx.try_recv_frame(&mut rx_buf).unwrap_err(),
            PacketRecvError::TruncatedFrame
        );

        // 9. send_chunked_message_v1 bounds: max_message_bytes, max_chunks == 0, MTU >= 517 (496B stride -> 512B frames),
        //    and mid-loop HAL failure short-circuit.
        assert_eq!(
            small_tx
                .send_chunked_message_v1(&peer_over_517, 1, 16, &[0x11; 257])
                .unwrap_err(),
            PacketSendError::Transport(BleTransportError::Chunk(ChunkError::MessageTooLarge))
        );
        assert_eq!(
            small_tx
                .send_chunked_message_v1(&peer_over_517, 1, 0, &[0x11; 10])
                .unwrap_err(),
            PacketSendError::Transport(BleTransportError::Chunk(ChunkError::InvalidChunkBounds))
        );

        let mut big_hal = MockBleGattHal::default();
        let peer_wide = BleEndpointV1(0x0060);
        big_hal.connect(peer_wide, 600, 10); // > 517 clamped to 512B -> stride 496B
        let mut big_tx =
            BlePacketTransportV1::try_new(big_hal, BleTransportConfigV1::default()).unwrap();
        assert_eq!(big_tx.effective_peer_payload_limit(peer_wide).unwrap(), 512);
        let wide_msg = alloc::vec![0x5Au8; 1_000]; // ceil(1000 / 496) = 3 chunks: 512, 512, 24 bytes
        assert_eq!(
            big_tx
                .send_chunked_message_v1(&peer_wide, 99, 16, &wide_msg)
                .unwrap(),
            3
        );
        let wide_lens: Vec<usize> = big_tx.hal().tx_queue.iter().map(|(_, f)| f.len()).collect();
        assert_eq!(wide_lens, alloc::vec![512, 512, 24]);

        // Pre-admission HAL failure on chunk 0: error returned, zero frames on the wire.
        big_tx.hal_mut().tx_queue.clear();
        big_tx.hal_mut().fail_next_gatt_err = Some(0xEE);
        assert_eq!(
            big_tx
                .send_chunked_message_v1(&peer_wide, 100, 16, &wide_msg)
                .unwrap_err(),
            PacketSendError::Transport(BleTransportError::Hal(0xEE))
        );
        assert!(big_tx.hal().tx_queue.is_empty());

        // True mid-loop HAL failure on chunk 1 of 3: chunk 0 is already on the wire (documented,
        // non-transactional), chunk 2 is never attempted, and the error surfaces immediately.
        big_tx.hal_mut().peers.get_mut(&peer_wide).unwrap().credits = 10;
        let attempts_before = big_tx.hal().tx_attempts;
        big_tx.hal_mut().fail_gatt_on_attempt = Some((attempts_before + 2, 0xD1));
        assert_eq!(
            big_tx
                .send_chunked_message_v1(&peer_wide, 101, 16, &wide_msg)
                .unwrap_err(),
            PacketSendError::Transport(BleTransportError::Hal(0xD1))
        );
        assert_eq!(big_tx.hal().tx_attempts, attempts_before + 2);
        assert_eq!(big_tx.hal().tx_queue.len(), 1);
        assert_eq!(big_tx.hal().tx_queue[0].1.len(), 512);
        assert_eq!(big_tx.hal().tx_credits(peer_wide), Some(9));
    }
}
