//! `no_std` LoRa sub-GHz physical packet transport (`ADR-0022`, `Wave 16E-1`).
//!
//! **Maturity Tier:** `Simulation verified` (hardware-agnostic HAL trait + in-memory simulated
//! radio link tests; physical RF hardware drivers such as `SX126x`/`SX127x` SPI transceivers
//! implement [`LoraRadioHal`] out of tree).
//!
//! Provides:
//! - [`LORA_MAX_PHY_MTU_BYTES`] (`250` bytes codec-visible payload after the 5-byte link header,
//!   fitting inside the 255-byte SX126x/SX127x FIFO limit) and [`LORA_CANONICAL_CHUNK_STRIDE_BYTES`]
//!   (`234` bytes for `ChunkedFrameCodec`).
//! - Exact integer-microsecond Semtech AN1200.13 Time-on-Air calculation
//!   ([`LoraModulationConfigV1::compute_airtime_micros`]) with zero floating-point arithmetic.
//! - Fixed-memory `no_std` sliding-window duty-cycle and FCC dwell-time enforcement
//!   ([`LoraDutyCycleLimiterV1`]) using a `prepare -> tx -> commit` [`LoraAirtimeReservationV1`]
//!   token so HAL `WouldBlock` or radio errors never consume airtime budget.
//! - Allocation-free [`LoraPacketTransportV1`] implementing [`PacketTransport`] over any
//!   [`LoraRadioHal`], preserving incoming frames losslessly on [`PacketRecvError::BufferTooSmall`].

use crate::chunked::CHUNK_HEADER_LEN;
use crate::traits::{
    BandwidthClass, PacketCapabilities, PacketRecvError, PacketSendError, PacketTransport,
    ReceivedFrame, ReliabilityClass,
};

/// SX126x / SX127x hardware physical FIFO limit in bytes (`255` bytes).
pub const LORA_SX12XX_MAX_FIFO_BYTES: usize = 255;

/// SwarmTorch LoRa link framing header length in bytes (`version[1] || src_u16_le[2] || dst_u16_le[2]`).
pub const LORA_LINK_HEADER_BYTES: usize = 5;

/// Link header version byte for SwarmTorch LoRa v1 frames.
pub const LORA_LINK_VERSION_V1: u8 = 0x01;

/// Maximum codec-visible LoRa frame payload in bytes (`255 - 5 = 250` bytes).
pub const LORA_MAX_PHY_MTU_BYTES: usize = LORA_SX12XX_MAX_FIFO_BYTES - LORA_LINK_HEADER_BYTES;

/// Canonical [`crate::chunked::ChunkedFrameCodec`] payload stride $P = M - 16 = 234$ bytes for LoRa.
pub const LORA_CANONICAL_CHUNK_STRIDE_BYTES: usize = LORA_MAX_PHY_MTU_BYTES - CHUNK_HEADER_LEN;

/// Default maximum reassembled message size advertised by [`LoraPacketTransportV1`] (`64 KiB`).
pub const LORA_DEFAULT_MAX_MESSAGE_BYTES: usize = 65_536;

/// Default ETSI EN 300 220 duty-cycle window (`1 hour = 3,600,000,000` microseconds).
pub const LORA_DEFAULT_DUTY_WINDOW_MICROS: u64 = 3_600_000_000;

/// Default EU868 1% duty cycle expressed in parts-per-million (`10,000` ppm = 1.0%).
pub const LORA_EU868_1_PERCENT_PPM: u32 = 10_000;

/// Default US915 FCC Part 15.247 maximum per-packet dwell time (`400 ms = 400,000` microseconds).
pub const LORA_US915_MAX_DWELL_MICROS: u64 = 400_000;

/// Number of sub-windows covering one full regulatory duty-cycle window (`60`).
const DUTY_WINDOW_SUBDIVISIONS: usize = 60;

/// Circular ring slots (`61 = 60 + 1`) so no bucket expires until every timestamp inside it is
/// strictly older than `window_micros` (conservative upper bound, zero early expiration).
const DUTY_RING_SLOTS: usize = DUTY_WINDOW_SUBDIVISIONS + 1;

/// Maximum number of foreign-destination unicast frames discarded in a single non-blocking
/// [`PacketTransport::try_recv_frame`] poll to bound worst-case IRQ/loop latency.
const MAX_FOREIGN_DISCARD_PER_POLL: usize = 8;

/// Unauthenticated 16-bit observed LoRa link endpoint address (`ADR-0022` §1.2).
///
/// Never conflated with a cryptographically verified `PeerId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LoraEndpointV1(pub u16);

impl LoraEndpointV1 {
    /// Broadcast link address (`0xFFFF`).
    pub const BROADCAST: Self = Self(0xFFFF);

    /// Return `true` if this endpoint is [`Self::BROADCAST`].
    #[inline]
    pub const fn is_broadcast(self) -> bool {
        self.0 == Self::BROADCAST.0
    }
}

/// LoRa Chirp Spread Spectrum (CSS) spreading factor (`SF7..=SF12`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum LoraSpreadingFactorV1 {
    /// Spreading Factor 7 (`2^7 = 128` chips/symbol).
    Sf7 = 7,
    /// Spreading Factor 8 (`2^8 = 256` chips/symbol).
    Sf8 = 8,
    /// Spreading Factor 9 (`2^9 = 512` chips/symbol).
    Sf9 = 9,
    /// Spreading Factor 10 (`2^10 = 1024` chips/symbol).
    Sf10 = 10,
    /// Spreading Factor 11 (`2^11 = 2048` chips/symbol).
    Sf11 = 11,
    /// Spreading Factor 12 (`2^12 = 4096` chips/symbol).
    Sf12 = 12,
}

impl LoraSpreadingFactorV1 {
    /// Numeric spreading factor value (`7..=12`).
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Standard LoRa channel bandwidths (`125 kHz`, `250 kHz`, `500 kHz`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LoraBandwidthV1 {
    /// 125 kHz channel bandwidth.
    Bw125kHz,
    /// 250 kHz channel bandwidth.
    Bw250kHz,
    /// 500 kHz channel bandwidth.
    Bw500kHz,
}

impl LoraBandwidthV1 {
    /// Bandwidth in Hertz.
    #[inline]
    pub const fn hz(self) -> u64 {
        match self {
            Self::Bw125kHz => 125_000,
            Self::Bw250kHz => 250_000,
            Self::Bw500kHz => 500_000,
        }
    }
}

/// LoRa forward error correction coding rate (`4/5`, `4/6`, `4/7`, `4/8`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum LoraCodingRateV1 {
    /// Coding rate 4/5 (`CR = 1`).
    Cr4_5 = 1,
    /// Coding rate 4/6 (`CR = 2`).
    Cr4_6 = 2,
    /// Coding rate 4/7 (`CR = 3`).
    Cr4_7 = 3,
    /// Coding rate 4/8 (`CR = 4`).
    Cr4_8 = 4,
}

impl LoraCodingRateV1 {
    /// Semtech AN1200.13 `CR` integer (`1..=4`).
    #[inline]
    pub const fn cr_param(self) -> u64 {
        self as u8 as u64
    }
}

/// Errors returned by LoRa configuration, duty-cycle accounting, and HAL integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraTransportError<E> {
    /// Invalid modulation, preamble, MTU, duty-cycle, or address configuration parameter.
    InvalidConfig,
    /// Physical payload length exceeded [`LORA_SX12XX_MAX_FIFO_BYTES`] (`255` bytes).
    PhyPayloadTooLarge {
        /// Attempted physical payload size in bytes.
        len: usize,
        /// Hardware FIFO maximum (`255` bytes).
        max: usize,
    },
    /// Monotonic microsecond clock moved backward (`now_micros < last_time_micros`).
    ClockRegression {
        /// Last observed monotonic timestamp.
        last_micros: u64,
        /// Regressed timestamp supplied by caller/HAL.
        now_micros: u64,
    },
    /// Packet Time-on-Air exceeded configured FCC/regional maximum dwell time.
    DwellTimeExceeded {
        /// Computed packet Time-on-Air in microseconds.
        airtime_micros: u64,
        /// Configured maximum dwell time in microseconds.
        max_dwell_micros: u64,
    },
    /// Packet Time-on-Air would exceed sliding-window regulatory duty-cycle budget.
    DutyCycleExceeded {
        /// Requested packet Time-on-Air in microseconds.
        requested_micros: u64,
        /// Remaining available airtime in the active regulatory window in microseconds.
        remaining_micros: u64,
    },
    /// Airtime reservation token was already superseded by another commit or bound to another limiter state.
    StaleReservation,
    /// Underlying radio hardware/driver error.
    Hal(E),
}

impl<E: core::fmt::Debug> core::fmt::Display for LoraTransportError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid LoRa modulation or duty-cycle configuration"),
            Self::PhyPayloadTooLarge { len, max } => {
                write!(f, "LoRa PHY payload {len} bytes exceeds {max}-byte FIFO cap")
            }
            Self::ClockRegression {
                last_micros,
                now_micros,
            } => write!(
                f,
                "LoRa monotonic clock regressed: now {now_micros} us < last {last_micros} us"
            ),
            Self::DwellTimeExceeded {
                airtime_micros,
                max_dwell_micros,
            } => write!(
                f,
                "LoRa packet airtime {airtime_micros} us exceeds max dwell {max_dwell_micros} us"
            ),
            Self::DutyCycleExceeded {
                requested_micros,
                remaining_micros,
            } => write!(
                f,
                "LoRa duty-cycle budget exceeded: requested {requested_micros} us, remaining {remaining_micros} us"
            ),
            Self::StaleReservation => {
                write!(f, "stale or superseded LoRa airtime reservation token")
            }
            Self::Hal(err) => write!(f, "LoRa HAL error: {err:?}"),
        }
    }
}

#[cfg(feature = "std")]
impl<E: core::fmt::Debug> std::error::Error for LoraTransportError<E> {}

/// Validated LoRa physical modulation parameters for exact Semtech AN1200.13 Time-on-Air computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraModulationConfigV1 {
    /// Spreading factor (`SF7..=SF12`).
    pub spreading_factor: LoraSpreadingFactorV1,
    /// Channel bandwidth (`125 / 250 / 500 kHz`).
    pub bandwidth: LoraBandwidthV1,
    /// Coding rate (`4/5 .. 4/8`).
    pub coding_rate: LoraCodingRateV1,
    /// Preamble symbol count (must be `>= 6` per SX126x/SX127x spec; canonical `8`).
    pub preamble_symbols: u16,
    /// Whether explicit PHY header mode is active (`true` -> $H = 0$, `false` -> $H = 1$).
    pub explicit_header: bool,
    /// Whether hardware PHY CRC is enabled (`true` -> $\text{CRC} = 1$).
    pub crc_enabled: bool,
    /// Explicit Low Data Rate Optimization flag (also automatically enforced when $T_{\text{sym}} \ge 16\text{ ms}$).
    pub low_data_rate_optimize: bool,
}

impl Default for LoraModulationConfigV1 {
    fn default() -> Self {
        Self::eu868_sf7_bw125()
    }
}

impl LoraModulationConfigV1 {
    /// Canonical EU868 DR5 configuration (`SF7`, `125 kHz`, `CR 4/5`, 8 preamble symbols, explicit header, CRC on).
    pub const fn eu868_sf7_bw125() -> Self {
        Self {
            spreading_factor: LoraSpreadingFactorV1::Sf7,
            bandwidth: LoraBandwidthV1::Bw125kHz,
            coding_rate: LoraCodingRateV1::Cr4_5,
            preamble_symbols: 8,
            explicit_header: true,
            crc_enabled: true,
            low_data_rate_optimize: false,
        }
    }

    /// Canonical long-range EU868 DR0 configuration (`SF12`, `125 kHz`, `CR 4/5`, LDRO enabled).
    pub const fn eu868_sf12_bw125() -> Self {
        Self {
            spreading_factor: LoraSpreadingFactorV1::Sf12,
            bandwidth: LoraBandwidthV1::Bw125kHz,
            coding_rate: LoraCodingRateV1::Cr4_5,
            preamble_symbols: 8,
            explicit_header: true,
            crc_enabled: true,
            low_data_rate_optimize: true,
        }
    }

    /// Validate modulation parameters.
    pub fn validate(&self) -> Result<(), LoraTransportError<()>> {
        if self.preamble_symbols < 6 {
            return Err(LoraTransportError::InvalidConfig);
        }
        Ok(())
    }

    /// Effective Low Data Rate Optimization bit (`DE = 1` when explicitly requested or whenever
    /// symbol duration $T_{\text{sym}} = 2^{\text{SF}} / \text{BW} \ge 16\text{ ms}$).
    #[inline]
    pub const fn effective_low_data_rate_optimize(&self) -> bool {
        let sf = self.spreading_factor.as_u8() as u32;
        let sym_chips = 1u64 << sf;
        let bw_hz = self.bandwidth.hz();
        // T_sym >= 16 ms  <==>  sym_chips * 1000 >= 16 * bw_hz
        self.low_data_rate_optimize || (sym_chips * 1_000 >= 16 * bw_hz)
    }

    /// Compute exact integer-microsecond LoRa Time-on-Air (ceiling-rounded to microseconds) per
    /// **Semtech AN1200.13 §4** without floating-point operations.
    ///
    /// `phy_payload_bytes` is the full over-the-air PHY payload length (including the 5-byte
    /// SwarmTorch LoRa link header), capped at [`LORA_SX12XX_MAX_FIFO_BYTES`] (`255` bytes).
    pub fn compute_airtime_micros(
        &self,
        phy_payload_bytes: usize,
    ) -> Result<u64, LoraTransportError<()>> {
        self.validate()?;
        if phy_payload_bytes > LORA_SX12XX_MAX_FIFO_BYTES {
            return Err(LoraTransportError::PhyPayloadTooLarge {
                len: phy_payload_bytes,
                max: LORA_SX12XX_MAX_FIFO_BYTES,
            });
        }

        let sf = i64::from(self.spreading_factor.as_u8());
        let pl = phy_payload_bytes as i64;
        let crc: i64 = if self.crc_enabled { 1 } else { 0 };
        let h: i64 = if self.explicit_header { 0 } else { 1 };
        let de: i64 = if self.effective_low_data_rate_optimize() {
            1
        } else {
            0
        };

        // Semtech AN1200.13 §4:
        // n_payload = 8 + max(ceil((8*PL - 4*SF + 28 + 16*CRC - 20*H) / (4*(SF - 2*DE))) * (CR + 4), 0)
        let num = 8 * pl - 4 * sf + 28 + 16 * crc - 20 * h;
        let den = 4 * (sf - 2 * de);
        let blocks: u64 = if num <= 0 {
            0
        } else {
            let n = num as u64;
            let d = den as u64;
            n.div_ceil(d)
        };
        let payload_symbols: u64 = 8 + blocks * (self.coding_rate.cr_param() + 4);

        // Total symbols in quarter-symbols: 4*(n_preamble + 4.25 + n_payload) = 4*n_preamble + 17 + 4*n_payload
        let total_quarter_symbols =
            u128::from(self.preamble_symbols) * 4 + 17 + u128::from(payload_symbols) * 4;
        let sym_chips = 1u128 << (self.spreading_factor.as_u8() as u32);
        let numerator = total_quarter_symbols * sym_chips * 1_000_000u128;
        let denominator = 4u128 * u128::from(self.bandwidth.hz());
        let toa_micros = numerator.div_ceil(denominator);

        u64::try_from(toa_micros).map_err(|_| LoraTransportError::InvalidConfig)
    }
}

/// Sealed reservation token produced by [`LoraDutyCycleLimiterV1::prepare_tx`] and consumed by
/// [`LoraDutyCycleLimiterV1::commit_tx`] only after the HAL confirms packet transmission (`Wave 16E-1`).
///
/// Dropping this token without calling `commit_tx` (e.g., on HAL `WouldBlock` or radio error)
/// leaves the regulatory duty-cycle budget untouched.
///
/// Binds the limiter's policy and monotonic state snapshot
/// `(window_micros, duty_cycle_ppm, max_dwell_micros, initialized, last_time_micros, commit_epoch)`
/// in pure `no_std` without atomics or heap allocation (note: two value-identical freshly cloned
/// limiters with identical state snapshots cannot be distinguished without an external instance tag).
#[derive(Debug, PartialEq, Eq)]
#[must_use = "commit_tx must be called only after HAL TX succeeds; drop on WouldBlock/error"]
pub struct LoraAirtimeReservationV1 {
    now_micros: u64,
    airtime_micros: u64,
    window_micros: u64,
    duty_cycle_ppm: u32,
    max_dwell_micros: Option<u64>,
    initialized: bool,
    last_time_micros: u64,
    commit_epoch: u64,
}

impl LoraAirtimeReservationV1 {
    /// Timestamp bound to this airtime reservation.
    #[inline]
    pub const fn now_micros(&self) -> u64 {
        self.now_micros
    }

    /// Reserved packet Time-on-Air in microseconds.
    #[inline]
    pub const fn airtime_micros(&self) -> u64 {
        self.airtime_micros
    }
}

/// Fixed-memory `no_std` sliding-window duty-cycle and dwell-time enforcer (`ADR-0022`, `Wave 16E-1`).
///
/// Uses a 61-slot (`60 + 1`) circular sub-window histogram so every transmitted packet remains
/// accounted for at least `window_micros`, preventing both 2x leaky-bucket burst bugs and
/// early boundary expiration.
#[derive(Debug, Clone)]
pub struct LoraDutyCycleLimiterV1 {
    window_micros: u64,
    slot_micros: u64,
    duty_cycle_ppm: u32,
    max_airtime_micros: u64,
    max_dwell_micros: Option<u64>,
    initialized: bool,
    last_time_micros: u64,
    last_slot: u64,
    commit_epoch: u64,
    slots: [u64; DUTY_RING_SLOTS],
}

impl LoraDutyCycleLimiterV1 {
    /// Construct a validated sliding-window duty-cycle limiter.
    ///
    /// - `window_micros` must be `>= 1`.
    /// - `duty_cycle_ppm` must be in `1..=1_000_000` (e.g. `10_000` = 1% EU868).
    /// - Resulting `max_airtime_micros = floor(window_micros * duty_cycle_ppm / 1_000_000)` must be `>= 1`.
    /// - `max_dwell_micros`, when `Some(d)`, must be `>= 1`.
    pub fn try_new(
        window_micros: u64,
        duty_cycle_ppm: u32,
        max_dwell_micros: Option<u64>,
    ) -> Result<Self, LoraTransportError<()>> {
        if window_micros == 0 || duty_cycle_ppm == 0 || duty_cycle_ppm > 1_000_000 {
            return Err(LoraTransportError::InvalidConfig);
        }
        if let Some(dwell) = max_dwell_micros {
            if dwell == 0 {
                return Err(LoraTransportError::InvalidConfig);
            }
        }
        // Floor division ensures regulatory duty cycle is never exceeded.
        let max_airtime_u128 =
            (u128::from(window_micros) * u128::from(duty_cycle_ppm)) / 1_000_000u128;
        let max_airtime_micros =
            u64::try_from(max_airtime_u128).map_err(|_| LoraTransportError::InvalidConfig)?;
        if max_airtime_micros == 0 {
            return Err(LoraTransportError::InvalidConfig);
        }
        let slot_micros = window_micros
            .div_ceil(DUTY_WINDOW_SUBDIVISIONS as u64)
            .max(1);
        Ok(Self {
            window_micros,
            slot_micros,
            duty_cycle_ppm,
            max_airtime_micros,
            max_dwell_micros,
            initialized: false,
            last_time_micros: 0,
            last_slot: 0,
            commit_epoch: 0,
            slots: [0u64; DUTY_RING_SLOTS],
        })
    }

    /// Canonical ETSI EN 300 220 EU868 1% duty-cycle limiter (`36s / hour`, no FCC dwell cap).
    pub fn eu868_1_percent() -> Self {
        Self::try_new(
            LORA_DEFAULT_DUTY_WINDOW_MICROS,
            LORA_EU868_1_PERCENT_PPM,
            None,
        )
        .expect("EU868 1% defaults are valid")
    }

    /// Canonical US915 FCC Part 15.247 dwell-time limiter (`400 ms` max per-packet dwell, 100% duty).
    pub fn us915_dwell_400ms() -> Self {
        Self::try_new(
            LORA_DEFAULT_DUTY_WINDOW_MICROS,
            1_000_000,
            Some(LORA_US915_MAX_DWELL_MICROS),
        )
        .expect("US915 400ms dwell defaults are valid")
    }

    /// Configured sliding window length in microseconds.
    #[inline]
    pub const fn window_micros(&self) -> u64 {
        self.window_micros
    }

    /// Configured duty cycle in parts-per-million (`1..=1_000_000`).
    #[inline]
    pub const fn duty_cycle_ppm(&self) -> u32 {
        self.duty_cycle_ppm
    }

    /// Maximum total airtime permitted in one regulatory window (`microseconds`).
    #[inline]
    pub const fn max_airtime_micros(&self) -> u64 {
        self.max_airtime_micros
    }

    /// Optional per-packet dwell-time ceiling in microseconds.
    #[inline]
    pub const fn max_dwell_micros(&self) -> Option<u64> {
        self.max_dwell_micros
    }

    /// Compute active airtime already consumed at `now_micros` without mutating internal state.
    pub fn active_airtime_at(&self, now_micros: u64) -> Result<u64, LoraTransportError<()>> {
        if !self.initialized {
            return Ok(0);
        }
        if now_micros < self.last_time_micros {
            return Err(LoraTransportError::ClockRegression {
                last_micros: self.last_time_micros,
                now_micros,
            });
        }
        let now_slot = now_micros / self.slot_micros;
        let delta_slots = now_slot - self.last_slot;
        if delta_slots >= DUTY_RING_SLOTS as u64 {
            return Ok(0);
        }
        let mut total = 0u64;
        // Retain only slots from `(now_slot - (DUTY_RING_SLOTS - 1))..=last_slot`.
        // Expired slots are `(last_slot + 1)..=now_slot` (wrapped mod DUTY_RING_SLOTS).
        for (idx, &slot_val) in self.slots.iter().enumerate() {
            if !slot_expires_on_advance(self.last_slot, delta_slots, idx) {
                total = total.saturating_add(slot_val);
            }
        }
        Ok(total)
    }

    /// **Phase 1 (`prepare`, read-only `&self`):** verify non-zero airtime, dwell time, clock
    /// monotonicity, and remaining sliding-window airtime without deducting any budget.
    pub fn prepare_tx(
        &self,
        now_micros: u64,
        airtime_micros: u64,
    ) -> Result<LoraAirtimeReservationV1, LoraTransportError<()>> {
        if airtime_micros == 0 {
            return Err(LoraTransportError::InvalidConfig);
        }
        let _ = self
            .commit_epoch
            .checked_add(1)
            .ok_or(LoraTransportError::InvalidConfig)?;
        if let Some(max_dwell) = self.max_dwell_micros {
            if airtime_micros > max_dwell {
                return Err(LoraTransportError::DwellTimeExceeded {
                    airtime_micros,
                    max_dwell_micros: max_dwell,
                });
            }
        }
        let used = self.active_airtime_at(now_micros)?;
        let remaining = self.max_airtime_micros.saturating_sub(used);
        if airtime_micros > remaining {
            return Err(LoraTransportError::DutyCycleExceeded {
                requested_micros: airtime_micros,
                remaining_micros: remaining,
            });
        }
        Ok(LoraAirtimeReservationV1 {
            now_micros,
            airtime_micros,
            window_micros: self.window_micros,
            duty_cycle_ppm: self.duty_cycle_ppm,
            max_dwell_micros: self.max_dwell_micros,
            initialized: self.initialized,
            last_time_micros: self.last_time_micros,
            commit_epoch: self.commit_epoch,
        })
    }

    /// **Phase 3 (`commit`):** record a previously prepared airtime reservation ONLY after the
    /// HAL confirms the packet was accepted for transmission (`Wave 16E-1`).
    ///
    /// Rejects stale/superseded reservations or reservations prepared against a different limiter
    /// policy/state snapshot with [`LoraTransportError::StaleReservation`], and re-verifies dwell,
    /// clock monotonicity, and remaining budget before mutating ring state.
    pub fn commit_tx(
        &mut self,
        reservation: LoraAirtimeReservationV1,
    ) -> Result<(), LoraTransportError<()>> {
        if reservation.commit_epoch != self.commit_epoch
            || reservation.window_micros != self.window_micros
            || reservation.duty_cycle_ppm != self.duty_cycle_ppm
            || reservation.max_dwell_micros != self.max_dwell_micros
            || reservation.initialized != self.initialized
            || reservation.last_time_micros != self.last_time_micros
        {
            return Err(LoraTransportError::StaleReservation);
        }
        let next_epoch = self
            .commit_epoch
            .checked_add(1)
            .ok_or(LoraTransportError::InvalidConfig)?;
        // Re-validate invariants before mutating ring state.
        let _ = self.prepare_tx(reservation.now_micros, reservation.airtime_micros)?;

        let now_slot = reservation.now_micros / self.slot_micros;
        if !self.initialized {
            self.initialized = true;
            self.last_time_micros = reservation.now_micros;
            self.last_slot = now_slot;
            self.slots = [0u64; DUTY_RING_SLOTS];
        } else {
            let delta_slots = now_slot - self.last_slot;
            if delta_slots >= DUTY_RING_SLOTS as u64 {
                self.slots = [0u64; DUTY_RING_SLOTS];
            } else {
                for step in 1..=delta_slots {
                    let clear_idx = ((self.last_slot + step) % (DUTY_RING_SLOTS as u64)) as usize;
                    self.slots[clear_idx] = 0;
                }
            }
            self.last_time_micros = reservation.now_micros;
            self.last_slot = now_slot;
        }
        let cur_idx = (now_slot % (DUTY_RING_SLOTS as u64)) as usize;
        self.slots[cur_idx] = self.slots[cur_idx].saturating_add(reservation.airtime_micros);
        self.commit_epoch = next_epoch;
        Ok(())
    }
}

#[inline]
fn slot_expires_on_advance(last_slot: u64, delta_slots: u64, ring_idx: usize) -> bool {
    if delta_slots == 0 {
        return false;
    }
    let ring_len = DUTY_RING_SLOTS as u64;
    let last_idx = last_slot % ring_len;
    let idx_u64 = ring_idx as u64;
    // Distance forward from `last_idx` in `(0..ring_len]`.
    let forward_dist = if idx_u64 > last_idx {
        idx_u64 - last_idx
    } else {
        ring_len - last_idx + idx_u64
    };
    forward_dist <= delta_slots
}

/// Non-blocking error returned by a low-level [`LoraRadioHal`] driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraHalError<E> {
    /// Radio TX FIFO is busy, CAD detected channel activity, or radio is currently transmitting.
    WouldBlock,
    /// Hardware/SPI/RF transceiver error.
    Radio(E),
}

/// Hardware-agnostic, non-blocking `no_std` LoRa transceiver HAL trait (`ADR-0022`, `Wave 16E-1`).
///
/// Downstream firmware crates implement this trait over `embedded-hal` SPI drivers (`SX126x`,
/// `SX127x`, `LR11xx`) using monotonic hardware timers.
pub trait LoraRadioHal {
    /// Driver-specific hardware error type.
    type Error: core::fmt::Debug;

    /// Read current monotonic microsecond timestamp (must never regress).
    fn now_micros(&self) -> u64;

    /// Attempt to transmit a raw physical packet (`phy_frame.len() <= 255`) without blocking.
    ///
    /// Returning `Err(LoraHalError::WouldBlock)` or `Err(LoraHalError::Radio(_))` signals that
    /// no RF transmission occurred; [`LoraPacketTransportV1`] does NOT deduct duty-cycle budget.
    fn try_send_phy(&mut self, phy_frame: &[u8]) -> Result<(), LoraHalError<Self::Error>>;

    /// Attempt to receive one raw physical packet into `phy_buf` (`phy_buf.len() >= 255`) without
    /// blocking. Returns `Ok(None)` when no packet is queued, or `Ok(Some(len))` when `len` bytes
    /// were written into `phy_buf[..len]`.
    fn try_recv_phy(&mut self, phy_buf: &mut [u8]) -> Result<Option<usize>, Self::Error>;
}

/// Configuration for [`LoraPacketTransportV1`].
#[derive(Debug, Clone)]
pub struct LoraTransportConfigV1 {
    /// Local 16-bit link endpoint address (must not be [`LoraEndpointV1::BROADCAST`]).
    pub local_addr: LoraEndpointV1,
    /// LoRa physical modulation parameters used for Time-on-Air computation.
    pub modulation: LoraModulationConfigV1,
    /// Sliding-window duty-cycle and dwell-time limiter.
    pub duty_limiter: LoraDutyCycleLimiterV1,
    /// Maximum reassembled message length advertised via [`PacketCapabilities::max_message_bytes`].
    pub max_message_bytes: usize,
}

impl LoraTransportConfigV1 {
    /// Construct a validated EU868 1% SF7/125kHz LoRa transport configuration.
    pub fn eu868_default(local_addr: LoraEndpointV1) -> Result<Self, LoraTransportError<()>> {
        let cfg = Self {
            local_addr,
            modulation: LoraModulationConfigV1::eu868_sf7_bw125(),
            duty_limiter: LoraDutyCycleLimiterV1::eu868_1_percent(),
            max_message_bytes: LORA_DEFAULT_MAX_MESSAGE_BYTES,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate transport configuration parameters.
    pub fn validate(&self) -> Result<(), LoraTransportError<()>> {
        if self.local_addr.is_broadcast() || self.max_message_bytes == 0 {
            return Err(LoraTransportError::InvalidConfig);
        }
        self.modulation.validate()?;
        Ok(())
    }
}

/// Preserved incoming frame staged when caller passed an undersized RX buffer (`BufferTooSmall`).
#[derive(Debug, Clone)]
struct PendingLoraRxFrame {
    source: LoraEndpointV1,
    len: usize,
    payload: [u8; LORA_MAX_PHY_MTU_BYTES],
}

/// Allocation-free `no_std` LoRa [`PacketTransport`] implementing 5-byte link framing, exact
/// Semtech AN1200.13 Time-on-Air calculation, `prepare -> tx -> commit` duty-cycle enforcement,
/// and lossless `BufferTooSmall` receive staging (`ADR-0022`, `Wave 16E-1`).
#[derive(Debug)]
pub struct LoraPacketTransportV1<R: LoraRadioHal> {
    radio: R,
    local_addr: LoraEndpointV1,
    modulation: LoraModulationConfigV1,
    duty_limiter: LoraDutyCycleLimiterV1,
    max_message_bytes: usize,
    pending_rx: Option<PendingLoraRxFrame>,
}

impl<R: LoraRadioHal> LoraPacketTransportV1<R> {
    /// Construct a validated [`LoraPacketTransportV1`] over `radio`.
    pub fn try_new(
        radio: R,
        config: LoraTransportConfigV1,
    ) -> Result<Self, LoraTransportError<R::Error>> {
        config.validate().map_err(map_unit_lora_err)?;
        Ok(Self {
            radio,
            local_addr: config.local_addr,
            modulation: config.modulation,
            duty_limiter: config.duty_limiter,
            max_message_bytes: config.max_message_bytes,
            pending_rx: None,
        })
    }

    /// Borrow the underlying [`LoraRadioHal`].
    pub const fn radio(&self) -> &R {
        &self.radio
    }

    /// Mutably borrow the underlying [`LoraRadioHal`].
    pub fn radio_mut(&mut self) -> &mut R {
        &mut self.radio
    }

    /// Borrow the active [`LoraDutyCycleLimiterV1`].
    pub const fn duty_limiter(&self) -> &LoraDutyCycleLimiterV1 {
        &self.duty_limiter
    }

    /// Borrow the active [`LoraModulationConfigV1`].
    pub const fn modulation(&self) -> &LoraModulationConfigV1 {
        &self.modulation
    }

    /// Local 16-bit observed link address.
    pub const fn local_addr(&self) -> LoraEndpointV1 {
        self.local_addr
    }
}

fn map_unit_lora_err<E>(err: LoraTransportError<()>) -> LoraTransportError<E> {
    match err {
        LoraTransportError::InvalidConfig => LoraTransportError::InvalidConfig,
        LoraTransportError::PhyPayloadTooLarge { len, max } => {
            LoraTransportError::PhyPayloadTooLarge { len, max }
        }
        LoraTransportError::ClockRegression {
            last_micros,
            now_micros,
        } => LoraTransportError::ClockRegression {
            last_micros,
            now_micros,
        },
        LoraTransportError::DwellTimeExceeded {
            airtime_micros,
            max_dwell_micros,
        } => LoraTransportError::DwellTimeExceeded {
            airtime_micros,
            max_dwell_micros,
        },
        LoraTransportError::DutyCycleExceeded {
            requested_micros,
            remaining_micros,
        } => LoraTransportError::DutyCycleExceeded {
            requested_micros,
            remaining_micros,
        },
        LoraTransportError::StaleReservation => LoraTransportError::StaleReservation,
        LoraTransportError::Hal(()) => LoraTransportError::InvalidConfig,
    }
}

impl<R: LoraRadioHal> PacketTransport for LoraPacketTransportV1<R> {
    type Endpoint = LoraEndpointV1;
    type Error = LoraTransportError<R::Error>;

    fn try_send_frame(
        &mut self,
        destination: &Self::Endpoint,
        frame: &[u8],
    ) -> Result<(), PacketSendError<Self::Error>> {
        if frame.len() > LORA_MAX_PHY_MTU_BYTES {
            return Err(PacketSendError::FrameTooLarge {
                len: frame.len(),
                max: LORA_MAX_PHY_MTU_BYTES,
            });
        }
        let phy_len = LORA_LINK_HEADER_BYTES + frame.len();
        let airtime_micros = self
            .modulation
            .compute_airtime_micros(phy_len)
            .map_err(|e| PacketSendError::Transport(map_unit_lora_err(e)))?;

        // Phase 1 (`prepare`, zero mutation): check monotonic clock, dwell, and duty-cycle budget.
        let now_micros = self.radio.now_micros();
        let reservation = self
            .duty_limiter
            .prepare_tx(now_micros, airtime_micros)
            .map_err(|e| PacketSendError::Transport(map_unit_lora_err(e)))?;

        // Build 5-byte link header + frame payload in a tiny fixed 255-byte stack buffer.
        let mut phy_buf = [0u8; LORA_SX12XX_MAX_FIFO_BYTES];
        phy_buf[0] = LORA_LINK_VERSION_V1;
        phy_buf[1..3].copy_from_slice(&self.local_addr.0.to_le_bytes());
        phy_buf[3..5].copy_from_slice(&destination.0.to_le_bytes());
        phy_buf[LORA_LINK_HEADER_BYTES..phy_len].copy_from_slice(frame);

        // Phase 2 (`tx`): attempt HAL transmission. On WouldBlock or Radio error, `reservation`
        // is dropped uncommitted so zero airtime budget is burned!
        match self.radio.try_send_phy(&phy_buf[..phy_len]) {
            Ok(()) => {
                // Phase 3 (`commit`): deduct airtime ONLY after HAL confirms transmission.
                self.duty_limiter
                    .commit_tx(reservation)
                    .map_err(|e| PacketSendError::Transport(map_unit_lora_err(e)))?;
                Ok(())
            }
            Err(LoraHalError::WouldBlock) => Err(PacketSendError::WouldBlock),
            Err(LoraHalError::Radio(err)) => {
                Err(PacketSendError::Transport(LoraTransportError::Hal(err)))
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

        let mut phy_buf = [0u8; LORA_SX12XX_MAX_FIFO_BYTES];
        for _ in 0..MAX_FOREIGN_DISCARD_PER_POLL {
            let phy_len = match self
                .radio
                .try_recv_phy(&mut phy_buf)
                .map_err(|e| PacketRecvError::Transport(LoraTransportError::Hal(e)))?
            {
                Some(len) => len,
                None => return Ok(None),
            };

            if !(LORA_LINK_HEADER_BYTES..=LORA_SX12XX_MAX_FIFO_BYTES).contains(&phy_len) {
                return Err(PacketRecvError::TruncatedFrame);
            }
            if phy_buf[0] != LORA_LINK_VERSION_V1 {
                return Err(PacketRecvError::TruncatedFrame);
            }

            let src_addr = LoraEndpointV1(u16::from_le_bytes([phy_buf[1], phy_buf[2]]));
            let dst_addr = LoraEndpointV1(u16::from_le_bytes([phy_buf[3], phy_buf[4]]));
            if src_addr.is_broadcast() {
                return Err(PacketRecvError::TruncatedFrame);
            }
            if dst_addr != self.local_addr && !dst_addr.is_broadcast() {
                // Drop foreign unicast traffic cleanly without surfacing to caller.
                continue;
            }

            let payload_len = phy_len - LORA_LINK_HEADER_BYTES;
            if buffer.len() < payload_len {
                let mut payload = [0u8; LORA_MAX_PHY_MTU_BYTES];
                payload[..payload_len].copy_from_slice(&phy_buf[LORA_LINK_HEADER_BYTES..phy_len]);
                self.pending_rx = Some(PendingLoraRxFrame {
                    source: src_addr,
                    len: payload_len,
                    payload,
                });
                return Err(PacketRecvError::BufferTooSmall {
                    required: payload_len,
                    available: buffer.len(),
                });
            }

            buffer[..payload_len].copy_from_slice(&phy_buf[LORA_LINK_HEADER_BYTES..phy_len]);
            return Ok(Some(ReceivedFrame {
                source: src_addr,
                len: payload_len,
            }));
        }

        Ok(None)
    }

    fn capabilities(&self) -> PacketCapabilities {
        PacketCapabilities {
            reliability: ReliabilityClass::BestEffort,
            bandwidth_class: BandwidthClass::UltraLow,
            max_frame_bytes: LORA_MAX_PHY_MTU_BYTES,
            max_message_bytes: self.max_message_bytes,
            supports_multicast: true,
            has_verified_link_principal: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "alloc")]
    use alloc::collections::VecDeque;
    #[cfg(feature = "alloc")]
    use alloc::vec::Vec;

    /// Deterministic in-memory LoRa HAL supporting clock control, `WouldBlock`, and fault injection.
    #[cfg(feature = "alloc")]
    #[derive(Debug, Default)]
    struct MockLoraHal {
        now_us: u64,
        tx_attempts: usize,
        tx_queue: VecDeque<Vec<u8>>,
        rx_queue: VecDeque<Vec<u8>>,
        fail_next_would_block: bool,
        fail_next_radio_err: Option<u8>,
        fail_next_rx_err: Option<u8>,
    }

    #[cfg(feature = "alloc")]
    impl LoraRadioHal for MockLoraHal {
        type Error = u8;

        fn now_micros(&self) -> u64 {
            self.now_us
        }

        fn try_send_phy(&mut self, phy_frame: &[u8]) -> Result<(), LoraHalError<Self::Error>> {
            self.tx_attempts += 1;
            if self.fail_next_would_block {
                self.fail_next_would_block = false;
                return Err(LoraHalError::WouldBlock);
            }
            if let Some(code) = self.fail_next_radio_err.take() {
                return Err(LoraHalError::Radio(code));
            }
            self.tx_queue.push_back(phy_frame.to_vec());
            Ok(())
        }

        fn try_recv_phy(&mut self, phy_buf: &mut [u8]) -> Result<Option<usize>, Self::Error> {
            if let Some(code) = self.fail_next_rx_err.take() {
                return Err(code);
            }
            let Some(pkt) = self.rx_queue.pop_front() else {
                return Ok(None);
            };
            let n = pkt.len().min(phy_buf.len());
            phy_buf[..n].copy_from_slice(&pkt[..n]);
            Ok(Some(pkt.len()))
        }
    }

    #[test]
    fn wave16e1_semtech_an1200_13_time_on_air_golden_vectors() {
        // Vector 1: SF7, 125kHz, CR4/5, preamble=8, explicit header, CRC=on, PL=20 -> 56,576 us
        let sf7_125 = LoraModulationConfigV1::eu868_sf7_bw125();
        assert!(!sf7_125.effective_low_data_rate_optimize());
        assert_eq!(sf7_125.compute_airtime_micros(20).unwrap(), 56_576);

        // Vector 2: SF12, 125kHz, CR4/5, preamble=8, explicit header, CRC=on, LDRO=on, PL=51 -> 2,465,792 us
        let sf12_125 = LoraModulationConfigV1::eu868_sf12_bw125();
        assert!(sf12_125.effective_low_data_rate_optimize());
        assert_eq!(sf12_125.compute_airtime_micros(51).unwrap(), 2_465_792);

        // Auto-LDRO kicks in whenever T_sym >= 16ms:
        // SF11/125kHz (16.384ms -> DE=1), SF12/125kHz (32.768ms -> DE=1), SF12/250kHz (16.384ms -> DE=1),
        // while SF10/125kHz (8.192ms -> DE=0) and SF12/500kHz (8.192ms -> DE=0).
        let mut sf12_auto = sf12_125;
        sf12_auto.low_data_rate_optimize = false;
        assert!(sf12_auto.effective_low_data_rate_optimize());
        assert_eq!(sf12_auto.compute_airtime_micros(51).unwrap(), 2_465_792);

        let sf11_125 = LoraModulationConfigV1 {
            spreading_factor: LoraSpreadingFactorV1::Sf11,
            bandwidth: LoraBandwidthV1::Bw125kHz,
            low_data_rate_optimize: false,
            ..sf7_125
        };
        assert!(sf11_125.effective_low_data_rate_optimize());
        assert_eq!(sf11_125.compute_airtime_micros(20).unwrap(), 741_376);

        let sf12_250 = LoraModulationConfigV1 {
            spreading_factor: LoraSpreadingFactorV1::Sf12,
            bandwidth: LoraBandwidthV1::Bw250kHz,
            low_data_rate_optimize: false,
            ..sf7_125
        };
        assert!(sf12_250.effective_low_data_rate_optimize());
        assert_eq!(sf12_250.compute_airtime_micros(20).unwrap(), 659_456);

        let sf10_125 = LoraModulationConfigV1 {
            spreading_factor: LoraSpreadingFactorV1::Sf10,
            bandwidth: LoraBandwidthV1::Bw125kHz,
            low_data_rate_optimize: false,
            ..sf7_125
        };
        assert!(!sf10_125.effective_low_data_rate_optimize());

        let sf12_500 = LoraModulationConfigV1 {
            spreading_factor: LoraSpreadingFactorV1::Sf12,
            bandwidth: LoraBandwidthV1::Bw500kHz,
            low_data_rate_optimize: false,
            ..sf7_125
        };
        assert!(!sf12_500.effective_low_data_rate_optimize());

        // Vector 3: SF7, 500kHz, CR4/5, preamble=8, explicit header, CRC=on, PL=255 -> 99,904 us
        let sf7_500 = LoraModulationConfigV1 {
            spreading_factor: LoraSpreadingFactorV1::Sf7,
            bandwidth: LoraBandwidthV1::Bw500kHz,
            coding_rate: LoraCodingRateV1::Cr4_5,
            preamble_symbols: 8,
            explicit_header: true,
            crc_enabled: true,
            low_data_rate_optimize: false,
        };
        assert_eq!(sf7_500.compute_airtime_micros(255).unwrap(), 99_904);

        // Implicit header (H=1), CRC off (CRC=0), PL=0 (negative numerator clamp -> blocks=0), preamble=6
        let implicit_zero = LoraModulationConfigV1 {
            preamble_symbols: 6,
            explicit_header: false,
            crc_enabled: false,
            coding_rate: LoraCodingRateV1::Cr4_8,
            ..sf7_125
        };
        assert_eq!(implicit_zero.compute_airtime_micros(0).unwrap(), 18_688);

        // Coding rate progression CR4/5 < CR4/6 < CR4/7 < CR4/8 and exact CR4/8 check (78,080 us)
        let cr4_6 = LoraModulationConfigV1 {
            coding_rate: LoraCodingRateV1::Cr4_6,
            ..sf7_125
        }
        .compute_airtime_micros(20)
        .unwrap();
        let cr4_7 = LoraModulationConfigV1 {
            coding_rate: LoraCodingRateV1::Cr4_7,
            ..sf7_125
        }
        .compute_airtime_micros(20)
        .unwrap();
        let cr4_8 = LoraModulationConfigV1 {
            coding_rate: LoraCodingRateV1::Cr4_8,
            ..sf7_125
        }
        .compute_airtime_micros(20)
        .unwrap();
        assert!(56_576 < cr4_6 && cr4_6 < cr4_7 && cr4_7 < cr4_8);
        assert_eq!(cr4_8, 78_080);

        // Extreme preamble (u16::MAX) at SF12/125kHz/CR4_8/PL=255 does not overflow u128/u64.
        let extreme = LoraModulationConfigV1 {
            preamble_symbols: u16::MAX,
            coding_rate: LoraCodingRateV1::Cr4_8,
            ..sf12_125
        };
        assert_eq!(extreme.compute_airtime_micros(255).unwrap(), 2_161_221_632);

        // Payload > 255 bytes rejected; preamble < 6 rejected
        assert_eq!(
            sf7_125.compute_airtime_micros(256).unwrap_err(),
            LoraTransportError::PhyPayloadTooLarge { len: 256, max: 255 }
        );
        let mut bad_pre = sf7_125;
        bad_pre.preamble_symbols = 5;
        assert_eq!(
            bad_pre.compute_airtime_micros(20).unwrap_err(),
            LoraTransportError::InvalidConfig
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn wave16e1_prepare_tx_commit_preserves_budget_on_would_block_and_enforces_sliding_window() {
        let modulation = LoraModulationConfigV1::eu868_sf7_bw125();
        // 15-byte frame -> 20-byte PHY packet -> exact 56,576 us ToA.
        let expected_toa = modulation.compute_airtime_micros(20).unwrap();
        assert_eq!(expected_toa, 56_576);

        // 1% duty over a 6-second window -> max_airtime_micros = 60,000 us (fits exactly 1 packet, not 2!).
        let limiter = LoraDutyCycleLimiterV1::try_new(6_000_000, 10_000, Some(400_000)).unwrap();
        assert_eq!(limiter.max_airtime_micros(), 60_000);

        let mut hal = MockLoraHal {
            now_us: 1_000_000,
            ..Default::default()
        };
        hal.fail_next_would_block = true;
        let cfg = LoraTransportConfigV1 {
            local_addr: LoraEndpointV1(0x0001),
            modulation,
            duty_limiter: limiter,
            max_message_bytes: LORA_DEFAULT_MAX_MESSAGE_BYTES,
        };
        let mut tx = LoraPacketTransportV1::try_new(hal, cfg).unwrap();

        let payload = [0x42u8; 15];
        let dst = LoraEndpointV1(0x0002);

        // 1. HAL WouldBlock MUST NOT burn any duty-cycle airtime!
        assert_eq!(
            tx.try_send_frame(&dst, &payload).unwrap_err(),
            PacketSendError::WouldBlock
        );
        assert_eq!(
            tx.duty_limiter().active_airtime_at(1_000_000).unwrap(),
            0,
            "WouldBlock must not consume duty-cycle budget"
        );

        // 2. HAL radio error MUST NOT burn any duty-cycle airtime!
        tx.radio_mut().fail_next_radio_err = Some(0xEE);
        assert_eq!(
            tx.try_send_frame(&dst, &payload).unwrap_err(),
            PacketSendError::Transport(LoraTransportError::Hal(0xEE))
        );
        assert_eq!(tx.duty_limiter().active_airtime_at(1_000_000).unwrap(), 0);

        // 3. First real TX succeeds and commits exactly 56,576 us!
        tx.try_send_frame(&dst, &payload).unwrap();
        assert_eq!(
            tx.duty_limiter().active_airtime_at(1_000_000).unwrap(),
            56_576
        );

        // 4. Second TX within the 6-second window fails closed with DutyCycleExceeded!
        tx.radio_mut().now_us = 6_999_999; // 5.999999s later (< 6.0s window!)
        assert_eq!(
            tx.try_send_frame(&dst, &payload).unwrap_err(),
            PacketSendError::Transport(LoraTransportError::DutyCycleExceeded {
                requested_micros: 56_576,
                remaining_micros: 60_000 - 56_576,
            })
        );

        // 5. Clock regression fails closed with ClockRegression!
        tx.radio_mut().now_us = 999_999;
        assert_eq!(
            tx.try_send_frame(&dst, &payload).unwrap_err(),
            PacketSendError::Transport(LoraTransportError::ClockRegression {
                last_micros: 1_000_000,
                now_micros: 999_999,
            })
        );

        // 6. After full conservative 61-slot window elapses (>= 6.1s), budget replenishes cleanly!
        tx.radio_mut().now_us = 7_100_000;
        assert_eq!(tx.duty_limiter().active_airtime_at(7_100_000).unwrap(), 0);
        tx.try_send_frame(&dst, &payload).unwrap();
        assert_eq!(tx.radio().tx_queue.len(), 2);

        // 7. Oversized frame (> 250 bytes) rejected with FrameTooLarge BEFORE calling HAL!
        let attempts_before = tx.radio().tx_attempts;
        let too_big = [0xAAu8; LORA_MAX_PHY_MTU_BYTES + 1];
        assert_eq!(
            tx.try_send_frame(&dst, &too_big).unwrap_err(),
            PacketSendError::FrameTooLarge {
                len: 251,
                max: LORA_MAX_PHY_MTU_BYTES,
            }
        );
        assert_eq!(
            tx.radio().tx_attempts,
            attempts_before,
            "FrameTooLarge must reject before touching HAL"
        );

        // 8. Dwell-time cap enforced (SF12/125kHz 51-byte packet = 2.46s > 400ms US915 dwell).
        let us915_cfg = LoraTransportConfigV1 {
            local_addr: LoraEndpointV1(0x0001),
            modulation: LoraModulationConfigV1::eu868_sf12_bw125(),
            duty_limiter: LoraDutyCycleLimiterV1::us915_dwell_400ms(),
            max_message_bytes: LORA_DEFAULT_MAX_MESSAGE_BYTES,
        };
        let mut us915_tx =
            LoraPacketTransportV1::try_new(MockLoraHal::default(), us915_cfg).unwrap();
        assert!(matches!(
            us915_tx.try_send_frame(&dst, &[0x11u8; 46]),
            Err(PacketSendError::Transport(
                LoraTransportError::DwellTimeExceeded {
                    airtime_micros: 2_465_792,
                    max_dwell_micros: 400_000,
                }
            ))
        ));
        assert_eq!(us915_tx.radio().tx_attempts, 0);
    }

    #[test]
    fn wave16e1_duty_limiter_rejects_invalid_configs_stale_reservations_and_enforces_exact_61_slot_window(
    ) {
        // Invalid duty-cycle configs fail closed.
        assert_eq!(
            LoraDutyCycleLimiterV1::try_new(0, 10_000, None).unwrap_err(),
            LoraTransportError::InvalidConfig
        );
        assert_eq!(
            LoraDutyCycleLimiterV1::try_new(6_000_000, 0, None).unwrap_err(),
            LoraTransportError::InvalidConfig
        );
        assert_eq!(
            LoraDutyCycleLimiterV1::try_new(6_000_000, 1_000_001, None).unwrap_err(),
            LoraTransportError::InvalidConfig
        );
        assert_eq!(
            LoraDutyCycleLimiterV1::try_new(6_000_000, 10_000, Some(0)).unwrap_err(),
            LoraTransportError::InvalidConfig
        );
        // Sub-microsecond floor budget fails closed.
        assert_eq!(
            LoraDutyCycleLimiterV1::try_new(100, 1, None).unwrap_err(),
            LoraTransportError::InvalidConfig
        );

        // Zero-microsecond reservation fails closed without touching state.
        let mut limiter = LoraDutyCycleLimiterV1::try_new(6_000_000, 10_000, None).unwrap();
        assert_eq!(
            limiter.prepare_tx(1_000, 0).unwrap_err(),
            LoraTransportError::InvalidConfig
        );

        // Cross-limiter reservation (EU868 -> US915) fails closed with StaleReservation!
        let eu = LoraDutyCycleLimiterV1::eu868_1_percent();
        let mut us = LoraDutyCycleLimiterV1::us915_dwell_400ms();
        let eu_res_100ms = eu.prepare_tx(1_000, 100_000).unwrap();
        assert_eq!(
            us.commit_tx(eu_res_100ms).unwrap_err(),
            LoraTransportError::StaleReservation
        );
        let eu_res_500ms = eu.prepare_tx(1_000, 500_000).unwrap();
        assert_eq!(
            us.commit_tx(eu_res_500ms).unwrap_err(),
            LoraTransportError::StaleReservation
        );
        assert_eq!(us.active_airtime_at(1_000).unwrap(), 0);

        // Cross-limiter with matching epoch==1 but different last_time_micros fails closed!
        let mut lim_a = LoraDutyCycleLimiterV1::try_new(6_000_000, 10_000, None).unwrap();
        let mut lim_b = LoraDutyCycleLimiterV1::try_new(6_000_000, 10_000, None).unwrap();
        let ra0 = lim_a.prepare_tx(100, 1_000).unwrap();
        lim_a.commit_tx(ra0).unwrap();
        let rb0 = lim_b.prepare_tx(500, 1_000).unwrap();
        lim_b.commit_tx(rb0).unwrap();
        let ra1 = lim_a.prepare_tx(600, 1_000).unwrap();
        assert_eq!(
            lim_b.commit_tx(ra1).unwrap_err(),
            LoraTransportError::StaleReservation
        );

        // Stale overlapping reservation cannot double-spend airtime.
        let r1 = limiter.prepare_tx(1_099_999, 50_000).unwrap();
        let r2 = limiter.prepare_tx(1_099_999, 50_000).unwrap();
        limiter.commit_tx(r1).unwrap();
        assert_eq!(
            limiter.commit_tx(r2).unwrap_err(),
            LoraTransportError::StaleReservation
        );
        assert_eq!(limiter.active_airtime_at(1_099_999).unwrap(), 50_000);

        // Exact 61-slot conservative boundary (slot_micros = 100_000):
        // Packet sent at 1_099_999 (slot 10) remains retained at 7_099_999 (slot 70, exact 6.0s delta)
        // and expires cleanly at 7_100_000 (slot 71, 6.000001s delta > 6.0s window).
        assert_eq!(limiter.active_airtime_at(7_099_999).unwrap(), 50_000);
        assert_eq!(limiter.active_airtime_at(7_100_000).unwrap(), 0);

        // Multi-commit wrap-around past slot 60 (delta_slots = 0, 1, 60, 61) + u64::MAX timestamp.
        let mut wrap = LoraDutyCycleLimiterV1::try_new(6_000_000, 10_000, None).unwrap();
        let c1 = wrap.prepare_tx(0, 10_000).unwrap();
        wrap.commit_tx(c1).unwrap(); // slot 0
        let c2 = wrap.prepare_tx(0, 10_000).unwrap(); // delta_slots = 0
        wrap.commit_tx(c2).unwrap();
        let c3 = wrap.prepare_tx(100_000, 10_000).unwrap(); // delta_slots = 1 (slot 1)
        wrap.commit_tx(c3).unwrap();
        assert_eq!(wrap.active_airtime_at(100_000).unwrap(), 30_000);
        // Advance delta_slots = 60 from slot 1 -> slot 61 (expires slot 0, retains slot 1!).
        assert_eq!(wrap.active_airtime_at(6_100_000).unwrap(), 10_000);
        let c4 = wrap.prepare_tx(6_100_000, 50_000).unwrap();
        wrap.commit_tx(c4).unwrap(); // slot 61 wraps cleanly to ring_idx 0!
        assert_eq!(wrap.active_airtime_at(6_100_000).unwrap(), 60_000);
        // Advance delta_slots = 61 -> all expired, including at u64::MAX without overflow!
        assert_eq!(wrap.active_airtime_at(12_200_000).unwrap(), 0);
        let c_max = wrap.prepare_tx(u64::MAX, 1_000).unwrap();
        wrap.commit_tx(c_max).unwrap();
        assert_eq!(wrap.active_airtime_at(u64::MAX).unwrap(), 1_000);

        // Non-divisible window_micros (1 and 61) preserve conservative retention.
        let mut tiny_w = LoraDutyCycleLimiterV1::try_new(1, 1_000_000, None).unwrap();
        let rt = tiny_w.prepare_tx(10, 1).unwrap();
        tiny_w.commit_tx(rt).unwrap();
        assert_eq!(tiny_w.active_airtime_at(11).unwrap(), 1);
        assert_eq!(tiny_w.active_airtime_at(71).unwrap(), 0);

        let mut odd_w = LoraDutyCycleLimiterV1::try_new(61, 1_000_000, None).unwrap();
        let ro = odd_w.prepare_tx(0, 61).unwrap();
        odd_w.commit_tx(ro).unwrap();
        assert_eq!(odd_w.active_airtime_at(61).unwrap(), 61);
        assert_eq!(odd_w.active_airtime_at(122).unwrap(), 0);

        // Monotonic commit_epoch overflow fails closed.
        limiter.commit_epoch = u64::MAX;
        assert_eq!(
            limiter.prepare_tx(10_000_000, 1_000).unwrap_err(),
            LoraTransportError::InvalidConfig
        );

        // Transport config rejects broadcast local address and zero max_message_bytes.
        assert_eq!(
            LoraTransportConfigV1::eu868_default(LoraEndpointV1::BROADCAST).unwrap_err(),
            LoraTransportError::InvalidConfig
        );
        let mut zero_msg = LoraTransportConfigV1::eu868_default(LoraEndpointV1(1)).unwrap();
        zero_msg.max_message_bytes = 0;
        assert_eq!(
            zero_msg.validate().unwrap_err(),
            LoraTransportError::InvalidConfig
        );
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn wave16e1_lora_chunked_frame_codec_reassembly_buffer_too_small_and_endpoint_filtering() {
        use crate::chunked::{ChunkedFrameCodec, ObservedSourceReassembler, ReassemblerLimits};

        let mut tx = LoraPacketTransportV1::try_new(
            MockLoraHal::default(),
            LoraTransportConfigV1 {
                local_addr: LoraEndpointV1(0x1001),
                modulation: LoraModulationConfigV1::eu868_sf7_bw125(),
                duty_limiter: LoraDutyCycleLimiterV1::try_new(
                    LORA_DEFAULT_DUTY_WINDOW_MICROS,
                    1_000_000,
                    None,
                )
                .unwrap(),
                max_message_bytes: LORA_DEFAULT_MAX_MESSAGE_BYTES,
            },
        )
        .unwrap();
        let mut rx = LoraPacketTransportV1::try_new(
            MockLoraHal::default(),
            LoraTransportConfigV1::eu868_default(LoraEndpointV1(0x2002)).unwrap(),
        )
        .unwrap();

        let caps = tx.capabilities();
        assert_eq!(caps.max_frame_bytes, LORA_MAX_PHY_MTU_BYTES);
        assert_eq!(caps.bandwidth_class, BandwidthClass::UltraLow);
        assert_eq!(caps.reliability, ReliabilityClass::BestEffort);
        assert!(caps.supports_multicast);
        assert!(!caps.has_verified_link_principal);

        // Fragment a 600-byte message across 3 LoRa frames (stride P = 234 -> 234 + 234 + 132)
        let message: Vec<u8> = (0..600usize)
            .map(|i| (i.wrapping_mul(31) & 0xFF) as u8)
            .collect();
        let mut frames = Vec::new();
        let stride = ChunkedFrameCodec::stride_for_mtu(caps.max_frame_bytes).unwrap();
        assert_eq!(stride, LORA_CANONICAL_CHUNK_STRIDE_BYTES);
        let total_chunks = message.len().div_ceil(stride);
        assert_eq!(total_chunks, 3);

        for chunk_idx in 0..total_chunks {
            let start = chunk_idx * stride;
            let end = (start + stride).min(message.len());
            let mut frame_buf = [0u8; LORA_MAX_PHY_MTU_BYTES];
            let written = ChunkedFrameCodec::encode_frame_into(
                77,
                chunk_idx as u16,
                total_chunks as u16,
                &message[start..end],
                &mut frame_buf,
            )
            .unwrap();
            tx.try_send_frame(&LoraEndpointV1(0x2002), &frame_buf[..written])
                .unwrap();
        }

        // Inject a foreign unicast packet addressed to 0x9999 ahead of the legitimate frames
        rx.radio_mut().rx_queue.push_back(alloc::vec![
            LORA_LINK_VERSION_V1,
            0x05,
            0x00,
            0x99,
            0x99,
            0xDE,
            0xAD
        ]);
        while let Some(pkt) = tx.radio_mut().tx_queue.pop_front() {
            rx.radio_mut().rx_queue.push_back(pkt);
        }

        // Repeated undersized RX calls (32 -> 100) must return BufferTooSmall WITHOUT dropping the frame!
        let mut tiny = [0u8; 32];
        assert_eq!(
            rx.try_recv_frame(&mut tiny).unwrap_err(),
            PacketRecvError::BufferTooSmall {
                required: LORA_MAX_PHY_MTU_BYTES,
                available: 32,
            }
        );
        let mut medium = [0u8; 100];
        assert_eq!(
            rx.try_recv_frame(&mut medium).unwrap_err(),
            PacketRecvError::BufferTooSmall {
                required: LORA_MAX_PHY_MTU_BYTES,
                available: 100,
            }
        );

        // Now receive all 3 chunks into a full 250-byte buffer and reassemble via ObservedSourceReassembler
        let mut reassembler = ObservedSourceReassembler::new(ReassemblerLimits {
            mtu: caps.max_frame_bytes,
            max_message_bytes: caps.max_message_bytes,
            ..ReassemblerLimits::default()
        })
        .unwrap();

        let mut rx_buf = [0u8; LORA_MAX_PHY_MTU_BYTES];
        let mut assembled = None;
        while let Some(rec) = rx.try_recv_frame(&mut rx_buf).unwrap() {
            assert_eq!(rec.source, LoraEndpointV1(0x1001));
            frames.push(rec.len);
            if let Some(done) = reassembler
                .ingest_frame(1, rec.source, &rx_buf[..rec.len], 1_000)
                .unwrap()
            {
                assembled = Some(done);
            }
        }
        assert_eq!(frames, alloc::vec![250, 250, 148]);
        assert_eq!(assembled.unwrap().payload, message);

        // Broadcast frame addressed to 0xFFFF is accepted by local peer.
        tx.try_send_frame(&LoraEndpointV1::BROADCAST, b"beacon")
            .unwrap();
        let bcast = tx.radio_mut().tx_queue.pop_front().unwrap();
        rx.radio_mut().rx_queue.push_back(bcast);
        let rec = rx.try_recv_frame(&mut rx_buf).unwrap().unwrap();
        assert_eq!(rec.source, LoraEndpointV1(0x1001));
        assert_eq!(&rx_buf[..rec.len], b"beacon");

        // Zero-byte payload (phy_len == 5) roundtrips cleanly.
        tx.try_send_frame(&LoraEndpointV1(0x2002), &[]).unwrap();
        let empty_pkt = tx.radio_mut().tx_queue.pop_front().unwrap();
        assert_eq!(empty_pkt.len(), LORA_LINK_HEADER_BYTES);
        rx.radio_mut().rx_queue.push_back(empty_pkt);
        let rec_empty = rx.try_recv_frame(&mut rx_buf).unwrap().unwrap();
        assert_eq!(rec_empty.source, LoraEndpointV1(0x1001));
        assert_eq!(rec_empty.len, 0);

        // Foreign-unicast saturation: MAX_FOREIGN_DISCARD_PER_POLL (8) foreign packets + 1 local
        // yields Ok(None) on first poll (bounded loop latency) and delivers local packet on next poll!
        for _ in 0..MAX_FOREIGN_DISCARD_PER_POLL {
            rx.radio_mut().rx_queue.push_back(alloc::vec![
                LORA_LINK_VERSION_V1,
                0x07,
                0x00,
                0x88,
                0x88,
                0xFE
            ]);
        }
        tx.try_send_frame(&LoraEndpointV1(0x2002), b"after-flood")
            .unwrap();
        let after_flood = tx.radio_mut().tx_queue.pop_front().unwrap();
        rx.radio_mut().rx_queue.push_back(after_flood);
        assert_eq!(rx.try_recv_frame(&mut rx_buf).unwrap(), None);
        let rec_after = rx.try_recv_frame(&mut rx_buf).unwrap().unwrap();
        assert_eq!(&rx_buf[..rec_after.len], b"after-flood");

        // HAL RX error propagates cleanly as PacketRecvError::Transport(LoraTransportError::Hal(_)).
        rx.radio_mut().fail_next_rx_err = Some(0x55);
        assert_eq!(
            rx.try_recv_frame(&mut rx_buf).unwrap_err(),
            PacketRecvError::Transport(LoraTransportError::Hal(0x55))
        );

        // Malformed RX frames (short header, oversized FIFO, bad version, broadcast src) fail closed.
        for bad in [
            alloc::vec![LORA_LINK_VERSION_V1, 0x01, 0x00, 0x02],
            alloc::vec![LORA_LINK_VERSION_V1; LORA_SX12XX_MAX_FIFO_BYTES + 1],
            alloc::vec![0xFF, 0x01, 0x00, 0x02, 0x20, 0xAA],
            alloc::vec![LORA_LINK_VERSION_V1, 0xFF, 0xFF, 0x02, 0x20, 0xAA],
        ] {
            rx.radio_mut().rx_queue.push_back(bad);
            assert_eq!(
                rx.try_recv_frame(&mut rx_buf).unwrap_err(),
                PacketRecvError::TruncatedFrame
            );
        }
    }
}
