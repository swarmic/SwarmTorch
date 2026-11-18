#![forbid(unsafe_code)]
//! Isolated QUIC / `Iroh` 1-RTT carrier adapter (`ADR-0022` §4, `W12-02`, `ST-08`, `15D-2`).
//!
//! **Maturity Tier:** `Integration verified` — deterministic ALPN, 1-RTT stage gating,
//! bounded framing, link-vs-originator binding (`StrictOneToOne` / `DecoupledRelayAllowed`),
//! bounded sliding-window pre-auth quotas checked **before** variable-length body reads,
//! CSPRNG fresh challenge nonces with reuse rejection, and bounded handshake/read timeouts
//! are enforced end-to-end over live `iroh 1.4` 1-RTT QUIC connections.

use core::fmt;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use iroh::endpoint::presets;
use iroh::{Endpoint, EndpointAddr, RelayMode, SecretKey, TransportAddr};
use swarm_torch_core::crypto::{KeyPair, MessageAuth, Signature};
use swarm_torch_core::replay::ReplayProtection;
use swarm_torch_core::traits::PeerId;
use swarm_torch_net::carrier::{
    sign_link_attestation, CarrierError, LinkBindingPolicyV1, LinkHandshakeAttestationV1,
    QuicHandshakeStageV1, QuicLengthPrefixedCodecV1, QuicStreamHandshakePolicyV1,
    VerifiedCarrierBindingV1, VerifiedLinkPrincipalV1, ALPN_SWARMTORCH_DELIVERY_V1,
};
use swarm_torch_net::gossip::{
    BundledGossipDeliveryV1, BundledGossipDeliveryV2, GossipVerifyError,
    DEFAULT_MAX_GOSSIP_DELIVERY_BYTES,
};
use swarm_torch_net::protocol::{EnvelopeVersionPolicyV1, MessageEnvelope, VerifyError};

/// Default maximum number of distinct QUIC link principals tracked simultaneously in
/// [`LinkFrameQuotaTrackerV1`] before fail-closed saturation (`ST-08`).
pub const DEFAULT_MAX_TRACKED_LINK_PRINCIPALS: usize = 1024;

/// Default maximum number of per-frame timestamps retained per link principal window (`ST-08`).
pub const DEFAULT_MAX_WINDOW_TIMESTAMPS_PER_LINK: usize = 1024;

/// Default sliding-window duration (in seconds) for [`LinkFrameQuotaTrackerV1`] (`ST-08`).
pub const DEFAULT_LINK_QUOTA_WINDOW_SECS: u64 = 60;

/// Errors produced while decoding or authenticating an Iroh bi-stream frame.
#[derive(Debug)]
pub enum IrohCarrierError {
    /// Carrier ALPN, early-data, binding policy, or length-prefix contract failed.
    Carrier(CarrierError),
    /// Per-link principal rate limit exceeded before cryptographic admission.
    LinkRateLimited { link_peer: PeerId },
    /// Per-link quota table saturated (`max_tracked_links`) with active non-expired principals.
    LinkQuotaTableFull { max_tracked_links: usize },
    /// Direct envelope verification failed.
    Envelope(VerifyError),
    /// Multi-hop gossip envelope verification failed.
    Gossip(GossipVerifyError),
    /// QUIC accept, 1-RTT handshake, or stream I/O exceeded its bounded timeout (`15D-2`).
    Timeout { stage: &'static str },
    /// Live `iroh` QUIC endpoint, dial, or stream I/O failure.
    Transport(String),
}

impl fmt::Display for IrohCarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Carrier(e) => write!(f, "iroh carrier error: {e}"),
            Self::LinkRateLimited { .. } => write!(f, "per-link pre-auth frame quota exceeded"),
            Self::LinkQuotaTableFull { max_tracked_links } => write!(
                f,
                "per-link quota table saturated at max_tracked_links={max_tracked_links}"
            ),
            Self::Envelope(e) => write!(f, "envelope verification failed: {e}"),
            Self::Gossip(e) => write!(f, "gossip verification failed: {e}"),
            Self::Timeout { stage } => write!(f, "iroh operation timed out during {stage}"),
            Self::Transport(msg) => write!(f, "iroh transport error: {msg}"),
        }
    }
}

impl std::error::Error for IrohCarrierError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Carrier(e) => Some(e),
            Self::Envelope(e) => Some(e),
            Self::Gossip(e) => Some(e),
            Self::LinkRateLimited { .. }
            | Self::LinkQuotaTableFull { .. }
            | Self::Timeout { .. }
            | Self::Transport(_) => None,
        }
    }
}

impl From<CarrierError> for IrohCarrierError {
    fn from(value: CarrierError) -> Self {
        Self::Carrier(value)
    }
}

/// Adapter-sealed verified QUIC/Iroh transport link principal (`ST-08`).
///
/// Fields are private so callers cannot forge an Iroh link identity outside
/// [`IrohCarrierAdapterV1::accept_attested_link`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrohVerifiedLinkPrincipalV1 {
    inner: VerifiedLinkPrincipalV1,
}

impl IrohVerifiedLinkPrincipalV1 {
    /// Raw 32-byte Ed25519 transport endpoint public key / Iroh `EndpointId`.
    pub const fn endpoint_public_key(&self) -> &[u8; 32] {
        self.inner.endpoint_public_key()
    }

    /// Canonical SwarmTorch `PeerId` derived from the verified link endpoint key.
    pub const fn link_peer_id(&self) -> PeerId {
        self.inner.link_peer_id()
    }

    /// Read-only view of the underlying verified carrier principal.
    pub const fn as_carrier_principal(&self) -> &VerifiedLinkPrincipalV1 {
        &self.inner
    }
}

/// Adapter-sealed carrier binding between verified Iroh transport peer and authenticated
/// cryptographic originator (`ST-08`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrohVerifiedCarrierBindingV1 {
    inner: VerifiedCarrierBindingV1,
}

impl IrohVerifiedCarrierBindingV1 {
    /// Verified transport link principal.
    pub const fn link_principal(&self) -> IrohVerifiedLinkPrincipalV1 {
        IrohVerifiedLinkPrincipalV1 {
            inner: self.inner.link_principal(),
        }
    }

    /// Cryptographically authenticated message originator `PeerId`.
    pub const fn originator_peer_id(&self) -> PeerId {
        self.inner.originator_peer_id()
    }

    /// Applied link-binding policy.
    pub const fn policy(&self) -> LinkBindingPolicyV1 {
        self.inner.policy()
    }

    /// True when the transport link peer differs from the cryptographic originator.
    pub const fn relayed_via_intermediary(&self) -> bool {
        self.inner.relayed_via_intermediary()
    }

    /// Read-only view of the underlying verified carrier binding.
    pub const fn as_carrier_binding(&self) -> &VerifiedCarrierBindingV1 {
        &self.inner
    }
}

/// Verified frame received over an Iroh 1-RTT connection.
#[derive(Debug, Clone)]
pub struct AcceptedIrohEnvelopeV1 {
    binding: IrohVerifiedCarrierBindingV1,
    /// Authenticated message envelope.
    pub envelope: MessageEnvelope,
}

impl AcceptedIrohEnvelopeV1 {
    /// Adapter-sealed transport link vs. cryptographic originator binding.
    pub const fn binding(&self) -> &IrohVerifiedCarrierBindingV1 {
        &self.binding
    }
}

/// Verified v1 gossip delivery frame received over an Iroh 1-RTT connection.
#[derive(Debug, Clone)]
pub struct AcceptedIrohGossipV1 {
    binding: IrohVerifiedCarrierBindingV1,
    /// Authenticated bundled gossip delivery (`InnerUpdateV1` + `RecipientWrapperV1`).
    pub delivery: BundledGossipDeliveryV1,
}

impl AcceptedIrohGossipV1 {
    /// Adapter-sealed transport link vs. cryptographic originator binding.
    pub const fn binding(&self) -> &IrohVerifiedCarrierBindingV1 {
        &self.binding
    }
}

/// Verified v2 gossip delivery frame received over an Iroh 1-RTT connection (`ST-01`, `ST-03`, `ST-08`).
#[derive(Debug, Clone)]
pub struct AcceptedIrohGossipV2 {
    binding: IrohVerifiedCarrierBindingV1,
    /// Authenticated bundled v2 gossip delivery (`InnerUpdateV2` + `RecipientWrapperV2`).
    pub delivery: BundledGossipDeliveryV2,
}

impl AcceptedIrohGossipV2 {
    /// Adapter-sealed transport link vs. cryptographic originator binding.
    pub const fn binding(&self) -> &IrohVerifiedCarrierBindingV1 {
        &self.binding
    }
}

/// Bounded sliding-window per-link pre-auth frame budget tracker keyed strictly on the TLS/QUIC
/// link principal (`PeerId`), ensuring spoofed inner sender headers or Sybil ephemeral keys cannot
/// poison another peer's quota or trigger unbounded memory growth (`ADR-0022` §4, `ST-08`).
#[derive(Debug, Clone)]
pub struct LinkFrameQuotaTrackerV1 {
    max_frames_per_link: u64,
    window_secs: u64,
    max_tracked_links: usize,
    max_timestamps_per_link: usize,
    counts: BTreeMap<PeerId, VecDeque<u64>>,
}

impl Default for LinkFrameQuotaTrackerV1 {
    fn default() -> Self {
        Self::new(0)
    }
}

impl LinkFrameQuotaTrackerV1 {
    /// Construct a per-link sliding-window frame quota tracker (`max_frames_per_link = 0` disables
    /// throttling) using default bounded table limits (`1024` tracked links, `1024` timestamps/link,
    /// `60s` sliding window).
    pub fn new(max_frames_per_link: u64) -> Self {
        Self::with_limits(
            max_frames_per_link,
            DEFAULT_LINK_QUOTA_WINDOW_SECS,
            DEFAULT_MAX_TRACKED_LINK_PRINCIPALS,
            DEFAULT_MAX_WINDOW_TIMESTAMPS_PER_LINK,
        )
    }

    /// Construct a per-link sliding-window quota tracker with explicit table and window limits.
    pub fn with_limits(
        max_frames_per_link: u64,
        window_secs: u64,
        max_tracked_links: usize,
        max_timestamps_per_link: usize,
    ) -> Self {
        Self {
            max_frames_per_link,
            window_secs,
            max_tracked_links,
            max_timestamps_per_link,
            counts: BTreeMap::new(),
        }
    }

    /// Current number of tracked link principals in the quota map.
    pub fn tracked_link_count(&self) -> usize {
        self.counts.len()
    }

    /// Current number of retained frame timestamps for `link_peer`.
    pub fn active_window_count(&self, link_peer: &PeerId) -> usize {
        self.counts.get(link_peer).map_or(0, VecDeque::len)
    }

    fn prune_deque(q: &mut VecDeque<u64>, now_secs: u64, window_secs: u64) {
        while let Some(&ts) = q.front() {
            let expired = if window_secs == 0 {
                ts < now_secs
            } else {
                now_secs.saturating_sub(ts) >= window_secs
            };
            if expired {
                q.pop_front();
            } else {
                break;
            }
        }
    }

    fn insert_sorted_timestamp(q: &mut VecDeque<u64>, ts: u64) {
        if q.back().is_none_or(|&last| ts >= last) {
            q.push_back(ts);
        } else {
            let (front_slice, back_slice) = q.as_slices();
            let pos = if back_slice.first().is_none_or(|&b| ts < b) {
                front_slice.partition_point(|&x| x <= ts)
            } else {
                front_slice.len() + back_slice.partition_point(|&x| x <= ts)
            };
            q.insert(pos, ts);
        }
    }

    /// Record one frame arrival at `now_secs` against `link_principal.link_peer_id()`, enforcing lazy
    /// window expiry and fail-closed table saturation.
    pub fn record_frame_at(
        &mut self,
        link_principal: &IrohVerifiedLinkPrincipalV1,
        now_secs: u64,
    ) -> Result<(), IrohCarrierError> {
        if self.max_frames_per_link == 0 {
            return Ok(());
        }
        let link_peer = link_principal.link_peer_id();
        if let Some(q) = self.counts.get_mut(&link_peer) {
            Self::prune_deque(q, now_secs, self.window_secs);
            if q.len() as u64 >= self.max_frames_per_link || q.len() >= self.max_timestamps_per_link
            {
                return Err(IrohCarrierError::LinkRateLimited { link_peer });
            }
            Self::insert_sorted_timestamp(q, now_secs);
            return Ok(());
        }

        if self.max_tracked_links == 0 {
            return Err(IrohCarrierError::LinkQuotaTableFull {
                max_tracked_links: self.max_tracked_links,
            });
        }

        if self.counts.len() >= self.max_tracked_links {
            let window_secs = self.window_secs;
            self.counts.retain(|_, q| {
                Self::prune_deque(q, now_secs, window_secs);
                !q.is_empty()
            });
            if self.counts.len() >= self.max_tracked_links {
                return Err(IrohCarrierError::LinkQuotaTableFull {
                    max_tracked_links: self.max_tracked_links,
                });
            }
        }

        if self.max_timestamps_per_link == 0 {
            return Err(IrohCarrierError::LinkRateLimited { link_peer });
        }

        let mut q = VecDeque::new();
        q.push_back(now_secs);
        self.counts.insert(link_peer, q);
        Ok(())
    }

    /// Record one frame arrival using the host Unix clock. Prefer [`Self::record_frame_at`] for
    /// deterministic time-window evaluation.
    pub fn record_frame(
        &mut self,
        link_principal: &IrohVerifiedLinkPrincipalV1,
    ) -> Result<(), IrohCarrierError> {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.record_frame_at(link_principal, now_secs)
    }
}

/// Adapter coordinating 1-RTT QUIC / Iroh handshake acceptance, stream framing, and delivery verification.
#[derive(Debug, Clone)]
pub struct IrohCarrierAdapterV1 {
    /// Policy governing ALPN, 1-RTT stage, max frame bytes, and link-vs-originator binding.
    pub policy: QuicStreamHandshakePolicyV1,
    /// Bounded length-prefixed framing codec.
    pub codec: QuicLengthPrefixedCodecV1,
}

impl Default for IrohCarrierAdapterV1 {
    fn default() -> Self {
        let policy = QuicStreamHandshakePolicyV1::default();
        Self {
            codec: QuicLengthPrefixedCodecV1 {
                max_frame_bytes: policy.max_frame_bytes,
            },
            policy,
        }
    }
}

impl IrohCarrierAdapterV1 {
    /// Canonical ALPN bytes expected on Iroh connections (`b"swarmtorch/delivery/v1"`).
    pub const fn expected_alpn() -> &'static [u8] {
        ALPN_SWARMTORCH_DELIVERY_V1
    }

    /// Construct an adapter with explicit link binding policy and max frame size.
    pub fn new(binding_policy: LinkBindingPolicyV1, max_frame_bytes: usize) -> Self {
        let policy = QuicStreamHandshakePolicyV1 {
            expected_alpn: ALPN_SWARMTORCH_DELIVERY_V1,
            binding_policy,
            max_frame_bytes,
        };
        Self {
            policy,
            codec: QuicLengthPrefixedCodecV1 { max_frame_bytes },
        }
    }

    /// Verify an attested 1-RTT QUIC/Iroh transport link against `expected_receiver_nonce`,
    /// `ALPN_SWARMTORCH_DELIVERY_V1`, and `OneRttComplete` (`ST-08`).
    ///
    /// Returns an adapter-sealed [`IrohVerifiedLinkPrincipalV1`]; bare unauthenticated `[u8; 32]`
    /// keys cannot mint a principal.
    pub fn accept_attested_link(
        &self,
        expected_receiver_nonce: &[u8; 16],
        handshake_stage: QuicHandshakeStageV1,
        attestation: &LinkHandshakeAttestationV1<'_>,
    ) -> Result<IrohVerifiedLinkPrincipalV1, IrohCarrierError> {
        let inner = self
            .policy
            .validate_attested_connection(expected_receiver_nonce, handshake_stage, attestation)
            .map_err(IrohCarrierError::Carrier)?;
        Ok(IrohVerifiedLinkPrincipalV1 { inner })
    }

    fn prepare_frame_and_quota<'a>(
        &self,
        link_principal: &IrohVerifiedLinkPrincipalV1,
        stream_bytes: &'a [u8],
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<(&'a [u8], &'a [u8]), IrohCarrierError> {
        if let Some(tracker) = quota {
            tracker.record_frame_at(link_principal, u64::from(current_time_secs))?;
        }
        self.codec
            .decode_next_frame(stream_bytes)
            .map_err(IrohCarrierError::Carrier)
    }

    fn bind_sealed_originator(
        &self,
        link_principal: IrohVerifiedLinkPrincipalV1,
        originator: PeerId,
    ) -> Result<IrohVerifiedCarrierBindingV1, IrohCarrierError> {
        let inner = self
            .policy
            .bind_originator(*link_principal.as_carrier_principal(), originator)?;
        Ok(IrohVerifiedCarrierBindingV1 { inner })
    }

    fn verify_and_bind_gossip_originator(
        &self,
        link_principal: IrohVerifiedLinkPrincipalV1,
        wrapper_originator_key: &[u8; 32],
        inner_originator_key: &[u8; 32],
    ) -> Result<(IrohVerifiedCarrierBindingV1, PeerId), IrohCarrierError> {
        if wrapper_originator_key != inner_originator_key {
            return Err(IrohCarrierError::Gossip(
                GossipVerifyError::OriginatorKeyMismatch,
            ));
        }
        let originator = PeerId::try_from_public_key_bytes(wrapper_originator_key)
            .map_err(|_| IrohCarrierError::Gossip(GossipVerifyError::InvalidOriginatorKey))?;
        let binding = self.bind_sealed_originator(link_principal, originator)?;
        Ok((binding, originator))
    }

    /// Frame an outbound serialized message (`MessageEnvelope` or `BundledGossipDeliveryV1`/`V2`).
    pub fn encode_wire_frame(&self, payload: &[u8]) -> Result<Vec<u8>, IrohCarrierError> {
        self.codec
            .encode_frame(payload)
            .map_err(IrohCarrierError::Carrier)
    }

    /// Accept one length-prefixed direct [`MessageEnvelope`] frame from `stream_bytes`,
    /// enforcing per-link quota and link-vs-originator binding policy **before** replay admission,
    /// and return `(accepted, remaining_bytes)`.
    /// Admit one wire [`MessageEnvelope`] frame under strict [`EnvelopeVersionPolicyV1::RequireV2`].
    pub fn accept_envelope_frame<'a>(
        &self,
        link_principal: IrohVerifiedLinkPrincipalV1,
        stream_bytes: &'a [u8],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<(AcceptedIrohEnvelopeV1, &'a [u8]), IrohCarrierError> {
        self.accept_envelope_frame_with_policy(
            link_principal,
            stream_bytes,
            replay_guard,
            current_time_secs,
            quota,
            EnvelopeVersionPolicyV1::RequireV2,
        )
    }

    /// Admit one wire [`MessageEnvelope`] frame under an explicit [`EnvelopeVersionPolicyV1`].
    pub fn accept_envelope_frame_with_policy<'a>(
        &self,
        link_principal: IrohVerifiedLinkPrincipalV1,
        stream_bytes: &'a [u8],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
        version_policy: EnvelopeVersionPolicyV1,
    ) -> Result<(AcceptedIrohEnvelopeV1, &'a [u8]), IrohCarrierError> {
        let (frame, rest) =
            self.prepare_frame_and_quota(&link_principal, stream_bytes, current_time_secs, quota)?;
        let envelope = MessageEnvelope::deserialize_bounded(frame, self.policy.max_frame_bytes)
            .map_err(IrohCarrierError::Envelope)?;
        let originator = envelope
            .sender_peer_id()
            .map_err(|_| IrohCarrierError::Envelope(VerifyError::InvalidSenderKey))?;
        let binding = self.bind_sealed_originator(link_principal, originator)?;
        envelope
            .verify_authenticated_with_policy(replay_guard, current_time_secs, version_policy)
            .map_err(IrohCarrierError::Envelope)?;
        Ok((AcceptedIrohEnvelopeV1 { binding, envelope }, rest))
    }

    /// Accept one length-prefixed bundled gossip delivery frame (`BundledGossipDeliveryV1`),
    /// checking wrapper/inner originator parity and carrier binding policy **before** mutating
    /// `replay_guard`.
    pub fn accept_gossip_frame<'a>(
        &self,
        link_principal: IrohVerifiedLinkPrincipalV1,
        recipient_peer: PeerId,
        stream_bytes: &'a [u8],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<(AcceptedIrohGossipV1, &'a [u8]), IrohCarrierError> {
        let (frame, rest) =
            self.prepare_frame_and_quota(&link_principal, stream_bytes, current_time_secs, quota)?;
        let max_gossip = self
            .policy
            .max_frame_bytes
            .min(DEFAULT_MAX_GOSSIP_DELIVERY_BYTES);
        let delivery = BundledGossipDeliveryV1::from_bytes_bounded(frame, max_gossip)
            .map_err(IrohCarrierError::Gossip)?;
        let (binding, originator) = self.verify_and_bind_gossip_originator(
            link_principal,
            &delivery.wrapper.originator_public_key,
            &delivery.inner.originator_public_key,
        )?;
        let verified_originator = delivery
            .verify_for_recipient(recipient_peer, replay_guard, current_time_secs)
            .map_err(IrohCarrierError::Gossip)?;
        debug_assert_eq!(verified_originator, originator);
        Ok((AcceptedIrohGossipV1 { binding, delivery }, rest))
    }

    /// Accept one length-prefixed bundled v2 gossip delivery frame (`BundledGossipDeliveryV2`),
    /// enforcing wrapper/inner originator parity, link binding policy, and receiver-bound
    /// `session_id` replay protection before mutating `replay_guard` (`ST-01`, `ST-03`, `ST-08`).
    pub fn accept_gossip_v2_frame<'a>(
        &self,
        link_principal: IrohVerifiedLinkPrincipalV1,
        recipient_peer: PeerId,
        stream_bytes: &'a [u8],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<(AcceptedIrohGossipV2, &'a [u8]), IrohCarrierError> {
        let (frame, rest) =
            self.prepare_frame_and_quota(&link_principal, stream_bytes, current_time_secs, quota)?;
        let max_gossip = self
            .policy
            .max_frame_bytes
            .min(DEFAULT_MAX_GOSSIP_DELIVERY_BYTES);
        let delivery = BundledGossipDeliveryV2::from_bytes_bounded(frame, max_gossip)
            .map_err(IrohCarrierError::Gossip)?;
        let (binding, originator) = self.verify_and_bind_gossip_originator(
            link_principal,
            &delivery.wrapper.originator_public_key,
            &delivery.inner.originator_public_key,
        )?;
        let verified_originator = delivery
            .verify_for_recipient_v2(recipient_peer, replay_guard, current_time_secs)
            .map_err(IrohCarrierError::Gossip)?;
        debug_assert_eq!(verified_originator, originator);
        Ok((AcceptedIrohGossipV2 { binding, delivery }, rest))
    }
}

/// Wire attestation header size sent ahead of the SwarmTorch length-prefixed frame:
/// `[att_pubkey(32) || att_sig(64)]`.
const LINK_ATTESTATION_HEADER_BYTES: usize = 32 + 64;

/// Default maximum time waiting for an incoming QUIC connection and 1-RTT TLS completion.
pub const DEFAULT_IROH_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// Default maximum time waiting for bi-stream open and attestation/frame reads/writes.
pub const DEFAULT_IROH_READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Default bounded capacity of recently accepted challenge nonces tracked against reuse.
const DEFAULT_MAX_RECENT_NONCES: usize = 4_096;

/// Explicit handshake and stream read/write deadlines enforced by [`IrohEndpointBridgeV1`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrohAcceptTimeoutsV1 {
    /// Deadline wrapping `endpoint.accept()` / `endpoint.connect()` and 1-RTT TLS completion.
    pub handshake_timeout: Duration,
    /// Deadline wrapping `accept_bi()` / `open_bi()`, hello, nonce, attestation header, frame
    /// reads, and the receiver's post-body ack write.
    pub read_timeout: Duration,
}

impl Default for IrohAcceptTimeoutsV1 {
    fn default() -> Self {
        Self {
            handshake_timeout: DEFAULT_IROH_HANDSHAKE_TIMEOUT,
            read_timeout: DEFAULT_IROH_READ_TIMEOUT,
        }
    }
}

#[derive(Debug, Default)]
struct RecentNonceCache {
    order: VecDeque<[u8; 16]>,
    seen: BTreeSet<[u8; 16]>,
}

impl RecentNonceCache {
    fn contains(&self, nonce: &[u8; 16]) -> bool {
        self.seen.contains(nonce)
    }

    fn record_unique(&mut self, nonce: [u8; 16]) -> bool {
        if !self.seen.insert(nonce) {
            return false;
        }
        if self.order.len() >= DEFAULT_MAX_RECENT_NONCES {
            if let Some(evicted) = self.order.pop_front() {
                self.seen.remove(&evicted);
            }
        }
        self.order.push_back(nonce);
        true
    }
}

/// Live `iroh 1.4` 1-RTT QUIC transport bridge enforcing SwarmTorch link attestation,
/// strict 1-RTT completion (zero 0-RTT fast-path), TLS-keyed link binding, bounded reads,
/// pre-auth header+quota gating before body allocation, bounded timeouts, CSPRNG nonces,
/// and replay-safe admission (`ST-08`, `ADR-0022`, `15D-2`).
pub struct IrohEndpointBridgeV1 {
    endpoint: Endpoint,
    link_auth: MessageAuth,
    link_peer_id: PeerId,
    adapter: IrohCarrierAdapterV1,
    timeouts: IrohAcceptTimeoutsV1,
    recent_nonces: Mutex<RecentNonceCache>,
}

impl IrohEndpointBridgeV1 {
    /// Bind a headless loopback `127.0.0.1:0` `iroh` QUIC endpoint with relay/DNS discovery
    /// disabled, deterministic Ed25519 key identity from `seed`, and SwarmTorch ALPN pinning.
    pub async fn bind_loopback(
        seed: [u8; 32],
        adapter: IrohCarrierAdapterV1,
    ) -> Result<Self, IrohCarrierError> {
        let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0));
        Self::bind_at_with_alpn(
            seed,
            addr,
            RelayMode::Disabled,
            adapter,
            IrohCarrierAdapterV1::expected_alpn(),
        )
        .await
    }

    /// Bind an `iroh` QUIC endpoint on `bind_addr` with the specified [`RelayMode`].
    pub async fn bind_at(
        seed: [u8; 32],
        bind_addr: SocketAddr,
        relay_mode: RelayMode,
        adapter: IrohCarrierAdapterV1,
    ) -> Result<Self, IrohCarrierError> {
        Self::bind_at_with_alpn(
            seed,
            bind_addr,
            relay_mode,
            adapter,
            IrohCarrierAdapterV1::expected_alpn(),
        )
        .await
    }

    async fn bind_at_with_alpn(
        seed: [u8; 32],
        bind_addr: SocketAddr,
        relay_mode: RelayMode,
        adapter: IrohCarrierAdapterV1,
        alpn: &[u8],
    ) -> Result<Self, IrohCarrierError> {
        let kp =
            KeyPair::from_seed(seed).map_err(|e| IrohCarrierError::Transport(e.to_string()))?;
        let link_peer_id = kp.peer_id();
        let link_auth = MessageAuth::new(kp);
        let secret_key = SecretKey::from_bytes(&seed);

        let builder = Endpoint::builder(presets::Minimal)
            .secret_key(secret_key)
            .alpns(vec![alpn.to_vec()])
            .relay_mode(relay_mode)
            .clear_address_lookup()
            .clear_addr_filter()
            .bind_addr(bind_addr)
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;
        let endpoint = builder
            .bind()
            .await
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;

        Ok(Self {
            endpoint,
            link_auth,
            link_peer_id,
            adapter,
            timeouts: IrohAcceptTimeoutsV1::default(),
            recent_nonces: Mutex::new(RecentNonceCache::default()),
        })
    }

    /// Override the bounded accept, 1-RTT handshake, and stream read/write deadlines.
    pub const fn with_timeouts(mut self, timeouts: IrohAcceptTimeoutsV1) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// Configured handshake and stream read/write deadlines.
    pub const fn timeouts(&self) -> IrohAcceptTimeoutsV1 {
        self.timeouts
    }

    /// Generate a fresh non-zero 128-bit challenge nonce backed by OS CSPRNG (`getrandom`).
    pub fn generate_fresh_challenge_nonce(&self) -> Result<[u8; 16], IrohCarrierError> {
        let guard = self
            .recent_nonces
            .lock()
            .map_err(|_| IrohCarrierError::Transport("nonce cache poisoned".to_string()))?;
        for _ in 0..8 {
            let mut nonce = [0u8; 16];
            getrandom::fill(&mut nonce)
                .map_err(|e| IrohCarrierError::Transport(format!("CSPRNG failed: {e}")))?;
            if nonce != [0u8; 16] && !guard.contains(&nonce) {
                return Ok(nonce);
            }
        }
        Err(IrohCarrierError::Carrier(
            CarrierError::InvalidLinkAttestation,
        ))
    }

    fn reserve_unique_nonce(&self, nonce: [u8; 16]) -> Result<(), IrohCarrierError> {
        if nonce == [0u8; 16] {
            return Err(IrohCarrierError::Carrier(
                CarrierError::InvalidLinkAttestation,
            ));
        }
        let mut guard = self
            .recent_nonces
            .lock()
            .map_err(|_| IrohCarrierError::Transport("nonce cache poisoned".to_string()))?;
        if !guard.record_unique(nonce) {
            return Err(IrohCarrierError::Carrier(
                CarrierError::InvalidLinkAttestation,
            ));
        }
        Ok(())
    }

    /// Access the underlying `iroh::Endpoint`.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Access the configured [`IrohCarrierAdapterV1`].
    pub fn adapter(&self) -> &IrohCarrierAdapterV1 {
        &self.adapter
    }

    /// Return the direct addressing info (`EndpointAddr`) for peers to dial this endpoint,
    /// falling back to immediate `bound_sockets()` loopback addresses when discovery is disabled.
    pub fn addr(&self) -> EndpointAddr {
        let mut ep_addr = self.endpoint.addr();
        if ep_addr.is_empty() {
            ep_addr = EndpointAddr::from_parts(
                self.endpoint.id(),
                self.endpoint
                    .bound_sockets()
                    .into_iter()
                    .map(TransportAddr::Ip),
            );
        }
        ep_addr
    }

    /// Return the canonical `PeerId` derived from this endpoint's Ed25519 transport public key.
    pub const fn link_peer_id(&self) -> PeerId {
        self.link_peer_id
    }

    /// Gracefully close the underlying `iroh::Endpoint`.
    pub async fn close(&self) {
        self.endpoint.close().await;
    }

    const STREAM_HELLO_V1: &'static [u8; 4] = b"STQ1";

    /// Connect to `target` over a strictly completed `1-RTT` `iroh` QUIC connection (never 0-RTT),
    /// exchange the receiver's fresh challenge nonce, sign a [`LinkHandshakeAttestationV1`], and
    /// stream the bounded length-prefixed payload.
    ///
    /// `Ok(())` means the frame was written and the stream finished. It does not confirm that
    /// the receiver admitted or verified the frame: the receiver's 1-byte ack is best-effort and
    /// is sent before envelope/gossip verification, so admission is only observable receiver-side.
    pub async fn send_frame_1rtt(
        &self,
        target: impl Into<EndpointAddr>,
        payload: &[u8],
    ) -> Result<(), IrohCarrierError> {
        let wire_frame = self.adapter.encode_wire_frame(payload)?;
        let conn = tokio::time::timeout(
            self.timeouts.handshake_timeout,
            self.endpoint
                .connect(target, IrohCarrierAdapterV1::expected_alpn()),
        )
        .await
        .map_err(|_| IrohCarrierError::Timeout {
            stage: "quic_connect",
        })?
        .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;

        let negotiated_alpn = conn.alpn();
        if negotiated_alpn != IrohCarrierAdapterV1::expected_alpn() {
            conn.close(1u32.into(), b"alpn mismatch");
            return Err(IrohCarrierError::Carrier(CarrierError::AlpnMismatch));
        }

        let (mut send, mut recv) = tokio::time::timeout(self.timeouts.read_timeout, conn.open_bi())
            .await
            .map_err(|_| {
                conn.close(2u32.into(), b"open_bi timeout");
                IrohCarrierError::Timeout { stage: "open_bi" }
            })?
            .map_err(|e| {
                conn.close(2u32.into(), b"open_bi failed");
                IrohCarrierError::Transport(e.to_string())
            })?;

        // Write 4-byte stream hello immediately so QUIC transmits client-initiated stream opening.
        tokio::time::timeout(
            self.timeouts.read_timeout,
            send.write_all(Self::STREAM_HELLO_V1),
        )
        .await
        .map_err(|_| {
            conn.close(2u32.into(), b"write hello timeout");
            IrohCarrierError::Timeout {
                stage: "write_hello",
            }
        })?
        .map_err(|e| {
            conn.close(2u32.into(), b"write hello failed");
            IrohCarrierError::Transport(e.to_string())
        })?;

        let mut receiver_nonce = [0u8; 16];
        tokio::time::timeout(
            self.timeouts.read_timeout,
            recv.read_exact(&mut receiver_nonce),
        )
        .await
        .map_err(|_| {
            conn.close(2u32.into(), b"read nonce timeout");
            IrohCarrierError::Timeout {
                stage: "read_nonce",
            }
        })?
        .map_err(|e| {
            conn.close(2u32.into(), b"read nonce failed");
            IrohCarrierError::Transport(e.to_string())
        })?;
        if receiver_nonce == [0u8; 16] {
            conn.close(2u32.into(), b"zero challenge nonce");
            return Err(IrohCarrierError::Carrier(
                CarrierError::InvalidLinkAttestation,
            ));
        }

        let att = sign_link_attestation(&self.link_auth, negotiated_alpn, receiver_nonce)?;
        let mut out = Vec::with_capacity(LINK_ATTESTATION_HEADER_BYTES + wire_frame.len());
        out.extend_from_slice(&att.endpoint_public_key);
        out.extend_from_slice(att.signature.as_bytes());
        out.extend_from_slice(&wire_frame);

        tokio::time::timeout(self.timeouts.read_timeout, send.write_all(&out))
            .await
            .map_err(|_| {
                conn.close(2u32.into(), b"write frame timeout");
                IrohCarrierError::Timeout {
                    stage: "write_frame",
                }
            })?
            .map_err(|e| {
                conn.close(2u32.into(), b"write frame failed");
                IrohCarrierError::Transport(e.to_string())
            })?;
        send.finish().map_err(|e| {
            conn.close(2u32.into(), b"finish stream failed");
            IrohCarrierError::Transport(e.to_string())
        })?;
        let _ = tokio::time::timeout(self.timeouts.read_timeout, recv.read_to_end(1)).await;
        Ok(())
    }

    /// Accept one incoming `iroh` connection strictly after `1-RTT` TLS completion (never 0-RTT),
    /// issue a CSPRNG challenge nonce, enforce link attestation and optional pre-auth `quota`
    /// **before** reading the variable-length payload, and return the sealed principal and frame.
    pub async fn accept_1rtt_frame_fresh_nonce(
        &self,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<(IrohVerifiedLinkPrincipalV1, Vec<u8>), IrohCarrierError> {
        let nonce = self.generate_fresh_challenge_nonce()?;
        self.accept_1rtt_frame_authenticated(nonce, current_time_secs, quota)
            .await
    }

    /// Accept one incoming `iroh` connection strictly after `1-RTT` TLS completion using a
    /// caller-supplied unique `receiver_nonce`, enforcing ALPN, handshake/read timeouts,
    /// signature attestation, and `quota` BEFORE reading/allocating the variable-length body.
    pub async fn accept_1rtt_frame_authenticated(
        &self,
        receiver_nonce: [u8; 16],
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<(IrohVerifiedLinkPrincipalV1, Vec<u8>), IrohCarrierError> {
        let (principal, frame, _conn) = self
            .accept_1rtt_frame_inner(receiver_nonce, current_time_secs, quota)
            .await?;
        Ok((principal, frame))
    }

    /// Accept one incoming `iroh` connection strictly after `1-RTT` TLS completion (never 0-RTT),
    /// issue `receiver_nonce` challenge, verify both ALPN and TLS `remote_id == attestation_key`,
    /// and return a sealed [`IrohVerifiedLinkPrincipalV1`] + bounded wire frame bytes.
    pub async fn accept_1rtt_frame(
        &self,
        receiver_nonce: [u8; 16],
    ) -> Result<(IrohVerifiedLinkPrincipalV1, Vec<u8>), IrohCarrierError> {
        let now_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|d| u32::try_from(d.as_secs()).ok())
            .unwrap_or(0);
        self.accept_1rtt_frame_authenticated(receiver_nonce, now_secs, None)
            .await
    }

    async fn accept_1rtt_frame_inner(
        &self,
        receiver_nonce: [u8; 16],
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<
        (
            IrohVerifiedLinkPrincipalV1,
            Vec<u8>,
            iroh::endpoint::Connection,
        ),
        IrohCarrierError,
    > {
        self.reserve_unique_nonce(receiver_nonce)?;

        let incoming =
            tokio::time::timeout(self.timeouts.handshake_timeout, self.endpoint.accept())
                .await
                .map_err(|_| IrohCarrierError::Timeout {
                    stage: "endpoint_accept",
                })?
                .ok_or_else(|| IrohCarrierError::Transport("iroh endpoint closed".to_string()))?;

        // Strictly require full 1-RTT QUIC/TLS completion (never call `.into_0rtt()`).
        let accepting = incoming
            .accept()
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;
        let conn = tokio::time::timeout(self.timeouts.handshake_timeout, accepting)
            .await
            .map_err(|_| IrohCarrierError::Timeout {
                stage: "quic_1rtt_handshake",
            })?
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;

        let negotiated_alpn = conn.alpn().to_vec();
        if negotiated_alpn.as_slice() != IrohCarrierAdapterV1::expected_alpn() {
            conn.close(1u32.into(), b"alpn mismatch");
            return Err(IrohCarrierError::Carrier(CarrierError::AlpnMismatch));
        }
        let remote_id_bytes: [u8; 32] = *conn.remote_id().as_bytes();

        let (mut send, mut recv) =
            tokio::time::timeout(self.timeouts.read_timeout, conn.accept_bi())
                .await
                .map_err(|_| {
                    conn.close(2u32.into(), b"accept_bi timeout");
                    IrohCarrierError::Timeout { stage: "accept_bi" }
                })?
                .map_err(|e| {
                    conn.close(2u32.into(), b"accept_bi failed");
                    IrohCarrierError::Transport(e.to_string())
                })?;

        let mut hello = [0u8; 4];
        tokio::time::timeout(self.timeouts.read_timeout, recv.read_exact(&mut hello))
            .await
            .map_err(|_| {
                conn.close(3u32.into(), b"read hello timeout");
                IrohCarrierError::Timeout {
                    stage: "read_hello",
                }
            })?
            .map_err(|e| {
                conn.close(3u32.into(), b"read hello failed");
                IrohCarrierError::Transport(e.to_string())
            })?;
        if &hello != Self::STREAM_HELLO_V1 {
            conn.close(3u32.into(), b"invalid stream hello");
            return Err(IrohCarrierError::Carrier(
                CarrierError::EmptyOrTruncatedFrame,
            ));
        }

        tokio::time::timeout(self.timeouts.read_timeout, send.write_all(&receiver_nonce))
            .await
            .map_err(|_| {
                conn.close(3u32.into(), b"write challenge timeout");
                IrohCarrierError::Timeout {
                    stage: "write_nonce",
                }
            })?
            .map_err(|e| {
                conn.close(3u32.into(), b"write challenge failed");
                IrohCarrierError::Transport(e.to_string())
            })?;

        // Step 1: Read ONLY the fixed 96-byte link attestation header (`pubkey[32] || sig[64]`)
        // under a bounded timeout BEFORE allocating or reading any variable-length payload!
        let mut header = [0u8; LINK_ATTESTATION_HEADER_BYTES];
        tokio::time::timeout(self.timeouts.read_timeout, recv.read_exact(&mut header))
            .await
            .map_err(|_| {
                conn.close(4u32.into(), b"read attestation header timeout");
                IrohCarrierError::Timeout {
                    stage: "read_attestation_header",
                }
            })?
            .map_err(|_| {
                conn.close(4u32.into(), b"truncated attestation header");
                IrohCarrierError::Carrier(CarrierError::EmptyOrTruncatedFrame)
            })?;

        let mut att_pubkey = [0u8; 32];
        att_pubkey.copy_from_slice(&header[..32]);
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&header[32..LINK_ATTESTATION_HEADER_BYTES]);

        // Step 2: Cryptographic binding between QUIC TLS 1.3 `EndpointId` and attestation key.
        if att_pubkey != remote_id_bytes {
            conn.close(5u32.into(), b"link key mismatch");
            return Err(IrohCarrierError::Carrier(
                CarrierError::InvalidLinkAttestation,
            ));
        }

        let att = LinkHandshakeAttestationV1 {
            endpoint_public_key: att_pubkey,
            negotiated_alpn: &negotiated_alpn,
            receiver_nonce,
            signature: Signature::from_bytes(sig_bytes),
        };
        let principal = match self.adapter.accept_attested_link(
            &receiver_nonce,
            QuicHandshakeStageV1::OneRttComplete,
            &att,
        ) {
            Ok(p) => p,
            Err(err) => {
                conn.close(5u32.into(), b"invalid link attestation");
                return Err(err);
            }
        };

        // Step 3: Enforce per-link quota BEFORE reading or allocating the variable-length body!
        if let Some(tracker) = quota {
            if let Err(err) = tracker.record_frame_at(&principal, u64::from(current_time_secs)) {
                conn.close(6u32.into(), b"link quota exceeded");
                return Err(err);
            }
        }

        // Step 4: Only authenticated, quota-admitted links may stream `wire_frame` up to cap.
        let max_body = 4usize.saturating_add(self.adapter.codec.max_frame_bytes);
        let wire_frame =
            tokio::time::timeout(self.timeouts.read_timeout, recv.read_to_end(max_body))
                .await
                .map_err(|_| {
                    conn.close(7u32.into(), b"read frame body timeout");
                    IrohCarrierError::Timeout {
                        stage: "read_frame_body",
                    }
                })?
                .map_err(|e| {
                    conn.close(7u32.into(), b"frame body too large or aborted");
                    if matches!(e, iroh::endpoint::ReadToEndError::TooLong) {
                        IrohCarrierError::Carrier(CarrierError::FrameTooLarge {
                            len: self.adapter.codec.max_frame_bytes.saturating_add(1),
                            max: self.adapter.codec.max_frame_bytes,
                        })
                    } else {
                        IrohCarrierError::Transport(e.to_string())
                    }
                })?;
        // Best-effort ack, bounded like every other stage: a peer that withholds stream
        // credit must not be able to park this accept forever after its body was read.
        let _ = tokio::time::timeout(self.timeouts.read_timeout, send.write_all(&[0x01])).await;
        let _ = send.finish();

        Ok((principal, wire_frame, conn))
    }

    /// Accept one live `1-RTT` QUIC frame using a CSPRNG challenge nonce and admit a direct
    /// [`MessageEnvelope`] under strict [`EnvelopeVersionPolicyV1::RequireV2`].
    pub async fn accept_envelope_over_quic_fresh_nonce(
        &self,
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<AcceptedIrohEnvelopeV1, IrohCarrierError> {
        let nonce = self.generate_fresh_challenge_nonce()?;
        self.accept_envelope_over_quic(nonce, replay_guard, current_time_secs, quota)
            .await
    }

    /// Accept one live `1-RTT` QUIC frame and admit a direct [`MessageEnvelope`] under
    /// pre-auth link quota + replay protection + strict [`EnvelopeVersionPolicyV1::RequireV2`].
    pub async fn accept_envelope_over_quic(
        &self,
        receiver_nonce: [u8; 16],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<AcceptedIrohEnvelopeV1, IrohCarrierError> {
        self.accept_envelope_over_quic_with_policy(
            receiver_nonce,
            replay_guard,
            current_time_secs,
            quota,
            EnvelopeVersionPolicyV1::RequireV2,
        )
        .await
    }

    /// Accept one live `1-RTT` QUIC frame and admit a direct [`MessageEnvelope`] under
    /// an explicit [`EnvelopeVersionPolicyV1`].
    pub async fn accept_envelope_over_quic_with_policy(
        &self,
        receiver_nonce: [u8; 16],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
        version_policy: EnvelopeVersionPolicyV1,
    ) -> Result<AcceptedIrohEnvelopeV1, IrohCarrierError> {
        // Quota is already checked and charged inside `accept_1rtt_frame_inner` prior to body read.
        let (principal, wire_frame, conn) = self
            .accept_1rtt_frame_inner(receiver_nonce, current_time_secs, quota)
            .await?;
        let (accepted, rest) = self
            .adapter
            .accept_envelope_frame_with_policy(
                principal,
                &wire_frame,
                replay_guard,
                current_time_secs,
                None,
                version_policy,
            )
            .inspect_err(|_| {
                conn.close(8u32.into(), b"envelope verification failed");
            })?;
        if !rest.is_empty() {
            conn.close(8u32.into(), b"trailing stream bytes");
            return Err(IrohCarrierError::Carrier(
                CarrierError::EmptyOrTruncatedFrame,
            ));
        }
        Ok(accepted)
    }

    /// Accept one live `1-RTT` QUIC frame using a CSPRNG challenge nonce and admit a v2 gossip
    /// [`BundledGossipDeliveryV2`].
    pub async fn accept_gossip_v2_over_quic_fresh_nonce(
        &self,
        recipient_peer: PeerId,
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<AcceptedIrohGossipV2, IrohCarrierError> {
        let nonce = self.generate_fresh_challenge_nonce()?;
        self.accept_gossip_v2_over_quic(
            nonce,
            recipient_peer,
            replay_guard,
            current_time_secs,
            quota,
        )
        .await
    }

    /// Accept one live `1-RTT` QUIC frame and admit a v2 gossip [`BundledGossipDeliveryV2`].
    pub async fn accept_gossip_v2_over_quic(
        &self,
        receiver_nonce: [u8; 16],
        recipient_peer: PeerId,
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<AcceptedIrohGossipV2, IrohCarrierError> {
        // Quota is already checked and charged inside `accept_1rtt_frame_inner` prior to body read.
        let (principal, wire_frame, conn) = self
            .accept_1rtt_frame_inner(receiver_nonce, current_time_secs, quota)
            .await?;
        let (accepted, rest) = self
            .adapter
            .accept_gossip_v2_frame(
                principal,
                recipient_peer,
                &wire_frame,
                replay_guard,
                current_time_secs,
                None,
            )
            .inspect_err(|_| {
                conn.close(8u32.into(), b"gossip verification failed");
            })?;
        if !rest.is_empty() {
            conn.close(8u32.into(), b"trailing stream bytes");
            return Err(IrohCarrierError::Carrier(
                CarrierError::EmptyOrTruncatedFrame,
            ));
        }
        Ok(accepted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use swarm_torch_core::crypto::{KeyPair, MessageAuth, SessionHandshakeChallengeV2};
    use swarm_torch_core::replay::{CohortAllowlist, ReplayStartupPolicyV1};
    use swarm_torch_net::carrier::sign_link_attestation;
    use swarm_torch_net::gossip::{InnerUpdateV1, InnerUpdateV2, PayloadKindV1};
    use swarm_torch_net::protocol::MessageType;

    fn attested_link(
        adapter: &IrohCarrierAdapterV1,
        auth: &MessageAuth,
        nonce: [u8; 16],
    ) -> IrohVerifiedLinkPrincipalV1 {
        let att = sign_link_attestation(auth, ALPN_SWARMTORCH_DELIVERY_V1, nonce).unwrap();
        adapter
            .accept_attested_link(&nonce, QuicHandshakeStageV1::OneRttComplete, &att)
            .unwrap()
    }

    #[test]
    fn iroh_carrier_rejects_early_data_enforces_binding_modes_and_authenticates_relay_gossip() {
        let relay_kp = KeyPair::from_seed([0x11; 32]).unwrap();
        let origin_kp = KeyPair::from_seed([0x22; 32]).unwrap();
        let coord_kp = KeyPair::from_seed([0x33; 32]).unwrap();
        let relay_auth = MessageAuth::new(relay_kp.clone());
        let origin_auth = MessageAuth::new(origin_kp.clone());
        let link_nonce = [0x5A; 16];

        let relay_adapter =
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::DecoupledRelayAllowed, 65_536);
        let strict_adapter = IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 65_536);

        let relay_att =
            sign_link_attestation(&relay_auth, ALPN_SWARMTORCH_DELIVERY_V1, link_nonce).unwrap();

        // 0-RTT, wrong ALPN, and wrong receiver nonce rejected pre-admission
        assert!(matches!(
            relay_adapter.accept_attested_link(
                &link_nonce,
                QuicHandshakeStageV1::ZeroRtt,
                &relay_att
            ),
            Err(IrohCarrierError::Carrier(CarrierError::EarlyDataRejected))
        ));
        let wrong_alpn_att = LinkHandshakeAttestationV1 {
            negotiated_alpn: b"other/proto",
            ..relay_att
        };
        assert!(matches!(
            relay_adapter.accept_attested_link(
                &link_nonce,
                QuicHandshakeStageV1::OneRttComplete,
                &wrong_alpn_att
            ),
            Err(IrohCarrierError::Carrier(CarrierError::AlpnMismatch))
        ));
        assert!(matches!(
            relay_adapter.accept_attested_link(
                &[0x99; 16],
                QuicHandshakeStageV1::OneRttComplete,
                &relay_att
            ),
            Err(IrohCarrierError::Carrier(
                CarrierError::InvalidLinkAttestation
            ))
        ));

        let relay_link = attested_link(&relay_adapter, &relay_auth, link_nonce);
        let direct_origin_link = attested_link(&strict_adapter, &origin_auth, [0x6B; 16]);

        // Legacy v1 path under explicit GenesisRosterBootstrap compatibility mode
        let env = MessageEnvelope::new_v1(
            *origin_kp.public_key(),
            [0xAB; 16],
            1,
            5,
            MessageType::GradientUpdate,
            vec![10, 20, 30],
        )
        .sign_with(&origin_auth);
        let wire_frame = relay_adapter
            .encode_wire_frame(&env.serialize().unwrap())
            .unwrap();

        let allowlist = CohortAllowlist::from_peers([origin_kp.peer_id()]);
        let mut strict_replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(allowlist.clone());
        let mut relay_replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(allowlist.clone());

        // Default RequireV2 rejects legacy v1 envelopes cleanly without mutating strict_replay.
        assert!(matches!(
            strict_adapter.accept_envelope_frame(
                direct_origin_link,
                &wire_frame,
                &mut strict_replay,
                env.timestamp,
                None
            ),
            Err(IrohCarrierError::Envelope(
                VerifyError::UnsupportedVersion { major: 0, minor: 2 }
            ))
        ));
        assert_eq!(strict_replay.cache_size(), 0);

        // StrictOneToOne rejects relay link != originator WITHOUT mutating strict_replay!
        assert!(matches!(
            strict_adapter.accept_envelope_frame_with_policy(
                relay_link,
                &wire_frame,
                &mut strict_replay,
                env.timestamp,
                None,
                EnvelopeVersionPolicyV1::AllowLegacyV1AndV2,
            ),
            Err(IrohCarrierError::Carrier(
                CarrierError::StrictBindingMismatch { .. }
            ))
        ));
        // Prove strict_replay remained clean: the exact same envelope succeeds over direct_origin_link!
        let (strict_direct, strict_rest) = strict_adapter
            .accept_envelope_frame_with_policy(
                direct_origin_link,
                &wire_frame,
                &mut strict_replay,
                env.timestamp,
                None,
                EnvelopeVersionPolicyV1::AllowLegacyV1AndV2,
            )
            .unwrap();
        assert!(strict_rest.is_empty());
        assert!(!strict_direct.binding().relayed_via_intermediary());

        // DecoupledRelayAllowed admits and flags relayed_via_intermediary = true
        let (accepted_env, rest) = relay_adapter
            .accept_envelope_frame_with_policy(
                relay_link,
                &wire_frame,
                &mut relay_replay,
                env.timestamp,
                None,
                EnvelopeVersionPolicyV1::AllowLegacyV1AndV2,
            )
            .unwrap();
        assert!(rest.is_empty());
        assert!(accepted_env.binding().relayed_via_intermediary());
        assert_eq!(
            accepted_env.binding().originator_peer_id(),
            origin_kp.peer_id()
        );

        // Relayed BundledGossipDeliveryV1 bound to coordinator
        let inner = InnerUpdateV1 {
            originator_public_key: *origin_kp.public_key(),
            swarm_instance_id: [0xAB; 16],
            round_id: 5,
            contribution_id: 2,
            base_checkpoint_digest: [0xCD; 32],
            payload_kind: PayloadKindV1::ModelDeltaV1,
            compression_schema_version: 0,
            num_elements: 4,
            reported_examples: 16,
            encoded_payload: vec![1, 2, 3, 4],
            originator_signature: Vec::new(),
        };
        let gossip = BundledGossipDeliveryV1::new_signed(
            &origin_auth,
            inner,
            coord_kp.peer_id(),
            2,
            1,
            1000,
        );
        let gossip_frame = relay_adapter
            .encode_wire_frame(&gossip.to_bytes().unwrap())
            .unwrap();

        let mut strict_gossip_replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(allowlist.clone());
        // StrictOneToOne rejects relayed gossip before touching strict_gossip_replay!
        assert!(matches!(
            strict_adapter.accept_gossip_frame(
                relay_link,
                coord_kp.peer_id(),
                &gossip_frame,
                &mut strict_gossip_replay,
                1000,
                None,
            ),
            Err(IrohCarrierError::Carrier(
                CarrierError::StrictBindingMismatch { .. }
            ))
        ));
        // Tampered inner payload over DecoupledRelayAllowed must fail without mutating replay!
        let mut tampered_gossip = gossip.clone();
        tampered_gossip.inner.encoded_payload[0] ^= 0xFF;
        let tampered_frame = relay_adapter
            .encode_wire_frame(&tampered_gossip.to_bytes().unwrap())
            .unwrap();
        assert!(matches!(
            relay_adapter.accept_gossip_frame(
                relay_link,
                coord_kp.peer_id(),
                &tampered_frame,
                &mut strict_gossip_replay,
                1000,
                None,
            ),
            Err(IrohCarrierError::Gossip(
                GossipVerifyError::InnerDigestMismatch
            ))
        ));

        // After both rejected attempts, the valid frame must still be admitted cleanly!
        let mut coord_replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(allowlist.clone());
        let (accepted_gossip, tail) = relay_adapter
            .accept_gossip_frame(
                relay_link,
                coord_kp.peer_id(),
                &gossip_frame,
                &mut coord_replay,
                1000,
                None,
            )
            .unwrap();
        assert!(tail.is_empty());
        assert!(accepted_gossip.binding().relayed_via_intermediary());
        assert_eq!(
            accepted_gossip.binding().originator_peer_id(),
            origin_kp.peer_id()
        );
        // Also prove strict_gossip_replay stayed clean across StrictBindingMismatch + InnerDigestMismatch!
        assert!(strict_adapter
            .accept_gossip_frame(
                direct_origin_link,
                coord_kp.peer_id(),
                &gossip_frame,
                &mut strict_gossip_replay,
                1000,
                None,
            )
            .is_ok());

        // Strict RequireSessionHandshake (default) + BundledGossipDeliveryV2 over relay
        let scope = [0xAB; 16];
        let run_id = [0xCD; 16];
        let policy_digest = [0xEF; 32];
        let mut v2_replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(allowlist);
        let challenge = SessionHandshakeChallengeV2 {
            swarm_instance_id: scope,
            run_id,
            policy_digest,
            receiver_peer: coord_kp.peer_id(),
            sender_peer: origin_kp.peer_id(),
            requested_epoch: 1,
            initial_sequence: 0,
            receiver_nonce: [0x77; 32],
            issued_at_secs: 1000,
        };
        let session_id = challenge.derive_session_id().unwrap();
        let inner_v2 = InnerUpdateV2 {
            originator_public_key: *origin_kp.public_key(),
            swarm_instance_id: scope,
            run_id,
            round_id: 5,
            contribution_id: 3,
            base_checkpoint_digest: [0x12; 32],
            context_policy_digest: policy_digest,
            payload_kind: PayloadKindV1::ModelDeltaV1,
            compression_schema_version: 0,
            num_elements: 4,
            reported_examples: 16,
            encoded_payload: vec![5, 6, 7, 8],
            originator_signature: Vec::new(),
        };
        let gossip_v2 = BundledGossipDeliveryV2::new_signed(
            &origin_auth,
            inner_v2,
            coord_kp.peer_id(),
            session_id,
            1,
            1,
            1000,
        );
        let gossip_v2_frame = relay_adapter
            .encode_wire_frame(&gossip_v2.to_bytes().unwrap())
            .unwrap();

        // Rejected before handshake completes!
        assert!(relay_adapter
            .accept_gossip_v2_frame(
                relay_link,
                coord_kp.peer_id(),
                &gossip_v2_frame,
                &mut v2_replay,
                1000,
                None,
            )
            .is_err());

        // Complete receiver-challenged v2 session handshake then admit relayed v2 frame
        v2_replay.register_recovery_challenge_v2(challenge).unwrap();
        let hs_sig = origin_auth.sign_session_handshake_v2(&challenge).unwrap();
        v2_replay
            .complete_recovery_handshake_v2(scope, run_id, origin_kp.public_key(), &hs_sig, 1000)
            .unwrap();
        let (accepted_v2, v2_tail) = relay_adapter
            .accept_gossip_v2_frame(
                relay_link,
                coord_kp.peer_id(),
                &gossip_v2_frame,
                &mut v2_replay,
                1000,
                None,
            )
            .unwrap();
        assert!(v2_tail.is_empty());
        assert!(accepted_v2.binding().relayed_via_intermediary());
        assert_eq!(
            accepted_v2.binding().originator_peer_id(),
            origin_kp.peer_id()
        );
        assert_eq!(accepted_v2.delivery.wrapper.session_id, session_id);
    }

    #[test]
    fn link_quota_tracker_enforces_sliding_window_expiry_and_fail_closed_table_saturation() {
        let adapter = IrohCarrierAdapterV1::default();
        let auth1 = MessageAuth::new(KeyPair::from_seed([0x41; 32]).unwrap());
        let auth2 = MessageAuth::new(KeyPair::from_seed([0x42; 32]).unwrap());
        let auth3 = MessageAuth::new(KeyPair::from_seed([0x43; 32]).unwrap());

        let p1 = attested_link(&adapter, &auth1, [0x01; 16]);
        let p2 = attested_link(&adapter, &auth2, [0x02; 16]);
        let p3 = attested_link(&adapter, &auth3, [0x03; 16]);

        let mut tracker = LinkFrameQuotaTrackerV1::with_limits(2, 60, 2, 16);

        // 2 frames allowed within the 60s window for p1; 3rd fails until window rolls over.
        tracker.record_frame_at(&p1, 100).unwrap();
        tracker.record_frame_at(&p1, 120).unwrap();
        assert!(matches!(
            tracker.record_frame_at(&p1, 159),
            Err(IrohCarrierError::LinkRateLimited { .. })
        ));
        assert_eq!(tracker.active_window_count(&p1.link_peer_id()), 2);

        // At t = 160, t=100 entry has expired (`160 - 100 >= 60`), freeing one slot.
        tracker.record_frame_at(&p1, 160).unwrap();
        assert_eq!(tracker.active_window_count(&p1.link_peer_id()), 2);

        // Second active principal fills max_tracked_links = 2.
        tracker.record_frame_at(&p2, 170).unwrap();
        assert_eq!(tracker.tracked_link_count(), 2);

        // Third principal at t = 175 must fail closed (`LinkQuotaTableFull`), NOT evict active p1/p2.
        assert!(matches!(
            tracker.record_frame_at(&p3, 175),
            Err(IrohCarrierError::LinkQuotaTableFull {
                max_tracked_links: 2
            })
        ));
        assert_eq!(tracker.tracked_link_count(), 2);

        // Once p1 and p2 entries both expire (`170 + 60 = 230`), lazy pruning reclaims them for p3.
        tracker.record_frame_at(&p3, 230).unwrap();
        assert_eq!(tracker.tracked_link_count(), 1);
        assert_eq!(tracker.active_window_count(&p3.link_peer_id()), 1);
    }

    #[tokio::test]
    async fn live_iroh_1rtt_quic_loopback_enforces_binding_alpn_quota_and_v2_gossip() {
        let relay_seed = [0x11; 32];
        let origin_seed = [0x22; 32];
        let coord_seed = [0x33; 32];

        let origin_kp = KeyPair::from_seed(origin_seed).unwrap();
        let origin_auth = MessageAuth::new(origin_kp.clone());
        let coord_kp = KeyPair::from_seed(coord_seed).unwrap();

        let relay_ep = IrohEndpointBridgeV1::bind_loopback(
            relay_seed,
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::DecoupledRelayAllowed, 65_536),
        )
        .await
        .unwrap();
        let strict_ep = IrohEndpointBridgeV1::bind_loopback(
            coord_seed,
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 65_536),
        )
        .await
        .unwrap();
        let relay_server_ep = IrohEndpointBridgeV1::bind_loopback(
            [0x44; 32],
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::DecoupledRelayAllowed, 65_536),
        )
        .await
        .unwrap();

        let allowlist = CohortAllowlist::from_peers([origin_kp.peer_id()]);
        let scope = [0x11; 16];
        let run_id = [0x22; 16];
        let policy_digest = [0x44; 32];
        let challenge = SessionHandshakeChallengeV2 {
            swarm_instance_id: scope,
            run_id,
            policy_digest,
            receiver_peer: coord_kp.peer_id(),
            sender_peer: origin_kp.peer_id(),
            requested_epoch: 1,
            initial_sequence: 0,
            receiver_nonce: [0x77; 32],
            issued_at_secs: 900,
        };
        let session_id = challenge.derive_session_id().unwrap();
        let hs_sig = origin_auth.sign_session_handshake_v2(&challenge).unwrap();

        let mut strict_replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(allowlist.clone());
        strict_replay
            .register_recovery_challenge_v2(challenge)
            .unwrap();
        strict_replay
            .complete_recovery_handshake_v2(scope, run_id, origin_kp.public_key(), &hs_sig, 900)
            .unwrap();

        let mut env_v2 = MessageEnvelope::new_v2(
            *origin_kp.public_key(),
            scope,
            run_id,
            1,
            session_id,
            1,
            MessageType::GradientUpdate,
            vec![1, 2, 3, 4],
        );
        env_v2.sequence = 1;
        env_v2.timestamp = 1000;
        let env_v2 = env_v2.sign_with(&origin_auth);
        let env_bytes = env_v2.serialize().unwrap();

        // 1. StrictOneToOne over live QUIC 1-RTT MUST reject relay != originator WITHOUT mutating replay_guard.
        let strict_addr = strict_ep.addr();
        let (send_res, recv_res) = tokio::join!(
            relay_ep.send_frame_1rtt(strict_addr, &env_bytes),
            strict_ep.accept_envelope_over_quic([0x5A; 16], &mut strict_replay, 1000, None),
        );
        send_res.unwrap();
        assert!(matches!(
            recv_res,
            Err(IrohCarrierError::Carrier(
                CarrierError::StrictBindingMismatch { .. }
            ))
        ));

        // 2. Mismatched ALPN endpoint dialed over live QUIC MUST fail closed before frame delivery.
        let bad_ep = IrohEndpointBridgeV1::bind_at_with_alpn(
            [0x55; 32],
            SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)),
            RelayMode::Disabled,
            IrohCarrierAdapterV1::default(),
            b"other/wrong-alpn",
        )
        .await
        .unwrap();
        let bad_addr = bad_ep.addr();
        let accept_task = tokio::spawn({
            let ep = bad_ep.endpoint().clone();
            async move {
                if let Some(inc) = ep.accept().await {
                    if let Ok(acc) = inc.accept() {
                        let _ = acc.await;
                    }
                }
            }
        });
        let err = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            relay_ep.send_frame_1rtt(bad_addr, &env_bytes),
        )
        .await
        .expect("ALPN mismatch dial timed out")
        .unwrap_err();
        assert!(matches!(err, IrohCarrierError::Transport(_)));
        bad_ep.close().await;
        let _ = accept_task.await;

        // 3. Relayed v2 gossip over live 1-RTT QUIC succeeds & records per-link quota!
        let inner_v2 = InnerUpdateV2 {
            originator_public_key: *origin_kp.public_key(),
            swarm_instance_id: scope,
            run_id,
            round_id: 9,
            contribution_id: 1,
            base_checkpoint_digest: [0x11; 32],
            context_policy_digest: policy_digest,
            payload_kind: PayloadKindV1::ModelDeltaV1,
            compression_schema_version: 0,
            num_elements: 2,
            reported_examples: 8,
            encoded_payload: vec![1, 2, 3, 4],
            originator_signature: Vec::new(),
        };
        let gossip_v2 = BundledGossipDeliveryV2::new_signed(
            &origin_auth,
            inner_v2,
            coord_kp.peer_id(),
            session_id,
            1,
            1,
            1000,
        );
        let gossip_bytes = gossip_v2.to_bytes().unwrap();

        let mut relay_replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(allowlist);
        relay_replay
            .register_recovery_challenge_v2(challenge)
            .unwrap();
        relay_replay
            .complete_recovery_handshake_v2(scope, run_id, origin_kp.public_key(), &hs_sig, 900)
            .unwrap();
        let mut quota = LinkFrameQuotaTrackerV1::with_limits(1, 60, 8, 16);

        let srv_addr = relay_server_ep.addr();
        let (send_g, recv_g) = tokio::join!(
            relay_ep.send_frame_1rtt(srv_addr.clone(), &gossip_bytes),
            relay_server_ep.accept_gossip_v2_over_quic(
                [0x6B; 16],
                coord_kp.peer_id(),
                &mut relay_replay,
                1000,
                Some(&mut quota),
            ),
        );
        send_g.unwrap();
        let accepted = recv_g.unwrap();
        assert!(accepted.binding().relayed_via_intermediary());
        assert_eq!(
            accepted.binding().link_principal().link_peer_id(),
            relay_ep.link_peer_id()
        );
        assert_eq!(accepted.binding().originator_peer_id(), origin_kp.peer_id());
        assert_eq!(accepted.delivery.wrapper.session_id, session_id);
        assert_eq!(quota.active_window_count(&relay_ep.link_peer_id()), 1);

        // 4. Quota limit (1 frame/min) rejects 2nd live frame pre-replay with LinkRateLimited!
        let (send_q, recv_q) = tokio::join!(
            relay_ep.send_frame_1rtt(srv_addr, &gossip_bytes),
            relay_server_ep.accept_gossip_v2_over_quic(
                [0x7C; 16],
                coord_kp.peer_id(),
                &mut relay_replay,
                1001,
                Some(&mut quota),
            ),
        );
        send_q.unwrap();
        assert!(matches!(
            recv_q,
            Err(IrohCarrierError::LinkRateLimited { .. })
        ));

        relay_ep.close().await;
        strict_ep.close().await;
        relay_server_ep.close().await;
    }

    #[tokio::test]
    async fn live_iroh_1rtt_direct_strict_delivery_and_tamper_rejection() {
        let origin_seed = [0x71; 32];
        let server_seed = [0x72; 32];
        let origin_kp = KeyPair::from_seed(origin_seed).unwrap();
        let origin_auth = MessageAuth::new(origin_kp.clone());
        let server_kp = KeyPair::from_seed(server_seed).unwrap();

        let client_ep = IrohEndpointBridgeV1::bind_loopback(
            origin_seed,
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 65_536),
        )
        .await
        .unwrap();
        let server_ep = IrohEndpointBridgeV1::bind_loopback(
            server_seed,
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 65_536),
        )
        .await
        .unwrap();

        let allowlist = CohortAllowlist::from_peers([origin_kp.peer_id()]);
        let scope = [0x11; 16];
        let run_id = [0x22; 16];
        let policy_digest = [0x55; 32];
        let challenge = SessionHandshakeChallengeV2 {
            swarm_instance_id: scope,
            run_id,
            policy_digest,
            receiver_peer: server_kp.peer_id(),
            sender_peer: origin_kp.peer_id(),
            requested_epoch: 1,
            initial_sequence: 0,
            receiver_nonce: [0x88; 32],
            issued_at_secs: 900,
        };
        let session_id = challenge.derive_session_id().unwrap();
        let hs_sig = origin_auth.sign_session_handshake_v2(&challenge).unwrap();
        let mut replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(allowlist);
        replay.register_recovery_challenge_v2(challenge).unwrap();
        replay
            .complete_recovery_handshake_v2(scope, run_id, origin_kp.public_key(), &hs_sig, 900)
            .unwrap();

        let mut env = MessageEnvelope::new_v2(
            *origin_kp.public_key(),
            scope,
            run_id,
            1,
            session_id,
            1,
            MessageType::GradientUpdate,
            vec![9, 8, 7],
        );
        env.sequence = 1;
        env.timestamp = 1000;
        let env = env.sign_with(&origin_auth);
        let valid_bytes = env.serialize().unwrap();
        let mut tampered_bytes = valid_bytes.clone();
        let last = tampered_bytes.len() - 1;
        tampered_bytes[last] ^= 0xFF;

        let addr = server_ep.addr();
        // 1. Tampered payload fails signature verification without advancing seq=1!
        let (s_bad, r_bad) = tokio::join!(
            client_ep.send_frame_1rtt(addr.clone(), &tampered_bytes),
            server_ep.accept_envelope_over_quic([0x81; 16], &mut replay, 1000, None),
        );
        s_bad.unwrap();
        assert!(matches!(r_bad, Err(IrohCarrierError::Envelope(_))));

        // 2. Genuine seq=1 still accepted directly over StrictOneToOne 1-RTT QUIC!
        let (s_ok, r_ok) = tokio::join!(
            client_ep.send_frame_1rtt(addr.clone(), &valid_bytes),
            server_ep.accept_envelope_over_quic([0x82; 16], &mut replay, 1000, None),
        );
        s_ok.unwrap();
        let accepted = r_ok.unwrap();
        assert!(!accepted.binding().relayed_via_intermediary());
        assert_eq!(accepted.binding().originator_peer_id(), origin_kp.peer_id());
        assert_eq!(accepted.envelope.payload, vec![9, 8, 7]);

        // 3. After idle window expiry (`t = 1200 > 1000 + 120`) and record reclamation,
        //    replaying seq=1 with a fresh timestamp over live 1-RTT QUIC MUST fail closed (`AR-14-05`, `14C-1`),
        //    while advancing to seq=2 succeeds!
        assert_eq!(replay.reclaim_expired(1200), 1);
        let mut replay_after_idle = MessageEnvelope::new_v2(
            *origin_kp.public_key(),
            scope,
            run_id,
            1,
            session_id,
            1,
            MessageType::GradientUpdate,
            vec![9, 8, 7],
        );
        replay_after_idle.sequence = 1;
        replay_after_idle.timestamp = 1200;
        let replay_bytes = replay_after_idle
            .sign_with(&origin_auth)
            .serialize()
            .unwrap();
        let (s_rep, r_rep) = tokio::join!(
            client_ep.send_frame_1rtt(addr.clone(), &replay_bytes),
            server_ep.accept_envelope_over_quic([0x83; 16], &mut replay, 1200, None),
        );
        s_rep.unwrap();
        assert!(matches!(r_rep, Err(IrohCarrierError::Envelope(_))));

        let mut next_env = MessageEnvelope::new_v2(
            *origin_kp.public_key(),
            scope,
            run_id,
            1,
            session_id,
            1,
            MessageType::GradientUpdate,
            vec![4, 5, 6],
        );
        next_env.sequence = 2;
        next_env.timestamp = 1200;
        let next_bytes = next_env.sign_with(&origin_auth).serialize().unwrap();
        let (s_next, r_next) = tokio::join!(
            client_ep.send_frame_1rtt(addr, &next_bytes),
            server_ep.accept_envelope_over_quic([0x84; 16], &mut replay, 1200, None),
        );
        s_next.unwrap();
        assert_eq!(r_next.unwrap().envelope.sequence, 2);

        client_ep.close().await;
        server_ep.close().await;
    }

    #[tokio::test]
    async fn iroh_accept_envelope_over_quic_rejects_legacy_v0_and_v1_by_default_without_replay_mutation(
    ) {
        let origin_seed = [0x81; 32];
        let server_seed = [0x82; 32];
        let origin_kp = KeyPair::from_seed(origin_seed).unwrap();
        let origin_auth = MessageAuth::new(origin_kp.clone());
        let scope = [0x61; 16];

        let allowlist = CohortAllowlist::from_peers([origin_kp.peer_id()]);
        let mut replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(allowlist);

        // Legacy v0 rejected at frame level under RequireV2 without replay mutation.
        let env_v0 = MessageEnvelope::new_legacy_v0(
            *origin_kp.public_key(),
            MessageType::Heartbeat,
            vec![1],
        )
        .with_sequence(1)
        .with_timestamp(1000)
        .sign_with(&origin_auth);
        let adapter = IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 65_536);
        let origin_link = attested_link(&adapter, &origin_auth, [0x19; 16]);
        let v0_frame = adapter
            .encode_wire_frame(&env_v0.serialize().unwrap())
            .unwrap();
        assert!(matches!(
            adapter.accept_envelope_frame(origin_link, &v0_frame, &mut replay, 1000, None),
            Err(IrohCarrierError::Envelope(
                VerifyError::UnsupportedVersion { major: 0, minor: 1 }
            ))
        ));
        assert_eq!(replay.cache_size(), 0);

        // Live 1-RTT QUIC: default accept_envelope_over_quic rejects v1 without replay mutation,
        // while accept_envelope_over_quic_with_policy(..., AllowLegacyV1AndV2) admits it!
        let server_ep = IrohEndpointBridgeV1::bind_loopback(server_seed, adapter)
            .await
            .unwrap();
        let client_ep =
            IrohEndpointBridgeV1::bind_loopback(origin_seed, IrohCarrierAdapterV1::default())
                .await
                .unwrap();

        let env_v1 = MessageEnvelope::new_v1(
            *origin_kp.public_key(),
            scope,
            1,
            3,
            MessageType::GradientUpdate,
            vec![7, 8, 9],
        )
        .with_sequence(1)
        .with_timestamp(1000)
        .sign_with(&origin_auth);
        let v1_bytes = env_v1.serialize().unwrap();
        let addr = server_ep.addr();

        let (s1, r1) = tokio::join!(
            client_ep.send_frame_1rtt(addr.clone(), &v1_bytes),
            server_ep.accept_envelope_over_quic([0x91; 16], &mut replay, 1000, None),
        );
        s1.unwrap();
        assert!(matches!(
            r1,
            Err(IrohCarrierError::Envelope(
                VerifyError::UnsupportedVersion { major: 0, minor: 2 }
            ))
        ));
        assert_eq!(replay.cache_size(), 0);

        let (s2, r2) = tokio::join!(
            client_ep.send_frame_1rtt(addr, &v1_bytes),
            server_ep.accept_envelope_over_quic_with_policy(
                [0x92; 16],
                &mut replay,
                1000,
                None,
                EnvelopeVersionPolicyV1::AllowLegacyV1AndV2,
            ),
        );
        s2.unwrap();
        let accepted = r2.unwrap();
        assert_eq!(accepted.envelope.payload, vec![7, 8, 9]);
        assert_eq!(replay.cache_size(), 1);

        client_ep.close().await;
        server_ep.close().await;
    }

    #[tokio::test]
    async fn iroh_accept_and_slowloris_stream_enforce_bounded_timeouts() {
        let server_ep = IrohEndpointBridgeV1::bind_loopback(
            [0xA1; 32],
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 65_536),
        )
        .await
        .unwrap()
        .with_timeouts(IrohAcceptTimeoutsV1 {
            handshake_timeout: Duration::from_millis(80),
            read_timeout: Duration::from_millis(80),
        });

        let start = std::time::Instant::now();
        let idle_res = server_ep.accept_1rtt_frame_fresh_nonce(1000, None).await;
        assert!(matches!(
            idle_res,
            Err(IrohCarrierError::Timeout {
                stage: "endpoint_accept"
            })
        ));
        assert!(start.elapsed() < Duration::from_secs(2));

        let client_ep = IrohEndpointBridgeV1::bind_loopback(
            [0xA2; 32],
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 65_536),
        )
        .await
        .unwrap();
        let addr = server_ep.addr();

        // Slowloris 1: send STQ1 + read nonce, then stall without sending attestation header.
        let slow_header_client = async {
            let conn = client_ep
                .endpoint
                .connect(addr.clone(), IrohCarrierAdapterV1::expected_alpn())
                .await
                .unwrap();
            let (mut send, mut recv) = conn.open_bi().await.unwrap();
            send.write_all(IrohEndpointBridgeV1::STREAM_HELLO_V1)
                .await
                .unwrap();
            let mut nonce = [0u8; 16];
            recv.read_exact(&mut nonce).await.unwrap();
            // Hold stream open until server's read_timeout closes the QUIC connection.
            let _ = tokio::time::timeout(Duration::from_secs(2), conn.closed()).await;
            drop(send);
        };
        let ((), srv_header_timeout) = tokio::join!(
            slow_header_client,
            server_ep.accept_1rtt_frame_fresh_nonce(1001, None),
        );
        assert!(matches!(
            srv_header_timeout,
            Err(IrohCarrierError::Timeout {
                stage: "read_attestation_header"
            })
        ));

        // Slowloris 2: send valid 96-byte attestation header, then stall before finishing body.
        let slow_body_client = async {
            let conn = client_ep
                .endpoint
                .connect(addr, IrohCarrierAdapterV1::expected_alpn())
                .await
                .unwrap();
            let (mut send, mut recv) = conn.open_bi().await.unwrap();
            send.write_all(IrohEndpointBridgeV1::STREAM_HELLO_V1)
                .await
                .unwrap();
            let mut nonce = [0u8; 16];
            recv.read_exact(&mut nonce).await.unwrap();
            let att = sign_link_attestation(
                &client_ep.link_auth,
                IrohCarrierAdapterV1::expected_alpn(),
                nonce,
            )
            .unwrap();
            let mut header = [0u8; LINK_ATTESTATION_HEADER_BYTES];
            header[..32].copy_from_slice(&att.endpoint_public_key);
            header[32..].copy_from_slice(att.signature.as_bytes());
            send.write_all(&header).await.unwrap();
            // Partial body stall: server must time out cleanly and close connection.
            send.write_all(&[0u8; 2]).await.unwrap();
            let _ = tokio::time::timeout(Duration::from_secs(2), conn.closed()).await;
            drop(send);
        };
        let ((), srv_body_timeout) = tokio::join!(
            slow_body_client,
            server_ep.accept_1rtt_frame_fresh_nonce(1002, None),
        );
        assert!(matches!(
            srv_body_timeout,
            Err(IrohCarrierError::Timeout {
                stage: "read_frame_body"
            })
        ));

        client_ep.close().await;
        server_ep.close().await;
    }

    #[tokio::test]
    async fn iroh_pre_body_quota_enforced_before_oversized_body_read() {
        let origin_seed = [0xA3; 32];
        let origin_kp = KeyPair::from_seed(origin_seed).unwrap();
        let origin_auth = MessageAuth::new(origin_kp.clone());
        let receiver_kp = KeyPair::from_seed([0xA4; 32]).unwrap();

        let client_ep = IrohEndpointBridgeV1::bind_loopback(
            origin_seed,
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 65_536),
        )
        .await
        .unwrap();
        let server_ep = IrohEndpointBridgeV1::bind_loopback(
            [0xA4; 32],
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 256),
        )
        .await
        .unwrap();
        let addr = server_ep.addr();

        let scope = [0x11; 16];
        let run_id = [0x22; 16];
        let challenge = SessionHandshakeChallengeV2 {
            swarm_instance_id: scope,
            run_id,
            policy_digest: [0x55; 32],
            receiver_peer: receiver_kp.peer_id(),
            sender_peer: origin_kp.peer_id(),
            requested_epoch: 1,
            initial_sequence: 0,
            receiver_nonce: [0x91; 32],
            issued_at_secs: 900,
        };
        let session_id = challenge.derive_session_id().unwrap();
        let hs_sig = origin_auth.sign_session_handshake_v2(&challenge).unwrap();
        let mut env = MessageEnvelope::new_v2(
            *origin_kp.public_key(),
            scope,
            run_id,
            1,
            session_id,
            1,
            MessageType::GradientUpdate,
            vec![1, 2, 3],
        );
        env.sequence = 1;
        env.timestamp = 1000;
        let frame_bytes = env.sign_with(&origin_auth).serialize().unwrap();

        let mut replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(CohortAllowlist::from_peers([origin_kp.peer_id()]));
        replay.register_recovery_challenge_v2(challenge).unwrap();
        replay
            .complete_recovery_handshake_v2(scope, run_id, origin_kp.public_key(), &hs_sig, 900)
            .unwrap();
        let mut quota = LinkFrameQuotaTrackerV1::with_limits(1, 60, 8, 16);

        // Consume the single allowed frame in the quota window.
        let (s1, r1) = tokio::join!(
            client_ep.send_frame_1rtt(addr.clone(), &frame_bytes),
            server_ep.accept_envelope_over_quic_fresh_nonce(&mut replay, 1000, Some(&mut quota)),
        );
        s1.unwrap();
        r1.unwrap();

        // Helper sending an attested but oversized raw body (> 256 bytes).
        let send_oversized_body = || async {
            let conn = client_ep
                .endpoint
                .connect(addr.clone(), IrohCarrierAdapterV1::expected_alpn())
                .await
                .unwrap();
            let (mut send, mut recv) = conn.open_bi().await.unwrap();
            send.write_all(IrohEndpointBridgeV1::STREAM_HELLO_V1)
                .await
                .unwrap();
            let mut nonce = [0u8; 16];
            recv.read_exact(&mut nonce).await.unwrap();
            let att = sign_link_attestation(
                &client_ep.link_auth,
                IrohCarrierAdapterV1::expected_alpn(),
                nonce,
            )
            .unwrap();
            let oversized = vec![0x5Au8; 512];
            let mut buf = Vec::with_capacity(LINK_ATTESTATION_HEADER_BYTES + 4 + oversized.len());
            buf.extend_from_slice(&att.endpoint_public_key);
            buf.extend_from_slice(att.signature.as_bytes());
            buf.extend_from_slice(&(oversized.len() as u32).to_le_bytes());
            buf.extend_from_slice(&oversized);
            let _ = send.write_all(&buf).await;
            let _ = send.finish();
            let _ = tokio::time::timeout(Duration::from_secs(2), conn.closed()).await;
        };

        // Pre-body quota must fail BEFORE body read/size check returns FrameTooLarge.
        let ((), r2) = tokio::join!(
            send_oversized_body(),
            server_ep.accept_1rtt_frame_fresh_nonce(1001, Some(&mut quota)),
        );
        assert!(matches!(r2, Err(IrohCarrierError::LinkRateLimited { .. })));

        // Without quota throttling, oversized body triggers bounded FrameTooLarge.
        let ((), r3) = tokio::join!(
            send_oversized_body(),
            server_ep.accept_1rtt_frame_fresh_nonce(1002, None),
        );
        assert!(matches!(
            r3,
            Err(IrohCarrierError::Carrier(CarrierError::FrameTooLarge {
                max: 256,
                ..
            }))
        ));

        client_ep.close().await;
        server_ep.close().await;
    }

    #[tokio::test]
    async fn iroh_fresh_csprng_nonces_and_static_nonce_reuse_rejected() {
        let origin_seed = [0xA5; 32];
        let origin_kp = KeyPair::from_seed(origin_seed).unwrap();
        let origin_auth = MessageAuth::new(origin_kp.clone());
        let receiver_kp = KeyPair::from_seed([0xA6; 32]).unwrap();

        let client_ep = IrohEndpointBridgeV1::bind_loopback(
            origin_seed,
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 65_536),
        )
        .await
        .unwrap();
        let server_ep = IrohEndpointBridgeV1::bind_loopback(
            [0xA6; 32],
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 65_536),
        )
        .await
        .unwrap()
        .with_timeouts(IrohAcceptTimeoutsV1 {
            handshake_timeout: Duration::from_millis(2000),
            read_timeout: Duration::from_millis(2000),
        });
        let addr = server_ep.addr();
        let scope = [0x44; 16];
        let run_id = [0x55; 16];
        let challenge = SessionHandshakeChallengeV2 {
            swarm_instance_id: scope,
            run_id,
            policy_digest: [0x66; 32],
            receiver_peer: receiver_kp.peer_id(),
            sender_peer: origin_kp.peer_id(),
            requested_epoch: 1,
            initial_sequence: 0,
            receiver_nonce: [0x92; 32],
            issued_at_secs: 900,
        };
        let session_id = challenge.derive_session_id().unwrap();
        let hs_sig = origin_auth.sign_session_handshake_v2(&challenge).unwrap();
        let mut replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_cohort_allowlist(CohortAllowlist::from_peers([origin_kp.peer_id()]));
        replay.register_recovery_challenge_v2(challenge).unwrap();
        replay
            .complete_recovery_handshake_v2(scope, run_id, origin_kp.public_key(), &hs_sig, 900)
            .unwrap();

        for seq in 1..=2u64 {
            let mut env = MessageEnvelope::new_v2(
                *origin_kp.public_key(),
                scope,
                run_id,
                1,
                session_id,
                1,
                MessageType::GradientUpdate,
                vec![seq as u8],
            );
            env.sequence = seq;
            env.timestamp = 1000 + seq as u32;
            let bytes = env.sign_with(&origin_auth).serialize().unwrap();
            let (s, r) = tokio::join!(
                client_ep.send_frame_1rtt(addr.clone(), &bytes),
                server_ep.accept_envelope_over_quic_fresh_nonce(
                    &mut replay,
                    1000 + seq as u32,
                    None
                ),
            );
            s.unwrap();
            let accepted = r.unwrap();
            assert_eq!(accepted.envelope.payload, vec![seq as u8]);
        }

        // Static non-zero nonce succeeds once and is rejected immediately on reuse.
        let mut env3 = MessageEnvelope::new_v2(
            *origin_kp.public_key(),
            scope,
            run_id,
            1,
            session_id,
            1,
            MessageType::GradientUpdate,
            vec![3],
        );
        env3.sequence = 3;
        env3.timestamp = 1003;
        let bytes3 = env3.sign_with(&origin_auth).serialize().unwrap();
        let static_nonce = [0xA7; 16];
        let (s3, r3) = tokio::join!(
            client_ep.send_frame_1rtt(addr, &bytes3),
            server_ep.accept_1rtt_frame(static_nonce),
        );
        s3.unwrap();
        r3.unwrap();

        let reuse_res = server_ep.accept_1rtt_frame(static_nonce).await;
        assert!(matches!(
            reuse_res,
            Err(IrohCarrierError::Carrier(
                CarrierError::InvalidLinkAttestation
            ))
        ));

        let zero_res = server_ep.accept_1rtt_frame([0u8; 16]).await;
        assert!(matches!(
            zero_res,
            Err(IrohCarrierError::Carrier(
                CarrierError::InvalidLinkAttestation
            ))
        ));

        client_ep.close().await;
        server_ep.close().await;
    }

    // -------------------------------------------------------------------------------------------
    // Independent clean-context T-Agent adversarial regressions (`15D-2`).
    // -------------------------------------------------------------------------------------------

    /// Attestation header variants a scripted raw peer sends after reading the receiver challenge.
    #[derive(Debug, Clone, Copy)]
    enum TAgentHeaderV1 {
        /// Valid attestation signed by the TLS-bound link key.
        Valid,
        /// Valid signature by a key that differs from the TLS `remote_id`.
        ForeignKey,
        /// Correct TLS-bound public key with a corrupted signature.
        CorruptSig,
    }

    /// Scripted raw-peer behaviours driven directly through an `iroh` endpoint.
    #[derive(Debug, Clone, Copy)]
    enum TAgentPeerScriptV1<'a> {
        /// Completes the 1-RTT handshake but never opens a bi-stream.
        ConnectOnly,
        /// Opens a bi-stream and writes only 2 of the 4 stream-hello bytes.
        PartialHello,
        /// Writes a wrong 4-byte stream hello.
        BadHello,
        /// Reads the challenge nonce, then never sends the attestation header.
        StallAfterNonce,
        /// Reads the challenge nonce, writes 50 of the 96 header bytes, then FIN.
        TruncatedHeader,
        /// Reads the challenge, writes a 96-byte header (`kind`), then `body`, then optional FIN.
        Header {
            kind: TAgentHeaderV1,
            body: &'a [u8],
            finish: bool,
        },
    }

    fn t_agent_app_close_code(reason: &Option<iroh::endpoint::ConnectionError>) -> Option<u64> {
        match reason {
            Some(iroh::endpoint::ConnectionError::ApplicationClosed(close)) => {
                Some(close.error_code.into_inner())
            }
            _ => None,
        }
    }

    fn t_agent_is_invalid_attestation<T>(res: &Result<T, IrohCarrierError>) -> bool {
        matches!(
            res,
            Err(IrohCarrierError::Carrier(
                CarrierError::InvalidLinkAttestation
            ))
        )
    }

    async fn t_agent_timed<F: std::future::Future>(fut: F) -> (F::Output, Duration) {
        let start = std::time::Instant::now();
        let out = fut.await;
        (out, start.elapsed())
    }

    /// Drive `script` against `addr` and return the close reason this peer observed within `hold`
    /// (`None` if the server never closed the connection in time).
    async fn t_agent_raw_peer(
        client: &IrohEndpointBridgeV1,
        addr: EndpointAddr,
        script: TAgentPeerScriptV1<'_>,
        hold: Duration,
    ) -> Option<iroh::endpoint::ConnectionError> {
        let alpn = IrohCarrierAdapterV1::expected_alpn();
        let conn = client.endpoint.connect(addr, alpn).await.ok()?;
        if matches!(script, TAgentPeerScriptV1::ConnectOnly) {
            return tokio::time::timeout(hold, conn.closed()).await.ok();
        }
        let (mut send, mut recv) = conn.open_bi().await.ok()?;
        let hello = IrohEndpointBridgeV1::STREAM_HELLO_V1;
        match script {
            TAgentPeerScriptV1::ConnectOnly => {}
            TAgentPeerScriptV1::PartialHello => {
                let _ = send.write_all(&hello[..2]).await;
            }
            TAgentPeerScriptV1::BadHello => {
                let _ = send.write_all(b"XXXX").await;
            }
            TAgentPeerScriptV1::StallAfterNonce | TAgentPeerScriptV1::TruncatedHeader => {
                send.write_all(hello).await.ok()?;
                let mut nonce = [0u8; 16];
                recv.read_exact(&mut nonce).await.ok()?;
                if matches!(script, TAgentPeerScriptV1::TruncatedHeader) {
                    let _ = send.write_all(&[0x33; 50]).await;
                    let _ = send.finish();
                }
            }
            TAgentPeerScriptV1::Header { kind, body, finish } => {
                send.write_all(hello).await.ok()?;
                let mut nonce = [0u8; 16];
                recv.read_exact(&mut nonce).await.ok()?;
                let foreign = MessageAuth::new(KeyPair::from_seed([0xEE; 32]).unwrap());
                let signer = match kind {
                    TAgentHeaderV1::ForeignKey => &foreign,
                    TAgentHeaderV1::Valid | TAgentHeaderV1::CorruptSig => &client.link_auth,
                };
                let att = sign_link_attestation(signer, alpn, nonce).unwrap();
                let mut header = [0u8; LINK_ATTESTATION_HEADER_BYTES];
                header[..32].copy_from_slice(&att.endpoint_public_key);
                header[32..].copy_from_slice(att.signature.as_bytes());
                if matches!(kind, TAgentHeaderV1::CorruptSig) {
                    header[40] ^= 0xFF;
                }
                let _ = send.write_all(&header).await;
                if !body.is_empty() {
                    let _ = send.write_all(body).await;
                }
                if finish {
                    let _ = send.finish();
                }
            }
        }
        // Keep both halves alive: dropping `SendStream` implicitly finishes it and dropping
        // `RecvStream` implicitly sends STOP_SENDING.
        let reason = tokio::time::timeout(hold, conn.closed()).await.ok();
        drop(send);
        drop(recv);
        reason
    }

    async fn t_agent_bind_raw_endpoint(
        seed: [u8; 32],
        stream_receive_window: Option<u32>,
    ) -> Endpoint {
        let mut builder = Endpoint::builder(presets::Minimal)
            .secret_key(SecretKey::from_bytes(&seed))
            .relay_mode(RelayMode::Disabled)
            .clear_address_lookup()
            .clear_addr_filter();
        if let Some(window) = stream_receive_window {
            builder = builder.transport_config(
                iroh::endpoint::QuicTransportConfig::builder()
                    .stream_receive_window(iroh::endpoint::VarInt::from_u32(window))
                    .build(),
            );
        }
        builder
            .bind_addr(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0)))
            .unwrap()
            .bind()
            .await
            .unwrap()
    }

    /// Peer that never reads the receiver challenge (it pre-signs a caller-supplied static nonce),
    /// so its QUIC stack never extends `MAX_STREAM_DATA` past its initial `stream_receive_window`.
    async fn t_agent_presigned_non_reading_peer(
        ep: &Endpoint,
        addr: EndpointAddr,
        auth: &MessageAuth,
        nonce: [u8; 16],
        wire_frame: &[u8],
        hold: Duration,
    ) -> Option<iroh::endpoint::ConnectionError> {
        let alpn = IrohCarrierAdapterV1::expected_alpn();
        let conn = ep.connect(addr, alpn).await.ok()?;
        let (mut send, recv_never_read) = conn.open_bi().await.ok()?;
        let att = sign_link_attestation(auth, alpn, nonce).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(IrohEndpointBridgeV1::STREAM_HELLO_V1);
        out.extend_from_slice(&att.endpoint_public_key);
        out.extend_from_slice(att.signature.as_bytes());
        out.extend_from_slice(wire_frame);
        send.write_all(&out).await.ok()?;
        send.finish().ok()?;
        let reason = tokio::time::timeout(hold, conn.closed()).await.ok();
        drop(recv_never_read);
        drop(send);
        reason
    }

    /// Invariant 1: every receiver-side stream write/read in `accept_1rtt_frame_inner` must be
    /// bounded. A peer that grants exactly 16 bytes of stream credit (the challenge nonce) and
    /// never extends it must not be able to wedge the accept call on the post-body ack write.
    #[tokio::test]
    async fn t_agent_post_body_ack_write_is_bounded_against_credit_starving_peer() {
        let handshake_timeout = Duration::from_secs(1);
        let read_timeout = Duration::from_millis(250);
        // Strictly larger than the sum of every bounded stage (accept + 1-RTT + 7 stream stages
        // incl. the ack), so exceeding it can only mean an await with no timeout at all.
        let outer_deadline = handshake_timeout * 2 + read_timeout * 8 + Duration::from_secs(1);
        let hold = outer_deadline + Duration::from_millis(500);
        let server_ep = IrohEndpointBridgeV1::bind_loopback(
            [0xB1; 32],
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 1024),
        )
        .await
        .unwrap()
        .with_timeouts(IrohAcceptTimeoutsV1 {
            handshake_timeout,
            read_timeout,
        });
        let addr = server_ep.addr();
        let wire_frame = server_ep.adapter().encode_wire_frame(&[0x42; 8]).unwrap();

        // Control: identical pre-signed, non-reading peer with default stream credit completes.
        let control_seed = [0xB2; 32];
        let control_auth = MessageAuth::new(KeyPair::from_seed(control_seed).unwrap());
        let control_ep = t_agent_bind_raw_endpoint(control_seed, None).await;
        let (_, control) = tokio::join!(
            t_agent_presigned_non_reading_peer(
                &control_ep,
                addr.clone(),
                &control_auth,
                [0xB3; 16],
                &wire_frame,
                hold,
            ),
            tokio::time::timeout(
                outer_deadline,
                server_ep.accept_1rtt_frame_authenticated([0xB3; 16], 1000, None),
            ),
        );
        let (principal, frame) = control
            .expect("control accept must not hang")
            .expect("control accept must succeed");
        assert_eq!(
            principal.link_peer_id(),
            KeyPair::from_seed(control_seed).unwrap().peer_id()
        );
        assert_eq!(frame, wire_frame);
        control_ep.close().await;

        // Attack: peer advertises a 16-byte stream receive window and never reads it.
        let attack_seed = [0xB5; 32];
        let attack_auth = MessageAuth::new(KeyPair::from_seed(attack_seed).unwrap());
        let starving_ep = t_agent_bind_raw_endpoint(attack_seed, Some(16)).await;
        let (_, attack) = tokio::join!(
            t_agent_presigned_non_reading_peer(
                &starving_ep,
                addr,
                &attack_auth,
                [0xB4; 16],
                &wire_frame,
                hold,
            ),
            tokio::time::timeout(
                outer_deadline,
                server_ep.accept_1rtt_frame_authenticated([0xB4; 16], 1000, None),
            ),
        );
        starving_ep.close().await;
        server_ep.close().await;
        assert!(
            attack.is_ok(),
            "accept_1rtt_frame_authenticated did not return within {outer_deadline:?} \
             (read_timeout = {read_timeout:?}): the post-body `send.write_all(&[0x01])` ack \
             is not bounded by `IrohAcceptTimeoutsV1`"
        );
    }

    /// Invariant 2: a quota-exhausted or unattested peer's body is never awaited; quota is charged
    /// only after attestation, exactly once, and BEFORE the bounded body read.
    #[tokio::test]
    async fn t_agent_unadmitted_peer_body_never_awaited_and_quota_charged_only_after_attestation() {
        let read_timeout = Duration::from_millis(1500);
        let server_ep = IrohEndpointBridgeV1::bind_loopback(
            [0xC1; 32],
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 256),
        )
        .await
        .unwrap()
        .with_timeouts(IrohAcceptTimeoutsV1 {
            handshake_timeout: Duration::from_secs(3),
            read_timeout,
        });
        let client_ep =
            IrohEndpointBridgeV1::bind_loopback([0xC2; 32], IrohCarrierAdapterV1::default())
                .await
                .unwrap();
        let addr = server_ep.addr();
        let link = client_ep.link_peer_id();
        let hold = Duration::from_secs(4);
        let stall = |kind: TAgentHeaderV1| TAgentPeerScriptV1::Header {
            kind,
            body: &[],
            finish: false,
        };

        // (a) Validly attested but quota-exhausted peer that never sends a body nor FIN.
        let principal = attested_link(
            &IrohCarrierAdapterV1::default(),
            &client_ep.link_auth,
            [0x01; 16],
        );
        let mut exhausted = LinkFrameQuotaTrackerV1::with_limits(1, 60, 8, 16);
        exhausted.record_frame_at(&principal, 1000).unwrap();
        let (close_a, (res_a, elapsed_a)) = tokio::join!(
            t_agent_raw_peer(&client_ep, addr.clone(), stall(TAgentHeaderV1::Valid), hold),
            t_agent_timed(server_ep.accept_1rtt_frame_fresh_nonce(1001, Some(&mut exhausted))),
        );
        assert!(
            matches!(res_a, Err(IrohCarrierError::LinkRateLimited { .. })),
            "{res_a:?}"
        );
        assert!(
            elapsed_a < read_timeout,
            "quota rejection waited on the body: {elapsed_a:?}"
        );
        assert_eq!(exhausted.active_window_count(&link), 1);
        assert_eq!(t_agent_app_close_code(&close_a), Some(6));

        // (b)/(c) Link-key mismatch and corrupted signature: rejected before quota and body.
        let mut open = LinkFrameQuotaTrackerV1::with_limits(4, 60, 8, 16);
        for kind in [TAgentHeaderV1::ForeignKey, TAgentHeaderV1::CorruptSig] {
            let (close, (res, elapsed)) = tokio::join!(
                t_agent_raw_peer(&client_ep, addr.clone(), stall(kind), hold),
                t_agent_timed(server_ep.accept_1rtt_frame_fresh_nonce(1002, Some(&mut open))),
            );
            assert!(t_agent_is_invalid_attestation(&res), "{kind:?}: {res:?}");
            assert!(
                elapsed < read_timeout,
                "{kind:?} rejection waited on the body: {elapsed:?}"
            );
            assert_eq!(
                open.tracked_link_count(),
                0,
                "{kind:?} charged quota before attestation"
            );
            assert_eq!(t_agent_app_close_code(&close), Some(5), "{kind:?}");
        }

        // (d) Neighbouring valid case: admitted peer is charged once pre-body, then its stalled
        //     body read is bounded by `read_timeout` and the connection is closed (code 7).
        let (close_d, (res_d, elapsed_d)) = tokio::join!(
            t_agent_raw_peer(&client_ep, addr, stall(TAgentHeaderV1::Valid), hold),
            t_agent_timed(server_ep.accept_1rtt_frame_fresh_nonce(1003, Some(&mut open))),
        );
        assert!(
            matches!(
                res_d,
                Err(IrohCarrierError::Timeout {
                    stage: "read_frame_body"
                })
            ),
            "{res_d:?}"
        );
        assert!(
            elapsed_d >= read_timeout && elapsed_d < read_timeout * 3,
            "{elapsed_d:?}"
        );
        assert_eq!(open.active_window_count(&link), 1);
        assert_eq!(t_agent_app_close_code(&close_d), Some(7));

        client_ep.close().await;
        server_ep.close().await;
    }

    /// Invariant 1: each receiver stage is individually bounded and closes the connection with a
    /// stage-specific application code on timeout or malformed preamble.
    #[tokio::test]
    async fn t_agent_receiver_stage_timeouts_and_malformed_preambles_close_with_stage_codes() {
        let read_timeout = Duration::from_millis(200);
        let server_ep = IrohEndpointBridgeV1::bind_loopback(
            [0xC5; 32],
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 256),
        )
        .await
        .unwrap()
        .with_timeouts(IrohAcceptTimeoutsV1 {
            handshake_timeout: Duration::from_secs(2),
            read_timeout,
        });
        let client_ep =
            IrohEndpointBridgeV1::bind_loopback([0xC6; 32], IrohCarrierAdapterV1::default())
                .await
                .unwrap();
        let addr = server_ep.addr();
        let hold = Duration::from_secs(3);
        let partial_body = [0x01u8, 0x00];
        // (script, expected timeout stage or `None` for `EmptyOrTruncatedFrame`, close code)
        let cases: [(TAgentPeerScriptV1<'_>, Option<&str>, u64); 6] = [
            (TAgentPeerScriptV1::ConnectOnly, Some("accept_bi"), 2),
            (TAgentPeerScriptV1::PartialHello, Some("read_hello"), 3),
            (TAgentPeerScriptV1::BadHello, None, 3),
            (
                TAgentPeerScriptV1::StallAfterNonce,
                Some("read_attestation_header"),
                4,
            ),
            (TAgentPeerScriptV1::TruncatedHeader, None, 4),
            (
                TAgentPeerScriptV1::Header {
                    kind: TAgentHeaderV1::Valid,
                    body: &partial_body,
                    finish: false,
                },
                Some("read_frame_body"),
                7,
            ),
        ];
        for (i, (script, expected_stage, expected_code)) in cases.into_iter().enumerate() {
            let (close, (res, elapsed)) = tokio::join!(
                t_agent_raw_peer(&client_ep, addr.clone(), script, hold),
                t_agent_timed(server_ep.accept_1rtt_frame_fresh_nonce(1000 + i as u32, None)),
            );
            let stage_ok = match (expected_stage, &res) {
                (Some(want), Err(IrohCarrierError::Timeout { stage })) => *stage == want,
                (None, Err(IrohCarrierError::Carrier(CarrierError::EmptyOrTruncatedFrame))) => true,
                _ => false,
            };
            assert!(
                stage_ok,
                "{script:?}: expected {expected_stage:?}, got {res:?}"
            );
            assert!(
                elapsed < Duration::from_millis(1500),
                "{script:?} not bounded: {elapsed:?}"
            );
            assert_eq!(
                t_agent_app_close_code(&close),
                Some(expected_code),
                "{script:?} close reason {close:?}"
            );
        }

        client_ep.close().await;
        server_ep.close().await;
    }

    /// Invariant 1 / sibling parity: `send_frame_1rtt` bounds dial+1-RTT completion and the nonce
    /// read, rejects a zero challenge, and closes the connection on every failure.
    #[tokio::test]
    async fn t_agent_sender_stage_timeouts_zero_challenge_and_cleanup() {
        let client_ep =
            IrohEndpointBridgeV1::bind_loopback([0xD1; 32], IrohCarrierAdapterV1::default())
                .await
                .unwrap()
                .with_timeouts(IrohAcceptTimeoutsV1 {
                    handshake_timeout: Duration::from_millis(300),
                    read_timeout: Duration::from_millis(300),
                });

        // (a) Listener that never calls `accept()`: dial + 1-RTT completion bounded.
        let silent_ep =
            IrohEndpointBridgeV1::bind_loopback([0xD2; 32], IrohCarrierAdapterV1::default())
                .await
                .unwrap();
        let (res_a, elapsed_a) =
            t_agent_timed(client_ep.send_frame_1rtt(silent_ep.addr(), b"payload")).await;
        assert!(
            matches!(
                res_a,
                Err(IrohCarrierError::Timeout {
                    stage: "quic_connect"
                })
            ),
            "{res_a:?}"
        );
        assert!(elapsed_a < Duration::from_secs(2), "{elapsed_a:?}");
        silent_ep.close().await;

        // (b) Server reads the hello but never issues a challenge; (c) server issues a zero nonce.
        let raw_ep =
            IrohEndpointBridgeV1::bind_loopback([0xD3; 32], IrohCarrierAdapterV1::default())
                .await
                .unwrap();
        for zero_nonce in [false, true] {
            let raw_server = async {
                let incoming = raw_ep.endpoint.accept().await?;
                let conn = incoming.accept().ok()?.await.ok()?;
                let (mut send, mut recv) = conn.accept_bi().await.ok()?;
                let mut hello = [0u8; 4];
                recv.read_exact(&mut hello).await.ok()?;
                if zero_nonce {
                    send.write_all(&[0u8; 16]).await.ok()?;
                }
                let reason = tokio::time::timeout(Duration::from_secs(3), conn.closed())
                    .await
                    .ok();
                drop(send);
                drop(recv);
                reason
            };
            let (reason, (res, elapsed)) = tokio::join!(
                raw_server,
                t_agent_timed(client_ep.send_frame_1rtt(raw_ep.addr(), b"payload")),
            );
            if zero_nonce {
                assert!(t_agent_is_invalid_attestation(&res), "{res:?}");
            } else {
                assert!(
                    matches!(
                        res,
                        Err(IrohCarrierError::Timeout {
                            stage: "read_nonce"
                        })
                    ),
                    "{res:?}"
                );
            }
            assert!(elapsed < Duration::from_secs(2), "{elapsed:?}");
            assert_eq!(
                t_agent_app_close_code(&reason),
                Some(2),
                "zero_nonce={zero_nonce}: {reason:?}"
            );
        }

        raw_ep.close().await;
        client_ep.close().await;
    }

    /// Invariant 2: quota is charged exactly once per frame through every bridge entry point
    /// (raw, fresh-nonce, envelope, envelope-with-policy, gossip v2), on failure and on success.
    #[tokio::test]
    async fn t_agent_quota_charged_exactly_once_per_frame_across_all_bridge_entry_points() {
        let origin_seed = [0xE1; 32];
        let origin_kp = KeyPair::from_seed(origin_seed).unwrap();
        let origin_auth = MessageAuth::new(origin_kp.clone());
        let server_ep = IrohEndpointBridgeV1::bind_loopback(
            [0xE2; 32],
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, 4096),
        )
        .await
        .unwrap();
        let client_ep =
            IrohEndpointBridgeV1::bind_loopback(origin_seed, IrohCarrierAdapterV1::default())
                .await
                .unwrap();
        let addr = server_ep.addr();
        let link = client_ep.link_peer_id();
        let recipient = KeyPair::from_seed([0xE3; 32]).unwrap().peer_id();
        let mut replay = ReplayProtection::try_with_config(64, 120)
            .unwrap()
            .with_startup_policy(ReplayStartupPolicyV1::GenesisRosterBootstrap)
            .with_cohort_allowlist(CohortAllowlist::from_peers([origin_kp.peer_id()]));
        let mut quota = LinkFrameQuotaTrackerV1::with_limits(32, 60, 8, 64);
        let garbage = [0xABu8; 8];

        macro_rules! assert_charged_once {
            ($expected:expr, $payload:expr, $accept:expr, $pat:pat) => {{
                let (sent, res) =
                    tokio::join!(client_ep.send_frame_1rtt(addr.clone(), $payload), $accept);
                sent.unwrap();
                #[allow(clippy::redundant_pattern_matching)]
                let matched = matches!(res, $pat);
                assert!(matched, "entry point #{}: {:?}", $expected, res);
                assert_eq!(
                    quota.active_window_count(&link),
                    $expected,
                    "entry point #{} did not charge quota exactly once",
                    $expected
                );
            }};
        }

        assert_charged_once!(
            1,
            &garbage,
            server_ep.accept_1rtt_frame_authenticated([0x11; 16], 1000, Some(&mut quota)),
            Ok(_)
        );
        assert_charged_once!(
            2,
            &garbage,
            server_ep.accept_1rtt_frame_fresh_nonce(1000, Some(&mut quota)),
            Ok(_)
        );
        assert_charged_once!(
            3,
            &garbage,
            server_ep.accept_envelope_over_quic([0x12; 16], &mut replay, 1000, Some(&mut quota)),
            Err(IrohCarrierError::Envelope(_))
        );
        assert_charged_once!(
            4,
            &garbage,
            server_ep.accept_envelope_over_quic_fresh_nonce(&mut replay, 1000, Some(&mut quota)),
            Err(IrohCarrierError::Envelope(_))
        );
        assert_charged_once!(
            5,
            &garbage,
            server_ep.accept_envelope_over_quic_with_policy(
                [0x13; 16],
                &mut replay,
                1000,
                Some(&mut quota),
                EnvelopeVersionPolicyV1::AllowLegacyV1AndV2,
            ),
            Err(IrohCarrierError::Envelope(_))
        );
        assert_charged_once!(
            6,
            &garbage,
            server_ep.accept_gossip_v2_over_quic(
                [0x14; 16],
                recipient,
                &mut replay,
                1000,
                Some(&mut quota),
            ),
            Err(IrohCarrierError::Gossip(_))
        );
        assert_charged_once!(
            7,
            &garbage,
            server_ep.accept_gossip_v2_over_quic_fresh_nonce(
                recipient,
                &mut replay,
                1000,
                Some(&mut quota),
            ),
            Err(IrohCarrierError::Gossip(_))
        );
        assert_eq!(replay.cache_size(), 0);

        // Successful admission is also charged exactly once (no second adapter-level charge).
        let env_v1 = MessageEnvelope::new_v1(
            *origin_kp.public_key(),
            [0x61; 16],
            1,
            3,
            MessageType::GradientUpdate,
            vec![7, 8, 9],
        )
        .with_sequence(1)
        .with_timestamp(1000)
        .sign_with(&origin_auth);
        let v1_bytes = env_v1.serialize().unwrap();
        assert_charged_once!(
            8,
            &v1_bytes,
            server_ep.accept_envelope_over_quic_with_policy(
                [0x15; 16],
                &mut replay,
                1000,
                Some(&mut quota),
                EnvelopeVersionPolicyV1::AllowLegacyV1AndV2,
            ),
            Ok(_)
        );
        assert_eq!(replay.cache_size(), 1);

        client_ep.close().await;
        server_ep.close().await;
    }

    /// Invariant 2 boundary: `4 + max_frame_bytes` is admitted byte-exact, one more byte maps the
    /// typed `ReadToEndError::TooLong` to `FrameTooLarge`, and both attempts are charged pre-body.
    #[tokio::test]
    async fn t_agent_body_cap_boundary_exact_admitted_plus_one_rejected_and_charged() {
        let max = 64usize;
        let server_ep = IrohEndpointBridgeV1::bind_loopback(
            [0xF1; 32],
            IrohCarrierAdapterV1::new(LinkBindingPolicyV1::StrictOneToOne, max),
        )
        .await
        .unwrap();
        let client_ep =
            IrohEndpointBridgeV1::bind_loopback([0xF2; 32], IrohCarrierAdapterV1::default())
                .await
                .unwrap();
        let addr = server_ep.addr();
        let link = client_ep.link_peer_id();
        let hold = Duration::from_secs(3);
        let mut quota = LinkFrameQuotaTrackerV1::with_limits(8, 60, 8, 16);
        let exact = server_ep.adapter().encode_wire_frame(&[0x5A; 64]).unwrap();
        assert_eq!(exact.len(), 4 + max);
        let mut over = exact.clone();
        over.push(0x00);

        let (_, res_exact) = tokio::join!(
            t_agent_raw_peer(
                &client_ep,
                addr.clone(),
                TAgentPeerScriptV1::Header {
                    kind: TAgentHeaderV1::Valid,
                    body: &exact,
                    finish: true,
                },
                hold,
            ),
            server_ep.accept_1rtt_frame_fresh_nonce(1000, Some(&mut quota)),
        );
        let (principal, frame) = res_exact.unwrap();
        assert_eq!(principal.link_peer_id(), link);
        assert_eq!(frame, exact);
        assert_eq!(quota.active_window_count(&link), 1);

        let (close_over, res_over) = tokio::join!(
            t_agent_raw_peer(
                &client_ep,
                addr,
                TAgentPeerScriptV1::Header {
                    kind: TAgentHeaderV1::Valid,
                    body: &over,
                    finish: true,
                },
                hold,
            ),
            server_ep.accept_1rtt_frame_fresh_nonce(1001, Some(&mut quota)),
        );
        assert!(
            matches!(
                res_over,
                Err(IrohCarrierError::Carrier(CarrierError::FrameTooLarge {
                    max: 64,
                    ..
                }))
            ),
            "{res_over:?}"
        );
        assert_eq!(quota.active_window_count(&link), 2);
        assert_eq!(t_agent_app_close_code(&close_over), Some(7));

        client_ep.close().await;
        server_ep.close().await;
    }

    /// Invariant 3: bounded FIFO nonce window, CSPRNG nonces, zero/duplicate rejection shared
    /// across every sibling entry point (even after an accept timeout), and fail-closed poison.
    #[tokio::test]
    async fn t_agent_nonce_cache_bounded_shared_across_siblings_burned_on_timeout_and_poison_fail_closed(
    ) {
        let nonce_n = |i: u64| {
            let mut n = [0u8; 16];
            n[..8].copy_from_slice(&(i + 1).to_le_bytes());
            n
        };
        let cap = DEFAULT_MAX_RECENT_NONCES as u64;
        let mut cache = RecentNonceCache::default();
        for i in 0..cap {
            assert!(cache.record_unique(nonce_n(i)));
        }
        assert!(!cache.record_unique(nonce_n(0)));
        assert!(!cache.record_unique(nonce_n(cap - 1)));
        assert_eq!(
            (cache.order.len(), cache.seen.len()),
            (DEFAULT_MAX_RECENT_NONCES, DEFAULT_MAX_RECENT_NONCES)
        );
        assert!(cache.record_unique(nonce_n(cap)));
        assert_eq!(
            (cache.order.len(), cache.seen.len()),
            (DEFAULT_MAX_RECENT_NONCES, DEFAULT_MAX_RECENT_NONCES)
        );
        assert!(
            !cache.contains(&nonce_n(0)),
            "oldest nonce must be evicted at capacity"
        );
        assert!(cache.contains(&nonce_n(1)) && cache.contains(&nonce_n(cap)));

        let server_ep =
            IrohEndpointBridgeV1::bind_loopback([0xA9; 32], IrohCarrierAdapterV1::default())
                .await
                .unwrap()
                .with_timeouts(IrohAcceptTimeoutsV1 {
                    handshake_timeout: Duration::from_millis(100),
                    read_timeout: Duration::from_millis(100),
                });
        let recipient = KeyPair::from_seed([0xAA; 32]).unwrap().peer_id();
        let mut replay = ReplayProtection::try_with_config(64, 120).unwrap();
        let mut quota = LinkFrameQuotaTrackerV1::with_limits(4, 60, 4, 4);
        let policy = EnvelopeVersionPolicyV1::AllowLegacyV1AndV2;

        // CSPRNG nonces are non-zero and pairwise distinct; generation alone reserves nothing.
        let mut drawn = BTreeSet::new();
        for _ in 0..256 {
            let n = server_ep.generate_fresh_challenge_nonce().unwrap();
            assert_ne!(n, [0u8; 16]);
            assert!(drawn.insert(n));
        }
        assert!(server_ep.recent_nonces.lock().unwrap().order.is_empty());

        // Zero caller nonce rejected by every sibling entry point BEFORE `endpoint.accept()`
        // (otherwise these would surface as `Timeout { stage: "endpoint_accept" }`).
        let zero = [0u8; 16];
        assert!(t_agent_is_invalid_attestation(
            &server_ep.accept_1rtt_frame(zero).await
        ));
        assert!(t_agent_is_invalid_attestation(
            &server_ep
                .accept_1rtt_frame_authenticated(zero, 1000, Some(&mut quota))
                .await
        ));
        assert!(t_agent_is_invalid_attestation(
            &server_ep
                .accept_envelope_over_quic(zero, &mut replay, 1000, Some(&mut quota))
                .await
        ));
        assert!(t_agent_is_invalid_attestation(
            &server_ep
                .accept_envelope_over_quic_with_policy(
                    zero,
                    &mut replay,
                    1000,
                    Some(&mut quota),
                    policy
                )
                .await
        ));
        assert!(t_agent_is_invalid_attestation(
            &server_ep
                .accept_gossip_v2_over_quic(zero, recipient, &mut replay, 1000, Some(&mut quota))
                .await
        ));

        // A reserved static nonce stays burned after an accept timeout, and the reservation is
        // shared across every sibling wrapper (fail-closed reuse rejection).
        let static_nonce = [0xA7; 16];
        assert!(matches!(
            server_ep
                .accept_envelope_over_quic(static_nonce, &mut replay, 1000, Some(&mut quota))
                .await,
            Err(IrohCarrierError::Timeout {
                stage: "endpoint_accept"
            })
        ));
        assert!(t_agent_is_invalid_attestation(
            &server_ep
                .accept_gossip_v2_over_quic(
                    static_nonce,
                    recipient,
                    &mut replay,
                    1000,
                    Some(&mut quota)
                )
                .await
        ));
        assert!(t_agent_is_invalid_attestation(
            &server_ep
                .accept_envelope_over_quic_with_policy(
                    static_nonce,
                    &mut replay,
                    1000,
                    Some(&mut quota),
                    policy
                )
                .await
        ));
        assert!(t_agent_is_invalid_attestation(
            &server_ep
                .accept_1rtt_frame_authenticated(static_nonce, 1000, Some(&mut quota))
                .await
        ));
        assert!(t_agent_is_invalid_attestation(
            &server_ep.accept_1rtt_frame(static_nonce).await
        ));
        assert_eq!(quota.tracked_link_count(), 0);
        assert_eq!(replay.cache_size(), 0);

        // A poisoned nonce-cache mutex fails closed with a typed error (no panic, no accept).
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = server_ep.recent_nonces.lock().unwrap();
            panic!("t-agent: deliberately poisoning recent_nonces");
        }));
        assert!(server_ep.recent_nonces.is_poisoned());
        assert!(matches!(
            server_ep.generate_fresh_challenge_nonce(),
            Err(IrohCarrierError::Transport(_))
        ));
        assert!(matches!(
            server_ep.accept_1rtt_frame_fresh_nonce(1000, None).await,
            Err(IrohCarrierError::Transport(_))
        ));
        assert!(matches!(
            server_ep
                .accept_1rtt_frame_authenticated([0xA8; 16], 1000, None)
                .await,
            Err(IrohCarrierError::Transport(_))
        ));
        assert!(matches!(
            server_ep
                .accept_envelope_over_quic_fresh_nonce(&mut replay, 1000, None)
                .await,
            Err(IrohCarrierError::Transport(_))
        ));
        assert!(matches!(
            server_ep
                .accept_gossip_v2_over_quic_fresh_nonce(recipient, &mut replay, 1000, None)
                .await,
            Err(IrohCarrierError::Transport(_))
        ));

        server_ep.close().await;
    }
}
