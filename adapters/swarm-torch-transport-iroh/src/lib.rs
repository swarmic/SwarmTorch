#![forbid(unsafe_code)]
//! Isolated QUIC / `Iroh` 1-RTT carrier adapter (`ADR-0022` §4, `W12-02`, `ST-08`).
//!
//! **Maturity Tier:** `Simulation verified` — deterministic ALPN, 1-RTT stage gating, bounded
//! framing, link-vs-originator binding (`StrictOneToOne` / `DecoupledRelayAllowed`), and bounded
//! sliding-window pre-auth rate limiting are enforced and tested locally. Live `iroh` QUIC endpoint
//! handshake integration is isolated for Phase `13D-2`.

use core::fmt;
use std::collections::{BTreeMap, VecDeque};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use std::time::{SystemTime, UNIX_EPOCH};

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
use swarm_torch_net::protocol::{MessageEnvelope, VerifyError};

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
            Self::LinkRateLimited { .. } | Self::LinkQuotaTableFull { .. } | Self::Transport(_) => {
                None
            }
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
    pub fn accept_envelope_frame<'a>(
        &self,
        link_principal: IrohVerifiedLinkPrincipalV1,
        stream_bytes: &'a [u8],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
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
            .verify_authenticated(replay_guard, current_time_secs)
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

/// Live `iroh 1.4` 1-RTT QUIC transport bridge enforcing SwarmTorch link attestation,
/// strict 1-RTT completion (zero 0-RTT fast-path), TLS-keyed link binding, bounded reads,
/// pre-auth link quotas, and replay-safe admission (`ST-08`, `ADR-0022`).
pub struct IrohEndpointBridgeV1 {
    endpoint: Endpoint,
    link_auth: MessageAuth,
    link_peer_id: PeerId,
    adapter: IrohCarrierAdapterV1,
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
        })
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
    pub async fn send_frame_1rtt(
        &self,
        target: impl Into<EndpointAddr>,
        payload: &[u8],
    ) -> Result<(), IrohCarrierError> {
        let wire_frame = self.adapter.encode_wire_frame(payload)?;
        let conn = self
            .endpoint
            .connect(target, IrohCarrierAdapterV1::expected_alpn())
            .await
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;

        let negotiated_alpn = conn.alpn();
        if negotiated_alpn != IrohCarrierAdapterV1::expected_alpn() {
            conn.close(1u32.into(), b"alpn mismatch");
            return Err(IrohCarrierError::Carrier(CarrierError::AlpnMismatch));
        }

        let (mut send, mut recv) = conn
            .open_bi()
            .await
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;

        // Write 4-byte stream hello immediately so QUIC transmits client-initiated stream opening.
        send.write_all(Self::STREAM_HELLO_V1)
            .await
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;

        let mut receiver_nonce = [0u8; 16];
        recv.read_exact(&mut receiver_nonce)
            .await
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;
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

        send.write_all(&out)
            .await
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;
        send.finish()
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;
        let _ = recv.read_to_end(1).await;
        Ok(())
    }

    /// Accept one incoming `iroh` connection strictly after `1-RTT` TLS completion (never 0-RTT),
    /// issue `receiver_nonce` challenge, verify both ALPN and TLS `remote_id == attestation_key`,
    /// and return a sealed [`IrohVerifiedLinkPrincipalV1`] + bounded wire frame bytes.
    pub async fn accept_1rtt_frame(
        &self,
        receiver_nonce: [u8; 16],
    ) -> Result<(IrohVerifiedLinkPrincipalV1, Vec<u8>), IrohCarrierError> {
        if receiver_nonce == [0u8; 16] {
            return Err(IrohCarrierError::Carrier(
                CarrierError::InvalidLinkAttestation,
            ));
        }

        let incoming = self
            .endpoint
            .accept()
            .await
            .ok_or_else(|| IrohCarrierError::Transport("iroh endpoint closed".to_string()))?;

        // Strictly require full 1-RTT QUIC/TLS completion (never call `.into_0rtt()`).
        let accepting = incoming
            .accept()
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;
        let conn = accepting
            .await
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;

        let negotiated_alpn = conn.alpn().to_vec();
        if negotiated_alpn.as_slice() != IrohCarrierAdapterV1::expected_alpn() {
            conn.close(1u32.into(), b"alpn mismatch");
            return Err(IrohCarrierError::Carrier(CarrierError::AlpnMismatch));
        }
        let remote_id_bytes: [u8; 32] = *conn.remote_id().as_bytes();

        let (mut send, mut recv) = conn
            .accept_bi()
            .await
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;

        let mut hello = [0u8; 4];
        recv.read_exact(&mut hello)
            .await
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;
        if &hello != Self::STREAM_HELLO_V1 {
            return Err(IrohCarrierError::Carrier(
                CarrierError::EmptyOrTruncatedFrame,
            ));
        }

        send.write_all(&receiver_nonce)
            .await
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;

        let max_read = LINK_ATTESTATION_HEADER_BYTES
            .saturating_add(4)
            .saturating_add(self.adapter.codec.max_frame_bytes);
        let buf = recv
            .read_to_end(max_read)
            .await
            .map_err(|e| IrohCarrierError::Transport(e.to_string()))?;
        let _ = send.write_all(&[0x01]).await;
        let _ = send.finish();

        if buf.len() < LINK_ATTESTATION_HEADER_BYTES {
            return Err(IrohCarrierError::Carrier(
                CarrierError::EmptyOrTruncatedFrame,
            ));
        }

        let mut att_pubkey = [0u8; 32];
        att_pubkey.copy_from_slice(&buf[..32]);
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&buf[32..LINK_ATTESTATION_HEADER_BYTES]);

        // Cryptographic binding between QUIC TLS 1.3 `EndpointId` and SwarmTorch attestation key.
        if att_pubkey != remote_id_bytes {
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
        let principal = self.adapter.accept_attested_link(
            &receiver_nonce,
            QuicHandshakeStageV1::OneRttComplete,
            &att,
        )?;
        Ok((principal, buf[LINK_ATTESTATION_HEADER_BYTES..].to_vec()))
    }

    /// Accept one live `1-RTT` QUIC frame and admit a direct [`MessageEnvelope`] under
    /// pre-auth link quota + replay protection.
    pub async fn accept_envelope_over_quic(
        &self,
        receiver_nonce: [u8; 16],
        replay_guard: &mut ReplayProtection,
        current_time_secs: u32,
        quota: Option<&mut LinkFrameQuotaTrackerV1>,
    ) -> Result<AcceptedIrohEnvelopeV1, IrohCarrierError> {
        let (principal, wire_frame) = self.accept_1rtt_frame(receiver_nonce).await?;
        let (accepted, rest) = self.adapter.accept_envelope_frame(
            principal,
            &wire_frame,
            replay_guard,
            current_time_secs,
            quota,
        )?;
        if !rest.is_empty() {
            return Err(IrohCarrierError::Carrier(
                CarrierError::EmptyOrTruncatedFrame,
            ));
        }
        Ok(accepted)
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
        let (principal, wire_frame) = self.accept_1rtt_frame(receiver_nonce).await?;
        let (accepted, rest) = self.adapter.accept_gossip_v2_frame(
            principal,
            recipient_peer,
            &wire_frame,
            replay_guard,
            current_time_secs,
            quota,
        )?;
        if !rest.is_empty() {
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

        // StrictOneToOne rejects relay link != originator WITHOUT mutating strict_replay!
        assert!(matches!(
            strict_adapter.accept_envelope_frame(
                relay_link,
                &wire_frame,
                &mut strict_replay,
                env.timestamp,
                None
            ),
            Err(IrohCarrierError::Carrier(
                CarrierError::StrictBindingMismatch { .. }
            ))
        ));
        // Prove strict_replay remained clean: the exact same envelope succeeds over direct_origin_link!
        let (strict_direct, strict_rest) = strict_adapter
            .accept_envelope_frame(
                direct_origin_link,
                &wire_frame,
                &mut strict_replay,
                env.timestamp,
                None,
            )
            .unwrap();
        assert!(strict_rest.is_empty());
        assert!(!strict_direct.binding().relayed_via_intermediary());

        // DecoupledRelayAllowed admits and flags relayed_via_intermediary = true
        let (accepted_env, rest) = relay_adapter
            .accept_envelope_frame(
                relay_link,
                &wire_frame,
                &mut relay_replay,
                env.timestamp,
                None,
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
            client_ep.send_frame_1rtt(addr, &valid_bytes),
            server_ep.accept_envelope_over_quic([0x82; 16], &mut replay, 1000, None),
        );
        s_ok.unwrap();
        let accepted = r_ok.unwrap();
        assert!(!accepted.binding().relayed_via_intermediary());
        assert_eq!(accepted.binding().originator_peer_id(), origin_kp.peer_id());
        assert_eq!(accepted.envelope.payload, vec![9, 8, 7]);

        client_ep.close().await;
        server_ep.close().await;
    }
}
